# SPDX-License-Identifier: GPL-3.0-only
"""Optional verified WiLoR detector, side continuity and packed hand scenes.

Only hand boxes/labels are reused; 3D reconstruction stays SAM 3D Body.
"""
import hashlib
import itertools
import time
import numpy as np
from hand_geometry import validate_bbox
from studio_paths import path

DETECTOR_SHA256='5ef3df44e42d2db52d4ffe91f83a22ce9925e2acc9abebf453f2c5d22e380033'


class HandDetector:
    def __init__(self,backend='native',optimized_preprocess=True):
        import torch
        from ultralytics import YOLO
        self.torch=torch; checkpoint=path('hand_detector_checkpoint')
        if not checkpoint.is_file():raise ValueError('缺少手部检测器，请配置 hand_detector_checkpoint：'+str(checkpoint))
        with checkpoint.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if digest!=DETECTOR_SHA256:raise ValueError('当前适配器仅接受已验证的 WiLoR detector.pt，请勿填入人体 Pose 权重')
        if backend=='trt_fp16':
            engine=path('hand_detector_engine')
            with engine.open('rb') as stream:engine_digest=hashlib.file_digest(stream,'sha256').hexdigest()
            from engine_registry import verify_engine
            verify_engine(engine,engine_digest)
            self.model=YOLO(str(engine),task='pose')
        elif backend=='native':self.model=YOLO(str(checkpoint)).to('cuda')
        else:raise ValueError('手部检测器方案无效')
        self.backend=backend;self.optimized_preprocess=bool(optimized_preprocess and backend=='trt_fp16')
        if self.model.names!={0:'left',1:'right'}:raise ValueError('检测器左右手类别不匹配')
        self.metadata=dict(name='WiLoR YOLOv8m-Pose hand detector',checkpoint=str(checkpoint),sha256=digest,
                           image_size=640,execution=backend,engine=(str(engine) if backend=='trt_fp16' else None),engine_sha256=(engine_digest if backend=='trt_fp16' else None),classes=self.model.names,role='boxes and sides only; no WiLoR/MANO reconstruction',preprocessing=('RGB pinned buffers v1' if self.optimized_preprocess else 'Ultralytics standard'))

    def detect(self,rgb,confidence=.4):
        if not .05<=confidence<=.95:raise ValueError('检测置信度范围应为 0.05–0.95')
        self.torch.cuda.synchronize();start=time.perf_counter()
        kwargs={}
        if self.optimized_preprocess:
            from hand_detector_preprocess import HandRGBPredictor
            kwargs['predictor']=HandRGBPredictor
        result=self.model.predict(source=rgb if self.optimized_preprocess else rgb[:,:,::-1].copy(),conf=confidence,imgsz=640,device=0,verbose=False,max_det=6,**kwargs)[0]
        boxes=result.boxes.xyxy.detach().cpu().numpy();scores=result.boxes.conf.detach().cpu().numpy();classes=result.boxes.cls.detach().cpu().numpy()
        h,w=rgb.shape[:2];hands=[]
        for box,score,cls in zip(boxes,scores,classes):
            box=box.copy();box[[0,2]]=np.clip(box[[0,2]],0,w-1);box[[1,3]]=np.clip(box[[1,3]],0,h-1)
            try:box=validate_bbox(box,w,h)
            except ValueError:continue
            side=self.model.names[int(cls)]
            hands.append(dict(bbox=box.tolist(),side=side,raw_side=side,score=float(score)))
        self.torch.cuda.synchronize()
        elapsed=(time.perf_counter()-start)*1000
        speed=result.speed or {}
        self.last_timings={key:float(speed.get(key,0.)) for key in ('preprocess','inference','postprocess')}
        self.last_timings['other']=elapsed-sum(self.last_timings.values())
        return sorted(hands,key=lambda hand:-hand['score']),elapsed


def iou(a,b):
    a=np.asarray(a);b=np.asarray(b);lo=np.maximum(a[:2],b[:2]);hi=np.minimum(a[2:],b[2:])
    area=float(np.maximum(hi-lo,0).prod());total=float((a[2:]-a[:2]).prod()+(b[2:]-b[:2]).prod()-area)
    return area/max(total,1e-9)


class HandSideTracker:
    """At most one subject, two slots. No geometry extrapolation or stale meshes."""
    def __init__(self):self.previous={}
    def reset(self):self.previous.clear()
    def update(self,candidates,timestamp,sides=('left','right')):
        candidates=candidates[:6];costs={}
        for side in sides:
            previous=self.previous.get(side)
            if previous and timestamp-previous[1]>.5:previous=None
            for index,hand in enumerate(candidates):
                mismatch=hand['raw_side']!=side and hand.get('forced_side')!=side
                cost=.25*(1-hand['score']) + (1.2 if mismatch else 0.)
                if previous:
                    old=previous[0];box=np.asarray(hand['bbox']);old=np.asarray(old)
                    distance=float(np.linalg.norm((box[:2]+box[2:]-old[:2]-old[2:])/2))/max(float(np.linalg.norm(old[2:]-old[:2])),1)
                    overlap=iou(old,box)
                    cost+=min(distance,2)*.35+(1-overlap)*.3
                    # A class correction requires strong spatial continuity.
                    if mismatch and (distance>.65 or overlap<.1):cost=1e6
                    elif mismatch:cost-=.9
                elif mismatch:cost=1e6
                costs[side,index]=cost
        best=None
        for assignment in itertools.product(range(-1,len(candidates)),repeat=len(sides)):
            used=[index for index in assignment if index>=0]
            if len(used)!=len(set(used)):continue
            cost=sum(1.15 if index<0 else costs[side,index] for side,index in zip(sides,assignment))
            if best is None or cost<best[0]:best=cost,assignment
        result=[]
        for side,index in zip(sides,best[1]):
            if index<0:continue
            candidate=candidates[index]
            hand=dict(candidate,side=side,track_id=side,identity_source=('user forced side + short spatial continuity' if candidate.get('forced_side') else 'detector + short spatial continuity'))
            self.previous[side]=(hand['bbox'],timestamp);result.append(hand)
        return result


def choose_hands(candidates):
    result=[]
    for side in ('left','right'):
        hand=next((hand for hand in candidates if hand['side']==side),None)
        if hand is not None:
            result.append(dict(hand,track_id=side,identity_source='user forced side' if hand.get('forced_side') else 'detector label'))
    return result


def pack_scene(hands,outputs,matrix):
    """Shared camera scene, plus each hand's local geometry and native parameters."""
    packed=dict(camera_matrix=np.asarray(matrix,np.float32),camera_translation=np.zeros(3,np.float32),
                sides=np.array([hand['side'] for hand in hands],dtype='U5'),
                bboxes=np.array([hand['bbox'] for hand in hands],np.float32).reshape(-1,4),
                scores=np.array([hand['score'] for hand in hands],np.float32))
    vertices=[];faces=[];joints=[];edges=[];vo=jo=0
    for hand,(data,raw,crop,report) in zip(hands,outputs):
        side=hand['side']
        vertices.append(data['vertices']+data['camera_translation']);faces.append(data['faces']+vo)
        joints.append(data['joints']+data['camera_translation']);edges.append(data['edges']+jo)
        for key in ('vertices','faces','joints','edges','global_rotations','camera_translation','vertices_2d','joint_names'):
            packed[side+'_'+key]=data[key]
        for key in ('raw_vertices','raw_joints','raw_global_rotations','raw_camera_translation','filtered_vertices','filtered_joints','filtered_global_rotations','filtered_camera_translation'):
            if key in data:packed[side+'_'+key]=data[key]
        for key in ('hand','shape','scale','mhr_model_params','pred_cam','global_rot'):
            if key in raw:packed[side+'_raw_'+key]=raw[key]
        vo+=len(data['vertices']);jo+=len(data['joints'])
    for key,items,width,dtype in [('vertices',vertices,3,np.float32),('faces',faces,3,np.int32),('joints',joints,3,np.float32),('edges',edges,2,np.int32)]:
        packed[key]=np.concatenate(items,axis=0) if items else np.empty((0,width),dtype)
    return packed


class AutomaticHands:
    def __init__(self,engine):
        self.engine=engine;self.detector=HandDetector(engine.execution.detector);self.tracker=HandSideTracker()
    def process(self,rgb,confidence=.4,padding=1.1,matrix=None,timestamp=0,stabilize=True,detect_only=False,hand_selection='both'):
        if hand_selection not in ('both','left','right'):raise ValueError('手部检测范围应为 both / left / right')
        sides=('left','right') if hand_selection=='both' else (hand_selection,)
        if getattr(self,'_hand_selection',None)!=hand_selection:
            self.tracker.reset();self._hand_selection=hand_selection
        start=time.perf_counter();raw,detect_ms=self.detector.detect(rgb,confidence)
        # In single-hand mode the user supplies handedness. Keep original labels
        # for diagnostics, but never reject a valid box because YOLO swapped it.
        eligible=sorted((h for h in raw if h['score']>=confidence),key=lambda h:-h['score'])
        if hand_selection!='both':
            eligible=[dict(h,side=hand_selection,forced_side=hand_selection) for h in eligible]
        hands=self.tracker.update(eligible,timestamp,sides) if stabilize else choose_hands(eligible)
        h,w=rgb.shape[:2];provided_matrix=matrix is not None
        if matrix is None:matrix=np.array([[np.hypot(w,h),0,w/2],[0,np.hypot(w,h),h/2],[0,0,1]],np.float32)
        outputs=self.engine.predict_many(rgb,hands,padding,matrix) if hands and not detect_only else []
        for output in outputs:output[3]['camera_source']='provided' if provided_matrix else 'estimated'
        packed=pack_scene(hands if outputs else [],outputs,matrix)
        report=dict(task='automatic_hands',hand_count=len(hands),hands=hands,raw_detections=raw,
                    detector=self.detector.metadata,detect_ms=detect_ms,detector_timings=getattr(self.detector,'last_timings',{}),
                    reconstruct_ms=outputs[0][3]['inference_ms'] if outputs else 0.,
                    reconstruction_timings=outputs[0][3]['timings'] if outputs else {},
                    elapsed_ms=(time.perf_counter()-start)*1000,
                    execution_counts=outputs[0][3]['execution_counts'] if outputs else dict(body=0,hand=0,backbone=0),
                    batch_size=len(outputs),hand_reports=[output[3] for output in outputs],
                    detection_options=dict(confidence=confidence,hand_selection=hand_selection),
                    missing=[side for side in sides if not any(hand['side']==side for hand in hands)])
        return packed,outputs,report
