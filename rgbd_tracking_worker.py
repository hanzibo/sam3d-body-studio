# SPDX-License-Identifier: GPL-3.0-only
"""Opt-in Fast worker job. Uses one exclusive GPU lease owned by fast_worker."""
from dataclasses import replace
import json
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from depth_pose_accelerated import CudaPoseFitter,BODY_IDS
from depth_pose_refinement import MhrSurfaceModel,PoseDepthConfig,project
from rgbd_sequence import RGBDSequence
from rgbd_geometry import matrix
from rgbd_tracking_points import SequenceGeometry,ProjectiveSurface,COCO_TO_MHR
from video_project import save_json,file_hash


class TrackingFitter(CudaPoseFitter):
    def __init__(self,reference):
        self.observed_uv=torch.zeros(len(BODY_IDS),2,device='cuda')
        self.observed_weights=torch.zeros(len(BODY_IDS),device='cuda')
        self.current_keypoints=None
        super().__init__(reference)

    @torch.no_grad()
    def prepare(self,reference,k,scale,translation,config):
        super().prepare(reference,k,scale,translation,config)
        self.observed_weights.zero_()
        if self.current_keypoints is not None:
            kp=np.asarray(self.current_keypoints)
            for coco,mhr in enumerate(COCO_TO_MHR):
                if mhr in BODY_IDS:
                    i=BODY_IDS.index(mhr)
                    self.observed_uv[i].copy_(torch.as_tensor(kp[coco,:2],device='cuda'))
                    self.observed_weights[i]=float(kp[coco,2]) if kp[coco,2]>=.35 else 0.

    def step(self):
        from torch.nn.functional import huber_loss
        _,v,kp=self.forward()
        d=v[self.ids]*self.scale+self.t+self.shift-self.target
        plane=(d*self.normals).sum(1)/.02
        uv=(self.project(kp*self.scale+self.t+self.shift)-self.observed_uv)/self.sigma
        loss=((huber_loss(plane,torch.zeros_like(plane),reduction='none')*self.valid).sum()/self.count
              +.15*(huber_loss(d/.04,torch.zeros_like(d),reduction='none')*self.valid[:,None]).sum()/(3*self.count)
              +2.*(huber_loss(uv,torch.zeros_like(uv),reduction='none')*self.observed_weights[:,None]).sum()/(2*self.observed_weights.sum().clamp(min=1))
              +.008*(self.angle/.15).square().mean()+.004*(self.shift/.05).square().mean())
        self.optimizer.zero_grad(set_to_none=True);loss.backward();self.optimizer.step()
        with torch.no_grad():
            self.angle.copy_(torch.maximum(self.lo,torch.minimum(self.hi,self.angle)))
            self.shift.copy_(torch.maximum(-self.shift_bound,torch.minimum(self.shift_bound,self.shift)))
        self.loss=loss.detach()


def save_pairs(file,surface):
    batches=[b for b in surface.batches if b.get('used_steps',0)>0]
    if not batches:
        np.savez_compressed(file,pixel_uv=np.empty((0,2),np.float32),vertex_ids=np.empty(0,int),batch=np.empty(0,int),target_xyz=np.empty((0,3),np.float32),normal_xyz=np.empty((0,3),np.float32),batch_step=np.empty(0,int),candidate_uv=np.empty((0,2),np.float32));return
    np.savez_compressed(file,**{k:np.concatenate([b[k] for b in batches]) for k in ('pixel_uv','vertex_ids','target_xyz','normal_xyz')},
                        batch=np.concatenate([np.full(len(b['vertex_ids']),i,np.int16) for i,b in enumerate(batches)]),
                        batch_step=np.array([b['first_step'] for b in batches]),batch_used_steps=np.array([b['used_steps'] for b in batches]),
                        candidate_uv=np.concatenate([b['model_uv'] for b in surface.candidates]))


def residual(fitter,params,translation,surface,keypoints):
    with torch.no_grad():
        old_base=fitter.base.clone();old_angle=fitter.angle.clone()
        fitter.base.copy_(torch.as_tensor(params,device='cuda'));fitter.angle.zero_()
        _,v,kp=fitter.forward();v=v.cpu().numpy()+translation
        uv=project(kp.cpu().numpy()+translation,surface.k)
        fitter.base.copy_(old_base);fitter.angle.copy_(old_angle)
    b=surface.batches[-1] if surface.batches else None
    depth=float(np.linalg.norm(v[b['vertex_ids']]-b['target_xyz'],axis=1).mean()*1000) if b is not None and len(b['vertex_ids']) else None
    errors=[]
    for coco,mhr in enumerate(COCO_TO_MHR):
        if mhr in BODY_IDS and keypoints[coco][2]>=.35:errors.append(np.linalg.norm(uv[BODY_IDS.index(mhr)]-keypoints[coco][:2]))
    return depth,float(np.mean(errors)) if errors else None,v,uv


def track_session(command,network_forward,faces,emit):
    directory=Path(command['directory']).resolve();mode=command['mode'];steps=int(command['steps'])
    if mode not in ('candidate','oracle') or steps not in (5,10,15,30):raise ValueError('invalid tracking configuration')
    policy=command.get('policy','periodic')
    if policy not in ('periodic','first_init') or (policy=='first_init' and mode!='candidate'):raise ValueError('invalid initialization policy')
    source=json.loads((directory/'inputs.json').read_text());sequence=RGBDSequence(source['source'])
    reference_folder=None
    if mode=='oracle':
        from rgbd_tracking_reference import reference_directory
        reference_folder=reference_directory(directory)
        reference_hash=file_hash(reference_folder/'manifest.json')
    source_hashes={p:file_hash(Path(__file__).parent/p) for p in ['rgbd_tracking_worker.py','rgbd_tracking_points.py','depth_pose_accelerated.py','fast_worker.py']}
    if source['index_sha256']!=file_hash(sequence.directory/'frames.jsonl') or source['camera_sha256']!=file_hash(sequence.directory/'camera.json'):raise ValueError('sequence provenance changed')
    dest=directory/(f'first-init-{steps}' if policy=='first_init' else f'{mode}-{steps}');dest.mkdir(exist_ok=False);(dest/'points').mkdir();(dest/'states').mkdir()
    if policy=='first_init':(dest/'meshes').mkdir()
    keyframes=directory/'network_keyframes';keyframes.mkdir(exist_ok=True)
    geometry=SequenceGeometry(sequence);k=geometry.k
    from depth_pose_render import FittingDepthRenderer
    renderer=FittingDepthRenderer(faces,geometry.small_k)
    model=MhrSurfaceModel();fitter=None;params=None;translation=None;last_keyframe=-1e6;failure=False;rows=[];cold_ms=0.
    config=PoseDepthConfig(steps=steps,correspond_every=5,max_angle_deg=20.,max_root_deg=10.,max_translation_m=.1)
    try:
        for local,entry in enumerate(source['frames']):
            began=time.perf_counter();stage={};index=entry['source_index'];stamp=entry['time_s']
            start=time.perf_counter();image,cloud,normals,pixels,rawrow=geometry.read(index);stage['decode_geometry_ms']=(time.perf_counter()-start)*1000
            if file_hash(directory/'frames'/f'{local:06d}.png')!=entry['rgb_sha256']:raise ValueError('prepared RGB changed')
            row=dict(frame=local,source_index=index,time_s=stamp,state='LOST',valid=False,keyframe=False,points=0,
                     detector_ms=entry['detector_seconds']*1000,detector_input='cached current-frame YOLO; measured extraction cost added separately')
            if entry['bbox'] is None:
                # First-init ablation keeps its last accepted pose, but never
                # displays it as a valid body while the person is absent.
                if policy=='periodic':params=None;failure=False
                surface=None;np.savez_compressed(dest/'points'/f'{local:06d}.npz',pixel_uv=np.empty((0,2)),batch=np.empty(0,int),vertex_ids=np.empty(0,int),candidate_uv=np.empty((0,2)),batch_step=np.empty(0,int))
            else:
                start=time.perf_counter();is_key=needs_initialization(params is None,failure,stamp-last_keyframe,policy)
                reason='initial_or_return' if params is None else 'quality_failure' if failure else 'periodic_1s' if is_key else None
                network_seconds=0.;cache_hit=False;new_identity=params is None
                if is_key:
                    cached=keyframes/f'{local:06d}.npz';meta=cached.with_suffix('.json')
                    if cached.exists():
                        cache_meta=json.loads(meta.read_text())
                        if cache_meta['rgb_sha256']!=entry['rgb_sha256']:raise ValueError('keyframe RGB changed')
                        with np.load(cached,allow_pickle=False) as data:a={key:data[key].copy() for key in data.files}
                        network_seconds=cache_meta['network_seconds'];cache_hit=True
                    else:
                        cam=torch.from_numpy(matrix(k).astype(np.float32)).unsqueeze(0).cuda()
                        outputs,network_seconds=network_forward(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),'body',entry['bbox'],cam)
                        a={key:value for key,value in outputs[0].items() if value is not None};a['faces']=faces
                        np.savez_compressed(cached,**a);save_json(meta,dict(rgb_sha256=entry['rgb_sha256'],network_seconds=network_seconds))
                    old_identity=model.identity.clone() if params is not None else None
                    old_expression=model.expression.clone() if params is not None else None
                    model.seed(a)
                    if not new_identity:model.identity=old_identity;model.expression=old_expression
                    params=model.base.numpy().copy();translation=a['pred_cam_t'].copy()
                    if fitter is None:
                        fitter=TrackingFitter(model);cold_ms=fitter.setup_ms
                    # Correct gross monocular Z offset before narrow depth correspondence gating.
                    with torch.no_grad():
                        fitter.current_keypoints=entry['keypoints'];fitter.prepare(model,k,1.,translation,config)
                        _,v,_=fitter.forward();v=v.cpu().numpy()+translation
                    zrender=renderer.render(v)
                    overlap=(zrender>.05)&(cloud[:,:,2]>.05)
                    bbox=entry['bbox'];inside=((pixels>=bbox[:2])&(pixels<=bbox[2:])).all(2);overlap &= inside
                    if overlap.sum()>100:
                        dz=float(np.median(cloud[:,:,2][overlap]-zrender[overlap]));translation[2]+=np.clip(dz,-.5,.5)
                    last_keyframe=stamp
                stage['keyframe_setup_ms']=(time.perf_counter()-start)*1000
                # Initial fit is identical for both methods; oracle only enters subsequent frames.
                oracle=None
                if mode=='oracle' and not is_key:
                    oracle=cv2.imread(str(reference_folder/f'{local:06d}.png'),0)
                    if oracle is None:raise ValueError('oracle mask missing')
                    oracle=oracle>0
                c=replace(config,steps=60 if is_key else steps)
                surface=ProjectiveSurface(cloud,normals,pixels,k,geometry.small_k,renderer.render,entry['bbox'],c,oracle)
                model.base=torch.tensor(params,dtype=torch.float32);fitter.current_keypoints=entry['keypoints']
                start=time.perf_counter();candidate,t,report=fitter.fit(model,surface,1.,translation)
                stage['fit_ms']=(time.perf_counter()-start)*1000
                start=time.perf_counter();before_d,before_uv,_,_=residual(fitter,params,translation,surface,entry['keypoints'])
                after_d,after_uv,v,uv=residual(fitter,candidate,t,surface,entry['keypoints'])
                sufficient=report['sufficient']
                # Absolute quality plus no large degradation; stationary solutions may be accepted.
                before_energy=(before_d if before_d is not None else 1e6)+.5*(before_uv or 0)
                after_energy=(after_d if after_d is not None else 1e6)+.5*(after_uv or 0)
                accepted=sufficient and after_d is not None and after_d<100 and (after_uv is None or after_uv<100) and after_energy<=before_energy+5
                if accepted:params,translation=candidate,t;failure=False
                else:failure=True
                row.update(state='TRACKING' if accepted else 'DEGRADED',valid=bool(accepted),keyframe=is_key,keyframe_reason=reason,
                           network_ms=network_seconds*1000,network_cached=cache_hit,points=report['points'],before_depth_mm=before_d,after_depth_mm=after_d,
                           before_keypoints_px=before_uv,after_keypoints_px=after_uv,fit=report)
                stage['accept_ms']=(time.perf_counter()-start)*1000
                start=time.perf_counter()
                save_pairs(dest/'points'/f'{local:06d}.npz',surface)
                np.savez_compressed(dest/'states'/f'{local:06d}.npz',params=params,translation=translation,identity=model.identity.numpy(),expression=model.expression.numpy(),faces=faces,valid=accepted)
                stage['export_ms']=(time.perf_counter()-start)*1000
            row['stages_ms']=stage;row['offline_processing_ms']=(time.perf_counter()-began)*1000
            # Includes cached observations' measured cost. Not a concurrent live-camera timing.
            row['estimated_observed_input_ms']=row['offline_processing_ms']+row['detector_ms']+(row.get('network_ms',0) if row.get('network_cached') else 0)
            rows.append(row)
            with (dest/'frames.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            if policy=='first_init':
                # Artifact generation is outside measured processing cost.
                # Invalid frames have no displayed geometry, including LOST.
                if row['valid']:
                    with torch.no_grad():vv,jj,kk,_=model.forward(torch.tensor(params,dtype=torch.float32))
                    np.savez_compressed(dest/'meshes'/f'{local:06d}.npz',vertices=vv.numpy()+translation,joints=jj.numpy()+translation,keypoints=kk.numpy()+translation)
            if local%10==0:emit('stage',message=f'{mode} {local}/{len(source["frames"])} {row["state"]} {row["offline_processing_ms"]:.1f} ms')
        save_json(dest/'complete.json',dict(schema='studio_rgbd_tracking_v1',mode=mode,steps=steps,frames=len(rows),cold_graph_setup_ms=cold_ms,
                   reference_manifest_sha256=reference_hash if mode=='oracle' else None,reference_directory=str(reference_folder) if mode=='oracle' else None,
                   inputs_sha256=file_hash(directory/'inputs.json'),source_sha256=source_hashes,uses_sam3=mode=='oracle',initial_fit='same mask-free 60 steps in both modes',
                   geometry='raw depth stride 2, measured normals, nearest-Z splat at 320x180; RGB pixel_uv retains full-resolution float coordinates',
                   initialization_policy=policy,
                   keyframe_policy='first usable frame only; 60 steps once, then fixed short steps; NO periodic/failure/return network reset; preserve last accepted pose across absence' if policy=='first_init' else 'SAM every 1s or quality failure/return; 60-step keyframe; shape frozen within segment',
                   timing_scope='offline replay with cached YOLO; includes PNG hash/export, cached network cost estimated; NOT measured live-camera Hz',
                   points='actual measured target RGB float pixels and optimizer correspondence batches; candidates are model projections'))
        emit('tracking_complete',directory=str(dest))
    finally:renderer.close()


def needs_initialization(uninitialized,failure,elapsed,policy='periodic'):
    return uninitialized or (policy=='periodic' and (failure or elapsed>=1.))
