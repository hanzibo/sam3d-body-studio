# SPDX-License-Identifier: GPL-3.0-only
"""Owned adapter for the original SAM hand branch. No upstream code is modified."""
from pathlib import Path
import gc
import copy
import hashlib
import sys
import subprocess
import time
import numpy as np
from studio_paths import path
from hand_geometry import validate_bbox, hand_topology, mirror_geometry, validate_hand_outputs


class HandEngine:
    def __init__(self, stage=lambda text: None, execution=None):
        import torch
        self.torch = torch
        torch.set_num_threads(4)
        if not torch.cuda.is_available(): raise RuntimeError('手部实验需要 CUDA')
        self.stream=torch.cuda.Stream()
        sys.path.insert(0, str(path('official_repo')))
        from sam_3d_body.models.meta_arch import SAM3DBody
        from sam_3d_body.utils.config import get_config
        from sam_3d_body import SAM3DBodyEstimator
        from offline_dinov3 import local_dinov3_source
        checkpoint = path('fast_checkpoint_dir')
        self.hashes = {}
        for name, expected in [('model.ckpt', 'b5a2f9d305dd02626b967aa2e86021fba07065df66ce7a7e00ffb9664f150abf'),
                               ('assets/mhr_model.pt', '352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc')]:
            with (checkpoint/name).open('rb') as f: sha = hashlib.file_digest(f,'sha256').hexdigest()
            if sha != expected: raise ValueError('手部模型文件哈希不匹配：'+name)
            self.hashes[name] = sha
        stage('初始化手部网络与基础权重 · 512；随后应用所选编码器 / 加速方案')
        cfg = get_config(str(checkpoint/'model_config.yaml')); cfg.defrost()
        cfg.MODEL.IMAGE_SIZE = [512,512]
        cfg.MODEL.MHR_HEAD.MHR_MODEL_PATH = str(checkpoint/'assets/mhr_model.pt')
        cfg.TRAIN.FP16_TYPE = 'float16'; cfg.freeze()
        with local_dinov3_source(): self.model = SAM3DBody(cfg)
        state = torch.load(checkpoint/'model.ckpt', map_location='cpu', weights_only=True, mmap=True)
        missing, unexpected = self.model.load_state_dict(state.get('state_dict',state),strict=False)
        invalid = [k for k in missing if not k.startswith(('head_pose.mhr.','head_pose_hand.mhr.'))
                   and k not in ('backbone.encoder.mask_token','head_pose.hand_pose_comps_ori','head_pose_hand.hand_pose_comps_ori')]
        unexpected = [k for k in unexpected if k not in ('hand_cls_embed.weight','hand_cls_embed.bias')]
        if invalid or unexpected: raise ValueError(f'手部权重不匹配：{invalid}, {unexpected}')
        if 'backbone.encoder.mask_token' in missing:
            with torch.no_grad(): self.model.backbone.encoder.mask_token.zero_()
        with torch.no_grad():
            for head in (self.model.head_pose,self.model.head_pose_hand):
                head.hand_pose_comps_ori.copy_(head.hand_pose_comps)
        del state; gc.collect()
        self.model = self.model.cuda().eval()
        self.estimator = SAM3DBodyEstimator(self.model,cfg)
        mhr = self.model.head_pose_hand.mhr
        self.names = list(mhr.get_joint_names())
        ct = mhr.character_torch; lbs = ct.linear_blend_skinning
        array = lambda t: t.detach().cpu().numpy()
        self.ids, self.faces, self.joint_ids, self.edges, self.wrist = hand_topology(
            self.names, array(ct.skeleton.joint_parents), array(ct.mesh.faces),
            array(lbs.vert_indices_flattened), array(lbs.skin_indices_flattened),
            array(lbs.skin_weights_flattened), len(ct.blend_shape.base_shape))
        self.source_commit = subprocess.check_output(['git','-C',str(path('official_repo')),'rev-parse','HEAD'],text=True).strip()
        from hand_acceleration import HandExecution,install_execution
        self.execution=execution or HandExecution()
        self.execution_actual=install_execution(self,self.execution,stage)
        self.calls = dict(body=0, hand=0, backbone=0)
        for attr, key in [('forward_decoder','body'),('forward_decoder_hand','hand')]:
            original = getattr(self.model,attr)
            def wrap(*a, _fn=original, _key=key, **kw):
                self.calls[_key] += 1
                return _fn(*a,**kw)
            setattr(self.model,attr,wrap)
        self.model.backbone.register_forward_hook(lambda *args: self.calls.__setitem__('backbone',self.calls['backbone']+1))

    def predict(self, rgb, bbox, side='right', padding=.9, matrix=None):
        return self.predict_many(rgb,[dict(bbox=bbox,side=side)],padding,matrix)[0]

    def predict_many(self, rgb, hands, padding=.9, matrix=None):
        """Batch independent canonical hand crops as B hands x 1 person.

        Each crop retains its own (possibly reflected) camera intrinsics.
        """
        import cv2
        from sam_3d_body.data.transforms import Compose, GetBBoxCenterScale, TopdownAffine, VisionTransformWrapper
        from torchvision.transforms import ToTensor
        from sam_3d_body.data.utils.prepare_batch import prepare_batch
        from sam_3d_body.utils import recursive_to
        torch=self.torch; h,w=rgb.shape[:2]
        if not 1 <= len(hands) <= 2:raise ValueError('每帧支持 1–2 只手')
        if not np.isfinite(padding) or not .7 <= padding <= 2.:raise ValueError('裁剪比例无效')
        supplied_matrix=matrix is not None
        if matrix is None:matrix=np.array([[np.hypot(w,h),0,w/2],[0,np.hypot(w,h),h/2],[0,0,1]],np.float32)
        matrix=np.asarray(matrix,np.float32)
        if matrix.shape!=(3,3) or not np.isfinite(matrix).all() or min(matrix[0,0],matrix[1,1])<=0:raise ValueError('相机内参无效')
        before=dict(self.calls);encoder_runs_before=getattr(self.model.backbone,'executions',None);torch.cuda.synchronize();
        torch.compiler.cudagraph_mark_step_begin()
        began=time.perf_counter()
        parts=[];prepared=[]
        transform=Compose([GetBBoxCenterScale(padding=padding),TopdownAffine(input_size=[512,512],use_udp=False),VisionTransformWrapper(ToTensor())])
        for hand in hands:
            side=hand['side'];bbox=validate_bbox(hand['bbox'],w,h)
            if side not in ('right','left'):raise ValueError('左右手标签无效')
            model_k=matrix.copy();box=bbox.copy();image=rgb
            if side=='left':
                image=rgb[:,::-1].copy();box[[0,2]]=w-1-bbox[[2,0]]
                model_k[0,2]=w-1-matrix[0,2];model_k[0,1]*=-1
            part=prepare_batch(image,transform,box[None],cam_int=torch.tensor(model_k)[None])
            affine=part['affine_trans'][0,0].numpy().copy()
            parts.append(part);prepared.append((side,bbox,image,affine))
        batch={key:torch.cat([part[key] for part in parts],dim=0) if isinstance(parts[0][key],torch.Tensor)
               else sum([part[key] for part in parts],[]) for key in parts[0]}
        prepared_at=time.perf_counter()
        events={}
        def begin_backbone(*args):events['backbone_start']=torch.cuda.Event(enable_timing=True);events['backbone_start'].record()
        def end_backbone(*args):events['backbone_end']=torch.cuda.Event(enable_timing=True);events['backbone_end'].record()
        h1=self.model.backbone.register_forward_pre_hook(begin_backbone)
        h2=self.model.backbone.register_forward_hook(end_backbone)
        old_hand=self.model.forward_decoder_hand
        def timed_hand(*args,**kwargs):
            events['decoder_start']=torch.cuda.Event(enable_timing=True);events['decoder_start'].record()
            value=old_hand(*args,**kwargs)
            events['decoder_end']=torch.cuda.Event(enable_timing=True);events['decoder_end'].record()
            return value
        self.model.forward_decoder_hand=timed_hand
        try:
            with torch.inference_mode(),torch.cuda.stream(self.stream):
                batch=recursive_to(batch,'cuda');self.model._initialize_batch(batch)
                result=self.model.run_inference(rgb,batch,inference_type='hand')['mhr_hand']
                torch.cuda.synchronize();network_at=time.perf_counter()
                arrays={key:value.detach().float().cpu().numpy() for key,value in result.items()
                        if isinstance(value,torch.Tensor) and value.ndim and value.shape[0]==len(hands)}
        finally:
            h1.remove();h2.remove();self.model.forward_decoder_hand=old_hand
        copied_at=time.perf_counter()
        counts={key:self.calls[key]-before[key] for key in before}
        if counts!=dict(body=0,hand=1,backbone=1):raise RuntimeError('手部执行路径异常：'+str(counts))
        timings=dict(prepare_cpu_ms=(prepared_at-began)*1000,network_transfer_wall_ms=(network_at-prepared_at)*1000,
                     backbone_gpu_ms=events['backbone_start'].elapsed_time(events['backbone_end']),
                     hand_decoder_including_mhr_gpu_ms=events['decoder_start'].elapsed_time(events['decoder_end']),
                     output_copy_ms=(copied_at-network_at)*1000)
        display_arrays=arrays;temporal_info=None
        request=getattr(self,"temporal_request",None)
        if request is not None:
            from hand_temporal import filtered_geometry
            display_arrays,temporal_info=filtered_geometry(self,arrays,hands,request)
        outputs=[]
        for index,(side,bbox,image,affine) in enumerate(prepared):
            raw={key:value[index].copy() for key,value in arrays.items()}
            source_raw=raw
            source_compact=len(source_raw['pred_vertices'])==len(self.ids)
            source_rows=np.arange(len(self.ids)) if source_compact else self.ids
            source_raw['pred_vertex_ids']=self.ids.copy() if source_compact else np.arange(len(source_raw['pred_vertices']))
            if temporal_info is not None:source_raw['pred_keypoints_2d_verts_valid']=validate_hand_outputs(source_raw,source_rows)[0]
            if temporal_info is not None:raw={key:value[index].copy() for key,value in display_arrays.items()}
            compact=len(raw['pred_vertices'])==len(self.ids)
            vertex_rows=np.arange(len(self.ids)) if compact else self.ids
            raw['pred_vertex_ids']=self.ids.copy() if compact else np.arange(len(raw['pred_vertices']))
            projection_valid,undefined_projections=validate_hand_outputs(raw,vertex_rows)
            raw['pred_keypoints_2d_verts_valid']=projection_valid
            vertices=raw['pred_vertices'][vertex_rows].copy();joints=raw['pred_joint_coords'][self.joint_ids].copy()
            basis=np.diag([1.,-1.,-1.]).astype(np.float32)
            rotations=basis @ raw['joint_global_rots'][self.joint_ids] @ basis
            camera=raw['pred_cam_t'].copy();faces=self.faces.copy()
            pixels=raw['pred_keypoints_2d_verts'][vertex_rows].copy()
            if temporal_info is not None:
                projection_matrix=matrix.copy()
                if side=='left':projection_matrix[0,2]=w-1-matrix[0,2];projection_matrix[0,1]*=-1
                project=(vertices+camera)@projection_matrix.T;pixels=project[:,:2]/project[:,2:]
            names=[self.names[i] for i in self.joint_ids]
            if side=='left':
                vertices,joints,rotations,camera,faces=mirror_geometry(vertices,joints,rotations,camera,faces)
                pixels[:,0]=w-1-pixels[:,0];names=[n.replace('r_','l_',1) for n in names]
            crop=cv2.warpAffine(image,affine,(512,512))
            data=dict(vertices=vertices,faces=faces,joints=joints,global_rotations=rotations,
                      edges=self.edges,camera_translation=camera,camera_matrix=matrix,vertices_2d=pixels,
                      canonical_vertex_ids=self.ids,canonical_joint_ids=self.joint_ids,
                      joint_ids=np.array([self.names.index(n) for n in names],np.int32),joint_names=np.array(names),
                      bbox=bbox,affine_model=affine,wrist=joints[list(self.joint_ids).index(self.wrist)])
            projected=(vertices+camera)@matrix.T;projected=projected[:,:2]/projected[:,2:]
            error=float(np.max(np.abs(projected-pixels)))
            if error>.05:raise RuntimeError(f'手部相机投影校验失败：{error:.3f}px')
            report=dict(task='hand',side=side,padding=padding,image_size=512,inference_ms=(network_at-began)*1000,
                        batch_size=len(hands),execution_counts=counts,backbone_engine_executions=(self.model.backbone.executions-encoder_runs_before if encoder_runs_before is not None else None),timings=timings,projection_roundtrip_max_px=error,
                        coordinates='CV x right, y down, z forward; meters; add camera_translation',
                        rotations='MHR global in CV basis C R C; mirrored left S R S',
                        timing_scope='shared batch: crop + H2D + network + MHR + sync; excludes D2H/export/render; GPU sub-times nested',
                        topology='r_wrist descendants skin weight >= 0.5; complete triangles',
                        raw_output_convention='canonical right hand; not native left MHR parameters',raw_vertex_layout='hand_only' if compact else 'full_body',
                        final_surface_parity_mm=getattr(getattr(self.model.head_pose_hand,'_studio_final_hand',None),'parity_mm',None),hashes=self.hashes,
                        backbone=({'trt_fp16':'TensorRT FP16','trt_int8':'TensorRT INT8/FP16 mixed','native':'native PyTorch FP16'}[self.execution.backbone]),decoder='FP32',mhr='FP32',
                        execution=self.execution_actual,mhr_graph_parity_mm=getattr(self.model.head_pose_hand.mhr,'parity_mm',None),backbone_runtime=(self.model.backbone.status() if hasattr(self.model.backbone,'status') else None),
                        feedback_runtime=copy.deepcopy(getattr(self.model,'_studio_hand_feedback',None)),
                        undefined_nonhand_projection_ids=undefined_projections.tolist(),
                        intermediate_parity_mm=getattr(getattr(self.model.head_pose_hand,'_studio_selected_hand',None),'parity_mm',None),
                        vertices=len(vertices),triangles=len(faces),joint_nodes=len(joints),source_commit=self.source_commit,
                        camera_source='provided' if supplied_matrix else 'estimated')
            if temporal_info is not None:
                report["one_euro"]=temporal_info
                for key in ("vertices","joints","global_rotations","camera_translation"):
                    data["filtered_"+key]=data[key].copy()
                raw_v=source_raw["pred_vertices"][vertex_rows].copy();raw_j=source_raw["pred_joint_coords"][self.joint_ids].copy()
                raw_r=basis @ source_raw["joint_global_rots"][self.joint_ids] @ basis;raw_c=source_raw["pred_cam_t"].copy()
                if side=="left":raw_v,raw_j,raw_r,raw_c,_=mirror_geometry(raw_v,raw_j,raw_r,raw_c,self.faces)
                data.update(raw_vertices=raw_v,raw_joints=raw_j,raw_global_rotations=raw_r,raw_camera_translation=raw_c)
            outputs.append((data,source_raw,crop,report))
        return outputs
