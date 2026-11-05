# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors, Studio modifications.
# Copyright (c) 2026 Fast SAM 3D Body Authors, upstream conversion workflow.
# Adapted preprocessing/output conventions from mhr2smpl/multi_view/
# infer_multiview.py and multiview_net.py at
# d72aa36913a1673ace029d345f437561a85ec9e2; retain LICENSES/Fast-SAM-3D-Body-MIT.txt.
# Modified: NumPy sampling/FK, joint-only asset access, raw/filtered outputs,
# explicit temporal resets and Studio protocol. Networks are imported separately.
"""Optional single-view community MHR->SMPL and center-window SmootherMLP.

No robot transport or retargeting. Raw parameters remain independent of smoothed
joint positions; smoothing does not produce valid smoothed joint rotations.
"""
from collections import deque
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import time
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parent
from studio_paths import path
FORK=path('fast_repo')/'mhr2smpl'
ASSETS=path('smpl_assets')
WEIGHTS=FORK/'experiments/multiview_n30000_e500/best_model.pth'
SAMPLES=FORK/'experiments/multiview_n30000_e500/sample_idx.npy'
SMOOTHER=FORK/'experiments/smoother_w5'
SMPL_NAMES=['pelvis','left_hip','right_hip','spine1','left_knee','right_knee','spine2',
 'left_ankle','right_ankle','spine3','left_foot','right_foot','neck','left_collar',
 'right_collar','head','left_shoulder','right_shoulder','left_elbow','right_elbow',
 'left_wrist','right_wrist','left_hand','right_hand']
SMPL_PARENTS=np.array([-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21])
SMPL_EDGES=np.array([(int(p),i) for i,p in enumerate(SMPL_PARENTS) if p>=0],dtype=np.int32)
CV_FLIP=np.diag([1.,-1.,-1.]).astype(np.float32)
SMPL_SHAPES={'smpl_global_orient':(3,), 'smpl_body_pose':(63,), 'smpl_betas':(10,),
 'smpl_joints_root':(24,3), 'smpl_keypoints_3d':(24,3),
 'smpl_global_rots_cv':(24,3,3), 'smpl_joints_smooth_root':(24,3),
 'smpl_keypoints_3d_smooth':(24,3), 'smpl_joints_one_euro_root':(24,3),
 'smpl_keypoints_3d_one_euro':(24,3)}


def model_path(config):
    if config.smpl_model_path:return Path(config.smpl_model_path).expanduser()
    for name in ('SMPL_NEUTRAL.npz','SMPL_NEUTRAL.pkl'):
        if (ASSETS/name).is_file():return ASSETS/name
    return ASSETS/'SMPL_NEUTRAL.pkl'


def resources(config):
    if not config.smpl_enabled:return []
    result=[WEIGHTS,SAMPLES,ASSETS/'mhr2smpl_mapping.npz',model_path(config),
            FORK/'multi_view/multiview_net.py']
    if config.smpl_smoothing:result.extend([SMOOTHER/'smoother_config.json',SMOOTHER/'smoother_best.pth',FORK/'smooth/smoother_net.py'])
    return result


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def fk_joints(rest,body_pose):
    """Exactly the 24 SMPL rigid FK joints, without surface LBS or extra vertices."""
    rotations=Rotation.from_rotvec(np.concatenate([np.zeros((1,3)),np.asarray(body_pose).reshape(21,3),np.zeros((2,3))])).as_matrix()
    positions=np.zeros((24,3),dtype=np.float64);global_rots=np.zeros((24,3,3))
    global_rots[0]=rotations[0]
    for i in range(1,24):
        parent=SMPL_PARENTS[i]
        positions[i]=positions[parent]+global_rots[parent]@(rest[i]-rest[parent])
        global_rots[i]=global_rots[parent]@rotations[i]
    return positions.astype(np.float32),global_rots.astype(np.float32)


def schema():
    return dict(smpl_joint_names=SMPL_NAMES,smpl_parents=SMPL_PARENTS.tolist(),
        smpl_parameters='single-view learned conversion; global_orient axis-angle radians in community YZ-flipped coordinates, body_pose 21x3 axis-angle, betas 10; final two hand rotations padded zero for FK',
        smpl_display='CV model space, pelvis anchored to MHR hip midpoint; add pred_cam_t for camera projection; not independently estimated metric root translation',
        smpl_smoothing='MLP predicts center of last 5 frames, 2-frame lag when buffer full; explicit source frame/time; only 24 joint positions are smoothed, parameters/rotations remain current RAW predictions',
        smpl_one_euro='causal per-coordinate filter of raw CV model-space 24 joints, current source timestamp; no future lookahead but frequency-dependent phase lag; canonical root coordinates use current raw orientation; positions only, no filtered rotations or exact bone constraints',
        smpl_lag_ms_definition='timestamp lookahead age only; one_euro lag_ms=0 means current source sample, NOT zero phase lag; filter configuration is stored in launch_config',
        smpl_control='pose data only; not robot joint commands; no IK, retargeting or actuator output')


class SmplStage:
    def __init__(self,config,device='cuda',require_model=True):
        import torch
        self.torch=torch;self.config=config;self.device=device
        net_module=load_module('studio_smpl_net',FORK/'multi_view/multiview_net.py')
        self.net=net_module.MinimalMultiViewNet().to(device).eval()
        self.net.load_state_dict(torch.load(WEIGHTS,map_location=device,weights_only=True),strict=True)
        mapping=np.load(ASSETS/'mhr2smpl_mapping.npz',allow_pickle=False)
        samples=np.load(SAMPLES,allow_pickle=False)
        if samples.shape!=(1500,) or samples.min()<0 or samples.max()>=6890:raise ValueError('SMPL 采样映射无效')
        self.ids=np.asarray(mapping['mhr_vert_ids'][samples],dtype=np.int64)
        self.bary=np.asarray(mapping['baryc_coords'][samples],dtype=np.float32)
        if self.ids.shape!=(1500,3) or self.ids.max()>=18439 or self.ids.min()<0 or not np.isfinite(self.bary).all():raise ValueError('SMPL 顶点映射无效')
        self.j_template=self.j_shape=None
        self.asset_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in resources(config) if p.is_file()}
        if require_model:self.load_body_model(model_path(config))
        self.smoother=None;self.window=1
        if config.smpl_smoothing:
            cfg=json.loads((SMOOTHER/'smoother_config.json').read_text())
            if cfg['window_size']!=5 or cfg['joint_dim']!=72:raise ValueError('当前平滑权重需要 5 帧、24 个关节')
            cls=load_module('studio_smpl_smoother',FORK/'smooth/smoother_net.py').SmootherMLP
            self.smoother=cls(window_size=5,joint_dim=72,hidden_dims=cfg['hidden_dims']).to(device).eval()
            self.smoother.load_state_dict(torch.load(SMOOTHER/'smoother_best.pth',map_location=device,weights_only=True),strict=True)
            self.window=5
        from one_euro import OneEuroFilter
        self.euro=OneEuroFilter(config.one_euro_min_cutoff,config.one_euro_beta,config.one_euro_d_cutoff) if config.smpl_one_euro_enabled else None
        self.history=deque(maxlen=self.window);self.sequence=None;self.last_time=None
        self.last_info={}

    def load_body_model(self,path):
        # Official user-provided SMPL asset. Numeric NPZ avoids pickle/chumpy.
        if not path.is_file():raise FileNotFoundError('缺少官方 SMPL neutral 模型：'+str(path))
        if path.suffix.lower()=='.npz':
            with np.load(path,allow_pickle=False) as data:body={k:data[k] for k in ['v_template','shapedirs','J_regressor','kintree_table']}
        else:
            try:
                with path.open('rb') as file:body=pickle.load(file,encoding='latin1')
            except ModuleNotFoundError as exc:
                raise ValueError('旧版 SMPL pickle 依赖 Chumpy；请将模型路径留空，使用已转换的 assets/smpl/SMPL_NEUTRAL.npz') from exc
        template=np.asarray(body['v_template'],dtype=np.float32)
        shapedirs=np.asarray(body['shapedirs'],dtype=np.float32)[:,:,:10]
        regressor=body['J_regressor']
        if hasattr(regressor,'toarray'):regressor=regressor.toarray()
        regressor=np.asarray(regressor,dtype=np.float32)
        parents=np.asarray(body['kintree_table'])[0,1:].astype(np.int64)
        if template.shape!=(6890,3) or shapedirs.shape!=(6890,3,10) or regressor.shape!=(24,6890) or not np.array_equal(parents,SMPL_PARENTS[1:]):raise ValueError('模型不是兼容的 24 关节 neutral SMPL')
        self.j_template=regressor@template
        self.j_shape=np.einsum('jv,vck->jck',regressor,shapedirs,optimize=True)
        if not np.isfinite(self.j_template).all() or not np.isfinite(self.j_shape).all():raise ValueError('SMPL 模型包含无效数值')

    def reset(self,reason='reset'):
        if self.euro:self.euro.reset()
        self.history.clear();self.sequence=None;self.last_time=None;self.reset_reason=reason

    def parameters(self,vertices,translation):
        torch=self.torch
        # Translation cancels in centroid centering; preserve source YZ convention.
        samples=(np.asarray(vertices,dtype=np.float32)[self.ids]*self.bary[:,:,None]).sum(axis=1)@CV_FLIP
        samples-=samples.mean(axis=0,keepdims=True)
        with torch.inference_mode():
            output=self.net.forward_single_view(torch.from_numpy(samples.flatten()).to(self.device).unsqueeze(0))
        values=output[0].float().cpu().numpy()
        a=values[:3];b=values[3:6];a=a/max(np.linalg.norm(a),1e-12);b=b-a*np.dot(a,b);b=b/max(np.linalg.norm(b),1e-12)
        mat=np.stack([a,b,np.cross(a,b)],axis=1)
        if not np.isfinite(values).all() or not np.allclose(mat.T@mat,np.eye(3),atol=1e-4):raise ValueError('SMPL 转换产生无效姿态')
        return Rotation.from_matrix(mat).as_rotvec().astype(np.float32),values[6:69],values[69:79]

    def process(self,person,command=None):
        began=time.perf_counter();command=command or {}
        sequence=command.get('stream_id') if command.get('type')=='stream' else None
        stamp=command.get('host_received_monotonic',time.monotonic())
        if sequence is None or sequence!=self.sequence or self.last_time is None or stamp-self.last_time>.5 or stamp<=self.last_time:
            self.reset('new_sequence_or_gap')
        self.sequence=sequence;self.last_time=stamp
        go,pose,betas=self.parameters(person['pred_vertices'],person['pred_cam_t'])
        person.update(smpl_global_orient=go,smpl_body_pose=pose.copy(),smpl_betas=betas.copy())
        if self.j_template is not None:
            rest=self.j_template+np.einsum('jck,k->jc',self.j_shape,betas)
            joints,rotations=fk_joints(rest,pose)
            to_cv=CV_FLIP@Rotation.from_rotvec(go).as_matrix()
            anchor=np.asarray(person['pred_keypoints_3d'])[[9,10]].mean(axis=0)
            cv_joints=(joints@to_cv.T+anchor).astype(np.float32)
            person.update(smpl_joints_root=joints,smpl_keypoints_3d=cv_joints,smpl_global_rots_cv=np.einsum('ij,njk->nik',to_cv,rotations).astype(np.float32))
            euro_info=None
            if self.euro:
                filter_began=time.perf_counter()
                filtered=self.euro(cv_joints,float(stamp))
                local=((filtered-filtered[0])@to_cv).astype(np.float32)
                euro_info=dict(source_frame_index=command.get('frame_index',0),source_monotonic=stamp,history_frames=self.euro.samples,processing_ms=(time.perf_counter()-filter_began)*1000,causal=True,min_cutoff=self.config.one_euro_min_cutoff,beta=self.config.one_euro_beta,d_cutoff=self.config.one_euro_d_cutoff)
                person.update(smpl_keypoints_3d_one_euro=filtered,smpl_joints_one_euro_root=local,smpl_one_euro_ms=np.array(euro_info['processing_ms']),smpl_one_euro_source_monotonic=np.array(stamp,dtype=np.float64),smpl_one_euro_source_frame_index=np.array(command.get('frame_index',0),dtype=np.int64),smpl_one_euro_history_frames=np.array(self.euro.samples,dtype=np.int64))
            entry=dict(joints=joints,rotation=to_cv,anchor=anchor,frame_index=command.get('frame_index',0),timestamp=stamp)
            self.history.append(entry)
            if self.smoother is not None:
                entries=[self.history[0]]*(self.window-len(self.history))+list(self.history)
                inputs=np.stack([x['joints'] for x in entries]).reshape(1,-1)
                with self.torch.inference_mode():
                    smooth=self.smoother(self.torch.from_numpy(inputs).to(self.device))[0].float().cpu().numpy().reshape(24,3)
                smooth-=smooth[0:1]
                center=entries[self.window//2]
                person.update(smpl_joints_smooth_root=smooth,smpl_keypoints_3d_smooth=(smooth@center['rotation'].T+center['anchor']).astype(np.float32))
                smooth_info=dict(source_frame_index=center['frame_index'],source_monotonic=center['timestamp'],
                    lag_ms=max(0.,(stamp-center['timestamp'])*1000),history_frames=len(self.history),window=5,center_window=True)
            else:smooth_info=None
        else:smooth_info=None;euro_info=None
        for key,value in person.items():
            if key.startswith('smpl_') and isinstance(value,np.ndarray) and not np.isfinite(value).all():raise ValueError('SMPL 结果数值无效')
        self.last_info=dict(enabled=True,smoothing=smooth_info,one_euro=euro_info,postprocess_ms=(time.perf_counter()-began)*1000,
                            raw_frame_index=command.get('frame_index',0),raw_source_monotonic=stamp)
        person['smpl_postprocess_ms']=np.array(self.last_info['postprocess_ms'],dtype=np.float64)
        if smooth_info:
            for key in ('source_frame_index','source_monotonic','lag_ms','history_frames'):
                person['smpl_smooth_'+key]=np.array(smooth_info[key],dtype=np.int64 if key in ('source_frame_index','history_frames') else np.float64)
        return self.last_info
