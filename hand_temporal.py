# SPDX-License-Identifier: GPL-3.0-only
"""Causal hand articulation / SO(3) wrist filtering; raw network arrays retained."""
import time
import numpy as np
from scipy.spatial.transform import Rotation
from one_euro import OneEuroFilter


def filter_options(options=None):
    source=options or {}
    result=dict(pose=bool(source.get('pose',False)),position=bool(source.get('position',False)),
                min_cutoff=float(source.get('min_cutoff',1.5)),beta=float(source.get('beta',.5)),
                d_cutoff=float(source.get('d_cutoff',1.)))
    OneEuroFilter(result['min_cutoff'],result['beta'],result['d_cutoff'])
    return result


class RotationOneEuro:
    """Shortest geodesic interpolation with angular-speed adaptive cutoff."""
    def __init__(self,min_cutoff,beta,d_cutoff):
        self.min_cutoff=min_cutoff;self.beta=beta;self.d_cutoff=d_cutoff
        self.last=None;self.filtered=None;self.stamp=None;self.speed=0.
    def __call__(self,matrix,stamp):
        value=Rotation.from_matrix(matrix)
        if self.stamp is None:
            self.filtered=value
        else:
            dt=stamp-self.stamp
            speed=np.linalg.norm((self.last.inv()*value).as_rotvec())/dt
            a=OneEuroFilter.alpha(self.d_cutoff,dt);self.speed=a*speed+(1-a)*self.speed
            a=OneEuroFilter.alpha(self.min_cutoff+self.beta*self.speed,dt)
            delta=(self.filtered.inv()*value).as_rotvec()
            self.filtered=self.filtered*Rotation.from_rotvec(a*delta)
        self.last=value;self.stamp=stamp
        return self.filtered.as_matrix()


class HandTemporalFilter:
    def __init__(self):self.reset()
    def reset(self):self.states={};self.options=None
    def parameters(self,arrays,hands,stamp,options):
        options=filter_options(options)
        if not np.isfinite(stamp):raise ValueError('手部滤波时间戳无效')
        if options!=self.options:self.reset();self.options=options
        current={h['side'] for h in hands}
        self.states={s:v for s,v in self.states.items() if s in current}
        values={k:arrays[k].copy() for k in ('global_rot','hand','pred_cam_t')}
        history=[]
        args=tuple(options[k] for k in ('min_cutoff','beta','d_cutoff'))
        for i,hand in enumerate(hands):
            side=hand['side'];state=self.states.get(side)
            box=np.asarray(hand['bbox'],float)
            if state is not None:
                old=state['box'];distance=np.linalg.norm((box[:2]+box[2:]-old[:2]-old[2:])/2)
                jump=distance>1.5*max(np.linalg.norm(old[2:]-old[:2]),1)
            else:jump=False
            if state is None or not 0<stamp-state['stamp']<=.5 or jump:
                state=dict(rotation=RotationOneEuro(*args),hand=OneEuroFilter(*args),position=OneEuroFilter(*args),samples=0)
                self.states[side]=state
            if options['pose']:
                # Same extrinsic xyz convention consumed by upstream mhr_forward.
                matrix=Rotation.from_euler('xyz',arrays['global_rot'][i]).as_matrix()
                values['global_rot'][i]=Rotation.from_matrix(state['rotation'](matrix,stamp)).as_euler('xyz')
                values['hand'][i]=state['hand'](arrays['hand'][i],stamp)
            if options['position']:values['pred_cam_t'][i]=state['position'](arrays['pred_cam_t'][i],stamp)
            state.update(stamp=stamp,box=box,samples=state['samples']+1);history.append(state['samples'])
        return values,history


def filtered_geometry(engine,arrays,hands,request):
    """Reuse final compact MHR once; never run encoder / decoder a second time."""
    began=time.perf_counter();stamp,options=request
    stage=getattr(engine,'_temporal_filter',None)
    if stage is None:stage=engine._temporal_filter=HandTemporalFilter()
    values,history=stage.parameters(arrays,hands,stamp,options)
    options=stage.options
    display=dict(arrays);display.update(values)
    if options['pose']:
        torch=engine.torch;head=engine.model.head_pose_hand
        def tensor(value):return torch.as_tensor(value,device='cuda',dtype=torch.float32)
        args=dict(global_trans=torch.zeros_like(tensor(values['global_rot'])),global_rot=tensor(values['global_rot']),
                  body_pose_params=tensor(arrays['body_pose']),hand_pose_params=tensor(values['hand']),
                  scale_params=tensor(arrays['scale']),shape_params=tensor(arrays['shape']),expr_params=tensor(arrays['face']),
                  do_pcblend=True,return_keypoints=True,return_joint_coords=True,return_model_params=True,return_joint_rotations=True)
        with torch.inference_mode(),torch.cuda.stream(engine.stream):
            function=getattr(head,'_studio_final_geometry',head.mhr_forward)
            vertices,points,joints,params,rotations=function(**args)
            for key,value in (('pred_vertices',vertices),('pred_joint_coords',joints),('joint_global_rots',rotations),('mhr_model_params',params)):
                value=value.detach().float().cpu().numpy().copy()
                if key in ('pred_vertices','pred_joint_coords'):value[...,1:]*=-1
                display[key]=value
        # Filtered vertices are projected later in original-image coordinates.
    return display,dict(**options,history_frames=history,processing_ms=(time.perf_counter()-began)*1000,
                        scope='hand latent coefficients + SO(3) wrist; optional canonical camera translation',
                        raw_preserved=True,extra_mhr_calls=int(options['pose']))
