# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Independent causal MHR position filtering. Raw model fields are never replaced."""
import time
import numpy as np
from one_euro import OneEuroFilter

MHR_FILTER_SHAPES = {'mhr_keypoints_3d_one_euro':(70,3),
                     'mhr_joint_coords_one_euro':(127,3),
                     'mhr_keypoints_2d_one_euro':(70,2)}
MHR_FILTER_SCALARS = ('mhr_one_euro_ms','mhr_one_euro_source_monotonic',
                      'mhr_one_euro_source_frame_index','mhr_one_euro_history_frames')

class MhrFilterStage:
    def __init__(self, config):
        self.config=config
        args=(config.one_euro_min_cutoff,config.one_euro_beta,config.one_euro_d_cutoff)
        self.landmarks=OneEuroFilter(*args);self.joints=OneEuroFilter(*args)
        self.reset()

    def reset(self,reason='reset'):
        self.landmarks.reset();self.joints.reset()
        self.sequence=None;self.last_time=None;self.reset_reason=reason

    def process(self, person, command=None, camera_info=None):
        began=time.perf_counter();command=command or {}
        sequence=command.get('stream_id') if command.get('type')=='stream' else None
        stamp=float(command.get('host_received_monotonic',time.monotonic()))
        if not np.isfinite(stamp):raise ValueError('MHR 滤波时间戳无效')
        if sequence is None or sequence!=self.sequence or self.last_time is None or not 0<stamp-self.last_time<=.5:
            self.reset('new_sequence_or_gap')
        for key,shape in [('pred_keypoints_3d',(70,3)),('pred_joint_coords',(127,3))]:
            if np.shape(person[key])!=shape or not np.isfinite(person[key]).all():raise ValueError('MHR 滤波输入无效：'+key)
        points=self.landmarks(person['pred_keypoints_3d'],stamp)
        joints=self.joints(person['pred_joint_coords'],stamp)
        person.update(mhr_keypoints_3d_one_euro=points,mhr_joint_coords_one_euro=joints)
        if camera_info is not None:
            matrix=np.asarray(camera_info['matrix'],dtype=np.float64)
            cam=points.astype(np.float64)+np.asarray(person['pred_cam_t']).reshape(1,3)
            if matrix.shape!=(3,3) or not np.isfinite(matrix).all() or not np.isfinite(cam).all():raise ValueError('MHR 投影内参或相机坐标无效')
            projected=cam@matrix.T
            valid=cam[:,2]>1e-6
            uv=np.zeros((70,2),dtype=np.float64)
            uv[valid]=projected[valid,:2]/projected[valid,2:3]
            valid &= np.isfinite(uv).all(axis=1) & (np.abs(uv)<1e8).all(axis=1)
            uv[~valid]=0
            person['mhr_keypoints_2d_one_euro']=uv.astype(np.float32)
            person['mhr_projection_valid']=valid
        cost=(time.perf_counter()-began)*1000
        person.update(mhr_one_euro_ms=np.array(cost),mhr_one_euro_source_monotonic=np.array(stamp),
                      mhr_one_euro_source_frame_index=np.array(command.get('frame_index',0),dtype=np.int64),
                      mhr_one_euro_history_frames=np.array(self.landmarks.samples,dtype=np.int64))
        info=dict(processing_ms=cost,source_frame_index=command.get('frame_index',0),source_monotonic=stamp,
                  history_frames=self.landmarks.samples,reset_reason=self.reset_reason,
                  min_cutoff=self.config.one_euro_min_cutoff,beta=self.config.one_euro_beta,d_cutoff=self.config.one_euro_d_cutoff,
                  scope='MHR70 landmarks and MHR127 joint positions only; raw mesh, rotations, parameters and camera translation unchanged')
        self.sequence=sequence;self.last_time=stamp
        return info
