# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Bounded asynchronous raw motion recording, readable JSONL and CSV."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import queue
import threading
import time
import numpy as np
from pose_data import NAMES, ANGLE_TRIPLES, joint_angles, schema
from smpl_stage import SMPL_NAMES
SMPL_ANGLES = {"left_elbow":(16,18,20), "right_elbow":(17,19,21), "left_knee":(1,4,7), "right_knee":(2,5,8)}


def json_default(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)


class MotionRecorder:
    def __init__(self, root, context):
        self.path = Path(root)/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        self.path.mkdir(parents=True)
        self.started = time.monotonic()
        self.unix_offset = time.time()-self.started
        self.context = dict(schema(), started_utc=datetime.now(timezone.utc).isoformat(), **context)
        self.queue = queue.Queue(maxsize=128)
        self.stop_event = threading.Event()
        self.count = self.gaps = 0
        self.error = None
        self.reason = ''
        self._metadata('recording')
        self.thread = threading.Thread(target=self._write, name='motion-writer', daemon=True)
        self.thread.start()

    def _metadata(self, status):
        data = dict(self.context, status=status, frames=self.count, gaps=self.gaps, error=self.error,
                    stop_reason=self.reason, duration_seconds=time.monotonic()-self.started)
        temp = self.path/'session.tmp'
        temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,default=json_default,allow_nan=False))
        temp.replace(self.path/'session.json')

    def append(self, event):
        if self.stop_event.is_set() or self.error: return False
        try: self.queue.put_nowait(event); return True
        except queue.Full:
            self.error = '写盘队列已满，录制停止；未静默丢弃数据'
            self.close('writer_overflow')
            return False

    def close(self, reason='user_stop'):
        self.reason = reason
        self.stop_event.set()

    @property
    def done(self): return not self.thread.is_alive()

    def _write(self):
        try:
            with (self.path/'frames.jsonl').open('w') as frames, (self.path/'angles.csv').open('w',newline='') as angles, (self.path/'keypoints.csv').open('w',newline='') as points, (self.path/'smpl_parameters.csv').open('w',newline='') as params, (self.path/'smpl_joints.csv').open('w',newline='') as smpl_points, (self.path/'smpl_angles.csv').open('w',newline='') as smpl_angles, (self.path/'mhr_filtered_keypoints.csv').open('w',newline='') as filtered_points, (self.path/'mhr_joints.csv').open('w',newline='') as mhr_joints, (self.path/'mhr_filtered_angles.csv').open('w',newline='') as filtered_angles:
                aw,pw = csv.writer(angles),csv.writer(points)
                aw.writerow(['frame_index','elapsed_seconds','host_received_unix','inference_ms',*ANGLE_TRIPLES])
                pw.writerow(['frame_index','elapsed_seconds','index','name','x_model_m','y_model_m','z_model_m','u_image_px','v_image_px'])
                sp,sj,sa=csv.writer(params),csv.writer(smpl_points),csv.writer(smpl_angles)
                sp.writerow(['frame_index','host_received_monotonic',*[f'global_orient_{i}' for i in range(3)],*[f'body_pose_{i}' for i in range(63)],*[f'beta_{i}' for i in range(10)]])
                sj.writerow(['frame_index','variant','source_frame_index','source_monotonic','lag_ms','index','name','x_root_m','y_root_m','z_root_m','x_cv_model_m','y_cv_model_m','z_cv_model_m'])
                sa.writerow(['frame_index','variant','source_frame_index','source_monotonic','lag_ms',*SMPL_ANGLES])
                fp,mj,fa=csv.writer(filtered_points),csv.writer(mhr_joints),csv.writer(filtered_angles)
                provenance=['frame_index','source_frame_index','source_monotonic','history_frames','filter_ms']
                fp.writerow([*provenance,'index','name','x_model_m','y_model_m','z_model_m','u_image_px','v_image_px','projection_valid'])
                mj.writerow(['frame_index','variant','source_monotonic','index','x_model_m','y_model_m','z_model_m'])
                fa.writerow([*provenance,*ANGLE_TRIPLES])
                last_flush = time.monotonic()
                while not self.stop_event.is_set() or not self.queue.empty():
                    try: event=self.queue.get(timeout=.1)
                    except queue.Empty: continue
                    frame=dict(event)
                    received=frame['host_received_monotonic']
                    frame['elapsed_seconds']=received-self.started
                    frame['host_received_unix']=received+self.unix_offset
                    if frame['type']=='pose':
                        pose=frame['pose'];frame['angles_deg']=joint_angles(pose['pred_keypoints_3d'])
                        aw.writerow([frame['frame_index'],frame['elapsed_seconds'],frame['host_received_unix'],frame['inference_ms'],*frame['angles_deg'].values()])
                        for i,(p,uv) in enumerate(zip(pose['pred_keypoints_3d'],pose['pred_keypoints_2d'])):
                            pw.writerow([frame['frame_index'],frame['elapsed_seconds'],i,NAMES[i],*p,*uv])
                        for variant,key in [('raw','pred_joint_coords'),('one_euro','mhr_joint_coords_one_euro')]:
                            if key not in pose:continue
                            stamp=float(pose['mhr_one_euro_source_monotonic']) if variant=='one_euro' else received
                            for i,p in enumerate(pose[key]):mj.writerow([frame['frame_index'],variant,stamp,i,*p])
                        if 'mhr_keypoints_3d_one_euro' in pose:
                            provenance=[frame['frame_index'],int(pose['mhr_one_euro_source_frame_index']),float(pose['mhr_one_euro_source_monotonic']),int(pose['mhr_one_euro_history_frames']),float(pose['mhr_one_euro_ms'])]
                            filtered=pose['mhr_keypoints_3d_one_euro']
                            frame['mhr_one_euro_angles_deg']=joint_angles(filtered)
                            fa.writerow([*provenance,*frame['mhr_one_euro_angles_deg'].values()])
                            uv=pose.get('mhr_keypoints_2d_one_euro');mask=pose.get('mhr_projection_valid')
                            for i,p in enumerate(filtered):
                                fp.writerow([*provenance,i,NAMES[i],*p,*(uv[i] if uv is not None else ['','']),bool(mask[i]) if mask is not None else False])
                        if 'smpl_body_pose' in pose:
                            sp.writerow([frame['frame_index'],received,*pose['smpl_global_orient'],*pose['smpl_body_pose'],*pose['smpl_betas']])
                            frame['smpl_angles_deg']={}
                            for variant,root_key,cv_key in [('raw','smpl_joints_root','smpl_keypoints_3d'),('smooth','smpl_joints_smooth_root','smpl_keypoints_3d_smooth'),('one_euro','smpl_joints_one_euro_root','smpl_keypoints_3d_one_euro')]:
                                if root_key not in pose:continue
                                source=int(pose['smpl_smooth_source_frame_index']) if variant=='smooth' else frame['frame_index']
                                stamp=float(pose['smpl_smooth_source_monotonic']) if variant=='smooth' else received
                                lag=float(pose['smpl_smooth_lag_ms']) if variant=='smooth' else 0.
                                if variant=='one_euro':
                                    source=int(pose['smpl_one_euro_source_frame_index']);stamp=float(pose['smpl_one_euro_source_monotonic'])
                                    lag=0. # No lookahead; not a measurement of phase lag.
                                angle=joint_angles(pose[root_key],SMPL_ANGLES)
                                frame['smpl_angles_deg'][variant]=angle
                                sa.writerow([frame['frame_index'],variant,source,stamp,lag,*angle.values()])
                                for i,(p,cv) in enumerate(zip(pose[root_key],pose[cv_key])):
                                    sj.writerow([frame['frame_index'],variant,source,stamp,lag,i,SMPL_NAMES[i],*p,*cv])
                        self.count+=1
                    else: self.gaps+=1
                    frames.write(json.dumps(frame,ensure_ascii=False,default=json_default,allow_nan=False)+'\n')
                    if time.monotonic()-last_flush>1:
                        for file in (frames,angles,points,params,smpl_points,smpl_angles,filtered_points,mhr_joints,filtered_angles): file.flush()
                        last_flush=time.monotonic()
        except Exception as exc:
            self.error=str(exc)
            self.stop_event.set()
        finally:
            try: self._metadata('error' if self.error else 'complete')
            except OSError as exc: self.error=str(exc)
