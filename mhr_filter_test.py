"""MHR filtering independence, projection, raw preservation, reset and export."""
from dataclasses import replace
import csv,json,tempfile,time
from pathlib import Path
import numpy as np
from launch_config import LaunchConfig,PRESETS
from mhr_filter import MhrFilterStage
from pose_data import extract_pose,joint_angles
from motion_recording import MotionRecorder

config=PRESETS['MHR 骨架＋One Euro'];assert not config.smpl_enabled
config.validate();assert config.mhr_one_euro_enabled and not config.smpl_one_euro_enabled
legacy=PRESETS['SMPL 身体＋One Euro'].to_dict();legacy.pop('one_euro_target')
assert LaunchConfig.from_dict(legacy).one_euro_target=='smpl'
assert replace(config,smpl_enabled=True,one_euro_target='both').validate().smpl_one_euro_enabled
rng=np.random.default_rng(42)
raw=dict(pred_keypoints_3d=rng.normal(0,.1,(70,3)).astype('f'),pred_joint_coords=rng.normal(0,.1,(127,3)).astype('f'),pred_keypoints_2d=np.zeros((70,2),'f'),pred_global_rots=np.tile(np.eye(3,dtype='f'),(127,1,1)),pred_cam_t=np.array([0,0,3.],'f'))
K=np.array([[600,0,640],[0,610,400],[0,0,1.]])
stage=MhrFilterStage(config)
def process(stamp,index=0,stream='a',delta=0):
 person={k:v.copy() for k,v in raw.items()}
 person['pred_keypoints_3d']+=delta;person['pred_joint_coords']+=delta
 before={k:v.copy() for k,v in person.items()}
 info=stage.process(person,dict(type='stream',stream_id=stream,host_received_monotonic=stamp,frame_index=index),dict(matrix=K))
 for k in before:np.testing.assert_array_equal(person[k],before[k])
 return extract_pose(person),info
first,_=process(1.)
np.testing.assert_array_equal(first['mhr_keypoints_3d_one_euro'],raw['pred_keypoints_3d'])
cam=raw['pred_keypoints_3d']+raw['pred_cam_t'];projection=cam@K.T
np.testing.assert_allclose(first['mhr_keypoints_2d_one_euro'],projection[:,:2]/projection[:,2:3],atol=1e-4)
second,_=process(1.13,2,delta=.2)
assert int(second['mhr_one_euro_history_frames'])==2
assert np.linalg.norm(second['mhr_keypoints_3d_one_euro']-first['mhr_keypoints_3d_one_euro'])<np.linalg.norm(second['pred_keypoints_3d']-raw['pred_keypoints_3d'])
for stamp,stream in [(1.1,'a'),(1.1,'a'),(2.,'a'),(2.1,'b')]:
 p,_=process(stamp,stream=stream);assert int(p['mhr_one_euro_history_frames'])==1
stage.reset('frame_error');assert int(process(2.2,stream='b')[0]['mhr_one_euro_history_frames'])==1
# Behind-camera points are excluded from projected lines, while exports remain finite.
p={k:v.copy() for k,v in raw.items()};p['pred_cam_t'][2]=-3
stage.process(p,camera_info=dict(matrix=K));assert not p['mhr_projection_valid'].any();extract_pose(p)
with tempfile.TemporaryDirectory() as tmp:
 rec=MotionRecorder(Path(tmp),dict(launch_config=config.to_dict()))
 rec.append(dict(type='pose',frame_index=2,host_received_monotonic=rec.started,inference_ms=10.,pose=second))
 rec.close();rec.thread.join(5);assert rec.done and not rec.error,rec.error
 with (rec.path/'frames.jsonl').open() as f:frame=json.loads(f.readline())
 assert frame['mhr_one_euro_angles_deg']==joint_angles(second['mhr_keypoints_3d_one_euro'])
 for filename,count in [('keypoints.csv',70),('mhr_filtered_keypoints.csv',70),('mhr_joints.csv',254),('angles.csv',1),('mhr_filtered_angles.csv',1)]:
  with (rec.path/filename).open() as f:assert len(list(csv.DictReader(f)))==count
 np.testing.assert_array_equal(frame['pose']['pred_global_rots'],second['pred_global_rots'])
print('PASS: MHR independent filtering, raw data, projection mask, resets, legacy profiles and CSV/JSONL export')
# Optional model acceptance uses an explicitly supplied private fixture.
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--smpl-input',type=Path,help='Optional local prediction NPZ; requires SMPL conversion assets')
args=parser.parse_args()
if args.smpl_input is not None:
    # Both targets coexist; each filters raw predictions, never each other's output.
    from smpl_stage import SmplStage
    path=args.smpl_input
    both=replace(config,smpl_enabled=True,smpl_smoothing=True,one_euro_target='both')
    smpl=SmplStage(both);mhr=MhrFilterStage(both)
    with np.load(path,allow_pickle=False) as data:original={k:data[k].copy() for k in data.files}
    for i in range(6):
     p={k:v.copy() for k,v in original.items()}
     if 'pred_vertices' not in p:p['pred_vertices']=p['vertices']
     command=dict(type='stream',stream_id='both',host_received_monotonic=10+i*.12,frame_index=i)
     smpl.process(p,command);mhr.process(p,command,dict(matrix=K));pose=extract_pose(p)
     assert int(pose['smpl_one_euro_history_frames'])==i+1==int(pose['mhr_one_euro_history_frames'])
     assert int(pose['smpl_smooth_source_frame_index'])==max(0,i-2)
     for key in ('pred_keypoints_3d','pred_joint_coords','pred_global_rots','pred_cam_t'):np.testing.assert_array_equal(pose[key],original[key])
    # Enabling conversion while selecting MHR only must not also create SMPL Euro output.
    conversion=SmplStage(replace(both,one_euro_target='mhr',smpl_smoothing=False))
    assert conversion.euro is None
    print('PASS: combined MHR/SMPL/MLP states and conversion with MHR-only target')
