# SPDX-License-Identifier: GPL-3.0-only
"""Offline temporal RGB-D experiment. Online candidate never reads SAM3 masks.

Commands use configured isolated environments. Outputs are immutable by default.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import cv2
import numpy as np
from rgbd_sequence import RGBDSequence
from rgbd_tracking_points import SequenceGeometry,sampled_indices
from video_project import save_json,file_hash,put_segment,add_track,SCHEMA

ROOT=Path(__file__).resolve().parent


def prepare(sequence,output,hz):
    from gpu_lease import acquire_gpu_lease
    from backends import YOLO_ENGINE
    from engine_registry import verify_engine
    from tools.build_detector import HumanDetector
    import torch
    lease=acquire_gpu_lease('exclusive')
    verify_engine(YOLO_ENGINE,file_hash(YOLO_ENGINE))
    detector=HumanDetector(name='yolo_pose',device='cuda',model=str(YOLO_ENGINE))
    source=RGBDSequence(sequence);geometry=SequenceGeometry(source)
    output=Path(output);output.mkdir(parents=True,exist_ok=False);(output/'frames').mkdir()
    indices=sampled_indices(source.frames,hz);rows=[]
    for local,index in enumerate(indices):
        began=time.perf_counter()
        raw,_,row=source.read(index)
        image=cv2.remap(raw,*geometry.maps,cv2.INTER_LINEAR)
        torch.cuda.synchronize();start=time.perf_counter()
        detected=detector.run_human_detection(image, default_to_full_image=False)
        torch.cuda.synchronize();seconds=time.perf_counter()-start
        boxes=np.asarray(detected['boxes']).reshape(-1,4);points=np.asarray(detected['keypoints'])
        # Single-person recordings only; reject ambiguity rather than switch identity.
        valid=len(boxes)==1 and len(points)==1 and (points[0,[5,6,11,12],2]>.3).sum()>=2
        file=output/'frames'/f'{local:06d}.png'
        if not cv2.imwrite(str(file),image):raise OSError(str(file))
        rows.append(dict(frame=local,source_index=index,time_s=row['relative_time_s'],rgb_sha256=file_hash(file),
                         bbox=boxes[0].tolist() if valid else None,keypoints=points[0].tolist() if valid else None,
                         detector_seconds=seconds,prepare_seconds=time.perf_counter()-began))
        if local%20==0:print('prepare',source.meta['name'],local,len(indices),flush=True)
    k=geometry.k
    project=dict(schema=SCHEMA,width=k['width'],height=k['height'],frame_count=len(rows),pts=[r['time_s'] for r in rows],
                 source=str(source.directory),source_sha256=file_hash(source.directory/'frames.jsonl'),tracks=[],
                 duration=rows[-1]['time_s']+1/hz,has_audio=False,source_kind='rgbd_tracking',
                 rgbd_source_indices=indices,source_time_origin=0.)
    track=add_track(project,'人体（自动肩髋中心提示，需核查分割）')
    starts=[i for i,r in enumerate(rows) if r['bbox'] and (i==0 or not rows[i-1]['bbox'])]
    for start in starts:
        end=start
        while end+1<len(rows) and rows[end+1]['bbox']:end+=1
        kp=np.array(rows[start]['keypoints']);good=kp[[5,6,11,12]];good=good[good[:,2]>.3]
        center=good[:,:2].mean(0);x1,y1,x2,y2=rows[start]['bbox']
        prompts=[[float(center[0]),float(center[1]),1]]
        # A torso-only click often segments the shirt. Include head/limbs to
        # express the intended whole-person object, not a clothing instance.
        for j in (0,7,8,13,14):
            x,y,confidence=kp[j]
            if confidence>.55 and 0<=x<k['width'] and 0<=y<k['height']:prompts.append([float(x),float(y),1])
        for x,y in [(max(0,x1-25),max(0,y1-25)),(min(k['width']-1,x2+25),min(k['height']-1,y2+25))]:prompts.append([float(x),float(y),0])
        put_segment(project,track['id'],start,end,prompts)
    save_json(output/'project.json',project)
    manifest=dict(schema='studio_rgbd_tracking_inputs_v1',source=str(source.directory),name=source.meta['name'],
                  index_sha256=file_hash(source.directory/'frames.jsonl'),camera_sha256=file_hash(source.directory/'camera.json'),
                  hz=hz,k=k,coordinate_grid='undistorted colour; original K',frames=rows,
                  prompt_source='automatic torso/head/elbow/knee foreground + exterior background; SAM3 is reference, not ground truth')
    save_json(output/'inputs.json',manifest)
    print('PREPARED',output,flush=True)


def segment(output):
    from sam3_video_worker import segment_video
    from video_project import load_project,mask_at
    output=Path(output)
    segment_video(output,lambda kind,**kw:print(kind,kw,flush=True))
    directory,project=load_project(output);dest=output/'reference_masks';dest.mkdir(exist_ok=False)
    rows=[]
    for i in range(project['frame_count']):
        source=mask_at(directory,project,project['tracks'][0],i)
        mask=cv2.imread(str(source),0) if source else np.zeros((project['height'],project['width']),np.uint8)
        if mask is None:raise ValueError('mask missing')
        file=dest/f'{i:06d}.png';cv2.imwrite(str(file),mask)
        rows.append(dict(frame=i,pixels=int((mask>0).sum()),sha256=file_hash(file),segmented=source is not None,source=str(source) if source else 'no unambiguous YOLO person; NOT independently SAM3-tested'))
    save_json(dest/'manifest.json',dict(grid='undistorted colour',frames=rows,source='SAM3 temporal propagation with automatic point prompts; absent intervals detector-labelled'))


def track(output,steps,modes):
    from studio_paths import environment,path
    from launch_config import LaunchConfig
    output=Path(output).resolve();config=LaunchConfig(warmup_runs=3,warmup_layout='sample')
    config.save(output/'launch.json')
    env=environment('fast');env.update(BODYSTUDIO_CONFIG=str(output/'launch.json'),BODYSTUDIO_GPU_KIND='exclusive')
    commands=[]
    for mode in modes:
        commands.append(dict(type='rgbd_track',job=mode,config_id=config.config_id,directory=str(output),mode=mode,steps=steps))
    commands.append(dict(type='stop'))
    # Stream output to a file; avoid pipe deadlocks from compiler diagnostics.
    with (output/('tracking-worker-'+'-'.join(modes)+f'-{steps}.log')).open('x') as log:
        result=subprocess.run([str(path('fast_python')),str(ROOT/'fast_worker.py')],input=''.join(json.dumps(c)+'\n' for c in commands),text=True,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=ROOT)
    if result.returncode:raise RuntimeError('tracking worker failed; see tracking-worker.log')
    for mode in modes:
        if not (output/f'{mode}-{steps}'/'complete.json').exists():raise RuntimeError('incomplete '+mode+'; see tracking-worker.log')


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('sequence');prep.add_argument('output');prep.add_argument('--hz',type=float,default=10.)
    seg=sub.add_parser('segment');seg.add_argument('output')
    tr=sub.add_parser('track');tr.add_argument('output');tr.add_argument('--steps',type=int,choices=[5,10,15,30],default=15);tr.add_argument('--modes',nargs='+',choices=['candidate','oracle'],default=['candidate','oracle'])
    ev=sub.add_parser('evaluate');ev.add_argument('output');ev.add_argument('--steps',type=int,default=15)
    p.add_argument('--internal',action='store_true');args=p.parse_args()
    if args.command in ('prepare','segment') and not args.internal:
        from studio_paths import environment,path
        kind='fast' if args.command=='prepare' else 'sam3'
        raise SystemExit(subprocess.call([str(path(kind+'_python')),str(Path(__file__).resolve()),'--internal',*sys.argv[1:]],env=environment(kind),cwd=ROOT))
    if args.command=='prepare':prepare(args.sequence,args.output,args.hz)
    elif args.command=='segment':segment(args.output)
    elif args.command=='track':track(args.output,args.steps,args.modes)
    else:
        from rgbd_tracking_report import evaluate
        evaluate(Path(args.output),args.steps)

if __name__=='__main__':main()
