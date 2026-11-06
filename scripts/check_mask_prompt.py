#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Sequential no-mask/raw/eroded/dilated inference on one saved SAM3 sample.

Run only after the SAM3 process exits. This tests conditioning and finite output,
not segmentation or pose accuracy. No upstream files or launch caches are edited.
"""
import argparse
from dataclasses import replace
from datetime import datetime
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from launch_config import LaunchConfig
from mask_prompt import morph_mask, sha256


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--backend',choices=['standard','fast'],required=True)
    parser.add_argument('--segmentation',type=Path,required=True,help='SAM3 output directory containing report.json')
    parser.add_argument('--timeout',type=float,default=900)
    args=parser.parse_args()
    sam=json.loads((args.segmentation/'report.json').read_text())
    image=Path(sam['image']);assert sha256(image)==sam['rgb_sha256']
    raw=cv2.imread(str(args.segmentation/f"mask_{sam['selected']:02d}.png"),0)
    assert raw is not None and raw.any()
    x,y,w,h=cv2.boundingRect(raw);bbox=[x,y,x+w,y+h]
    config=LaunchConfig(warmup_runs=2,warmup_layout='camera')
    if args.backend=='standard':
        config=replace(config,backend='standard',image_size=512,trt_backbone=False,trt_detector=False,
                       decoder_compile=False,mhr_compile=False,interm_compile=False)
    config.validate()
    output=ROOT/'outputs/mask-validation'/f'{datetime.now():%Y%m%d-%H%M%S-%f}-{args.backend}'
    output.mkdir(parents=True);profile=output/'config.json';config.save(profile)
    variants=[('off',None),('raw',0),('erode',-8),('dilate',8)]
    commands=[]
    for name,radius in variants:
        dest=output/name;dest.mkdir()
        info=None
        if radius is not None:
            target=dest/'mask_prompt.png';mask=morph_mask(raw,radius)
            assert mask.any();cv2.imwrite(str(target),mask*255)
            info=dict(path=str(target),mask_sha256=sha256(target),image_sha256=sha256(image),
                      morphology_radius_px=radius,original_mask_sha256=sha256(args.segmentation/f"mask_{sam['selected']:02d}.png"))
        command=dict(type='run',job=name,config_id=config.config_id,backend_id=args.backend,session_id='mask-check',
                     image=str(image),bbox=bbox,inference_type=config.mode,output=str(dest),mask_prompt=info)
        (dest/'request.json').write_text(json.dumps(command,indent=2));commands.append(command)
    env=dict(os.environ,BODYSTUDIO_CONFIG=str(profile),BODYSTUDIO_SESSION='mask-check')
    launcher=ROOT/('launch_fast_model.sh' if args.backend=='fast' else 'launch_model.sh')
    process=subprocess.Popen([str(launcher)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                             text=True,bufsize=1,env=env)
    lines=queue.Queue()
    def reader():
        for line in process.stdout: lines.put(line)
        lines.put(None)
    threading.Thread(target=reader,daemon=True).start()
    reports=[];index=0;deadline=time.monotonic()+args.timeout
    def send():
        process.stdin.write(json.dumps(commands[index])+'\n');process.stdin.flush()
    try:
        with (output/'worker.log').open('w') as log:
            while time.monotonic()<deadline:
                try:line=lines.get(timeout=1)
                except queue.Empty:continue
                if line is None:raise RuntimeError('Worker exited: '+str(output/'worker.log'))
                log.write(line);log.flush()
                if not line.startswith('@@SAM3D '):continue
                event=json.loads(line[len('@@SAM3D '):])
                if event['type']=='stage':print(event.get('message',''),flush=True)
                elif event['type'] in ('fatal','error'):raise RuntimeError(event.get('message'))
                elif event['type']=='ready':send()
                elif event['type']=='mesh':
                    report=event['report'];assert report['all_outputs_finite']
                    assert report['mask_prompt']['enabled']==(index>0)
                    reports.append(report);print(f"{variants[index][0]}: {report['inference_seconds']*1000:.1f} ms",flush=True)
                    index+=1
                    if index==len(commands):break
                    send()
        if len(reports)!=4:raise TimeoutError(str(output/'worker.log'))
        assert all(p['bbox_xyxy']==reports[0]['bbox_xyxy'] for p in reports), 'Crop changed between mask variants'
        vertices=[np.load(output/name/'person_00.npz')['pred_vertices'] for name,_ in variants]
        diffs=[float(np.abs(v-vertices[0]).max()) for v in vertices[1:]]
        assert all(d>1e-7 for d in diffs), 'Mask-conditioned predictions must differ from no-mask at the same crop'
        assert not np.array_equal(vertices[1],vertices[2]) and not np.array_equal(vertices[1],vertices[3])
        checks=dict(passed=True,backend=args.backend,config=config.to_dict(),bbox=bbox,
                    variants=[dict(name=n,radius_px=r,inference_ms=p['inference_seconds']*1000,
                                   mask_prompt=p['mask_prompt']) for (n,r),p in zip(variants,reports)],
                    max_vertex_difference_vs_off=diffs,
                    scope='saved RGB-D RGB; fixed selection ROI; raw RGB unchanged; finite mesh and mask sensitivity, not accuracy or camera latency')
        (output/'checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2))
        print(json.dumps(dict(passed=True,output=str(output),max_vertex_difference_vs_off=diffs)),flush=True)
    finally:
        if process.poll() is None:
            try:
                process.stdin.write(json.dumps(dict(type='stop'))+'\n');process.stdin.flush();process.wait(timeout=15)
            except (BrokenPipeError,subprocess.TimeoutExpired):
                process.terminate()
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait()
        process.stdin.close();process.stdout.close()


if __name__=='__main__':main()
