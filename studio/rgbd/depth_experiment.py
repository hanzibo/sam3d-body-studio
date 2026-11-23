# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Resumable static RGB-D benchmark. SAM3 must exit before body workers start."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import cv2
from studio.core.launch_config import PRESETS
from studio.video.video_pipeline import BodyProcess
from studio_paths import path, ROOT


def save(file,value):
    file=Path(file);temp=file.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2));temp.replace(file)


def main():
    p=argparse.ArgumentParser();p.add_argument('--masks',required=True);p.add_argument('--output',required=True)
    p.add_argument('--profiles',nargs='+',default=['standard512','fast512','fast384']);p.add_argument('--evaluate-only',action='store_true')
    args=p.parse_args();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    rows=json.loads(Path(args.masks).read_text());configs=dict(standard512=PRESETS['标准版身体'],fast512=PRESETS['身体 512 对照'],fast384=PRESETS['演示最快'])
    for name in args.profiles:
        config=replace(configs[name],warmup_runs=3,warmup_layout='camera');folder=out/name;folder.mkdir(exist_ok=True)
        body=None
        try:
            for row in rows:
                dest=folder/f"pose-{row['pose']:03d}";dest.mkdir(exist_ok=True)
                sample=Path(row['sample']);mask=Path(row['mask'])
                if hashlib.sha256((sample/'rgb.png').read_bytes()).hexdigest()!=row['rgb_sha256']:raise ValueError('RGB changed')
                if hashlib.sha256(mask.read_bytes()).hexdigest()!=row['mask_sha256']:raise ValueError('mask changed')
                prediction=dest/'prediction'/'person_00.npz'
                if not prediction.is_file() and not args.evaluate_only:
                    if body is None:
                        body=BodyProcess(config,folder);ready=body.event('ready',timeout=1800)
                        if ready['actual_config']!=config.to_dict():raise ValueError('actual config mismatch')
                        save(folder/'ready.json',ready)
                    x,y,w,h=cv2.boundingRect(cv2.imread(str(mask),0));camera=json.loads((sample/'camera.json').read_text())
                    command=dict(type='benchmark',job=f"pose-{row['pose']}",image=str(sample/'rgb.png'),output=str(dest/'prediction'),
                        bbox=[x,y,x+w,y+h],inference_type=config.mode,camera_intrinsics=camera['intrinsics'],
                        camera_intrinsics_origin='frozen RGB-D capture camera.json',runs=5,warmup_runs=2,
                        config_id=config.config_id,backend_id=config.backend,session_id=body.session)
                    save(dest/'request.json',command);body.process.stdin.write(json.dumps(command)+'\n');body.process.stdin.flush()
                    event=body.event('mesh',timeout=1800);save(dest/'event.json',event)
                    print(f"{name} pose {row['pose']}: {event['report']['benchmark']['mean_ms']:.1f}ms",flush=True)
        finally:
            if body is not None:body.close()
        # Body process has exited and released VRAM before EGL evaluation.
        for row in rows:
            dest=folder/f"pose-{row['pose']:03d}";prediction=dest/'prediction'/'person_00.npz'
            if not prediction.is_file():continue
            if (dest/'evaluation'/'refinement.json').is_file():continue
            if (dest/'evaluation').exists():raise ValueError('incomplete evaluation exists; preserve and rename before retry')
            with (dest/'evaluation.log').open('w') as log:
                result=subprocess.run([str(path('standard_python')),'-u','-m','studio.rgbd.rgbd_evaluate',
                    '--sample',row['sample'],'--prediction',str(prediction),'--mask',row['mask'],
                    '--output',str(dest/'evaluation'),'--refine'],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
            if result.returncode:raise RuntimeError(f'evaluation failed: {dest}/evaluation.log')
            metric=json.loads((dest/'evaluation'/'metrics.json').read_text());ref=metric['depth_refinement']
            print(f"{name} pose {row['pose']}: {metric['mae_mm']:.1f} -> {ref['metrics']['mae_mm']:.1f} mm / {ref['reason']}",flush=True)
    print('complete',flush=True)

if __name__=='__main__':main()
