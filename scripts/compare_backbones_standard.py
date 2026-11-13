#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Compare actual Standard512, Fast384 and Embedl512 workers on frozen poses.

Fast runs YOLO every iteration. Standard uses the same detected boxes and RGB
intrinsics, without mask prompts or depth fitting. Workers run sequentially.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from launch_config import PRESETS
from benchmark_embedl_encoder import memory_guard


def sha256(file):
    with Path(file).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def run(config, label, cases, output, runs):
    folder=output/label;folder.mkdir()
    profile=folder/'config.json';config.save(profile)
    env=dict(os.environ,BODYSTUDIO_CONFIG=str(profile),BODYSTUDIO_SESSION='standard-reference',
             BODYSTUDIO_GPU_KIND='exclusive')
    launcher=ROOT/('launch_model.sh' if config.backend=='standard' else 'launch_fast_model.sh')
    process=subprocess.Popen([str(launcher)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,text=True,bufsize=1,env=env)
    lines=queue.Queue()
    def reader():
        for line in process.stdout:lines.put(line)
        lines.put(None)
    thread=threading.Thread(target=reader,daemon=True);thread.start()
    log=(folder/'worker.log').open('w')
    def until(kind, timeout=900):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            try:line=lines.get(timeout=1)
            except queue.Empty:continue
            if line is None:raise RuntimeError(f'Worker exited: {folder}/worker.log')
            log.write(line);log.flush()
            if not line.startswith('@@SAM3D '):continue
            event=json.loads(line[len('@@SAM3D '):])
            if event['type'] in ('fatal','error'):raise RuntimeError(str(event))
            if event['type']=='stage':print(label,event.get('message'),flush=True)
            if event['type']==kind:return event
        raise TimeoutError(str(folder/'worker.log'))
    reports=[]
    try:
        ready=until('ready');assert ready['actual_config']==config.to_dict()
        (folder/'ready.json').write_text(json.dumps(ready,ensure_ascii=False,indent=2))
        for index,case in enumerate(cases):
            dest=folder/f'pose-{case["id"]:02d}'
            command=dict(type='benchmark',job=str(index),config_id=config.config_id,backend_id=config.backend,
                session_id='standard-reference',inference_type='body',image=case['image'],
                bbox=case['bbox'] if config.backend=='standard' else None,
                camera_intrinsics=case['intrinsics'],output=str(dest),warmup_runs=3,runs=runs)
            process.stdin.write(json.dumps(command)+'\n');process.stdin.flush()
            report=until('mesh',180)['report']
            assert report['actual_config']==config.to_dict() and report['all_outputs_finite']
            assert report['camera_intrinsics_used']['source']=='provided_rgb'
            assert report['vertex_shapes']==[[18439,3]]
            assert report['image_sha256']==case['image_sha256']
            if config.backend=='fast':
                assert report['detector_runs']==config.warmup_runs+(index+1)*(runs+3)
                assert report['selection_roi'] is None
                if label=='fast384':
                    case['bbox']=report['bbox_xyxy']
                    case['bbox_source']='YOLO11m-Pose TensorRT; actual Fast384 selection'
                else:
                    import numpy as np
                    np.testing.assert_allclose(report['bbox_xyxy'],case['bbox'],rtol=0,atol=1e-4)
            reports.append(dict(id=case['id'],mean_ms=report['benchmark']['mean_ms'],
                                output=str(dest),benchmark=report['benchmark']))
            (folder/'cases.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2))
            print(label,case['id'],f'{reports[-1]["mean_ms"]:.2f} ms',flush=True)
        return reports
    finally:
        if process.poll() is None:
            try:
                process.stdin.write('{"type":"stop"}\n');process.stdin.flush();process.wait(timeout=20)
            except (BrokenPipeError,subprocess.TimeoutExpired):
                process.terminate()
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait()
        thread.join(timeout=5);log.close();process.stdin.close();process.stdout.close()


def main():
    import cv2
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dataset',type=Path,default=ROOT/'outputs/rgbd-sets/recording-20261106-135615-851252')
    p.add_argument('--runs',type=int,default=20,help='Repeated timed inference per pose per configuration')
    a=p.parse_args();guard=memory_guard()
    if not 1<=a.runs<=200:raise ValueError('runs must be 1–200')
    a.output.mkdir(parents=True,exist_ok=False)
    source=json.loads((a.dataset/'pose_set.json').read_text())
    cases=[]
    for pose in source['poses']:
        sample=a.dataset/pose['sample'];image=sample/'rgb.png'
        rgb=cv2.imread(str(image));assert rgb is not None
        metadata=json.loads((sample/'camera.json').read_text())
        cases.append(dict(id=pose['id'],name=pose['name'],notes=pose.get('notes',''),
            image=str(image.resolve()),image_sha256=sha256(image),bbox=None,
            bbox_source='to_be_detected_by_Fast384_YOLO',
            intrinsics=metadata['intrinsics']))
    manifest=dict(cases=cases,dataset=str(a.dataset.resolve()),memory_guard=guard,
        scope='Fast executes YOLO every iteration (audited count), Standard consumes the same detected bbox; same original RGB and calibrated K, body only, 3 feedbacks, no masks/depth refinement/SMPL/smoothing; timing includes detector for Fast, excludes load/decode/export/render')
    (a.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    for label,name in [('fast384','演示最快'),('int8512','Fast INT8 512 身体（Embedl）'),('std512','标准版身体')]:
        config=replace(PRESETS[name],warmup_runs=10,warmup_layout='camera').validate()
        run(config,label,cases,a.output,a.runs)
        (a.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print('ALL_VARIANTS_OK',a.output,flush=True)


if __name__=='__main__':main()
