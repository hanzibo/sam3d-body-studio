#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Real one-image GPU acceptance; writes private artifacts only under outputs/."""
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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from launch_config import LaunchConfig
from studio_paths import path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=['standard','fast'],required=True)
    parser.add_argument('--eager',action='store_true',help='Fast 512 without TRT or torch.compile')
    parser.add_argument('--stream-smoke',action='store_true',help='Also send three different recorded Gemini images through the real stream protocol')
    parser.add_argument('--timeout',type=float,default=600)
    args=parser.parse_args()
    config=LaunchConfig(warmup_runs=2,warmup_layout='sample')
    if args.backend=='standard' or args.eager:
        config=replace(config,backend=args.backend,image_size=512,trt_backbone=False,trt_detector=False,
                       decoder_compile=False,mhr_compile=False,interm_compile=False)
    config.validate()
    output=ROOT/'outputs/release-validation'/f'{datetime.now():%Y%m%d-%H%M%S-%f}-{args.backend}'
    output.mkdir(parents=True)
    profile=output/'config.json';profile.write_text(json.dumps(config.to_dict(),indent=2))
    image=path('official_repo')/'notebook/images/dancing.jpg'
    if not image.is_file():raise FileNotFoundError(image)
    env=dict(os.environ,BODYSTUDIO_CONFIG=str(profile),BODYSTUDIO_SESSION='release-check')
    launcher=ROOT/('launch_fast_model.sh' if args.backend=='fast' else 'launch_model.sh')
    process=subprocess.Popen([str(launcher)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                             text=True,bufsize=1,env=env)
    lines=queue.Queue()
    def read():
        for line in process.stdout:lines.put(line)
        lines.put(None)
    threading.Thread(target=read,daemon=True).start()
    deadline=time.monotonic()+args.timeout
    ready=False;report=None
    try:
        with (output/'worker.log').open('w') as log:
            while time.monotonic()<deadline:
                try:line=lines.get(timeout=1)
                except queue.Empty:continue
                if line is None:raise RuntimeError('Worker exited; see '+str(output/'worker.log'))
                log.write(line);log.flush()
                if not line.startswith('@@SAM3D '):continue
                event=json.loads(line[len('@@SAM3D '):])
                if event['type']=='stage':print(event.get('message',''),flush=True)
                if event['type'] in ('fatal','error'):raise RuntimeError(event.get('message',str(event)))
                if event['type']=='ready':
                    assert event['actual_config']==config.to_dict()
                    assert not ready
                    ready=True
                    command=dict(type='run',job='release-check',config_id=config.config_id,
                                 backend_id=args.backend,session_id='release-check',image=str(image),
                                 bbox=[732,114,1523,1451],inference_type='body',output=str(output/'prediction'))
                    process.stdin.write(json.dumps(command)+'\n');process.stdin.flush()
                elif event['type']=='mesh':
                    report=event['report'];break
            if report is None:raise TimeoutError('No mesh before timeout; see '+str(output/'worker.log'))
            assert ready and report['actual_config']==config.to_dict()
            assert report['model_inference_completed'] and report['all_outputs_finite']
            assert report['vertex_shapes']==[[18439,3]]
            if args.backend=='fast':
                keypoints=report['keypoint_path']
                assert keypoints['keypoints']==70 and keypoints['vertices']==468
                expected_graph=bool(keypoints['enabled'] and config.interm_compile and config.body_interm_layers)
                assert keypoints['graph_captured']==expected_graph
                execution=report['execution']
                if execution['final_replay'] is not None:
                    assert execution['final_replay']['graph_captured']==execution['final_replay']['enabled']
                assert execution['body_io']['enabled']==(os.environ.get('BODYSTUDIO_BODY_IO','1')!='0')
            assert (output/'prediction/person_00.npz').is_file()
            stream_rows=[]
            if args.stream_smoke:
                import base64
                import cv2
                import numpy as np
                source=ROOT/'outputs/rgbd-sets/recording-20261106-135615-851252'
                poses=json.loads((source/'pose_set.json').read_text())['poses']
                stream_dir=output/'stream';stream_dir.mkdir()
                for index,row in enumerate([x for x in poses if x.get('sample')][:3]):
                    rgb=cv2.imread(str(source/row['sample']/'rgb.png'));assert rgb is not None
                    ok,jpeg=cv2.imencode('.jpg',rgb);assert ok
                    command=dict(type='stream',job=f'stream-{index}',stream_id='acceptance',frame_index=index,
                        host_received_monotonic=time.monotonic(),jpeg=base64.b64encode(jpeg).decode(),
                        config_id=config.config_id,backend_id=args.backend,session_id='release-check',
                        inference_type='body',output=str(stream_dir))
                    process.stdin.write(json.dumps(command)+'\n');process.stdin.flush()
                    until=time.monotonic()+60
                    while time.monotonic()<until:
                        try:line=lines.get(timeout=1)
                        except queue.Empty:continue
                        if line is None:raise RuntimeError('Worker exited during stream check')
                        log.write(line);log.flush()
                        if not line.startswith('@@SAM3D '):continue
                        event=json.loads(line[len('@@SAM3D '):])
                        if event['type'] in ('fatal','error'):raise RuntimeError(str(event))
                        if event['type']!='stream_mesh':continue
                        assert event['job']==command['job'] and event['frame_index']==index
                        artifact=Path(event['path'])
                        with np.load(artifact,allow_pickle=False) as arrays:
                            assert arrays['pred_vertices'].shape==(18439,3)
                            assert np.isfinite(arrays['pred_vertices']).all()
                        artifact.rename(stream_dir/f'frame-{index:02d}.npz')
                        stream_rows.append(dict(frame_index=index,inference_ms=event['inference_seconds']*1000,
                                                worker_ms=event['worker_seconds']*1000))
                        break
                    else:raise TimeoutError('Stream smoke check timed out')
                (output/'stream-checks.json').write_text(json.dumps(dict(passed=True,frames=stream_rows,
                    scope='Recorded Gemini frames through production stream protocol; no live-camera/display FPS measurement'),indent=2))
            (output/'checks.json').write_text(json.dumps(dict(passed=True,backend=args.backend,
                config_id=config.config_id,vertex_shapes=report['vertex_shapes'],
                inference_ms=report['inference_seconds']*1000,scope='one image, two warmups; no accuracy or camera benchmark'),indent=2))
            print(json.dumps(dict(passed=True,backend=args.backend,inference_ms=report['inference_seconds']*1000,
                                  output=str(output)),ensure_ascii=False),flush=True)
    finally:
        if process.poll() is None:
            try:
                process.stdin.write(json.dumps(dict(type='stop',backend_id=args.backend,session_id='release-check'))+'\n')
                process.stdin.flush();process.wait(timeout=15)
            except (BrokenPipeError,subprocess.TimeoutExpired):
                process.terminate()
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait()
        process.stdout.close();process.stdin.close()

if __name__=='__main__':main()
