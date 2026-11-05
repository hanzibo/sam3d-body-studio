#!/usr/bin/env python3
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
            assert (output/'prediction/person_00.npz').is_file()
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
