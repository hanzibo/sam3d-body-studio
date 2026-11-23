#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Benchmark actual Studio workers sequentially; keep artifacts on Data.

Run under a systemd memory-limited scope. Never loads both model workers at once.
"""
import argparse
import base64
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
from studio.core.launch_config import PRESETS
from benchmark_embedl_encoder import memory_guard


def run_variant(config, output, image, runs):
    import cv2
    import numpy as np
    output.mkdir(parents=True)
    profile=output/'config.json';config.save(profile)
    env=dict(os.environ,BODYSTUDIO_CONFIG=str(profile),BODYSTUDIO_SESSION='embedl-profile-check',
             BODYSTUDIO_GPU_KIND='exclusive')
    process=subprocess.Popen([str(ROOT/'launch_fast_model.sh')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,text=True,bufsize=1,env=env)
    lines=queue.Queue()
    def reader():
        for line in process.stdout:lines.put(line)
        lines.put(None)
    thread=threading.Thread(target=reader,daemon=True);thread.start()
    log=(output/'worker.log').open('w')
    def event_until(kind, timeout=900):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            try:line=lines.get(timeout=1)
            except queue.Empty:continue
            if line is None:raise RuntimeError(f'Worker exited ({process.poll()}): {output}/worker.log')
            log.write(line);log.flush()
            if not line.startswith('@@SAM3D '):continue
            event=json.loads(line[len('@@SAM3D '):])
            if event['type'] in ('fatal','error'):raise RuntimeError(str(event))
            if event['type']=='stage':print(event['message'],flush=True)
            if event['type']==kind:return event
        raise TimeoutError(str(output/'worker.log'))
    def send(command):
        command.update(config_id=config.config_id,backend_id='fast',session_id='embedl-profile-check',
                       inference_type=config.mode)
        process.stdin.write(json.dumps(command)+'\n');process.stdin.flush()
    try:
        ready=event_until('ready')
        assert ready['actual_config']==config.to_dict()
        (output/'ready.json').write_text(json.dumps(ready,ensure_ascii=False,indent=2))
        # No ROI: include the real YOLO detection pass in every timed iteration.
        send(dict(type='benchmark',job='timing',image=str(image),output=str(output/'prediction'),
                  warmup_runs=10,runs=runs))
        report=event_until('mesh')['report']
        assert report['actual_config']==config.to_dict() and report['all_outputs_finite']
        assert report['vertex_shapes']==[[18439,3]]
        assert report['compile_actual']==dict(decoder=True,mhr=True,interm=True)
        assert report['feedback_cache']['enabled'] and report['feedback_cache']['last']['hits']>0
        assert report['keypoint_path']['enabled'] and report['keypoint_path']['graph_captured']
        assert report['execution']['final_replay']['graph_captured']
        assert report['execution']['body_io']['enabled']
        if config.backbone_precision=='int8':assert report['backbone_runtime']['calls']>=runs+config.warmup_runs+10
        with np.load(output/'prediction/person_00.npz',allow_pickle=False) as result:
            assert result['pred_vertices'].shape==(18439,3) and np.isfinite(result['pred_vertices']).all()
        stream=[]
        if config.backbone_precision=='int8':
            source=ROOT/'outputs/rgbd-sets/recording-20261106-135615-851252'
            poses=json.loads((source/'pose_set.json').read_text())['poses']
            stream_dir=output/'stream';stream_dir.mkdir()
            for index,row in enumerate([x for x in poses if x.get('sample')][:3]):
                rgb=cv2.imread(str(source/row['sample']/'rgb.png'))
                ok,jpeg=cv2.imencode('.jpg',rgb);assert ok
                send(dict(type='stream',job=f'stream-{index}',stream_id='embedl-check',frame_index=index,
                    host_received_monotonic=time.monotonic(),jpeg=base64.b64encode(jpeg).decode(),
                    output=str(stream_dir),runtime_status=True))
                event=event_until('stream_mesh',60)
                artifact=Path(event['path'])
                with np.load(artifact,allow_pickle=False) as result:
                    assert result['pred_vertices'].shape==(18439,3) and np.isfinite(result['pred_vertices']).all()
                artifact.rename(stream_dir/f'frame-{index:02d}.npz')
                stream.append(event)
            (output/'stream.json').write_text(json.dumps(stream,ensure_ascii=False,indent=2))
        print(config.backbone_precision,report['benchmark']['mean_ms'],flush=True)
        return dict(config=config.to_dict(),config_id=config.config_id,benchmark=report['benchmark'],
                    runtime=ready['runtime_status'],stream_frames=len(stream),output=str(output))
    finally:
        if process.poll() is None:
            try:
                process.stdin.write('{"type":"stop"}\n');process.stdin.flush();process.wait(timeout=20)
            except (BrokenPipeError,subprocess.TimeoutExpired):
                process.terminate()
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait()
        thread.join(timeout=5)
        log.close();process.stdout.close();process.stdin.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--image',type=Path,default=ROOT/'outputs/gui/20261103-174926-349375/input.png')
    p.add_argument('--runs',type=int,default=200)
    a=p.parse_args()
    guard=memory_guard()
    if not 1<=a.runs<=200:raise ValueError('runs must be 1–200')
    a.output.mkdir(parents=True,exist_ok=False)
    results={}
    for label,name in [('int8','Fast INT8 512 身体（Embedl）'),('fp16','身体 512 对照')]:
        config=replace(PRESETS[name],warmup_layout='sample').validate()
        results[label]=run_variant(config,a.output/label,a.image,a.runs)
        (a.output/'results.json').write_text(json.dumps(dict(results=results,memory_guard=guard),ensure_ascii=False,indent=2))
    old,new=(results[k]['benchmark']['mean_ms'] for k in ('fp16','int8'))
    summary=dict(passed=True,results=results,saved_ms=old-new,reduction_percent=(old-new)/old*100,
        scope='Actual Studio workers, no ROI, YOLO detection included, synchronized inference, excludes load/render/exports; recorded stream protocol smoke only',
        date=datetime.now().isoformat(),memory_guard=guard)
    (a.output/'results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(dict(passed=True,int8_ms=new,fp16_ms=old,saved_ms=old-new),indent=2),flush=True)


if __name__=='__main__':main()
