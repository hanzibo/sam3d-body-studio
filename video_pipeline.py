# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Isolated offline stages, cancellable resident-worker bridge and resumable runs."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time
import traceback
import cv2
import numpy as np
from video_project import (ROOT,prepare,load_project,frame_path,mask_at,require_masks,annotation_key,run_key,
                           save_json,file_hash,digest)
from mask_prompt import morph_mask
from launch_config import LaunchConfig


def emit(kind,**values):print('@@VIDEO '+json.dumps(dict(type=kind,**values),ensure_ascii=False),flush=True)
def interrupted(*_):raise KeyboardInterrupt()


class BodyProcess:
    def __init__(self,config,folder):
        self.config=config;self.folder=folder;self.lines=queue.Queue();self.log=(folder/'worker.log').open('a');self.tail=[]
        profile=folder/'config.json';config.save(profile)
        env=dict(os.environ,BODYSTUDIO_CONFIG=str(profile),BODYSTUDIO_SESSION='video-'+folder.name,BODYSTUDIO_GPU_KIND='exclusive')
        self.process=subprocess.Popen([str(ROOT/('launch_fast_model.sh' if config.backend=='fast' else 'launch_model.sh'))],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=env)
        self.session=env['BODYSTUDIO_SESSION']
        def reader():
            for line in self.process.stdout:self.lines.put(line)
            self.lines.put(None)
        threading.Thread(target=reader,daemon=True).start()

    def event(self,expected,timeout=600):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            try:line=self.lines.get(timeout=.5)
            except queue.Empty:continue
            if line is None:raise RuntimeError('人体模型进程退出：'+''.join(self.tail)[-1200:])
            self.log.write(line);self.log.flush()
            self.tail=(self.tail+[line])[-12:]
            if not line.startswith('@@SAM3D '):continue
            event=json.loads(line[len('@@SAM3D '):])
            if event['type']=='stage':emit('stage',message=event.get('message',''))
            if event['type'] in ('error','fatal'):raise RuntimeError(event.get('message','模型失败'))
            if event['type']==expected:return event
        raise TimeoutError('等待人体模型超时')

    def request(self,command):
        command.update(config_id=self.config.config_id,backend_id=self.config.backend,session_id=self.session)
        self.process.stdin.write(json.dumps(command)+'\n');self.process.stdin.flush()
        return self.event('video_mesh',timeout=180)

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
        self.process.stdin.close();self.process.stdout.close();self.log.close()


def reconstruct(directory,config_file,radius):
    directory,project=load_project(directory);require_masks(directory,project)
    config=LaunchConfig.from_dict(json.loads(Path(config_file).read_text()))
    if config.one_euro_enabled or config.smpl_smoothing:
        raise ValueError('首版视频保留原始输出，请关闭 One Euro 和 MLP；避免不同人物共享时序状态')
    config.validate();folder=directory/'runs'/run_key(project,config,radius);folder.mkdir(parents=True,exist_ok=True)
    save_json(folder/'run.json',dict(schema='studio_video_run_v1',source_sha256=project['source_sha256'],
        annotations=annotation_key(project),config=config.to_dict(),radius=radius,status='running'))
    body=BodyProcess(config,folder);records=[];began=time.monotonic();cached=0
    try:
        ready=body.event('ready')
        if ready.get('actual_config')!=config.to_dict():raise ValueError('实际加载方案与视频冻结配置不一致')
        for index in range(project['frame_count']):
            image=frame_path(directory,index);image_hash=None;reserved=[]
            for track in project['tracks']:
                raw_path=mask_at(directory,project,track,index)
                if raw_path is None:continue
                raw=cv2.imread(str(raw_path),0)
                if raw is None or raw.shape!=(project['height'],project['width']):raise ValueError('视频掩膜尺寸错误')
                dest=folder/'predictions'/f'{index:06d}'/f"person-{track['id']:03d}";dest.mkdir(parents=True,exist_ok=True)
                meta=dest/'result.json'
                if meta.is_file():
                    record=json.loads(meta.read_text())
                    if record.get('status') in ('ok','not_visible') and (record['status']!='ok' or (dest/'prediction.npz').is_file()):
                        if record['status']=='ok':reserved.append(record['bbox_xyxy'])
                        records.append(record);cached+=1;continue
                record=dict(frame=index,person_id=track['id'],status='not_visible')
                if raw.any():
                    try:
                        if image_hash is None:image_hash=file_hash(image)
                        mask=morph_mask(raw,radius)
                        if not mask.any():raise ValueError('形态调整后掩膜为空')
                        target=dest/'prompt-mask.png'
                        if not cv2.imwrite(str(target),mask*255):raise OSError('提示掩膜写入失败')
                        info=dict(path=str(target),image_sha256=image_hash,mask_sha256=file_hash(target),morphology_radius_px=radius)
                        selection=dict(path=str(raw_path),image_sha256=image_hash,mask_sha256=file_hash(raw_path))
                        x,y,w,h=cv2.boundingRect(raw)
                        command=dict(type='video',job=f'{index}-{track["id"]}',image=str(image),output=str(dest),
                            topology=str(folder/'faces.npy'),bbox=[x,y,x+w,y+h],inference_type=config.mode,
                            mask_prompt=info,selection_mask=selection,video_frame_key=f'{directory}:{index}',video_reserved_boxes=reserved)
                        event=body.request(command)
                        record.update(event,status='ok',input_sha256=image_hash,selection_mask_sha256=selection['mask_sha256'],mask_prompt=info)
                        reserved.append(record['bbox_xyxy'])
                    except (RuntimeError,ValueError) as exc:
                        record.update(status='failed',message=str(exc))
                        if body.process.poll() is not None:raise
                save_json(meta,record);records.append(record)
            if index%5==0:emit('progress',message=f'人体重建 · 帧 {index+1}/{project["frame_count"]}',current=index+1,total=project['frame_count'])
        summary=dict(schema='studio_video_run_v1',source_sha256=project['source_sha256'],annotations=annotation_key(project),
            config=config.to_dict(),radius=radius,status='complete',seconds=time.monotonic()-began,records=records,
            successful=sum(r['status']=='ok' for r in records),failed=sum(r['status']=='failed' for r in records))
        summary['cached_person_frames']=cached
        summary['seconds_scope']='current invocation wall time including model load and cache checks'
        summary['prediction_revision']=digest([(r['frame'],r['person_id'],r['status'],
            (folder/'predictions'/f'{r["frame"]:06d}'/f'person-{r["person_id"]:03d}'/'prediction.npz').stat().st_mtime_ns if r['status']=='ok' else 0) for r in records])
        save_json(folder/'run.json',summary);emit('reconstructed',run=str(folder),successful=summary['successful'],failed=summary['failed'])
    finally:body.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['prepare','segment','reconstruct','render'],required=True)
    parser.add_argument('--directory',required=True);parser.add_argument('--source');parser.add_argument('--config')
    parser.add_argument('--radius',type=int,default=0);parser.add_argument('--run');parser.add_argument('--layers',choices=['mesh','skeleton','both'],default='mesh')
    args=parser.parse_args()
    try:os.setsid()
    except OSError:pass
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        if args.mode=='prepare':prepare(args.source,args.directory,emit)
        elif args.mode=='segment':
            from sam3_video_worker import segment_video
            segment_video(args.directory,emit)
        elif args.mode=='reconstruct':reconstruct(args.directory,args.config,args.radius)
        elif args.mode=='render':
            from video_render import render_video
            render_video(args.directory,args.run,args.layers,emit)
        emit('done',mode=args.mode)
    except KeyboardInterrupt:emit('cancelled');return 130
    except Exception as exc:traceback.print_exc();emit('error',message=str(exc));return 1
    return 0

if __name__=='__main__':sys.exit(main())
