# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Offline video manifests, independent object annotations and immutable job keys."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent
SCHEMA='studio_video_v1'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def file_hash(file):
    with Path(file).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def save_json(file,value):
    file=Path(file);temp=file.with_suffix(file.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2));temp.replace(file)


def load_project(directory):
    directory=Path(directory).resolve();value=json.loads((directory/'project.json').read_text())
    if value['schema']!=SCHEMA or len(value['pts'])!=value['frame_count']:raise ValueError('视频任务格式错误')
    return directory,value


def frame_path(directory,index):return Path(directory)/'frames'/f'{int(index):06d}.png'


def prepare(source,directory,emit):
    source=Path(source).expanduser().resolve();directory=Path(directory)
    if not source.is_file():raise FileNotFoundError(source)
    directory.mkdir(parents=True,exist_ok=False);(directory/'frames').mkdir()
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_frames',
        '-show_entries','frame=best_effort_timestamp_time,pkt_duration_time:stream=avg_frame_rate:format=duration',
        '-of','json',str(source)]))
    samples=probe.get('frames',[]);pts=[float(f['best_effort_timestamp_time']) for f in samples]
    if not pts or any(b<=a for a,b in zip(pts,pts[1:])):raise ValueError('无法建立严格递增的视频显示时间戳')
    origin=pts[0];pts=[x-origin for x in pts]
    cap=cv2.VideoCapture(str(source));count=0
    try:
        while True:
            ok,image=cap.read()
            if not ok:break
            if count==0:height,width=image.shape[:2]
            if image.shape[:2]!=(height,width):raise ValueError('视频中途改变分辨率')
            if not cv2.imwrite(str(frame_path(directory,count)),image):raise OSError('保存视频帧失败')
            count+=1
            if count%15==0:emit('progress',message=f'解码 {count}/{len(pts)} 帧',current=count,total=len(pts))
    finally:cap.release()
    if count!=len(pts):raise ValueError(f'解码帧数 {count} 与时间戳 {len(pts)} 不一致，不继续错位处理')
    tail=float(samples[-1].get('pkt_duration_time') or (pts[-1]-pts[-2] if count>1 else 1/30))
    streams=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(source)]))['streams']
    value=dict(schema=SCHEMA,created=datetime.now().isoformat(),source=str(source),source_sha256=file_hash(source),
               width=width,height=height,frame_count=count,pts=pts,source_time_origin=origin,
               duration=pts[-1]+tail,has_audio=any(s['codec_type']=='audio' for s in streams),tracks=[])
    save_json(directory/'project.json',value);emit('prepared',directory=str(directory),frames=count)


def add_track(project,name):
    ident=max([x['id'] for x in project['tracks']]+[0])+1
    track=dict(id=ident,name=name or f'人物 {ident}',segments=[]);project['tracks'].append(track);return track


def put_segment(project,ident,start,end,points):
    if not 0<=start<=end<project['frame_count']:raise ValueError('分段范围错误')
    if not 1<=len(points)<=128 or not any(p[2]==1 for p in points):raise ValueError('需要 1～128 个提示点，至少一个人体前景点')
    for x,y,label in points:
        if not (0<=x<project['width'] and 0<=y<project['height'] and label in (0,1)):raise ValueError('提示点越界')
    track=next(t for t in project['tracks'] if t['id']==ident)
    others=[s for s in track['segments'] if s['start']!=start]
    for segment in others:
        if segment['start']<start<=segment['end']:segment['end']=start-1
        if start<segment['start']<=end:end=segment['start']-1
    track['segments']=sorted(others+[dict(start=int(start),end=int(end),points=points)],key=lambda s:s['start'])


def segment_key(project,track,segment):
    return digest(dict(source=project['source_sha256'],track=track['id'],segment=segment,version=1))[:20]


def segment_folder(directory,project,track,segment):
    return Path(directory)/'masks'/f"person-{track['id']:03d}"/segment_key(project,track,segment)


def mask_at(directory,project,track,index):
    for segment in track['segments']:
        if segment['start']<=index<=segment['end']:
            folder=segment_folder(directory,project,track,segment)
            if not (folder/'complete.json').is_file():return None
            file=folder/f'{index:06d}.png'
            return file if file.is_file() else None
    return None


def annotation_key(project):
    return digest(dict(source=project['source_sha256'],tracks=project['tracks']))


def require_masks(directory,project):
    if not project['tracks'] or not any(t['segments'] for t in project['tracks']):raise ValueError('请先给人物保存至少一段提示')
    for track in project['tracks']:
        for segment in track['segments']:
            folder=segment_folder(directory,project,track,segment)
            report=json.loads((folder/'complete.json').read_text())
            if report['key']!=segment_key(project,track,segment):raise ValueError('分割结果已失效')
            for index in range(segment['start'],segment['end']+1):
                file=folder/f'{index:06d}.png'
                if not file.is_file():raise ValueError(f'掩膜缺失：人物 {track["id"]} / 帧 {index}')
    return True


def run_key(project,config,radius):
    return config.config_id+'-'+digest(dict(annotations=annotation_key(project),radius=radius,version=1))[:12]
