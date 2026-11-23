# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Named, timestamped raw RGB-D sequences. No camera, Qt or model imports.

Lossless raw frames are written on a bounded queue. Only pairs committed to the
JSONL index are usable; raw colour and depth grids must be registered offline.
"""
from datetime import datetime, timezone
import fcntl
import json
import math
from pathlib import Path
import queue
import re
import threading
from studio.rgbd.rgbd_batch import save_json

PRESETS = {
    'arms': ('双臂缓慢抬放', '缓慢抬起、放下双臂，肘部与躯干保持间隙', '双臂慢慢回到自然站立'),
    'cross': ('交叉与屈肘', '缓慢交叉双臂、屈肘，再向两侧展开', '缓慢展开双臂，回到自然站立'),
    'turn': ('转身与移动', '小幅左右转身，再缓慢前后移动，保持全身入画', '缓慢回到起始位置，面向相机'),
    'recover': ('遮挡与恢复', '短暂遮挡上身或移出画面，然后返回原位', '返回画面中央，面向相机保持全身入画'),
    'custom': ('自定义动作', '按自定义说明缓慢动作，保持全身入画', '缓慢回到起始姿态'),
}


def utc_now():return datetime.now(timezone.utc).isoformat()


def safe_name(name):
    name = re.sub(r'[^\w\-\u4e00-\u9fff]+', '-', str(name).strip(), flags=re.UNICODE).strip('-_')[:64]
    if not name:raise ValueError('请输入数据名称（中文、字母、数字均可）')
    return name


def action_plan(duration, preset='arms', custom=''):
    duration = float(duration)
    if not math.isfinite(duration) or not 5 <= duration <= 300:raise ValueError('录制时长应为 5–300 秒')
    title, action, recovery = PRESETS[preset]
    return [dict(start_s=0., end_s=2., text='自然站立，保持静止'),
            dict(start_s=2., end_s=duration*.7, text=custom.strip() or action),
            dict(start_s=duration*.7, end_s=duration-1., text=recovery),
            dict(start_s=duration-1., end_s=duration, text='保持最后姿态，即将结束')]


def action_at(plan, elapsed):
    return next((p['text'] for p in plan if p['start_s'] <= elapsed < p['end_s']), plan[-1]['text'])


class SequenceWriter:
    """One writer owns a new directory and an OS cleanup-protection lock."""
    def __init__(self, directory, name, camera, settings, queue_size=12):
        if queue_size<1:raise ValueError('写入队列容量必须大于零')
        self.directory=Path(directory);self.directory.mkdir(parents=True, exist_ok=False)
        self.lock=(self.directory/'.recorder.lock').open('a+b')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.frames=queue.Queue(maxsize=queue_size);self.error=None;self.closed=False
        self.enqueued=0;self.written=0;self.dropped_queue=0;self.bytes=0
        self.first_time=None;self.last_time=None;self.capture_times=[]
        self.meta=dict(schema='studio_rgbd_sequence_v1', name=str(name).strip(), created_utc=utc_now(),
                       status='recording', settings=settings, camera='camera.json', index='frames.jsonl',
                       representation='raw_colour_and_raw_uint16_depth; register using camera.json before geometry',
                       coordinate_system='raw grids; calibrated D2C yields x_right_y_down_z_forward',
                       colour_encoding='camera MJPEG passthrough if available; otherwise lossless PNG',
                       depth_encoding='lossless PNG uint16; metres = raw * per-frame depth_scale_mm / 1000')
        try:
            for sub in ('rgb','depth'):(self.directory/sub).mkdir()
            save_json(self.directory/'camera.json', camera)
            save_json(self.directory/'sequence.json', self.meta)
            self.index=(self.directory/'frames.jsonl').open('x')
            self.thread=threading.Thread(target=self._run, name='rgbd-sequence-writer', daemon=False);self.thread.start()
        except BaseException:
            self.lock.close();raise

    def submit(self, image, depth, metadata, rgb_encoded=None):
        if self.closed:raise ValueError('录制已经结束')
        if self.error:raise OSError(self.error)
        if self.frames.full():self.dropped_queue+=1;return False
        import numpy as np
        valid_colour = (image is not None and image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3)
        if rgb_encoded is not None:
            if not isinstance(rgb_encoded, bytes) or not rgb_encoded.startswith(b'\xff\xd8'):
                raise ValueError('原始 MJPEG 数据无效')
            valid_colour=True
        if not valid_colour or depth.dtype != np.uint16 or depth.ndim != 2:
            raise ValueError('原始 RGB/深度格式不正确')
        scale=metadata.get('depth_scale_mm', 0)
        if not math.isfinite(scale) or scale<=0:raise ValueError('深度单位比例不正确')
        for name in ('color_timestamp_ms','depth_timestamp_ms','host_monotonic_ns','relative_time_s','skew_ms'):
            if not math.isfinite(metadata[name]):raise ValueError('帧时间戳不正确：'+name)
        if metadata['skew_ms']<0 or metadata['skew_ms']>self.meta['settings']['max_skew_ms']:
            raise ValueError('帧时间差超限')
        if self.last_time is not None and metadata['relative_time_s']<=self.last_time:raise ValueError('帧时间戳必须严格递增')
        number=self.enqueued
        self.frames.put_nowait((number,image.copy() if image is not None else None,depth.copy(),dict(metadata),rgb_encoded))
        self.enqueued+=1;self.last_time=metadata['relative_time_s']
        if self.first_time is None:self.first_time=self.last_time
        return True

    def _run(self):
        import cv2
        while True:
            item=self.frames.get()
            try:
                if item is None:return
                if self.error:continue
                number,image,depth,meta,encoded=item
                rgb=f'rgb/{number:06d}.jpg' if encoded is not None else f'rgb/{number:06d}.png'
                dep=f'depth/{number:06d}.png'
                if encoded is not None:
                    dest=self.directory/rgb;temp=dest.with_suffix('.part.jpg')
                    temp.write_bytes(encoded);temp.replace(dest)
                for name,array in (((dep,depth),) if encoded is not None else ((rgb,image),(dep,depth))):
                    dest=self.directory/name;temp=dest.with_suffix('.part.png')
                    if not cv2.imwrite(str(temp),array,[cv2.IMWRITE_PNG_COMPRESSION,1]):raise OSError('写入失败：'+name)
                    temp.replace(dest)
                row=dict(meta, frame=number, rgb=rgb, depth=dep,
                         rgb_encoding='camera_mjpg_passthrough' if encoded is not None else 'lossless_png')
                self.index.write(json.dumps(row,ensure_ascii=False)+'\n');self.index.flush()
                self.capture_times.append(meta['relative_time_s'])
                self.written+=1;self.bytes+=(self.directory/rgb).stat().st_size+(self.directory/dep).stat().st_size
            except Exception as exc:self.error=str(exc)
            finally:self.frames.task_done()

    def stats(self):
        times=self.capture_times
        return dict(enqueued=self.enqueued, saved_frames=self.written, queue_pending=self.frames.qsize(),
                    dropped_queue=self.dropped_queue, bytes=self.bytes,
                    actual_saved_fps=(len(times)-1)/(times[-1]-times[0]) if len(times)>1 and times[-1]>times[0] else 0.)

    def close(self, status='completed', **extra):
        if self.closed:return self.meta
        self.closed=True
        try:
            self.frames.put(None);self.thread.join();self.index.close()
            times=self.capture_times
            self.meta.update(status='error' if self.error else status, finished_utc=utc_now(),
                             **self.stats(), **extra)
            self.meta.update(recorded_span_s=times[-1]-times[0] if len(times)>1 else 0., error=self.error)
            save_json(self.directory/'sequence.json',self.meta)
            return self.meta
        finally:
            fcntl.flock(self.lock,fcntl.LOCK_UN);self.lock.close()


class RGBDSequence:
    """Reader for later tracking. Does not mistake raw depth for colour-aligned."""
    def __init__(self, directory, allow_partial=False):
        self.directory=Path(directory).resolve()
        self.meta=json.loads((self.directory/'sequence.json').read_text())
        if self.meta.get('schema')!='studio_rgbd_sequence_v1':raise ValueError('不是 RGB-D 连续序列')
        if self.meta['status'] not in ('completed','stopped') and not allow_partial:raise ValueError('序列未正常结束；可显式读取已提交的部分帧')
        self.camera=json.loads((self.directory/'camera.json').read_text())
        self.frames=[]
        for line in (self.directory/'frames.jsonl').read_text().splitlines():
            try:row=json.loads(line)
            except ValueError:
                if allow_partial:break
                raise
            for key in ('rgb','depth'):
                p=(self.directory/row[key]).resolve();p.relative_to(self.directory)
                if not p.is_file():raise ValueError('缺少已登记帧：'+str(p))
            self.frames.append(row)
        if not allow_partial and len(self.frames)!=self.meta['saved_frames']:raise ValueError('帧清单计数不一致')

    def read(self, index, registered=False):
        import cv2
        import numpy as np
        row=self.frames[index]
        image=cv2.imread(str(self.directory/row['rgb']),cv2.IMREAD_COLOR)
        raw=cv2.imread(str(self.directory/row['depth']),cv2.IMREAD_UNCHANGED)
        if image is None or raw is None or raw.dtype!=np.uint16:raise ValueError('原始 RGB-D 帧损坏')
        if not registered:return image, raw, row
        from studio.rgbd.rgbd_geometry import register_depth, matrix, coefficients
        c=self.camera['raw_calibration'];k=c['color_intrinsics']
        rgb=cv2.undistort(image,matrix(k),coefficients(c['color_distortion']),None,matrix(k))
        z,xyz=register_depth(raw.astype(np.float32)*row['depth_scale_mm']/1000,
                             c['depth_intrinsics'],c['depth_distortion'],k,
                             c['depth_to_color_R'],c['depth_to_color_t_m'])
        return rgb,z,xyz,row
