# SPDX-License-Identifier: GPL-3.0-only
"""Calibrated V4L2 RGB source using the shared Studio socket protocol."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import select
import socket
import struct
import time
import cv2
import numpy as np
from studio_paths import path
from camera_intrinsics import resolve_intrinsics


def load_calibration(filename):
    raw=Path(filename).read_bytes();value=json.loads(raw)
    if value.get('schema')!='apriltag-webcam-intrinsics-v1':raise ValueError('不支持的前置摄像头标定格式')
    intrinsic=value['color_intrinsic'];width,height=intrinsic['width'],intrinsic['height']
    matrix,_=resolve_intrinsics(intrinsic,width,height)
    distortion=value['color_distortion']
    if distortion.get('model')!='BROWN_CONRADY':raise ValueError('不支持的畸变模型')
    d=np.array([distortion.get(k,0) for k in ('k1','k2','p1','p2','k3','k4','k5','k6')],np.float64)
    if not np.isfinite(d).all():raise ValueError('无效畸变系数')
    if not str(value['device']).startswith('/dev/'):raise ValueError('标定文件缺少 V4L2 设备路径')
    return value,matrix,d,hashlib.sha256(raw).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--socket',required=True);args=parser.parse_args()
    sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);sock.settimeout(2);sock.connect(args.socket)
    cap=None
    def send(kind,**values):
        data=json.dumps(dict(type=kind,**values)).encode();sock.sendall(struct.pack('!I',len(data))+data)
    try:
        calibration,k,d,digest=load_calibration(path('webcam_calibration'))
        intrinsic=calibration['color_intrinsic'];width,height=intrinsic['width'],intrinsic['height']
        cap=cv2.VideoCapture(calibration['device'],cv2.CAP_V4L2)
        if not cap.isOpened():raise RuntimeError('无法打开前置摄像头，请关闭其他占用程序')
        cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,width);cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
        cap.set(cv2.CAP_PROP_FPS,30);cap.set(cv2.CAP_PROP_BUFFERSIZE,1);cap.set(cv2.CAP_PROP_AUTOFOCUS,0)
        maps=cv2.initUndistortRectifyMap(k,d,None,k,(width,height),cv2.CV_16SC2)
        metadata=dict(device='电脑前置摄像头',device_path=calibration['device'],width=width,height=height,
            fps=cap.get(cv2.CAP_PROP_FPS) or 30,format='rectified BGR',sdk='OpenCV/V4L2',depth_enabled=False,
            intrinsics=intrinsic,rgb_rectified=True,effective_distortion=[0]*8,mirrored=False,
            calibration_sha256=digest,raw_distortion=calibration['color_distortion'],
            calibration_validation=calibration.get('validation',{}),timestamp_source='host_after_v4l2_read')
        pending=None;buffer=b'';fps=15;next_preview=0;index=0;ready=False
        while True:
            if select.select([sock],[],[],0)[0]:
                chunk=sock.recv(65536)
                if not chunk:break
                buffer+=chunk
                while len(buffer)>=4:
                    size=struct.unpack('!I',buffer[:4])[0]
                    if size>1048576:raise ValueError('控制消息过大')
                    if len(buffer)<size+4:break
                    cmd=json.loads(buffer[4:4+size]);buffer=buffer[4+size:]
                    if cmd['type']=='stop':return 0
                    if cmd['type']=='preview_rate':fps=max(1,min(30,int(cmd.get('fps',15))));next_preview=0
                    if cmd['type']=='cancel':pending=None;send('cancelled')
                    if cmd['type']=='capture' and ready:
                        delay=max(0,min(60,float(cmd['delay'])));pending=dict(cmd,delay=delay,deadline=time.monotonic()+delay)
                        send('countdown',job=cmd['job'],delay=delay)
            ok,raw=cap.read();now=time.monotonic()
            if not ok:raise RuntimeError('前置摄像头读取失败，请检查连接或占用')
            if raw.shape[:2]!=(height,width):raise ValueError('实际分辨率与标定不一致，拒绝套用内参')
            index+=1
            if index>=5 and not ready:ready=True;send('ready',metadata=metadata)
            capture_due=pending is not None and now>=pending['deadline']
            if now<next_preview and not capture_due:continue
            image=cv2.remap(raw,*maps,interpolation=cv2.INTER_LINEAR)
            if capture_due:
                target=Path(pending['directory']);target.mkdir(parents=True,exist_ok=False);photo=target/'photo.png'
                if not cv2.imwrite(str(photo),image):raise OSError('无法保存照片')
                record=dict(metadata,frame_index=index,received_monotonic=now,device_timestamp_us=None,
                    captured_utc=datetime.now(timezone.utc).isoformat(),job=pending['job'],countdown_seconds=pending['delay'])
                (target/'camera.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
                send('photo',job=pending['job'],path=str(photo),metadata=record);pending=None
            if now>=next_preview:
                ok,jpeg=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,85])
                if ok:send('frame',jpeg=base64.b64encode(jpeg).decode(),index=index,received_monotonic=now,
                    device_timestamp_us=None,preview_fps=fps,receipt_stage='v4l2_read_complete')
                next_preview=now+1/fps
    except Exception as exc:
        try:send('error',message=str(exc))
        except OSError:pass
        return 1
    finally:
        if cap is not None:cap.release()
        sock.close()
    return 0

if __name__=='__main__':raise SystemExit(main())
