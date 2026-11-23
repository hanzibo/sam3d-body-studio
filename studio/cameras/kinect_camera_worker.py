# SPDX-License-Identifier: GPL-3.0-only
"""Kinect v2 RGB adapter for the Studio camera socket protocol."""
import argparse
import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import select
import socket
import struct
import time
import cv2
from kinect2 import Kinect2Camera


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--socket',required=True);args=parser.parse_args()
    sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);sock.settimeout(2);sock.connect(args.socket)
    def send(kind,**values):
        data=json.dumps(dict(type=kind,**values)).encode();sock.sendall(struct.pack('!I',len(data))+data)
    try:
        with Kinect2Camera() as camera:
            calibration=camera.calibration()['color']
            metadata=dict(device='Kinect v2',serial=camera.serial,firmware=camera.firmware_version,
                sdk='kinect2/libfreenect2',width=1920,height=1080,fps=30,format='BGRX',depth_enabled=False,
                intrinsics=dict(width=1920,height=1080,**{k:calibration[k] for k in ('fx','fy','cx','cy')}))
            buffer=b'';pending=None;fps=15;next_frame=0;ready=False;count=0
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
                        if cmd['type']=='preview_rate':fps=max(1,min(30,int(cmd.get('fps',15))));next_frame=0
                        if cmd['type']=='cancel':pending=None;send('cancelled')
                        if cmd['type']=='capture' and ready:
                            delay=max(0,min(60,float(cmd['delay'])));pending=dict(cmd,delay=delay,deadline=time.monotonic()+delay)
                            send('countdown',job=cmd['job'],delay=delay)
                frames=camera.get_frames(timeout_ms=3000);now=time.monotonic();count+=1
                image=frames['color'][...,:3];meta=frames['metadata']['color']
                if count>=5 and not ready:ready=True;send('ready',metadata=metadata)
                if pending and now>=pending['deadline']:
                    target=Path(pending['directory']);target.mkdir(parents=True,exist_ok=False)
                    photo=target/'photo.png'
                    if not cv2.imwrite(str(photo),image):raise OSError('无法保存照片')
                    record=dict(metadata,frame_index=int(meta['sequence']),device_timestamp_us=float(meta['timestamp_ms'])*1000,
                        received_monotonic=now,captured_utc=datetime.now(timezone.utc).isoformat(),job=pending['job'],countdown_seconds=pending['delay'])
                    (target/'camera.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
                    send('photo',job=pending['job'],path=str(photo),metadata=record);pending=None
                if now>=next_frame:
                    ok,jpeg=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,85])
                    if ok:send('frame',jpeg=base64.b64encode(jpeg).decode(),index=int(meta['sequence']),
                        received_monotonic=now,device_timestamp_us=float(meta['timestamp_ms'])*1000,preview_fps=fps,receipt_stage='sdk_color_frame_available')
                    next_frame=now+1/fps
    except Exception as exc:
        try:send('error',message=str(exc))
        except OSError:pass
        return 1
    finally:sock.close()
    return 0

if __name__=='__main__':raise SystemExit(main())
