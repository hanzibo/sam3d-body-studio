# SPDX-License-Identifier: GPL-3.0-only
"""Kinect registered RGB-D sample in the undistorted depth camera frame."""
import argparse
import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import select
import signal
import sys
import time
import cv2
import numpy as np
from kinect2 import Kinect2Camera
from studio.rgbd.rgbd_geometry import write_ply


def emit(kind,**values):print('@@RGBD '+json.dumps(dict(type=kind,**values)),flush=True)
def capture(args):
    output=Path(args.output)
    if not args.preview:output.mkdir(parents=True,exist_ok=False)
    with Kinect2Camera() as camera:
        calibration=camera.calibration();k=dict(width=512,height=424,**{n:calibration['depth'][n] for n in ('fx','fy','cx','cy')})
        serial=camera.serial;firmware=camera.firmware_version
        deadline=None if args.preview else time.monotonic()+max(0,args.delay)+2
        emit('ready',message='Kinect v2 · SDK配准 RGB-D 512×424 · 深度相机坐标系')
        last_preview=0;last=-1
        while True:
            if args.preview and select.select([sys.stdin],[],[],0)[0]:
                line=sys.stdin.readline()
                if not line:return
                cmd=json.loads(line)
                if cmd['type']=='stop':return
                if cmd['type']=='capture' and deadline is None:
                    output.mkdir(parents=True,exist_ok=False);deadline=time.monotonic()+max(0,float(cmd.get('delay',args.delay)))
                    emit('stage',message='倒计时已开始，请摆好姿势')
            frames=camera.get_frames(timeout_ms=3000,registered=True)
            now=time.monotonic();cm=frames['metadata']['color'];dm=frames['metadata']['depth'];skew=abs(cm['timestamp_ms']-dm['timestamp_ms'])
            rgb=frames['registered_color'][...,:3].copy();raw=frames['undistorted_depth']
            depth=raw.astype(np.float32)/1000;depth[~np.isfinite(depth) | (depth<=0)]=0
            if now-last_preview>.1:
                ok,jpeg=cv2.imencode('.jpg',rgb)
                extra={}
                if args.depth_preview:
                    disp=cv2.applyColorMap(np.uint8(np.clip(depth/6,0,1)*255),cv2.COLORMAP_TURBO);disp[depth==0]=0
                    _,dj=cv2.imencode('.jpg',disp);extra['depth_jpeg']=base64.b64encode(dj).decode()
                if ok:emit('frame',jpeg=base64.b64encode(jpeg).decode(),skew_ms=skew,**extra)
                last_preview=now
            if deadline is None:continue
            remain=max(0,int(np.ceil(deadline-now)))
            if remain!=last:emit('countdown',seconds=remain);last=remain
            if now<deadline:continue
            if skew>args.max_skew_ms:
                if now>deadline+20:raise RuntimeError('RGB-D 时间差超限，未保存样本')
                continue
            break
    yy,xx=np.indices(depth.shape);xyz=np.stack(((xx-k['cx'])*depth/k['fx'],(yy-k['cy'])*depth/k['fy'],depth),axis=-1).astype(np.float32)
    valid=depth>.05;xyz[~valid]=0
    for name,image in [('rgb.png',rgb),('rgb_raw.png',rgb),('depth_raw.png',np.uint16(np.clip(depth*1000,0,65535)))]:
        if not cv2.imwrite(str(output/name),image):raise OSError('保存失败：'+name)
    np.savez_compressed(output/'rgbd.npz',depth_m=depth,xyz_m=xyz)
    write_ply(output/'scene.ply',xyz[valid],rgb[:,:,::-1][valid])
    disp=cv2.applyColorMap(np.uint8(np.clip(depth/6,0,1)*255),cv2.COLORMAP_TURBO);disp[~valid]=0;cv2.imwrite(str(output/'depth_preview.png'),disp)
    timestamps=dict(color_timestamp_ms=cm['timestamp_ms'],depth_timestamp_ms=dm['timestamp_ms'],skew_ms=skew)
    metadata=dict(schema='studio_rgbd_v1',device='Kinect v2',serial=serial,firmware=firmware,intrinsics=k,
        created_utc=datetime.now(timezone.utc).isoformat(),raw_calibration=calibration,
        streams=dict(color=dict(width=512,height=424,format='registered BGR'),depth=dict(width=512,height=424,format='undistorted depth')),
        registration='libfreenect2 color registered to undistorted depth; rgb_raw.png is registered RGB, not native 1080p',
        rgb_rectified=True,effective_distortion=[0]*8,coordinate_system='x_right_y_down_z_forward',units='metres',
        depth_enabled=True,timestamps=timestamps,max_timestamp_skew_ms=args.max_skew_ms,
        capture_quality=dict(registered_valid_fraction=float(valid.mean()),registered_valid_points=int(valid.sum())))
    (output/'camera.json').write_text(json.dumps(metadata,indent=2));emit('done',directory=str(output),valid_points=int(valid.sum()),timestamps=timestamps)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--delay',type=float,default=5)
    p.add_argument('--max-skew-ms',type=float,default=20);p.add_argument('--preview',action='store_true');p.add_argument('--depth-preview',action='store_true');args=p.parse_args()
    def stop(*_):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,stop)
    try:capture(args)
    except KeyboardInterrupt:emit('cancelled');sys.exit(130)
    except Exception as exc:emit('error',message=str(exc));sys.exit(1)
