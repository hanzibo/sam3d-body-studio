"""Isolated Gemini RGB-D acquisition; stdout is a bounded JSON event stream."""
import argparse
import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import select
import sys
import time
import cv2
import numpy as np
from pyorbbecsdk import Config, Context, OBSensorType, OBFrameAggregateOutputMode, Pipeline
from rgbd_geometry import register_depth, matrix, coefficients, write_ply
from orbbec_color import color_frame_to_bgr


def emit(kind, **kwargs): print('@@RGBD '+json.dumps(dict(type=kind,**kwargs),ensure_ascii=False),flush=True)
def intrinsic(v): return {n:getattr(v,n) for n in ('width','height','fx','fy','cx','cy')}
def distortion(v): return dict(model=str(v.model),**{n:getattr(v,n) for n in ('k1','k2','p1','p2','k3','k4','k5','k6')})
def profile(p): return dict(width=p.get_width(),height=p.get_height(),fps=p.get_fps(),format=str(p.get_format()))
def interrupted(*_): raise KeyboardInterrupt()


def capture(output, delay=5, max_skew_ms=20, preview=False):
    output=Path(output)
    # Preview must not create a sample; only an explicit capture may write one.
    if not preview:output.mkdir(parents=True,exist_ok=False)
    context=Context();devices=context.query_devices()
    device=next((devices.get_device_by_index(i) for i in range(devices.get_count())
                 if devices.get_device_by_index(i).get_device_info().get_pid()==0x0807),None)
    if device is None: raise RuntimeError('未找到 Gemini 336L，请检查连接或其他程序占用')
    info=device.get_device_info();pipeline=Pipeline(device)
    cp=pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR).get_default_video_stream_profile()
    dp=pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR).get_default_video_stream_profile()
    config=Config();config.enable_stream(cp);config.enable_stream(dp)
    config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
    pipeline.enable_frame_sync()
    pipeline.start(config)
    try:
        camera=pipeline.get_camera_param()
        if camera.is_mirrored: raise ValueError('验证暂不接受镜像流，请关闭相机镜像')
        ck,dk=intrinsic(camera.rgb_intrinsic),intrinsic(camera.depth_intrinsic)
        cd,dd=distortion(camera.rgb_distortion),distortion(camera.depth_distortion)
        # These cameras use the Brown rational model; reject unsupported models.
        if any(not any(s in d['model'].lower() for s in ('brown','none','unknown')) for d in (cd,dd)):
            raise ValueError('未支持的畸变模型：'+str((cd,dd)))
        rotation=np.asarray(camera.transform.rot).reshape(3,3)
        translation=np.asarray(camera.transform.transform)/1000
        began=time.monotonic();deadline=None if preview else began+max(0,delay)+2
        emit('ready',message=f'RGB {cp.get_width()}×{cp.get_height()} / 深度 {dp.get_width()}×{dp.get_height()} · '+('实时预览，请让头脚完整入画' if preview else '请保持静止'))
        count=0;last=-1;selected=None;last_preview=0;last_frame=began
        while deadline is None or time.monotonic()<deadline+20:
            if preview and select.select([sys.stdin],[],[],0)[0]:
                line=sys.stdin.readline()
                if not line:return
                command=json.loads(line)
                if command.get('type')=='stop':return
                if command.get('type')=='capture' and deadline is None:
                    output.mkdir(parents=True,exist_ok=False)
                    deadline=time.monotonic()+max(0,float(command.get('delay',delay)))
                    emit('stage',message='倒计时已开始，请摆好姿势')
            if deadline is not None:
                remain=max(0,int(np.ceil(deadline-time.monotonic())))
                if remain!=last:emit('countdown',seconds=remain);last=remain
            frames=pipeline.wait_for_frames(1000)
            if not frames:
                if time.monotonic()-last_frame>10:raise RuntimeError('相机连续 10 秒没有返回 RGB-D 帧')
                continue
            color,depth=frames.get_color_frame(),frames.get_depth_frame()
            if not color or not depth:continue
            count+=1;last_frame=time.monotonic()
            if last_frame-last_preview>=.1:
                image=color_frame_to_bgr(color)
                small=cv2.resize(image,(800,round(image.shape[0]*800/image.shape[1])))
                ok,jpeg=cv2.imencode('.jpg',small,[cv2.IMWRITE_JPEG_QUALITY,80])
                if ok:emit('frame',jpeg=base64.b64encode(jpeg).decode('ascii'))
                last_preview=last_frame
            skew=abs(color.get_timestamp()-depth.get_timestamp())
            if deadline is None or time.monotonic()<deadline or count<15 or skew>max_skew_ms:continue
            image=color_frame_to_bgr(color)
            raw=np.frombuffer(depth.get_data(),np.uint16).reshape(depth.get_height(),depth.get_width()).copy()
            selected=dict(color_timestamp_ms=color.get_timestamp(),depth_timestamp_ms=depth.get_timestamp(),
                skew_ms=skew,color_index=color.get_index(),depth_index=depth.get_index(),
                depth_scale_mm=float(depth.get_depth_scale()))
            break
        if selected is None: raise RuntimeError('未取得时间差合格的 RGB-D 帧，采集超时')
    finally: pipeline.stop()
    if image.shape[:2]!=(ck['height'],ck['width']) or raw.shape!=(dk['height'],dk['width']):
        raise ValueError('活动图像尺寸与 SDK 标定不一致')
    emit('stage',message='相机已释放，正在进行标定配准与点云导出')
    rectified=cv2.undistort(image,matrix(ck),coefficients(cd),None,matrix(ck))
    depth_m=raw.astype(np.float32)*selected['depth_scale_mm']/1000
    aligned,xyz=register_depth(depth_m,dk,dd,ck,rotation,translation)
    for name,img in [('rgb_raw.png',image),('rgb.png',rectified),('depth_raw.png',raw)]:
        if not cv2.imwrite(str(output/name),img):raise OSError('保存失败：'+name)
    np.savez_compressed(output/'rgbd.npz',depth_m=aligned,xyz_m=xyz)
    valid=aligned>.05
    write_ply(output/'scene.ply',xyz[valid],rectified[:,:,::-1][valid])
    disp=cv2.applyColorMap(np.uint8(np.clip(aligned/6,0,1)*255),cv2.COLORMAP_TURBO);disp[~valid]=0
    cv2.imwrite(str(output/'depth_preview.png'),disp)
    metadata=dict(schema='studio_rgbd_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        device=info.get_name(),serial=info.get_serial_number(),firmware=info.get_firmware_version(),
        streams=dict(color=profile(cp),depth=profile(dp)),intrinsics=ck,
        raw_calibration=dict(color_intrinsics=ck,color_distortion=cd,depth_intrinsics=dk,depth_distortion=dd,
            depth_to_color_R=rotation.tolist(),depth_to_color_t_m=translation.tolist()),
        registration='depth Brown deprojection -> D2C -> rectified RGB pinhole; nearest Z; no hole filling',
        rgb_rectified=True,effective_distortion=[0]*8,coordinate_system='x_right_y_down_z_forward',
        units='metres',depth_enabled=True,timestamps=selected,max_timestamp_skew_ms=max_skew_ms)
    (output/'camera.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    emit('done',directory=str(output),valid_points=int(valid.sum()),timestamps=selected)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--delay',type=float,default=5)
    p.add_argument('--max-skew-ms',type=float,default=20);p.add_argument('--preview',action='store_true');args=p.parse_args()
    signal.signal(signal.SIGTERM,interrupted)
    try:capture(args.output,args.delay,args.max_skew_ms,args.preview)
    except KeyboardInterrupt:emit('cancelled');sys.exit(130)
    except Exception as exc:emit('error',message=str(exc));sys.exit(1)
