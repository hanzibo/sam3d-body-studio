# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Isolated Gemini sequence acquisition. Preview first, record on JSON command."""
import argparse
import base64
import json
from pathlib import Path
import select
import shutil
import signal
import sys
import time
import cv2
import numpy as np
from pyorbbecsdk import Config, Context, OBSensorType, OBFrameAggregateOutputMode, Pipeline
from studio.cameras.rgbd_capture import emit, intrinsic, distortion, profile, interrupted
from studio.cameras.orbbec_color import color_frame_to_bgr
from studio.rgbd.rgbd_sequence import SequenceWriter, action_plan, action_at


def capture(args):
    context=Context();devices=context.query_devices()
    device=next((devices.get_device_by_index(i) for i in range(devices.get_count())
                 if devices.get_device_by_index(i).get_device_info().get_pid()==0x0807),None)
    if device is None:raise RuntimeError('未找到 Gemini 336L，请检查连接或其他程序占用')
    info=device.get_device_info();pipeline=Pipeline(device)
    cp=pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR).get_default_video_stream_profile()
    dp=pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR).get_default_video_stream_profile()
    config=Config();config.enable_stream(cp);config.enable_stream(dp)
    config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
    pipeline.enable_frame_sync();pipeline.start(config)
    writer=None;outcome='interrupted';extra={};failure=None
    try:
        camera=pipeline.get_camera_param()
        if camera.is_mirrored:raise ValueError('暂不接受镜像相机流')
        ck,dk=intrinsic(camera.rgb_intrinsic),intrinsic(camera.depth_intrinsic)
        cd,dd=distortion(camera.rgb_distortion),distortion(camera.depth_distortion)
        if any(not any(s in d['model'].lower() for s in ('brown','none','unknown')) for d in (cd,dd)):
            raise ValueError('不支持的相机畸变模型')
        calibration=dict(schema='studio_rgbd_sequence_camera_v1',device=info.get_name(),serial=info.get_serial_number(),
            firmware=info.get_firmware_version(),streams=dict(color=profile(cp),depth=profile(dp)),
            raw_calibration=dict(color_intrinsics=ck,color_distortion=cd,depth_intrinsics=dk,depth_distortion=dd,
                depth_to_color_R=np.asarray(camera.transform.rot).reshape(3,3).tolist(),
                depth_to_color_t_m=(np.asarray(camera.transform.transform)/1000).tolist()),
            rgb_rectified=False,depth_registered=False,
            timestamp_unit='milliseconds',timestamp_source='Orbbec frame.get_timestamp()',
            host_clock='time.monotonic_ns; receive time, not exposure time',
            registration='offline: raw depth Brown deprojection -> D2C -> rectified RGB pinhole; nearest Z; no hole filling')
        emit('ready',message=f'RGB {cp.get_width()}×{cp.get_height()} / 深度 {dp.get_width()}×{dp.get_height()} · 预览不保存',
             camera=calibration)
        deadline=None;settings=None;began=None;last_frame=time.monotonic();last_preview=0.;last_stat=0.;last_count=-1
        last_device=None;last_depth=None;next_sample=None;received=0;bad_skew=0;duplicates=0;throttled=0
        while True:
            if select.select([sys.stdin],[],[],0)[0]:
                line=sys.stdin.readline()
                command=json.loads(line) if line else dict(type='stop')
                kind=command.get('type')
                if kind=='stop':outcome='stopped';break
                if kind=='cancel' and writer is None:outcome='cancelled';break
                if kind=='record' and deadline is None:
                    duration=float(command['duration']);fps=int(command['fps']);delay=float(command['delay'])
                    if fps not in (10,15,30) or not np.isfinite(delay) or not 0<=delay<=30:raise ValueError('录制参数不正确')
                    plan=action_plan(duration,command.get('preset','arms'),command.get('custom',''))
                    settings=dict(duration_s=duration,target_fps=fps,countdown_s=delay,max_skew_ms=args.max_skew_ms,
                                  preset=command.get('preset','arms'),notes=command.get('custom',''),action_plan=plan)
                    deadline=time.monotonic()+delay
                    emit('stage',message='开录倒计时已开始，请保持全身入画')
            now=time.monotonic()
            if deadline is not None and writer is None:
                remain=max(0,int(np.ceil(deadline-now)))
                if remain!=last_count:emit('countdown',seconds=remain);last_count=remain
                if now>=deadline:
                    parent=Path(args.output).parent;parent.mkdir(parents=True,exist_ok=True)
                    if shutil.disk_usage(parent).free<1024**3:raise OSError('可用空间不足 1 GiB，请释放空间后重试')
                    writer=SequenceWriter(args.output,args.name,calibration,settings)
                    began=time.monotonic();next_sample=None
                    emit('recording',directory=str(Path(args.output).resolve()),duration_s=settings['duration_s'],plan=settings['action_plan'])
                    now=time.monotonic()
            if writer is not None:
                if writer.error:raise OSError(writer.error)
                elapsed=now-began
                if elapsed>=settings['duration_s']:outcome='completed';break
                if now-last_stat>=.2:
                    emit('progress',elapsed_s=elapsed,remaining_s=max(0,settings['duration_s']-elapsed),
                         action=action_at(settings['action_plan'],elapsed),rejected_skew=bad_skew,duplicates=duplicates,
                         received_pairs=received,**writer.stats())
                    last_stat=now
                    if shutil.disk_usage(writer.directory).free<512*1024**2:
                        raise OSError('可用空间不足 512 MiB，已停止录制；已提交帧保留')
            frames=pipeline.wait_for_frames(250)
            if not frames:
                if time.monotonic()-last_frame>10:raise RuntimeError('相机连续 10 秒没有返回 RGB-D 帧')
                continue
            color,depth=frames.get_color_frame(),frames.get_depth_frame()
            if not color or not depth:continue
            host=time.monotonic_ns();last_frame=host/1e9
            ct,dt=float(color.get_timestamp()),float(depth.get_timestamp());skew=abs(ct-dt)
            want_preview=last_frame-last_preview>=.2
            image=None
            if want_preview:
                image=color_frame_to_bgr(color)
                small=cv2.resize(image,(800,round(image.shape[0]*800/image.shape[1])))
                ok,jpg=cv2.imencode('.jpg',small,[cv2.IMWRITE_JPEG_QUALITY,75])
                if ok:
                    raw=np.frombuffer(depth.get_data(),np.uint16).reshape(depth.get_height(),depth.get_width())
                    display=cv2.applyColorMap(np.uint8(np.clip(raw.astype(np.float32)*depth.get_depth_scale()/6000,0,1)*255),cv2.COLORMAP_TURBO)
                    display[raw==0]=0;display=cv2.resize(display,(480,round(display.shape[0]*480/display.shape[1])))
                    dok,djpg=cv2.imencode('.jpg',display,[cv2.IMWRITE_JPEG_QUALITY,70])
                    emit('frame',jpeg=base64.b64encode(jpg).decode(),depth_jpeg=base64.b64encode(djpg).decode() if dok else '',
                         skew_ms=skew,valid_depth_fraction=float(np.mean(raw>0)))
                last_preview=last_frame
            if writer is None:continue
            if last_frame-began>=settings['duration_s']:outcome='completed';break
            received+=1
            if skew>args.max_skew_ms or not np.isfinite(skew):bad_skew+=1;continue
            if last_device is not None:
                if ct<last_device or dt<last_depth:raise RuntimeError('相机时间戳倒退，已停止以避免生成不连续序列')
                if ct==last_device or dt==last_depth:duplicates+=1;continue
            last_device,last_depth=ct,dt
            if next_sample is not None and ct<next_sample-.5:throttled+=1;continue
            interval=1000/settings['target_fps']
            next_sample=(ct if next_sample is None else max(next_sample,ct-interval*.5))+interval
            encoded=np.asarray(color.get_data()).tobytes() if 'MJPG' in str(color.get_format()).upper() else None
            if image is None and encoded is None:image=color_frame_to_bgr(color)
            raw=np.frombuffer(depth.get_data(),np.uint16).reshape(depth.get_height(),depth.get_width())
            if (color.get_height(),color.get_width())!=(ck['height'],ck['width']) or raw.shape!=(dk['height'],dk['width']):
                raise ValueError('实际帧尺寸与标定不一致')
            writer.submit(image,raw,dict(color_timestamp_ms=ct,depth_timestamp_ms=dt,skew_ms=skew,
                color_index=int(color.get_index()),depth_index=int(depth.get_index()),depth_scale_mm=float(depth.get_depth_scale()),
                host_monotonic_ns=host,relative_time_s=last_frame-began),rgb_encoded=encoded)
    except BaseException as exc:
        failure=exc;outcome='interrupted' if isinstance(exc,KeyboardInterrupt) else 'error'
    finally:
        try:pipeline.stop()
        except Exception as exc:
            if failure is None:failure=exc;outcome='error'
        if writer is not None:
            emit('saving',message='相机已释放，正在保存写入队列；请勿关闭窗口')
            extra=dict(received_pairs=received,rejected_skew=bad_skew,rejected_duplicate_timestamp=duplicates,
                       rate_limited=throttled,actual_wall_duration_s=time.monotonic()-began,
                       failure=str(failure) if failure else None)
            meta=writer.close(outcome,**extra)
            emit('sequence_done',directory=str(writer.directory.resolve()),summary=meta)
            if writer.error and failure is None:failure=OSError(writer.error)
        else:emit('preview_closed',message='预览已关闭；没有创建录制数据')
    if failure is not None:raise failure


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--name',required=True)
    p.add_argument('--max-skew-ms',type=float,default=20);args=p.parse_args()
    if not np.isfinite(args.max_skew_ms) or not 0<args.max_skew_ms<=100:raise ValueError('时间差阈值不正确')
    signal.signal(signal.SIGTERM,interrupted)
    try:capture(args)
    except KeyboardInterrupt:emit('cancelled');return 130
    except Exception as exc:emit('error',message=str(exc));return 1
    return 0


if __name__=='__main__':sys.exit(main())
