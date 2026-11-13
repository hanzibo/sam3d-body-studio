# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""RGB-only Orbbec worker. SDK stays in its existing Python environment."""
import argparse
import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import select
import socket
import struct
import sys
import time

import cv2
from pyorbbecsdk import Config, Context, OBSensorType, Pipeline, get_version
from orbbec_color import color_frame_to_bgr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--socket', required=True)
    args = parser.parse_args()
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(2)
    sock.connect(args.socket)
    def send(kind, **values):
        data = json.dumps(dict(type=kind, **values)).encode()
        sock.sendall(struct.pack('!I', len(data)) + data)
    pipeline = None
    buffer = b''
    pending = None
    try:
        context = Context()
        devices = context.query_devices()
        device = next((devices.get_device_by_index(i) for i in range(devices.get_count())
                       if devices.get_device_by_index(i).get_device_info().get_pid() == 0x0807), None)
        if device is None:
            raise RuntimeError('未找到 Gemini 336L，请检查 USB 连接或其他占用程序')
        info = device.get_device_info()
        pipeline = Pipeline(device)
        profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        profile = next((p for i in range(len(profiles))
                        if (p := profiles[i].as_video_stream_profile()).get_width() == 1280
                        and p.get_height() == 800 and p.get_fps() == 30 and 'MJPG' in str(p.get_format())), None)
        if profile is None:
            raise RuntimeError('相机不支持所需的 1280×800 MJPG 30 FPS 配置')
        intrinsic = profile.get_intrinsic()
        metadata = dict(device=info.get_name(), serial=info.get_serial_number(),
                        firmware=info.get_firmware_version(), sdk=str(get_version()),
                        width=1280, height=800, fps=30, format='MJPG',
                        intrinsics={k:getattr(intrinsic,k) for k in ['width','height','fx','fy','cx','cy']},
                        depth_enabled=False)
        config = Config()
        config.enable_stream(profile)
        pipeline.start(config)
        last_frame = time.monotonic()
        preview_at = 0
        preview_fps = 15
        count = 0
        ready = False
        restarted = False
        previous_index = None
        previous_timestamp = None
        running = True
        while running:
            if select.select([sock], [], [], 0)[0]:
                data = sock.recv(65536)
                if not data:
                    break
                buffer += data
                while len(buffer) >= 4:
                    size = struct.unpack('!I', buffer[:4])[0]
                    if size > 1048576:
                        raise ValueError('控制消息过大')
                    if len(buffer) < size+4:
                        break
                    command = json.loads(buffer[4:4+size])
                    buffer = buffer[4+size:]
                    if command['type'] == 'stop':
                        running = False
                        pending = None
                    elif command['type'] == 'preview_rate':
                        preview_fps=max(1,min(30,int(command.get('fps',15))));preview_at=0
                    elif command['type'] == 'cancel':
                        pending = None
                        send('cancelled')
                    elif command['type'] == 'capture' and ready and running:
                        delay = max(0, min(60, float(command['delay'])))
                        pending = dict(command, deadline=time.monotonic()+delay)
                        send('countdown', job=command['job'], delay=delay)
            if not running:
                break
            frames = pipeline.wait_for_frames(100)
            frame = frames.get_color_frame() if frames else None
            # MJPG live preview already has an encoded image. Decode only for
            # PNG capture or fallback formats, never on every preview frame.
            encoded = bytes(frame.get_data()) if frame else b''
            jpeg = encoded.startswith(b'\xff\xd8')
            image = color_frame_to_bgr(frame) if frame and not jpeg else None
            now = time.monotonic()
            valid = frame is not None and (jpeg or image is not None)
            if valid:
                index, timestamp = frame.get_index(), frame.get_timestamp_us()
                if previous_index is not None and (index <= previous_index or timestamp <= previous_timestamp):
                    valid = False
                else:
                    previous_index, previous_timestamp = index, timestamp
            if not valid:
                if now-last_frame > 3:
                    if not ready and not restarted:
                        send('status', message='首次取帧超时，正在重新启动采集…')
                        pipeline.stop()
                        pipeline.start(config)
                        restarted = True
                        previous_index = previous_timestamp = None
                        last_frame = time.monotonic()
                        count = 0
                        continue
                    raise RuntimeError('连续 3 秒没有收到有效图像，请检查相机连接')
                continue
            last_frame = now
            count += 1
            if count >= 15 and not ready:
                ready = True
                send('ready', metadata=metadata)
            if pending and now >= pending['deadline']:
                if image is None:
                    image = color_frame_to_bgr(frame)
                    if image is None:
                        continue
                target = Path(pending['directory'])
                target.mkdir(parents=True, exist_ok=False)
                photo = target/'photo.png'
                if not cv2.imwrite(str(photo), image):
                    raise OSError('无法保存相机照片')
                record = dict(metadata, frame_index=frame.get_index(), device_timestamp_us=frame.get_timestamp_us(),
                              received_monotonic=now, captured_utc=datetime.now(timezone.utc).isoformat(),
                              countdown_seconds=float(pending['delay']), deadline_monotonic=pending['deadline'],
                              deadline_to_arrival_seconds=now-pending['deadline'], job=pending['job'])
                (target/'camera.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
                send('photo', job=pending['job'], path=str(photo), metadata=record)
                pending = None
            if now+.001 >= preview_at:
                # The selected camera profile already emits JPEG; preserve its pixels
                # rather than decode/encode a second lossy JPEG for the socket.
                if not jpeg:
                    ok, encoded=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,85])
                    if not ok:continue
                send('frame',jpeg=base64.b64encode(encoded).decode(),index=frame.get_index(),
                     received_monotonic=now,device_timestamp_us=frame.get_timestamp_us(),preview_fps=preview_fps,
                     receipt_stage='sdk_color_frame_available')
                preview_at+=1/preview_fps
                if preview_at<now-1/preview_fps:preview_at=now+1/preview_fps
    except Exception as exc:
        try:
            send('error', message=str(exc))
        except OSError:
            pass
        print(str(exc), file=sys.stderr, flush=True)
        return 1
    finally:
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass
        sock.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
