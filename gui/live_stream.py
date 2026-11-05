"""Latest-frame-only scheduling on the Qt thread; inference remains in its worker."""
import base64
from collections import deque
from datetime import datetime
import json
from pathlib import Path
import tempfile
import time
import uuid
import numpy as np
from PyQt5 import QtCore, QtGui
from gui_camera import select_camera_intrinsics

ROOT = Path(__file__).resolve().parents[1]


class LiveStream(QtCore.QObject):
    def __init__(self, window):
        super().__init__(window)
        self.w = window
        self.active = False
        self.pending = None
        self.paint_pending = None
        self.temp = None
        self.samples = deque(maxlen=2000)
        self.attempts = self.successes = self.errors = 0
        self.last_summary = None
        self.error_counts = {}
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.poll)
        self.w.viewer.frameSwapped.connect(lambda: self.painted('mhr' if getattr(self,'display_mode',None)=='mhr' else '3d'))
        self.w.overlay.frameSwapped.connect(lambda: self.painted('overlay'))
        self.w.smpl_view.frameSwapped.connect(lambda:self.painted('smpl'))

    @property
    def running(self):
        return self.active or self.pending is not None or self.paint_pending is not None

    def toggle(self):
        if self.active:
            self.stop()
        elif not self.running:
            self.start()

    def start(self):
        w = self.w
        if w.sam3_blocks_models():
            w.status.setText('SAM3 正在分割，暂不能启动人体实时流');return
        if w.camera_state != 'preview' or w.inference_busy or w.model_state != 'ready' or w.draft_config() != w.active_config:
            w.status.setText('请先应用并预热方案，再打开相机预览')
            return
        if w.stream_mode.currentData()=='smpl' and not w.active_config.smpl_enabled:
            w.status.setText('请先启用 SMPL 转换并应用预热，或选择其他连续显示模式');return
        if w.stream_mode.currentData()=='overlay' and w.overlay.failed:
            w.status.setText('叠加渲染器不可用，请重开窗口或选择 3D 预览');return
        self.active = True
        self.temp = tempfile.TemporaryDirectory(prefix='sam3d-live-')
        self.samples.clear()
        self.error_counts = {}
        self.attempts = self.successes = self.errors = 0
        self.last_index = None
        self.next_due = 0
        self.started = time.monotonic()
        self.stream_id=uuid.uuid4().hex
        self.roi = list(w.canvas.bbox) if w.canvas.bbox else None
        self.use_intrinsics = w.intrinsics_check.isChecked()
        self.metadata = dict(w.capture_metadata or {})
        self.config = w.active_config
        self.display_mode = w.stream_mode.currentData()
        self.input_image = None
        w.clear_result()
        w.viewer.stream_anchor = None
        if self.display_mode=='overlay':w.tabs.setCurrentWidget(w.overlay)
        elif self.display_mode=='smpl':w.tabs.setCurrentWidget(w.smpl_view)
        else:w.tabs.setCurrentIndex(0)
        if not self.active:
            return
        w.tabs.setTabText(1, '图像对照 · 连续模式不生成')
        w.set_busy(True)
        w.status.setText('连续推理已启动，等待最新相机帧…')
        self.timer.start()
        self.poll()

    def poll(self):
        if self.pending:
            if time.monotonic()-self.sent_at > 30:
                self.stop('连续推理超时，正在释放模型')
                self.w.model_state = 'stopping'
                self.w.process.terminate()
                self.w.kill_timer.start(2500)
            return
        if self.paint_pending:
            # Hidden/minimized windows do not swap. Resume dispatch when visible.
            return
        if not self.active:
            return
        w = self.w
        packet = w.camera.latest_frame
        if not packet or packet['index'] == self.last_index or w.camera_state != 'preview':
            return
        now = time.monotonic()
        if now < self.next_due:
            return
        if now-packet.get('received_monotonic', 0) > .5:
            w.status.setText('连续推理：等待新鲜相机帧…')
            return
        try:
            matrix, origin = select_camera_intrinsics(self.use_intrinsics, 1280, 800, self.metadata)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            self.stop('连续推理内参无效：'+str(exc));return
        self.last_index = packet['index']
        self.pending = uuid.uuid4().hex
        self.sent_at = now
        self.received_at = packet['received_monotonic']
        self.attempts += 1
        if self.display_mode == 'overlay':
            self.input_image = QtGui.QImage.fromData(base64.b64decode(packet['jpeg']), 'JPG')
            if self.input_image.isNull():
                self.pending = None
                self.stop('相机帧解码失败');return
        command = dict(type='stream', job=self.pending,stream_id=self.stream_id,host_received_monotonic=self.received_at, jpeg=packet['jpeg'], frame_index=packet['index'],
            output=self.temp.name, bbox=self.roi, inference_type=self.config.mode,
            camera_intrinsics=matrix, camera_intrinsics_origin=origin,
            backend_id=w.backend_id, session_id=w.model_session, config_id=self.config.config_id)
        w.process.write((json.dumps(command)+'\n').encode())

    def handle_model_event(self, event):
        if not self.pending or event.get('job') != self.pending:
            return False
        if event['type'] not in ('stream_mesh', 'error'):
            return False
        self.pending = None
        if not self.active:
            self.finish()
            return True
        if event['type'] == 'error':
            self.errors += 1
            message = event['message']
            key = message if message in self.error_counts or len(self.error_counts)<10 else '其他错误'
            self.error_counts[key] = self.error_counts.get(key, 0)+1
            self.next_due = time.monotonic()+.2
            self.w.status.setText('连续推理：'+event['message']+' · 将重试最新帧')
            self.w.viewer.clear_mesh()
            self.w.motion.clear_pose()
            self.w.motion.record_event(dict(type='gap', frame_index=event.get('frame_index',self.last_index),
                host_received_monotonic=self.received_at,message=event['message']))
            if self.display_mode=='overlay':
                self.w.overlay.clear_frame(self.input_image)
            self.w.result_label.setText('当前帧未获得有效人体 · 等待重新入画')
            # Retry at most 5 Hz on invalid/no-person frames.
            QtCore.QTimer.singleShot(200, self.poll)
            return True
        try:
            with np.load(event['path'], allow_pickle=False) as data:
                vertices, faces = data['pred_vertices'], data['faces']
                data_pose = {key:data[key].copy() for key in data.files if key not in ('pred_vertices','faces')}
                if not np.isfinite(vertices).all():
                    raise ValueError('网格数值无效')
                self.w.lab.set_latest(data, f'连续推理帧 {event["frame_index"]} · {self.w.backend_id}')
                if self.display_mode == 'overlay':
                    translation = data['pred_cam_t']
                    self.w.overlay.set_frame(self.input_image, vertices, translation, faces,
                        event['camera_intrinsics_used']['matrix'], event['frame_index'])
                elif self.display_mode=='3d':
                    self.w.viewer.set_mesh(vertices, faces, self.w.wire_check.isChecked(), continuous=True)
            self.w.motion.set_pose(data_pose, dict(frame_index=event['frame_index'],
                host_received_monotonic=self.received_at, inference_ms=event['inference_seconds']*1000,
                camera_intrinsics=event['camera_intrinsics_used'],backend_id=self.w.backend_id,
                config_id=self.config.config_id, launch_config=self.config.to_dict()))
            self.w.motion.record_event(dict(type='pose',frame_index=event['frame_index'],
                host_received_monotonic=self.received_at, inference_ms=event['inference_seconds']*1000,
                camera_intrinsics=event['camera_intrinsics_used'],pose=self.w.motion.pose))
            self.paint_pending = dict(frame_index=event['frame_index'], inference_ms=event['inference_seconds']*1000,
                worker_ms=event['worker_seconds']*1000, received_at=self.received_at,
                camera_intrinsics=event['camera_intrinsics_used'],smpl_info=event.get('smpl_info'),mhr_info=event.get('mhr_info'))
            (self.w.overlay if self.display_mode=='overlay' else self.w.smpl_view if self.display_mode=='smpl' else self.w.viewer).update()
        except (OSError, ValueError, KeyError) as exc:
            self.stop('连续网格读取失败：'+str(exc))
        return True

    def painted(self, mode):
        if not self.paint_pending or mode != self.display_mode:
            return
        sample = self.paint_pending
        self.paint_pending = None
        if not self.active:
            self.finish();return
        now = time.monotonic()
        sample['display_monotonic'] = now
        sample['frame_to_display_ms'] = (now-sample.pop('received_at'))*1000
        self.samples.append(sample)
        self.successes += 1
        recent = list(self.samples)[-30:]
        fps = (len(recent)-1)/(recent[-1]['display_monotonic']-recent[0]['display_monotonic']) if len(recent)>1 else 0
        k = '真实内参' if sample['camera_intrinsics']['source']=='provided_rgb' else '默认内参'
        self.w.result_label.setText(f'{"彩色叠加" if self.display_mode=="overlay" else "SMPL" if self.display_mode=="smpl" else "MHR 骨架" if self.display_mode=="mhr" else "连续 3D"} · {fps:.1f} FPS · 帧到显示 {sample["frame_to_display_ms"]:.0f} ms · 模型 {sample["inference_ms"]:.0f} ms · {k}')
        self.w.status.setText(f'正在连续推理 · 已更新 {self.successes} 帧 · 失败 {self.errors} 帧 · 点停止结束')
        QtCore.QTimer.singleShot(0, self.poll)

    def stop(self, message='连续推理已停止，保留最后的 3D 预览'):
        if not self.running:
            return
        self.active = False
        self.w.motion.stop_record(message)
        self.stop_message = message
        if self.paint_pending:
            self.paint_pending = None
        self.w.stream_btn.setText('正在停止…' if self.pending else '开始连续推理')
        self.w.update_camera_controls()
        if not self.pending:
            self.finish()

    def abort(self, message):
        if self.running:
            self.w.motion.stop_record(message)
            self.pending = self.paint_pending = None
            self.active = False
            self.stop_message = message
            self.finish()

    def finish(self):
        self.timer.stop()
        duration = time.monotonic()-self.started
        samples = list(self.samples)
        self.last_summary = dict(duration_seconds=duration, attempts=self.attempts, successes=self.successes, errors=self.errors,
            display_mode=self.display_mode, gpu_renderer=self.w.overlay.renderer_name if self.display_mode=='overlay' else None,
            launch_config=self.config.to_dict(), bbox=self.roi, use_gemini_intrinsics=self.use_intrinsics, error_counts=self.error_counts,
            camera_metadata=self.metadata, smpl_assets=getattr(self.w,'smpl_asset_hashes',None), samples=samples, retained_samples=len(samples),
            timing_scope='host RGB receipt to OpenGL frameSwapped; excludes sensor exposure and USB delivery; not accuracy validation')
        if samples:
            for field in ('inference_ms','worker_ms','frame_to_display_ms'):
                values=[v[field] for v in samples]
                self.last_summary[field] = dict(mean=float(np.mean(values)),median=float(np.median(values)),p95=float(np.percentile(values,95)))
            mhr_stages=[x['mhr_info']['processing_ms'] for x in samples if x.get('mhr_info')]
            if mhr_stages:self.last_summary['mhr_filter_ms']=dict(mean=float(np.mean(mhr_stages)),p95=float(np.percentile(mhr_stages,95)))
            stages=[x['smpl_info']['postprocess_ms'] for x in samples if x.get('smpl_info')]
            if stages:self.last_summary['smpl_postprocess_ms']=dict(mean=float(np.mean(stages)),p95=float(np.percentile(stages,95)))
            if len(samples)>1:
                self.last_summary['display_fps']=(len(samples)-1)/(samples[-1]['display_monotonic']-samples[0]['display_monotonic'])
        target=ROOT/'outputs/streams';target.mkdir(parents=True, exist_ok=True)
        path=target/(datetime.now().strftime('%Y%m%d-%H%M%S-%f')+'.json')
        path.write_text(json.dumps(self.last_summary,ensure_ascii=False,indent=2))
        self.summary_path=path
        self.input_image = None
        if self.temp:
            self.temp.cleanup();self.temp=None
        self.w.set_busy(False)
        self.w.status.setText(self.stop_message)
        self.w.tabs.setTabText(1, '图像对照')
