# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""First-tab point prompting, CPU morphology and isolated one-shot SAM3/RGB-D jobs."""
import base64
from datetime import datetime
import json
from pathlib import Path
import time
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from mask_prompt import morph_mask, sha256
from studio_paths import path

ROOT = Path(__file__).resolve().parents[1]


class ImageSegmentation(QtWidgets.QGroupBox):
    def __init__(self, host):
        super().__init__('SAM3 点提示 / 掩膜提示实验')
        self.host = host
        self.process = QtCore.QProcess(self)
        self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.read_output)
        self.process.finished.connect(self.finished)
        self.process.errorOccurred.connect(self.process_error)
        self.mode = None; self.buffer = ''; self.event = None; self.log = ''
        self.raw = self.effective = None; self.directory = None; self.result = None
        self.input_file = None; self.cancelled = False; self.started = 0; self.last_error = ''
        self.kill_timer = QtCore.QTimer(self); self.kill_timer.setSingleShot(True); self.kill_timer.timeout.connect(self.process.kill)
        self.timer = QtCore.QTimer(self); self.timer.setInterval(1000); self.timer.timeout.connect(self.check_timeout)
        layout = QtWidgets.QVBoxLayout(self)
        self.rgbd_btn = QtWidgets.QPushButton('RGB-D 倒计时拍照（先分割，再重建）')
        self.rgbd_btn.clicked.connect(self.capture); layout.addWidget(self.rgbd_btn)
        row = QtWidgets.QHBoxLayout()
        self.points = QtWidgets.QCheckBox('点提示模式'); self.points.toggled.connect(self.point_mode)
        undo = QtWidgets.QPushButton('撤销点'); undo.clicked.connect(self.undo)
        clear = QtWidgets.QPushButton('清除掩膜 / 点'); clear.clicked.connect(self.clear)
        for w in (self.points, undo, clear): row.addWidget(w)
        self.undo_btn, self.clear_btn = undo, clear
        layout.addLayout(row)
        row = QtWidgets.QHBoxLayout()
        self.segment_btn = QtWidgets.QPushButton('SAM3 分割'); self.segment_btn.clicked.connect(self.segment)
        self.cancel_btn = QtWidgets.QPushButton('取消'); self.cancel_btn.clicked.connect(self.cancel)
        row.addWidget(self.segment_btn); row.addWidget(self.cancel_btn); layout.addLayout(row)
        self.candidates = QtWidgets.QComboBox(); self.candidates.currentIndexChanged.connect(self.choose_candidate)
        layout.addWidget(self.candidates)
        self.use_mask = QtWidgets.QCheckBox('向人体网络传入掩膜提示（关闭可做对照）')
        self.use_mask.setChecked(True); self.use_mask.toggled.connect(lambda: self.host.canvas.update())
        layout.addWidget(self.use_mask)
        row = QtWidgets.QHBoxLayout()
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.slider.setRange(-40, 40)
        self.slider.setValue(0); self.slider.valueChanged.connect(self.morph)
        self.radius_label = QtWidgets.QLabel('原始掩膜 · 0 px')
        row.addWidget(self.slider, 1); row.addWidget(self.radius_label); layout.addLayout(row)
        self.status = QtWidgets.QLabel('左键：人体前景点；右键：背景点。负值腐蚀，正值膨胀（原图像素）。\n先分割，再预热人体模型；加载人体模型后禁用 SAM3。人物框不随滑杆改变。')
        self.status.setWordWrap(True); layout.addWidget(self.status)
        host.canvas.promptAdded.connect(self.add_point)

    @property
    def active(self):
        return self.mode is not None or self.process.state() != QtCore.QProcess.NotRunning

    @property
    def segmenting(self):
        return self.mode == 'segment' or (self.active and self.mode != 'capture')

    def blocked(self):
        h = self.host
        if h.model_state != 'off' or h.process.state() != QtCore.QProcess.NotRunning:
            return '人体模型已加载或尚未完全退出，请先释放模型'
        if h.lab.client.device != 'cpu' and h.lab.client.process.state() != QtCore.QProcess.NotRunning:
            return 'MHR GPU 模型未释放，请先在实验室释放模型'
        return self.common_block()

    def common_block(self):
        h = self.host
        if hasattr(h,'video') and h.video.active:return '视频任务正在运行，请先等待结束或取消'
        if h.inference_busy or h.live.running or h.camera_state not in ('off', 'photo') or h.camera.active:
            return '请先停止相机预览 / 重建 / 实时流'
        if h.rgbd.mode is not None or h.rgbd.segmenting or h.rgbd.pending_camera_action is not None or h.rgbd.batch.active:
            return 'RGB-D 标签页任务尚未结束'
        return ''

    def refresh(self):
        h = self.host
        idle = not self.active and not h.inference_busy and h.camera_state in ('off', 'photo') and not h.live.running
        has_image = h.source_path is not None and not h.canvas.image.isNull()
        self.segment_btn.setEnabled(idle and has_image and not self.blocked())
        self.segment_btn.setToolTip(self.blocked() or '需要至少一个人体前景点')
        self.rgbd_btn.setEnabled(idle and not self.common_block())
        for w in (self.points, self.undo_btn, self.clear_btn): w.setEnabled(idle and has_image)
        for w in (self.slider, self.use_mask): w.setEnabled(idle and self.raw is not None)
        self.candidates.setEnabled(idle and self.result is not None)
        self.cancel_btn.setEnabled(self.active)

    def point_mode(self, enabled):
        self.host.canvas.prompt_mode = enabled
        if enabled: self.host.auto_reconstruct.setChecked(False)

    def add_point(self, x, y, label):
        if self.active or len(self.host.canvas.prompts) >= 128: return
        self.host.canvas.prompts.append([x, y, label])
        # Changed prompts invalidate a previous segmentation until rerun.
        self.invalidate_mask(); self.host.canvas.update()

    def undo(self):
        if self.active: return
        if self.host.canvas.prompts: self.host.canvas.prompts.pop()
        self.invalidate_mask(); self.host.canvas.update()

    def invalidate_mask(self):
        self.raw = self.effective = None; self.result = None
        self.host.canvas.mask_overlay = QtGui.QImage()
        self.candidates.blockSignals(True); self.candidates.clear(); self.candidates.blockSignals(False)
        self.refresh()

    def clear(self):
        if self.active: return
        self.host.canvas.prompts = []; self.invalidate_mask()
        self.slider.setValue(0); self.host.canvas.update()

    def reset_image(self):
        self.clear(); self.directory = None; self.input_file = None

    def launch(self, mode, python, script, args):
        self.mode = mode; self.cancelled = False; self.buffer = ''; self.event = None; self.log = ''; self.last_error = ''; self.started = time.monotonic()
        self.process.setWorkingDirectory(str(ROOT))
        env = QtCore.QProcessEnvironment.systemEnvironment()
        env.remove('QT_QPA_PLATFORM_PLUGIN_PATH'); env.remove('QT_QPA_FONTDIR')
        self.process.setProcessEnvironment(env)
        self.process.start(str(python), [str(ROOT/script)] + args)
        self.timer.start(); self.host.update_camera_controls()

    def segment(self):
        if self.active: return
        reason = self.blocked()
        if reason: self.status.setText(reason); return
        h = self.host
        if h.source_path is None or not any(p[2] == 1 for p in h.canvas.prompts):
            self.status.setText('请启用点提示模式，在人体上至少点一个前景点'); return
        if not path('sam3_python').is_file() or not path('sam3_checkpoint').is_file():
            self.status.setText('找不到 SAM3 环境或权重'); return
        try:
            self.directory = ROOT/'outputs/segmentation'/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
            self.directory.mkdir(parents=True)
            self.input_file = self.directory/'input.png'
            if not h.canvas.image.save(str(self.input_file)): raise OSError('无法保存 SAM3 输入')
            req = dict(image=str(self.input_file), rgb_sha256=sha256(self.input_file), output=str(self.directory/'sam3'),
                       points_pixels=[p[:2] for p in h.canvas.prompts], point_labels=[p[2] for p in h.canvas.prompts])
            request = self.directory/'request.json'; request.write_text(json.dumps(req, ensure_ascii=False, indent=2))
        except OSError as exc:
            self.status.setText('无法保存分割输入：'+str(exc)); return
        self.status.setText('正在加载 SAM3；分割进程完全退出后才允许预热人体模型…')
        self.launch('segment', path('sam3_python'), 'sam3_segment.py', ['--request', str(request)])

    def capture(self):
        if self.active or self.common_block(): return
        self.host.auto_reconstruct.setChecked(False)
        self.reset_image(); self.host.clear_result()
        self.host.source_path = None
        self.directory = ROOT/'outputs/rgbd'/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        self.status.setText('RGB-D 倒计时拍照；请让头脚完整入画')
        self.launch('capture', path('camera_python'), 'rgbd_capture.py',
                    ['--output', str(self.directory), '--delay', str(self.host.delay_combo.currentData())])

    def read_output(self):
        self.buffer += bytes(self.process.readAllStandardOutput()).decode('utf-8', 'replace')
        while '\n' in self.buffer:
            line, self.buffer = self.buffer.split('\n', 1); self.log += line+'\n'
            if not line.startswith('@@RGBD '): continue
            try: event = json.loads(line[len('@@RGBD '):])
            except ValueError: continue
            kind = event.get('type')
            if kind in ('segmented', 'done'): self.event = event
            elif kind in ('error', 'stage', 'ready'):
                self.status.setText(event.get('message', ''))
                if kind == 'error': self.last_error = event.get('message', '')
            elif kind == 'countdown': self.host.canvas.overlay = str(event['seconds']); self.host.canvas.update()
            elif kind == 'frame' and self.mode == 'capture':
                image = QtGui.QImage.fromData(base64.b64decode(event['jpeg']))
                self.host.canvas.set_preview(image)

    def finished(self, code, *_):
        self.read_output(); mode = self.mode; self.mode = None; self.timer.stop(); self.kill_timer.stop()
        self.host.canvas.overlay = ''
        try:
            if self.directory and self.directory.is_dir(): (self.directory/'worker.log').write_text(self.log)
        except OSError as exc: self.last_error = '无法保存日志：'+str(exc)
        if code == 0 and self.event and not self.cancelled:
            try:
                if mode == 'capture':
                    self.host.load_image(str(Path(self.event['directory'])/'rgb.png'))
                    self.status.setText('RGB-D 已保存，相机已释放；可选点分割，再预热重建')
                elif mode == 'segment':
                    self.result = self.event['report']
                    self.candidates.blockSignals(True); self.candidates.clear()
                    for candidate in self.result['candidates']:
                        self.candidates.addItem(f"候选 {candidate['index']+1} · {candidate['score']:.3f}", candidate['index'])
                    self.candidates.setCurrentIndex(self.candidates.findData(self.result['selected']))
                    self.candidates.blockSignals(False); self.choose_candidate()
                    self.status.setText('分割完成，SAM3 进程已退出。调整掩膜后可预热；左右键仍可补点重分割。')
            except (OSError, ValueError, KeyError, TypeError) as exc: self.invalidate_mask(); self.status.setText('结果无效：'+str(exc))
        else:
            if mode == 'capture':
                self.host.source_path = None; self.host.canvas.set_image(QtGui.QImage()); self.reset_image()
            self.status.setText('已取消' if self.cancelled else '任务失败：'+(self.last_error or self.log[-700:])+'；完整日志在该次数据目录 worker.log')
        self.host.canvas.update(); self.host.update_camera_controls()

    def choose_candidate(self, *_):
        if self.active or not self.result or self.candidates.currentData() is None: return
        raw = cv2.imread(str(self.directory/'sam3'/f'mask_{self.candidates.currentData():02d}.png'), 0)
        if raw is None or raw.shape != (self.host.canvas.image.height(), self.host.canvas.image.width()) or not raw.any():
            self.raw = self.effective = None; self.host.canvas.mask_overlay = QtGui.QImage()
            self.host.canvas.update(); self.refresh()
            self.status.setText('候选掩膜为空或尺寸错误，请换候选'); return
        self.raw = (raw > 0).astype(np.uint8)
        x, y, w, h = cv2.boundingRect(self.raw)
        self.host.canvas.bbox = [x, y, x+w, y+h]; self.host.bbox_changed()
        self.morph(); self.refresh()

    def morph(self, *_):
        radius = self.slider.value()
        self.radius_label.setText(('膨胀' if radius > 0 else '腐蚀' if radius < 0 else '原始')+f' {radius} px')
        if self.raw is None: return
        self.effective = morph_mask(self.raw, radius)
        rgba = np.zeros((*self.effective.shape, 4), np.uint8)
        rgba[self.effective > 0] = [60, 230, 150, 90]
        self.host.canvas.mask_overlay = QtGui.QImage(rgba.data, rgba.shape[1], rgba.shape[0], rgba.strides[0], QtGui.QImage.Format_RGBA8888).copy()
        self.host.canvas.update()
        if not self.effective.any(): self.status.setText('腐蚀后掩膜为空，请减小腐蚀半径；禁止重建')

    def snapshot(self, directory, image):
        if self.use_mask.isChecked() and self.result is not None and self.raw is None:
            raise ValueError('所选候选掩膜无效，请换候选或清除掩膜')
        if self.raw is None or not self.use_mask.isChecked(): return None
        if self.effective is None or not self.effective.any(): raise ValueError('掩膜为空，请减小腐蚀半径')
        original = QtGui.QImage(str(self.input_file))
        if original.convertToFormat(QtGui.QImage.Format_RGBA8888) != self.host.canvas.image.convertToFormat(QtGui.QImage.Format_RGBA8888): raise ValueError('图片已改变，请重新分割')
        target = Path(directory)/'mask_prompt.png'; raw_path = Path(directory)/'mask_original.png'
        if not cv2.imwrite(str(target), self.effective*255) or not cv2.imwrite(str(raw_path), self.raw*255): raise OSError('保存掩膜失败')
        info = dict(path=str(target), mask_sha256=sha256(target), image_sha256=sha256(image),
                    original_mask_sha256=sha256(raw_path), morphology_radius_px=self.slider.value(),
                    method='sam3_point_prompts', selected=self.candidates.currentData(),
                    points_pixels=[p[:2] for p in self.host.canvas.prompts], point_labels=[p[2] for p in self.host.canvas.prompts],
                    bbox_xyxy=self.host.canvas.bbox, segmentation_report=self.result)
        return info

    def process_error(self, error):
        if error == QtCore.QProcess.FailedToStart: self.finished(-1)

    def check_timeout(self):
        if time.monotonic()-self.started > (240 if self.mode == 'segment' else 120):
            self.status.setText('任务超时，正在取消'); self.cancel()

    def cancel(self):
        if not self.active: return
        self.cancelled = True; self.process.terminate()
        self.kill_timer.start(2500)

    def shutdown(self):
        self.cancelled = True; self.timer.stop(); self.kill_timer.stop()
        if self.process.state() != QtCore.QProcess.NotRunning:
            self.process.kill(); self.process.waitForFinished(5000)
