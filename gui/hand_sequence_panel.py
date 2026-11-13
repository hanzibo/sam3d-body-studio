# SPDX-License-Identifier: GPL-3.0-only
"""Offline Gemini sequence jobs, scrubbing and calibrated two-hand playback."""
from datetime import datetime
import json
from pathlib import Path
import uuid
import numpy as np
from PyQt5 import QtCore,QtGui,QtWidgets
from studio_paths import path
from hand_sequence import read_sequence
from overlay_view import OverlayView


class HandSequencePanel(QtWidgets.QWidget):
    def __init__(self,host,viewer_factory):
        super().__init__();self.host=host;self.source=None;self.directory=None;self.rows=[];self.info_data=None
        self.timer=QtCore.QTimer(self);self.timer.timeout.connect(self.advance)
        layout=QtWidgets.QVBoxLayout(self)
        hint=QtWidgets.QLabel('离线录制目录：manifest.json + timestamps.jsonl + frames/。每帧自动找手；同一人最多左、右各一只。未检测到的手不沿用旧网格。')
        hint.setWordWrap(True);layout.addWidget(hint)
        row=QtWidgets.QHBoxLayout();layout.addLayout(row)
        self.open=QtWidgets.QPushButton('打开录制目录…');self.open.clicked.connect(self.choose_source);row.addWidget(self.open)
        self.existing=QtWidgets.QPushButton('打开已有结果…');self.existing.clicked.connect(self.choose_result);row.addWidget(self.existing)
        row.addWidget(QtWidgets.QLabel('抽帧间隔'))
        self.stride=QtWidgets.QSpinBox();self.stride.setRange(1,120);self.stride.setValue(1);row.addWidget(self.stride)
        self.stabilize=QtWidgets.QCheckBox('左右标签连续性');self.stabilize.setChecked(True);row.addWidget(self.stabilize)
        self.video=QtWidgets.QCheckBox('导出叠加 MP4');self.video.setChecked(True);row.addWidget(self.video)
        self.start=QtWidgets.QPushButton('逐帧自动重建');self.start.clicked.connect(self.run);row.addWidget(self.start)
        self.label=QtWidgets.QLabel('请打开数据集，再使用下方按钮预加载手部模型。');self.label.setWordWrap(True);layout.addWidget(self.label)
        self.progress=QtWidgets.QProgressBar();layout.addWidget(self.progress)
        row=QtWidgets.QHBoxLayout();layout.addLayout(row)
        self.play=QtWidgets.QPushButton('播放');self.play.clicked.connect(self.toggle_play);row.addWidget(self.play)
        self.slider=QtWidgets.QSlider(QtCore.Qt.Horizontal);self.slider.valueChanged.connect(self.show_frame);row.addWidget(self.slider,1)
        self.position=QtWidgets.QLabel('0 / 0');row.addWidget(self.position)
        self.view=QtWidgets.QComboBox();self.view.addItems(['原图叠加','三维场景']);row.addWidget(self.view)
        self.open_video=QtWidgets.QPushButton('打开 MP4');self.open_video.clicked.connect(self.launch_video);row.addWidget(self.open_video)
        self.stack=QtWidgets.QStackedWidget();layout.addWidget(self.stack,1)
        self.overlay=OverlayView();self.overlay.set_layers(True,False,False);self.stack.addWidget(self.overlay)
        self.viewer=viewer_factory();self.viewer.grid.setSize(.7,.7);self.viewer.grid.setSpacing(.05,.05);self.stack.addWidget(self.viewer)
        self.view.currentIndexChanged.connect(self.stack.setCurrentIndex);self.update_controls()

    def update_controls(self):
        busy=self.host.state in ('loading','busy','switching','stopping') or getattr(getattr(self.host,'live',None),'running',False)
        self.open.setEnabled(not busy);self.existing.setEnabled(not busy);self.stride.setEnabled(not busy)
        self.stabilize.setEnabled(not busy);self.video.setEnabled(not busy)
        self.start.setEnabled(self.source is not None and self.host.state=='ready' and not busy)
        self.play.setEnabled(bool(self.rows) and not busy);self.slider.setEnabled(bool(self.rows) and not busy)
        self.open_video.setEnabled(self.directory is not None and (self.directory/'overlay.mp4').is_file() and not busy)

    def choose_source(self):
        name=QtWidgets.QFileDialog.getExistingDirectory(self,'选择 Gemini 录制目录')
        if name:self.load_source(Path(name))

    def load_source(self,directory):
        try:
            manifest,records,k=read_sequence(directory);self.source=Path(directory).resolve()
            self.host.automatic.setChecked(True)
            self.label.setText(f"{manifest.get('label',self.source.name)} · {len(records)} 帧 · {manifest['profile']['fps']} FPS · 使用录制内参")
            self.timer.stop();self.play.setText('播放');self.rows=[];self.viewer.clear_mesh();self.overlay.clear_frame();self.viewer._hand_active=False
            self.update_controls()
        except Exception as exc:self.label.setText('录制目录无效：'+str(exc))

    def run(self):
        if self.host.state!='ready' or self.source is None or self.host.live.running:return
        self.timer.stop();self.play.setText('播放');self.rows=[];self.viewer._hand_active=False
        output=path('hand_sequence_outputs')/(datetime.now().strftime('%Y%m%d-%H%M%S-%f')+'-'+self.source.name)
        self.directory=output;self.host.result_dir=output;self.host.job=uuid.uuid4().hex
        self.host.state='busy';self.host.update_controls()
        command=dict(type='sequence',job=self.host.job,source=str(self.source),output=str(output),stride=self.stride.value(),
                     confidence=self.host.confidence.value(),hand_selection=self.host.hand_selection.currentData(),padding=self.host.padding.value(),stabilize=self.stabilize.isChecked(),render_video=self.video.isChecked())
        self.host.process.write((json.dumps(command)+'\n').encode());self.label.setText('开始逐帧重建…')

    def on_frame(self,event):
        self.directory=Path(event['output']);row=event['row'];self.rows.append(row)
        self.progress.setRange(0,event['total']);self.progress.setValue(len(self.rows))
        source=self.source
        self.info_data=dict(source=str(source))
        self.draw(row)
        self.label.setText(f"处理 {len(self.rows)}/{event['total']} · 原帧 {row['source_index']} · {row['report']['hand_count']} 手 · 检测 {row['report']['detect_ms']:.1f} ms / 重建 {row['report']['reconstruct_ms']:.1f} ms")

    def completed(self,event):
        self.load_result(Path(event['output']))
        info=event['report'];s=info['summary']
        self.label.setText(f"完成 {s['frames']} 帧 · 检测均值 {s['detection_mean_ms']:.1f} ms · 重建均值 {s['reconstruction_mean_ms']:.1f} ms · 含读写 {s['frame_with_io_mean_ms']:.1f} ms。"+('视频合成失败：'+info['video_error'] if 'video_error' in info else ''))

    def choose_result(self):
        name=QtWidgets.QFileDialog.getExistingDirectory(self,'选择包含 sequence.json 的手部结果目录',str(path('hand_sequence_outputs')))
        if name:self.load_result(Path(name))

    def load_result(self,directory):
        try:
            info=json.loads((directory/'sequence.json').read_text())
            if not info.get('complete'):raise ValueError('任务未完成；保留已输出帧供排查，不能当作完整结果播放')
            rows=[json.loads(line) for line in (directory/'frames.jsonl').read_text().splitlines()]
            if not rows or len(rows)!=info['frames']:raise ValueError('结果帧数不一致')
            self.timer.stop();self.play.setText('播放');self.directory=directory;self.info_data=info;self.rows=rows
            self.source=Path(info['source']);self.host.result_dir=directory;self.viewer._hand_active=False
            self.slider.blockSignals(True);self.slider.setRange(0,len(rows)-1);self.slider.setValue(0);self.slider.blockSignals(False)
            self.progress.setRange(0,len(rows));self.progress.setValue(len(rows));self.show_frame(0);self.update_controls()
            summary=info.get('summary',{})
            self.label.setText(f"已打开 {self.source.name} · {len(rows)} 帧 · 检测均值 {summary.get('detection_mean_ms',0):.1f} ms / 重建均值 {summary.get('reconstruction_mean_ms',0):.1f} ms · 可直接回放，无需加载模型。")
        except Exception as exc:self.label.setText('无法打开结果：'+str(exc))

    def draw(self,row):
        image=QtGui.QImage(str(Path(self.info_data['source'])/row['input_file']))
        if image.isNull():raise ValueError('源帧缺失')
        with np.load(self.directory/row['scene'],allow_pickle=False) as data:
            self.host.display_scene(image,data,row['report'],self.viewer,self.overlay,continuous=True)
        self.position.setText(f"{row['index']+1} / {len(self.rows)} · t={row['timestamp']:.2f}s")

    def show_frame(self,index):
        if not self.rows:return
        try:self.draw(self.rows[index])
        except Exception as exc:self.timer.stop();self.label.setText('回放失败：'+str(exc))

    def toggle_play(self):
        if self.timer.isActive():self.timer.stop();self.play.setText('播放');return
        if self.slider.value()==len(self.rows)-1:self.slider.setValue(0)
        self.timer.start(max(10,round(1000/float(self.info_data['playback_fps']))));self.play.setText('暂停')

    def advance(self):
        index=self.slider.value()+1
        if index>=len(self.rows):self.timer.stop();self.play.setText('播放');return
        self.slider.setValue(index)

    def launch_video(self):
        if self.directory:QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(self.directory/'overlay.mp4')))
