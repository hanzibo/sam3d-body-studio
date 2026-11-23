# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Model-free named RGB-D stream recording panel; SDK runs out of process."""
from datetime import datetime
import json
from pathlib import Path
import shutil
import time
from PyQt5 import QtCore, QtGui, QtWidgets
from studio_paths import ROOT, environment, path
from studio.rgbd.rgbd_sequence import PRESETS, action_plan, safe_name


class StreamRecorder(QtWidgets.QWidget):
    busyChanged=QtCore.pyqtSignal(bool)

    def __init__(self, preview_class, camera_python=None, capture_script=None, output_root=None):
        super().__init__()
        self.camera_python=str(camera_python or path('camera_python'))
        self.capture_entry=[str(capture_script)] if capture_script is not None else ['-m','studio.cameras.rgbd_stream_capture']
        self.output_root=Path(output_root or ROOT/'outputs/rgbd-sequences')
        self.state='idle';self.pending=None;self.summary=None;self.buffer='';self.frames=0;self.last_frame=0.
        self.started=0.;self.last_action='';self.log_lines=[];self.stop_requested=False
        self.process=QtCore.QProcess(self);self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.read_output);self.process.finished.connect(self.finished)
        self.process.errorOccurred.connect(self.process_error)
        self.timer=QtCore.QTimer(self);self.timer.setInterval(500);self.timer.timeout.connect(self.tick);self.timer.start()
        layout=QtWidgets.QVBoxLayout(self)
        title=QtWidgets.QLabel('连续 RGB-D 录制');title.setObjectName('title');layout.addWidget(title)
        sub=QtWidgets.QLabel('先预览调整站位，再倒计时开录。只采集相机，不加载模型；原始 RGB＋无损深度＋标定＋逐帧时间戳一并保存。')
        sub.setWordWrap(True);sub.setObjectName('muted');layout.addWidget(sub)
        form=QtWidgets.QGridLayout()
        self.name=QtWidgets.QLineEdit('双臂抬放');self.name.setMaxLength(80);self.name.setPlaceholderText('给本段数据命名，例如：双臂抬放_正面_2米_01')
        self.name.setToolTip('名称在连接前设置；预览期间可调整动作与时长，修改名称请先断开。')
        form.addWidget(QtWidgets.QLabel('数据名称'),0,0);form.addWidget(self.name,0,1,1,3)
        self.preset=QtWidgets.QComboBox()
        for key,(label,*_) in PRESETS.items():self.preset.addItem(label,key)
        form.addWidget(QtWidgets.QLabel('动作方案'),0,4);form.addWidget(self.preset,0,5)
        self.duration=QtWidgets.QSpinBox();self.duration.setRange(5,300);self.duration.setValue(20);self.duration.setSuffix(' 秒')
        self.delay=QtWidgets.QSpinBox();self.delay.setRange(0,30);self.delay.setValue(5);self.delay.setSuffix(' 秒')
        self.fps=QtWidgets.QComboBox()
        for value in (30,15,10):self.fps.addItem(f'{value} FPS',value)
        form.addWidget(QtWidgets.QLabel('录制时长'),1,0);form.addWidget(self.duration,1,1)
        form.addWidget(QtWidgets.QLabel('开录倒计时'),1,2);form.addWidget(self.delay,1,3)
        form.addWidget(QtWidgets.QLabel('目标保存帧率'),1,4);form.addWidget(self.fps,1,5)
        self.notes=QtWidgets.QLineEdit();self.notes.setPlaceholderText('选填：替换中段动作提示，也会随数据保存；例如距离、遮挡条件')
        form.addWidget(QtWidgets.QLabel('动作 / 备注'),2,0);form.addWidget(self.notes,2,1,1,5);layout.addLayout(form)
        splitter=QtWidgets.QSplitter();layout.addWidget(splitter,1)
        side=QtWidgets.QWidget();sl=QtWidgets.QVBoxLayout(side);sl.setContentsMargins(0,0,6,0)
        sl.addWidget(QtWidgets.QLabel('本次动作时间表'))
        self.plan=QtWidgets.QListWidget();self.plan.setMaximumHeight(230);self.plan.setWordWrap(True);sl.addWidget(self.plan)
        self.plan.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        hint=QtWidgets.QLabel('开头 2 秒自然站立。动作尽量缓慢，全身完整入画。\n\n实际帧率以保存结果为准；预览固定低帧率，不代表录制帧率。')
        hint.setWordWrap(True);hint.setObjectName('muted');sl.addWidget(hint)
        sl.addWidget(QtWidgets.QLabel('最近录制（双击打开目录）'))
        self.history=QtWidgets.QListWidget();self.history.setWordWrap(True);self.history.itemDoubleClicked.connect(self.open_history);sl.addWidget(self.history,1)
        self.history.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        views=QtWidgets.QWidget();vl=QtWidgets.QVBoxLayout(views);vl.setContentsMargins(0,0,0,0)
        hero=QtWidgets.QHBoxLayout()
        self.clock=QtWidgets.QLabel('待机');self.clock.setAlignment(QtCore.Qt.AlignCenter);self.clock.setMinimumWidth(130)
        self.clock.setStyleSheet('font-size:34px; font-weight:bold; color:#8bc4ff; padding:6px;')
        self.prompt=QtWidgets.QLabel('连接相机，调整站位后开始录制');self.prompt.setWordWrap(True);self.prompt.setObjectName('poseTitle')
        hero.addWidget(self.clock);hero.addWidget(self.prompt,1);vl.addLayout(hero)
        preview_split=QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.rgb=preview_class('彩色预览 · 点击“连接并预览”');self.depth=preview_class('原始深度伪彩 · 黑色为无效深度')
        for caption,view in (('彩色预览',self.rgb),('原始深度（未配准）',self.depth)):
            container=QtWidgets.QWidget();box=QtWidgets.QVBoxLayout(container);box.setContentsMargins(0,0,0,0)
            box.addWidget(QtWidgets.QLabel(caption));box.addWidget(view,1);preview_split.addWidget(container)
        preview_split.setSizes([720,280]);vl.addWidget(preview_split,1)
        self.progress=QtWidgets.QProgressBar();self.progress.setRange(0,1000);self.progress.setValue(0);self.progress.setFormat('等待开始');vl.addWidget(self.progress)
        self.stats=QtWidgets.QLabel('尚未保存帧 · 预览不会创建数据目录');self.stats.setWordWrap(True);vl.addWidget(self.stats)
        splitter.addWidget(side);splitter.addWidget(views);splitter.setSizes([300,900])
        self.location=QtWidgets.QLabel('保存根目录：'+str(self.output_root));self.location.setWordWrap(True)
        self.location.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse);layout.addWidget(self.location)
        row=QtWidgets.QHBoxLayout()
        self.connect_btn=QtWidgets.QPushButton('连接并预览');self.connect_btn.clicked.connect(self.start_preview);row.addWidget(self.connect_btn)
        self.sound=QtWidgets.QCheckBox('倒计时 / 换动作提示音');self.sound.setChecked(True);row.addWidget(self.sound)
        self.disk=QtWidgets.QLabel();row.addWidget(self.disk);row.addStretch()
        self.record_btn=QtWidgets.QPushButton('倒计时开始录制');self.record_btn.setObjectName('primary');self.record_btn.clicked.connect(self.begin_record);row.addWidget(self.record_btn)
        self.stop_btn=QtWidgets.QPushButton('断开相机');self.stop_btn.clicked.connect(self.stop);row.addWidget(self.stop_btn)
        self.folder_btn=QtWidgets.QPushButton('打开数据目录');self.folder_btn.clicked.connect(self.open_folder);row.addWidget(self.folder_btn);layout.addLayout(row)
        self.status=QtWidgets.QLabel('先关闭 Studio 等其他程序的相机预览。Ctrl+Enter 开始录制；Esc 提前结束并保存已采集帧。')
        self.status.setWordWrap(True);self.status.setObjectName('status');layout.addWidget(self.status)
        self.logs_btn=QtWidgets.QCheckBox('显示采集日志');layout.addWidget(self.logs_btn)
        self.log=QtWidgets.QPlainTextEdit();self.log.setReadOnly(True);self.log.setMaximumHeight(110);self.log.hide()
        self.log.document().setMaximumBlockCount(300);self.logs_btn.toggled.connect(self.log.setVisible);layout.addWidget(self.log)
        for control in (self.duration,self.preset):
            (control.valueChanged if control is self.duration else control.currentIndexChanged).connect(self.update_plan)
        self.notes.textChanged.connect(self.update_plan)
        self.preset.currentIndexChanged.connect(self.preset_name)
        self.name.textChanged.connect(self.refresh)
        self.update_plan();self.refresh_history();self.refresh();self.tick()

    @property
    def busy(self):return self.state!='idle' or self.process.state()!=QtCore.QProcess.NotRunning

    def preset_name(self):
        titles=[v[0] for v in PRESETS.values()]+['双臂抬放']
        if self.state=='idle' and self.name.text() in titles:self.name.setText(PRESETS[self.preset.currentData()][0])

    def update_plan(self):
        self.plan.clear()
        for p in action_plan(self.duration.value(),self.preset.currentData(),self.notes.text()):
            self.plan.addItem(f"{p['start_s']:g}–{p['end_s']:g} 秒  {p['text']}")

    def start_preview(self):
        if self.busy:return
        try:
            safe=safe_name(self.name.text())
            if not Path(self.camera_python).is_file():raise ValueError('相机 Python 环境不存在，请检查路径配置')
            self.pending=self.output_root/(safe+'__'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
        except Exception as exc:self.fail(str(exc));return
        self.summary=None;self.frames=0;self.buffer='';self.log_lines=[];self.last_frame=0.;self.last_action='';self.stop_requested=False
        self.progress.setValue(0);self.progress.setFormat('等待开录');self.stats.setText('实时预览不保存 · 调整头脚完整入画')
        self.started=time.monotonic();self.state='connecting'
        env=QtCore.QProcessEnvironment()
        for k,v in environment('camera').items():env.insert(k,v)
        self.process.setProcessEnvironment(env);self.process.setWorkingDirectory(str(ROOT))
        self.process.start(self.camera_python,['-u',*self.capture_entry,'--output',str(self.pending),'--name',self.name.text().strip()])
        self.location.setText('本次计划保存到：'+str(self.pending));self.status.setText('正在连接 Gemini 336L…');self.refresh()

    def begin_record(self):
        if not self.record_btn.isEnabled():return
        if time.monotonic()-self.last_frame>3:self.fail('预览已中断，请断开后重新连接');return
        self.state='countdown';self.started=time.monotonic()
        self.process.write((json.dumps(dict(type='record',duration=self.duration.value(),delay=self.delay.value(),
            fps=self.fps.currentData(),preset=self.preset.currentData(),custom=self.notes.text().strip()))+'\n').encode())
        self.status.setText('倒计时结束后自动录制；请按屏幕动作提示缓慢运动。');self.refresh()

    def read_output(self):
        self.buffer+=bytes(self.process.readAllStandardOutput()).decode('utf-8',errors='replace')
        while '\n' in self.buffer:
            line,self.buffer=self.buffer.split('\n',1)
            if not line.startswith('@@RGBD '):self.add_log(line);continue
            try:self.handle_event(json.loads(line[7:]))
            except Exception as exc:self.fail('采集事件无效：'+str(exc));self.stop()

    def handle_event(self,event):
        kind=event['type']
        if kind=='frame':
            self.rgb.set_jpeg(event['jpeg'])
            if event.get('depth_jpeg'):self.depth.set_jpeg(event['depth_jpeg'])
            self.frames+=1;self.last_frame=time.monotonic()
            if self.state=='preview':self.stats.setText(f"预览 · RGB-D 时间差 {event.get('skew_ms',0):.1f} ms · 原始深度有效 {event.get('valid_depth_fraction',0):.1%}（全图）")
        elif kind=='ready':
            if self.state!='stopping':
                self.state='preview';self.clock.setText('预览');self.prompt.setText('调整站位，全身入画；准备好后点击开始录制')
                self.status.setText(event['message']);self.rgb.overlay='预览 · 尚未录制';self.rgb.update()
        elif kind=='countdown':
            n=event['seconds'];self.clock.setText(str(n));self.prompt.setText('秒后开始 · 自然站立，先保持静止')
            self.rgb.overlay=f'{n} 秒后开始录制';self.rgb.update()
            if self.sound.isChecked() and 0<n<=3:QtWidgets.QApplication.beep()
        elif kind=='recording':
            self.state='recording';self.clock.setText(f'{self.duration.value()}s');self.prompt.setText('自然站立，保持静止')
            self.status.setText('正在录制 · 按动作提示缓慢运动；可随时“结束并保存”。');self.started=time.monotonic()
            if self.sound.isChecked():QtWidgets.QApplication.beep()
        elif kind=='progress':
            remaining=event['remaining_s'];elapsed=event['elapsed_s'];action=event['action']
            self.clock.setText(f'{remaining:.0f}s');self.prompt.setText(action)
            self.rgb.overlay='● 录制中 · '+action;self.rgb.update()
            self.progress.setValue(round(1000*min(1,elapsed/self.duration.value())))
            self.progress.setFormat(f'{elapsed:.1f} / {self.duration.value()} 秒')
            drops=event['dropped_queue'];skew=event['rejected_skew']
            self.stats.setText(f"已写 {event['saved_frames']} 帧 · 实际保存 {event['actual_saved_fps']:.1f} FPS · 排队 {event['queue_pending']} · 写入丢帧 {drops} · 时间差剔除 {skew} · {event['bytes']/1024**2:.0f} MiB")
            self.stats.setStyleSheet('color:#ffc277;' if drops or skew else '')
            if self.last_action!=action:
                if self.last_action and self.sound.isChecked():QtWidgets.QApplication.beep()
                self.last_action=action
                plan=action_plan(self.duration.value(),self.preset.currentData(),self.notes.text())
                self.plan.setCurrentRow(next((i for i,p in enumerate(plan) if p['start_s']<=elapsed<p['end_s']),len(plan)-1))
        elif kind=='saving':
            self.state='saving';self.started=time.monotonic();self.clock.setText('保存');self.prompt.setText('相机已释放，正在保存剩余帧…')
            self.rgb.overlay='正在保存…';self.rgb.update();self.status.setText(event['message'])
        elif kind=='sequence_done':
            if Path(event['directory']).resolve()!=self.pending.resolve():raise ValueError('返回的录制路径不匹配')
            self.summary=event['summary']
        elif kind=='error':self.fail(event['message'])
        elif kind=='stage':self.status.setText(event['message'])
        if kind not in ('frame','progress'):self.add_log(json.dumps(event,ensure_ascii=False))
        self.refresh()

    def finished(self,code,status):
        self.read_output();self.state='idle';self.clock.setText('完成' if self.summary else '待机')
        if self.summary:
            m=self.summary;n=m['saved_frames'];fps=m['actual_saved_fps'];state=m['status']
            ok=state in ('completed','stopped') and n>0 and code==0
            label='已完成' if state=='completed' else ('提前结束，已保存' if state=='stopped' else '异常中断，保留部分数据')
            self.status.setText(f"{label}：{m['name']} · {n} 帧 · {fps:.1f} FPS · 写入丢帧 {m['dropped_queue']} · 时间差剔除 {m.get('rejected_skew',0)}。"+(' 可开始下一段。' if ok else ' 请查看日志与 sequence.json，不要当作完整序列。'))
            if m.get('failure') or m.get('error'):self.add_log(str(m.get('failure') or m.get('error')))
            self.stats.setText(f"已保存 {n} 帧 · {m['recorded_span_s']:.2f} 秒帧跨度 · {fps:.1f} FPS · {m['bytes']/1024**2:.1f} MiB")
            self.progress.setValue(1000 if state=='completed' else self.progress.value());self.progress.setFormat(label)
            self.rgb.overlay=label;self.rgb.update()
            if self.sound.isChecked():QtWidgets.QApplication.beep()
        elif self.stop_requested:
            self.status.setText('已取消并释放相机；开录前取消不创建数据目录。');self.rgb.overlay='已断开';self.rgb.update()
        else:self.fail('相机进程未完成录制，请查看日志并重新连接。')
        if self.pending and self.pending.is_dir():
            try:(self.pending/'worker.log').write_text('\n'.join(self.log_lines))
            except OSError as exc:self.add_log(str(exc))
        self.refresh_history();self.refresh()

    def process_error(self,error):
        if error==QtCore.QProcess.FailedToStart:
            self.state='idle';self.fail('相机进程启动失败：'+self.process.errorString());self.refresh()

    def stop(self):
        if self.process.state()==QtCore.QProcess.NotRunning:return
        if self.state in ('saving','stopping'):return
        self.stop_requested=True;self.state='stopping';self.started=time.monotonic()
        self.process.write(b'{"type":"stop"}\n');self.status.setText('正在结束录制，保存已采集帧并释放相机…');self.refresh()

    def shutdown(self):
        self.timer.stop();self.stop()
        if self.process.state()!=QtCore.QProcess.NotRunning and not self.process.waitForFinished(10000):
            self.process.terminate()
            if not self.process.waitForFinished(5000):self.process.kill();self.process.waitForFinished(1500)

    def tick(self):
        base=self.output_root
        while not base.exists():base=base.parent
        try:self.disk.setText(f'可用 {shutil.disk_usage(base).free/1024**3:.1f} GiB')
        except OSError:self.disk.setText('空间未知')
        elapsed=time.monotonic()-self.started
        if self.state=='connecting' and elapsed>20:self.stop();self.fail('相机连接超过 20 秒，请检查设备连接或占用')
        if self.state in ('preview','recording','countdown') and self.last_frame and time.monotonic()-self.last_frame>12:
            self.stop();self.fail('相机预览中断，正在停止并保留已采集数据')
        if self.state=='stopping' and elapsed>15:self.process.terminate()
        if self.state in ('saving','stopping') and elapsed>45:self.process.kill()

    def refresh(self):
        idle=not self.busy
        self.name.setEnabled(idle)
        for w in (self.preset,self.duration,self.delay,self.fps,self.notes):w.setEnabled(self.state in ('idle','preview'))
        self.connect_btn.setEnabled(idle and bool(self.name.text().strip()))
        self.record_btn.setEnabled(self.state=='preview' and self.frames>0)
        self.stop_btn.setEnabled(self.busy and self.state not in ('saving','stopping'))
        self.stop_btn.setText('结束并保存' if self.state=='recording' else ('取消倒计时' if self.state=='countdown' else '断开相机'))
        self.folder_btn.setEnabled(self.output_root.is_dir());self.history.setEnabled(idle)
        self.busyChanged.emit(self.busy)

    def refresh_history(self):
        self.history.clear()
        if not self.output_root.is_dir():return
        for directory in sorted(self.output_root.iterdir(),reverse=True,key=lambda p:p.name.split('__')[-1])[:100]:
            try:
                m=json.loads((directory/'sequence.json').read_text());state={'completed':'完成','stopped':'提前结束','error':'异常','interrupted':'中断','recording':'未正常结束'}.get(m['status'],m['status'])
                display=QtGui.QFontMetrics(self.history.font()).elidedText(m['name'],QtCore.Qt.ElideRight,195)
                when=datetime.fromisoformat(m['created_utc']).astimezone().strftime('%m-%d %H:%M')
                item=QtWidgets.QListWidgetItem(f"{display} · {state}\n{when} · {m.get('saved_frames',0)} 帧 / {m.get('actual_saved_fps',0):.1f} FPS")
                item.setData(QtCore.Qt.UserRole,str(directory));item.setToolTip(m['name']+'\n'+str(directory));self.history.addItem(item)
            except (OSError,ValueError,KeyError):continue

    def open_history(self,item):QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(item.data(QtCore.Qt.UserRole)))
    def open_folder(self):
        folder=self.pending if self.pending and self.pending.is_dir() else self.output_root
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(folder)))
    def add_log(self,text):
        self.log_lines.append(text);self.log_lines=self.log_lines[-300:];self.log.appendPlainText(text)
    def fail(self,message):self.status.setText('⚠ '+message);self.add_log(message)
