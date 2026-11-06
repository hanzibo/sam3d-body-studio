# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Model-free Gemini pose recorder. Camera SDK stays in its own subprocess."""
import argparse
import base64
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from studio_paths import environment, path
from recorder_session import RecordingSession
from PyQt5 import QtCore, QtGui, QtWidgets


def stamp():return datetime.now().strftime('%Y%m%d-%H%M%S-%f')


class Preview(QtWidgets.QWidget):
    def __init__(self, empty):
        super().__init__();self.image=QtGui.QImage();self.overlay='';self.empty=empty
        self.setMinimumSize(240,180)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,QtWidgets.QSizePolicy.Expanding)

    def paintEvent(self,event):
        painter=QtGui.QPainter(self);painter.fillRect(self.rect(),QtGui.QColor('#080d17'))
        if self.image.isNull():
            painter.setPen(QtGui.QColor('#a4b3c9'));painter.drawText(self.rect(),QtCore.Qt.AlignCenter,self.empty);return
        size=self.image.size().scaled(self.size(),QtCore.Qt.KeepAspectRatio)
        rectangle=QtCore.QRect((self.width()-size.width())//2,(self.height()-size.height())//2,size.width(),size.height())
        painter.drawImage(rectangle,self.image)
        if self.overlay:
            painter.fillRect(0,0,self.width(),54,QtGui.QColor(3,9,18,210));painter.setPen(QtGui.QColor('white'))
            painter.setFont(QtGui.QFont('Noto Sans CJK SC',17,QtGui.QFont.Bold))
            painter.drawText(self.rect().adjusted(10,0,-10,0),QtCore.Qt.AlignTop|QtCore.Qt.AlignHCenter,self.overlay)

    def set_jpeg(self,encoded):
        image=QtGui.QImage.fromData(base64.b64decode(encoded),'JPG')
        if not image.isNull():self.image=image;self.update()


class Recorder(QtWidgets.QMainWindow):
    def __init__(self, camera_python=None, capture_script=None):
        super().__init__();self.setWindowTitle('RGB-D 姿态采集 · SAM 3D Body Studio');self.resize(1180,830)
        self.camera_python=str(camera_python or path('camera_python'));self.capture_script=str(capture_script or ROOT/'rgbd_capture.py')
        self.session=None;self.state='idle';self.pending=None;self.pending_index=None
        self.event=None;self.buffer='';self.cancelled=False;self.auto_sequence=False
        self.frame_count=0;self.last_frame=0.;self.log_lines=[];self.worker_started=0.
        self.process=QtCore.QProcess(self);self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.read_output);self.process.finished.connect(self.finished)
        self.process.errorOccurred.connect(self.process_error)
        self.next_timer=QtCore.QTimer(self);self.next_timer.setSingleShot(True);self.next_timer.timeout.connect(self.capture)
        self.timer=QtCore.QTimer(self);self.timer.setInterval(500);self.timer.timeout.connect(self.tick);self.timer.start()
        central=QtWidgets.QWidget();self.setCentralWidget(central);layout=QtWidgets.QVBoxLayout(central)
        title=QtWidgets.QLabel('RGB-D 姿态采集');title.setObjectName('title');layout.addWidget(title)
        sub=QtWidgets.QLabel('仅采集相机，不加载人体模型或 SAM3。关闭 Studio 的相机预览后再连接；照片不会自动上传。')
        sub.setWordWrap(True);sub.setObjectName('muted');layout.addWidget(sub)
        row=QtWidgets.QHBoxLayout();row.addWidget(QtWidgets.QLabel('姿态数量'))
        self.count=QtWidgets.QSpinBox();self.count.setRange(1,100);self.count.setValue(8);row.addWidget(self.count)
        self.create_btn=QtWidgets.QPushButton('新建采集');self.create_btn.clicked.connect(self.new_session);row.addWidget(self.create_btn)
        self.open_btn=QtWidgets.QPushButton('继续已有采集');self.open_btn.clicked.connect(self.open_session);row.addWidget(self.open_btn)
        self.folder_btn=QtWidgets.QPushButton('打开数据目录');self.folder_btn.clicked.connect(self.open_folder);row.addWidget(self.folder_btn)
        row.addStretch();self.disk=QtWidgets.QLabel();row.addWidget(self.disk);layout.addLayout(row)
        self.location=QtWidgets.QLabel('先新建采集，或继续已有姿态集。默认保存到 outputs/rgbd-sets/。')
        self.location.setWordWrap(True);self.location.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse);layout.addWidget(self.location)
        split=QtWidgets.QSplitter();layout.addWidget(split,1)
        side=QtWidgets.QWidget();sl=QtWidgets.QVBoxLayout(side);sl.setContentsMargins(0,0,4,0)
        self.poses=QtWidgets.QListWidget();self.poses.setIconSize(QtCore.QSize(88,50));self.poses.currentRowChanged.connect(self.select_pose);sl.addWidget(self.poses,1)
        sl.addWidget(QtWidgets.QLabel('当前姿态说明（可修改）'))
        self.notes=QtWidgets.QPlainTextEdit();self.notes.setMaximumHeight(95);self.notes.setPlaceholderText('例如：正面、双臂前伸、距离约 2 米')
        self.notes.textChanged.connect(self.save_notes);sl.addWidget(self.notes)
        self.progress=QtWidgets.QProgressBar();self.progress.setFormat('%v / %m 个姿态已保存');sl.addWidget(self.progress)
        previews=QtWidgets.QWidget();pl=QtWidgets.QVBoxLayout(previews);pl.setContentsMargins(0,0,0,0)
        self.pose_title=QtWidgets.QLabel('等待创建采集');self.pose_title.setObjectName('poseTitle');self.pose_title.setWordWrap(True);pl.addWidget(self.pose_title)
        views=QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.rgb=Preview('彩色预览 · 点击“连接相机”');self.depth=Preview('深度预览 · 黑色代表无有效深度')
        views.addWidget(self.rgb);views.addWidget(self.depth);views.setSizes([420,210]);pl.addWidget(views,1)
        self.quality=QtWidgets.QLabel('预览不保存；只有倒计时完成并成功导出后，才计为已拍摄。')
        self.quality.setWordWrap(True);pl.addWidget(self.quality)
        split.addWidget(side);split.addWidget(previews);split.setSizes([285,870])
        row=QtWidgets.QHBoxLayout();self.connect_btn=QtWidgets.QPushButton('连接相机');self.connect_btn.clicked.connect(self.start_preview);row.addWidget(self.connect_btn)
        row.addWidget(QtWidgets.QLabel('倒计时'));self.delay=QtWidgets.QSpinBox();self.delay.setRange(0,30);self.delay.setValue(5);self.delay.setSuffix(' 秒');row.addWidget(self.delay)
        self.auto=QtWidgets.QCheckBox('自动依次拍摄');self.auto.toggled.connect(self.refresh);row.addWidget(self.auto)
        row.addWidget(QtWidgets.QLabel('换姿势间隔'));self.gap=QtWidgets.QSpinBox();self.gap.setRange(1,60);self.gap.setValue(8);self.gap.setSuffix(' 秒');row.addWidget(self.gap)
        self.sound=QtWidgets.QCheckBox('提示音');self.sound.setChecked(True);row.addWidget(self.sound)
        self.capture_btn=QtWidgets.QPushButton('倒计时拍摄');self.capture_btn.setObjectName('primary');self.capture_btn.clicked.connect(self.begin_capture);row.addWidget(self.capture_btn)
        self.stop_btn=QtWidgets.QPushButton('停止 / 断开');self.stop_btn.clicked.connect(self.stop);row.addWidget(self.stop_btn)
        layout.addLayout(row)
        self.status=QtWidgets.QLabel('建议全身入画、光线稳定；每次倒计时结束时保持静止。')
        self.status.setObjectName('status');self.status.setWordWrap(True);self.status.setMinimumHeight(50);layout.addWidget(self.status)
        self.show_log=QtWidgets.QCheckBox('显示采集日志');layout.addWidget(self.show_log)
        self.log=QtWidgets.QPlainTextEdit();self.log.setReadOnly(True);self.log.setMaximumHeight(120);self.log.hide()
        self.show_log.toggled.connect(self.log.setVisible);layout.addWidget(self.log)
        self.shortcut=QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+Return'),self);self.shortcut.activated.connect(self.begin_capture)
        self.escape=QtWidgets.QShortcut(QtGui.QKeySequence('Escape'),self);self.escape.activated.connect(self.stop)
        self.capture_btn.setToolTip('Ctrl+Enter 拍摄；Esc 取消并断开。重拍保留旧版本，不覆盖文件。')
        self.setStyleSheet(STYLE);self.refresh();self.tick()

    @property
    def busy(self):return self.process.state()!=QtCore.QProcess.NotRunning or self.state!='idle' or self.next_timer.isActive()

    def new_session(self):
        if self.busy:return
        try:self.use_session(RecordingSession.create(ROOT/'outputs/rgbd-sets'/('recording-'+stamp()),self.count.value()))
        except Exception as exc:self.fail(str(exc))

    def open_session(self):
        if self.busy:return
        folder=QtWidgets.QFileDialog.getExistingDirectory(self,'选择含 pose_set.json 的采集目录',str(ROOT/'outputs/rgbd-sets'))
        if folder:
            try:self.use_session(RecordingSession(folder))
            except Exception as exc:self.fail(str(exc))

    def use_session(self,session):
        if self.session:self.session.close()
        self.session=session;self.count.setValue(len(session.poses.poses))
        self.poses.blockSignals(True);self.poses.clear()
        for pose in session.poses.poses:self.poses.addItem(pose['name'])
        self.poses.blockSignals(False);self.refresh_list()
        index=next((i for i,p in enumerate(session.poses.poses) if not p.get('sample')),0)
        self.poses.setCurrentRow(index);self.select_pose(index)
        self.location.setText(str(session.directory));self.location.setToolTip(str(session.directory))
        self.status.setText('采集已就绪。点击“连接相机”预览，然后倒计时拍摄；每张姿态可修改说明。');self.refresh()

    def refresh_list(self):
        if not self.session:return
        for i,pose in enumerate(self.session.poses.poses):
            item=self.poses.item(i);item.setText(('✓ ' if pose.get('sample') else '○ ')+pose['name'])
            item.setToolTip(pose.get('notes',''))
            sample=self.session.poses.sample_path(pose)
            if sample:item.setIcon(QtGui.QIcon(str(sample/'rgb.png')))
        self.progress.setRange(0,len(self.session.poses.poses));self.progress.setValue(self.session.completed)

    def select_pose(self,index):
        if not self.session or index<0:return
        pose=self.session.poses.poses[index]
        self.notes.blockSignals(True);self.notes.setPlainText(pose.get('notes',''));self.notes.blockSignals(False)
        self.pose_title.setText(f"{pose['name']} / 共 {len(self.session.poses.poses)} 张 · "+pose.get('notes',''))
        if self.state=='idle':
            sample=self.session.poses.sample_path(pose)
            self.rgb.image=QtGui.QImage(str(sample/'rgb.png')) if sample else QtGui.QImage()
            self.depth.image=QtGui.QImage(str(sample/'depth_preview.png')) if sample else QtGui.QImage()
            self.rgb.overlay='已保存 · 点击连接相机可重拍' if sample else '';self.rgb.update();self.depth.update()
            if sample:
                try:self.show_quality(json.loads((sample/'camera.json').read_text()))
                except (ValueError,OSError):self.quality.setText('历史元数据无法读取；原始文件仍保留。')
            else:self.quality.setText('当前姿态尚未拍摄。')
        self.refresh()

    def save_notes(self):
        if not self.session or self.poses.currentRow()<0:return
        pose=self.session.poses.poses[self.poses.currentRow()];pose['notes']=self.notes.toPlainText()
        try:self.session.poses.save();self.refresh_list()
        except OSError as exc:self.fail('说明保存失败：'+str(exc))
        self.pose_title.setText(f"{pose['name']} · "+pose['notes'])

    def start_preview(self):
        if not self.session or self.process.state()!=QtCore.QProcess.NotRunning or self.state!='idle':return
        if not Path(self.camera_python).is_file():self.fail('相机 Python 环境不存在，请检查本机路径配置');return
        self.cancelled=False;self.event=None;self.buffer='';self.log_lines=[];self.frame_count=0;self.last_frame=0.
        self.pending_index=self.poses.currentRow()
        self.pending=self.session.directory/'samples'/f'pose-{self.pending_index+1:03d}'/stamp()
        env=QtCore.QProcessEnvironment()
        for key,value in environment('camera').items():env.insert(key,value)
        self.process.setProcessEnvironment(env);self.process.setWorkingDirectory(str(ROOT))
        self.state='connecting';self.worker_started=time.monotonic();self.rgb.overlay='正在连接相机…';self.rgb.update()
        self.status.setText('正在连接 Gemini 336L；若提示设备占用，请关闭其他窗口中的相机预览。')
        self.process.start(self.camera_python,['-u',self.capture_script,'--output',str(self.pending),'--preview','--depth-preview'])
        self.refresh()

    def begin_capture(self):
        if not self.capture_btn.isEnabled():return
        self.auto_sequence=self.auto.isChecked();self.capture()

    def capture(self):
        if self.state!='preview' or not self.session:return
        if time.monotonic()-self.last_frame>3:self.fail('预览已中断，请停止后重新连接');self.stop();return
        if shutil.disk_usage(self.session.directory).free<512*1024**2:
            self.fail('可用空间不足 512 MiB，已停止采集，请释放空间或换用其他数据盘');self.stop();return
        self.state='countdown';self.worker_started=time.monotonic();self.next_timer.stop()
        self.process.write((json.dumps(dict(type='capture',delay=self.delay.value()))+'\n').encode())
        self.status.setText('倒计时拍摄中，请保持全身入画；Esc 可取消。');self.refresh()

    def read_output(self):
        self.buffer+=bytes(self.process.readAllStandardOutput()).decode('utf-8',errors='replace')
        while '\n' in self.buffer:
            line,self.buffer=self.buffer.split('\n',1)
            if not line.startswith('@@RGBD '):self.add_log(line);continue
            try:self.handle_event(json.loads(line[7:]))
            except Exception as exc:self.add_log('结果读取失败：'+str(exc));self.fail(str(exc))

    def handle_event(self,event):
        kind=event['type']
        if kind=='frame':
            self.rgb.set_jpeg(event['jpeg'])
            if event.get('depth_jpeg'):self.depth.set_jpeg(event['depth_jpeg'])
            self.frame_count+=1;self.last_frame=time.monotonic()
            self.quality.setText(f"预览 RGB-D 时间差 {event.get('skew_ms',0):.1f} ms · 深度有效像素 {event.get('valid_depth_fraction',0):.1%}（整幅深度图，非人体区域）")
            self.refresh();return
        self.add_log(json.dumps(event,ensure_ascii=False))
        if kind=='ready' and not self.cancelled:
            self.state='preview';self.rgb.overlay='预览 · 请按姿态说明摆好姿势';self.rgb.update();self.status.setText(event['message'])
            if self.auto_sequence:
                self.next_timer.start(self.gap.value()*1000)
                self.rgb.overlay=f'请换姿势 · {self.gap.value()} 秒后开始倒计时';self.rgb.update()
                self.status.setText('已自动切换下一姿态。换姿势间隔结束后，将开始拍摄倒计时。')
        elif kind=='countdown':
            self.rgb.overlay=f"{event['seconds']} 秒后拍摄 · {self.session.poses.poses[self.pending_index]['name']}";self.rgb.update()
            if self.sound.isChecked() and 0<event['seconds']<=3:QtWidgets.QApplication.beep()
        elif kind=='stage':
            self.status.setText(event['message'])
            if '相机已释放' in event['message']:self.state='saving';self.rgb.overlay='正在保存 RGB-D…';self.rgb.update()
        elif kind=='done':self.event=event
        elif kind=='error':self.fail(event['message'])
        self.refresh()

    def add_log(self,line):
        self.log_lines.append(line);self.log_lines=self.log_lines[-500:];self.log.appendPlainText(line)
        self.log.document().setMaximumBlockCount(500)

    def finished(self,code,status):
        self.read_output();self.next_timer.stop();self.state='idle';self.rgb.overlay='';self.rgb.update()
        accepted=False
        if not self.cancelled and code==0 and self.event:
            try:
                output=Path(self.event['directory']).resolve()
                if output!=self.pending.resolve():raise ValueError('相机返回了错误的采集路径')
                meta=self.session.attach(self.pending_index,output);self.show_quality(meta);accepted=True
                self.refresh_list();self.select_pose(self.pending_index)
                self.status.setText(f'已保存 {self.session.completed}/{len(self.session.poses.poses)} 个姿态。可点击其他姿态继续；重拍会保留旧版本。')
                if self.sound.isChecked():QtWidgets.QApplication.beep()
            except Exception as exc:self.fail('采集未加入样本集：'+str(exc))
        elif self.cancelled:self.status.setText('已停止并释放相机。已完成照片保留；未完成采集不计入姿态集。')
        else:self.fail('采集未完成。请检查相机连接或日志，然后重新连接。')
        if self.pending and self.pending.is_dir():
            try:(self.pending/'worker.log').write_text('\n'.join(self.log_lines))
            except OSError as exc:self.add_log('日志保存失败：'+str(exc))
        if accepted and self.auto_sequence:
            # Capture exactly this and subsequent poses, including already recorded ones as new versions.
            index=self.pending_index+1
            if index<len(self.session.poses.poses):
                self.poses.setCurrentRow(index);QtCore.QTimer.singleShot(0,self.continue_sequence)
            else:
                self.auto_sequence=False;self.status.setText('本轮拍摄完成，相机已释放。可在 Studio 的 RGB-D 几何验证页打开该样本集。')
        elif not accepted:self.auto_sequence=False
        self.refresh()

    def continue_sequence(self):
        if self.auto_sequence:self.start_preview()

    def show_quality(self,meta):
        q=meta.get('capture_quality',{});ts=meta.get('timestamps',{})
        self.quality.setText(f"已保存 · RGB-D 时间差 {ts.get('skew_ms',0):.1f} ms · 配准深度有效比例 {q.get('registered_valid_fraction',0):.1%}（全图） · 原始 RGB / 深度、内外参和相机系点云齐全")

    def process_error(self,error):
        if error==QtCore.QProcess.FailedToStart:
            self.state='idle';self.auto_sequence=False;self.fail('无法启动相机进程：'+self.process.errorString());self.refresh()
        elif not self.cancelled:self.add_log('相机进程：'+self.process.errorString())

    def stop(self):
        self.next_timer.stop();self.auto_sequence=False;self.cancelled=True
        if self.process.state()!=QtCore.QProcess.NotRunning:
            self.state='stopping';self.status.setText('正在取消并释放相机…')
            self.process.terminate();pid=self.process.processId()
            QtCore.QTimer.singleShot(2500,lambda:self.process.kill() if self.process.processId()==pid and self.process.state()!=QtCore.QProcess.NotRunning else None)
        else:self.state='idle'
        self.refresh()

    def fail(self,message):
        self.status.setText('⚠ '+message);self.add_log(message)

    def tick(self):
        folder=self.session.directory if self.session else ROOT
        try:self.disk.setText(f'可用空间 {shutil.disk_usage(folder).free/1024**3:.1f} GiB')
        except OSError:self.disk.setText('无法读取磁盘空间')
        elapsed=time.monotonic()-self.worker_started
        if self.state=='connecting' and elapsed>20:self.stop();self.fail('相机连接超时，请检查其他程序是否占用设备')
        elif self.state in ('countdown','saving') and elapsed>self.delay.value()+90:self.stop();self.fail('拍摄或导出超时，原有已完成数据保留')
        elif self.state=='preview' and self.last_frame and time.monotonic()-self.last_frame>10:self.stop();self.fail('相机连续 10 秒未返回预览，请重新连接')

    def refresh(self):
        idle=not self.busy;available=self.session is not None
        for widget in (self.create_btn,self.open_btn,self.count):widget.setEnabled(idle)
        self.poses.setEnabled(self.state in ('idle','preview') and not self.auto_sequence)
        self.notes.setEnabled(available and self.state in ('idle','preview') and not self.auto_sequence)
        self.connect_btn.setEnabled(available and idle);self.folder_btn.setEnabled(available)
        self.capture_btn.setEnabled(available and self.state=='preview' and self.frame_count>0 and not self.next_timer.isActive())
        pose=self.session.poses.poses[self.poses.currentRow()] if available and self.poses.currentRow()>=0 else {}
        self.capture_btn.setText('开始依次拍摄' if self.auto.isChecked() else ('倒计时重拍（保留旧版）' if pose.get('sample') else '倒计时拍摄'))
        self.stop_btn.setEnabled(self.busy)
        for widget in (self.auto,self.delay,self.gap):widget.setEnabled(self.state in ('idle','preview') and not self.auto_sequence)
        self.gap.setEnabled(self.gap.isEnabled() and self.auto.isChecked())
        # The live camera process owns one destination. Switching poses needs a reconnect.
        if self.state=='preview':self.poses.setEnabled(False)

    def open_folder(self):
        if self.session:QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(self.session.directory)))

    def closeEvent(self,event):
        self.timer.stop();self.stop()
        if self.process.state()!=QtCore.QProcess.NotRunning:
            if not self.process.waitForFinished(3000):self.process.kill();self.process.waitForFinished(1500)
        if self.session:self.session.close()
        event.accept()


STYLE='''
QWidget { background:#101827; color:#e5edf8; font-family:"Noto Sans CJK SC","DejaVu Sans"; font-size:13px; }
QLabel#title { font-size:25px; font-weight:bold; }
QLabel#muted { color:#99abc3; }
QLabel#poseTitle { color:#83bdff; font-size:16px; padding:5px; }
QLabel#status { background:#172941; border:1px solid #29466b; border-radius:6px; padding:9px; }
QPushButton { background:#20324b; border:1px solid #3a5271; border-radius:6px; padding:9px 13px; }
QPushButton:hover { background:#2c476a; }
QPushButton#primary { background:#216ddd; font-weight:bold; }
QPushButton:disabled { background:#182336; color:#67758d; border-color:#28364b; }
QSpinBox,QPlainTextEdit,QListWidget { background:#0b1220; border:1px solid #34465f; border-radius:5px; padding:5px; }
QListWidget::item { padding:8px; }
QListWidget::item:selected { background:#244f80; }
QCheckBox { spacing:6px; }
QProgressBar { background:#182b42; border:0; border-radius:4px; text-align:center; }
QProgressBar::chunk { background:#247dcc; }
QSplitter::handle { background:#263b54; height:5px; width:5px; }
QToolTip { background:#20324b; color:white; border:1px solid #3a5271; }
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--session',type=Path)
    args=parser.parse_args()
    app=QtWidgets.QApplication(sys.argv[:1]);app.setStyle('Fusion')
    w=Recorder()
    if args.session:
        try:w.use_session(RecordingSession(args.session))
        except Exception as exc:w.fail(str(exc))
    w.show();sys.exit(app.exec_())


if __name__=='__main__':main()
