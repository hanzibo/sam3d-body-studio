# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Independent offline video workspace: segmented tracks, isolated GPU stages."""
from dataclasses import replace
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import cv2
from PyQt5 import QtCore,QtGui,QtWidgets
from rgbd_validation import MaskCanvas
from profile_panel import ProfilePanel
from launch_config import LaunchConfig
from studio_paths import path,environment
from video_project import ROOT,load_project,frame_path,add_track,put_segment,save_json,mask_at,require_masks


class VideoReconstruction(QtWidgets.QWidget):
    def __init__(self,host):
        super().__init__();self.host=host;self.directory=None;self.project=None;self.mode=None;self.buffer='';self.done=False;self.cancelled=False;self.pid=0
        self.output=None;self.run=None;self.pending_run=None
        self.process=QtCore.QProcess(self);self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.read);self.process.finished.connect(self.finished);self.process.errorOccurred.connect(self.error)
        self.kill_timer=QtCore.QTimer(self);self.kill_timer.setSingleShot(True);self.kill_timer.timeout.connect(self.kill)
        self.play_timer=QtCore.QTimer(self);self.play_timer.timeout.connect(self.advance)
        layout=QtWidgets.QVBoxLayout(self)
        note=QtWidgets.QLabel('离线视频 · 人物 ID → 分段点提示 → SAM3 掩膜 → 逐帧人体重建 → 合成视频。SAM3 / 人体模型分阶段独占显存。')
        note.setWordWrap(True);layout.addWidget(note)
        row=QtWidgets.QHBoxLayout();layout.addLayout(row)
        self.import_btn=self.button(row,'导入视频…',self.import_video);self.open_btn=self.button(row,'打开已有任务…',self.open_project)
        self.folder_btn=self.button(row,'打开任务目录',lambda:self.open_path(self.directory));self.cancel_btn=self.button(row,'取消任务',self.cancel)
        self.info=QtWidgets.QLabel('请选择视频；原文件保留，解码帧与各次结果统一保存在 outputs/video。');self.info.setWordWrap(True);layout.addWidget(self.info)
        split=QtWidgets.QSplitter();layout.addWidget(split,1)
        left=QtWidgets.QWidget();ll=QtWidgets.QVBoxLayout(left);ll.setContentsMargins(0,0,0,0);split.addWidget(left)
        self.canvas=MaskCanvas();self.canvas.empty_text='导入视频后，在人物出现帧添加前景 / 背景提示点';ll.addWidget(self.canvas,1)
        timeline=QtWidgets.QHBoxLayout();ll.addLayout(timeline)
        self.play_btn=self.button(timeline,'播放 / 暂停',self.toggle_play)
        self.slider=QtWidgets.QSlider(QtCore.Qt.Horizontal);self.slider.valueChanged.connect(self.seek);timeline.addWidget(self.slider,1)
        self.frame=QtWidgets.QSpinBox();self.frame.valueChanged.connect(self.slider.setValue);timeline.addWidget(self.frame)
        self.time=QtWidgets.QLabel('0 / 0');timeline.addWidget(self.time)
        self.result_preview=QtWidgets.QCheckBox('预览所选记录的渲染帧');self.result_preview.toggled.connect(lambda:self.seek(self.slider.value()));ll.addWidget(self.result_preview)
        panel=QtWidgets.QWidget();pl=QtWidgets.QVBoxLayout(panel);split.addWidget(panel);split.setStretchFactor(0,3);split.setStretchFactor(1,2)
        row=QtWidgets.QHBoxLayout();pl.addLayout(row);self.people=QtWidgets.QComboBox();self.people.currentIndexChanged.connect(self.person_changed);row.addWidget(self.people,1)
        self.add_btn=self.button(row,'新增人物',self.new_person);self.remove_btn=self.button(row,'移除人物',self.remove_person)
        self.segments=QtWidgets.QListWidget();self.segments.currentRowChanged.connect(self.segment_selected);pl.addWidget(self.segments)
        row=QtWidgets.QHBoxLayout();pl.addLayout(row);row.addWidget(QtWidgets.QLabel('当前帧作为起点，终点帧：'))
        self.end=QtWidgets.QSpinBox();row.addWidget(self.end)
        row=QtWidgets.QHBoxLayout();pl.addLayout(row)
        self.fg=self.button(row,'前景 ＋',lambda:self.prompt_mode(1));self.bg=self.button(row,'背景 −',lambda:self.prompt_mode(0))
        self.undo=self.button(row,'撤销点',self.undo_point);self.clear=self.button(row,'清空点',self.clear_points)
        self.save_btn=QtWidgets.QPushButton('保存当前帧提示 / 新建或修正分段');self.save_btn.clicked.connect(self.save_segment);pl.addWidget(self.save_btn)
        self.remove_segment_btn=QtWidgets.QPushButton('移除选中分段');self.remove_segment_btn.clicked.connect(self.remove_segment);pl.addWidget(self.remove_segment_btn)
        help_label=QtWidgets.QLabel('左键添加前景，右键添加背景。中途入场：到出现帧新增人物；遮挡后失跟：到失跟帧重新点选并保存分段。每人至少一个前景点；保存前请勿切换帧。分段之外不预测该人物。')
        help_label.setWordWrap(True);pl.addWidget(help_label)
        self.segment_btn=QtWidgets.QPushButton('① SAM3 分割全部人物 / 未完成分段');self.segment_btn.clicked.connect(lambda:self.start_stage('segment'));pl.addWidget(self.segment_btn)
        row=QtWidgets.QHBoxLayout();pl.addLayout(row);row.addWidget(QtWidgets.QLabel('掩膜膨胀＋ / 腐蚀−（像素）'))
        self.radius=QtWidgets.QSpinBox();self.radius.setRange(-40,40);row.addWidget(self.radius)
        self.reconstruct_btn=QtWidgets.QPushButton('② 应用本页方案、预热并逐帧重建');self.reconstruct_btn.clicked.connect(lambda:self.start_stage('reconstruct'));pl.addWidget(self.reconstruct_btn)
        self.runs=QtWidgets.QComboBox();self.runs.currentIndexChanged.connect(self.select_run);pl.addWidget(self.runs)
        row=QtWidgets.QHBoxLayout();pl.addLayout(row);self.layers=QtWidgets.QComboBox()
        for title,value in [('网格叠加','mesh'),('骨架叠加','skeleton'),('网格＋骨架','both')]:self.layers.addItem(title,value)
        self.layers.currentIndexChanged.connect(lambda:self.seek(self.slider.value()));row.addWidget(self.layers)
        self.render_btn=self.button(row,'③ 合成视频',lambda:self.start_stage('render'))
        self.video_btn=QtWidgets.QPushButton('打开导出视频');self.video_btn.clicked.connect(self.open_video);pl.addWidget(self.video_btn);pl.addStretch()
        self.profile=ProfilePanel(replace(LaunchConfig(),warmup_runs=3,warmup_layout='sample'))
        settings=QtWidgets.QGroupBox('本视频独立启动方案（点击展开；不修改第一页配置）');settings.setCheckable(True);settings.setChecked(False)
        sl=QtWidgets.QVBoxLayout(settings);sl.addWidget(self.profile);self.profile.hide();settings.toggled.connect(self.profile.setVisible);layout.addWidget(settings)
        self.progress=QtWidgets.QProgressBar();layout.addWidget(self.progress)
        self.status=QtWidgets.QLabel('准备就绪。首次分割前请释放第一页与 MHR 实验室中的模型。');self.status.setWordWrap(True);layout.addWidget(self.status)
        self.log=QtWidgets.QPlainTextEdit();self.log.setReadOnly(True);self.log.setMaximumHeight(100);self.log.document().setMaximumBlockCount(1000);layout.addWidget(self.log)
        self.guard_timer=QtCore.QTimer(self);self.guard_timer.setInterval(500);self.guard_timer.timeout.connect(self.refresh);self.guard_timer.start();self.refresh()

    @property
    def active(self):return self.mode is not None or self.process.state()!=QtCore.QProcess.NotRunning

    @staticmethod
    def button(row,text,slot):
        button=QtWidgets.QPushButton(text);button.clicked.connect(slot);row.addWidget(button);return button

    def open_path(self,value):
        if value:QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(value)))

    def other_busy(self):
        h=self.host
        if not hasattr(h,'live') or not hasattr(h,'image_seg'):return '界面初始化中'
        if h.model_state!='off' or h.process.state()!=QtCore.QProcess.NotRunning:return '请先在第一页释放人体模型'
        if h.inference_busy or h.camera.active or h.camera_state not in ('off','photo') or h.live.running:return '请先停止第一页相机 / 推理 / 实时流'
        if h.image_seg.active:return '请先等待第一页分割 / 采集退出'
        v=h.rgbd
        if v.mode is not None or v.expected is not None or v.segmentation_pending is not None or v.pending_camera_action is not None or v.batch.active:return '请先停止 RGB-D 页面任务'
        lab=h.lab.client
        if lab.process.state()!=QtCore.QProcess.NotRunning:return '请先释放 MHR 实验室模型'
        if hasattr(h,'data_manager') and h.data_manager.deleting:return '数据管理正在删除，请等待完成'
        if any(p.state()!=QtCore.QProcess.NotRunning for p in h.render_processes.values()):return '请等待已有对照图渲染结束'
        return ''

    def refresh(self):
        idle=not self.active;loaded=self.project is not None
        for control in (self.import_btn,self.open_btn):control.setEnabled(idle)
        for control in (self.people,self.add_btn,self.remove_btn,self.segments,self.end,self.fg,self.bg,self.undo,self.clear,self.save_btn,self.remove_segment_btn,self.radius,self.profile):control.setEnabled(idle and loaded)
        self.canvas.setEnabled(idle and loaded and not self.result_preview.isChecked())
        self.slider.setEnabled(idle and loaded);self.frame.setEnabled(idle and loaded);self.play_btn.setEnabled(idle and loaded)
        self.result_preview.setEnabled(idle and self.run is not None);self.runs.setEnabled(idle);self.layers.setEnabled(idle)
        reason=self.other_busy() if idle else '本页任务正在运行'
        self.segment_btn.setEnabled(idle and loaded and any(t['segments'] for t in self.project['tracks']) and not reason)
        self.reconstruct_btn.setEnabled(idle and loaded and not reason)
        self.render_btn.setEnabled(idle and self.run is not None and not reason)
        for button in (self.segment_btn,self.reconstruct_btn,self.render_btn):button.setToolTip(reason)
        self.folder_btn.setEnabled(loaded);self.cancel_btn.setEnabled(self.active and not self.cancelled)
        self.video_btn.setEnabled(self.video_file() is not None)

    def import_video(self):
        if self.active:return
        source,_=QtWidgets.QFileDialog.getOpenFileName(self,'导入视频','','视频 (*.mp4 *.mov *.mkv *.avi);;全部 (*)')
        if not source:return
        folder=ROOT/'outputs/video'/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        self.directory=folder;self.project=None;self.run=None
        self.launch('prepare',['--source',source])

    def open_project(self):
        if self.active:return
        folder=QtWidgets.QFileDialog.getExistingDirectory(self,'打开含 project.json 的任务目录',str(ROOT/'outputs/video'))
        if folder:
            try:self.load(folder)
            except Exception as exc:self.status.setText(str(exc))

    def load(self,directory):
        self.directory,self.project=load_project(directory);p=self.project
        self.info.setText(f'{self.directory.name} · {p["width"]}×{p["height"]} · {p["frame_count"]} 帧 · {p["duration"]:.3f} 秒 · '+('有音轨' if p['has_audio'] else '无音轨'))
        self.slider.setRange(0,p['frame_count']-1);self.frame.setRange(0,p['frame_count']-1);self.end.setRange(0,p['frame_count']-1);self.end.setValue(p['frame_count']-1)
        self.reload_people();self.reload_runs();self.seek(0);self.refresh()

    def track(self):
        return next((t for t in self.project['tracks'] if t['id']==self.people.currentData()),None) if self.project else None

    def persist(self):save_json(self.directory/'project.json',self.project)

    def reload_people(self):
        selected=self.people.currentData();self.people.blockSignals(True);self.people.clear()
        for t in self.project['tracks']:self.people.addItem(f'ID {t["id"]} · {t["name"]}',t['id'])
        self.people.setCurrentIndex(max(0,self.people.findData(selected)));self.people.blockSignals(False);self.person_changed()

    def new_person(self):
        name,ok=QtWidgets.QInputDialog.getText(self,'新增人物','人物名称（入场后也可从当前帧开始）：')
        if ok:
            t=add_track(self.project,name);self.persist();self.reload_people();self.people.setCurrentIndex(self.people.findData(t['id']));self.clear_points()

    def remove_person(self):
        t=self.track()
        if t and QtWidgets.QMessageBox.question(self,'移除人物','移除此人物的提示？已有文件保留在任务目录中。')==QtWidgets.QMessageBox.Yes:
            self.project['tracks'].remove(t);self.persist();self.reload_people()

    def person_changed(self,*_):
        self.segments.blockSignals(True);self.segments.clear();t=self.track()
        if t:
            for s in t['segments']:self.segments.addItem(f'帧 {s["start"]} → {s["end"]} · {len(s["points"])} 点')
        self.segments.blockSignals(False);self.seek(self.slider.value())

    def segment_selected(self,row):
        t=self.track()
        if t and 0<=row<len(t['segments']):
            s=t['segments'][row];self.slider.setValue(s['start']);self.seek(s['start']);self.end.setValue(s['end'])

    def seek(self,index):
        if not self.project:return
        self.frame.blockSignals(True);self.frame.setValue(index);self.frame.blockSignals(False)
        image=frame_path(self.directory,index);t=self.track();mask=None;points=[]
        if t:
            file=mask_at(self.directory,self.project,t,index)
            if file:mask=cv2.imread(str(file),0)
            s=next((s for s in t['segments'] if s['start']==index),None)
            if s:points=list(s['points'])
        if self.result_preview.isChecked() and self.run:
            rendered=self.run/('render-'+self.layers.currentData())/f'{index:06d}.png'
            if rendered.exists() and self.current_render():image=rendered;mask=None;points=[]
        self.canvas.load(image,mask);self.canvas.prompts=points;self.canvas.prompt_mode=True;self.canvas.update()
        self.time.setText(f'{index}/{self.project["frame_count"]-1} · {self.project["pts"][index]:.3f}s')

    def prompt_mode(self,label):self.canvas.prompt_label=label;self.canvas.prompt_mode=True
    def undo_point(self):
        if self.canvas.prompts:self.canvas.prompts.pop();self.canvas.update()
    def clear_points(self):self.canvas.prompts=[];self.canvas.update()

    def save_segment(self):
        t=self.track()
        if not t:self.status.setText('请先新增人物');return
        try:
            put_segment(self.project,t['id'],self.slider.value(),self.end.value(),[list(p) for p in self.canvas.prompts])
            self.persist();self.person_changed();self.status.setText('分段提示已保存；修正后需要重新分割，再重建。')
        except Exception as exc:self.status.setText(str(exc))

    def remove_segment(self):
        t=self.track();row=self.segments.currentRow()
        if t and 0<=row<len(t['segments']):t['segments'].pop(row);self.persist();self.person_changed()

    def toggle_play(self):
        if self.play_timer.isActive():self.play_timer.stop()
        elif self.project:
            if self.slider.value()==self.slider.maximum():self.slider.setValue(0)
            self.schedule_play()
    def schedule_play(self):
        index=self.slider.value();p=self.project
        if index>=p['frame_count']-1:self.play_timer.stop();return
        self.play_timer.start(max(1,int((p['pts'][index+1]-p['pts'][index])*1000)))
    def advance(self):
        if not self.project or self.active:self.play_timer.stop();return
        self.slider.setValue(min(self.slider.value()+1,self.slider.maximum()));self.schedule_play()

    def reload_runs(self):
        self.runs.blockSignals(True);self.runs.clear()
        for file in sorted((self.directory/'runs').glob('*/run.json'),reverse=True):
            try:
                r=json.loads(file.read_text())
                if r['status']=='complete':
                    c=LaunchConfig.from_dict(r['config']);self.runs.addItem(f'{c.backend}/{c.image_size}/{c.config_id} · 成功 {r["successful"]} / 失败 {r["failed"]}',str(file.parent))
            except (OSError,ValueError,KeyError):continue
        self.runs.blockSignals(False);self.select_run()
    def select_run(self,*_):
        value=self.runs.currentData();self.run=Path(value) if value else None
        self.runs.setToolTip((self.run/'config.json').read_text() if self.run and (self.run/'config.json').is_file() else '')
        self.seek(self.slider.value());self.refresh()
    def video_file(self):
        file=self.run/f'overlay-{self.layers.currentData()}.mp4' if self.run else None
        return file if file and file.is_file() and self.current_render() else None
    def current_render(self):
        if not self.run:return False
        try:
            summary=json.loads((self.run/'run.json').read_text())
            rendered=json.loads((self.run/('render-'+self.layers.currentData())/'complete.json').read_text())
            return summary['status']=='complete' and summary.get('prediction_revision')==rendered.get('prediction_revision')
        except (ValueError,OSError,KeyError):return False
    def open_video(self):self.open_path(self.video_file())

    def start_stage(self,mode):
        if self.active:return
        try:
            reason=self.other_busy()
            if reason:raise ValueError(reason)
            if self.project is None:raise ValueError('请先导入视频')
            args=[]
            if mode=='reconstruct':
                require_masks(self.directory,self.project);config=self.profile.config()
                if config.one_euro_enabled or config.smpl_smoothing:raise ValueError('视频首版请关闭 One Euro / MLP；人物时序状态暂不混用')
                file=self.directory/'requested-config.json';config.save(file);args=['--config',str(file),'--radius',str(self.radius.value())]
            elif mode=='render':
                if self.run is None:raise ValueError('请先选择完成的重建记录')
                args=['--run',str(self.run),'--layers',self.layers.currentData()]
            self.launch(mode,args)
        except Exception as exc:self.status.setText(str(exc))

    def launch(self,mode,args):
        self.play_timer.stop();self.kill_timer.stop();self.mode=mode;self.done=False;self.cancelled=False;self.buffer='';self.pid=0;self.log.clear();self.progress.setValue(0)
        env=QtCore.QProcessEnvironment()
        for k,v in environment('standard').items():env.insert(k,v)
        self.process.setProcessEnvironment(env);self.process.setWorkingDirectory(str(ROOT))
        executable=path('sam3_python' if mode=='segment' else 'standard_python')
        self.process.start(str(executable),['-u',str(ROOT/'video_pipeline.py'),'--mode',mode,'--directory',str(self.directory),*args])
        self.status.setText('任务启动中；可以取消，已完成分段和预测会保留。');self.refresh();self.host.update_camera_controls()

    def read(self):
        self.pid=int(self.process.processId()) or self.pid
        self.buffer+=bytes(self.process.readAllStandardOutput()).decode(errors='replace')
        while '\n' in self.buffer:
            line,self.buffer=self.buffer.split('\n',1);self.log.appendPlainText(line)
            if not line.startswith('@@VIDEO '):continue
            try:event=json.loads(line[8:])
            except ValueError:continue
            kind=event['type']
            if kind in ('stage','error','progress'):self.status.setText(event.get('message',''))
            if kind=='progress':self.progress.setMaximum(max(1,event.get('total',1)));self.progress.setValue(event.get('current',0))
            if kind=='done':self.done=True
            if kind=='rendered':self.output=Path(event['video'])
            if kind=='reconstructed':self.pending_run=Path(event['run'])
        self.refresh()

    def error(self,error):
        if error==QtCore.QProcess.FailedToStart:
            self.mode=None;self.kill_timer.stop();self.status.setText('无法启动环境：'+self.process.errorString());self.refresh();self.host.update_camera_controls()

    def finished(self,code,*_):
        self.read();self.kill_timer.stop();previous=self.mode;self.mode=None;self.pid=0
        if not self.cancelled and code==0 and self.done:
            self.status.setText('阶段完成；分割进程已退出，可以加载人体模型。' if previous=='segment' else '阶段完成。')
            if previous=='prepare':self.load(self.directory)
            else:
                self.reload_runs()
                if self.pending_run:self.runs.setCurrentIndex(max(0,self.runs.findData(str(self.pending_run))));self.pending_run=None
                self.seek(self.slider.value())
            self.progress.setValue(self.progress.maximum())
        elif self.cancelled:self.status.setText('任务已取消，进程已退出；再次启动可复用完整分段 / 成功预测。')
        else:self.status.setText('任务失败，详细原因请查看下方日志；完整结果保留。')
        self.refresh();self.host.update_camera_controls()

    def cancel(self):
        if not self.active:return
        self.cancelled=True;self.process.terminate();self.kill_timer.start(3000);self.refresh()
    def kill(self):
        if self.process.state()==QtCore.QProcess.NotRunning:return
        pid=self.pid or int(self.process.processId())
        try:
            if pid and os.getpgid(pid)==pid:os.killpg(pid,signal.SIGKILL)
            else:self.process.kill()
        except ProcessLookupError:pass
    def shutdown(self):
        self.guard_timer.stop();self.play_timer.stop();self.kill_timer.stop()
        if self.active:
            self.cancel();self.kill_timer.stop()
            if not self.process.waitForFinished(3000):self.kill();self.process.waitForFinished(3000)
