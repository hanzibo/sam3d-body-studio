# SPDX-License-Identifier: GPL-3.0-only
"""Model-free synchronized mesh/point-cloud playback; bounded asynchronous disk IO."""
import bisect,json,time
from pathlib import Path
import numpy as np
from PyQt5 import QtCore,QtGui,QtWidgets
from rgbd_validation import CameraScene,to_view
from studio.rgbd.rgbd_comparison_data import METHODS,LABELS,ALL_METHODS,METHOD_LABELS,comparison_methods,load_npz
from studio.video.video_project import file_hash

ROOT=Path(__file__).resolve().parents[1]

class ReplayReader(QtCore.QObject):
    ready=QtCore.pyqtSignal(int,int,object)
    @QtCore.pyqtSlot(int,int,str)
    def read(self,generation,index,folder):
        try:
            root=Path(folder);file=root/'frames'/f'{index:06d}.npz';info=json.loads(file.with_suffix('.json').read_text())
            if file_hash(file)!=info['artifact_sha256']:raise ValueError('人体结果校验失败')
            result=dict(mesh=load_npz(file),cloud=load_npz(root/'report/replay'/f'{index:06d}.npz'),info=info,image=QtGui.QImage(str(root/'report/replay'/f'{index:06d}.jpg')))
        except Exception as e:result=dict(error=str(e))
        self.ready.emit(generation,index,result)

class RgbdComparison(QtWidgets.QWidget):
    request=QtCore.pyqtSignal(int,int,str)
    def __init__(self,host=None):
        super().__init__(host);self.directory=None;self.source=None;self.faces=None;self.generation=0;self.busy=False;self.pending=None;self.driver=0;self.loaded_index=-1;self.playing=False;self.case_paths=[]
        self.thread=QtCore.QThread(self);self.reader=ReplayReader();self.reader.moveToThread(self.thread);self.request.connect(self.reader.read);self.reader.ready.connect(self.loaded);self.thread.finished.connect(self.reader.deleteLater);self.thread.start()
        self.timer=QtCore.QTimer(self);self.timer.setInterval(20);self.timer.timeout.connect(self.tick)
        self.sync=QtCore.QTimer(self);self.sync.setInterval(40);self.sync.timeout.connect(self.sync_views)
        layout=QtWidgets.QVBoxLayout(self)
        title=QtWidgets.QLabel('多方案 RGB-D 连续对照 · 真实相机坐标 / 米制 / 无额外对齐');title.setWordWrap(True);layout.addWidget(title)
        self.note=QtWidgets.QLabel('回放不加载人体模型或SAM3。');self.note.setWordWrap(True);layout.addWidget(self.note)
        row=QtWidgets.QHBoxLayout();layout.addLayout(row)
        self.cases=QtWidgets.QComboBox();self.cases.currentIndexChanged.connect(self.case_changed);row.addWidget(self.cases,1)
        b=QtWidgets.QPushButton('扫描已有对照');b.clicked.connect(self.scan);row.addWidget(b)
        b=QtWidgets.QPushButton('打开对照目录…');b.clicked.connect(self.choose);row.addWidget(b)
        self.report_btn=QtWidgets.QPushButton('量化报告 / 视频');self.report_btn.clicked.connect(self.open_report);row.addWidget(self.report_btn)
        self.split=QtWidgets.QSplitter(QtCore.Qt.Horizontal);layout.addWidget(self.split,1);self.scenes=[];self.stats=[];self.pages=[]
        for i,label in enumerate(METHOD_LABELS.values()):
            page=QtWidgets.QWidget();col=QtWidgets.QVBoxLayout(page);col.setContentsMargins(2,2,2,2);col.addWidget(QtWidgets.QLabel(label))
            scene=CameraScene();scene.setMinimumSize(230,250);scene.installEventFilter(self);scene._mesh_data.clear();self.scenes.append(scene);col.addWidget(scene,1)
            stats=QtWidgets.QLabel('等待载入');stats.setWordWrap(True);self.stats.append(stats);col.addWidget(stats);self.split.addWidget(page);self.pages.append(page)
        row=QtWidgets.QHBoxLayout();layout.addLayout(row)
        self.play_btn=QtWidgets.QPushButton('播放');self.play_btn.clicked.connect(self.toggle);row.addWidget(self.play_btn)
        for label,step in [('上一帧',-1),('下一帧',1)]:
            b=QtWidgets.QPushButton(label);b.clicked.connect(lambda _,delta=step:self.step(delta));row.addWidget(b)
        self.slider=QtWidgets.QSlider(QtCore.Qt.Horizontal);self.slider.valueChanged.connect(self.seek);row.addWidget(self.slider,1)
        self.index=QtWidgets.QSpinBox();self.index.valueChanged.connect(self.slider.setValue);row.addWidget(self.index)
        self.time_label=QtWidgets.QLabel('—');row.addWidget(self.time_label)
        row=QtWidgets.QHBoxLayout();layout.addLayout(row)
        self.loop=QtWidgets.QCheckBox('四组顺序循环');self.loop.setChecked(True);row.addWidget(self.loop)
        self.person=QtWidgets.QCheckBox('仅参考人体点云');self.person.setChecked(True);self.person.toggled.connect(self.cloud_mode);row.addWidget(self.person)
        self.cloud_check=QtWidgets.QCheckBox('点云');self.cloud_check.setChecked(True);self.cloud_check.toggled.connect(lambda yes:[s.cloud.setVisible(yes) for s in self.scenes]);row.addWidget(self.cloud_check)
        self.mesh_check=QtWidgets.QCheckBox('人体网格');self.mesh_check.setChecked(True);self.mesh_check.toggled.connect(self.mesh_visibility);row.addWidget(self.mesh_check)
        row.addWidget(QtWidgets.QLabel('网格透明度'))
        self.alpha=QtWidgets.QSlider(QtCore.Qt.Horizontal);self.alpha.setRange(5,100);self.alpha.setValue(55);self.alpha.valueChanged.connect(lambda val:[s.set_opacity(val) for s in self.scenes]);row.addWidget(self.alpha)
        for label,angle in [('相机正面',-90),('侧面',0),('斜视',-45)]:
            b=QtWidgets.QPushButton(label);b.clicked.connect(lambda _,a=angle:self.view(a));row.addWidget(b)
        self.rgb=QtWidgets.QLabel();self.rgb.setMinimumHeight(110);self.rgb.setMaximumHeight(170);self.rgb.setAlignment(QtCore.Qt.AlignCenter);layout.addWidget(self.rgb)
        self.status=QtWidgets.QLabel('请扫描或打开多方案对照目录。播放保持源时间，较慢设备会跳过显示帧；暂停可逐帧检查。');self.status.setWordWrap(True);layout.addWidget(self.status)
        self.all_scenes=list(self.scenes);self.all_stats=list(self.stats);self.scenes=self.all_scenes[:3];self.stats=self.all_stats[:3];self.pages[3].hide()
        self.rows={};self.methods=METHODS;self.scan()
    def scan(self):
        old=self.directory;paths=sorted((ROOT/'outputs/depth-fusion').glob('tracking-comparison-*/*/report/complete.json'),key=lambda p:(str(p.parents[2]), {'arms':0,'cross':1,'turn':2,'recover':3}.get(p.parents[1].name,4)))
        self.cases.blockSignals(True);self.cases.clear();self.case_paths=[]
        for f in paths:
            directory=f.parents[1]
            try:name=json.loads((directory/'inputs.json').read_text())['name']
            except Exception:continue
            self.case_paths.append(directory);self.cases.addItem(name+' · '+directory.parent.name)
        self.cases.blockSignals(False)
        if old in self.case_paths:self.cases.setCurrentIndex(self.case_paths.index(old))
        elif self.case_paths:self.cases.setCurrentIndex(0);self.case_changed(0)
    def choose(self):
        folder=QtWidgets.QFileDialog.getExistingDirectory(self,'选择含 manifest.json 的单组对照目录',str(ROOT/'outputs/depth-fusion'))
        if not folder:return
        folder=Path(folder)
        if folder.name=='report':folder=folder.parent
        if folder not in self.case_paths:self.case_paths.append(folder);self.cases.addItem(folder.name)
        self.cases.setCurrentIndex(self.case_paths.index(folder));self.case_changed(self.cases.currentIndex())
    def case_changed(self,index):
        if not 0<=index<len(self.case_paths):return
        self.pause()
        try:
            directory=self.case_paths[index]
            if not (directory/'report/complete.json').exists():raise ValueError('对照导出尚未完成')
            self.methods=comparison_methods(directory);self.scenes=self.all_scenes[:len(self.methods)];self.stats=self.all_stats[:len(self.methods)];self.driver=min(self.driver,len(self.scenes)-1)
            for i,page in enumerate(self.pages):page.setVisible(i<len(self.methods))
            self.note.setText('依次：逐帧网络 / 独立＋CUDA60（离线SAM3辅助）/ 周期初始化＋15步'+(' / 仅首帧初始化＋15步（失效后不重置）' if len(self.methods)==4 else '')+'。不加载模型；各列视角同步。')
            self.rgb.clear()
            source=json.loads((directory/'inputs.json').read_text());self.faces=load_npz(directory/'topology.npz')['faces']
            self.directory=directory;self.source=source;self.generation+=1;self.pending=None;self.loaded_index=-1
            self.times=[r['time_s'] for r in source['frames']];self.rows={}
            for r in json.loads((directory/'report/rows.json').read_text()):self.rows[(r['frame'],r['method'])]=r
            self.slider.blockSignals(True);self.slider.setRange(0,len(self.times)-1);self.slider.setValue(0);self.slider.blockSignals(False);self.index.setRange(0,len(self.times)-1)
            for scene in self.all_scenes:
                if scene.body:scene.removeItem(scene.body);scene.body=None
                scene._mesh_data.clear();scene.mesh_vertices=None;scene.cloud.setData(pos=np.empty((0,3)),color=np.empty((0,4)),size=scene.point_size,pxMode=True)
            self.status.setText('已载入 '+source['name']+'。鼠标旋转任一场景，其他列同步；缺失帧不显示旧人体。')
            self.seek(0)
        except Exception as exc:self.directory=None;self.status.setText('载入失败：'+str(exc))
    def seek(self,index):
        if self.directory is None:return
        self.index.blockSignals(True);self.index.setValue(index);self.index.blockSignals(False)
        if self.playing:self.play_origin=time.monotonic();self.play_source=self.times[index]
        self.queue(index)
    def queue(self,index):
        if self.busy:self.pending=index;return
        if index==self.loaded_index:return
        self.busy=True;self.pending=None;self.request.emit(self.generation,index,str(self.directory))
    @QtCore.pyqtSlot(int,int,object)
    def loaded(self,generation,index,result):
        self.busy=False
        if generation!=self.generation:
            if self.directory:self.queue(self.pending if self.pending is not None else self.slider.value())
            return
        if 'error' in result:self.pause();self.status.setText('读取失败：'+result['error']);return
        a=result['mesh'];cloud=result['cloud'];self.loaded_index=index
        for m,scene in enumerate(self.scenes):
            scene.raw_cloud=to_view(cloud['cloud']);scene.colours=cloud['colors'].astype(np.float32)/255;scene.person=cloud['person'];scene.show_person=self.person.isChecked();scene.refresh_cloud();scene.cloud.setVisible(self.cloud_check.isChecked())
            if bool(a['valid'][m]):
                view=dict(scene.opts);scene.set_mesh(a['vertices'][m],self.faces,key=(self.generation,index,m));scene.opts.update(view);scene.body.setVisible(self.mesh_check.isChecked())
            elif scene.body:scene.removeItem(scene.body);scene.body=None;scene.mesh_vertices=None
            r=self.rows.get((index,self.methods[m]),{});mae=r.get('mae_mm');iou=r.get('silhouette_iou');elapsed=r.get('core_ms')
            surface=r.get('point_surface_mae_mm');status=str(r.get('status','—'))+(' · 初始化帧' if r.get('keyframe') else '')
            if mae is not None and iou is not None and elapsed is not None:
                surface_text=f'{surface:.1f}' if surface is not None else '—'
                self.stats[m].setText(status+f'\n深度MAE {mae:.1f} mm · 点面MAE {surface_text} mm\nIoU {iou:.3f} · 核心成本 {elapsed:.1f} ms')
            else:self.stats[m].setText(status+' · 无有效人体')
        if self.rows.get((index,'tracking'),{}).get('reference_needs_review'):self.status.setText('此帧 SAM3 参考掩膜待核查，量化结果暂定；不改变已保存人体结果。')
        else:self.status.setText('相机系米制 · 未额外对齐 · 视角同步 · 播放速度不代表处理帧率')
        self.time_label.setText(f'{index}/{len(self.times)-1} · {self.times[index]:.3f}s · 源帧{self.source["frames"][index]["source_index"]}')
        self.rgb.setPixmap(QtGui.QPixmap.fromImage(result['image']).scaled(self.rgb.size(),QtCore.Qt.KeepAspectRatio,QtCore.Qt.SmoothTransformation))
        if index==0:self.view(-45)
        self.sync_views()
        self.slider.blockSignals(True);self.slider.setValue(index);self.slider.blockSignals(False);self.index.blockSignals(True);self.index.setValue(index);self.index.blockSignals(False)
        pending=self.pending;self.pending=None
        if pending is not None and pending!=index:self.queue(pending)
    def toggle(self):
        if self.playing:self.pause();return
        if self.directory is None:return
        if self.loaded_index>=len(self.times)-1:self.slider.setValue(0)
        self.playing=True;self.play_origin=time.monotonic();self.play_source=self.times[max(0,self.slider.value())];self.play_btn.setText('暂停');self.timer.start()
    def pause(self):self.playing=False;self.timer.stop();self.play_btn.setText('播放')
    def step(self,delta):self.pause();self.slider.setValue(max(0,min(self.slider.maximum(),self.slider.value()+delta)))
    def tick(self):
        if self.busy:return
        t=self.play_source+time.monotonic()-self.play_origin
        i=min(len(self.times)-1,max(0,bisect.bisect_right(self.times,t)-1))
        if i!=self.loaded_index:self.queue(i)
        if t>self.times[-1]+.1:
            if self.loop.isChecked() and self.case_paths:
                nxt=(self.cases.currentIndex()+1)%len(self.case_paths);self.cases.setCurrentIndex(nxt)
                if len(self.case_paths)==1:self.case_changed(nxt)
                self.toggle()
            else:self.pause()
    def cloud_mode(self,yes):
        for scene in self.scenes:scene.show_person=yes;scene.refresh_cloud()
    def mesh_visibility(self,yes):
        for scene in self.scenes:
            if scene.body:scene.body.setVisible(yes)
    def view(self,angle):
        # One shared framing for all methods; never re-centre each method independently.
        scene=self.scenes[2]
        if scene.mesh_vertices is None:scene=self.scenes[0]
        scene.focus();center=scene.opts['center'];distance=scene.opts['distance']
        for s in self.scenes:s.opts['center']=QtGui.QVector3D(center);s.setCameraPosition(distance=distance,elevation=8 if angle==-45 else 0,azimuth=angle)
    def eventFilter(self,obj,event):
        if obj in self.scenes and event.type() in (QtCore.QEvent.MouseButtonPress,QtCore.QEvent.Wheel):self.driver=self.scenes.index(obj)
        return super().eventFilter(obj,event)
    def sync_views(self):
        source=self.scenes[self.driver]
        for i,s in enumerate(self.scenes):
            if i==self.driver:continue
            for k in ('center','distance','elevation','azimuth','fov'):s.opts[k]=QtGui.QVector3D(source.opts[k]) if k=='center' else source.opts[k]
            s.update()
    def showEvent(self,event):self.sync.start();super().showEvent(event)
    def hideEvent(self,event):self.pause();self.sync.stop();super().hideEvent(event)
    def open_report(self):
        if self.directory:QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(self.directory.parent/'report.html')))
    def shutdown(self):
        self.pause();self.sync.stop();self.thread.quit();self.thread.wait(5000)
        for scene in self.all_scenes:
            if scene.isValid():scene.makeCurrent();scene.cloud.release_gl();scene.doneCurrent()
