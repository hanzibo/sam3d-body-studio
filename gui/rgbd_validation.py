# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Frozen RGB-D validation workspace. Reuses Studio's resident inference worker."""
from datetime import datetime
import base64
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem
from OpenGL import GL
from pose_data import EDGES
from mesh_geometry import FastMeshData
from pose_set import PoseSetPanel
from record_history import RecordDialog

ROOT=Path(__file__).resolve().parents[1]
from studio_paths import path
CAMERA_PYTHON=str(path('camera_python'))
SAM3_PYTHON=str(path('sam3_python'))


def stamp():return datetime.now().strftime('%Y%m%d-%H%M%S-%f')
def to_view(xyz):return np.ascontiguousarray(np.asarray(xyz)[:,[0,2,1]]*[1,1,-1],dtype=np.float32)


class MaskCanvas(QtWidgets.QWidget):
    changed=QtCore.pyqtSignal()
    promptsChanged=QtCore.pyqtSignal()
    def __init__(self):
        super().__init__();self.setMinimumSize(280,220)
        self.image=QtGui.QImage();self.mask=None;self.points=[];self.drawing=False;self.prompt_mode=False;self.prompt_label=1;self.prompts=[]
        self.live_image=None;self.live_text=''
    def load(self,path,mask=None):
        self.image=QtGui.QImage(str(path)) if path else QtGui.QImage();self.mask=mask;self.points=[];self.drawing=False;self.prompt_mode=False;self.prompts=[];self.update()
    def rectangle(self):
        size=self.image.size().scaled(self.size(),QtCore.Qt.KeepAspectRatio)
        return QtCore.QRectF((self.width()-size.width())/2,(self.height()-size.height())/2,size.width(),size.height())
    def paintEvent(self,event):
        p=QtGui.QPainter(self);p.fillRect(self.rect(),QtGui.QColor('#080d17'))
        if self.live_image is not None:
            size=self.live_image.size().scaled(self.size(),QtCore.Qt.KeepAspectRatio)
            r=QtCore.QRectF((self.width()-size.width())/2,(self.height()-size.height())/2,size.width(),size.height())
            p.drawImage(r,self.live_image);p.fillRect(0,0,self.width(),42,QtGui.QColor(0,0,0,175))
            p.setPen(QtGui.QColor('white'));p.drawText(self.rect().adjusted(8,0,-8,0),QtCore.Qt.AlignTop|QtCore.Qt.AlignHCenter,self.live_text)
            return
        if self.image.isNull():p.setPen(QtGui.QColor('white'));p.drawText(self.rect(),QtCore.Qt.AlignCenter,'请采集或打开 RGB-D 样本');return
        r=self.rectangle();p.drawImage(r,self.image)
        if self.mask is not None:
            rgba=np.zeros((*self.mask.shape,4),np.uint8);rgba[self.mask>0]=[40,220,140,80]
            im=QtGui.QImage(rgba.data,rgba.shape[1],rgba.shape[0],rgba.strides[0],QtGui.QImage.Format_RGBA8888)
            p.drawImage(r,im)
        p.setPen(QtGui.QPen(QtGui.QColor('#ffe16b'),2))
        pts=[QtCore.QPointF(r.x()+x*r.width()/self.image.width(),r.y()+y*r.height()/self.image.height()) for x,y in self.points]
        for a,b in zip(pts,pts[1:]):p.drawLine(a,b)
        for point in pts:p.drawEllipse(point,3,3)
        for x,y,label in self.prompts:
            at=QtCore.QPointF(r.x()+x*r.width()/self.image.width(),r.y()+y*r.height()/self.image.height())
            p.setPen(QtGui.QPen(QtGui.QColor('#ffffff'),1));p.setBrush(QtGui.QColor('#29e68b' if label else '#ff5265'))
            p.drawEllipse(at,6,6)
            p.setPen(QtGui.QColor('#07111c'));p.drawText(QtCore.QRectF(at.x()-5,at.y()-9,10,18),QtCore.Qt.AlignCenter,'+' if label else '−')
    def mousePressEvent(self,event):
        if self.image.isNull() or not self.isEnabled():return
        r=self.rectangle()
        if self.prompt_mode:
            if r.contains(event.pos()) and event.button() in (QtCore.Qt.LeftButton,QtCore.Qt.RightButton):
                x=min(self.image.width()-1,max(0,int((event.x()-r.x())*self.image.width()/r.width())))
                y=min(self.image.height()-1,max(0,int((event.y()-r.y())*self.image.height()/r.height())))
                if len(self.prompts)<128:
                    self.prompts.append([x,y,0 if event.button()==QtCore.Qt.RightButton else self.prompt_label])
                    self.update();self.promptsChanged.emit()
            return
        if not self.drawing:return
        if event.button()==QtCore.Qt.RightButton:
            if self.points:self.points.pop();self.update()
        elif r.contains(event.pos()):
            x=int((event.x()-r.x())*self.image.width()/r.width());y=int((event.y()-r.y())*self.image.height()/r.height())
            self.points.append([min(x,self.image.width()-1),min(y,self.image.height()-1)]);self.update()
    def finish(self):
        if len(self.points)<3:raise ValueError('至少点击三个点，再完成圈选')
        mask=np.zeros((self.image.height(),self.image.width()),np.uint8)
        cv2.fillPoly(mask,[np.asarray(self.points,np.int32)],255)
        self.mask=mask;self.drawing=False;self.points=[];self.update();self.changed.emit()


class PointCloudItem(GLGraphicsItem):
    """Simple depth-tested GL_POINTS; avoids legacy point-sprite texture wrappers."""
    def __init__(self):
        super().__init__();self.setGLOptions('opaque')
        self.positions=np.empty((0,3),np.float32);self.colors=np.empty((0,4),np.float32);self.size=2
    def setData(self,pos,color,size,pxMode=True):
        self.positions=np.ascontiguousarray(pos,dtype=np.float32)
        self.colors=np.ascontiguousarray(color,dtype=np.float32);self.size=size;self.update()
    def paint(self):
        if not len(self.positions):return
        self.setupGLState();GL.glPointSize(self.size)
        GL.glEnableClientState(GL.GL_VERTEX_ARRAY);GL.glEnableClientState(GL.GL_COLOR_ARRAY)
        try:
            GL.glVertexPointerf(self.positions);GL.glColorPointerf(self.colors)
            GL.glDrawArrays(GL.GL_POINTS,0,len(self.positions))
        finally:
            GL.glDisableClientState(GL.GL_COLOR_ARRAY);GL.glDisableClientState(GL.GL_VERTEX_ARRAY)


class CameraScene(gl.GLViewWidget):
    def __init__(self):
        super().__init__();self.setMinimumSize(400,300);self.setBackgroundColor('#0a1120')
        self.cloud=PointCloudItem()
        self.addItem(self.cloud)
        self.bones=gl.GLLinePlotItem(width=3,mode='lines',antialias=True,color=(1,.7,.15,1));self.bones.hide();self.addItem(self.bones)
        self.body=None;self.raw_cloud=None;self.colours=None;self.person=None
        self.show_person=False;self.opacity=.55;self.point_size=2
        self.mesh_vertices=None;self.mesh_faces=None
        axis=gl.GLAxisItem();axis.setSize(.5,.5,.5);self.addItem(axis)
        self.opts['center']=QtGui.QVector3D(0,2,0);self.setCameraPosition(distance=4,elevation=0,azimuth=-90)
    def set_cloud(self,xyz,rgb,mask):
        good=np.isfinite(xyz).all(2)&(xyz[:,:,2]>.05)&(xyz[:,:,2]<10)
        # Deterministic pixel stride for rendering only. Full cloud used in metrics/export.
        yy,xx=np.indices(good.shape);good&=(yy%2==0)&(xx%2==0)
        self.raw_cloud=to_view(xyz[good]);self.colours=rgb[good].astype(np.float32)/255
        self.person=(mask[good]>0) if mask is not None else np.zeros(good.sum(),bool)
        self.refresh_cloud()
    def refresh_cloud(self):
        if self.raw_cloud is None:return
        select=self.person if self.show_person else slice(None)
        colors=self.colours[select];colors=np.column_stack((colors,np.ones(len(colors))))
        self.cloud.setData(pos=self.raw_cloud[select],color=colors,size=self.point_size,pxMode=True)
    def set_mesh(self,vertices,faces):
        first=self.body is None
        self.mesh_vertices=np.array(vertices);self.mesh_faces=np.array(faces)
        if self.body:self.removeItem(self.body)
        self.body=gl.GLMeshItem(meshdata=FastMeshData(vertexes=to_view(vertices),faces=faces),smooth=True,
            shader='body_studio',color=(.68,.82,1,self.opacity),glOptions='translucent',computeNormals=True)
        self.addItem(self.body)
        if first:self.front()
    def set_opacity(self,value):
        self.opacity=value/100
        if self.body:self.body.setColor((.68,.82,1,self.opacity))
    def focus(self):
        if self.mesh_vertices is not None:
            world=to_view(self.mesh_vertices);center=(world.min(0)+world.max(0))/2
            self.opts['center']=QtGui.QVector3D(*map(float,center))
            self.opts['distance']=float(max(np.ptp(world,axis=0).max()*2,.5))
        else:self.opts['center']=QtGui.QVector3D(0,2,0);self.opts['distance']=4
    def front(self):self.focus();self.setCameraPosition(elevation=0,azimuth=-90)
    def side(self):self.focus();self.setCameraPosition(elevation=0,azimuth=0)


class RgbdValidation(QtWidgets.QWidget):
    def __init__(self,host):
        super().__init__();self.host=host;self.sample=None;self.metadata=None;self.xyz=None;self.rgb=None
        self.results=[];self.expected=None;self.active_result=None;self.eval_context=None
        self.segmentation_pending=None;self.segmented_event=None;self.segmentation_result=None;self._applying_sam=False
        self.segment_timer=QtCore.QTimer(self);self.segment_timer.setInterval(100);self.segment_timer.timeout.connect(self.poll_segmentation)
        self.process=QtCore.QProcess(self);self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.read_output);self.process.finished.connect(self.finished)
        self.process.errorOccurred.connect(self.process_error)
        self.buffer='';self.log_text='';self.mode=None;self.done=False;self.pending_sample=None;self.operation_cancelled=False
        self.started=0;self.timeout=QtCore.QTimer(self);self.timeout.setInterval(1000);self.timeout.timeout.connect(self.check_timeout)
        self.preview_frames=0;self.pending_camera_action=None
        self.camera_handoff=QtCore.QTimer(self);self.camera_handoff.setInterval(100);self.camera_handoff.timeout.connect(self.poll_camera_handoff)
        root=QtWidgets.QVBoxLayout(self)
        intro=QtWidgets.QLabel('同一份 RGB-D 对照不同启动方案 · 相机坐标 / 米制 / 原始预测不做对齐 · 仅验证可见表面')
        intro.setWordWrap(True);root.addWidget(intro)
        row=QtWidgets.QHBoxLayout();self.preview_btn=QtWidgets.QPushButton('开启相机预览');self.preview_btn.clicked.connect(self.toggle_preview)
        self.capture_btn=QtWidgets.QPushButton('倒计时采集 RGB-D');self.capture_btn.clicked.connect(self.capture)
        self.delay=QtWidgets.QSpinBox();self.delay.setRange(0,30);self.delay.setValue(5);self.delay.setSuffix(' 秒')
        self.open_btn=QtWidgets.QPushButton('打开已有样本');self.open_btn.clicked.connect(self.open_sample)
        self.cancel_btn=QtWidgets.QPushButton('取消');self.cancel_btn.clicked.connect(self.cancel);self.cancel_btn.setEnabled(False)
        self.folder_btn=QtWidgets.QPushButton('打开样本目录');self.folder_btn.clicked.connect(self.open_folder)
        for w in (self.preview_btn,self.capture_btn,self.delay,self.open_btn,self.cancel_btn,self.folder_btn):row.addWidget(w)
        row.addStretch();root.addLayout(row)
        self.status=QtWidgets.QLabel('开启相机预览后摆好姿势，再倒计时采集；拍摄时保持静止，头脚完整入画。');self.status.setWordWrap(True);root.addWidget(self.status)
        split=QtWidgets.QSplitter();root.addWidget(split,1)
        left=QtWidgets.QWidget();ll=QtWidgets.QVBoxLayout(left);ll.setContentsMargins(0,0,0,0)
        self.canvas=MaskCanvas();self.canvas.changed.connect(self.mask_changed);self.canvas.promptsChanged.connect(self.prompts_changed);ll.addWidget(self.canvas,1)
        promptrow=QtWidgets.QHBoxLayout()
        self.fg_btn=QtWidgets.QPushButton('人体前景点 ＋');self.fg_btn.setCheckable(True)
        self.bg_btn=QtWidgets.QPushButton('背景点 −');self.bg_btn.setCheckable(True)
        self.point_group=QtWidgets.QButtonGroup(self);self.point_group.setExclusive(True)
        self.point_group.addButton(self.fg_btn);self.point_group.addButton(self.bg_btn)
        self.fg_btn.clicked.connect(lambda:self.begin_prompts(1));self.bg_btn.clicked.connect(lambda:self.begin_prompts(0))
        self.undo_point=QtWidgets.QPushButton('撤销点');self.undo_point.clicked.connect(self.undo_prompt)
        self.clear_points=QtWidgets.QPushButton('清空点');self.clear_points.clicked.connect(self.clear_prompts)
        for b in (self.fg_btn,self.bg_btn,self.undo_point,self.clear_points):promptrow.addWidget(b)
        ll.addLayout(promptrow)
        samrow=QtWidgets.QHBoxLayout();self.segment_btn=QtWidgets.QPushButton('SAM3 自动分割（先释放人体模型）')
        self.segment_btn.clicked.connect(self.start_segmentation);samrow.addWidget(self.segment_btn)
        self.candidates=QtWidgets.QComboBox();self.candidates.addItem('尚未分割');self.candidates.currentIndexChanged.connect(self.choose_candidate);samrow.addWidget(self.candidates)
        ll.addLayout(samrow)
        self.seg_info=QtWidgets.QLabel('先标人体前景点；右键可直接添加背景点。分割结束后自动释放 SAM3，再预热人体模型。')
        self.seg_info.setWordWrap(True);ll.addWidget(self.seg_info)
        maskrow=QtWidgets.QHBoxLayout()
        for text,slot in [('手工圈选（备用）',self.begin_mask),('完成圈选',self.finish_mask),('载入掩膜',self.load_mask),('清除',self.clear_mask)]:
            b=QtWidgets.QPushButton(text);b.clicked.connect(slot);maskrow.addWidget(b)
        ll.addLayout(maskrow)
        helptext=QtWidgets.QLabel('优先用 SAM3 点提示分割；残留背景处补红点，漏掉的人体处补绿点，再次分割。SAM 掩膜仍需目视检查。\n手工圈选/载入 PNG 保留作备用；各人体模型方案使用同一份最终掩膜。')
        helptext.setWordWrap(True);ll.addWidget(helptext)
        self.preview=QtWidgets.QLabel();self.preview.setAlignment(QtCore.Qt.AlignCenter);self.preview.setMinimumHeight(100);ll.addWidget(self.preview)
        self.error_image=None
        right=QtWidgets.QWidget();rl=QtWidgets.QVBoxLayout(right);rl.setContentsMargins(0,0,0,0)
        self.scene=CameraScene();rl.addWidget(self.scene,1)
        layers=QtWidgets.QHBoxLayout();self.cloud_check=QtWidgets.QCheckBox('点云');self.cloud_check.setChecked(True)
        self.cloud_check.toggled.connect(self.scene.cloud.setVisible)
        self.mesh_check=QtWidgets.QCheckBox('网格');self.mesh_check.setChecked(True);self.mesh_check.toggled.connect(lambda v:self.scene.body.setVisible(v) if self.scene.body else None)
        self.skeleton_check=QtWidgets.QCheckBox('骨架');self.skeleton_check.toggled.connect(self.scene.bones.setVisible)
        self.person_check=QtWidgets.QCheckBox('仅人体点云');self.person_check.toggled.connect(self.set_person)
        for w in (self.cloud_check,self.mesh_check,self.skeleton_check,self.person_check):layers.addWidget(w)
        for text,slot in [('相机正面',self.scene.front),('侧面',self.scene.side)]:
            b=QtWidgets.QPushButton(text);b.clicked.connect(slot);layers.addWidget(b)
        rl.addLayout(layers)
        params=QtWidgets.QHBoxLayout();params.addWidget(QtWidgets.QLabel('网格透明度'))
        self.alpha=QtWidgets.QSlider(QtCore.Qt.Horizontal);self.alpha.setRange(0,100);self.alpha.setValue(55);self.alpha.valueChanged.connect(self.scene.set_opacity);params.addWidget(self.alpha)
        params.addWidget(QtWidgets.QLabel('点大小'));size=QtWidgets.QSpinBox();size.setRange(1,8);size.setValue(2)
        size.valueChanged.connect(self.point_size);params.addWidget(size);rl.addLayout(params)
        left_scroll=QtWidgets.QScrollArea();left_scroll.setWidgetResizable(True);left_scroll.setFrameShape(QtWidgets.QFrame.NoFrame);left_scroll.setWidget(left)
        split.addWidget(left_scroll);split.addWidget(right);split.setSizes([480,720])
        row=QtWidgets.QHBoxLayout();self.profile_label=QtWidgets.QLabel();self.profile_label.setWordWrap(True);row.addWidget(self.profile_label,1)
        self.profile_btn=QtWidgets.QPushButton('设置启动方案 / 预热');self.profile_btn.clicked.connect(self.configure);row.addWidget(self.profile_btn)
        self.target_mask=QtWidgets.QCheckBox('掩膜框指定人物');self.target_mask.setChecked(True);row.addWidget(self.target_mask)
        self.run_btn=QtWidgets.QPushButton('用当前方案重建此样本');self.run_btn.clicked.connect(self.reconstruct);row.addWidget(self.run_btn)
        self.evaluate_btn=QtWidgets.QPushButton('评估选中结果');self.evaluate_btn.clicked.connect(self.evaluate);row.addWidget(self.evaluate_btn)
        root.addLayout(row)
        self.table=QtWidgets.QTableWidget(0,7);self.table.setHorizontalHeaderLabels(['配置 / 时间','推理 ms','深度 MAE mm','深度 P90 mm','轮廓 IoU','覆盖率','点面 MAE mm'])
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows);self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers);self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(0,QtWidgets.QHeaderView.Stretch);self.table.setMaximumHeight(170)
        self.table.itemSelectionChanged.connect(self.select_result);root.addWidget(self.table)
        self.table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.history_menu)
        self.table.cellDoubleClicked.connect(lambda row,col:self.show_record(row))
        self.table.setToolTip('右键历史项：展示详细配置、删除该项、与另一项对比；双击查看详情。')
        tail=QtWidgets.QHBoxLayout();self.details=QtWidgets.QLabel('误差：模型 Z − 实测 Z；热图蓝色偏前 / 红色偏后，范围 ±50 mm。');self.details.setWordWrap(True);tail.addWidget(self.details,1)
        for text,slot in [('导出完整实验包',self.export_bundle),('保存场景截图',self.save_screenshot)]:
            b=QtWidgets.QPushButton(text);b.clicked.connect(slot);tail.addWidget(b)
        root.addLayout(tail)
        self.host.reconstructionFinished.connect(self.reconstruction_finished)
        self.batch=PoseSetPanel(self);root.insertWidget(1,self.batch)
        self.refresh_timer=QtCore.QTimer(self);self.refresh_timer.setInterval(500);self.refresh_timer.timeout.connect(self.refresh_controls);self.refresh_timer.start()
        self.refresh_controls()

    @property
    def segmenting(self):return self.segmentation_pending is not None or self.mode in ('segment','segment_batch')

    def begin_prompts(self,label):
        if self.sample is None or not self.canvas.isEnabled():return
        self.canvas.drawing=False;self.canvas.points=[];self.canvas.prompt_mode=True;self.canvas.prompt_label=label
        self.fg_btn.setChecked(label==1);self.bg_btn.setChecked(label==0);self.canvas.update()
        self.status.setText('点击人体添加绿点，点击误选背景添加红点；右键总是添加背景点，随后点击 SAM3 自动分割。')
    def prompts_changed(self):
        if not self.sample:return
        rgb_sha=hashlib.sha256((self.sample/'rgb.png').read_bytes()).hexdigest()
        prompt=dict(rgb_sha256=rgb_sha,points_pixels=[p[:2] for p in self.canvas.prompts],point_labels=[p[2] for p in self.canvas.prompts])
        (self.sample/'sam3_prompts.json').write_text(json.dumps(prompt,ensure_ascii=False,indent=2))
        positive=sum(p[2] for p in self.canvas.prompts)
        self.seg_info.setText(f'人体前景 {positive} 点 / 背景 {len(self.canvas.prompts)-positive} 点 · 提示已修改，请点击自动分割生成新掩膜。')
    def undo_prompt(self):
        if self.canvas.isEnabled() and self.canvas.prompts:self.canvas.prompts.pop();self.canvas.update();self.prompts_changed()
    def clear_prompts(self):
        if self.canvas.isEnabled():self.canvas.prompts=[];self.canvas.update();self.prompts_changed()
    def start_segmentation(self):
        host=self.host
        if self.sample is None or self.process.state()!=QtCore.QProcess.NotRunning or self.expected or self.segmenting:return
        if host.inference_busy or host.live.running:
            self.status.setText('请先停止人体重建 / 实时流，再运行 SAM3 分割');return
        if not any(p[2]==1 for p in self.canvas.prompts):
            self.status.setText('请先在人体上至少点击一个绿色前景点');return
        if not Path(SAM3_PYTHON).is_file() or not path('sam3_checkpoint').is_file():
            self.status.setText('找不到本机 SAM3 环境或权重，请确认 Data 已挂载');return
        output=self.sample/'segmentations'/stamp();output.parent.mkdir(exist_ok=True)
        request=dict(image=str(self.sample/'rgb.png'),rgb_sha256=hashlib.sha256((self.sample/'rgb.png').read_bytes()).hexdigest(),
            points_pixels=[p[:2] for p in self.canvas.prompts],point_labels=[p[2] for p in self.canvas.prompts],output=str(output))
        request_path=output.parent/(output.name+'-request.json');request_path.write_text(json.dumps(request,ensure_ascii=False,indent=2))
        self.segmentation_output=str(output);self.begin_segmentation(request_path)
    def begin_segmentation(self,request_path,mode='segment'):
        host=self.host
        self.segmentation_pending=dict(path=str(request_path),started=time.monotonic(),mode=mode)
        self.segmented_event=None;self.log_text=''
        # Suppress profile-change auto relaunch before releasing the body process.
        host.pending_launch=False
        if host.model_state in ('loading','ready'):host.release_model()
        if host.lab.client.device!='cpu' and host.lab.client.process.state()!=QtCore.QProcess.NotRunning:host.lab.release()
        self.status.setText('正在释放人体 / MHR GPU 模型，等待进程完全退出后加载 SAM3…')
        self.segment_timer.start();self.refresh_controls();host.update_camera_controls()
    def poll_segmentation(self):
        if self.segmentation_pending is None:self.segment_timer.stop();return
        host=self.host
        if time.monotonic()-self.segmentation_pending['started']>20:
            self.segmentation_pending=None;self.segment_timer.stop();self.status.setText('人体模型释放超时，已取消 SAM3 加载');self.refresh_controls();host.update_camera_controls();return
        if host.process.state()!=QtCore.QProcess.NotRunning or host.model_state!='off':return
        if host.lab.client.device!='cpu' and host.lab.client.process.state()!=QtCore.QProcess.NotRunning:return
        path=self.segmentation_pending['path'];mode=self.segmentation_pending.get('mode','segment');self.segment_timer.stop()
        # Set mode before clearing pending: no gap in the frontend GPU interlock.
        self.mode=mode;self.segmentation_pending=None
        self.launch(SAM3_PYTHON,'sam3_segment.py',['--request',path],mode);host.update_camera_controls()
    def setup_candidates(self,output,report,selected=None):
        self.segmentation_result=dict(output=str(output),report=report)
        self.candidates.blockSignals(True);self.candidates.clear()
        for candidate in report['candidates']:
            self.candidates.addItem(f"候选 {candidate['index']+1} · 预测质量 {candidate['score']:.3f} · 提示 {candidate['prompt_matches']}/{len(report['point_labels'])}",candidate['index'])
        self.candidates.setCurrentIndex(self.candidates.findData(report['selected'] if selected is None else selected))
        self.candidates.blockSignals(False)
    def choose_candidate(self,*_):
        if not self.segmentation_result or self.candidates.currentData() is None:return
        if self.process.state()!=QtCore.QProcess.NotRunning:return
        data=self.segmentation_result;index=self.candidates.currentData();path=Path(data['output'])/f'mask_{index:02d}.png'
        mask=cv2.imread(str(path),0)
        if mask is None or mask.shape!=self.xyz.shape[:2]:self.status.setText('分割候选掩膜尺寸错误');return
        if not mask.any():self.status.setText('所选候选为空，请换候选或补点分割');return
        self.canvas.mask=mask;self.canvas.drawing=False;self.canvas.points=[]
        self._applying_sam=True
        try:self.mask_changed()
        finally:self._applying_sam=False
        source=dict(method='sam3_point_prompts',segmentation=Path(data['output']).name,selected=index,
            mask_sha256=self.mask_hash(),rgb_sha256=data['report']['rgb_sha256'])
        (self.sample/'mask_source.json').write_text(json.dumps(source,ensure_ascii=False,indent=2))
        report=data['report']
        self.seg_info.setText(f"SAM3 候选 {index+1} 已应用 · 加载 {report['load_seconds']:.1f}s / 分割 {report['inference_seconds']:.2f}s · 峰值保留显存 {report['peak_reserved_gib']:.2f} GiB · 模型进程已退出")
    def restore_segmentation(self):
        self.segmentation_result=None;self.candidates.blockSignals(True);self.candidates.clear();self.candidates.addItem('尚未分割');self.candidates.blockSignals(False)
        self.seg_info.setText('先标人体前景点；右键添加背景点。分割结束后释放 SAM3，再预热人体模型。')
        rgb_sha=hashlib.sha256((self.sample/'rgb.png').read_bytes()).hexdigest()
        prompt_file=self.sample/'sam3_prompts.json'
        if prompt_file.is_file():
            prompt=json.loads(prompt_file.read_text())
            if prompt.get('rgb_sha256')==rgb_sha:self.canvas.prompts=[[*p,l] for p,l in zip(prompt['points_pixels'],prompt['point_labels'])]
        source_file=self.sample/'mask_source.json'
        if source_file.is_file():
            source=json.loads(source_file.read_text());output=self.sample/'segmentations'/source.get('segmentation','')
            if source.get('rgb_sha256')==rgb_sha and source.get('mask_sha256')==self.mask_hash() and (output/'report.json').is_file():
                report=json.loads((output/'report.json').read_text());self.setup_candidates(output,report,source['selected'])
                self.seg_info.setText('已恢复 SAM3 掩膜与提示点，可补点重新分割或直接预热人体模型。')

    def refresh_controls(self):
        host=self.host
        busy=self.process.state()!=QtCore.QProcess.NotRunning or self.expected is not None or self.segmentation_pending is not None or self.pending_camera_action is not None
        free=not host.inference_busy and host.camera_state in ('off','photo') and not host.camera.active
        previewing=self.mode=='preview' and self.process.state()==QtCore.QProcess.Running
        can_camera=not host.inference_busy and not (getattr(host,'live',None) and host.live.active)
        self.preview_btn.setText('关闭相机预览' if self.mode=='preview' else '开启相机预览')
        self.preview_btn.setEnabled(previewing or (not busy and can_camera))
        self.capture_btn.setEnabled(previewing or (not busy and can_camera));self.open_btn.setEnabled(not busy and not host.inference_busy)
        self.cancel_btn.setEnabled(busy);self.run_btn.setEnabled(self.sample is not None and not busy and free and host.model_state!='stopping')
        self.evaluate_btn.setEnabled(self.active_result is not None and self.canvas.mask is not None and not busy)
        config=host.draft_config()
        self.profile_label.setText('当前方案：'+(config.summary() if config else '参数组合无效，请到第一页调整'))
        self.profile_label.setToolTip(json.dumps(config.to_dict(),ensure_ascii=False,indent=2) if config else '')
        if config is None:self.run_btn.setEnabled(False)
        self.segment_btn.setEnabled(self.sample is not None and not busy and not host.inference_busy and not (getattr(host,'live',None) and host.live.running))
        self.canvas.setEnabled(not busy)
        for button in (self.fg_btn,self.bg_btn,self.undo_point,self.clear_points):button.setEnabled(self.sample is not None and not busy)
        self.candidates.setEnabled(self.segmentation_result is not None and not busy)
        if hasattr(self,'batch'):self.batch.refresh()

    def launch(self,python,script,args,mode):
        self.mode=mode;self.done=False;self.operation_cancelled=False;self.buffer='';self.log_text='';self.started=time.monotonic()
        env=QtCore.QProcessEnvironment.systemEnvironment()
        for key in ('PYTHONPATH','QT_QPA_PLATFORM_PLUGIN_PATH','QT_QPA_FONTDIR'):env.remove(key)
        self.process.setProcessEnvironment(env);self.process.setWorkingDirectory(str(ROOT))
        self.process.start(python,['-u',str(ROOT/script),*args]);self.timeout.start();self.refresh_controls()
    def capture(self):
        if not self.capture_btn.isEnabled():return
        if not self.batch.confirm_capture():return
        if self.mode=='preview':
            self.mode='capture';self.started=time.monotonic()
            self.process.write((json.dumps(dict(type='capture',delay=self.delay.value()))+'\n').encode())
            self.status.setText('倒计时已开始，请摆好姿势');self.refresh_controls();self.host.update_camera_controls();return
        self.request_camera('capture')
    def toggle_preview(self):
        if self.mode=='preview':self.cancel();return
        if self.preview_btn.isEnabled():self.request_camera('preview')
    def request_camera(self,action):
        self.preview_frames=0
        self.pending_camera_action=action;self.handoff_started=time.monotonic()
        if self.host.camera.active:self.host.close_camera()
        self.status.setText('正在切换到 RGB-D 相机…');self.camera_handoff.start();self.refresh_controls();self.host.update_camera_controls()
    def poll_camera_handoff(self):
        if time.monotonic()-self.handoff_started>10:
            self.pending_camera_action=None;self.camera_handoff.stop();self.status.setText('相机释放超时，请关闭其他占用相机的程序');self.refresh_controls();return
        if self.host.camera.active or self.host.camera_state not in ('off','photo'):return
        action=self.pending_camera_action;self.pending_camera_action=None;self.camera_handoff.stop()
        self.pending_sample=self.batch.capture_directory() or ROOT/'outputs/rgbd'/stamp()
        self.preview_frames=0;self.canvas.live_text='正在连接 RGB-D 相机…'
        args=['--output',str(self.pending_sample),'--delay',str(self.delay.value())]
        if action=='preview':args.append('--preview')
        self.launch(CAMERA_PYTHON,'rgbd_capture.py',args,action);self.host.update_camera_controls()
    def read_output(self):
        self.buffer+=bytes(self.process.readAllStandardOutput()).decode('utf8',errors='replace')
        while '\n' in self.buffer:
            line,self.buffer=self.buffer.split('\n',1)
            if not line.startswith('@@RGBD {"type": "frame"'):self.log_text+=line+'\n'
            if not line.startswith('@@RGBD '):continue
            try:
                event=json.loads(line[7:]);kind=event['type']
                self.batch.worker_event(event)
                if kind=='frame':
                    im=QtGui.QImage.fromData(base64.b64decode(event['jpeg']),'JPG')
                    if not im.isNull():self.canvas.live_image=im;self.preview_frames+=1;self.canvas.update()
                elif kind in ('ready','stage','error'):
                    message=event['message']
                    if kind=='ready' and self.batch.session and self.batch.capture_index is not None:message=f'姿态{self.batch.capture_index+1}/{len(self.batch.session.poses)} · '+message
                    self.status.setText(message);self.canvas.live_text=message;self.canvas.update()
                    if kind=='ready':self.refresh_controls()
                elif kind=='countdown':
                    prefix=f'姿态{self.batch.capture_index+1}/{len(self.batch.session.poses)} · ' if self.batch.session and self.batch.capture_index is not None else ''
                    self.status.setText(prefix+f"还有 {event['seconds']} 秒拍摄，请摆好姿势并保持静止")
                    self.canvas.live_text=self.status.text();self.canvas.update()
                elif kind=='done':self.done=True;self.pending_sample=Path(event['directory'])
                elif kind=='segmented':self.segmented_event=event
                elif kind=='batch_complete':self.done=True
                elif kind=='evaluated':
                    self.done=True;self.eval_context['evaluation']=event['output'];self.eval_context['metrics']=event['metrics']
                    self.save_results();self.update_table()
                    if self.active_result:self.display_evaluation(self.active_result)
                    self.status.setText('评估完成 · 保留原始相机位置与尺度 · 可切换方案重建同一样本进行对照')
            except Exception as exc:self.status.setText('结果处理失败：'+str(exc));self.done=False
    def finished(self,code,status):
        self.read_output();self.timeout.stop()
        finished_mode=self.mode
        self.canvas.live_image=None;self.canvas.live_text='';self.canvas.update()
        if self.operation_cancelled:
            self.status.setText('相机已释放，原始样本与已保存掩膜保留。' if self.mode in ('preview','capture') else '已取消，原始样本与已保存掩膜保留；模型进程已退出。')
        elif self.mode=='segment' and code==0 and self.segmented_event is not None:
            try:
                event=self.segmented_event;self.setup_candidates(event['output'],event['report'])
                self.done=True;self.choose_candidate()
                self.status.setText('SAM3 已退出并释放显存；检查绿色掩膜，满意后可预热人体模型。')
            except Exception as exc:self.status.setText('分割结果载入失败：'+str(exc))
        elif self.mode=='capture' and self.done and code==0:
            try:self.load_sample(self.pending_sample)
            except Exception as exc:self.status.setText('样本载入失败：'+str(exc))
        elif not self.done and code!=0:
            self.status.setText(self.status.text()+' · 操作未完成（可重新采集/评估）')
        logdir=self.pending_sample if self.mode=='capture' else (Path(self.segmentation_output) if self.mode=='segment' else (Path(self.eval_context['evaluation_pending']) if self.eval_context else None))
        if logdir:
            logdir.mkdir(parents=True,exist_ok=True);(logdir/'worker.log').write_text(self.log_text)
        self.mode=None;self.eval_context=None;self.refresh_controls()
        self.host.update_camera_controls()
        self.batch.operation_finished(finished_mode,code==0 and self.done and not self.operation_cancelled)
    def process_error(self,error):
        if self.operation_cancelled and error==QtCore.QProcess.Crashed:return
        self.status.setText('进程错误：'+self.process.errorString());self.timeout.stop()
        if error==QtCore.QProcess.FailedToStart:
            self.mode=None;self.eval_context=None;self.canvas.live_image=None;self.canvas.update();self.host.update_camera_controls()
        self.refresh_controls()
    def check_timeout(self):
        if self.mode=='preview':return
        limit=90 if self.mode=='capture' else (180+30*len(self.batch.session.poses) if self.mode=='segment_batch' and self.batch.session else 180)
        if time.monotonic()-self.started>limit:
            self.status.setText('操作超时，正在取消');self.cancel()
    def cancel(self):
        if self.batch.active:self.batch.stop();return
        self.operation_cancelled=True
        self.pending_camera_action=None;self.camera_handoff.stop()
        self.segmentation_pending=None;self.segment_timer.stop()
        if self.expected:
            self.expected=None;self.host.cancel_run()
        if self.process.state()!=QtCore.QProcess.NotRunning:
            pid=self.process.processId();self.process.terminate()
            QtCore.QTimer.singleShot(2500,lambda:self.kill_if_running(pid))
        self.status.setText('已请求取消，正在释放资源…');self.refresh_controls();self.host.update_camera_controls()
    def kill_if_running(self,pid):
        if self.process.state()!=QtCore.QProcess.NotRunning and self.process.processId()==pid:self.process.kill()
    def open_sample(self):
        directory=QtWidgets.QFileDialog.getExistingDirectory(self,'选择含 camera.json / rgbd.npz 的样本目录',str(ROOT/'outputs/rgbd'))
        if directory:
            try:self.load_sample(directory)
            except Exception as exc:self.status.setText('载入失败：'+str(exc))
    def clear_sample(self):
        self.sample=None;self.metadata=None;self.xyz=None;self.rgb=None;self.results=[];self.active_result=None
        self.canvas.load('');self.segmentation_result=None;self.candidates.clear();self.candidates.addItem('尚未分割')
        if self.scene.body:self.scene.removeItem(self.scene.body);self.scene.body=None
        self.scene.mesh_vertices=None;self.scene.mesh_faces=None;self.scene.raw_cloud=None
        self.scene.cloud.setData(pos=np.empty((0,3),np.float32),color=np.empty((0,4),np.float32))
        self.scene.bones.setData(pos=np.empty((0,3),np.float32));self.table.setRowCount(0);self.preview.clear()
        self.details.setText('请拍摄当前姿态。');self.seg_info.setText('拍摄后可为当前姿态标注前景 / 背景点。')
        self.status.setText('当前姿态尚未拍摄，请开启相机预览后倒计时采集。');self.refresh_controls()
    def load_sample(self,directory):
        if self.segmenting:raise ValueError('SAM3 分割尚未退出，不能切换样本')
        directory=Path(directory).resolve();meta=json.loads((directory/'camera.json').read_text())
        if meta.get('schema')!='studio_rgbd_v1' or not meta.get('rgb_rectified') or meta.get('units')!='metres':raise ValueError('样本必须为 Studio 标定配准后的 RGB-D 格式')
        with np.load(directory/'rgbd.npz',allow_pickle=False) as data:xyz=data['xyz_m'].copy();depth=data['depth_m']
        rgb=cv2.imread(str(directory/'rgb.png'))
        if rgb is None or xyz.shape!=(*rgb.shape[:2],3) or depth.shape!=rgb.shape[:2]:raise ValueError('RGB-D 尺寸不一致')
        mask=cv2.imread(str(directory/'mask.png'),0) if (directory/'mask.png').is_file() else None
        if mask is not None and mask.shape!=depth.shape:raise ValueError('人体掩膜尺寸不一致')
        self.sample=directory;self.metadata=meta;self.xyz=xyz;self.rgb=rgb[:,:,::-1].copy()
        self.canvas.load(directory/'rgb.png',mask);self.restore_segmentation();self.scene.set_cloud(xyz,self.rgb,mask)
        if self.scene.body:self.scene.removeItem(self.scene.body);self.scene.body=None
        self.scene.bones.setData(pos=np.empty((0,3),np.float32))
        self.scene.mesh_vertices=None;self.scene.mesh_faces=None
        self.active_result=None;self.results=[]
        if (directory/'results.json').is_file():
            self.results=json.loads((directory/'results.json').read_text())
            for result in self.results:
                result['prediction']=str(directory/'predictions'/Path(result['prediction']).parent.name/'person_00.npz')
                if result.get('evaluation'):result['evaluation']=str(directory/'evaluations'/Path(result['evaluation']).name)
        self.update_table()
        if self.results:self.table.selectRow(0);self.select_result()
        self.status.setText(f"样本 {directory.name} · 深度有效点 {(depth>.05).sum():,} · RGB-D 时间差 {meta['timestamps']['skew_ms']:.1f} ms · 可标点分割或直接重建")
        pix=QtGui.QPixmap(str(directory/'depth_preview.png'));self.preview.setPixmap(pix.scaled(420,140,QtCore.Qt.KeepAspectRatio,QtCore.Qt.SmoothTransformation))
        self.refresh_controls()
    def begin_mask(self):
        if self.sample and self.process.state()==QtCore.QProcess.NotRunning and self.expected is None and not self.segmenting:
            self.canvas.points=[];self.canvas.drawing=True;self.canvas.prompt_mode=False;self.canvas.update();self.status.setText('沿人体轮廓逐点点击，右键撤销，点击“完成圈选”保存')
    def finish_mask(self):
        if not self.canvas.drawing or not self.canvas.isEnabled():return
        try:self.canvas.finish()
        except ValueError as exc:self.status.setText(str(exc))
    def load_mask(self):
        if not self.sample or self.process.state()!=QtCore.QProcess.NotRunning or self.expected or self.segmenting:return
        path,_=QtWidgets.QFileDialog.getOpenFileName(self,'选择独立人体二值掩膜','','PNG (*.png)')
        if path:
            mask=cv2.imread(path,0)
            if mask is None or mask.shape!=self.xyz.shape[:2]:self.status.setText('掩膜必须与 RGB 尺寸完全相同');return
            self.canvas.mask=np.uint8(mask>0)*255;self.canvas.drawing=False;self.canvas.points=[];self.mask_changed()
    def clear_mask(self):
        if self.sample and self.process.state()==QtCore.QProcess.NotRunning and self.expected is None and not self.segmenting:
            self.canvas.mask=None;self.canvas.drawing=False;self.canvas.points=[];self.mask_changed()
    def mask_changed(self):
        if not self.sample:return
        if not self._applying_sam:(self.sample/'mask_source.json').unlink(missing_ok=True)
        if self.canvas.mask is not None:cv2.imwrite(str(self.sample/'mask.png'),self.canvas.mask)
        else:(self.sample/'mask.png').unlink(missing_ok=True)
        self.scene.set_cloud(self.xyz,self.rgb,self.canvas.mask);self.canvas.update()
        # Existing evaluations remain archived, but scores for a different mask are hidden.
        self.update_table();self.refresh_controls()
        if self.active_result:self.display_evaluation(self.active_result)
    def set_person(self,value):self.scene.show_person=value;self.scene.refresh_cloud()
    def point_size(self,value):self.scene.point_size=value;self.scene.refresh_cloud()
    def configure(self):
        self.host.workspace_tabs.setCurrentIndex(0);self.host.profile_toggle.setChecked(True)
    def reconstruct(self):
        if not self.run_btn.isEnabled():return
        host=self.host
        if host.model_state=='ready' and host.draft_config()!=host.active_config:
            self.status.setText('启动方案已修改，请到第一页应用并预热后再重建');return
        bbox=None
        if self.target_mask.isChecked() and self.canvas.mask is not None and self.canvas.mask.any():
            x,y,width,height=cv2.boundingRect(self.canvas.mask);bbox=[x,y,x+width,y+height]
        host.load_image(self.sample/'rgb.png',bbox);host.intrinsics_check.setChecked(True)
        self.expected=str(self.sample/'rgb.png');host.start_reconstruction()
        if not host.inference_busy:self.expected=None;self.status.setText(host.status.text())
        else:self.status.setText('正在用当前启动方案重建固定 RGB-D 样本…')
        self.refresh_controls()
    def reconstruction_finished(self,ok):
        if self.expected is None:return
        expected=self.expected;self.expected=None
        if not ok or str(self.host.source_path)!=expected:self.status.setText('重建未完成，样本保持不变');self.refresh_controls();return
        run=self.sample/'predictions'/stamp();run.mkdir(parents=True)
        for name in ('person_00.npz','request.json','inference.json'):
            source=self.host.result_dir/name
            if source.is_file():shutil.copy2(source,run/name)
        report=self.host.report
        result=dict(label=report.get('config_id','')+' / '+run.name,config=report.get('actual_config'),
            prediction=str(run/'person_00.npz'),inference_ms=report['inference_seconds']*1000,
            rgb_sha256=hashlib.sha256((self.sample/'rgb.png').read_bytes()).hexdigest(),
            person_bbox_xyxy=self.host.canvas.bbox)
        self.results.append(result);self.save_results();self.update_table();self.table.selectRow(len(self.results)-1)
        self.status.setText('重建已完成，已保留真实尺度与相机平移；可圈选后评估，或切换方案重复重建')
        self.refresh_controls()
        if self.canvas.mask is not None:QtCore.QTimer.singleShot(0,self.evaluate)
    def save_results(self):
        temporary=self.sample/'results.json.tmp'
        temporary.write_text(json.dumps(self.results,ensure_ascii=False,indent=2))
        temporary.replace(self.sample/'results.json')
        with (self.sample/'comparison.csv.tmp').open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['config','inference_ms','mask_sha256','depth_mae_mm','depth_p90_mm','iou','coverage','point_surface_mae_mm'])
            for r in self.results:
                m=r.get('metrics',{});writer.writerow([r['label'],r['inference_ms'],m.get('mask_sha256'),m.get('mae_mm'),m.get('p90_mm'),m.get('silhouette_iou'),m.get('model_coverage_fraction'),m.get('point_surface_mae_mm')])
        (self.sample/'comparison.csv.tmp').replace(self.sample/'comparison.csv')

    def history_idle(self):
        return (self.sample is not None and self.process.state()==QtCore.QProcess.NotRunning
                and self.expected is None and not self.segmenting and not self.batch.active
                and not self.host.inference_busy and self.pending_camera_action is None)

    def build_history_menu(self,row):
        if not 0<=row<len(self.results):return None
        self.table.selectRow(row)
        menu=QtWidgets.QMenu(self.table)
        for text,callback,enabled in [
                ('展示详细配置',lambda:self.show_record(row),self.history_idle()),
                ('删除该项',lambda:self.delete_record(row),self.history_idle()),
                ('与另一项进行详细配置及推理等信息对比',lambda:self.show_record(row,True),self.history_idle() and len(self.results)>1)]:
            action=menu.addAction(text);action.setEnabled(enabled);action.triggered.connect(callback)
        return menu

    def history_menu(self,pos):
        menu=self.build_history_menu(self.table.rowAt(pos.y()))
        if menu:
            menu.exec_(self.table.viewport().mapToGlobal(pos));menu.deleteLater()

    def show_record(self,row,compare=False):
        if not self.history_idle() or not 0<=row<len(self.results) or (compare and len(self.results)<2):return
        dialog=RecordDialog(self,row,compare);dialog.exec_();dialog.deleteLater()

    def delete_record(self,row):
        if not self.history_idle() or not 0<=row<len(self.results):return False
        # Archive the entire index before changing it. Meshes/reports/metrics stay on disk.
        backup=self.sample/'history-trash'/('results-'+stamp()+'.json')
        original=list(self.results)
        try:
            backup.parent.mkdir(exist_ok=True)
            backup.write_text(json.dumps(dict(deleted_index=row,deleted_at=stamp(),results=original),ensure_ascii=False,indent=2))
            self.results.pop(row);self.save_results()
        except Exception as exc:
            self.results=original
            # Repair a partially written index/CSV if possible; retain the recovery archive.
            try:self.save_results()
            except Exception:pass
            self.status.setText('删除未完成：'+str(exc));return False
        self.active_result=None;self.eval_context=None;self.update_table()
        if self.results:self.table.selectRow(min(row,len(self.results)-1));self.select_result()
        else:
            if self.scene.body:self.scene.removeItem(self.scene.body);self.scene.body=None
            self.scene.mesh_vertices=None;self.scene.mesh_faces=None
            self.scene.bones.setData(pos=np.empty((0,3),np.float32))
            self.preview.clear();self.details.setText('暂无历史结果，可使用当前方案重新重建。')
        self.refresh_controls()
        message='已移除历史项，预测及评估文件保留。恢复索引：'+str(backup)
        if self.batch.session:
            try:self.batch.session.export_csv(self.batch.session.directory);self.batch.refresh()
            except Exception as exc:message+='；样本集汇总更新失败：'+str(exc)
        self.status.setText(message);return True
    def mask_hash(self):
        path=self.sample/'mask.png' if self.sample else None
        return hashlib.sha256(path.read_bytes()).hexdigest() if path and path.is_file() else None
    def update_table(self):
        selected=self.table.currentRow();self.table.blockSignals(True);self.table.setRowCount(len(self.results))
        mask_hash=self.mask_hash()
        for row,r in enumerate(self.results):
            m=r.get('metrics',{})
            if m.get('mask_sha256')!=mask_hash:m={}
            config=r.get('config') or {};label=f"{config.get('backend','?')} / {config.get('mode','?')} / {config.get('image_size','?')} / {r['label']}"
            vals=[label,f"{r['inference_ms']:.1f}"]+[f'{m[k]:.3f}' if k in m else '—' for k in ('mae_mm','p90_mm','silhouette_iou','model_coverage_fraction','point_surface_mae_mm')]
            for col,value in enumerate(vals):self.table.setItem(row,col,QtWidgets.QTableWidgetItem(value))
        self.table.blockSignals(False)
        if 0<=selected<len(self.results):self.table.selectRow(selected)
    def select_result(self):
        row=self.table.currentRow()
        if not 0<=row<len(self.results):return
        try:
            result=self.results[row]
            with np.load(result['prediction'],allow_pickle=False) as d:
                self.scene.set_mesh(d['pred_vertices']+d['pred_cam_t'].reshape(1,3),d['faces'])
                if 'pred_keypoints_3d' in d:
                    points=to_view(d['pred_keypoints_3d']+d['pred_cam_t'].reshape(1,3))
                    self.scene.bones.setData(pos=points[EDGES].reshape(-1,3))
            self.scene.body.setVisible(self.mesh_check.isChecked());self.active_result=result;self.display_evaluation(result)
        except Exception as exc:self.status.setText('结果载入失败：'+str(exc))
        self.refresh_controls()
    def evaluate(self):
        if not self.active_result or self.canvas.mask is None or self.process.state()!=QtCore.QProcess.NotRunning:return
        out=self.sample/'evaluations'/stamp();out.parent.mkdir(exist_ok=True)
        self.eval_context=self.active_result;self.eval_context['evaluation_pending']=str(out)
        self.status.setText('正在渲染相机视角深度并计算误差，保持原始位置与尺度…')
        self.launch(sys.executable,'rgbd_evaluate.py',['--sample',str(self.sample),'--prediction',self.active_result['prediction'],
            '--mask',str(self.sample/'mask.png'),'--output',str(out)],'evaluate')
    def display_evaluation(self,result):
        m=result.get('metrics')
        if not m or m.get('mask_sha256')!=self.mask_hash():
            self.details.setText('当前掩膜尚未评估。原始结果不做平移、缩放或 ICP；仅验证可见表面。');self.preview.clear();return
        self.details.setText(f"深度有符号均值 {m['signed_mean_mm']:.1f} mm（正值偏后） · 有效深度 {m['valid_depth_fraction']:.1%} · <1/2/5 cm：{m['within_10mm']:.1%} / {m['within_20mm']:.1%} / {m['within_50mm']:.1%}\n热图蓝色偏前 / 红色偏后（±50 mm），灰暗区域不参与深度误差；点面距离采样 {m['point_surface_count']} 点。")
        pix=QtGui.QPixmap(str(Path(result['evaluation'])/'depth_error.png'));self.preview.setPixmap(pix.scaled(420,160,QtCore.Qt.KeepAspectRatio,QtCore.Qt.SmoothTransformation))
    def open_folder(self):
        if self.sample:QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(self.sample)))
    def export_bundle(self):
        if not self.sample or self.process.state()!=QtCore.QProcess.NotRunning or self.expected or self.segmenting:return
        parent=QtWidgets.QFileDialog.getExistingDirectory(self,'选择实验包导出位置')
        if parent:
            target=Path(parent)/('rgbd-experiment-'+self.sample.name)
            try:
                # Cannot export into the source tree and recurse indefinitely.
                if target.resolve().is_relative_to(self.sample.resolve()):raise ValueError('请选择样本目录之外的位置')
                shutil.copytree(self.sample,target)
                # All paths in results must point inside the copied package.
                if (target/'results.json').is_file():
                    text=(target/'results.json').read_text().replace(str(self.sample),str(target))
                    (target/'results.json').write_text(text)
                self.status.setText('实验包已导出：'+str(target))
            except Exception as exc:self.status.setText('导出失败：'+str(exc))
    def save_screenshot(self):
        if not self.sample:return
        path,_=QtWidgets.QFileDialog.getSaveFileName(self,'保存当前场景',str(self.sample/('scene-'+stamp()+'.png')),'PNG (*.png)')
        if path:self.scene.grabFramebuffer().save(path)
    def shutdown(self):
        self.batch.shutdown()
        self.camera_handoff.stop();self.pending_camera_action=None
        self.refresh_timer.stop();self.timeout.stop();self.segment_timer.stop();self.segmentation_pending=None;self.expected=None
        if self.process.state()!=QtCore.QProcess.NotRunning:
            self.process.terminate()
            if not self.process.waitForFinished(2500):self.process.kill();self.process.waitForFinished(1500)
