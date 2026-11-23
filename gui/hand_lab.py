# SPDX-License-Identifier: GPL-3.0-only
"""Manual ROI hand experiment, isolated from the body reconstruction pipeline."""
from datetime import datetime
import json
import subprocess
from pathlib import Path
import uuid
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from studio.hand.hand_geometry import validate_bbox
from studio.cameras.gui_camera import select_camera_intrinsics
from overlay_view import OverlayView
from hand_sequence_panel import HandSequencePanel
from hand_live import HandLivePanel
from studio_paths import path as configured_path

ROOT = Path(__file__).resolve().parents[1]


class HandLab(QtWidgets.QWidget):
    def __init__(self, host, canvas_factory, viewer_factory):
        super().__init__(); self.host = host
        self.source = None; self.metadata = None; self.result_dir = None
        self.state = 'off'; self.job = None; self.buffer = ''; self.log_file = None
        self.startup_warmup=None
        self.startup_warmup_check=QtWidgets.QCheckBox('加载后自动离线预热（单手＋双手）')
        self.startup_warmup_check.setChecked(host.settings.value('hand/startup_warmup',True,type=bool))
        self.startup_warmup_source=QtWidgets.QLineEdit(str(host.settings.value('hand/warmup_image',str(configured_path('hand_warmup_image')))))
        self.startup_warmup_source.setPlaceholderText('选择一张已有的清晰手部照片，无需连接相机')
        self.startup_warmup_source.setToolTip('只读取本地样本用于编译/图捕获，不导出该照片的网格，也不上传。')
        self.startup_warmup_browse=QtWidgets.QPushButton('选择预热样本…');self.startup_warmup_browse.clicked.connect(self.choose_warmup_sample)
        self.process = QtCore.QProcess(self)
        self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.read_output)
        self.process.finished.connect(self.finished)
        self.process.errorOccurred.connect(self.process_error)
        self.switch_timer = QtCore.QTimer(self); self.switch_timer.setInterval(100)
        self.switch_timer.timeout.connect(self.wait_body)
        layout = QtWidgets.QVBoxLayout(self)
        title = QtWidgets.QLabel('手部实验 · 手动 / 自动找手 → 独立手部分支 → 单帧 / 双手序列')
        title.setWordWrap(True); layout.addWidget(title)
        hint = QtWidgets.QLabel('手部 512 · 编码器按启动方案 / 解码器与 MHR FP32。跳过身体解码；手框应包含整只手和少量手腕。左手以镜像右手路径实验，原始参数仍为右手规范。')
        hint.setWordWrap(True); layout.addWidget(hint)
        row = QtWidgets.QHBoxLayout(); layout.addLayout(row)
        self.open_btn = QtWidgets.QPushButton('打开图片…'); self.open_btn.clicked.connect(self.open_image); row.addWidget(self.open_btn)
        self.use_btn = QtWidgets.QPushButton('使用第一页照片'); self.use_btn.clicked.connect(self.use_photo); row.addWidget(self.use_btn)
        self.side = QtWidgets.QComboBox(); self.side.addItem('右手','right'); self.side.addItem('左手（镜像实验）','left'); row.addWidget(self.side)
        row.addWidget(QtWidgets.QLabel('裁剪比例'))
        self.padding = QtWidgets.QDoubleSpinBox(); self.padding.setRange(.7,2); self.padding.setSingleStep(.05); self.padding.setValue(.9); row.addWidget(self.padding)
        self.intrinsics = QtWidgets.QCheckBox('使用照片 / Gemini 内参'); row.addWidget(self.intrinsics); row.addStretch()
        acceleration=QtWidgets.QHBoxLayout();layout.addLayout(acceleration)
        acceleration.addWidget(QtWidgets.QLabel('手部启动方案'))
        self.backbone_choice=QtWidgets.QComboBox();self.backbone_choice.addItem('原生 FP16 编码器（基线）','native');self.backbone_choice.addItem('TensorRT FP16 · 512','trt_fp16');self.backbone_choice.addItem('TensorRT INT8 / FP16 · 512（实验）','trt_int8');acceleration.addWidget(self.backbone_choice)
        self.intermediate_compile=QtWidgets.QCheckBox('中间预测回调整体编译')
        self.intermediate_compile.setToolTip('编译姿态参数预测、相机投影及裁剪坐标转换；保留 MHR Graph 边界。首次预热需编译。')
        self.prediction_count=QtWidgets.QComboBox()
        for count in range(5,-1,-1):
            layers=','.join(map(str,range(count))) if count else '无'
            self.prediction_count.addItem(f'{count} 次（层 {layers}）'+(' · 原始默认' if count==5 else ' · 实验'),count)
        self.prediction_count.setToolTip('六层解码与最终预测始终保留。只在选定层预测并反馈；减少次数可能改变重建精度，须重新加载 / 预热。')
        self.decoder_compile=QtWidgets.QCheckBox('解码层编译');acceleration.addWidget(self.decoder_compile)
        self.mhr_compile=QtWidgets.QCheckBox('MHR 前后处理编译');acceleration.addWidget(self.mhr_compile)
        self.mhr_graph=QtWidgets.QCheckBox('MHR CUDA Graph');acceleration.addWidget(self.mhr_graph)
        self.compile_mode=QtWidgets.QComboBox();self.compile_mode.addItem('编译：default','default');self.compile_mode.addItem('编译：reduce-overhead（实验）','reduce-overhead');acceleration.addWidget(self.compile_mode)
        acceleration.addStretch()
        feedback_row=QtWidgets.QHBoxLayout();layout.addLayout(feedback_row)
        self.feedback_cache=QtWidgets.QCheckBox('反馈复制优化');feedback_row.addWidget(self.feedback_cache)
        self.feedback_cache.setToolTip('合并 2D/3D 反馈写入；只在同次调用、同一姿态对象内缓存，不复用上一帧。')
        self.intermediate_keypoints=QtWidgets.QCheckBox('中间预测仅计算手部关键点');feedback_row.addWidget(self.intermediate_keypoints)
        self.intermediate_keypoints.setToolTip('保持所选中间预测与最终预测。按实际回归权重选择依赖，中间跳过无关网格；最终表面由独立开关控制，姿态修正保留。')
        self.final_hand_vertices=QtWidgets.QCheckBox('最终仅生成手部网格');feedback_row.addWidget(self.final_hand_vertices)
        self.final_hand_vertices.setToolTip('仅计算并投影手部 2321 顶点；保留完整骨架依赖、形状与姿态修正。关闭可与原生全身表面再截取对照。')
        feedback_row.addStretch()
        detector_row=QtWidgets.QHBoxLayout();layout.addLayout(detector_row);detector_row.addWidget(QtWidgets.QLabel('自动手框执行'))
        self.detector_choice=QtWidgets.QComboBox();self.detector_choice.addItem('原生 WiLoR YOLOv8m-Pose','native');self.detector_choice.addItem('TensorRT FP16 · 640','trt_fp16');detector_row.addWidget(self.detector_choice)
        self.accelerated_btn=QtWidgets.QPushButton('采用 512 加速方案');self.accelerated_btn.clicked.connect(lambda:self.set_execution_preset(True));detector_row.addWidget(self.accelerated_btn)
        self.int8_btn=QtWidgets.QPushButton('试用 INT8 512');self.int8_btn.clicked.connect(self.set_int8_preset);detector_row.addWidget(self.int8_btn)
        self.int8_btn.setToolTip('复用 Embedl 混合 INT8/FP16 编码器；双手分别编码后合批解码。量化可能改变预测，保留 FP16 对照。')
        self.baseline_btn=QtWidgets.QPushButton('恢复原生基线');self.baseline_btn.clicked.connect(lambda:self.set_execution_preset(False));detector_row.addWidget(self.baseline_btn);detector_row.addStretch()
        self.execution_hint=QtWidgets.QLabel('启动选项在加载前设置；修改须先释放模型。单 / 双手形状首次运行会分别预热，首次编译可能较慢；解码器与 MHR 保持 FP32。')
        self.execution_hint.setWordWrap(True);layout.addWidget(self.execution_hint)
        self.tabs = QtWidgets.QTabWidget(); layout.addWidget(self.tabs,1)
        class DetectionCanvas(canvas_factory):
            detected=[]
            def paintEvent(canvas,event):
                super().paintEvent(event)
                if canvas.image.isNull():return
                painter=QtGui.QPainter(canvas);r=canvas.image_rect()
                for hand in canvas.detected:
                    x1,y1,x2,y2=hand['bbox'];sx=r.width()/canvas.image.width();sy=r.height()/canvas.image.height()
                    painter.setPen(QtGui.QPen(QtGui.QColor('#52e8a0' if hand['side']=='left' else '#ffaf54'),2))
                    painter.drawRect(QtCore.QRectF(r.x()+x1*sx,r.y()+y1*sy,(x2-x1)*sx,(y2-y1)*sy))
                    painter.drawText(QtCore.QPointF(r.x()+x1*sx,r.y()+y1*sy-5),f"{hand['side']} {hand['score']:.2f}")
        options=QtWidgets.QHBoxLayout();layout.insertLayout(layout.indexOf(self.tabs),options)
        self.automatic=QtWidgets.QCheckBox('自动找手 / 双手重建');options.addWidget(self.automatic)
        options.addWidget(QtWidgets.QLabel('手部检测置信度'))
        self.confidence=QtWidgets.QDoubleSpinBox();self.confidence.setRange(.05,.95);self.confidence.setSingleStep(.05);self.confidence.setValue(host.settings.value('hand/detection_confidence',.4,type=float));options.addWidget(self.confidence)
        self.confidence.setToolTip('只保留达到该阈值的检测框；提高可减少误检，但也可能漏检。实时流下一帧生效，无需重新加载。')
        self.hand_selection=QtWidgets.QComboBox()
        for text,value in (('双手（各最多一只）','both'),('单手：强制右手','right'),('单手：强制左手','left')):self.hand_selection.addItem(text,value)
        self.hand_selection.setCurrentIndex(max(0,self.hand_selection.findData(host.settings.value('hand/detection_selection','both'))))
        self.hand_selection.setToolTip('单手模式忽略 YOLO 左右标签，只按检测框和置信度选一只，按用户指定的左右手重建。右手指人体自身右手。双手模式使用检测器分类；实时下一帧生效。')
        self.confidence.valueChanged.connect(lambda value:host.settings.setValue('hand/detection_confidence',value))
        self.hand_selection.currentIndexChanged.connect(lambda _:host.settings.setValue('hand/detection_selection',self.hand_selection.currentData()))
        self.detect_btn=QtWidgets.QPushButton('只检测手框');self.detect_btn.clicked.connect(lambda:self.submit('auto_detect'));options.addWidget(self.detect_btn)
        options.addWidget(QtWidgets.QLabel('WiLoR 专用 YOLOv8m-Pose · left/right · 21 点（仅使用手框）'));options.addStretch()
        self.automatic.toggled.connect(self.auto_changed)
        self.canvas = DetectionCanvas(selectable=True); self.canvas.setMinimumHeight(350)
        self.canvas.fileDropped.connect(self.load_image); self.canvas.bboxChanged.connect(self.roi_changed)
        self.tabs.addTab(self.canvas,'① 框选单只手')
        self.viewer = viewer_factory(); self.viewer.empty.setText('手部重建后可旋转查看')
        self.viewer.grid.setSize(.5,.5); self.viewer.grid.setSpacing(.025,.025)
        self.tabs.addTab(self.viewer,'② 三维手部')
        self.overlay = OverlayView(); self.overlay.set_layers(True,False,False)
        self.tabs.addTab(self.overlay,'③ 原图投影')
        self.crop = QtWidgets.QLabel('推理后显示实际送入网络的 512×512 裁剪图'); self.crop.setAlignment(QtCore.Qt.AlignCenter)
        self.tabs.addTab(self.crop,'④ 实际裁剪')
        self.sequence=HandSequencePanel(self,viewer_factory)
        self.tabs.addTab(self.sequence,'⑤ 连续序列')
        self.info = QtWidgets.QLabel('请打开图片并框选手部。预加载前会释放空闲的人体模型；SAM3 / 其他计算任务运行时禁止加载。')
        self.info.setWordWrap(True); layout.addWidget(self.info)
        row = QtWidgets.QHBoxLayout(); layout.addLayout(row)
        self.preload = QtWidgets.QPushButton('按上方方案加载手部模型'); self.preload.setToolTip('此按钮加载上方当前选择的方案；需要 TRT 时请先选择加速方案。加载后如需更改配置，先释放模型。'); self.preload.clicked.connect(self.start); row.addWidget(self.preload)
        self.warm = QtWidgets.QPushButton('预热当前手框 ×3'); self.warm.clicked.connect(lambda:self.submit('warmup')); row.addWidget(self.warm)
        self.run = QtWidgets.QPushButton('重建手部'); self.run.setObjectName('primary'); self.run.clicked.connect(lambda:self.submit('run')); row.addWidget(self.run)
        self.stop = QtWidgets.QPushButton('释放 / 取消'); self.stop.clicked.connect(self.shutdown); row.addWidget(self.stop)
        self.skeleton = QtWidgets.QCheckBox('显示手骨架'); self.skeleton.setChecked(True); self.skeleton.toggled.connect(lambda v:self.viewer.set_layers(True,v)); row.addWidget(self.skeleton)
        self.folder = QtWidgets.QPushButton('打开结果目录'); self.folder.clicked.connect(self.open_result); row.addWidget(self.folder)
        self.details = QtWidgets.QPlainTextEdit(); self.details.setReadOnly(True); self.details.setMaximumHeight(110)
        self.details.setMaximumBlockCount(200); layout.addWidget(self.details)
        self.side.currentIndexChanged.connect(self.roi_changed); self.padding.valueChanged.connect(self.roi_changed); self.intrinsics.toggled.connect(self.roi_changed)
        self.live=HandLivePanel(self,canvas_factory,viewer_factory)
        self.tabs.addTab(self.live,'⑥ 相机实时')
        self.organize_layout(layout)
        self.update_controls()

    def one_euro_options(self):
        return dict(pose=self.pose_filter.isChecked(),position=self.position_filter.isChecked(),**{key:widget.value() for key,widget in self.filter_parameters.items()})

    def organize_layout(self,layout):
        """Keep the existing controls/signals; give shared settings a clear hierarchy."""
        def clear(box):
            while box.count():
                item=box.takeAt(0)
                if item.layout():clear(item.layout())
                elif item.widget():item.widget().hide()
        clear(layout)
        layout.setSpacing(10)
        def label(text):
            result=QtWidgets.QLabel(text);result.setWordWrap(True);return result
        def card():
            result=QtWidgets.QFrame();result.setObjectName('card');return result
        self.profile_toggle=QtWidgets.QToolButton();self.profile_toggle.setText('启动方案设置 ▾');self.profile_toggle.setCheckable(True);self.profile_toggle.setChecked(True)
        layout.addWidget(self.profile_toggle)
        self.profile_card=card();profile=QtWidgets.QVBoxLayout(self.profile_card);profile.setContentsMargins(16,12,16,12);profile.setSpacing(10)
        presets=QtWidgets.QHBoxLayout();presets.addWidget(label('快捷方案'));presets.addWidget(self.accelerated_btn);presets.addWidget(self.int8_btn);presets.addWidget(self.baseline_btn);presets.addStretch();profile.addLayout(presets)
        grid=QtWidgets.QGridLayout();grid.setHorizontalSpacing(18);grid.setVerticalSpacing(10)
        grid.addWidget(label('编码器 / 输入'),0,0);grid.addWidget(self.backbone_choice,0,1)
        grid.addWidget(label('手部检测器'),0,2);grid.addWidget(self.detector_choice,0,3)
        grid.addWidget(label('手部中间预测'),1,0);grid.addWidget(self.prediction_count,1,1);grid.addWidget(self.intermediate_compile,1,2,1,2)
        grid.addWidget(label('自动找手范围'),2,0);grid.addWidget(self.hand_selection,2,1);grid.addWidget(label('检测置信度'),2,2);grid.addWidget(self.confidence,2,3)
        grid.setColumnStretch(1,1);grid.setColumnStretch(3,1);profile.addLayout(grid)
        self.pose_filter=QtWidgets.QCheckBox('One Euro 姿态（手指＋手掌朝向）')
        self.position_filter=QtWidgets.QCheckBox('同时滤波位置')
        for widget,key,default in ((self.pose_filter,'pose',False),(self.position_filter,'position',False)):
            widget.setChecked(self.host.settings.value('hand/one_euro_'+key,default,type=bool))
            widget.toggled.connect(lambda value,k=key:self.host.settings.setValue('hand/one_euro_'+k,value))
        filter_row=QtWidgets.QHBoxLayout();filter_row.addWidget(self.pose_filter);filter_row.addWidget(self.position_filter);filter_row.addStretch();profile.addLayout(filter_row)
        self.filter_parameters={};filter_grid=QtWidgets.QGridLayout()
        for column,(key,title,default,low,high) in enumerate((('min_cutoff','最小截止 Hz',1.5,.05,30.),('beta','速度响应 β',.5,0.,20.),('d_cutoff','导数截止 Hz',1.,.05,30.))):
            widget=QtWidgets.QDoubleSpinBox();widget.setDecimals(2);widget.setRange(low,high);widget.setSingleStep(.1)
            widget.setValue(self.host.settings.value('hand/one_euro_'+key,default,type=float))
            widget.valueChanged.connect(lambda value,k=key:self.host.settings.setValue('hand/one_euro_'+k,value))
            self.filter_parameters[key]=widget;filter_grid.addWidget(label(title),0,column*2);filter_grid.addWidget(widget,0,column*2+1)
        profile.addLayout(filter_grid)
        self.pose_filter.setToolTip('仅相机实时流生效；滤波手指潜在系数与 SO(3) 手掌旋转后重新生成网格。不会把滤波反馈进下一帧网络。开关与参数实时生效，无需重载。')
        self.position_filter.setToolTip('单独滤波相机系位置；默认关闭，优先验证姿态。切换配置或手部消失会重置历史。')
        for key,tip in {'min_cutoff':'越低静止越稳，但慢动作延迟越明显。','beta':'越大越跟手，但动态时滤波减弱。手指系数与旋转的单位不同，请实测调整。','d_cutoff':'控制速度估计的平滑程度；通常先保持 1 Hz。'}.items():self.filter_parameters[key].setToolTip(tip+' 仅相机实时流；下一次推理生效。')

        warmup_row=QtWidgets.QHBoxLayout();warmup_row.addWidget(self.startup_warmup_check);warmup_row.addWidget(self.startup_warmup_source,1);warmup_row.addWidget(self.startup_warmup_browse);profile.addLayout(warmup_row)
        self.advanced_toggle=QtWidgets.QToolButton();self.advanced_toggle.setText('高级加速设置 ▸');self.advanced_toggle.setCheckable(True);profile.addWidget(self.advanced_toggle)
        self.advanced_card=QtWidgets.QWidget();advanced=QtWidgets.QGridLayout(self.advanced_card);advanced.setContentsMargins(0,0,0,0);advanced.setHorizontalSpacing(20);advanced.setVerticalSpacing(10)
        for i,w in enumerate((self.decoder_compile,self.mhr_compile,self.mhr_graph,self.feedback_cache,self.intermediate_keypoints,self.final_hand_vertices)):advanced.addWidget(w,i//3,i%3)
        advanced.addWidget(label('编译模式'),2,0);advanced.addWidget(self.compile_mode,2,1,1,2);profile.addWidget(self.advanced_card)
        self.execution_hint.setObjectName('muted');self.execution_hint.setText('手部输入 512 · 解码器 / MHR 为 FP32。中间预测默认 5 次；减少次数可能改变姿态与位置，六层解码及最终预测始终保留。加载后用已有照片自动预热 B1/B2。');profile.addWidget(self.execution_hint)
        layout.addWidget(self.profile_card)
        self.profile_toggle.toggled.connect(lambda on:(self.profile_card.setVisible(on),self.profile_toggle.setText('启动方案设置 ▾' if on else '启动方案设置 ▸')))
        self.advanced_toggle.toggled.connect(lambda on:(self.advanced_card.setVisible(on),self.advanced_toggle.setText('高级加速设置 ▾' if on else '高级加速设置 ▸')))
        model=QtWidgets.QHBoxLayout();self.preload.setText('应用并加载 / 预热');self.preload.setObjectName('primary');model.addWidget(self.preload);model.addWidget(self.stop)
        self.info.setObjectName('muted');model.addWidget(self.info,1);layout.addLayout(model)
        self.workspace_split=QtWidgets.QSplitter(QtCore.Qt.Horizontal);layout.addWidget(self.workspace_split,1)
        self.input_card=card();inputs=QtWidgets.QVBoxLayout(self.input_card);inputs.setContentsMargins(16,16,16,16);inputs.setSpacing(10)
        heading=label('01  图片与手框');inputs.addWidget(heading)
        inputs.addWidget(self.open_btn);inputs.addWidget(self.use_btn)
        instructions=label('框住整只手及少量手腕；也可开启自动找手，同时重建左右手。');instructions.setObjectName('muted');inputs.addWidget(instructions)
        inputs.addWidget(self.automatic)
        for control in (self.side,self.padding,self.confidence):control.setMinimumHeight(32)
        form=QtWidgets.QFormLayout();form.setSpacing(10);form.addRow('手部方向',self.side);form.addRow('裁剪比例',self.padding);inputs.addLayout(form)
        inputs.addWidget(self.intrinsics);inputs.addWidget(self.detect_btn)
        line=QtWidgets.QFrame();line.setFrameShape(QtWidgets.QFrame.HLine);inputs.addWidget(line)
        inputs.addWidget(label('02  预热与重建'));inputs.addWidget(self.warm);inputs.addWidget(self.run);inputs.addWidget(self.skeleton);inputs.addWidget(self.folder);inputs.addStretch()
        self.input_card.setMinimumWidth(255);self.input_card.setMaximumWidth(350);self.input_scroll=QtWidgets.QScrollArea();self.input_scroll.setFrameShape(QtWidgets.QFrame.NoFrame);self.input_scroll.setWidgetResizable(True);self.input_scroll.setMinimumWidth(270);self.input_scroll.setMaximumWidth(365);self.input_scroll.setWidget(self.input_card)
        self.workspace_split.addWidget(self.input_scroll);self.workspace_split.addWidget(self.tabs)
        self.workspace_split.setStretchFactor(0,0);self.workspace_split.setStretchFactor(1,1);self.workspace_split.setSizes([285,1050])
        self.tabs.currentChanged.connect(lambda index:self.input_scroll.setVisible(index<4))
        detail_row=QtWidgets.QHBoxLayout();self.details_toggle=QtWidgets.QToolButton();self.details_toggle.setText('实际配置与任务报告 ▸');self.details_toggle.setCheckable(True);detail_row.addWidget(self.details_toggle);detail_row.addStretch();layout.addLayout(detail_row)
        self.details.setMaximumHeight(160);layout.addWidget(self.details)
        self.details_toggle.toggled.connect(lambda on:(self.details.setVisible(on),self.details_toggle.setText('实际配置与任务报告 ▾' if on else '实际配置与任务报告 ▸')))
        # Widgets detached from the original rows must be explicitly shown again.
        for w in (self.profile_toggle,self.profile_card,self.accelerated_btn,self.int8_btn,self.baseline_btn,self.backbone_choice,self.detector_choice,self.execution_hint,self.advanced_toggle,self.preload,self.stop,self.info,self.tabs,self.input_card,self.open_btn,self.use_btn,self.automatic,self.side,self.padding,self.confidence,self.intrinsics,self.detect_btn,self.warm,self.run,self.skeleton,self.folder):w.show()
        for w in (self.decoder_compile,self.mhr_compile,self.mhr_graph,self.feedback_cache,self.intermediate_keypoints,self.final_hand_vertices,self.compile_mode,self.intermediate_compile,self.prediction_count,self.hand_selection):w.show()
        self.advanced_card.hide();self.details.hide()

    @property
    def active(self): return self.state != 'off'

    def update_controls(self):
        busy = self.state in ('switching','loading','busy','stopping') or getattr(getattr(self,'live',None),'running',False)
        valid = self.source is not None and (self.automatic.isChecked() or self.canvas.bbox is not None)
        self.open_btn.setEnabled(not busy); self.use_btn.setEnabled(not busy)
        self.canvas.selection_enabled = not busy
        for widget in (self.side,self.padding,self.intrinsics): widget.setEnabled(not busy)
        self.preload.setEnabled(self.state == 'off')
        for widget in (self.accelerated_btn,self.int8_btn,self.baseline_btn,self.backbone_choice,self.detector_choice,self.prediction_count,self.intermediate_compile,self.decoder_compile,self.mhr_compile,self.mhr_graph,self.compile_mode,self.feedback_cache,self.intermediate_keypoints,self.final_hand_vertices,self.startup_warmup_check,self.startup_warmup_source,self.startup_warmup_browse):widget.setEnabled(self.state=='off')
        self.warm.setEnabled(self.state == 'ready' and valid and not busy); self.run.setEnabled(self.state == 'ready' and valid and not busy)
        self.stop.setEnabled(self.active); self.folder.setEnabled(self.result_dir is not None)
        self.automatic.setEnabled(not busy)
        detection_editable=not busy or getattr(getattr(self,'live',None),'running',False)
        self.confidence.setEnabled(detection_editable);self.hand_selection.setEnabled(detection_editable)
        self.detect_btn.setEnabled(self.state=='ready' and self.source is not None and not busy)
        self.side.setEnabled(not busy and not self.automatic.isChecked())
        self.run.setText('自动重建左右手' if self.automatic.isChecked() else '重建手部')
        if hasattr(self,'sequence'):self.sequence.update_controls()
        if hasattr(self,'live'):self.live.refresh()

    def auto_changed(self,enabled):
        if enabled:self.manual_bbox=self.canvas.bbox;self.canvas.bbox=None
        else:self.canvas.bbox=getattr(self,'manual_bbox',None);self.canvas.detected=[]
        self.canvas.update();self.padding.setValue(1.1 if enabled else .9);self.roi_changed()

    def open_image(self):
        name,_ = QtWidgets.QFileDialog.getOpenFileName(self,'打开手部照片','','图片 (*.png *.jpg *.jpeg *.bmp)')
        if name:self.load_image(name)

    def load_image(self, name, metadata=None):
        if self.state in ('busy','loading','stopping','switching') or self.live.running:return
        self.sequence.timer.stop()
        image = QtGui.QImage(str(name))
        if image.isNull(): self.info.setText('图片无法打开'); return
        self.source = Path(name); self.metadata = metadata
        self.metadata_error = None
        if metadata is None:
            camera = self.source.parent/'camera.json'
            manifest = self.source.parent.parent/'manifest.json'
            try:
                if camera.is_file():self.metadata = json.loads(camera.read_text())
                elif manifest.is_file():
                    value = json.loads(manifest.read_text()).get('profile',{}).get('intrinsics')
                    if value:self.metadata = {'intrinsics':value}
            except (ValueError,OSError) as exc:self.metadata_error = str(exc)
        self.intrinsics.setChecked(self.metadata is not None)
        self.canvas.detected=[];self.manual_bbox=None
        self.canvas.set_image(image); self.viewer.clear_mesh(); self.overlay.clear_frame(image)
        self.crop.clear(); self.crop.setText('推理后显示实际裁剪图'); self.result_dir = None
        self.tabs.setCurrentIndex(0); self.roi_changed()

    def use_photo(self):
        if self.host.source_path:
            self.load_image(self.host.source_path,self.host.capture_metadata)
        else:self.info.setText('第一页还没有照片，请先拍摄或打开图片。')

    def roi_changed(self):
        self.info.setText('手框 / 参数已更新；已有预览属于上次结果。点击“重建手部”生成新结果。' if self.result_dir else '请框住整只手，然后预加载、预热并重建。')
        self.update_controls()

    def blocking_reason(self):
        h = self.host
        if h.inference_busy or h.model_state in ('loading','stopping'):return '人体模型正在计算 / 加载 / 释放'
        if h.sam3_blocks_models():return 'SAM3 / 视频任务正在运行'
        lab = h.lab
        if lab.client.inflight is not None or lab.client.pending is not None or lab.timer.isActive():return 'MHR 实验室正在计算'
        r = h.rgbd
        if r.mode is not None or r.batch.active or r.process.state() != QtCore.QProcess.NotRunning:return 'RGB-D 任务正在运行'
        if getattr(h.live,'running',False):return '请先停止实时重建'
        return ''

    def set_execution_preset(self,accelerated):
        if self.state!='off':return
        self.backbone_choice.setCurrentIndex(1 if accelerated else 0)
        self.detector_choice.setCurrentIndex(1 if accelerated else 0)
        for widget in (self.decoder_compile,self.mhr_compile,self.mhr_graph,self.feedback_cache,self.intermediate_keypoints,self.final_hand_vertices):widget.setChecked(accelerated)
        self.intermediate_compile.setChecked(accelerated)
        self.prediction_count.setCurrentIndex(self.prediction_count.findData(5))
        self.compile_mode.setCurrentIndex(0)

    def set_int8_preset(self):
        if self.state!='off':return
        self.set_execution_preset(True)
        self.backbone_choice.setCurrentIndex(self.backbone_choice.findData('trt_int8'))

    def choose_warmup_sample(self):
        name,_=QtWidgets.QFileDialog.getOpenFileName(self,'选择离线预热样本',self.startup_warmup_source.text(),'图片 (*.jpg *.jpeg *.png *.bmp)')
        if name:self.startup_warmup_source.setText(name)

    def execution_config(self):
        from studio.hand.hand_config import HandExecution
        return HandExecution(backbone=self.backbone_choice.currentData(),detector=self.detector_choice.currentData(),decoder_compile=self.decoder_compile.isChecked(),mhr_compile=self.mhr_compile.isChecked(),mhr_graph=self.mhr_graph.isChecked(),compile_mode=self.compile_mode.currentData(),feedback_cache=self.feedback_cache.isChecked(),intermediate_keypoints=self.intermediate_keypoints.isChecked(),final_hand_vertices=self.final_hand_vertices.isChecked(),intermediate_compile=self.intermediate_compile.isChecked(),intermediate_predictions=self.prediction_count.currentData()).to_dict()

    def start(self):
        reason = self.blocking_reason()
        if reason:self.info.setText('暂不能加载手部模型：'+reason);return
        self.host.pending_launch = False
        if self.host.model_state == 'ready': self.host.release_model()
        self.host.lab.shutdown()
        self.state = 'switching'; self.info.setText('正在释放空闲人体 / MHR 模型，随后独占 GPU 加载手部模型…')
        self.switch_deadline = QtCore.QDeadlineTimer(15000)
        self.switch_timer.start(); self.update_controls()

    def wait_body(self):
        if self.host.process.state() != QtCore.QProcess.NotRunning or self.host.lab.client.process.state() != QtCore.QProcess.NotRunning:
            if self.switch_deadline.hasExpired():
                self.switch_timer.stop(); self.state='off'; self.info.setText('原模型尚未退出，请释放后重试。'); self.update_controls()
            return
        self.switch_timer.stop(); self.state='loading'; self.buffer=''; self.update_controls()
        self.startup_warmup=None
        logs = ROOT/'outputs/sessions'; logs.mkdir(parents=True,exist_ok=True)
        self.log_file = logs/('hand-'+uuid.uuid4().hex+'.log')
        env = QtCore.QProcessEnvironment.systemEnvironment()
        env.remove('QT_QPA_PLATFORM_PLUGIN_PATH'); env.remove('QT_QPA_FONTDIR')
        self.process.setProcessEnvironment(env); self.process.setWorkingDirectory(str(ROOT))
        self.scope = 'hand-studio-'+uuid.uuid4().hex[:10]
        arguments=['--execution',json.dumps(self.execution_config())]
        enabled=self.startup_warmup_check.isChecked();source=self.startup_warmup_source.text().strip()
        self.host.settings.setValue('hand/startup_warmup',enabled);self.host.settings.setValue('hand/warmup_image',source)
        if enabled:
            if source:arguments+=['--startup-warmup-image',source]
        else:arguments+=['--no-startup-warmup']
        self.process.start('systemd-run',['--user','--scope','--quiet','--unit='+self.scope,'-p','MemoryMax=7G','-p','MemorySwapMax=2G','/usr/bin/python3',str(ROOT/'launch_hand.py'),*arguments])

    def submit(self, kind):
        if self.state != 'ready' or self.live.running:return
        try:
            auto = self.automatic.isChecked() or kind=='auto_detect'
            bbox = None if auto else validate_bbox(self.canvas.bbox,self.canvas.image.width(),self.canvas.image.height())
            if auto and kind in ('warmup','run'):kind='auto_'+kind
            intr,origin = select_camera_intrinsics(self.intrinsics.isChecked(),self.canvas.image.width(),self.canvas.image.height(),self.metadata,getattr(self,'metadata_error',None))
            matrix = [[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1]] if intr else None
            stamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
            directory = ROOT/'outputs/hands'/stamp; directory.mkdir(parents=True)
            # Freeze the displayed image, so an external file edit cannot change this job.
            snapshot = directory/'source.png'
            if not self.canvas.image.save(str(snapshot)):raise ValueError('无法保存任务输入')
            self.job = uuid.uuid4().hex
            command = dict(type=kind,job=self.job,image=str(snapshot),hand_bbox=bbox.tolist() if bbox is not None else None,confidence=self.confidence.value(),hand_selection=self.hand_selection.currentData(),side=self.side.currentData(),padding=self.padding.value(),camera_matrix=matrix,camera_source=origin['source'],output=str(directory))
            self.state='busy'; self.info.setText('正在预热…' if kind=='warmup' else '正在进行独立手部重建…'); self.update_controls()
            self.process.write((json.dumps(command)+'\n').encode())
        except Exception as exc:self.info.setText(str(exc))

    def read_output(self):
        text = bytes(self.process.readAllStandardOutput()).decode('utf-8',errors='replace')
        if self.log_file:
            with self.log_file.open('a') as stream:stream.write(text)
        self.buffer += text
        while '\n' in self.buffer:
            line,self.buffer = self.buffer.split('\n',1)
            if not line.startswith('@@HAND '):continue
            try:event=json.loads(line[7:])
            except ValueError:continue
            if self.live.handle_event(event):continue
            kind=event['type']
            if kind=='stage': self.info.setText(event['message'])
            elif kind=='ready':
                self.state='ready'
                self.startup_warmup=event.get('startup_warmup',{'status':'disabled'})
                actual=event.get('execution',{})
                encoder={'native':'原生 PyTorch FP16','trt_fp16':'TensorRT FP16','trt_int8':'TensorRT INT8/FP16 混合精度'}.get(actual.get('backbone'),'未知')
                detector={'native':'原生 PyTorch','trt_fp16':'TensorRT FP16'}.get(actual.get('detector'),'未知')
                warm=self.startup_warmup
                ready=warm['status']=='ready'
                suffix=(f"离线预热完成 B1+B2（{warm['seconds']:.1f}s），可直接开始实时重建。" if ready else '尚未完成离线预热：'+warm.get('message','已关闭自动预热，可手动补充。'))
                predictions=actual.get('intermediate_predictions',5)
                compiled='开' if actual.get('intermediate_compile') else '关'
                self.info.setText(f"已加载（{event['load_seconds']:.1f}s）· 编码器：{encoder} / 512 · 检测器：{detector} · 中间预测 {predictions} 次 / 回调编译 {compiled} · "+suffix)
                self.details.setPlainText('实际启动配置与离线预热：\n'+json.dumps(dict(execution=event.get('execution',{}),startup_warmup=warm),ensure_ascii=False,indent=2))
                self.live.status.setText('离线预热已完成，无需在线摆手预热；打开相机后直接开始实时重建。' if ready else suffix)
            elif kind=='frame' and event.get('job')==self.job:
                try:self.sequence.on_frame(event)
                except Exception as exc:self.info.setText('帧预览失败（任务继续）：'+str(exc))
            elif kind=='sequence_done' and event.get('job')==self.job:
                self.state='ready';self.job=None;self.sequence.completed(event);self.info.setText('离线逐帧重建完成，可在连续序列页播放 / 拖动时间轴。')
            elif kind=='auto_result' and event.get('job')==self.job:
                self.state='ready';self.job=None
                try:self.show_auto_result(event)
                except Exception as exc:self.info.setText('自动手部结果读取失败：'+str(exc))
            elif kind in ('warmed','result','error') and event.get('job')==self.job:
                self.state='ready'
                if kind=='error':self.info.setText('手部任务失败：'+event['message'])
                elif kind=='warmed':self.info.setText(f"预热完成 · 最后一次 {event['report']['inference_ms']:.1f} ms（含裁剪 / CPU→GPU，未含导出）")
                else:
                    try:self.show_result(Path(event['output']),event['report'])
                    except Exception as exc:self.info.setText('结果读取失败：'+str(exc))
                self.job=None
            elif kind=='fatal':self.info.setText('手部模型加载失败：'+event['message']);self.details.appendPlainText(event['message'])
            self.update_controls()

    def show_result(self, directory, report):
        with np.load(directory/'hand.npz',allow_pickle=False) as data:
            self.viewer.set_mesh(data['vertices'],data['faces'])
            self.viewer.set_pose(data['joints'],data['edges']); self.viewer.set_layers(True,self.skeleton.isChecked())
            image=QtGui.QImage(str(directory/'input.png'))
            self.overlay.set_frame(image,data['vertices'],data['camera_translation'],data['faces'],data['camera_matrix'],0)
        self.crop.setPixmap(QtGui.QPixmap(str(directory/'crop.png')).scaled(512,512,QtCore.Qt.KeepAspectRatio,QtCore.Qt.SmoothTransformation))
        self.result_dir=directory; self.tabs.setCurrentIndex(1)
        self.info.setText(f"{report['side']} · 推理 {report['inference_ms']:.1f} ms · {report['vertices']} 顶点 / {report['joint_nodes']} 骨架节点 · 投影校验 {report['projection_roundtrip_max_px']:.4f} px。结果含 hand.ply / hand.npz / raw_hand.npz。")
        self.details.setPlainText(json.dumps(report,ensure_ascii=False,indent=2));self.update_controls()

    def display_scene(self,image,data,report,viewer=None,overlay=None,continuous=False,mode=None):
        viewer=viewer if viewer is not None else self.viewer
        overlay=overlay if overlay is not None else self.overlay
        if mode!='overlay' and len(data['vertices']):
            first=not continuous or not getattr(viewer,'_hand_active',False)
            viewer.set_mesh(data['vertices'],data['faces'],continuous=not first)
            if first:
                viewer.stream_anchor=viewer.anchor.copy()
                view_vertices=np.asarray(data['vertices'])[:,[0,2,1]].copy();view_vertices[:,2]*=-1
                center=(view_vertices.min(0)+view_vertices.max(0))/2-viewer.anchor
                viewer.opts['center']=QtGui.QVector3D(*map(float,center))
                # GLViewWidget's fov is horizontal. Fit a bounding sphere to
                # both axes, including depth, with room around the fingers.
                radius=float(np.linalg.norm(np.ptp(view_vertices,axis=0)/2))
                aspect=max(viewer.width(),1)/max(viewer.height(),1)
                half_horizontal=np.radians(viewer.opts['fov'])/2
                half_vertical=np.arctan(np.tan(half_horizontal)/aspect)
                distance=radius/max(float(np.sin(min(half_horizontal,half_vertical))),1e-3)*1.3
                viewer.setCameraPosition(distance=max(distance,.5),elevation=8,azimuth=-90)
            viewer._hand_active=True;viewer.set_pose(data['joints'],data['edges']);viewer.set_layers(True,self.skeleton.isChecked())
        elif mode!='overlay':
            viewer.clear_mesh();viewer._hand_active=False;viewer.empty.setText('当前帧未检测到手部');viewer.empty.show()
        if mode!='3d':
            if len(data['vertices']):overlay.set_frame(image,data['vertices'],np.zeros(3),data['faces'],data['camera_matrix'],0)
            else:overlay.clear_frame(image)

    def show_auto_result(self,event):
        directory=Path(event['output']);report=event['report'];self.result_dir=directory
        self.canvas.detected=report['hands'];self.canvas.update()
        image=QtGui.QImage(str(directory/'input.png'))
        with np.load(directory/'hands.npz',allow_pickle=False) as data:self.display_scene(image,data,report)
        crops=[]
        for side in ('left','right'):
            file=directory/(side+'-crop.png')
            if file.exists():crops.append(QtGui.QPixmap(str(file)).scaled(320,320,QtCore.Qt.KeepAspectRatio,QtCore.Qt.SmoothTransformation))
        if crops:
            joined=QtGui.QPixmap(sum(p.width() for p in crops),max(p.height() for p in crops));joined.fill(QtGui.QColor('#111827'));p=QtGui.QPainter(joined);x=0
            for crop in crops:p.drawPixmap(x,0,crop);x+=crop.width()
            p.end();self.crop.setPixmap(joined)
        else:self.crop.setText('只检测，或当前没有可用手框；未进行三维重建')
        self.tabs.setCurrentIndex(0 if event['kind']=='auto_detect' or not report['hand_count'] else 1)
        self.details.setPlainText(json.dumps(report,ensure_ascii=False,indent=2))
        self.info.setText(f"检测 {report['hand_count']} 只手 · 检测 {report['detect_ms']:.1f} ms · 整批重建 {report['reconstruct_ms']:.1f} ms · 缺失：{','.join(report['missing']) or '无'}。"+('预热完成。' if event['kind']=='auto_warmup' else ''))
        self.update_controls()

    def process_error(self, error):
        if error == QtCore.QProcess.FailedToStart:
            self.state='off';self.info.setText('手部进程无法启动：'+self.process.errorString());self.update_controls()

    def finished(self, code, status):
        self.read_output()
        cancelled=self.state=='stopping';self.state='off';self.job=None;self.startup_warmup=None
        self.live.abort('手部模型进程已退出，实时重建已停止')
        self.info.setText('手部模型已释放，可返回身体页预加载。' if cancelled else '手部进程已退出；如非主动释放，请查看日志：'+str(self.log_file))
        self.update_controls()

    def shutdown(self):
        self.live.end('正在释放手部模型')
        self.sequence.timer.stop()
        self.switch_timer.stop()
        if self.process.state()!=QtCore.QProcess.NotRunning:
            self.state='stopping'
            try:
                subprocess.run(['systemctl','--user','stop',self.scope+'.scope'],timeout=3,capture_output=True,check=False)
            except (OSError,subprocess.TimeoutExpired):
                subprocess.run(['systemctl','--user','kill','--signal=SIGKILL',self.scope+'.scope'],timeout=3,capture_output=True,check=False)
            self.process.terminate()
            if not self.process.waitForFinished(2500):self.process.kill();self.process.waitForFinished(3000)
        self.live.abort('手部模型已释放')
        self.state='off';self.job=None;self.startup_warmup=None;self.info.setText('手部模型已释放，可返回身体页预加载。');self.update_controls()

    def open_result(self):
        if self.result_dir:QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(self.result_dir)))
