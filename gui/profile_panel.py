"""Launch settings UI, with presets and explicit application by the main window."""
from dataclasses import replace
import json
from pathlib import Path
import uuid
from PyQt5 import QtCore, QtWidgets
from launch_config import LaunchConfig, PRESETS, ROOT

class ProfilePanel(QtWidgets.QGroupBox):
    changed=QtCore.pyqtSignal()

    def __init__(self,config,parent=None):
        super().__init__('启动方案 · 先选择，再应用并预热',parent)
        self.block=False
        layout=QtWidgets.QVBoxLayout(self)
        top=QtWidgets.QHBoxLayout()
        self.presets=QtWidgets.QComboBox()
        self.presets.addItem('自定义',None)
        self.reload_profiles()
        top.addWidget(self.presets,1)
        for label,slot in [('保存方案',self.save_profile),('导入',self.import_profile),('导出',self.export_profile),('恢复默认',lambda:self.set_config(PRESETS['演示最快']))]:
            button=QtWidgets.QPushButton(label);button.clicked.connect(slot);top.addWidget(button)
        layout.addLayout(top)
        grid=QtWidgets.QGridLayout();layout.addLayout(grid)
        self.backend=QtWidgets.QComboBox();self.backend.addItem('Fast','fast');self.backend.addItem('标准版（ComfyUI）','standard')
        self.hand=QtWidgets.QCheckBox('身体＋手部精修')
        self.hand.setToolTip('不勾选为身体快速模式；仍输出完整网格')
        self.size=QtWidgets.QComboBox()
        for value in [384,512]:self.size.addItem(str(value),value)
        self.backbone=self.execution_combo();self.detector=self.execution_combo()
        self.decoder=QtWidgets.QCheckBox('解码器编译')
        for column,(label,widget) in enumerate([('实现',self.backend),('模式',self.hand),('网络分辨率',self.size)]):
            grid.addWidget(QtWidgets.QLabel(label),0,column*2);grid.addWidget(widget,0,column*2+1)
        for column,(label,widget) in enumerate([('骨干执行',self.backbone),('检测器执行',self.detector),('',self.decoder)]):
            grid.addWidget(QtWidgets.QLabel(label),1,column*2);grid.addWidget(widget,1,column*2+1)
        feedback_row=QtWidgets.QHBoxLayout()
        feedback_row.addWidget(QtWidgets.QLabel('身体中间预测'))
        self.feedback=QtWidgets.QComboBox()
        for label,value in [('3次（原方案：0,1,2）',3),('2次（实验：0,2）',2),('1次（实验：0）',1),('0次（实验：仅最终预测）',0)]:self.feedback.addItem(label,value)
        self.feedback.setToolTip('仅 Fast 身体模式。所有 Transformer 层仍运行，最终预测始终保留；减少反馈可能损失精度，修改后须重新预热。')
        feedback_row.addWidget(self.feedback,1);layout.addLayout(feedback_row)
        smpl_row=QtWidgets.QHBoxLayout()
        self.smpl=QtWidgets.QCheckBox('SMPL 转换（24 关节）')
        self.smoothing=QtWidgets.QCheckBox('MLP 平滑（延后约 2 帧）')
        self.smoothing.setToolTip('仅平滑 SMPL 关节位置，5 帧窗口预测中心帧；姿态参数和旋转仍为原始值，不直接用于机器人控制。')
        self.euro=QtWidgets.QCheckBox('One Euro（低延迟位置滤波）')
        smpl_row.addWidget(self.smpl);smpl_row.addWidget(self.smoothing);smpl_row.addWidget(self.euro);smpl_row.addStretch()
        layout.addLayout(smpl_row)
        euro_row=QtWidgets.QHBoxLayout()
        self.euro_min=self.filter_spin(.01,30.,1.5);self.euro_beta=self.filter_spin(0.,100.,5.);self.euro_deriv=self.filter_spin(.01,30.,1.)
        for label,widget,tip in [('最小截止 Hz',self.euro_min,'越小静止时越稳，但跟随滞后越大'),('速度响应 β',self.euro_beta,'越大运动时响应越快，动态平滑越弱；单位 Hz/(m/s)'),('导数截止 Hz',self.euro_deriv,'速度估计的低通截止频率，通常保持 1 Hz')]:
            euro_row.addWidget(QtWidgets.QLabel(label));euro_row.addWidget(widget);widget.setToolTip(tip);widget.valueChanged.connect(self.edited)
        self.euro_target=QtWidgets.QComboBox()
        for label,value in [('MHR 骨架','mhr'),('SMPL 骨架','smpl'),('MHR＋SMPL','both')]:self.euro_target.addItem(label,value)
        self.euro_target.setToolTip('MHR：70 个关键点＋127 个关节位置，无需 SMPL 模型；两者使用独立滤波状态。网格、旋转、相机平移和原始数据不变。')
        self.euro_target.currentIndexChanged.connect(self.edited)
        euro_row.insertWidget(0,self.euro_target)
        euro_row.insertWidget(0,QtWidgets.QLabel('滤波目标'))
        layout.addLayout(euro_row)
        smpl_path_row=QtWidgets.QHBoxLayout()
        smpl_path_row.addWidget(QtWidgets.QLabel('SMPL 模型'))
        self.smpl_path=QtWidgets.QLineEdit();self.smpl_path.setPlaceholderText('留空：assets/smpl/SMPL_NEUTRAL.pkl 或 .npz')
        smpl_path_row.addWidget(self.smpl_path,1)
        self.smpl_browse=QtWidgets.QPushButton('选择…');self.smpl_browse.clicked.connect(self.choose_smpl)
        smpl_path_row.addWidget(self.smpl_browse);layout.addLayout(smpl_path_row)
        advanced_button=QtWidgets.QToolButton();advanced_button.setText('高级设置 ▸');advanced_button.setCheckable(True)
        layout.addWidget(advanced_button)
        self.advanced=QtWidgets.QWidget();advanced=QtWidgets.QHBoxLayout(self.advanced)
        advanced.setContentsMargins(0,0,0,0)
        self.mhr=QtWidgets.QCheckBox('MHR 编译');self.interm=QtWidgets.QCheckBox('中间预测编译')
        self.warmup=QtWidgets.QSpinBox();self.warmup.setRange(1,30)
        self.layout_combo=QtWidgets.QComboBox()
        for label,value in [('上传＋相机','both'),('上传示例','sample'),('相机尺寸','camera')]:self.layout_combo.addItem(label,value)
        self.layout_combo.setToolTip('使用内置示例预热；相机尺寸会缩放到 1280×800，不读取相机，无需连接设备。')
        for widget in [self.mhr,self.interm,QtWidgets.QLabel('预热次数/布局'),self.warmup,self.layout_combo]:advanced.addWidget(widget)
        advanced.addStretch();layout.addWidget(self.advanced);self.advanced.hide()
        advanced_button.toggled.connect(self.advanced.setVisible)
        self.note=QtWidgets.QLabel('身体模式不做独立手部精修；384 可能损失精度。Fast 固定 FP16 骨干、FP32 解码器，PyTorch 骨干不编译。')
        self.note.setWordWrap(True);self.note.setObjectName('muted');layout.addWidget(self.note)
        self.summary=QtWidgets.QLabel();self.summary.setWordWrap(True);layout.addWidget(self.summary)
        self.presets.currentIndexChanged.connect(self.pick_preset)
        self.backend.currentIndexChanged.connect(self.backend_changed)
        for widget in [self.hand,self.decoder,self.mhr,self.interm,self.smpl,self.smoothing,self.euro]:widget.toggled.connect(self.edited)
        for widget in [self.size,self.backbone,self.detector,self.layout_combo,self.feedback]:widget.currentIndexChanged.connect(self.edited)
        self.warmup.valueChanged.connect(self.edited)
        self.smpl_path.textChanged.connect(self.edited)
        self.set_config(config)

    def filter_spin(self,low,high,value):
        spin=QtWidgets.QDoubleSpinBox();spin.setDecimals(2);spin.setRange(low,high);spin.setSingleStep(.1);spin.setValue(value);return spin

    def choose_smpl(self):
        path,_=QtWidgets.QFileDialog.getOpenFileName(self,'选择官方 neutral SMPL 模型',str(ROOT/'assets/smpl'),'SMPL (*.pkl *.npz)')
        if path:self.smpl_path.setText(path)

    def execution_combo(self):
        combo=QtWidgets.QComboBox();combo.addItem('TensorRT',True);combo.addItem('PyTorch',False);return combo

    def reload_profiles(self):
        while self.presets.count()>1:self.presets.removeItem(1)
        for name,config in PRESETS.items():self.presets.addItem(name,config.to_dict())
        for path in sorted((ROOT/'profiles').glob('*.json')):
            try:
                data=json.loads(path.read_text());config=LaunchConfig.from_dict(data['config'])
                self.presets.addItem('自存 · '+data['name'],config.to_dict())
            except (ValueError,KeyError,OSError):continue

    def config(self):
        return LaunchConfig(backend=self.backend.currentData(),mode='full' if self.hand.isChecked() else 'body',
            image_size=self.size.currentData(),trt_backbone=self.backbone.currentData(),trt_detector=self.detector.currentData(),
            decoder_compile=self.decoder.isChecked(),mhr_compile=self.mhr.isChecked(),interm_compile=self.interm.isChecked(),
            warmup_runs=self.warmup.value(),warmup_layout=self.layout_combo.currentData(),
            smpl_enabled=self.smpl.isChecked(),smpl_smoothing=self.smoothing.isChecked(),smpl_model_path=self.smpl_path.text().strip(),body_interm_predictions=self.feedback.currentData(),one_euro_enabled=self.euro.isChecked(),one_euro_target=self.euro_target.currentData(),one_euro_min_cutoff=self.euro_min.value(),one_euro_beta=self.euro_beta.value(),one_euro_d_cutoff=self.euro_deriv.value()).validate()

    def set_config(self,config):
        self.block=True
        for widget,value in [(self.backend,config.backend),(self.size,config.image_size),
                             (self.backbone,config.trt_backbone),(self.detector,config.trt_detector),(self.layout_combo,config.warmup_layout),(self.feedback,config.body_interm_predictions)]:
            widget.setCurrentIndex(widget.findData(value))
        for widget,value in [(self.hand,config.mode=='full'),(self.decoder,config.decoder_compile),
                             (self.mhr,config.mhr_compile),(self.interm,config.interm_compile),(self.smpl,config.smpl_enabled),(self.smoothing,config.smpl_smoothing),(self.euro,config.one_euro_enabled)]:widget.setChecked(value)
        self.euro_target.setCurrentIndex(self.euro_target.findData(config.one_euro_target))
        self.euro_min.setValue(config.one_euro_min_cutoff);self.euro_beta.setValue(config.one_euro_beta);self.euro_deriv.setValue(config.one_euro_d_cutoff)
        self.smpl_path.setText(config.smpl_model_path)
        self.warmup.setValue(config.warmup_runs)
        self.presets.setCurrentIndex(0)
        for index in range(1,self.presets.count()):
            if self.presets.itemData(index)==config.to_dict():self.presets.setCurrentIndex(index);break
        self.block=False;self.refresh();self.changed.emit()

    def backend_changed(self):
        if self.block:return
        config=PRESETS['标准版身体'] if self.backend.currentData()=='standard' else PRESETS['演示最快']
        self.set_config(config)

    def pick_preset(self):
        if not self.block and self.presets.currentData() is not None:
            index=self.presets.currentIndex()
            self.set_config(LaunchConfig.from_dict(self.presets.currentData()))
            self.presets.blockSignals(True);self.presets.setCurrentIndex(index);self.presets.blockSignals(False)

    def edited(self):
        if self.block:return
        if self.hand.isChecked() and self.feedback.currentData()!=3:
            self.feedback.blockSignals(True);self.feedback.setCurrentIndex(self.feedback.findData(3));self.feedback.blockSignals(False)
        if not self.smpl.isChecked():
            self.smoothing.blockSignals(True);self.smoothing.setChecked(False);self.smoothing.blockSignals(False)
            self.euro_target.blockSignals(True);self.euro_target.setCurrentIndex(self.euro_target.findData('mhr'));self.euro_target.blockSignals(False)
        self.presets.blockSignals(True);self.presets.setCurrentIndex(0);self.presets.blockSignals(False)
        self.refresh();self.changed.emit()

    def refresh(self):
        fast=self.backend.currentData()=='fast'
        self.smoothing.setEnabled(self.smpl.isChecked())
        for i in (1,2):self.euro_target.model().item(i).setEnabled(self.smpl.isChecked())
        self.euro_target.setEnabled(self.euro.isChecked())
        self.feedback.setEnabled(fast and not self.hand.isChecked())
        for widget in (self.euro_min,self.euro_beta,self.euro_deriv):widget.setEnabled(self.euro.isChecked())
        self.smpl_path.setEnabled(self.smpl.isChecked());self.smpl_browse.setEnabled(self.smpl.isChecked())
        for widget in [self.size,self.backbone,self.detector,self.decoder,self.mhr,self.interm]:widget.setEnabled(fast)
        try:
            self.summary.setText(self.config().summary())
            self.summary.setStyleSheet('')
        except ValueError as exc:
            self.summary.setText('不可启动：'+str(exc));self.summary.setStyleSheet('color:#fca5a5')

    def save_profile(self):
        try:config=self.config()
        except ValueError as exc:QtWidgets.QMessageBox.warning(self,'方案无效',str(exc));return
        name,ok=QtWidgets.QInputDialog.getText(self,'保存启动方案','方案名称：')
        if not ok or not name.strip():return
        directory=ROOT/'profiles';directory.mkdir(exist_ok=True)
        (directory/(uuid.uuid4().hex+'.json')).write_text(json.dumps({'name':name.strip()[:80],'config':config.to_dict()},ensure_ascii=False,indent=2))
        self.block=True;self.reload_profiles();self.block=False;self.set_config(config)

    def import_profile(self):
        path,_=QtWidgets.QFileDialog.getOpenFileName(self,'导入启动方案','','JSON (*.json)')
        if not path:return
        try:
            data=json.loads(Path(path).read_text());self.set_config(LaunchConfig.from_dict(data.get('config',data)))
        except (ValueError,OSError,AttributeError) as exc:QtWidgets.QMessageBox.warning(self,'导入失败',str(exc))

    def export_profile(self):
        try:config=self.config()
        except ValueError as exc:QtWidgets.QMessageBox.warning(self,'方案无效',str(exc));return
        path,_=QtWidgets.QFileDialog.getSaveFileName(self,'导出启动方案','studio-profile.json','JSON (*.json)')
        if path:config.save(path)
