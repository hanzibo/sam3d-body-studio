# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Read-only inspection of frozen RGB-D run records; never loads model weights."""
import copy
import hashlib
import json
import math
from pathlib import Path
from PyQt5 import QtCore, QtGui, QtWidgets

MISSING = '未记录'
CONFIG_FIELDS = {
    'schema_version':'方案格式版本', 'backend':'实现', 'mode':'身体 / 完整模式',
    'image_size':'网络分辨率', 'trt_backbone':'骨干 TensorRT', 'trt_detector':'检测器 TensorRT',
    'decoder_compile':'解码器编译', 'mhr_compile':'MHR 编译', 'interm_compile':'中间预测编译',
    'body_interm_predictions':'身体中间预测次数', 'warmup_runs':'每种布局预热次数',
    'warmup_layout':'预热布局', 'smpl_enabled':'SMPL 转换', 'smpl_smoothing':'MLP 平滑',
    'smpl_model_path':'SMPL 模型路径', 'one_euro_enabled':'One Euro 滤波',
    'one_euro_target':'One Euro 目标', 'one_euro_min_cutoff':'最小截止 Hz',
    'one_euro_beta':'速度响应 β', 'one_euro_d_cutoff':'导数截止 Hz',
}
EXEC_FIELDS = {
    'backend_id':'后端标识', 'implementation':'实现来源', 'config_id':'配置 ID',
    'backbone_execution':'骨干实际执行', 'detector':'检测器实际执行',
    'compile_requested':'请求编译状态', 'compile_actual':'实际编译状态',
    'checkpoint_dtype':'权重精度', 'body_interm_layers':'实际中间预测层',
    'body_decoder_depth':'解码器深度', 'observed_backbone_shapes':'实际骨干输入尺寸',
    'model_reused':'复用驻留模型', 'warmup':'预热记录', 'benchmark':'重复计时记录',
    'input_hw':'输入图像尺寸 H×W', 'inference_type':'推理类型', 'person_count':'检测人数',
    'bbox_xyxy':'检测框 xyxy', 'selection_roi':'人物选择 ROI',
    'camera_intrinsics_used':'使用相机内参', 'camera_intrinsics_origin':'内参来源',
    'camera':'相机信息', 'smpl_info':'SMPL 实际状态', 'mhr_info':'MHR 实际状态',
    'gpu':'GPU', 'torch':'PyTorch 版本', 'repo_commit':'源码版本',
    'checkpoint_verification':'模型文件校验', 'engine_verification':'引擎文件校验',
    'all_outputs_finite':'输出均为有限值', 'validation_scope':'验证范围',
}
METRICS = {
    'mae_mm':'深度 MAE mm ↓', 'p90_mm':'深度 P90 mm ↓',
    'silhouette_iou':'轮廓 IoU ↑', 'model_coverage_fraction':'覆盖率 ↑',
    'point_surface_mae_mm':'点面 MAE mm ↓', 'point_surface_p90_mm':'点面 P90 mm ↓',
    'signed_mean_mm':'深度有符号均值 mm', 'valid_depth_fraction':'有效深度比例',
    'within_10mm':'深度误差 ≤10 mm 比例 ↑', 'within_20mm':'深度误差 ≤20 mm 比例 ↑',
    'within_50mm':'深度误差 ≤50 mm 比例 ↑', 'point_surface_count':'点面距离采样点数',
    'mask_pixels':'人体掩膜像素数', 'overlap_pixels':'深度重叠像素数',
}


def pretty(value):
    if value is None:return '空值'
    if isinstance(value,bool):return '开启' if value else '关闭'
    if isinstance(value,float):return f'{value:.6g}'
    if value=='':return '留空'
    if isinstance(value,(dict,list)):return json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)
    return str(value)


def read_json(path, errors):
    if not path.is_file():return {}
    try:
        data=json.loads(path.read_text())
        if not isinstance(data,dict):raise ValueError('应为 JSON 对象')
        return data
    except (OSError,ValueError) as exc:
        errors.append(f'{path.name} 读取失败：{exc}');return {}


def snapshot(sample, result, current_mask):
    """Resolve assets inside moved/exported sample first; no current GUI config."""
    result=copy.deepcopy(result);errors=[];sample=Path(sample)
    folder=sample/'predictions'/Path(result.get('prediction','')).parent.name
    report=read_json(folder/'inference.json',errors);request=read_json(folder/'request.json',errors)
    candidates=[(report.get('actual_config'),'inference.json / actual_config'),
                (result.get('config'),'results.json / config'),
                (report.get('launch_config'),'inference.json / launch_config（请求值）'),
                (request.get('launch_config'),'request.json / launch_config（请求值）')]
    config,origin=next(((c,o) for c,o in candidates if isinstance(c,dict) and c),({},'未保存配置'))
    if not report:errors.append('缺少可读推理报告，部分实际执行信息未记录。')
    metrics=result.get('metrics') or {}
    return dict(result=result,report=report,request=request,config=config,origin=origin,
                sample=str(sample),folder=str(folder),current_mask=current_mask,errors=errors,
                mask_current=bool(metrics.get('mask_sha256') and metrics['mask_sha256']==current_mask))


def configuration(s):
    c=s['config'];fields={**CONFIG_FIELDS,**{k:k for k in c if k not in CONFIG_FIELDS}}
    return {f'{label} [{key}]':c.get(key,MISSING) for key,label in fields.items()}


def information(s):
    r=s['result'];p=s['report'];q=s['request'];m=r.get('metrics') or {}
    info={'记录':r.get('label',MISSING),'配置来源':s['origin'],
          '模型推理 ms':r.get('inference_ms',MISSING)}
    for k,label in [('worker_seconds_to_mesh','工作进程到网格 ms'),('gui_mesh_loaded_seconds','GUI 网格就绪 ms'),('load_seconds','模型首次加载 ms')]:
        value=p.get(k)
        info[label]=value*1000 if isinstance(value,(int,float)) else MISSING
    info['评估与当前掩膜一致']=s['mask_current'] if m else MISSING
    info.update({label:m.get(k,MISSING) for k,label in METRICS.items()})
    info.update({label:p.get(k,MISSING) for k,label in EXEC_FIELDS.items()})
    info.update({'RGB SHA256':r.get('rgb_sha256',p.get('image_sha256',MISSING)),
                 '当次人物选择框':r.get('person_bbox_xyxy',q.get('bbox_xyxy',MISSING)),
                 '请求相机内参':q.get('camera_intrinsics',MISSING),
                 '批处理输入指纹':r.get('batch_input',MISSING),
                 '输入选择方式':r.get('input_selection',MISSING),
                 '评估掩膜 SHA256':m.get('mask_sha256',MISSING),
                 '评估掩膜来源':m.get('mask_source',MISSING),
                 '评估与当前掩膜一致':s['mask_current'] if m else MISSING})
    info['预测文件']=str(Path(s['folder'])/'person_00.npz')
    info['推理报告']=str(Path(s['folder'])/'inference.json')
    return info


def warnings(a,b=None):
    messages=list(a['errors'])
    for tag,s in [('A',a)]+([('B',b)] if b else []):
        if s['result'].get('metrics') and not s['mask_current']:
            messages.append(tag+' 的指标来自历史掩膜，与当前掩膜不一致；仍保留历史值。')
    if b:
        messages+=b['errors']
        ia,ib=information(a),information(b)
        for key in ('RGB SHA256','当次人物选择框','请求相机内参','批处理输入指纹','输入选择方式','评估掩膜 SHA256'):
            va,vb=ia[key],ib[key]
            if va==MISSING or vb==MISSING:messages.append(key+'：记录不完整，无法确认输入一致。')
            elif va!=vb:messages.append(key+' 不同，指标不能直接作为同输入下的精度比较。')
    messages.append('↓ 越低越好，↑ 越高越好；差值为 B − A。单次推理时间不代表稳定帧率或相机端到端延迟。' if b else '详情来自该次保存的记录；未记录字段不使用当前界面设置补齐。')
    return '\n'.join(dict.fromkeys(messages))


def make_table(a,b=None):
    table=QtWidgets.QTableWidget();table.setColumnCount(4 if b is not None else 2)
    table.setHorizontalHeaderLabels(['参数 / 信息','A','B','B − A'] if b is not None else ['参数 / 信息','保存值'])
    table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
    table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
    table.setWordWrap(True);table.verticalHeader().hide()
    keys=list(a)+([k for k in b if k not in a] if b is not None else [])
    table.setRowCount(len(keys));table.difference_rows=[]
    for row,key in enumerate(keys):
        va=a.get(key,MISSING);vb=b.get(key,MISSING) if b is not None else None
        vals=[key,pretty(va)];different=b is not None and va!=vb
        if b is not None:
            delta=''
            if all(type(v) in (float,int) and math.isfinite(v) for v in (va,vb)):delta=f'{vb-va:+.6g}'
            vals.extend([pretty(vb),delta])
        table.difference_rows.append(different)
        for col,value in enumerate(vals):
            item=QtWidgets.QTableWidgetItem(value);item.setToolTip(value)
            if different:item.setBackground(QtGui.QColor('#5b421b'));item.setForeground(QtGui.QColor('#fff2ce'))
            table.setItem(row,col,item)
    header=table.horizontalHeader();header.setSectionResizeMode(0,QtWidgets.QHeaderView.Interactive);table.setColumnWidth(0,295)
    for col in range(1,table.columnCount()):header.setSectionResizeMode(col,QtWidgets.QHeaderView.Stretch)
    table.resizeRowsToContents()
    # Hash inventories remain scrollable/copyable without swallowing the entire table.
    for row in range(table.rowCount()):table.setRowHeight(row,min(150,max(32,table.rowHeight(row))))
    return table


class RecordDialog(QtWidgets.QDialog):
    def __init__(self,workspace,index,compare=False):
        super().__init__(workspace);self.v=workspace;self.index=index;self.scenes=[];self._view_states=[]
        self.setWindowTitle('历史记录：详细配置与推理对比' if compare else '历史记录：详细配置')
        self.resize(1250,850);layout=QtWidgets.QVBoxLayout(self)
        self.a=snapshot(workspace.sample,workspace.results[index],workspace.mask_hash());self.b=None
        c=self.a['config']
        title=QtWidgets.QLabel(f"A：{c.get('backend','?')} / {c.get('mode','?')} / {c.get('image_size','?')} / "+workspace.results[index]['label']);title.setWordWrap(True);layout.addWidget(title)
        self.selector=None
        if compare:
            self.selector=QtWidgets.QComboBox()
            for i,r in enumerate(workspace.results):
                if i!=index:
                    c=r.get('config') or {}
                    self.selector.addItem(f"B：{c.get('backend','?')} / {c.get('mode','?')} / {c.get('image_size','?')} / {r['label']}",i)
            layout.addWidget(self.selector);self.selector.currentIndexChanged.connect(self.rebuild)
        self.notice=QtWidgets.QLabel();self.notice.setWordWrap(True);self.notice.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse);layout.addWidget(self.notice)
        self.diff=QtWidgets.QCheckBox('仅显示有差异的配置 / 信息');self.diff.setVisible(compare);self.diff.toggled.connect(self.filter_rows);layout.addWidget(self.diff)
        self.tabs=QtWidgets.QTabWidget();layout.addWidget(self.tabs,1);self.tables=[];self.tabs.currentChanged.connect(self.tab_changed)
        buttons=QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        buttons.button(QtWidgets.QDialogButtonBox.Close).setText('关闭')
        buttons.rejected.connect(self.reject);copy_btn=buttons.addButton('复制完整记录 JSON',QtWidgets.QDialogButtonBox.ActionRole);copy_btn.clicked.connect(self.copy_json)
        layout.addWidget(buttons)
        self.sync_timer=QtCore.QTimer(self);self.sync_timer.setInterval(60);self.sync_timer.timeout.connect(self.sync_views)
        self.rebuild()

    def rebuild(self):
        self.sync_timer.stop();self.scenes=[];self.tables=[]
        self.tabs.blockSignals(True)
        while self.tabs.count():
            widget=self.tabs.widget(0);self.tabs.removeTab(0);widget.deleteLater()
        self.b=snapshot(self.v.sample,self.v.results[self.selector.currentData()],self.v.mask_hash()) if self.selector else None
        self.notice.setText(warnings(self.a,self.b))
        for name,fn in [('详细启动配置',configuration),('实际执行 / 耗时 / 评估',information)]:
            table=make_table(fn(self.a),fn(self.b) if self.b else None);self.tables.append(table);self.tabs.addTab(table,name)
        self.raw=QtWidgets.QPlainTextEdit();self.raw.setReadOnly(True);self.raw.setPlainText(self.json_text());self.tabs.addTab(self.raw,'原始记录 JSON')
        self.visual=QtWidgets.QWidget();self.visual_layout=QtWidgets.QVBoxLayout(self.visual);self.tabs.addTab(self.visual,'模型 / 误差图对照（点击加载）')
        self.tabs.blockSignals(False);self.filter_rows()

    def json_text(self):return json.dumps({'A':self.a,**({'B':self.b} if self.b else {})},ensure_ascii=False,indent=2)
    def copy_json(self):QtWidgets.QApplication.clipboard().setText(self.json_text())
    def filter_rows(self):
        for t in self.tables:
            for row,different in enumerate(t.difference_rows):t.setRowHidden(row,self.diff.isChecked() and not different)

    def tab_changed(self,index):
        if self.tabs.widget(index) is not self.visual or self.scenes:return
        # Defer mesh and OpenGL work until this tab is explicitly requested.
        import numpy as np
        row=QtWidgets.QHBoxLayout();self.visual_layout.addLayout(row,1)
        for tag,s in [('A',self.a)]+([('B',self.b)] if self.b else []):
            column=QtWidgets.QVBoxLayout();row.addLayout(column,1)
            label=QtWidgets.QLabel(tag+'：'+s['result'].get('label',''));label.setWordWrap(True);column.addWidget(label)
            scene=type(self.v.scene)();column.addWidget(scene,1);self.scenes.append(scene)
            try:
                with np.load(Path(s['folder'])/'person_00.npz',allow_pickle=False) as data:
                    scene.set_mesh(data['pred_vertices']+data['pred_cam_t'].reshape(1,3),data['faces'])
                # Current RGB-D can only be shown as reference for a matching saved RGB.
                current_hash=hashlib.sha256((Path(s['sample'])/'rgb.png').read_bytes()).hexdigest()
                if s['result'].get('rgb_sha256')==current_hash and self.v.xyz is not None:
                    scene.set_cloud(self.v.xyz,self.v.rgb,self.v.canvas.mask)
            except Exception as exc:column.addWidget(QtWidgets.QLabel('模型载入失败：'+str(exc)))
            evaluation=s['result'].get('evaluation')
            path=Path(s['sample'])/'evaluations'/Path(evaluation).name/'depth_error.png' if evaluation else None
            if path and path.is_file():
                image=QtWidgets.QLabel();image.setAlignment(QtCore.Qt.AlignCenter)
                image.setPixmap(QtGui.QPixmap(str(path)).scaled(500,145,QtCore.Qt.KeepAspectRatio,QtCore.Qt.SmoothTransformation));column.addWidget(image)
            column.addWidget(QtWidgets.QLabel('误差图为当次历史评估；模型保持原始相机坐标、尺度与平移。'))
        actions=QtWidgets.QHBoxLayout();self.visual_layout.addLayout(actions)
        for text,angle in [('统一正面',-90),('统一侧面',0)]:
            btn=QtWidgets.QPushButton(text);btn.clicked.connect(lambda checked=False,a=angle:self.reset_views(a));actions.addWidget(btn)
        self.reset_views(-90);self.sync_timer.start()

    def camera_state(self,scene):
        o=scene.opts;c=o['center'];return (c.x(),c.y(),c.z(),o['distance'],o['elevation'],o['azimuth'],o['fov'])
    def apply_camera(self,scene,state):
        scene.opts.update(center=QtGui.QVector3D(*state[:3]),distance=state[3],elevation=state[4],azimuth=state[5],fov=state[6]);scene.update()
    def reset_views(self,angle):
        if not self.scenes:return
        self.scenes[0].front();self.scenes[0].setCameraPosition(azimuth=angle)
        state=self.camera_state(self.scenes[0])
        for scene in self.scenes[1:]:self.apply_camera(scene,state)
        self._view_states=[self.camera_state(scene) for scene in self.scenes]
    def sync_views(self):
        if not self.isVisible() or self.tabs.currentWidget() is not self.visual:return
        states=[self.camera_state(scene) for scene in self.scenes]
        for i,state in enumerate(states):
            if state!=self._view_states[i]:
                for j,scene in enumerate(self.scenes):
                    if i!=j:self.apply_camera(scene,state)
                break
        self._view_states=[self.camera_state(scene) for scene in self.scenes]
    def done(self,result):self.sync_timer.stop();super().done(result)
