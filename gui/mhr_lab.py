"""Independent MHR parameter laboratory tab; GUI never imports torch."""
import copy
import json
from pathlib import Path
import tempfile
import time
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph.opengl as gl
from mesh_geometry import FastMeshData

ROOT = Path(__file__).resolve().parents[1]
from studio_paths import path
PYTHON = str(path('fast_python'))


def meaning(name):
    special = {'root_tx':'整体 X 平移', 'root_ty':'整体 Y 平移', 'root_tz':'整体 Z 平移',
               'root_rx':'整体 X 旋转','root_ry':'整体 Y 旋转','root_rz':'整体 Z 旋转',
               'scale_shoulder_width':'肩宽尺度','scale_uparms':'上臂尺度','scale_lowarms':'前臂尺度',
               'scale_uplegs':'大腿尺度','scale_lowlegs':'小腿尺度', 'scale_spine_length':'躯干长度',
               'scale_neck_length':'颈部长度','scale_hip_width':'髋宽尺度'}
    if name in special:return special[name]
    s = name
    for a,b in [('r_', '右侧 '),('l_', '左侧 '),('c_', '中线 ')]:
        if s.startswith(a):s=b+s[len(a):];break
    for a,b in [('clavicle','锁骨'),('uparm','上臂'),('lowarm','前臂'),('elbow','肘'),
                ('wrist','腕'),('upleg','大腿'),('lowleg','小腿'),('knee','膝'),('foot','足'),
                ('spine','脊柱'),('head','头'),('neck','颈'),('thumb','拇指'),('index','食指'),
                ('middle','中指'),('ring','无名指'),('pinky','小指'),('jaw','下颌'),('eye','眼'),
                ('tongue','舌'),('twist','扭转'),('bend','弯曲'),('lean','侧倾'),('scale','尺度')]:
        s=s.replace(a,b)
    return s


class LabClient(QtCore.QObject):
    ready = QtCore.pyqtSignal(dict)
    frame = QtCore.pyqtSignal(dict)
    error = QtCore.pyqtSignal(str)
    activity = QtCore.pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.process=QtCore.QProcess(self)
        self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.read)
        self.process.finished.connect(self.finished)
        self.process.errorOccurred.connect(self.process_error)
        self.temp=None; self.buffer=''; self.loaded=False; self.inflight=None; self.pending=None
        self.serial=0; self.stopping=False; self.log=''; self.device='auto'
        self.watchdog=QtCore.QTimer(self);self.watchdog.setInterval(1000);self.watchdog.timeout.connect(self.check_timeout)
        self.last_event=time.monotonic()

    def start(self):
        if self.process.state()!=QtCore.QProcess.NotRunning:return
        self.temp=tempfile.TemporaryDirectory(prefix='mhr-lab-')
        self.loaded=False;self.buffer='';self.stopping=False;self.log='';self.last_event=time.monotonic()
        self.process.setWorkingDirectory(str(ROOT))
        env=QtCore.QProcessEnvironment.systemEnvironment();env.remove('PYTHONPATH');env.insert('PYTHONDONTWRITEBYTECODE','1')
        self.process.setProcessEnvironment(env)
        self.process.start(PYTHON,['-u',str(ROOT/'mhr_lab_worker.py'),'--output',self.temp.name,'--device',self.device])
        self.watchdog.start()

    def submit(self, kind, **kw):
        self.serial+=1;self.pending=dict(type=kind,request_id=self.serial,**kw)
        self.activity.emit(True);self.start();self.dispatch()
        return self.serial

    def dispatch(self):
        if not self.loaded or self.inflight is not None or self.pending is None:return
        self.inflight=self.pending;self.pending=None;self.last_event=time.monotonic()
        self.process.write((json.dumps(self.inflight,ensure_ascii=False)+'\n').encode())

    def read(self):
        self.buffer+=bytes(self.process.readAllStandardOutput()).decode('utf-8',errors='replace')
        while '\n' in self.buffer:
            line,self.buffer=self.buffer.split('\n',1)
            if not line.startswith('@@MHRLAB '):
                self.log=(self.log+line+'\n')[-6000:];continue
            try:event=json.loads(line[len('@@MHRLAB '):])
            except ValueError:continue
            self.last_event=time.monotonic();kind=event.get('type')
            if kind=='ready':
                self.loaded=True;self.ready.emit(event['metadata']);self.dispatch()
            elif kind in ('frame','error'):
                if self.inflight is None or event.get('request_id')!=self.inflight['request_id']:continue
                event['operation']=self.inflight['type'];self.inflight=None
                if kind=='frame':self.frame.emit(event)
                else:self.error.emit(event['message'])
                self.dispatch();self.activity.emit(self.inflight is not None or self.pending is not None)
            elif kind=='fatal':
                self.pending=None;self.error.emit(event['message'])

    def process_error(self, error):
        if error==QtCore.QProcess.FailedToStart:
            self.pending=None;self.error.emit('实验室进程启动失败：'+self.process.errorString());self.activity.emit(False)

    def finished(self, *args):
        unexpected=not self.stopping and (self.inflight is not None or self.pending is not None or self.loaded)
        self.loaded=False;self.inflight=None;self.pending=None;self.watchdog.stop()
        if unexpected:self.error.emit('实验室进程退出，可重新创建或导入。\n'+self.log[-1500:])
        self.activity.emit(False)
        if self.temp:self.temp.cleanup();self.temp=None

    def check_timeout(self):
        if (not self.loaded or self.inflight) and time.monotonic()-self.last_event>90:
            self.error.emit('实验室加载/计算超时，请释放后重试或选择 CPU。');self.stop()

    def stop(self):
        self.stopping=True;self.pending=None;self.watchdog.stop()
        if self.process.state()!=QtCore.QProcess.NotRunning:
            self.process.terminate()
            if not self.process.waitForFinished(1500):self.process.kill();self.process.waitForFinished(1500)
        self.loaded=False;self.inflight=None


class MhrLab(QtWidgets.QWidget):
    def __init__(self, window, viewer_class):
        super().__init__()
        self.window=window;self.client=LabClient(self);self.metadata=None;self.state=None;self.base=None
        self.frame=None;self.frame_state=None;self.latest=None;self.joint=0;self.param=None
        self.param_items={};self.joint_items={};self.loading=False;self.seeding=False;self.reference=None;self.raw_parameters={}
        self.timer=QtCore.QTimer(self);self.timer.setSingleShot(True);self.timer.setInterval(16);self.timer.timeout.connect(self.render)
        self.client.ready.connect(self.setup_metadata);self.client.frame.connect(self.receive)
        self.client.error.connect(self.failed);self.client.activity.connect(self.busy)
        layout=QtWidgets.QVBoxLayout(self);layout.setContentsMargins(16,12,16,12)
        top=QtWidgets.QHBoxLayout()
        self.default_btn=QtWidgets.QPushButton('新建默认人体');self.default_btn.clicked.connect(self.new_default)
        self.import_btn=QtWidgets.QPushButton('导入最近一次预测');self.import_btn.setEnabled(False);self.import_btn.clicked.connect(self.import_latest)
        self.open_btn=QtWidgets.QPushButton('打开预测 / 实验 NPZ…');self.open_btn.clicked.connect(self.open_file)
        self.device=QtWidgets.QComboBox();self.device.addItem('自动 / CUDA','auto');self.device.addItem('CPU','cpu')
        self.device.currentIndexChanged.connect(self.device_changed)
        release=QtWidgets.QPushButton('释放实验室模型');release.clicked.connect(self.release)
        for w in [self.default_btn,self.import_btn,self.open_btn,self.device,release]:top.addWidget(w)
        top.addStretch();layout.addLayout(top)
        self.source=QtWidgets.QLabel('独立编辑页 · 创建默认人体，或导入第一个页面最近一次预测。');self.source.setWordWrap(True);layout.addWidget(self.source)
        split=QtWidgets.QSplitter();layout.addWidget(split,1)
        left=QtWidgets.QWidget();ll=QtWidgets.QVBoxLayout(left);ll.setContentsMargins(0,0,0,0)
        self.search=QtWidgets.QLineEdit();self.search.setPlaceholderText('搜索参数或关节：肘 / elbow / scale…');self.search.textChanged.connect(self.filter_items);ll.addWidget(self.search)
        self.lists=QtWidgets.QTabWidget();ll.addWidget(self.lists)
        self.parameters=QtWidgets.QTreeWidget();self.parameters.setHeaderLabels(['参数 / 原始名称','当前','初始'])
        self.parameters.header().setSectionResizeMode(0,QtWidgets.QHeaderView.Stretch)
        for col in (1,2):
            self.parameters.header().setSectionResizeMode(col,QtWidgets.QHeaderView.Fixed);self.parameters.setColumnWidth(col,76)
        self.parameters.currentItemChanged.connect(self.select_param);self.lists.addTab(self.parameters,'原生参数 321')
        self.joints=QtWidgets.QTreeWidget();self.joints.setHeaderLabels(['127 节点 / 原始名称'])
        self.joints.currentItemChanged.connect(self.select_joint);self.lists.addTab(self.joints,'骨架节点 127')
        self.raw_view=QtWidgets.QPlainTextEdit();self.raw_view.setReadOnly(True)
        self.raw_view.setPlainText('导入预测后显示未修改的网络输出值；左侧参数页展示可编辑的 MHR 原生参数。')
        self.lists.addTab(self.raw_view,'预测原始值')
        left.setMinimumWidth(290);split.addWidget(left)
        middle=QtWidgets.QWidget();ml=QtWidgets.QVBoxLayout(middle);ml.setContentsMargins(0,0,0,0)
        views=QtWidgets.QHBoxLayout()
        for label,az in [('正面',-90),('背面',90),('左侧',180),('右侧',0)]:
            b=QtWidgets.QPushButton(label);b.clicked.connect(lambda _,a=az:self.viewer.preset(a));views.addWidget(b)
        reset=QtWidgets.QPushButton('复位视角');reset.clicked.connect(self.fit_view);views.addWidget(reset);ml.addLayout(views)
        self.viewer=viewer_class();self.viewer.setMinimumSize(340,350)
        self.viewer.empty.setText('创建或导入人体后，拖动参数实时观察形变')
        self.viewer.stream_anchor=None;ml.addWidget(self.viewer,1)
        self.axes=gl.GLLinePlotItem(mode='lines',width=4,antialias=True);self.viewer.addItem(self.axes)
        self.marker=gl.GLLinePlotItem(mode='lines',width=4,color=(1,.75,.15,1),antialias=True);self.viewer.addItem(self.marker)
        options=QtWidgets.QHBoxLayout()
        self.skeleton=QtWidgets.QCheckBox('127 骨架');self.skeleton.setChecked(True)
        self.ghost=QtWidgets.QCheckBox('初始线框');self.heat=QtWidgets.QCheckBox('位移热力图')
        for w in [self.skeleton,self.ghost,self.heat]:options.addWidget(w);w.toggled.connect(self.draw)
        ml.addLayout(options)
        legend=QtWidgets.QLabel('左键旋转 · 滚轮缩放 · 中键平移\n选中节点：X 红 / Y 绿 / Z 蓝；热力图：蓝 → 红，表面位移由小到大')
        legend.setObjectName('muted');legend.setWordWrap(True);ml.addWidget(legend)
        split.addWidget(middle)
        right_scroll=QtWidgets.QScrollArea();right_scroll.setWidgetResizable(True);right_scroll.setMinimumWidth(300)
        right=QtWidgets.QWidget();rl=QtWidgets.QVBoxLayout(right);right_scroll.setWidget(right);split.addWidget(right_scroll)
        self.param_title=QtWidgets.QLabel('选择一个原生参数');self.param_title.setWordWrap(True);rl.addWidget(self.param_title)
        self.value=QtWidgets.QDoubleSpinBox();self.value.setDecimals(5);self.value.setRange(-1e4,1e4);self.value.valueChanged.connect(self.value_changed);rl.addWidget(self.value)
        self.slider=QtWidgets.QSlider(QtCore.Qt.Horizontal);self.slider.setRange(0,1000);self.slider.valueChanged.connect(self.slider_changed);rl.addWidget(self.slider)
        self.range_label=QtWidgets.QLabel();self.range_label.setWordWrap(True);self.range_label.setObjectName('muted');rl.addWidget(self.range_label)
        self.param_reset=QtWidgets.QPushButton('恢复当前参数');self.param_reset.clicked.connect(self.reset_param);rl.addWidget(self.param_reset)
        self.joint_title=QtWidgets.QLabel('关节局部变换');self.joint_title.setWordWrap(True);rl.addWidget(self.joint_title)
        note=QtWidgets.QLabel('高级：直接覆盖局部关节参数。旋转 XYZ 为相对绑定预旋转的欧拉角；平移是绑定偏移外的增量。覆盖后可超出原生 204 参数的表达范围。')
        note.setWordWrap(True);note.setObjectName('muted');rl.addWidget(note)
        self.joint_edit=QtWidgets.QCheckBox('直接编辑所选节点（自定义骨架）');self.joint_edit.toggled.connect(self.enable_joint);rl.addWidget(self.joint_edit)
        self.joint_controls=[]
        for i,label in enumerate(['X 旋转 °','Y 旋转 °','Z 旋转 °','X 平移 cm','Y 平移 cm','Z 平移 cm']):
            row=QtWidgets.QHBoxLayout();row.addWidget(QtWidgets.QLabel(label))
            spin=QtWidgets.QDoubleSpinBox();spin.setDecimals(3);spin.setRange(-180,180) if i<3 else spin.setRange(-500,500)
            slide=QtWidgets.QSlider(QtCore.Qt.Horizontal);slide.setRange(-1800,1800) if i<3 else slide.setRange(-5000,5000)
            row.addWidget(spin);rl.addLayout(row);rl.addWidget(slide)
            spin.valueChanged.connect(lambda v,k=i:self.joint_value_changed(k,v));slide.valueChanged.connect(lambda v,s=spin:s.setValue(v/10))
            self.joint_controls.append((spin,slide))
        self.related=QtWidgets.QLabel();self.related.setWordWrap(True);self.related.setObjectName('muted');rl.addWidget(self.related)
        self.transforms=QtWidgets.QPlainTextEdit();self.transforms.setReadOnly(True);self.transforms.setMinimumHeight(210);rl.addWidget(self.transforms)
        self.correctives=QtWidgets.QCheckBox('姿态表面修正');self.correctives.setChecked(True);self.correctives.toggled.connect(self.correctives_changed);rl.addWidget(self.correctives)
        restore=QtWidgets.QPushButton('全部恢复至导入 / 默认状态');restore.clicked.connect(self.reset_all);rl.addWidget(restore)
        self.export_btn=QtWidgets.QPushButton('导出参数 + 网格…');self.export_btn.clicked.connect(self.export_dialog);rl.addWidget(self.export_btn)
        rl.addStretch();split.setSizes([350,600,330])
        self.status=QtWidgets.QLabel('实验室模型尚未加载 · 不需要预热 SAM 网络');self.status.setWordWrap(True);layout.addWidget(self.status)
        self.set_editable(False)

    def set_editable(self, enabled):
        for w in [self.parameters,self.joints,self.value,self.slider,self.param_reset,self.joint_edit,self.correctives]:w.setEnabled(enabled)
        self.export_btn.setEnabled(enabled and self.frame is not None)
        for spin,slide in self.joint_controls:spin.setEnabled(enabled and self.joint_edit.isChecked());slide.setEnabled(enabled and self.joint_edit.isChecked())

    def set_latest(self, data, source):
        keys=('mhr_model_params','global_rot','body_pose','hand','scale','shape','face','pred_vertices','body_pose_params','hand_pose_params','scale_params','shape_params','expr_params','pred_pose_raw','pred_cam_t')
        self.latest=({k:np.array(data[k],copy=True) for k in keys if k in data}, str(source))
        self.import_btn.setEnabled(not self.seeding)
        self.import_btn.setToolTip('复制预测快照：'+str(source))

    def new_default(self):self.seed(source='默认中性人体：204 姿态/尺度、45 体型、72 表情均为零')

    def import_latest(self):
        if self.latest:self.seed(data={k:v.tolist() for k,v in self.latest[0].items()},source=self.latest[1])

    def seed(self, **kw):
        if self.client.device!='cpu' and self.window.sam3_blocks_models():
            self.status.setText('SAM3 正在分割，暂不能加载 MHR GPU 模型；可等待或选择 CPU。');return
        self.timer.stop();self.seeding=True;self.set_editable(False)
        self.default_btn.setEnabled(False);self.open_btn.setEnabled(False);self.import_btn.setEnabled(False)
        self.status.setText('正在加载 MHR / 创建人体…');self.client.submit('seed',**kw)

    def open_file(self):
        path,_=QtWidgets.QFileDialog.getOpenFileName(self,'打开预测或实验快照',str(ROOT/'outputs'),'NumPy (*.npz)')
        if path:self.seed(path=path,source=path)

    def device_changed(self):
        self.release();self.client.device=self.device.currentData();self.status.setText('运行设备已切换，点击新建或导入重新加载。')

    def release(self):
        self.timer.stop();self.client.stop();self.state=None;self.base=None;self.frame_state=None;self.seeding=False
        self.set_editable(False);self.default_btn.setEnabled(True);self.open_btn.setEnabled(True);self.import_btn.setEnabled(self.latest is not None)
        self.status.setText('实验室模型已释放；可重新创建或导入。')

    def busy(self, on):
        self.export_btn.setEnabled(not on and not self.timer.isActive() and self.frame is not None and self.state is not None and self.frame_state==self.state)
        if on and not self.seeding:self.status.setText('正在更新人体…')

    def failed(self, text):
        self.seeding=False;self.default_btn.setEnabled(True);self.open_btn.setEnabled(True);self.import_btn.setEnabled(self.latest is not None)
        self.set_editable(self.state is not None and self.client.loaded)
        self.export_btn.setEnabled(False);self.status.setText('实验室：'+text)

    def setup_metadata(self, meta):
        self.metadata=meta;self.parameters.clear();self.joints.clear();self.param_items={};self.joint_items={}
        for title,key,start,end in [('整体变换','model',0,6),('身体 / 手部姿态与长度','model',6,136),('骨骼尺度','model',136,204),('体型混合系数','shape',0,45),('表情混合系数','expression',0,72)]:
            group=QtWidgets.QTreeWidgetItem(self.parameters,[title])
            for i in range(start,end):
                name=meta['names'][i] if key=='model' else f'{key}_{i:02d}'
                label=meaning(name) if key=='model' else ('体型基 ' if key=='shape' else '表情基 ')+str(i)
                item=QtWidgets.QTreeWidgetItem(group,[f'{label} · {name}']);item.setData(0,QtCore.Qt.UserRole,(key,i));item.setToolTip(0,name)
                self.param_items[(key,i)]=item
            group.setExpanded(key=='model' and start<136)
        # Build before attaching, because the rig need not be topologically sorted.
        for j,name in enumerate(meta['joint_names']):
            item=QtWidgets.QTreeWidgetItem([f'{j:03d}  {meaning(name)} · {name}']);item.setData(0,QtCore.Qt.UserRole,j);self.joint_items[j]=item
        for j,p in enumerate(meta['parents']):
            if p<0 or p>=127 or p==j:self.joints.addTopLevelItem(self.joint_items[j])
            else:self.joint_items[p].addChild(self.joint_items[j])
        self.joints.expandToDepth(2)

    def receive(self, event):
        try:
            with np.load(event['path'],allow_pickle=False) as f:self.frame={k:f[k].copy() for k in f.files}
            self.frame_state=copy.deepcopy(event['state'])
            if event['operation']=='seed':
                self.state=copy.deepcopy(event['state']);self.base=copy.deepcopy(event['base']);self.seeding=False
                self.raw_parameters=event.get('raw_parameters') or {}
                lines=['导入时的原始数据快照（只读，不随实验修改）', '原生编辑参数：204；体型：45；表情：72。\n']
                for key,value in self.raw_parameters.items():
                    a=np.asarray(value);lines.append(f'{key}  shape={a.shape}\n'+np.array2string(a,precision=5,threshold=1200,max_line_width=65)+'\n')
                self.raw_view.setPlainText('\n'.join(lines) if self.raw_parameters else '默认人体：原生姿态/尺度 204、体型 45、表情 72 均为零。\n没有运行图像网络，因此没有预测原始值。')
                self.default_btn.setEnabled(True);self.open_btn.setEnabled(True);self.import_btn.setEnabled(self.latest is not None)
                self.source.setText('当前实验起点：'+event['source']+' · 预测与相机页面保留原始数据')
                self.viewer.stream_anchor=None
                self.viewer.clear_mesh()
                if self.reference is not None:self.viewer.removeItem(self.reference);self.reference=None
                self.loading=True;self.correctives.setChecked(self.state['correctives']);self.loading=False
                self.set_editable(True)
                self.parameters.setCurrentItem(self.param_items[('model',36)])
                self.joints.setCurrentItem(self.joint_items[40])
            self.refresh_values();self.refresh_param();self.refresh_joint();self.draw()
            displacement=np.linalg.norm(self.frame['vertices']-self.frame['baseline_vertices'],axis=1)
            self.status.setText(f'{self.metadata["device"]} · MHR+取回 {float(self.frame["compute_ms"]):.1f} ms · '
                                f'最大表面位移 {displacement.max()*100:.2f} cm · {len(self.state["overrides"])} 个局部覆盖')
            if event['operation']=='seed' and 'import_max_error_mm' in self.frame:
                self.status.setText(self.status.text()+f' · 导入复现最大差异 {float(self.frame["import_max_error_mm"]):.3f} mm · 表面修正'+('开' if self.state['correctives'] else '关'))
            if self.client.pending is not None or self.timer.isActive():self.status.setText(self.status.text()+' · 正在更新最新修改')
        except Exception as exc:self.failed('读取人体失败：'+str(exc))

    def factor(self,key,i):return 180/np.pi if key=='model' and self.metadata['kinds'][i]=='angle' else 1.

    def refresh_values(self):
        if self.state is None:return
        for (key,i),item in self.param_items.items():
            f=self.factor(key,i);cur=self.state[key][i];base=self.base[key][i]
            suffix='°' if f!=1 else ''
            item.setText(1,f'{cur*f:.3f}{suffix}');item.setText(2,f'{base*f:.3f}{suffix}')
            item.setForeground(1,QtGui.QBrush(QtGui.QColor('#ffbf65' if abs(cur-base)>1e-6 else '#e2e8f0')))

    def select_param(self,item,*args):
        self.param=item.data(0,QtCore.Qt.UserRole) if item else None;self.refresh_param()

    def refresh_param(self):
        if self.state is None or self.param is None:return
        key,i=self.param;f=self.factor(key,i);value=self.state[key][i]*f
        if key=='model':
            name=self.metadata['names'][i];lo,hi=np.array(self.metadata['limits'][i])*f
            title=f'{meaning(name)}\n{name} · 索引 {i}'
        else:
            name=f'{key}_{i}';lo,hi=(-3,3);title=f'{"体型" if key=="shape" else "表情"}基 {i}\n混合系数可能影响多个区域，观察热力图判断。'
        lo=min(float(lo),value,self.base[key][i]*f);hi=max(float(hi),value,self.base[key][i]*f)
        self.loading=True
        if hi-lo<1e-8:
            self.value.setRange(lo,hi);self.slider.setEnabled(False)
        else:
            self.value.setRange(lo,hi);self.slider.setEnabled(True)
        self.value.setSuffix(' °' if f!=1 else '');self.value.setSingleStep(1 if f!=1 else .01);self.value.setValue(value)
        self.slide_bounds=(lo,hi);self.slider.setValue(round((value-lo)/max(hi-lo,1e-9)*1000))
        self.param_title.setText(title);self.range_label.setText(f'编辑范围 [{lo:.3f}, {hi:.3f}]'+('°' if f!=1 else '（原生单位）')+'\n模型 min/max 用作编辑范围；导入值不会被自动裁剪。')
        self.loading=False

    def value_changed(self,value):
        if self.loading or self.state is None or self.param is None:return
        key,i=self.param;self.state[key][i]=float(value/self.factor(key,i));lo,hi=self.slide_bounds
        self.loading=True;self.slider.setValue(round((value-lo)/max(hi-lo,1e-9)*1000));self.loading=False
        self.refresh_values();self.schedule()

    def slider_changed(self,value):
        if self.loading or not hasattr(self,'slide_bounds'):return
        lo,hi=self.slide_bounds;self.value.setValue(lo+(hi-lo)*value/1000)

    def reset_param(self):
        if self.state is not None and self.param is not None:
            key,i=self.param;self.state[key][i]=self.base[key][i];self.refresh_values();self.refresh_param();self.schedule()

    def select_joint(self,item,*args):
        if item is not None:self.joint=int(item.data(0,QtCore.Qt.UserRole));self.refresh_joint();self.draw_selection()

    def refresh_joint(self):
        if self.frame is None or self.state is None:return
        j=self.joint;name=self.metadata['joint_names'][j]
        self.joint_title.setText(f'{j:03d} · {meaning(name)}\n{name}')
        values=self.state['overrides'].get(str(j),self.frame['joint_parameters'][j,:6].tolist())
        self.loading=True;self.joint_edit.setChecked(str(j) in self.state['overrides'])
        for k,(spin,slide) in enumerate(self.joint_controls):
            v=float(np.degrees(values[k+3]) if k<3 else values[k-3]);bound=max(180 if k<3 else 500,abs(v)+1)
            spin.setRange(-bound,bound);slide.setRange(-round(bound*10),round(bound*10));spin.setValue(v);slide.setValue(round(v*10))
            spin.setEnabled(self.joint_edit.isChecked());slide.setEnabled(self.joint_edit.isChecked())
        self.loading=False
        related=self.metadata['joint_parameters'][j]
        self.related.setText('关联原生参数：'+(', '.join(self.metadata['names'][i] for i in related) if related else '没有直接驱动参数（辅助节点）'))
        def mat(key):return np.array2string(self.frame[key][j],precision=3,suppress_small=True)
        self.transforms.setPlainText('MHR 原生坐标（位置 cm）\n局部位置 '+mat('local_positions')+'\n局部旋转矩阵\n'+mat('local_rotations')+'\n全局位置 '+mat('global_positions')+'\n全局旋转矩阵\n'+mat('global_rotations'))

    def enable_joint(self,on):
        if self.loading or self.state is None or self.frame is None:return
        if on:self.state['overrides'][str(self.joint)]=self.frame['joint_parameters'][self.joint,:6].tolist()
        else:self.state['overrides'].pop(str(self.joint),None)
        self.refresh_joint();self.schedule()

    def joint_value_changed(self,k,v):
        if self.loading or self.state is None or not self.joint_edit.isChecked():return
        values=self.state['overrides'][str(self.joint)]
        values[k+3 if k<3 else k-3]=float(np.radians(v) if k<3 else v)
        self.loading=True;self.joint_controls[k][1].setValue(round(v*10));self.loading=False;self.schedule()

    def correctives_changed(self,on):
        if not self.loading and self.state is not None:self.state['correctives']=on;self.schedule()

    def reset_all(self):
        if self.base is None:return
        self.state=copy.deepcopy(self.base);self.loading=True;self.correctives.setChecked(self.state['correctives']);self.loading=False
        self.refresh_values();self.refresh_param();self.refresh_joint();self.schedule()

    def schedule(self):
        self.export_btn.setEnabled(False)
        # Throttle rather than debounce: sustained dragging must still produce frames.
        if not self.timer.isActive():self.timer.start()

    def render(self):
        if self.state is not None and not self.seeding:self.client.submit('render',state=copy.deepcopy(self.state))

    def filter_items(self,text):
        text=text.lower().strip()
        def visit(item):
            own=text in item.text(0).lower();child=False
            for i in range(item.childCount()):child=visit(item.child(i)) or child
            item.setHidden(not(own or child))
            if text and child:item.setExpanded(True)
            return own or child
        for tree in [self.parameters,self.joints]:
            for i in range(tree.topLevelItemCount()):visit(tree.topLevelItem(i))

    def draw(self,*args):
        if self.frame is None:return
        f=self.frame;first=self.viewer.stream_anchor is None;self.viewer.set_mesh(f['vertices'],f['faces'],continuous=True)
        edges=np.array([(p,j) for j,p in enumerate(self.metadata['parents']) if 0<=p<127 and p!=j],np.int32)
        self.viewer.set_layers(True,self.skeleton.isChecked());self.viewer.set_pose(f['joints'],edges)
        if self.heat.isChecked():
            d=np.linalg.norm(f['vertices']-f['baseline_vertices'],axis=1);t=np.clip(d/max(float(d.max()),.001),0,1)
            colors=np.column_stack([.15+.85*t,.45*(1-t),1-.85*t,np.ones(len(t))]).astype(np.float32)
            self.viewer.mesh.opts['meshdata'].setVertexColors(colors);self.viewer.mesh.meshDataChanged()
        if self.reference is None:
            v=f['baseline_vertices'][:,[0,2,1]].copy();v[:,2]*=-1;v-=self.viewer.anchor
            self.reference=gl.GLMeshItem(meshdata=FastMeshData(vertexes=v,faces=f['faces']),drawFaces=False,drawEdges=True,edgeColor=(.65,.8,.9,.18),glOptions='translucent')
            self.viewer.addItem(self.reference)
        self.reference.setVisible(self.ghost.isChecked());self.draw_selection()
        if first:self.fit_view()

    def fit_view(self):
        if self.frame is None or self.viewer.anchor is None:return
        v=self.frame['vertices'][:,[0,2,1]].copy();v[:,2]*=-1;v-=self.viewer.anchor
        low,high=v.min(axis=0),v.max(axis=0);center=(low+high)/2
        ratio=max(self.viewer.width(),1)/max(self.viewer.height(),1)
        tangent=np.tan(np.radians(self.viewer.opts['fov'])/2)
        distance=max(float(high[0]-low[0]),float(high[2]-low[2])*ratio)/(2*tangent)*1.3
        self.viewer.opts['center']=QtGui.QVector3D(*map(float,center))
        self.viewer.setCameraPosition(distance=max(distance,.4),elevation=8,azimuth=-90);self.viewer.update()

    def draw_selection(self):
        if self.frame is None or self.viewer.anchor is None:return
        j=self.joint;C=np.array([[1,0,0],[0,0,-1],[0,1,0]],np.float32)
        p=C@self.frame['global_positions'][j]*.01-self.viewer.anchor
        R=C@self.frame['global_rotations'][j]
        pos=np.array([[p,p+R[:,k]*.12] for k in range(3)]).reshape(-1,3)
        self.axes.setData(pos=pos,color=np.repeat(np.array([[1,.2,.2,1],[.2,1,.2,1],[.3,.5,1,1]],np.float32),2,axis=0))
        cross=np.array([[p-np.eye(3)[k]*.02,p+np.eye(3)[k]*.02] for k in range(3)],np.float32).reshape(-1,3)
        self.marker.setData(pos=cross);self.viewer.update()

    def export_dialog(self):
        path,_=QtWidgets.QFileDialog.getSaveFileName(self,'导出实验参数及人体网格',str(ROOT/'outputs/mhr-lab-experiment.npz'),'实验快照 (*.npz)')
        if path:
            try:self.export_to(path)
            except Exception as exc:self.failed('导出失败：'+str(exc))

    def export_to(self,path):
        if self.frame is None or self.frame_state is None:raise ValueError('没有可导出的结果')
        if self.timer.isActive() or self.client.inflight or self.client.pending:raise ValueError('请等待最新修改生成完成后再导出')
        path=Path(path).with_suffix('.npz');path.parent.mkdir(parents=True,exist_ok=True)
        state=self.frame_state
        np.savez_compressed(path,**self.frame,mhr_model_params=np.array(state['model'],np.float32),shape=np.array(state['shape'],np.float32),
                            face=np.array(state['expression'],np.float32),lab_state_json=np.array(json.dumps(state)),
                            lab_base_json=np.array(json.dumps(self.base)),lab_raw_json=np.array(json.dumps(self.raw_parameters)))
        doc=dict(format='mhr-lab-v1',source=self.source.text(),state=state,baseline=self.base,
                 parameter_names=self.metadata['names'],joint_names=self.metadata['joint_names'],
                 mesh_coordinates='Studio camera axes; metres',joint_transform_coordinates='MHR native axes; centimetres',
                 custom_joint_overrides=bool(state['overrides']))
        path.with_suffix('.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2))
        # ASCII PLY avoids importing an additional mesh package into the GUI.
        v=self.frame['vertices'];faces=self.frame['faces']
        with path.with_suffix('.ply').open('w') as out:
            out.write(f'ply\nformat ascii 1.0\nelement vertex {len(v)}\nproperty float x\nproperty float y\nproperty float z\nelement face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n')
            np.savetxt(out,v,fmt='%.7g');np.savetxt(out,np.column_stack([np.full(len(faces),3),faces]),fmt='%d')
        self.status.setText('已导出参数 NPZ / JSON 与网格 PLY：'+str(path));return path

    def shutdown(self):self.timer.stop();self.client.stop()
