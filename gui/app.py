"""Local image-to-body desktop application. Inference runs in a separate process."""
from datetime import datetime
import json
import math
import uuid
from pathlib import Path
import sys
import time

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph.opengl as gl
from OpenGL import GL
from pyqtgraph.opengl import shaders
from camera_client import CameraClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backends import BACKENDS, missing_resources
from launch_config import LaunchConfig, PRESETS
from profile_panel import ProfilePanel
from run_history import record_report, show_history
from gui_camera import select_camera_intrinsics
from live_stream import LiveStream
from mesh_geometry import FastMeshData
from mhr_lab import MhrLab
from rgbd_validation import RgbdValidation
from overlay_view import OverlayView
from motion_panel import MotionPanel
from pose_data import EDGES, COLORS
from smpl_stage import SMPL_EDGES
from studio_paths import path
REPO = path('official_repo')
CHECKPOINT = path('standard_checkpoint')
SAMPLE = REPO / 'notebook/images/dancing.jpg'


shaders.ShaderProgram('body_studio', [
    shaders.VertexShader("""
        varying vec3 normal;
        void main() {
            normal = normalize(gl_NormalMatrix * gl_Normal);
            gl_FrontColor = gl_Color;
            gl_BackColor = gl_Color;
            gl_Position = ftransform();
        }
    """),
    shaders.FragmentShader("""
        varying vec3 normal;
        void main() {
            vec3 n = normalize(normal);
            float key = max(dot(n, normalize(vec3(-0.4, 0.6, 1.0))), 0.0);
            float fill = max(dot(n, normalize(vec3(0.8, -0.2, 0.6))), 0.0);
            gl_FragColor = vec4(gl_Color.rgb * (0.40 + 0.45 * key + 0.15 * fill), gl_Color.a);
        }
    """),
])


class ImageCanvas(QtWidgets.QWidget):
    fileDropped = QtCore.pyqtSignal(str)
    bboxChanged = QtCore.pyqtSignal()

    def __init__(self, selectable=False):
        super().__init__()
        self.image = QtGui.QImage()
        self.bbox = None
        self.drag_start = None
        self.overlay = ''
        self.selectable = selectable
        self.selection_enabled = True
        self.setAcceptDrops(selectable)
        self.setMinimumSize(240, 230)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        if selectable:
            self.setCursor(QtCore.Qt.CrossCursor)

    def set_image(self, image):
        self.image = image
        self.bbox = None
        self.update()

    def set_preview(self, image):
        self.image = image
        self.update()

    def image_rect(self):
        if self.image.isNull():
            return QtCore.QRectF()
        size = self.image.size().scaled(self.size() - QtCore.QSize(24, 24), QtCore.Qt.KeepAspectRatio)
        return QtCore.QRectF((self.width()-size.width())/2, (self.height()-size.height())/2, size.width(), size.height())

    def image_point(self, pos):
        r = self.image_rect()
        return QtCore.QPointF(np.clip((pos.x()-r.x())/r.width(), 0, 1)*self.image.width(),
                             np.clip((pos.y()-r.y())/r.height(), 0, 1)*self.image.height())

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor('#111827'))
        if self.image.isNull():
            p.setPen(QtGui.QColor('#8c9bb5'))
            p.drawText(self.rect(), QtCore.Qt.AlignCenter, '选择或拖入人体图片' if self.selectable else '重建完成后显示图像对照')
            return
        r = self.image_rect()
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        p.drawImage(r, self.image)
        if self.bbox:
            x1, y1, x2, y2 = self.bbox
            box = QtCore.QRectF(r.x()+x1/self.image.width()*r.width(), r.y()+y1/self.image.height()*r.height(),
                                (x2-x1)/self.image.width()*r.width(), (y2-y1)/self.image.height()*r.height())
            p.setPen(QtGui.QPen(QtGui.QColor('#54e5d0'), 2))
            p.setBrush(QtGui.QColor(84, 229, 208, 22))
            p.drawRect(box)

        if self.overlay:
            p.setPen(QtGui.QColor('white'))
            p.setBrush(QtGui.QColor(0, 0, 0, 150))
            box = QtCore.QRect(self.width()//2-65, self.height()//2-65, 130, 130)
            p.drawRoundedRect(box, 20, 20)
            p.setFont(QtGui.QFont('DejaVu Sans', 54, QtGui.QFont.Bold))
            p.drawText(box, QtCore.Qt.AlignCenter, self.overlay)

    def mousePressEvent(self, event):
        if self.selectable and self.selection_enabled and not self.image.isNull() and event.button() == QtCore.Qt.LeftButton and self.image_rect().contains(event.pos()):
            self.drag_start = self.image_point(event.pos())

    def mouseMoveEvent(self, event):
        if self.drag_start is not None:
            end = self.image_point(event.pos())
            self.bbox = [min(self.drag_start.x(), end.x()), min(self.drag_start.y(), end.y()),
                         max(self.drag_start.x(), end.x()), max(self.drag_start.y(), end.y())]
            self.update()

    def mouseReleaseEvent(self, event):
        if self.drag_start is not None:
            self.mouseMoveEvent(event)
            if self.bbox[2]-self.bbox[0] < 10 or self.bbox[3]-self.bbox[1] < 10:
                self.bbox = None
            self.drag_start = None
            self.bboxChanged.emit()
            self.update()

    def dragEnterEvent(self, event):
        if self.selection_enabled and event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        if self.selection_enabled and event.mimeData().hasUrls():
            path = event.mimeData().urls()[0].toLocalFile()
            if path:
                self.fileDropped.emit(path)


class BodyViewer(gl.GLViewWidget):
    def __init__(self):
        super().__init__()
        self.setBackgroundColor('#111827')
        self.setMinimumSize(350, 300)
        self.mesh = None
        self.anchor = None
        self.show_mesh = True
        self.show_skeleton = True
        self.bones = gl.GLLinePlotItem(width=3, mode='lines', antialias=True)
        self.addItem(self.bones)
        self.bones.hide()
        self.grid = gl.GLGridItem()
        self.grid.setSize(4, 4)
        self.grid.setSpacing(0.25, 0.25)
        self.grid.setColor((92, 118, 154, 55))
        self.addItem(self.grid)
        self.height_span = 1.8
        self.empty = QtWidgets.QLabel('重建后，可在这里旋转查看三维人体', self)
        self.empty.setAlignment(QtCore.Qt.AlignCenter)
        self.empty.setStyleSheet('color:#8c9bb5; background:transparent; font-size:15px;')
        self.empty.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.reset_view()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.empty.setGeometry(0, self.height()//2-25, self.width(), 50)

    def set_layers(self, mesh, skeleton):
        self.show_mesh, self.show_skeleton = mesh, skeleton
        if self.mesh is not None:self.mesh.setVisible(mesh)
        self.bones.setVisible(skeleton and getattr(self, 'pose_points', None) is not None)
        self.update()

    def set_pose(self, points, edges=None):
        self.pose_points = points
        if points is None or self.anchor is None:
            self.bones.hide();return
        v = np.asarray(points, dtype=np.float32)[:, [0,2,1]].copy()
        v[:,2] *= -1
        v -= self.anchor
        edges=EDGES if edges is None else edges
        colors=np.repeat(COLORS,2,axis=0) if edges is EDGES else np.tile([.45,.9,1.,1.],(len(edges)*2,1))
        self.bones.setData(pos=v[edges].reshape(-1,3), color=colors)
        self.bones.setGLOptions('translucent')
        self.bones.updateGLOptions({GL.GL_DEPTH_TEST: False})
        self.bones.setDepthValue(10)
        self.bones.setVisible(self.show_skeleton)

    def set_smpl_frame(self, points, continuous=False):
        self.set_skeleton_frame(points, SMPL_EDGES, continuous)

    def set_skeleton_frame(self, points, edges=None, continuous=False):
        first = not continuous or getattr(self, 'stream_anchor', None) is None
        v=np.asarray(points,dtype=np.float32)[:,[0,2,1]].copy();v[:,2]*=-1
        anchor=np.array([*(v[:,:2].min(axis=0)+v[:,:2].max(axis=0))/2,v[:,2].min()])
        if first:self.stream_anchor=anchor
        self.anchor=self.stream_anchor
        self.height_span=max(float(np.ptp(v[:,2])),float(np.ptp(v[:,0])),1.8)
        self.set_pose(points,edges);self.empty.hide()
        if first:self.reset_view()

    def clear_mesh(self):
        self.set_pose(None)
        self.anchor = None
        if self.mesh is not None:
            self.removeItem(self.mesh)
            self.mesh = None
        self.empty.show()

    def set_mesh(self, vertices, faces, wireframe=False, continuous=False):
        if not continuous:
            self.clear_mesh()
        # Camera-space x/right, y/down, z/depth -> preview x/right, y/depth, z/up.
        v = np.asarray(vertices, dtype=np.float32)[:, [0, 2, 1]].copy()
        v[:, 2] *= -1
        anchor = np.array([*(v[:, :2].min(axis=0)+v[:, :2].max(axis=0))/2, v[:, 2].min()])
        first = continuous and getattr(self, 'stream_anchor', None) is None
        if first:
            self.stream_anchor = anchor
        self.anchor = self.stream_anchor if continuous else anchor
        v -= self.anchor
        if continuous and self.mesh is not None:
            self.mesh.setMeshData(meshdata=FastMeshData(vertexes=v, faces=np.asarray(faces, dtype=np.uint32)))
            self.mesh.opts['drawEdges'] = wireframe
            self.update()
            return
        self.height_span = max(float(np.ptp(v[:, 2])), float(np.ptp(v[:, 0])), 2.0 if continuous else 0.2)
        self.mesh = gl.GLMeshItem(meshdata=FastMeshData(vertexes=v, faces=np.asarray(faces, dtype=np.uint32)),
                                  smooth=True, shader='body_studio', color=(0.57, 0.77, 0.94, 1),
                                  drawEdges=wireframe, edgeColor=(0.12, 0.21, 0.35, 0.65))
        self.addItem(self.mesh)
        self.mesh.setVisible(self.show_mesh)
        self.empty.hide()
        if not continuous or first:
            self.reset_view()

    def reset_view(self):
        self.opts['center'] = QtGui.QVector3D(0, 0, self.height_span*0.48)
        self.setCameraPosition(distance=self.height_span*2.1, elevation=8, azimuth=-90)
        self.update()

    def preset(self, azimuth):
        self.setCameraPosition(azimuth=azimuth, elevation=8)


class MainWindow(QtWidgets.QMainWindow):
    reconstructionFinished = QtCore.pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.setWindowTitle('SAM 3D Body Studio')
        self.setWindowIcon(QtGui.QIcon(str(ROOT/'gui/icon.svg')))
        self.resize(1320, 850)
        self.setMinimumSize(1000, 680)
        self.source_path = None
        self.result_dir = None
        self.running_dir = None
        self.report = None
        self.cancelled = False
        self.inference_busy = False
        self.started_at = 0
        self.model_state = 'off'
        self.settings = QtCore.QSettings('LocalVision', 'SAM3DBodyStudio')
        try:
            initial_config=LaunchConfig.from_dict(json.loads(str(self.settings.value('launch_config','null'))))
        except (ValueError,TypeError):
            initial_config=PRESETS['演示最快']
        self.initial_config=initial_config
        self.backend_id=initial_config.backend
        self.active_config=None
        self.pending_launch=False
        self.job_progress=''
        self.model_session = None
        self.pending_backend = None
        self.model_failure = None
        self.model_buffer = ''
        self.model_log_path = None
        self.pending_model_request = None
        self.active_job = None
        self.render_processes = {}
        self.kill_timer = QtCore.QTimer(self)
        self.kill_timer.setSingleShot(True)
        self.kill_timer.timeout.connect(self.kill_process)
        self.process = QtCore.QProcess(self)
        self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.read_log)
        self.process.finished.connect(self.process_finished)
        self.process.errorOccurred.connect(self.process_error)
        self.clock = QtCore.QTimer(self)
        self.clock.timeout.connect(self.tick)
        self.rotation = QtCore.QTimer(self)
        self.rotation.setInterval(35)
        self.rotation.timeout.connect(self.rotate_step)
        self.camera = CameraClient(self)
        self.camera.messageReceived.connect(self.camera_event)
        self.camera.frame.connect(self.camera_frame)
        self.camera.stopped.connect(self.camera_stopped)
        self.camera.log.connect(lambda text: self.log_box.appendPlainText(text.rstrip()))
        self.camera_state = 'off'
        self.camera_job = None
        self.pending_photo = None
        self.capture_metadata = None
        self.capture_metadata_error = None
        self.camera_error = None
        self.closing = False
        self.countdown_clock = QtCore.QTimer(self)
        self.countdown_clock.setInterval(100)
        self.countdown_clock.timeout.connect(self.countdown_tick)
        self.build_ui()
        self.live = LiveStream(self)
        self.update_bbox_label()
        self.update_camera_controls()
        self.setStyleSheet(STYLE)
        previous = str(self.settings.value("lab_latest_prediction_path", ""))
        if previous and Path(previous).is_file():
            try:
                with np.load(previous, allow_pickle=False) as data:
                    self.lab.set_latest(data, previous)
            except (OSError, ValueError, KeyError):
                pass

    def build_ui(self):
        root = QtWidgets.QWidget()
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setWidget(root)
        self.workspace_tabs = QtWidgets.QTabWidget()
        self.workspace_tabs.addTab(scroll, "SAM 3D 重建")
        self.setCentralWidget(self.workspace_tabs)
        layout = QtWidgets.QVBoxLayout(root)
        layout.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        layout.setContentsMargins(24, 20, 24, 18)
        layout.setSpacing(16)
        header = QtWidgets.QHBoxLayout()
        title_col = QtWidgets.QVBoxLayout()
        title = QtWidgets.QLabel('SAM 3D Body Studio')
        title.setObjectName('title')
        title_col.addWidget(title)
        subtitle = QtWidgets.QLabel('从一张图片重建人体 · 本地 GPU 运行')
        subtitle.setObjectName('muted')
        title_col.addWidget(subtitle)
        header.addLayout(title_col)
        header.addStretch()
        self.demo_btn = QtWidgets.QPushButton('载入示例')
        self.demo_btn.clicked.connect(lambda: self.load_image(SAMPLE, [732, 114, 1523, 1451]))
        self.open_btn = QtWidgets.QPushButton('选择图片…')
        self.open_btn.clicked.connect(self.choose_image)
        header.addWidget(self.demo_btn)
        header.addWidget(self.open_btn)
        layout.addLayout(header)
        self.profile_panel=ProfilePanel(self.initial_config)
        self.profile_toggle=QtWidgets.QToolButton()
        self.profile_toggle.setText('启动方案设置 ▾')
        self.profile_toggle.setCheckable(True)
        self.profile_toggle.setChecked(True)
        self.profile_toggle.toggled.connect(self.profile_panel.setVisible)
        layout.addWidget(self.profile_toggle)
        self.backend_combo=self.profile_panel.backend
        self.hand_check=self.profile_panel.hand
        self.profile_panel.changed.connect(self.profile_changed)
        layout.addWidget(self.profile_panel)
        model_row=QtWidgets.QHBoxLayout()
        self.preload_btn=QtWidgets.QPushButton('应用并预热')
        self.preload_btn.clicked.connect(self.preload_model)
        self.release_btn=QtWidgets.QPushButton('释放模型')
        self.release_btn.clicked.connect(self.release_model)
        self.restore_profile_btn=QtWidgets.QPushButton('恢复已加载方案')
        self.restore_profile_btn.clicked.connect(lambda:self.profile_panel.set_config(self.active_config) if self.active_config else None)
        history=QtWidgets.QPushButton('运行历史 / 对照')
        history.clicked.connect(lambda:show_history(self))
        self.model_label=QtWidgets.QLabel('模型未加载 · 选择方案后点击应用并预热')
        self.model_label.setWordWrap(True);self.model_label.setObjectName('muted')
        for widget in [self.preload_btn,self.release_btn,self.restore_profile_btn,history]:model_row.addWidget(widget)
        model_row.addWidget(self.model_label,1)
        layout.addLayout(model_row)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        left = QtWidgets.QFrame()
        left.setObjectName('card')
        ll = QtWidgets.QVBoxLayout(left)
        ll.setContentsMargins(16, 16, 16, 16)
        ll.addWidget(QtWidgets.QLabel('01  输入图片'))
        self.source_name = QtWidgets.QLabel('尚未选择图片')
        self.source_name.setObjectName('muted')
        self.source_name.setWordWrap(True)
        ll.addWidget(self.source_name)
        self.canvas = ImageCanvas(selectable=True)
        self.canvas.fileDropped.connect(self.load_image)
        self.canvas.bboxChanged.connect(self.bbox_changed)
        self.camera_panel = QtWidgets.QWidget()
        camera_layout = QtWidgets.QVBoxLayout(self.camera_panel)
        camera_layout.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        camera_layout.setContentsMargins(0, 0, 0, 0)
        camera_row = QtWidgets.QHBoxLayout()
        self.camera_btn = QtWidgets.QPushButton('打开相机 / 重拍')
        self.camera_btn.clicked.connect(self.open_camera)
        self.camera_close_btn = QtWidgets.QPushButton('关闭相机')
        self.camera_close_btn.clicked.connect(self.close_camera)
        camera_row.addWidget(self.camera_btn)
        camera_row.addWidget(self.camera_close_btn)
        camera_layout.addLayout(camera_row)
        capture_row = QtWidgets.QHBoxLayout()
        capture_row.addWidget(QtWidgets.QLabel('延时'))
        self.delay_combo = QtWidgets.QComboBox()
        for seconds in [3, 5, 10, 15]:
            self.delay_combo.addItem(f'{seconds} 秒', seconds)
        self.delay_combo.setCurrentIndex(2)
        capture_row.addWidget(self.delay_combo)
        self.capture_btn = QtWidgets.QPushButton('拍照并重建')
        self.capture_btn.clicked.connect(self.capture_camera)
        capture_row.addWidget(self.capture_btn, 1)
        camera_layout.addLayout(capture_row)
        self.auto_reconstruct = QtWidgets.QCheckBox('拍照后自动重建')
        self.auto_reconstruct.setChecked(True)
        self.auto_reconstruct.toggled.connect(lambda on: self.capture_btn.setText('拍照并重建' if on else '倒计时拍照'))
        self.auto_reconstruct.toggled.connect(self.update_camera_controls)
        camera_layout.addWidget(self.auto_reconstruct)
        self.stream_btn = QtWidgets.QPushButton('开始连续推理')
        self.stream_btn.setToolTip('先应用并预热方案，再打开相机。仅处理最新帧，不保存视频或逐帧对照图；可随时停止。建议演示最快方案。')
        self.stream_btn.clicked.connect(lambda: self.live.toggle())
        camera_layout.addWidget(self.stream_btn)
        mode_row = QtWidgets.QHBoxLayout()
        mode_row.addWidget(QtWidgets.QLabel('连续显示'))
        self.stream_mode = QtWidgets.QComboBox()
        self.stream_mode.addItem('彩色图叠加（网格 / 骨架）', 'overlay')
        self.stream_mode.addItem('可旋转 3D 预览', '3d')
        self.stream_mode.addItem('MHR 身体骨架预览', 'mhr')
        self.stream_mode.addItem('SMPL 身体骨架预览', 'smpl')
        saved_mode = self.settings.value('stream_display_mode', 'overlay')
        self.stream_mode.setCurrentIndex(max(0, self.stream_mode.findData(saved_mode)))
        self.stream_mode.setToolTip('开始前选择。彩色叠加显示对应推理输入帧，无需重新预热；3D 模式可旋转查看。')
        self.stream_mode.currentIndexChanged.connect(lambda: self.settings.setValue('stream_display_mode', self.stream_mode.currentData()))
        mode_row.addWidget(self.stream_mode, 1)
        camera_layout.addLayout(mode_row)
        self.intrinsics_check = QtWidgets.QCheckBox('使用 Gemini 真实内参')
        self.intrinsics_check.setChecked(self.settings.value('use_gemini_intrinsics', True, type=bool))
        self.intrinsics_check.setToolTip('照片元数据优先；否则 1280×800 图片使用本机 Gemini 固定内参。其他相机图片请关闭。切换无需重新预热。')
        self.intrinsics_check.toggled.connect(self.intrinsics_changed)
        camera_layout.addWidget(self.intrinsics_check)
        self.intrinsics_label = QtWidgets.QLabel()
        self.intrinsics_label.setWordWrap(True)
        self.intrinsics_label.setObjectName('muted')
        camera_layout.addWidget(self.intrinsics_label)
        self.camera_label = QtWidgets.QLabel('Gemini 336L · 请让头脚完整入画')
        self.camera_label.setObjectName('muted')
        self.camera_label.setWordWrap(True)
        camera_layout.addWidget(self.camera_label)
        ll.addWidget(self.camera_panel)
        ll.addWidget(self.canvas, 1)
        self.bbox_label = QtWidgets.QLabel('可拖动框选一人；未框选时使用整张图片')
        self.bbox_label.setObjectName('muted')
        self.bbox_label.setWordWrap(True)
        ll.addWidget(self.bbox_label)
        self.reset_bbox_btn = QtWidgets.QPushButton('清除人物框 / 使用整图')
        self.reset_bbox_btn.clicked.connect(self.reset_bbox)
        ll.addWidget(self.reset_bbox_btn)
        splitter.addWidget(left)

        right = QtWidgets.QFrame()
        right.setObjectName('card')
        rl = QtWidgets.QVBoxLayout(right)
        rl.setContentsMargins(16, 16, 16, 16)
        rl.addWidget(QtWidgets.QLabel('02  重建结果'))
        self.tabs = QtWidgets.QTabWidget()
        view_page = QtWidgets.QWidget()
        vl = QtWidgets.QVBoxLayout(view_page)
        vl.setContentsMargins(0, 12, 0, 0)
        tools = QtWidgets.QHBoxLayout()
        self.view_buttons = {}
        for label, azimuth in [('正面', -90), ('背面', 90), ('左侧', 180), ('右侧', 0)]:
            button = QtWidgets.QPushButton(label)
            button.clicked.connect(lambda checked=False, a=azimuth: self.viewer.preset(a))
            tools.addWidget(button)
            self.view_buttons[label] = button
        self.reset_view_btn = QtWidgets.QPushButton('复位')
        self.reset_view_btn.clicked.connect(lambda: self.viewer.reset_view())
        tools.addWidget(self.reset_view_btn)
        tools.addStretch()
        vl.addLayout(tools)
        self.viewer = BodyViewer()
        vl.addWidget(self.viewer, 1)
        options = QtWidgets.QHBoxLayout()
        self.wire_check = QtWidgets.QCheckBox('线框')
        self.wire_check.toggled.connect(self.set_wireframe)
        self.rotate_check = QtWidgets.QCheckBox('自动旋转')
        self.rotate_check.toggled.connect(lambda active: self.rotation.start() if active else self.rotation.stop())
        options.addWidget(self.wire_check)
        options.addWidget(self.rotate_check)
        options.addStretch()
        help_label = QtWidgets.QLabel('左键旋转 · 滚轮缩放 · 中键平移')
        help_label.setObjectName('muted')
        options.addWidget(help_label)
        vl.addLayout(options)
        self.tabs.addTab(view_page, '3D 预览')
        self.comparison = ImageCanvas()
        self.tabs.addTab(self.comparison, '图像对照')
        self.overlay = OverlayView()
        self.overlay.renderFailed.connect(lambda message: (self.live.abort('叠加渲染失败：'+message), self.status.setText('叠加渲染失败：'+message)))
        self.tabs.addTab(self.overlay, '彩色叠加')
        self.smpl_view=BodyViewer()
        self.smpl_view.empty.setText('启用 SMPL 转换并预热后显示 24 关节身体骨架')
        self.tabs.addTab(self.smpl_view, 'SMPL 骨架')
        rl.addWidget(self.tabs, 1)
        self.motion = MotionPanel(self)
        rl.addWidget(self.motion)
        self.result_label = QtWidgets.QLabel('等待重建 · 支持 JPG / PNG / WEBP 等图片')
        self.result_label.setObjectName('muted')
        self.result_label.setWordWrap(True)
        rl.addWidget(self.result_label)
        export_row = QtWidgets.QHBoxLayout()
        self.folder_btn = QtWidgets.QPushButton('打开结果目录')
        self.folder_btn.clicked.connect(self.open_result_folder)
        self.folder_btn.setEnabled(False)
        self.export_btn = QtWidgets.QPushButton('另存网格 PLY…')
        self.export_btn.clicked.connect(self.export_mesh)
        self.export_btn.setEnabled(False)
        export_row.addWidget(self.folder_btn)
        export_row.addWidget(self.export_btn)
        export_row.addStretch()
        rl.addLayout(export_row)
        splitter.addWidget(right)
        splitter.setSizes([440, 790])
        layout.addWidget(splitter, 1)

        self.log_box = QtWidgets.QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumBlockCount(2500)
        self.log_box.setMaximumHeight(150)
        self.log_box.hide()
        layout.addWidget(self.log_box)
        bottom = QtWidgets.QHBoxLayout()
        self.status = QtWidgets.QLabel('请选择一张清晰的单人全身图片')
        self.status.setWordWrap(True)
        bottom.addWidget(self.status, 1)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setFixedWidth(100)
        self.progress.hide()
        bottom.addWidget(self.progress)
        log_btn = QtWidgets.QPushButton('运行日志')
        log_btn.setCheckable(True)
        log_btn.toggled.connect(self.log_box.setVisible)
        bottom.addWidget(log_btn)
        self.cancel_btn = QtWidgets.QPushButton('取消')
        self.cancel_btn.clicked.connect(self.cancel_run)
        self.cancel_btn.setEnabled(False)
        bottom.addWidget(self.cancel_btn)
        self.run_btn = QtWidgets.QPushButton('开始重建')
        self.run_btn.setObjectName('primary')
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self.start_reconstruction)
        self.benchmark_btn=QtWidgets.QPushButton('同图测速…')
        self.benchmark_btn.clicked.connect(self.benchmark_dialog)
        bottom.addWidget(self.benchmark_btn)
        bottom.addWidget(self.run_btn)
        layout.addLayout(bottom)
        self.lab = MhrLab(self, BodyViewer)
        self.workspace_tabs.addTab(self.lab, "MHR 参数实验室")
        self.rgbd = RgbdValidation(self)
        self.workspace_tabs.addTab(self.rgbd, "RGB-D 几何验证")

    def update_camera_controls(self):
        busy = self.inference_busy
        active = self.camera_state not in ('off', 'photo')
        waiting = self.camera_state in ('countdown', 'stopping', 'starting')
        rgbd_camera=hasattr(self,'rgbd') and (self.rgbd.mode in ('capture','preview') or self.rgbd.pending_camera_action is not None)
        self.camera_btn.setEnabled(not busy and not active and not rgbd_camera)
        self.camera_close_btn.setEnabled(active and self.camera_state != 'stopping')
        self.capture_btn.setEnabled(self.camera_state == 'preview' and not busy)
        self.delay_combo.setEnabled(not waiting and not busy)
        self.auto_reconstruct.setEnabled(not waiting and not busy)
        self.intrinsics_check.setEnabled(not waiting and not busy)
        self.open_btn.setEnabled(not active and not busy)
        self.demo_btn.setEnabled(not active and not busy)
        self.canvas.selection_enabled = not waiting and not busy
        self.reset_bbox_btn.setEnabled(not waiting and not busy)
        self.run_btn.setEnabled(not active and not busy and self.source_path is not None and self.model_state != 'stopping')
        self.cancel_btn.setEnabled(busy or self.camera_state == 'countdown')
        config=self.draft_config()
        dirty=self.active_config is not None and config != self.active_config
        can_apply=config is not None and not busy and not waiting and self.model_state in ('off','ready')
        self.preload_btn.setEnabled(can_apply and (self.model_state=='off' or dirty))
        self.release_btn.setEnabled(self.model_state in ('loading','ready') and not busy)
        self.restore_profile_btn.setEnabled(self.model_state=='ready' and not busy and not waiting)
        self.profile_panel.setEnabled(not busy and not waiting and self.model_state not in ('loading','stopping'))
        can_run=config is not None and self.model_state in ('off','ready') and not (self.model_state=='ready' and dirty)
        try:
            _, camera_origin = self.selected_camera_intrinsics()
            self.intrinsics_label.setText(camera_origin['label'])
            self.intrinsics_label.setStyleSheet('')
        except (ValueError, OSError, KeyError, TypeError) as exc:
            self.intrinsics_label.setText('内参不可用：'+str(exc)+' · 可关闭开关使用默认估计')
            self.intrinsics_label.setStyleSheet('color:#fca5a5')
            can_run = False
        self.run_btn.setEnabled(not active and not busy and self.source_path is not None and can_run)
        self.benchmark_btn.setEnabled(not active and not busy and self.source_path is not None and can_run)
        self.capture_btn.setEnabled(self.camera_state=='preview' and not busy and self.model_state!='stopping'
                                    and (not self.auto_reconstruct.isChecked() or can_run))

        streaming = self.live.running
        self.stream_mode.setEnabled(not streaming and not busy)
        self.tabs.tabBar().setEnabled(not streaming)
        self.stream_btn.setText('停止连续推理' if self.live.active else ('正在停止…' if streaming else '开始连续推理'))
        self.stream_btn.setEnabled(self.live.active or (not streaming and not busy and self.camera_state=='preview'
                                   and self.model_state=='ready' and not dirty and can_run))

        if self.sam3_blocks_models():
            for button in (self.preload_btn,self.run_btn,self.benchmark_btn,self.stream_btn):button.setEnabled(False)
        if hasattr(self,'rgbd') and self.rgbd.batch.active:
            for widget in (self.preload_btn,self.release_btn,self.run_btn,self.benchmark_btn,self.stream_btn,self.camera_btn,self.open_btn,self.demo_btn,self.profile_panel):widget.setEnabled(False)

    def selected_camera_intrinsics(self):
        width, height = (self.canvas.image.width(), self.canvas.image.height()) if self.source_path else (1280, 800)
        return select_camera_intrinsics(self.intrinsics_check.isChecked(), width, height,
                                        self.capture_metadata, self.capture_metadata_error)

    def intrinsics_changed(self):
        self.settings.setValue('use_gemini_intrinsics', self.intrinsics_check.isChecked())
        self.update_camera_controls()

    def draft_config(self):
        try:return self.profile_panel.config()
        except ValueError:return None

    def profile_changed(self):
        config=self.draft_config()
        if config is not None:
            self.settings.setValue('launch_config',json.dumps(config.to_dict()))
            if self.model_state=='off':self.backend_id=config.backend
        self.update_bbox_label()
        if self.model_state=='ready':
            self.model_label.setText('已就绪 · '+self.active_config.summary() if config==self.active_config else
                                     '配置已修改 · 点击应用并预热，或恢复已加载方案')
        self.update_camera_controls()

    def change_backend(self):
        # Kept for external automation; changes are applied only by preload_model.
        self.profile_changed()

    def apply_backend(self,target):
        self.profile_panel.set_config(PRESETS['标准版身体'] if target=='standard' else PRESETS['演示最快'])

    def benchmark_dialog(self):
        dialog=QtWidgets.QDialog(self);dialog.setWindowTitle('同图测速')
        form=QtWidgets.QFormLayout(dialog)
        warm=QtWidgets.QSpinBox();warm.setRange(1,30);warm.setValue(10)
        runs=QtWidgets.QSpinBox();runs.setRange(1,200);runs.setValue(50)
        form.addRow('额外同图预热次数',warm);form.addRow('正式测量次数',runs)
        note=QtWidgets.QLabel('固定当前图片与人物选择框。排除加载、读图、导出与显示时间；取消会释放模型并保留部分记录。')
        note.setWordWrap(True);form.addRow(note)
        buttons=QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok|QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept);buttons.rejected.connect(dialog.reject);form.addRow(buttons)
        if dialog.exec_()==QtWidgets.QDialog.Accepted:
            self.start_reconstruction(True,warm.value(),runs.value())

    def open_camera(self):
        if self.camera.active or self.inference_busy or self.rgbd.mode in ('capture','preview') or self.rgbd.pending_camera_action is not None or self.rgbd.batch.active:
            return
        self.camera_job = None
        self.pending_photo = None
        self.capture_metadata = None
        self.capture_metadata_error = None
        self.camera_error = None
        self.source_path = None
        self.clear_result()
        self.canvas.set_image(QtGui.QImage())
        self.bbox_changed()
        self.source_name.setText('实时相机 · 1280 × 800')
        self.camera_state = 'starting'
        self.camera_label.setText('正在连接和预热相机…')
        self.status.setText('相机预热中，请稍候')
        self.update_camera_controls()
        self.camera.start()

    def close_camera(self):
        self.live.stop('相机关闭，连续推理已停止')
        self.camera_job = None
        self.pending_photo = None
        self.countdown_clock.stop()
        self.canvas.overlay = ''
        self.canvas.update()
        self.camera_state = 'stopping'
        self.camera_label.setText('正在释放相机…')
        self.update_camera_controls()
        self.camera.stop()

    def camera_frame(self, image):
        if self.camera_state in ('starting', 'preview', 'countdown'):
            self.canvas.set_preview(image)
            self.live.poll()

    def capture_camera(self):
        if self.camera_state != 'preview':
            return
        self.camera_job = uuid.uuid4().hex
        self.photo_bbox = list(self.canvas.bbox) if self.canvas.bbox else None
        self.photo_auto = self.auto_reconstruct.isChecked()
        delay = self.delay_combo.currentData()
        directory = ROOT/'outputs/camera'/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        self.camera_state = 'countdown'
        self.countdown_deadline = time.monotonic()+delay
        self.camera.send('capture', job=self.camera_job, delay=delay, directory=str(directory))
        self.countdown_clock.start()
        self.countdown_tick()
        self.update_camera_controls()

    def countdown_tick(self):
        remaining = max(0, math.ceil(self.countdown_deadline-time.monotonic()))
        self.canvas.overlay = str(remaining) if remaining else '…'
        self.canvas.update()
        self.status.setText(f'请摆好姿势 · {remaining} 秒后拍照' if remaining else '正在等待新帧并保存照片…')

    def camera_event(self, event):
        kind = event['type']
        if kind == 'ready' and self.camera_state == 'starting':
            self.camera_state = 'preview'
            self.capture_metadata = event['metadata']
            self.camera_label.setText('Gemini 336L · 1280 × 800 / 30 FPS · 请让头脚完整入画')
            self.status.setText('相机已就绪，可框选人物后倒计时拍照')
            self.update_camera_controls()
        elif kind == 'countdown' and event['job'] == self.camera_job and self.camera_state == 'countdown':
            self.countdown_deadline = time.monotonic()+event['delay']
        elif kind == 'photo' and event['job'] == self.camera_job and self.camera_state == 'countdown':
            self.pending_photo = event
            self.countdown_clock.stop()
            self.canvas.overlay = ''
            self.camera_state = 'stopping'
            self.camera_label.setText('照片已保存，正在释放相机…')
            self.camera.stop()
            self.update_camera_controls()
        elif kind == 'status':
            self.camera_label.setText(event['message'])
        elif kind == 'error':
            self.live.stop('相机出错，连续推理已停止')
            self.camera_error = event['message']
            self.log_box.appendPlainText('相机：'+self.camera_error)
            self.pending_photo = None
            self.camera_job = None
            self.countdown_clock.stop()
            self.canvas.overlay = ''
            self.canvas.update()
            self.camera_state = 'stopping'
            self.camera_label.setText(self.camera_error)
            self.status.setText('相机出错：'+self.camera_error)
            self.camera.stop()
            self.update_camera_controls()

    def camera_stopped(self):
        if self.closing:
            return
        photo = self.pending_photo
        self.pending_photo = None
        self.camera_state = 'off'
        self.canvas.overlay = ''
        self.countdown_clock.stop()
        if photo and photo['job'] == self.camera_job:
            self.load_image(photo['path'], self.photo_bbox)
            self.capture_metadata = photo['metadata']
            self.capture_metadata_error = None
            self.camera_state = 'photo'
            self.camera_label.setText('照片已保存 · 相机已释放 · 可重拍')
            self.status.setText('拍照完成，可重建或重拍')
            self.update_camera_controls()
            if self.photo_auto:
                self.start_reconstruction()
        else:
            self.source_path = None
            self.canvas.set_image(QtGui.QImage())
            self.camera_label.setText(self.camera_error or '相机已关闭，可重新打开')
            self.status.setText(('相机出错：'+self.camera_error) if self.camera_error else '相机已关闭')
            self.update_camera_controls()

    def choose_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, '选择人体图片', str(Path.home()),
            '图片 (*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff);;所有文件 (*)')
        if path:
            self.load_image(path)

    def load_image(self, path, bbox=None):
        if self.inference_busy or self.camera_state not in ('off', 'photo'):
            return
        reader = QtGui.QImageReader(str(path))
        reader.setAutoTransform(True)
        image = reader.read()
        if image.isNull():
            QtWidgets.QMessageBox.warning(self, '无法打开图片', '请选择有效的图片文件。\n'+reader.errorString())
            return
        self.source_path = Path(path)
        self.capture_metadata = None
        self.capture_metadata_error = None
        sidecar = self.source_path.parent/'camera.json'
        if sidecar.is_file():
            try:
                metadata = json.loads(sidecar.read_text())
                if not isinstance(metadata, dict):
                    raise ValueError('元数据必须是 JSON 对象')
                self.capture_metadata = metadata
            except (ValueError, OSError) as exc:
                self.capture_metadata_error = str(exc)
        self.canvas.set_image(image.convertToFormat(QtGui.QImage.Format_RGB888))
        self.canvas.bbox = bbox
        self.bbox_changed()
        self.source_name.setText(f'{self.source_path.name}  ·  {image.width()} × {image.height()}')
        self.clear_result()
        self.run_btn.setEnabled(True)
        self.update_camera_controls()
        self.status.setText('图片已载入，可框选目标人物后开始重建')

    def bbox_changed(self):
        if self.result_dir is not None:
            self.clear_result()
            self.status.setText('人物框已更新，请重新重建')
        self.update_bbox_label()

    def update_bbox_label(self):
        if self.canvas.bbox:
            self.bbox_label.setText(('Fast 选人框：' if self.backend_id == 'fast' else '人物框：')+
                                   ', '.join(str(round(v)) for v in self.canvas.bbox))
        else:
            self.bbox_label.setText('Fast 自动检测单人；多人图片请框选目标' if self.backend_id == 'fast' else
                                   '可拖动框选一人；未框选时使用整张图片')

    def reset_bbox(self):
        self.canvas.bbox = None
        self.canvas.update()
        self.bbox_changed()

    def clear_result(self):
        for process in list(self.render_processes.values()):
            if process.state() != QtCore.QProcess.NotRunning:
                process.kill()
        self.viewer.clear_mesh()
        self.smpl_view.clear_mesh()
        self.smpl_view.stream_anchor=None
        self.overlay.clear_frame()
        self.motion.clear_pose()
        self.comparison.set_image(QtGui.QImage())
        self.rotate_check.setChecked(False)
        self.result_dir = None
        self.report = None
        self.folder_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.result_label.setText('等待当前图片的重建结果')
        self.tabs.setTabText(1, '图像对照')

    def set_busy(self, busy):
        self.inference_busy = busy
        for widget in [self.open_btn, self.demo_btn, self.hand_check, self.reset_bbox_btn]:
            widget.setEnabled(not busy)
        self.canvas.selection_enabled = not busy
        self.run_btn.setEnabled(not busy and self.source_path is not None)
        self.cancel_btn.setEnabled(busy)
        self.progress.setVisible(busy)
        self.progress.setRange(0, 0 if busy else 1)
        if busy:
            self.clock.start(250)
        else:
            self.clock.stop()
        self.update_camera_controls()

    def start_reconstruction(self, benchmark=False, warmup_runs=10, runs=50):
        if self.sam3_blocks_models():
            self.status.setText("SAM3 正在分割，暂不能启动人体重建");return
        request_started = time.monotonic()
        if self.camera_state not in ('off', 'photo'):
            return
        if self.source_path is None or self.inference_busy or self.model_state == 'stopping':
            return
        if not self.check_backend():
            return
        try:
            camera_intrinsics, camera_origin = self.selected_camera_intrinsics()
        except (ValueError, OSError, KeyError, TypeError) as exc:
            self.status.setText('内参不可用：'+str(exc));return
        config=self.draft_config()
        if self.model_state=='ready' and config != self.active_config:
            self.status.setText('配置已修改，请先应用并预热');return
        if benchmark:
            for process in list(self.render_processes.values()):
                process.kill();process.waitForFinished(3000)
        self.job_progress=''
        self.running_dir = ROOT/'outputs/gui'/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        self.running_dir.mkdir(parents=True)
        input_path = self.running_dir/'input.png'
        if not self.canvas.image.save(str(input_path)):
            QtWidgets.QMessageBox.warning(self, '无法保存输入', '请检查输出目录是否可写。')
            return
        if self.capture_metadata:
            (self.running_dir/'camera.json').write_text(json.dumps(self.capture_metadata, ensure_ascii=False, indent=2))
        (self.running_dir/'request.json').write_text(json.dumps({
            'source_image': str(self.source_path), 'normalized_image': str(input_path),
            'bbox_xyxy': self.canvas.bbox, 'hand_refinement': self.hand_check.isChecked(),
            'backend_id': config.backend,'launch_config':config.to_dict(),'config_id':config.config_id,
            'benchmark':bool(benchmark), 'use_gemini_intrinsics': self.intrinsics_check.isChecked(),
            'camera_intrinsics': camera_intrinsics, 'camera_intrinsics_origin': camera_origin}, ensure_ascii=False, indent=2))
        self.clear_result()
        self.log_box.clear()
        self.cancelled = False
        self.started_at = request_started
        self.set_busy(True)
        self.active_job = uuid.uuid4().hex
        self.pending_model_request = dict(type='benchmark' if benchmark else 'run', job=self.active_job, image=str(input_path),
            output=str(self.running_dir), bbox=self.canvas.bbox,
            inference_type=config.mode,config_id=config.config_id,warmup_runs=warmup_runs,runs=runs,
            camera_intrinsics=camera_intrinsics, camera_intrinsics_origin=camera_origin)
        if self.model_state == 'ready':
            self.dispatch_model_request()
        else:
            self.preload_model()
        self.tick()

    def sam3_blocks_models(self):
        return hasattr(self,'rgbd') and self.rgbd.segmenting

    def preload_model(self):
        if self.sam3_blocks_models():
            self.status.setText('SAM3 分割尚未退出，已阻止人体模型加载；请等待分割结束或取消。');return
        if self.model_state=='ready':
            if self.draft_config()==self.active_config:return
            if not self.check_backend():return
            self.pending_launch=True
            self.release_model()
            return
        if self.model_state!='off' or not self.check_backend():return
        config=self.draft_config()
        self.backend_id=config.backend
        self.active_config=config
        self.model_state='loading'
        self.model_failure=None
        self.model_buffer=''
        self.preload_started=time.monotonic()
        self.model_session=uuid.uuid4().hex
        session_dir=ROOT/'outputs/sessions';session_dir.mkdir(parents=True,exist_ok=True)
        config_path=session_dir/(self.model_session+'.json');config.save(config_path)
        self.model_log_path=session_dir/(self.model_session+'.log')
        self.model_log_path.write_text('启动配置：'+str(config_path)+'\n',encoding='utf-8')
        self.model_label.setText('正在加载并预热 · '+config.summary())
        self.update_camera_controls()
        self.process.setWorkingDirectory(str(ROOT))
        env=QtCore.QProcessEnvironment.systemEnvironment()
        env.remove('QT_QPA_PLATFORM_PLUGIN_PATH');env.remove('QT_QPA_FONTDIR')
        env.insert('BODYSTUDIO_SESSION',self.model_session)
        env.insert('BODYSTUDIO_CONFIG',str(config_path))
        self.process.setProcessEnvironment(env)
        self.process.start('/bin/bash',[str(BACKENDS[self.backend_id]['launcher'])])

    def check_backend(self):
        try:config=self.profile_panel.config()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self,'启动方案无效',str(exc));return False
        missing=missing_resources(config.backend,config)
        if missing:
            QtWidgets.QMessageBox.warning(self,'后端资源不可用','请确认 Data 已挂载并检查：\n'+'\n'.join(missing))
            return False
        return True

    def dispatch_model_request(self):
        if self.pending_model_request and self.model_state == 'ready':
            if self.pending_model_request['config_id'] != self.active_config.config_id:
                self.fail_model_job('任务配置与已加载模型不一致');return
            self.pending_model_request.update(backend_id=self.backend_id, session_id=self.model_session)
            self.process.write((json.dumps(self.pending_model_request)+'\n').encode())
            self.pending_model_request = None

    def release_model(self):
        if self.inference_busy or self.model_state not in ('loading','ready'):
            return
        self.model_state = 'stopping'
        self.model_label.setText('正在释放模型…')
        self.update_camera_controls()
        self.process.terminate()
        self.kill_timer.start(2500)

    def tick(self):
        if self.live.running:
            return
        stage = '正在加载并预热模型…' if self.model_state == 'loading' else (self.job_progress or '正在重建人体…')
        self.status.setText(('正在取消…' if self.cancelled else stage)+f'  {time.monotonic()-self.started_at:.1f} 秒')

    def read_log(self):
        chunk=bytes(self.process.readAllStandardOutput()).decode('utf-8', errors='replace')
        if chunk and self.model_log_path:
            try:
                with self.model_log_path.open('a',encoding='utf-8') as file:file.write(chunk)
            except OSError as exc:
                self.log_box.appendPlainText('无法保存模型日志：'+str(exc));self.model_log_path=None
        self.model_buffer += chunk
        while '\n' in self.model_buffer:
            line, self.model_buffer = self.model_buffer.split('\n', 1)
            if line.startswith('@@SAM3D '):
                try:
                    self.model_event(json.loads(line[8:]))
                except (ValueError, KeyError, OSError) as exc:
                    self.log_box.appendPlainText(str(exc))
                    self.fail_model_job('无法读取模型结果')
            elif line:
                self.log_box.appendPlainText(line.replace('\r','\n'))
                if '显存互斥保护：' in line:self.model_failure='显存互斥保护：'+line.split('显存互斥保护：',1)[1]

    def model_event(self, event):
        if event.get('backend_id') != self.backend_id or event.get('session_id') != self.model_session:
            return
        if self.active_config is None or event.get('config_id') != self.active_config.config_id:
            return
        kind = event['type']
        if self.live.handle_model_event(event):
            return
        if kind == 'progress' and event.get('job')==self.active_job and self.inference_busy:
            self.job_progress=event['message'];self.tick()
        elif kind == 'stage' and self.model_state == 'loading':
            self.model_label.setText(BACKENDS[self.backend_id]['label']+' · '+event['message'])
        elif kind == 'ready' and self.model_state == 'loading':
            if event.get('actual_config') != self.active_config.to_dict():
                self.model_failure='后端实际配置与所选方案不一致'
                if self.inference_busy:self.fail_model_job(self.model_failure)
                self.release_model();return
            self.smpl_asset_hashes=event.get('smpl_assets')
            self.model_state = 'ready'
            self.profile_toggle.setChecked(False)
            self.model_label.setText(f'已就绪 · {self.active_config.summary()} · 加载及预热 {time.monotonic()-self.preload_started:.1f} 秒')
            self.update_camera_controls()
            self.dispatch_model_request()
        elif kind == 'mesh' and event['job'] == self.active_job and self.inference_busy and not self.cancelled:
            self.show_mesh(event)
        elif kind == 'error' and event.get('job') == self.active_job:
            self.fail_model_job('重建失败：'+event['message'])
        elif kind == 'fatal':
            self.live.abort('模型出错，连续推理已停止')
            self.model_failure = event['message']
            self.log_box.appendPlainText(event['message'])
            self.log_box.show()
            if self.model_log_path:self.log_box.appendPlainText('完整预加载日志：'+str(self.model_log_path))
            if self.inference_busy:
                self.fail_model_job('模型加载失败：'+event['message'])

    def fail_model_job(self, message):
        self.pending_model_request = None
        self.active_job = None
        self.set_busy(False)
        self.status.setText(message)
        self.log_box.show()
        if self.running_dir:
            (self.running_dir/'run.log').write_text(self.log_box.toPlainText())
        self.reconstructionFinished.emit(False)

    def show_mesh(self, event):
        directory = Path(event['output'])
        if directory != self.running_dir:
            return
        with np.load(directory/'person_00.npz', allow_pickle=False) as data:
            vertices, faces = data['pred_vertices'], data['faces']
            if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
                raise ValueError('网格顶点无效')
            if faces.ndim != 2 or faces.shape[1] != 3 or faces.min() < 0 or faces.max() >= len(vertices):
                raise ValueError('网格拓扑无效')
            self.viewer.set_mesh(vertices, faces, self.wire_check.isChecked())
            self.motion.set_pose(data, dict(source_path=str(self.source_path), report=event['report'],
                                           inference_ms=event['report']['inference_seconds']*1000))
            self.lab.set_latest(data, str(directory/'person_00.npz'))
            self.settings.setValue('lab_latest_prediction_path', str(directory/'person_00.npz'))
            k = event['report'].get('camera_intrinsics_used', {}).get('matrix')
            if k is not None:
                self.overlay.set_frame(QtGui.QImage(str(self.source_path)), vertices, data['pred_cam_t'], faces, k, 0)
        self.report = event['report']
        self.report['gui_mesh_loaded_seconds'] = time.monotonic()-self.started_at
        (directory/'inference.json').write_text(json.dumps(self.report, ensure_ascii=False, indent=2))
        (directory/'run.log').write_text(self.log_box.toPlainText())
        record_report(directory,self.report)
        self.result_dir = directory
        self.tabs.setCurrentIndex(0)
        self.folder_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        self.result_label.setText(f'{BACKENDS[self.report["backend_id"]]["label"]} · {len(vertices):,} 顶点 · {len(faces):,} 面 · '+
            f'3D 就绪 {self.report["gui_mesh_loaded_seconds"]:.2f} 秒 · 模型计算 {self.report["inference_seconds"]:.2f} 秒')
        if self.report.get('benchmark'):
            bench=self.report['benchmark']
            self.result_label.setText(f'同图测速 {bench["completed_runs"]} 次 · 平均 {bench["mean_ms"]:.1f} ms · 中位数 {bench["median_ms"]:.1f} ms · P95 {bench["p95_ms"]:.1f} ms')
        camera_source = self.report.get('camera_intrinsics_used', {}).get('source')
        self.result_label.setText(self.result_label.text()+' · '+('真实相机内参' if camera_source=='provided_rgb' else '默认估计内参'))
        self.set_busy(False)
        self.status.setText('3D 已就绪，可旋转和导出 · 图像对照正在后台生成')
        if self.report.get('hand_warning'):
            self.status.setText(self.status.text()+' · '+self.report['hand_warning'])
        self.tabs.setTabText(1, '图像对照 · 生成中')
        self.comparison.set_image(QtGui.QImage())
        self.start_comparison(directory, self.active_job)
        self.reconstructionFinished.emit(True)

    def start_comparison(self, directory, job):
        process = QtCore.QProcess(self)
        self.render_processes[job] = process
        process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(lambda: self.read_render_log(process, directory))
        process.finished.connect(lambda code, status: self.comparison_finished(process, directory, job, code, status))
        process.errorOccurred.connect(lambda error: self.comparison_error(process, directory, job, error))
        process.setWorkingDirectory(str(ROOT))
        env = QtCore.QProcessEnvironment.systemEnvironment()
        env.remove('QT_QPA_PLATFORM_PLUGIN_PATH')
        env.remove('QT_QPA_FONTDIR')
        process.setProcessEnvironment(env)
        process.start('/bin/bash', [str(ROOT/'launch_render.sh'), str(directory)])

    def read_render_log(self, process, directory):
        text = bytes(process.readAllStandardOutput()).decode(errors='replace')
        if text:
            with (directory/'comparison.log').open('a') as file:
                file.write(text)

    def comparison_error(self, process, directory, job, error):
        if error == QtCore.QProcess.FailedToStart:
            self.comparison_finished(process, directory, job, 1, QtCore.QProcess.CrashExit)

    def comparison_finished(self, process, directory, job, code, exit_status):
        self.read_render_log(process, directory)
        self.render_processes.pop(job, None)
        if code != 0:
            try:
                report = json.loads((directory/'inference.json').read_text())
                if report.get('comparison_status') == 'pending':
                    report['comparison_status'] = 'failed' if directory == self.result_dir and not self.closing else 'cancelled'
                    (directory/'inference.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
            except (OSError, ValueError):
                pass
        if directory == self.result_dir and job == self.active_job and not self.closing:
            image = QtGui.QImage(str(directory/'reconstruction.jpg'))
            if code == 0 and exit_status == QtCore.QProcess.NormalExit and not image.isNull():
                self.comparison.set_image(image)
                self.tabs.setTabText(1, '图像对照')
                self.report = json.loads((directory/'inference.json').read_text())
                if not self.inference_busy:
                    self.status.setText('重建与图像对照均已完成 · 可继续拍摄')
                    if self.report.get('hand_warning'):
                        self.status.setText(self.status.text()+' · '+self.report['hand_warning'])
            else:
                self.tabs.setTabText(1, '图像对照 · 失败')
                self.status.setText('图像对照生成失败，3D 网格仍可使用；详情见 comparison.log')
        process.deleteLater()

    def process_error(self, error):
        if error == QtCore.QProcess.FailedToStart:
            self.model_state = 'off'
            self.model_label.setText('模型未加载 · 启动失败')
            if self.inference_busy:
                self.fail_model_job(self.process.errorString())
            self.update_camera_controls()

    def process_finished(self, code, exit_status):
        self.live.abort('模型进程退出，连续推理已停止')
        self.kill_timer.stop()
        self.read_log()
        if code!=0 and self.model_state=='loading' and not self.model_failure:
            self.model_failure=f'加载进程异常退出（代码 {code}），请查看运行日志'
            self.log_box.show()
        self.model_state = 'off'
        self.model_label.setText(('模型加载失败：'+self.model_failure) if self.model_failure else '模型已释放 · 可重新预加载')
        if self.inference_busy:
            self.fail_model_job('已取消，可重新加载后重建' if self.cancelled else '模型进程退出，请查看运行日志')
        self.update_camera_controls()
        if self.pending_launch and not self.closing:
            self.pending_launch=False
            self.preload_model()

    def cancel_run(self):
        if self.live.running:
            self.live.stop()
            return
        if self.camera_state == 'countdown':
            self.camera_job = None
            self.camera.send('cancel')
            self.countdown_clock.stop()
            self.canvas.overlay = ''
            self.canvas.update()
            self.camera_state = 'preview'
            self.status.setText('倒计时已取消，可重新拍照')
            self.update_camera_controls()
            return
        if self.inference_busy:
            self.cancelled = True
            self.pending_model_request = None
            self.cancel_btn.setEnabled(False)
            self.model_state = 'stopping'
            self.process.terminate()
            self.kill_timer.start(2500)

    def kill_process(self):
        if self.process.state() != QtCore.QProcess.NotRunning:
            self.process.kill()

    def set_wireframe(self, enabled):
        if self.viewer.mesh:
            self.viewer.mesh.opts['drawEdges'] = enabled
            self.viewer.mesh.meshDataChanged()

    def rotate_step(self):
        if self.viewer.mesh:
            self.viewer.orbit(0.6, 0)

    def open_result_folder(self):
        if self.result_dir:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(self.result_dir)))

    def export_mesh(self):
        if not self.result_dir:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, '另存人体网格', str(Path.home()/'人体重建.ply'), 'PLY 网格 (*.ply)')
        if path:
            import shutil
            try:
                target = Path(path)
                if target.suffix.lower() != '.ply':
                    target = target.with_suffix('.ply')
                source = self.result_dir/'person_00.ply'
                if source.resolve() != target.resolve():
                    shutil.copyfile(source, target)
                self.status.setText(f'网格已保存：{target}')
            except OSError as exc:
                QtWidgets.QMessageBox.warning(self, '保存失败', str(exc))

    def closeEvent(self, event):
        self.closing = True
        self.rgbd.shutdown()
        self.lab.shutdown()
        self.live.abort('窗口已关闭')
        self.motion.shutdown()
        self.pending_backend = None
        self.pending_launch=False
        self.countdown_clock.stop()
        self.camera.shutdown()
        self.rotation.stop()
        for process in list(self.render_processes.values()):
            process.kill()
            process.waitForFinished(2000)
        if self.process.state() != QtCore.QProcess.NotRunning:
            self.cancelled = True
            self.process.terminate()
            if not self.process.waitForFinished(2500):
                self.process.kill()
                self.process.waitForFinished(3000)
        self.overlay.release()
        event.accept()


STYLE = '''
QWidget { background:#0b1120; color:#e2e8f0; font-family:"Noto Sans CJK SC","DejaVu Sans"; font-size:13px; }
QFrame#card { background:#151e30; border:1px solid #27354e; border-radius:12px; }
QFrame#card QLabel, QFrame#card QCheckBox { background:transparent; }
QLabel#title { font-size:25px; font-weight:700; color:#f1f5f9; }
QLabel#muted { color:#94a3b8; font-size:12px; }
QPushButton { background:#1d2a41; border:1px solid #34435e; border-radius:7px; padding:9px 14px; }
QPushButton:hover { background:#2a3c58; border-color:#5b7ea8; }
QPushButton:pressed { background:#344867; }
QPushButton:disabled { color:#59677f; background:#172033; border-color:#27334a; }
QPushButton#primary { background:#2878eb; color:white; border-color:#4a91f5; font-weight:600; padding:11px 30px; }
QPushButton#primary:hover { background:#398af7; }
QPushButton#primary:disabled { background:#1c365b; color:#7188a7; border-color:#284566; }
QTabWidget::pane { border:0; }
QTabBar::tab { background:#172239; padding:9px 20px; border-bottom:2px solid transparent; }
QTabBar::tab:selected { color:#76b7ff; border-bottom-color:#4697ff; }
QCheckBox { spacing:8px; padding:4px 0; }
QCheckBox::indicator { width:15px; height:15px; border:1px solid #5b6d89; border-radius:3px; background:#111827; }
QCheckBox::indicator:checked { background:#4297f8; border-color:#73b4ff; }
QPlainTextEdit { background:#090f1a; color:#a3c1d9; border:1px solid #2b3d57; border-radius:6px; font-family:monospace; font-size:11px; }
QProgressBar { border:0; background:#182b47; border-radius:4px; height:7px; max-height:7px; }
QProgressBar::chunk { background:#4297f8; border-radius:4px; }
QSplitter::handle { background:#0b1120; width:14px; }
QToolTip { background:#1d2a41; color:white; border:1px solid #34435e; }
'''


def create_application():
    # OpenCV's wheel injects its incompatible bundled Qt plugin directory at import.
    import os
    os.environ.pop('QT_QPA_PLATFORM_PLUGIN_PATH', None)
    os.environ.pop('QT_QPA_FONTDIR', None)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)
    # PyQtGraph caches shaders globally; reopened windows must share GL resources.
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)
    app = QtWidgets.QApplication(['sam3d-body-studio'])
    app.setApplicationName('SAM 3D Body Studio')
    app.setDesktopFileName('sam3d-body-studio')
    app.setWindowIcon(QtGui.QIcon(str(ROOT/'gui/icon.svg')))
    app.setFont(QtGui.QFont('Noto Sans CJK SC', 10))
    return app


if __name__ == '__main__':
    app = create_application()
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
