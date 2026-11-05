# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Shared pose layers, angles and recording controls for still/live results."""
from pathlib import Path
from PyQt5 import QtCore, QtGui, QtWidgets
from pose_data import extract_pose, joint_angles, angle_text
from motion_recording import MotionRecorder


class MotionPanel(QtWidgets.QWidget):
    def __init__(self, window):
        super().__init__(window)
        self.w=window;self.pose=None;self.recorder=None;self.last_path=None;self.context={}
        layout=QtWidgets.QVBoxLayout(self);layout.setContentsMargins(0,0,0,0)
        row=QtWidgets.QHBoxLayout()
        self.mesh=QtWidgets.QCheckBox('网格');self.mesh.setChecked(True)
        self.skeleton=QtWidgets.QCheckBox('骨架');self.skeleton.setChecked(True)
        self.angles=QtWidgets.QCheckBox('肘膝角度');self.angles.setChecked(True)
        for item in (self.mesh,self.skeleton,self.angles):
            row.addWidget(item);item.toggled.connect(self.layers_changed)
        self.export=QtWidgets.QPushButton('导出当前骨架…');self.export.clicked.connect(self.export_pose)
        row.addWidget(self.export);layout.addLayout(row)
        mhr_row=QtWidgets.QHBoxLayout()
        mhr_row.addWidget(QtWidgets.QLabel('MHR 骨架 / 角度'))
        self.mhr_variant=QtWidgets.QComboBox()
        self.mhr_variant.addItem('原始（当前帧）','raw');self.mhr_variant.addItem('One Euro（位置滤波）','one_euro')
        self.mhr_variant.currentIndexChanged.connect(self.render_mhr)
        mhr_row.addWidget(self.mhr_variant)
        self.mhr_label=QtWidgets.QLabel('原始输出');self.mhr_label.setWordWrap(True)
        mhr_row.addWidget(self.mhr_label,1);layout.addLayout(mhr_row)
        self.angle_label=QtWidgets.QLabel('左肘 —  ·  右肘 —  ·  左膝 —  ·  右膝 —')
        self.angle_label.setWordWrap(True)
        self.angle_label.setToolTip('三维关键点的三点夹角，伸直约 180°；不是解剖学屈曲角或局部关节旋转。')
        layout.addWidget(self.angle_label)
        smpl_row=QtWidgets.QHBoxLayout()
        smpl_row.addWidget(QtWidgets.QLabel('SMPL 预览'))
        self.smpl_variant=QtWidgets.QComboBox()
        self.smpl_variant.addItem('原始（当前帧）','raw');self.smpl_variant.addItem('MLP 平滑（中心帧）','smooth');self.smpl_variant.addItem('One Euro（当前帧滤波）','one_euro')
        self.smpl_variant.currentIndexChanged.connect(self.render_smpl)
        smpl_row.addWidget(self.smpl_variant)
        self.smpl_label=QtWidgets.QLabel('SMPL 未启用');self.smpl_label.setWordWrap(True)
        smpl_row.addWidget(self.smpl_label,1);layout.addLayout(smpl_row)
        self.smpl_angles=QtWidgets.QLabel();self.smpl_angles.setWordWrap(True);layout.addWidget(self.smpl_angles)
        row=QtWidgets.QHBoxLayout()
        self.record=QtWidgets.QPushButton('开始动作录制');self.record.clicked.connect(self.toggle_record)
        self.folder=QtWidgets.QPushButton('打开动作目录');self.folder.clicked.connect(self.open_folder)
        row.addWidget(self.record);row.addWidget(self.folder)
        self.label=QtWidgets.QLabel('仅录制推理骨架与角度，不保存 RGB 视频');self.label.setWordWrap(True)
        row.addWidget(self.label,1);layout.addLayout(row)
        self.timer=QtCore.QTimer(self);self.timer.setInterval(200);self.timer.timeout.connect(self.refresh);self.timer.start()
        self.refresh()

    def layers_changed(self):
        self.w.viewer.set_layers(self.mesh.isChecked(),self.skeleton.isChecked())
        self.w.overlay.set_layers(self.mesh.isChecked(),self.skeleton.isChecked(),self.angles.isChecked())
        self.angle_label.setVisible(self.angles.isChecked())
        self.smpl_angles.setVisible(self.angles.isChecked())

    def set_pose(self, data, context=None):
        self.pose=extract_pose(data);self.context=context or {}
        self.render_mhr()
        self.render_smpl()
        self.refresh()

    def render_mhr(self):
        available=self.pose is not None and 'mhr_keypoints_3d_one_euro' in self.pose
        self.mhr_variant.model().item(1).setEnabled(available)
        if not available and self.mhr_variant.currentData()=='one_euro':
            self.mhr_variant.blockSignals(True);self.mhr_variant.setCurrentIndex(0);self.mhr_variant.blockSignals(False)
        if self.pose is None:
            self.mhr_label.setText('无骨架数据');return
        euro=self.mhr_variant.currentData()=='one_euro'
        points=self.pose['mhr_keypoints_3d_one_euro' if euro else 'pred_keypoints_3d']
        angles=joint_angles(points)
        live=getattr(self.w,'live',None)
        if live and live.active and live.display_mode=='mhr':
            self.w.viewer.set_skeleton_frame(points,continuous=True)
        else:self.w.viewer.set_pose(points)
        uv=self.pose.get('mhr_keypoints_2d_one_euro') if euro else self.pose['pred_keypoints_2d']
        mask=self.pose.get('mhr_projection_valid') if euro else None
        self.w.overlay.set_pose(uv,angles,mask)
        self.angle_label.setText(('MHR One Euro · ' if euro else 'MHR 原始 · ')+angle_text(angles))
        text='原始输出（原始与滤波数据均保留）' if available else 'One Euro 未启用，显示原始输出'
        if euro:text=f'滤波 {float(self.pose["mhr_one_euro_ms"]):.2f} ms · 来源帧 {int(self.pose["mhr_one_euro_source_frame_index"])} · 网格保持原始'
        self.mhr_label.setText(text)

    def render_smpl(self):
        variant=self.smpl_variant.currentData();smooth=variant=='smooth';euro=variant=='one_euro'
        key='smpl_keypoints_3d_smooth' if smooth else 'smpl_keypoints_3d_one_euro' if euro else 'smpl_keypoints_3d'
        self.smpl_variant.model().item(1).setEnabled(self.pose is not None and 'smpl_keypoints_3d_smooth' in self.pose)
        self.smpl_variant.model().item(2).setEnabled(self.pose is not None and 'smpl_keypoints_3d_one_euro' in self.pose)
        if self.pose is None or key not in self.pose:
            self.w.smpl_view.clear_mesh();self.smpl_angles.clear()
            self.smpl_label.setText('当前方案没有所选滤波结果' if smooth or euro else 'SMPL 未启用')
            return
        live=getattr(self.w,'live',None)
        self.w.smpl_view.set_smpl_frame(self.pose[key],continuous=bool(live and live.active))
        points=self.pose['smpl_joints_smooth_root' if smooth else 'smpl_joints_one_euro_root' if euro else 'smpl_joints_root']
        angles=joint_angles(points,{'left_elbow':(16,18,20),'right_elbow':(17,19,21),'left_knee':(1,4,7),'right_knee':(2,5,8)})
        self.smpl_angles.setText(('SMPL MLP · ' if smooth else 'SMPL One Euro · ' if euro else 'SMPL 原始 · ')+angle_text(angles))
        cost=float(self.pose.get('smpl_postprocess_ms',0))
        text=f'转换＋关节处理 {cost:.1f} ms'
        if smooth:
            lag=float(self.pose['smpl_smooth_lag_ms']);frame=int(self.pose['smpl_smooth_source_frame_index'])
            text+=f' · 延后 {lag:.0f} ms · 来源帧 {frame}'
        if euro:text+=f' · 滤波 {float(self.pose["smpl_one_euro_ms"]):.2f} ms · 因果滤波，无固定等待帧'
        self.smpl_label.setText(text)

    def clear_pose(self):
        self.pose=None
        self.render_mhr()
        self.render_smpl()
        self.angle_label.setText('左肘 —  ·  右肘 —  ·  左膝 —  ·  右膝 —')
        self.w.viewer.set_pose(None);self.w.overlay.set_pose(None,{})
        self.refresh()

    def refresh(self):
        live=getattr(self.w,'live',None)
        rec=self.recorder
        closing=rec is not None and rec.stop_event.is_set()
        self.record.setEnabled(bool(live and live.active) and not closing)
        self.record.setText('正在保存…' if closing and not rec.done else '停止动作录制' if rec else '开始动作录制')
        self.export.setEnabled(self.pose is not None and rec is None and not (live and live.running))
        self.folder.setEnabled(self.last_path is not None)
        if rec:
            self.label.setText(f'录制 {rec.count} 帧 · 失败 {rec.gaps} 帧' if not closing else '正在保存动作数据…')
            if rec.done:
                self.last_path=rec.path
                self.label.setText(('保存失败：'+rec.error) if rec.error else f'已保存 {rec.count} 帧 · {rec.path.name}')
                self.recorder=None
                self.record.setText('开始动作录制')
                self.folder.setEnabled(True)

    def toggle_record(self):
        if self.recorder:
            self.stop_record('user_stop');return
        live=self.w.live
        if not live.active:return
        try:
            self.recorder=MotionRecorder(Path(__file__).resolve().parents[1]/'outputs/motions', dict(
                launch_config=live.config.to_dict(),config_id=live.config.config_id,backend_id=self.w.backend_id,
                model_session=self.w.model_session,smpl_assets=getattr(self.w,'smpl_asset_hashes',None),roi=live.roi,camera_metadata=live.metadata,
                use_gemini_intrinsics=live.use_intrinsics,display_mode=live.display_mode))
            self.last_path=self.recorder.path
        except (OSError,ValueError,TypeError) as exc:
            self.label.setText('无法开始录制：'+str(exc))
        self.refresh()

    def record_event(self, event):
        rec=self.recorder
        if rec and event['host_received_monotonic']>=rec.started:
            rec.append(event)
        self.refresh()

    def stop_record(self, reason):
        if self.recorder:self.recorder.close(reason)
        self.refresh()

    def shutdown(self):
        self.stop_record('window_close')
        if self.recorder:self.recorder.thread.join(timeout=5)

    def open_folder(self):
        if self.last_path:QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(self.last_path)))

    def export_pose(self):
        if self.pose is None:return
        path=QtWidgets.QFileDialog.getExistingDirectory(self,'选择骨架导出父目录',str(Path.home()))
        if not path:return
        try:
            rec=MotionRecorder(Path(path),dict(source='single_result_export',result_context=self.context))
            rec.append(dict(type='pose',frame_index=self.context.get('frame_index',0),host_received_monotonic=rec.started,
                            inference_ms=self.context.get('inference_ms',0),pose=self.pose))
            rec.close('snapshot_export')
            self.recorder=rec;self.last_path=rec.path;self.refresh()
        except (OSError,ValueError,TypeError) as exc:self.label.setText('导出失败：'+str(exc))
