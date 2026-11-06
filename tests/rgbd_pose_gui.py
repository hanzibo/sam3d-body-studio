# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Post-fit history rows use their own mesh and never double-apply translation."""
import hashlib
from pathlib import Path
import sys
import tempfile
import numpy as np
import cv2
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'gui'),str(ROOT)]
from PyQt5 import QtCore
from app import MainWindow, create_application
from record_history import information, snapshot

with tempfile.TemporaryDirectory() as folder:
    QtCore.QSettings.setPath(QtCore.QSettings.NativeFormat,QtCore.QSettings.UserScope,folder)
    app=create_application();window=MainWindow();v=window.rgbd;sample=Path(folder)
    cv2.imwrite(str(sample/'rgb.png'),np.zeros((20,20,3),np.uint8));cv2.imwrite(str(sample/'mask.png'),np.ones((20,20),np.uint8)*255)
    for name in ('rgbd.npz','camera.json'):(sample/name).write_bytes(b'synthetic')
    raw=sample/'predictions/raw/person_00.npz';raw.parent.mkdir(parents=True)
    fitted=sample/'predictions/pose/person_00.npz';fitted.parent.mkdir(parents=True)
    vertices=np.array([[0.,0.,2.],[1,0,2],[0,1,2]],np.float32);faces=np.array([[0,1,2]])
    np.savez(raw,pred_vertices=vertices,pred_cam_t=np.zeros(3),faces=faces,pred_keypoints_3d=np.zeros((70,3)))
    np.savez(fitted,pred_vertices=vertices+[.02,0,0],pred_cam_t=np.array([0,0,-.03]),faces=faces,pred_keypoints_3d=np.zeros((70,3)))
    fit=dict(stage='pose',stage_label='身体姿态',accepted=True,reason='accepted',scale=1.,total_seconds=1.5,
             source_prediction=str(raw),source_prediction_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
             fit_inputs={name:hashlib.sha256((sample/name).read_bytes()).hexdigest() for name in ('rgb.png','mask.png','rgbd.npz','camera.json')})
    result=dict(label='postfit',config={},prediction=str(fitted),inference_ms=66,depth_fit=fit,metrics={})
    v.sample=sample;v.results=[result];v.canvas.mask=np.ones((20,20),np.uint8)*255;v.display_evaluation=lambda _:None
    v.update_table();v.table.selectRow(0);v.select_result()
    np.testing.assert_allclose(v.scene.mesh_vertices,vertices+[.02,0,-.03])
    assert not v.refine_btn.isEnabled();assert v.depth_view.isHidden();assert not v.stage_compare.isHidden()
    assert '不含后处理' in v.depth_info.text()
    v.stage_compare.setChecked(True)
    np.testing.assert_allclose(v.scene.mesh_vertices,vertices)
    assert v.scene.comparison is not None
    assert information(snapshot(sample,result,''))['本阶段 CPU 后处理 ms']==1500
    (sample/'camera.json').write_text('changed');v.select_result()
    assert '实验输入已改变' in v.depth_info.text()
    raw.write_bytes(b'changed');v.select_result();assert '原始预测已改变' in v.status.text()
    window.close();app.processEvents()
    print('RGBD_POSE_GUI_OK: fitted mesh, raw overlay, stale input/source guards, timing details')
