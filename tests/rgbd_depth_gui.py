# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Raw/refined display, stale-input rejection and asynchronous request wiring."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import numpy as np
import cv2
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'gui'),str(ROOT)]
from PyQt5 import QtCore
from app import MainWindow,create_application

with tempfile.TemporaryDirectory() as folder:
    QtCore.QSettings.setPath(QtCore.QSettings.NativeFormat,QtCore.QSettings.UserScope,folder)
    app=create_application();window=MainWindow();v=window.rgbd;sample=Path(folder)
    cv2.imwrite(str(sample/'rgb.png'),np.zeros((20,20,3),np.uint8));cv2.imwrite(str(sample/'mask.png'),np.ones((20,20),np.uint8)*255)
    pred=sample/'person_00.npz';vertices=np.array([[0.,0.,2.],[1,0,2],[0,1,2]],np.float32);faces=np.array([[0,1,2]])
    np.savez(pred,pred_vertices=vertices,pred_cam_t=np.zeros(3),faces=faces,pred_keypoints_3d=np.zeros((70,3)))
    (sample/'rgbd.npz').write_bytes(b'synthetic RGB-D');(sample/'camera.json').write_text('{}')
    ref=dict(accepted=True,reason='accepted',delta_translation_m=[.01,.02,-.03],total_seconds=.05,
        prediction_sha256=hashlib.sha256(pred.read_bytes()).hexdigest(),mask_sha256=hashlib.sha256((sample/'mask.png').read_bytes()).hexdigest(),
        rgb_sha256=hashlib.sha256((sample/'rgb.png').read_bytes()).hexdigest(),metrics=dict(mae_mm=10))
    ref['rgbd_sha256']=hashlib.sha256((sample/'rgbd.npz').read_bytes()).hexdigest();ref['camera_sha256']=hashlib.sha256((sample/'camera.json').read_bytes()).hexdigest()
    result=dict(label='synthetic',config={},prediction=str(pred),inference_ms=10,metrics=dict(mask_sha256=ref['mask_sha256'],depth_refinement=ref,mae_mm=20))
    v.sample=sample;v.results=[result];v.canvas.mask=np.ones((20,20),np.uint8)*255;v.display_evaluation=lambda _:None
    v.update_table();v.table.selectRow(0);v.select_result();assert np.allclose(v.scene.mesh_vertices,vertices)
    v.depth_view.setCurrentIndex(1);assert np.allclose(v.scene.mesh_vertices,vertices+[.01,.02,-.03]);assert '已接受' in v.depth_info.text()
    v.depth_view.setCurrentIndex(2);assert v.scene.comparison is not None;assert np.allclose(v.scene.mesh_vertices,vertices)
    v.mesh_visibility(False);assert not v.scene.comparison.visible()
    (sample/'mask.png').write_bytes(b'changed mask');v.select_result();assert v.scene.comparison is None;assert '输入已改变' in v.depth_info.text()
    calls=[];v.launch=lambda *args:calls.append(args);v.evaluate(refine=True);assert '--refine' in calls[-1][2]
    window.close();app.processEvents();print('RGBD_DEPTH_GUI_OK: raw/refined/overlay, stale mask, visibility, async request')
