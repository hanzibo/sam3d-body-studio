# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Thumbnail invalidation, shared geometry and static point-buffer draw ranges."""
from pathlib import Path
import json
import sys
import tempfile
from unittest.mock import patch,MagicMock
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'gui'),str(ROOT)]
from PyQt5 import QtCore,QtGui
from app import MainWindow,create_application
from rgbd_batch import PoseSet
import rgbd_validation as module

with tempfile.TemporaryDirectory() as tmp:
    QtCore.QSettings.setPath(QtCore.QSettings.NativeFormat,QtCore.QSettings.UserScope,tmp)
    app=create_application();w=MainWindow();v=w.rgbd;p=v.batch
    session=PoseSet.create(Path(tmp)/'set',12)
    for i in range(12):
        folder=session.directory/str(i);folder.mkdir();cv2.imwrite(str(folder/'rgb.png'),np.full((72,128,3),i,np.uint8))
        (folder/'camera.json').write_text('{}');(folder/'rgbd.npz').write_bytes(b'synthetic')
        session.attach(i,folder);p.poses.addItem(f'姿态{i+1}')
    p.session=session
    reader=QtGui.QImageReader
    with patch.object(QtGui,'QImageReader',wraps=reader) as spy:
        p.refresh(force_poses=True);assert spy.call_count==12
        for _ in range(20):p.refresh(force_poses=True)
        assert spy.call_count==12,'unchanged PNGs were decoded again'
        sample=session.sample_path(session.poses[0]);(sample/'sam3_prompts.json').write_text(json.dumps(dict(point_labels=[1,0])))
        p.refresh(force_poses=True);assert spy.call_count==12;assert '1前/1后' in p.poses.itemText(0)
        cv2.imwrite(str(sample/'rgb.png'),np.ones((72,128,3),np.uint8)*255)
        p.refresh(force_poses=True);assert spy.call_count==13
        (sample/'rgb.png').unlink();p.refresh(force_poses=True);assert p.poses.itemIcon(0).isNull()
    # Geometry/normal buffers survive translation changes; raw vertex arrays stay intact.
    scene=v.scene;vertices=np.array([[0.,0.,2.],[1,0,2],[0,1,2]],np.float32);faces=np.array([[0,1,2]]);source=vertices.copy()
    scene.set_mesh(vertices,faces,key=('a',));body=scene.body;data=body.opts['meshdata']
    scene.set_mesh(vertices,faces,key=('a',),translation=[.1,.2,.3]);assert scene.body is body
    scene.set_comparison(vertices,faces,key=('a',),translation=[.1,.2,.3]);assert scene.comparison.opts['meshdata'] is data
    assert np.array_equal(vertices,source);assert np.allclose(scene.mesh_vertices,source+[.1,.2,.3])
    scene.set_mesh(vertices,faces,key=('b',));assert scene.body.opts['meshdata'] is not data
    # Mock only GL/Qt buffer, exercising upload and draw logic on CPU CI workers.
    cloud=scene.cloud;pos=np.arange(300,dtype=np.float32).reshape(100,3);rgba=np.ones((100,4),np.float32)
    buffer=MagicMock();buffer.create.return_value=True;buffer.bind.return_value=True;context=object()
    with patch.object(module.QtGui,'QOpenGLBuffer',return_value=buffer),patch.object(module.QtGui.QOpenGLContext,'currentContext',return_value=context),patch.object(module,'GL') as gl:
        cloud.setData(pos,rgba,2);cloud.paint();cloud.paint()
        assert buffer.allocate.call_count==1;assert cloud.upload_count==1
        cloud.set_interacting(True);cloud.paint();assert gl.glDrawArrays.call_args.args[1:]==(100,25)
        cloud.set_interacting(False);cloud.paint();assert gl.glDrawArrays.call_args.args[1:]==(0,100)
        assert buffer.allocate.call_count==1;assert np.array_equal(cloud.positions,pos)
        cloud.setData(pos+1,rgba,2);cloud.paint();assert buffer.allocate.call_count==2
        cloud.release_gl();assert buffer.destroy.called
    w.close();app.processEvents();print('RGBD_PREVIEW_GUI_OK: thumbnails reuse/invalidate, immutable shared mesh, VBO upload once, LOD restores full range')
