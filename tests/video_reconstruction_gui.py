# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Video workspace annotations and cross-tab exclusion without GPU loading."""
from pathlib import Path
import sys
import tempfile
import json
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'gui'),str(ROOT)]
from PyQt5 import QtCore
from app import MainWindow,create_application
from video_project import save_json,add_track

with tempfile.TemporaryDirectory() as tmp:
    QtCore.QSettings.setPath(QtCore.QSettings.NativeFormat,QtCore.QSettings.UserScope,tmp)
    app=create_application();window=MainWindow();video=window.video
    folder=Path(tmp)/'video';(folder/'frames').mkdir(parents=True)
    for i in range(3):cv2.imwrite(str(folder/'frames'/f'{i:06d}.png'),np.zeros((80,100,3),np.uint8))
    project=dict(schema='studio_video_v1',source_sha256='test',width=100,height=80,frame_count=3,
                 pts=[0.,.04,.08],duration=.12,has_audio=False,tracks=[])
    add_track(project,'test');save_json(folder/'project.json',project)
    video.load(folder);video.canvas.prompts=[[25,30,1],[70,30,0]];video.save_segment()
    assert video.project['tracks'][0]['segments'][0]['start']==0
    video.slider.setValue(1);video.canvas.prompts=[[30,30,1]];video.save_segment()
    assert video.project['tracks'][0]['segments'][0]['end']==0
    video.slider.setValue(0);assert video.canvas.prompts==[[25,30,1],[70,30,0]]
    before=window.active_config
    video.mode='segment';window.update_camera_controls()
    assert window.sam3_blocks_models() and not window.preload_btn.isEnabled()
    assert not window.camera_btn.isEnabled() and not window.rgbd.isEnabled()
    assert window.image_seg.common_block() and window.data_manager.busy_reason()
    video.mode=None;window.update_camera_controls();assert window.rgbd.isEnabled()
    window.model_state='ready';assert '释放' in video.other_busy();window.model_state='off'
    assert window.active_config==before
    run=folder/'runs/test';render=run/'render-mesh';render.mkdir(parents=True)
    save_json(run/'run.json',dict(status='complete',prediction_revision='new'))
    save_json(render/'complete.json',dict(prediction_revision='old'))
    (run/'overlay-mesh.mp4').write_bytes(b'test');video.run=run
    assert video.video_file() is None
    save_json(render/'complete.json',dict(prediction_revision='new'))
    assert video.video_file() is not None
    window.close();app.processEvents()
    print('VIDEO_GUI_OK: corrections, prompts, cross-tab guards, independent config, stale exports blocked')
