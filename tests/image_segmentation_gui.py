# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Offscreen first-tab controls, immutable inputs and load interlocks; no models."""
import json
from pathlib import Path
import sys
import tempfile
import time
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'gui'),str(ROOT)]
from app import MainWindow, create_application
from mask_prompt import read_prompt

with tempfile.TemporaryDirectory() as folder:
    QtCore.QSettings.setPath(QtCore.QSettings.NativeFormat,QtCore.QSettings.UserScope,folder)
    app=create_application();w=MainWindow();seg=w.image_seg
    image=QtGui.QImage(160,120,QtGui.QImage.Format_RGB888);image.fill(QtGui.QColor('gray'))
    source=Path(folder)/'rgb.png';image.save(str(source));w.load_image(str(source))
    w.intrinsics_check.setChecked(False)
    assert seg.segment_btn.isEnabled()
    seg.points.setChecked(True);assert not w.auto_reconstruct.isChecked()
    seg.add_point(80,60,1);assert w.canvas.prompts==[[80,60,1]]
    seg.raw=np.zeros((120,160),np.uint8);seg.raw[20:100,40:120]=1
    seg.input_file=source;seg.result=dict(test=True);seg.morph()
    bbox=[40,20,120,100];w.canvas.bbox=bbox.copy()
    seg.slider.setValue(4);assert w.canvas.bbox==bbox
    positive=seg.effective.sum();seg.slider.setValue(-4);assert seg.effective.sum()<positive
    normalized=Path(folder)/'input.png';w.canvas.image.save(str(normalized))
    info=seg.snapshot(Path(folder),normalized)
    actual,report=read_prompt(dict(type='run',image=str(normalized),mask_prompt=info),(120,160))
    assert report['enabled'] and actual.sum()>0
    w.model_state='ready';w.update_camera_controls();assert not seg.segment_btn.isEnabled()
    assert seg.slider.isEnabled()  # CPU morphology still works with a loaded body model.
    w.model_state='off';seg.mode='segment';w.update_camera_controls()
    assert w.sam3_blocks_models() and not w.preload_btn.isEnabled() and not w.open_btn.isEnabled()
    seg.mode=None;w.update_camera_controls()
    seg.use_mask.setChecked(False);assert seg.snapshot(Path(folder),normalized) is None
    seg.add_point(70,60,0);assert seg.raw is None
    # A result event alone must not unlock models or apply a mask before exit.
    seg.directory=Path(folder)/'seg';(seg.directory/'sam3').mkdir(parents=True)
    candidate=np.zeros((120,160),np.uint8);candidate[20:100,40:120]=255
    cv2.imwrite(str(seg.directory/'sam3/mask_00.png'),candidate)
    fake=Path(folder)/'fake.py'
    event=dict(type='segmented',report=dict(selected=0,candidates=[dict(index=0,score=.9)]))
    fake.write_text("import time; print("+repr('@@RGBD '+json.dumps(event))+",flush=True); time.sleep(.8)")
    seg.launch('segment',sys.executable,fake,[])
    def wait_for(predicate):
        deadline=time.monotonic()+5
        while not predicate() and time.monotonic()<deadline:
            app.processEvents();time.sleep(.01)
        assert predicate()
    wait_for(lambda:seg.event is not None)
    assert seg.active and seg.raw is None and w.sam3_blocks_models()
    wait_for(lambda:not seg.active)
    assert seg.raw is not None and not w.sam3_blocks_models()
    cv2.imwrite(str(seg.directory/'sam3/mask_01.png'),np.zeros_like(candidate))
    seg.candidates.addItem('empty',1);seg.candidates.setCurrentIndex(1)
    assert seg.raw is None and seg.candidates.isEnabled()
    seg.candidates.setCurrentIndex(0);assert seg.raw is not None
    fake.write_text('import time; time.sleep(30)')
    seg.launch('segment',sys.executable,fake,[]);seg.cancel()
    assert w.sam3_blocks_models()  # terminate is asynchronous
    wait_for(lambda:not seg.active)
    assert '已取消' in seg.status.text()
    # Failed executable must recover controls as well.
    seg.launch('segment',Path(folder)/'missing-python',fake,[])
    wait_for(lambda:not seg.active)
    assert not w.sam3_blocks_models()
    w.close();app.processEvents()
    print('IMAGE_SEGMENTATION_GUI_OK: snapshots, morphology, invalidation and model interlocks')
