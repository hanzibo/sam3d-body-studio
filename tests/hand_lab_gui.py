# SPDX-License-Identifier: GPL-3.0-only
"""Real isolated Qt/GPU hand pipeline on user-supplied Gemini RGB sequences."""
from pathlib import Path
import json
import sys
import tempfile
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'gui'),str(ROOT)]
from PyQt5 import QtCore
from app import create_application,MainWindow
from studio.core.gpu_lease import acquire_gpu_lease
OUT=Path('/run/media/hzb/Data/experiments/hand-lab-20261112')
DATA=Path('/run/media/hzb/Data/experiments/wilor/datasets')

def wait(app,predicate,seconds=90):
    end=time.monotonic()+seconds
    while not predicate():
        app.processEvents();time.sleep(.01)
        if time.monotonic()>end:raise AssertionError('GUI timed out')
    app.processEvents()

def pause(app,seconds=.5):
    end=time.monotonic()+seconds
    while time.monotonic()<end:app.processEvents();time.sleep(.01)

with tempfile.TemporaryDirectory() as settings:
    QtCore.QSettings.setPath(QtCore.QSettings.NativeFormat,QtCore.QSettings.UserScope,settings)
    app=create_application();host=MainWindow();host.resize(1550,1050)
    hand=host.hand;host.workspace_tabs.setCurrentWidget(hand);host.show();pause(app)
    assert host.workspace_tabs.tabText(host.workspace_tabs.count()-1)=='手部实验'
    single=DATA/'20261112_173053_single_hand_d4f1fc/frames/000150.jpg'
    hand.load_image(single);assert hand.intrinsics.isChecked();assert not hand.run.isEnabled()
    hand.canvas.bbox=[422,365,540,538];hand.roi_changed()
    hand.start();wait(app,lambda:hand.state in ('ready','off'))
    assert hand.state=='ready',hand.info.text()
    assert host.model_state=='off'
    host.preload_model();assert host.model_state=='off' and '手部' in host.status.text()
    try:acquire_gpu_lease('sam3')
    except RuntimeError:pass
    else:raise AssertionError('exclusive hand lease failed')
    hand.submit('warmup');wait(app,lambda:hand.state=='ready');assert '预热完成' in hand.info.text(),hand.info.text()
    cases=[('single-right',single,[422,365,540,538],0),
           ('single-fist',single.with_name('000330.jpg'),[399,421,492,552],0),
           ('dual-right',DATA/'20261112_173116_two_hands_acdf29/frames/000150.jpg',[337,364,459,539],0),
           ('dual-left',DATA/'20261112_173116_two_hands_acdf29/frames/000150.jpg',[607,330,724,494],1)]
    results=[]
    for label,image,box,side in cases:
        hand.load_image(image);hand.canvas.bbox=box;hand.side.setCurrentIndex(side);hand.roi_changed()
        hand.submit('run');wait(app,lambda:hand.state=='ready');assert hand.result_dir is not None,hand.info.text()
        report=json.loads((hand.result_dir/'report.json').read_text());assert report['execution_counts']==dict(body=0,hand=1,backbone=1)
        assert report['camera_source']=='photo_metadata';assert report['projection_roundtrip_max_px']<.05
        with np.load(hand.result_dir/'hand.npz',allow_pickle=False) as data:
            np.testing.assert_allclose(data['camera_matrix'][0,0],607.8025512695312)
            assert np.linalg.det(data['global_rotations']).min()>.99
        for index,name in [(1,'3d'),(2,'overlay'),(3,'crop')]:
            hand.tabs.setCurrentIndex(index);pause(app)
            host.grab().save(str(OUT/f'{label}-{name}.png'))
            if index==2:hand.overlay.grabFramebuffer().save(str(OUT/f'{label}-overlay-frame.png'))
        results.append(dict(label=label,directory=str(hand.result_dir),report=report))
    hand.shutdown();assert hand.process.state()==QtCore.QProcess.NotRunning
    lease=acquire_gpu_lease('sam3');lease.close()
    host.close();app.processEvents()
    (OUT/'gui-validation.json').write_text(json.dumps(results,indent=2,ensure_ascii=False))
    print('HAND_GUI_REAL_GPU_OK: single / dual right / dual left; calibrated overlay; prewarm; body block; SAM3 exclusion; release')
