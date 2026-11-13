# SPDX-License-Identifier: GPL-3.0-only
"""Real saved-frame GUI regression, no body/SAM3 model, camera or inference."""
import sys,time,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'gui'))
from PyQt5 import QtCore,QtTest
from app import create_application,MainWindow
from rgbd_comparison_data import load_npz

app=create_application();w=MainWindow();w.resize(1640,1000);w.show();v=w.tracking_comparison
output=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else ROOT/'outputs/depth-fusion/tracking-comparison-v1';checks={}
def wait(fn,timeout=30):
    limit=time.monotonic()+timeout
    while not fn():
        app.processEvents();QtTest.QTest.qWait(15)
        if '失败' in v.status.text():raise RuntimeError(v.status.text())
        if time.monotonic()>limit:raise TimeoutError(v.status.text())
try:
    assert any(w.workspace_tabs.tabText(i)=='RGB-D 跟踪对照' for i in range(w.workspace_tabs.count()))
    w.workspace_tabs.setCurrentWidget(v);v.scan();assert len(v.case_paths)>=4
    for case in ('arms','cross','turn','recover'):
        index=next(i for i,p in enumerate(v.case_paths) if p.name==case and p.parent==output)
        v.cases.setCurrentIndex(index);wait(lambda:v.loaded_index==0 and not v.busy)
        v.slider.setValue(50);wait(lambda:v.loaded_index==50 and not v.busy)
        actual=load_npz(v.directory/'frames/000050.npz')
        assert len(v.scenes)==len(actual['vertices'])
        for i,scene in enumerate(v.scenes):
            if actual['valid'][i]:np.testing.assert_allclose(scene.mesh_vertices,actual['vertices'][i],rtol=0,atol=0)
            else:assert scene.body is None
        v.view(0);v.driver=1;v.scenes[1].setCameraPosition(azimuth=30);v.sync_views();assert all(s.opts['azimuth']==30 for s in v.scenes)
        v.mesh_check.setChecked(False);assert all(s.body is None or not s.body.visible() for s in v.scenes);v.mesh_check.setChecked(True)
        before=v.loaded_index;v.toggle();wait(lambda:v.loaded_index>before+2);v.pause();checks[case]='real frame + exact saved camera mesh + synchronized view + playback OK'
        if case=='arms':w.grab().save(str(output/'gui-comparison.png'));v.scenes[2].grabFramebuffer().save(str(output/'gui-tracking-scene.png'))
    v.slider.setValue(110);wait(lambda:v.loaded_index==110 and not v.busy)
    assert all(s.body is None for s in v.scenes);checks['lost']='no stale body on out-of-frame interval'
    assert w.model_state!='ready';checks['models']='no inference model loaded'
    (output/'gui-checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2));print('REAL_GUI_COMPARISON_PLAYBACK_OK',flush=True)
finally:w.close();app.processEvents()
