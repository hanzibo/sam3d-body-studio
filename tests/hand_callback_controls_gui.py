# SPDX-License-Identifier: GPL-3.0-only
"""Qt configuration roundtrip and loaded-model locking, no camera or GPU."""
import sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'gui'),str(ROOT)]
from PyQt5 import QtCore
from app import create_application,MainWindow

with tempfile.TemporaryDirectory() as settings:
    QtCore.QSettings.setPath(QtCore.QSettings.NativeFormat,QtCore.QSettings.UserScope,settings)
    app=create_application();window=MainWindow();hand=window.hand
    hand.accelerated_btn.click()
    assert hand.execution_config()['intermediate_compile'] is True
    assert hand.execution_config()['intermediate_predictions']==5
    for count in range(6):
        hand.prediction_count.setCurrentIndex(hand.prediction_count.findData(count))
        assert hand.execution_config()['intermediate_predictions']==count
    assert hand.prediction_count.isEnabled() and hand.intermediate_compile.isEnabled()
    hand.state='ready';hand.update_controls()
    assert not hand.prediction_count.isEnabled() and not hand.intermediate_compile.isEnabled()
    hand.state='off';hand.update_controls();hand.baseline_btn.click()
    assert hand.execution_config()['intermediate_compile'] is False
    assert hand.execution_config()['intermediate_predictions']==5
    hand.int8_btn.click();assert hand.execution_config()['intermediate_compile'] is True
    assert hand.execution_config()['backbone']=='trt_int8'
    for value in ('both','right','left'):
        hand.hand_selection.setCurrentIndex(hand.hand_selection.findData(value))
        assert window.settings.value('hand/detection_selection','both')==value
    hand.confidence.setValue(.75)
    assert window.settings.value('hand/detection_confidence',type=float)==.75
    hand.state='busy';hand.live.active=True;hand.update_controls()
    assert hand.confidence.isEnabled() and hand.hand_selection.isEnabled()
    hand.pose_filter.setChecked(True);hand.position_filter.setChecked(False)
    hand.filter_parameters['min_cutoff'].setValue(.9)
    assert hand.one_euro_options()==dict(pose=True,position=False,min_cutoff=.9,beta=.5,d_cutoff=1.)
    assert window.settings.value('hand/one_euro_pose',type=bool)
    assert hand.pose_filter.isEnabled() and hand.filter_parameters['beta'].isEnabled()
    hand.live.active=False;hand.state='off'
    window.close();app.processEvents()
    print('HAND_CALLBACK_CONTROLS_OK')
