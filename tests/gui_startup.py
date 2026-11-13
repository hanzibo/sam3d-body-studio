# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Construct all Studio tabs without GPU/model loading; isolate user settings."""
import os
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'gui'),str(ROOT)]
from PyQt5 import QtCore
from app import MainWindow,create_application

with tempfile.TemporaryDirectory() as folder:
    QtCore.QSettings.setPath(QtCore.QSettings.NativeFormat,QtCore.QSettings.UserScope,folder)
    app=create_application();w=MainWindow()
    assert w.workspace_tabs.count()==7
    assert w.workspace_tabs.tabText(6)=='手部实验'
    assert w.hand.state=='off'
    assert w.hand.tabs.count()==6 and w.hand.tabs.tabText(5)=='⑥ 相机实时'
    assert not w.hand.live.running and not w.hand.live.start.isEnabled()
    assert w.workspace_tabs.tabText(3)=='数据管理'
    assert w.workspace_tabs.tabText(4)=='视频重建'
    assert w.model_state=='off'
    assert w.process.state()==QtCore.QProcess.NotRunning
    w.close();app.processEvents()
    print('GUI_STARTUP_OK: seven tabs; no model or camera started')
