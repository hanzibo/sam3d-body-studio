# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Offscreen data browser and real system-trash test on disposable fixtures."""
from pathlib import Path
import shutil
import sys
import tempfile
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'gui'), str(ROOT)]
from PyQt5 import QtCore, QtGui, QtWidgets
from app import MainWindow, create_application
from data_manager import DataManager


def wait(app, predicate):
    deadline = time.monotonic() + 15
    while not predicate():
        app.processEvents()
        if time.monotonic() > deadline: raise AssertionError('background operation timed out')
        time.sleep(.01)
    app.processEvents()


with tempfile.TemporaryDirectory() as folder:
    root = Path(folder)
    QtCore.QSettings.setPath(QtCore.QSettings.NativeFormat, QtCore.QSettings.UserScope, str(root/'settings'))
    app = create_application(); host = MainWindow()
    sample = root/'outputs/rgbd/disposable-pose'; sample.mkdir(parents=True)
    (sample/'camera.json').write_text('{"schema":"demo","note":"fixture only"}')
    image = QtGui.QImage(320, 240, QtGui.QImage.Format_RGB32); image.fill(QtGui.QColor('#268c92')); image.save(str(sample/'rgb.png'))
    (sample/'rgbd.npz').write_bytes(b'not deserialized by preview')
    logs = root/'outputs/sessions'; logs.mkdir(parents=True); (logs/'old.log').write_text('fixture session')
    run = root/'outputs/gui/disposable-run'; run.mkdir(parents=True); (run/'request.json').write_text('{"resolution":384}')
    panel = DataManager(host, root); panel.resize(1250, 800); panel.show()
    try:
        wait(app, lambda: panel.scanned and panel.task is None)
        assert host.workspace_tabs.count() == 5 and len(panel.entries) == 3
        row = next(i for i,e in enumerate(panel.entries) if e.path == sample)
        panel.table.selectRow(row); app.processEvents()
        assert panel.files.topLevelItemCount() == 3 and panel.delete_btn.isEnabled()
        for i in range(panel.files.topLevelItemCount()):
            item = panel.files.topLevelItem(i)
            if item.text(0) == 'camera.json': panel.files.setCurrentItem(item)
        assert 'fixture only' in panel.text.toPlainText()
        panel.preview_path(sample/'rgb.png'); assert panel.image.pixmap() is not None
        panel.preview_path(sample/'rgbd.npz'); assert '不反序列化' in panel.text.toPlainText()
        host.source_path = sample/'rgb.png'; panel.update_controls(); assert not panel.delete_btn.isEnabled()
        host.source_path = None; host.inference_busy = True; panel.update_controls(); assert not panel.delete_btn.isEnabled()
        # Even direct method invocation must enforce protection, not just the button.
        panel.delete_selected(); assert sample.exists()
        host.inference_busy = False; host.camera_state = 'countdown'; assert panel.busy_reason()
        host.camera_state = 'off'; host.model_state = 'loading'; assert panel.busy_reason()
        host.model_state = 'off'; host.rgbd.segmentation_pending = {}; assert panel.busy_reason()
        host.rgbd.segmentation_pending = None
        host.rgbd.mode = 'capture'; assert panel.busy_reason(); host.rgbd.mode = None
        host.lab.client.inflight = {}; assert panel.busy_reason(); host.lab.client.inflight = None
        host.motion.recorder = object(); assert panel.busy_reason(); host.motion.recorder = None
        host.render_processes['fixture'] = type('Process', (), {'state': lambda self: QtCore.QProcess.Running})()
        assert panel.busy_reason(); host.render_processes.clear()
        # Idle resident model may stay loaded, but its own log/config stay protected.
        host.model_state = 'ready'; host.model_log_path = logs/'old.log'
        logrow = next(i for i,e in enumerate(panel.entries) if e.path == logs/'old.log')
        panel.table.selectRow(logrow); panel.update_controls(); assert not panel.delete_btn.isEnabled()
        host.model_state = 'off'; host.model_log_path = None; panel.table.selectRow(row)
        host.rgbd.batch.session = type('Session', (), {'directory': sample})()
        panel.update_controls(); assert not panel.delete_btn.isEnabled(); host.rgbd.batch.session = None
        panel.order.setCurrentIndex(1)
        assert [e.size for e in panel.entries] == sorted([e.size for e in panel.entries], reverse=True)
        panel.table.horizontalHeader().sectionClicked.emit(2)
        assert panel.order.currentIndex() == 2
        assert [e.size for e in panel.entries] == sorted(e.size for e in panel.entries)
        assert panel.table.horizontalHeader().sortIndicatorOrder() == QtCore.Qt.AscendingOrder
        panel.table.horizontalHeader().sectionClicked.emit(2)
        assert panel.order.currentIndex() == 1
        assert [e.size for e in panel.entries] == sorted([e.size for e in panel.entries], reverse=True)
        row = next(i for i,e in enumerate(panel.entries) if e.path == sample); panel.table.selectRow(row)
        panel.search.setText('no-match'); assert not panel.selected_entries()
        panel.search.clear(); panel.table.selectRow(row); panel.update_controls(); assert panel.delete_btn.isEnabled()
        # Cancellation must leave fixture untouched.
        with patch.object(QtWidgets.QMessageBox, 'question', return_value=QtWidgets.QMessageBox.No): panel.delete_selected()
        assert sample.exists()
        # Actual recycle operation; save its returned location and restore it after proof.
        original_trash = QtCore.QFile.moveToTrash; trashed = []
        def move(path):
            result = original_trash(path)
            if result[0]: trashed.append((Path(path), Path(result[1])))
            return result
        try:
            with patch.object(QtWidgets.QMessageBox, 'question', return_value=QtWidgets.QMessageBox.Yes), patch.object(QtCore.QFile, 'moveToTrash', side_effect=move):
                panel.delete_selected()
                wait(app, lambda: panel.task is None and not panel.deleting)
            assert not sample.exists() and len(trashed) == 1
            assert '已移入回收站 1 项' in panel.status.text()
            assert len(panel.entries) == 2 and run.exists()
        finally:
            for original, trash in trashed:
                if trash.exists():
                    shutil.move(str(trash), str(original))
                    info = trash.parent.parent/'info'/(trash.name+'.trashinfo')
                    if info.exists(): info.unlink()
        # Failure is explicit and never falls back to permanent removal.
        panel.refresh(); wait(app, lambda: panel.task is None)
        row = next(i for i,e in enumerate(panel.entries) if e.path == sample); panel.table.selectRow(row)
        with patch.object(QtWidgets.QMessageBox, 'question', return_value=QtWidgets.QMessageBox.Yes), patch.object(QtWidgets.QMessageBox, 'warning'), patch.object(QtCore.QFile, 'moveToTrash', return_value=(False, '')):
            panel.delete_selected(); wait(app, lambda: panel.task is None and not panel.deleting)
        assert sample.exists() and '部分失败' in panel.status.text()
        def cancelled_export(parent, caption, filename, filefilter):
            assert Path(filename).parent.is_dir()
            assert Path(filename).parent.parent == root/'outputs/mhr-lab'
            return '', ''
        with patch('mhr_lab.ROOT', root), patch.object(QtWidgets.QFileDialog, 'getSaveFileName', side_effect=cancelled_export):
            host.lab.export_dialog()
        assert not list((root/'outputs/mhr-lab').iterdir())
        assert host.model_state == 'off' and host.process.state() == QtCore.QProcess.NotRunning
        print('DATA_MANAGER_OK: grouping, previews, filters, active-job/path guards, cancellation, real trash/restore, failure preservation; no models/camera loaded')
    finally:
        panel.shutdown(); panel.close(); host.close(); app.processEvents()
