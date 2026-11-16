# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Qt process/socket adapter; the GUI never imports the camera SDK."""
import json
from pathlib import Path
import struct
import tempfile
import time
from PyQt5 import QtCore, QtGui, QtNetwork
from camera_receiver import CameraReceiver

ROOT = Path(__file__).resolve().parents[1]


class CameraClient(QtCore.QObject):
    messageReceived = QtCore.pyqtSignal(dict)
    frame = QtCore.pyqtSignal(QtGui.QImage)
    stopped = QtCore.pyqtSignal()
    log = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.process = QtCore.QProcess(self)
        self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.read_log)
        self.process.finished.connect(self.finished)
        self.process.errorOccurred.connect(self.process_error)
        self.server = QtNetwork.QLocalServer(self)
        self.server.newConnection.connect(self.connected)
        self.socket = None
        self.receiver = self.io_thread = None
        self.temp = None
        self.buffer = b''
        self.stopping = False
        self.last_event = 0
        self.latest_frame = None
        self.watchdog = QtCore.QTimer(self)
        self.watchdog.setInterval(500)
        self.watchdog.timeout.connect(self.check_timeout)
        self.kill_timer = QtCore.QTimer(self)
        self.kill_timer.setSingleShot(True)
        self.kill_timer.timeout.connect(self.process.kill)

    @property
    def active(self):
        return self.process.state() != QtCore.QProcess.NotRunning

    def start(self, backend="orbbec"):
        if self.active:
            return
        self.close_receiver()
        self.stopping = False
        self.latest_frame = None
        self.buffer = b''
        self.temp = tempfile.TemporaryDirectory(prefix='sam3d-camera-')
        path = str(Path(self.temp.name)/'camera.sock')
        if not self.server.listen(path):
            self.messageReceived.emit(dict(type='error', message=self.server.errorString()))
            self.finished()
            return
        self.last_event = time.monotonic()
        self.process.setWorkingDirectory(str(ROOT))
        self.process.start('/bin/bash', [str(ROOT/'launch_camera.sh'), '--socket', path, '--backend', backend])
        self.watchdog.start()

    def connected(self):
        self.socket = self.server.nextPendingConnection()
        self.socket.setParent(None)
        self.io_thread = QtCore.QThread(self)
        self.receiver = CameraReceiver(self.socket)
        self.socket.moveToThread(self.io_thread)
        self.receiver.moveToThread(self.io_thread)
        self.io_thread.started.connect(self.receiver.start)
        self.receiver.available.connect(self.read_messages)
        self.receiver.message.connect(self.receive_message)
        self.receiver.closed.connect(self.io_thread.quit, QtCore.Qt.DirectConnection)
        self.io_thread.finished.connect(self.receiver.deleteLater)
        self.io_thread.start()
        self.server.close()

    def send(self, kind, **values):
        if self.receiver is not None and self.io_thread.isRunning():
            data = json.dumps(dict(type=kind, **values)).encode()
            self.receiver.command.emit(struct.pack('!I',len(data))+data)

    def receive_message(self, message):
        if self.sender() is not self.receiver or self.stopping:
            return
        self.last_event = time.monotonic()
        self.messageReceived.emit(message)

    def read_messages(self):
        if self.sender() is not self.receiver or self.receiver is None:
            return
        latest = self.receiver.take()
        if latest and not self.stopping:
            self.last_event = time.monotonic()
            latest['gui_receive_monotonic'] = self.last_event
            self.latest_frame = latest
            self.frame.emit(latest['_image'])

    def check_timeout(self):
        if not self.stopping and time.monotonic()-self.last_event > 10:
            self.messageReceived.emit(dict(type='error', message='相机启动或通信超时'))
            self.stop()

    def stop(self):
        self.stopping = True
        self.watchdog.stop()
        self.send('stop')
        if self.active:
            self.kill_timer.start(3000)

    def read_log(self):
        self.log.emit(bytes(self.process.readAllStandardOutput()).decode(errors='replace'))

    def process_error(self, error):
        if error == QtCore.QProcess.FailedToStart:
            self.messageReceived.emit(dict(type='error', message=self.process.errorString()))
            self.finished()

    def finished(self, *args):
        unexpected = not self.stopping
        self.watchdog.stop()
        self.kill_timer.stop()
        self.read_log()
        self.server.close()
        self.close_receiver()
        if self.temp:
            self.temp.cleanup()
            self.temp = None
        if unexpected:
            self.messageReceived.emit(dict(type='error', message='相机进程已退出，请检查连接或运行日志'))
        self.stopped.emit()

    def shutdown(self):
        self.stop()
        if self.active and not self.process.waitForFinished(1500):
            self.process.kill()
            self.process.waitForFinished(2000)
        self.close_receiver()

    def close_receiver(self):
        if self.io_thread is not None:
            thread = self.io_thread
            if thread.isRunning():
                self.receiver.closeRequested.emit()
                # No GUI callback is required to finish the receiver thread.
                thread.wait()
            self.receiver = self.io_thread = self.socket = None
            thread.deleteLater()
