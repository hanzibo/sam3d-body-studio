"""Qt process/socket adapter; the GUI never imports the camera SDK."""
import base64
import json
from pathlib import Path
import struct
import tempfile
import time
from PyQt5 import QtCore, QtGui, QtNetwork

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

    def start(self):
        if self.active:
            return
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
        self.process.start('/bin/bash', [str(ROOT/'launch_camera.sh'), '--socket', path])
        self.watchdog.start()

    def connected(self):
        self.socket = self.server.nextPendingConnection()
        self.socket.readyRead.connect(self.read_messages)
        self.server.close()

    def send(self, kind, **values):
        if self.socket is not None and self.socket.state() == QtNetwork.QLocalSocket.ConnectedState:
            data = json.dumps(dict(type=kind, **values)).encode()
            self.socket.write(struct.pack('!I',len(data))+data)
            self.socket.flush()

    def read_messages(self):
        self.buffer += bytes(self.socket.readAll())
        latest = None
        while len(self.buffer) >= 4:
            size = struct.unpack('!I', self.buffer[:4])[0]
            if size > 8*1024*1024:
                self.messageReceived.emit(dict(type='error', message='相机消息格式无效'))
                self.stop()
                return
            if len(self.buffer) < size+4:
                break
            message = json.loads(self.buffer[4:4+size])
            self.buffer = self.buffer[4+size:]
            self.last_event = time.monotonic()
            if message['type'] == 'frame':
                latest = message
            else:
                self.messageReceived.emit(message)
        if latest and not self.stopping:
            image = QtGui.QImage.fromData(base64.b64decode(latest['jpeg']), 'JPG')
            if not image.isNull():
                self.latest_frame = latest
                self.frame.emit(image)

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
        if self.socket:
            self.socket.close()
            self.socket.deleteLater()
            self.socket = None
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
