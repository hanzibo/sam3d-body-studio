# SPDX-License-Identifier: GPL-3.0-only
"""Socket parsing and JPEG decoding off the GUI; one coalesced frame mailbox."""
import base64
import json
import struct
import threading
import time
from PyQt5 import QtCore, QtGui


class CameraReceiver(QtCore.QObject):
    available = QtCore.pyqtSignal()
    message = QtCore.pyqtSignal(dict)
    closed = QtCore.pyqtSignal()
    command = QtCore.pyqtSignal(bytes)
    closeRequested = QtCore.pyqtSignal()

    def __init__(self, socket):
        super().__init__()
        self.socket = socket
        self.buffer = bytearray()
        self.lock = threading.Lock()
        self.latest = None
        self.notified = False
        self.stopped = False
        self.command.connect(self.send, QtCore.Qt.QueuedConnection)
        self.closeRequested.connect(self.close, QtCore.Qt.QueuedConnection)

    @QtCore.pyqtSlot()
    def start(self):
        self.socket.readyRead.connect(self.read)
        self.socket.disconnected.connect(self.close)
        self.read()

    @QtCore.pyqtSlot(bytes)
    def send(self, data):
        if not self.stopped:
            self.socket.write(data)
            self.socket.flush()

    @QtCore.pyqtSlot()
    def read(self):
        if self.stopped:
            return
        self.buffer.extend(bytes(self.socket.readAll()))
        latest = None
        try:
            offset = 0
            while len(self.buffer)-offset >= 4:
                size = struct.unpack_from('!I', self.buffer, offset)[0]
                if size > 8*1024*1024:
                    raise ValueError('相机消息格式无效')
                if len(self.buffer)-offset < size+4:
                    break
                packet = json.loads(self.buffer[offset+4:offset+4+size])
                offset += size+4
                if packet['type'] == 'frame':
                    packet['client_receive_monotonic'] = time.monotonic()
                    latest = packet
                else:
                    self.message.emit(packet)
            del self.buffer[:offset]
            if latest is None:
                return
            began = time.monotonic()
            raw = base64.b64decode(latest['jpeg'], validate=True)
            image = QtGui.QImage.fromData(raw, 'JPG')
            latest['client_decode_ms'] = (time.monotonic()-began)*1000
            if image.isNull():
                raise ValueError('无法解码相机彩色图像')
            # QImage owns its storage; readers never mutate the published image.
            latest.update(_image=image, _jpeg_bytes=raw, decoded_monotonic=time.monotonic())
            with self.lock:
                self.latest = latest
                notify = not self.notified
                self.notified = True
            if notify:
                self.available.emit()
        except (ValueError, KeyError, TypeError) as exc:
            self.buffer.clear()
            self.message.emit(dict(type='error', message=str(exc)))

    def take(self):
        with self.lock:
            latest, self.latest = self.latest, None
            self.notified = False
            return latest

    @QtCore.pyqtSlot()
    def close(self):
        if self.stopped:
            return
        self.stopped = True
        self.socket.close()
        self.socket.deleteLater()
        with self.lock:
            self.latest = None
        self.closed.emit()
