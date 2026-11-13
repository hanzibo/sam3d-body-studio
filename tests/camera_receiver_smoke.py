# SPDX-License-Identifier: GPL-3.0-only
"""Real socket/thread tests, plus optional physical MJPG preview/PNG capture."""
import base64
import json
from pathlib import Path
import socket
import struct
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'gui'),str(ROOT)]
from PyQt5 import QtCore, QtGui
from camera_client import CameraClient

app=QtCore.QCoreApplication([])
def wait(predicate,seconds=10):
    end=time.monotonic()+seconds
    while not predicate():
        app.processEvents();time.sleep(.002)
        if time.monotonic()>end:raise AssertionError('timeout')
    app.processEvents()

def packet(message):
    raw=json.dumps(message).encode()
    return struct.pack('!I',len(raw))+raw

with tempfile.TemporaryDirectory() as directory:
    c=CameraClient();frames=[];messages=[]
    c.frame.connect(frames.append);c.messageReceived.connect(messages.append)
    assert c.server.listen(str(Path(directory)/'camera.sock'))
    peer=socket.socket(socket.AF_UNIX);peer.connect(c.server.fullServerName())
    try:
        wait(lambda:c.receiver is not None)
        image=QtGui.QImage(64,48,QtGui.QImage.Format_RGB888);image.fill(QtGui.QColor('red'))
        data=QtCore.QByteArray();buffer=QtCore.QBuffer(data);buffer.open(QtCore.QIODevice.WriteOnly)
        assert image.save(buffer,'JPG');raw=bytes(data);encoded=base64.b64encode(raw).decode()
        make=lambda index:packet(dict(type='frame',index=index,jpeg=encoded,received_monotonic=time.monotonic()))
        # Partial framing must not publish an incomplete JPEG.
        p=make(1);peer.sendall(p[:7]);app.processEvents();assert not frames
        peer.sendall(p[7:]);wait(lambda:c.latest_frame is not None)
        assert c.latest_frame['_image']==QtGui.QImage.fromData(raw,'JPG')
        assert c.receiver.thread()!=app.thread()
        n=len(frames)
        # Block GUI dispatch: the receiver must still drain/decode the socket.
        for i in range(2,32):peer.sendall(make(i))
        time.sleep(.15)
        with c.receiver.lock:
            assert c.receiver.latest['index']==31
        assert len(frames)==n
        wait(lambda:c.latest_frame['index']==31)
        assert len(frames)==n+1  # One wake-up, not 30 queued images.
        peer.sendall(packet(dict(type='ready',metadata={'fps':30})))
        wait(lambda:any(m['type']=='ready' for m in messages))
        peer.sendall(packet(dict(type='frame',index=32,jpeg='bad',received_monotonic=time.monotonic())))
        wait(lambda:any(m['type']=='error' for m in messages))
        peer.sendall(make(33));wait(lambda:c.latest_frame['index']==33)
        c.send('preview_rate',fps=30)
        peer.settimeout(2);p=peer.recv(4096);assert b'preview_rate' in p
    finally:
        c.shutdown();peer.close();app.processEvents()
    assert c.io_thread is None
    print('CAMERA_RECEIVER_THREAD_OK',flush=True)

    if '--physical' in sys.argv:
        c=CameraClient();messages=[];c.messageReceived.connect(messages.append)
        try:
            for cycle in range(2):
                messages.clear();c.start();wait(lambda:any(m['type']=='ready' for m in messages),30)
                wait(lambda:c.latest_frame is not None)
                c.send('preview_rate',fps=30)
                index=c.latest_frame['index'];wait(lambda:c.latest_frame['index']>index+15)
                assert c.latest_frame['_image'].width()==1280
                assert c.latest_frame.get('receipt_stage')=='sdk_color_frame_available'
                if cycle==0:
                    target=Path(directory)/'capture';c.send('capture',job='smoke',delay=0,directory=str(target))
                    wait(lambda:any(m['type']=='photo' for m in messages))
                    photo=QtGui.QImage(str(target/'photo.png'));assert not photo.isNull()
                    assert photo.width()==1280 and photo.height()==800
                c.stop();wait(lambda:not c.active);wait(lambda:c.io_thread is None)
            print('CAMERA_PHYSICAL_PREVIEW_CAPTURE_RESTART_OK',flush=True)
        finally:c.shutdown()
