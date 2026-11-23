# SPDX-License-Identifier: GPL-3.0-only
"""Recorded camera packets through the real local socket, worker and overlay."""
import base64
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'gui'), str(ROOT)]
from PyQt5 import QtCore, QtNetwork
from app import MainWindow, create_application
from studio.hand.hand_sequence import read_sequence


def wait(app, predicate, seconds=180):
    deadline = time.monotonic()+seconds
    while not predicate():
        app.processEvents()
        time.sleep(.001)
        if time.monotonic()>deadline:
            raise AssertionError('Timed out')
    app.processEvents()


with tempfile.TemporaryDirectory() as settings:
    QtCore.QSettings.setPath(QtCore.QSettings.NativeFormat, QtCore.QSettings.UserScope, settings)
    app = create_application()
    w = MainWindow(); h = w.hand; live = h.live
    w.resize(1600, 1050); w.workspace_tabs.setCurrentWidget(h)
    h.tabs.setCurrentWidget(live); w.show()
    source = Path('/run/media/hzb/Data/experiments/wilor/datasets/20261112_173053_single_hand_d4f1fc')
    manifest, records, matrix = read_sequence(source)
    # Prepare compressed packets before timing; playback cannot stall on disk IO.
    images = [(source/r['file']).read_bytes() for r in records]
    peer = QtNetwork.QLocalSocket()
    timer = QtCore.QTimer(); timer.setTimerType(QtCore.Qt.PreciseTimer)
    index = [0]
    def feed():
        i = index[0]; index[0] += 1
        raw = images[(150+i)%len(images)]
        packet = json.dumps(dict(type='frame', index=i, jpeg=base64.b64encode(raw).decode(),
                                 received_monotonic=time.monotonic())).encode()
        peer.write(struct.pack('!I', len(packet))+packet); peer.flush()
    timer.timeout.connect(feed)
    try:
        h.int8_btn.click(); h.start()
        wait(app, lambda:h.state in ('ready','off')); assert h.state=='ready',h.info.text()
        assert w.camera.server.listen(str(Path(settings)/'replay.sock'))
        peer.connectToServer(w.camera.server.fullServerName())
        wait(app, lambda:w.camera.socket is not None)
        w.camera_state='preview'; w.capture_metadata=manifest['profile']
        timer.start(33); feed(); wait(app, lambda:w.camera.latest_frame is not None)
        live.warmup(); wait(app, lambda:not live.running,300)
        assert '预热完成' in live.status.text(),live.status.text()
        live.begin(); wait(app, lambda:len(live.samples)>=int(os.environ.get('HAND_BENCH_FRAMES','70')),120)
        live.end('benchmark'); wait(app, lambda:not live.running)
        result = live.last_summary
        assert all(r['hands']==1 for r in result['samples'])
        assert all(a['frame_index']<b['frame_index'] for a,b in zip(result['samples'],result['samples'][1:]))
        destination = Path(os.environ['HAND_BENCH_OUT']); destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_text(json.dumps(result,indent=2))
        print(json.dumps({k:v for k,v in result.items() if k not in ('samples','camera','runtime')},indent=2),flush=True)
        if os.environ.get('HAND_BENCH_PARITY'):
            timer.stop()
            captures=[];display=h.display_scene;write=h.process.write;compact=[False]
            def capture(image,data,*args,**kwargs):
                captures.append({k:np.array(v,copy=True) for k,v in data.items()})
                return display(image,data,*args,**kwargs)
            def send(data):
                command=json.loads(bytes(data))
                command['compact_result']=compact[0]
                return write((json.dumps(command)+'\n').encode())
            h.display_scene=capture;h.process.write=send
            live.limit.setValue(1)
            # Freeze exactly one compressed image, with a new stream/tracker each run.
            raw=images[150%len(images)]
            for mode in (False,True):
                compact[0]=mode
                p=json.dumps(dict(type='frame',index=index[0],jpeg=base64.b64encode(raw).decode(),received_monotonic=time.monotonic())).encode()
                index[0]+=1;peer.write(struct.pack('!I',len(p))+p);peer.flush()
                wait(app,lambda:w.camera.latest_frame['index']==index[0]-1)
                live.begin();wait(app,lambda:len(live.samples)==1)
                live.end('parity');wait(app,lambda:not live.running)
            assert len(captures)==2
            differences={k:float(np.max(np.abs(captures[0][k].astype(float)-captures[1][k].astype(float)))) if captures[0][k].size else 0 for k in captures[0]}
            assert all(v<=1e-6 for v in differences.values()),differences
            destination.with_suffix('.parity.json').write_text(json.dumps(differences,indent=2))
            print('FULL_COMPACT_RENDER_PARITY',json.dumps(differences),flush=True)
            h.display_scene=display;h.process.write=write
    finally:
        timer.stop(); peer.close(); w.camera_state='off'
        w.camera.shutdown(); h.shutdown(); w.close(); app.processEvents()
