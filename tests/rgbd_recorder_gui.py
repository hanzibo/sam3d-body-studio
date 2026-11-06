# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Protocol replay GUI acceptance; synthetic captures, no camera or models."""
import json
from pathlib import Path
import sys
import tempfile
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'gui'),str(ROOT)]
from PyQt5 import QtWidgets
from rgbd_recorder import Recorder
from recorder_session import RecordingSession,recording_active
from rgbd_batch import PoseSet

app=QtWidgets.QApplication([])

def wait_for(predicate,timeout=6):
    deadline=time.monotonic()+timeout
    while not predicate() and time.monotonic()<deadline:app.processEvents();time.sleep(.01)
    assert predicate()

with tempfile.TemporaryDirectory() as temporary:
    root=Path(temporary);fake=root/'fake_capture.py'
    fake.write_text('''import argparse,base64,json,time,sys
from pathlib import Path
import cv2,numpy as np
p=argparse.ArgumentParser();p.add_argument('--output');args,_=p.parse_known_args()
def emit(kind,**kw):print('@@RGBD '+json.dumps(dict(type=kind,**kw)),flush=True)
image=np.full((120,160,3),100,np.uint8);_,jpeg=cv2.imencode('.jpg',image)
emit('ready',message='Synthetic camera protocol test')
emit('frame',jpeg=base64.b64encode(jpeg).decode(),depth_jpeg=base64.b64encode(jpeg).decode(),skew_ms=1.,valid_depth_fraction=1.)
line=sys.stdin.readline();command=json.loads(line)
emit('countdown',seconds=1);time.sleep(.3)
output=Path(args.output);output.mkdir(parents=True)
for name in ('rgb.png','rgb_raw.png','depth_preview.png'):cv2.imwrite(str(output/name),image)
cv2.imwrite(str(output/'depth_raw.png'),np.full((120,160),2000,np.uint16))
np.savez_compressed(output/'rgbd.npz',depth_m=np.full((120,160),2.,np.float32),xyz_m=np.ones((120,160,3),np.float32))
meta=dict(schema='studio_rgbd_v1',units='metres',rgb_rectified=True,timestamps=dict(skew_ms=1.),max_timestamp_skew_ms=20)
(output/'camera.json').write_text(json.dumps(meta))
emit('done',directory=str(output))
''')
    w=Recorder(camera_python=sys.executable,capture_script=fake);w.sound.setChecked(False);w.show()
    session=RecordingSession.create(root/'session',2);w.use_session(session)
    assert not w.capture_btn.isEnabled() and not w.process.processId()
    w.start_preview();wait_for(lambda:w.frame_count>0 and w.state=='preview')
    assert w.capture_btn.isEnabled() and not w.poses.isEnabled()
    w.begin_capture();wait_for(lambda:w.state=='idle')
    assert session.completed==1 and (session.directory/'pose_set.json').is_file()
    first=session.poses.sample_path(session.poses.poses[0]);assert first.is_dir()
    # A failed/cancelled retake must preserve the selected completed snapshot.
    w.start_preview();wait_for(lambda:w.capture_btn.isEnabled());w.begin_capture();w.stop()
    wait_for(lambda:w.state=='idle')
    assert session.poses.sample_path(session.poses.poses[0])==first
    # Retake 0, automatically advance to 1 and finish with both snapshots on disk.
    w.auto.setChecked(True);w.gap.setValue(1);w.start_preview()
    wait_for(lambda:w.capture_btn.isEnabled());w.begin_capture()
    wait_for(lambda:session.completed==2 and not w.busy,12)
    assert not w.auto_sequence
    p=PoseSet(session.directory);assert p.poses[0]['previous_samples'] and first.is_dir()
    assert w.progress.value()==2
    w.close();app.processEvents();assert not recording_active(session.directory)
    assert 'torch' not in sys.modules and 'pyorbbecsdk' not in sys.modules
    # Missing interpreter releases controls without marking a pose as captured.
    w=Recorder(camera_python=root/'missing');w.use_session(RecordingSession(root/'session'))
    w.start_preview();assert w.state=='idle' and '不存在' in w.status.text();w.close()
print('RGBD_RECORDER_GUI_OK: preview, save, cancel, retake, automatic sequence, resume, no model imports')
