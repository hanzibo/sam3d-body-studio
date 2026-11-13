# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""GUI protocol acceptance; --hardware explicitly opens the connected Gemini."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'gui'),str(ROOT)]
from PyQt5 import QtWidgets
from rgbd_recorder import Recorder
from rgbd_sequence import RGBDSequence
from recorder_session import recording_active

p=argparse.ArgumentParser();p.add_argument('--hardware',action='store_true');p.add_argument('--duration',type=int,default=5);p.add_argument('--fps',type=int,default=30);args=p.parse_args()
app=QtWidgets.QApplication([])

def wait(predicate,timeout=20):
    deadline=time.monotonic()+timeout
    while not predicate() and time.monotonic()<deadline:app.processEvents();time.sleep(.01)
    if not predicate():raise AssertionError('timeout: '+w.stream.status.text()+'\n'+w.stream.log.toPlainText()[-1500:])

with tempfile.TemporaryDirectory() as temp:
    temp=Path(temp)
    fake=temp/'fake_stream.py'
    fake.write_text('''import argparse,base64,json,time,sys
from pathlib import Path
import cv2,numpy as np
import os
sys.path.insert(0,os.getcwd())
from rgbd_sequence import SequenceWriter,action_plan
p=argparse.ArgumentParser();p.add_argument('--output');p.add_argument('--name');a=p.parse_args()
def emit(kind,**kw):print('@@RGBD '+json.dumps(dict(type=kind,**kw)),flush=True)
rgb=np.full((120,160,3),100,np.uint8);raw=np.full((120,160),2000,np.uint16)
_,jpg=cv2.imencode('.jpg',rgb);encoded=base64.b64encode(jpg).decode()
emit('ready',message='synthetic protocol replay')
emit('frame',jpeg=encoded,depth_jpeg=encoded,skew_ms=1,valid_depth_fraction=1)
c=json.loads(sys.stdin.readline())
if c['type']=='stop':sys.exit(0)
emit('countdown',seconds=1);time.sleep(.15)
s=dict(max_skew_ms=20,duration_s=c['duration'],action_plan=action_plan(c['duration']))
w=SequenceWriter(a.output,a.name,{},s)
emit('recording',directory=str(Path(a.output).resolve()))
for i in range(4):
 w.submit(rgb,raw,dict(color_timestamp_ms=i*100,depth_timestamp_ms=i*100+1,host_monotonic_ns=i*10**8,relative_time_s=i*.1,skew_ms=1,depth_scale_mm=1))
 time.sleep(.08)
 emit('frame',jpeg=encoded,depth_jpeg=encoded,skew_ms=1,valid_depth_fraction=1)
 emit('progress',elapsed_s=i*.1,remaining_s=c['duration']-i*.1,action='自然站立，保持静止',rejected_skew=0,**w.stats())
emit('saving',message='saving')
m=w.close('completed');emit('sequence_done',directory=str(Path(a.output).resolve()),summary=m)
''')
    if args.hardware:
        w=Recorder(sequence_root=ROOT/'outputs/rgbd-sequences')
    else:
        w=Recorder(camera_python=sys.executable,stream_script=fake,sequence_root=temp/'sequences')
    w.show();w.mode_tabs.setCurrentIndex(1);s=w.stream;s.sound.setChecked(False)
    s.name.setText('验证_相机链路_非动作数据' if args.hardware else '测试_双臂_2米')
    s.duration.setValue(args.duration);s.delay.setValue(1);s.fps.setCurrentIndex(s.fps.findData(args.fps))
    try:
        s.start_preview();wait(lambda:s.state=='preview' and s.frames>0)
        assert not w.mode_tabs.isTabEnabled(0) and not s.pending.exists()
        # Cancel before recording: no saved sequence, camera must be reusable.
        s.stop();wait(lambda:not s.busy)
        assert not s.pending.exists() and w.mode_tabs.isTabEnabled(0)
        if args.hardware:
            s.delay.setValue(5);s.start_preview();wait(lambda:s.state=='preview' and s.frames>0)
            s.begin_record();wait(lambda:s.clock.text()=='5')
            s.stop();wait(lambda:not s.busy)
            assert not s.pending.exists()
            print('STREAM_COUNTDOWN_CANCEL_OK',flush=True)
            s.delay.setValue(1)
        s.start_preview();wait(lambda:s.state=='preview' and s.frames>0)
        s.begin_record();wait(lambda:s.state=='recording')
        assert recording_active(s.pending)
        image_root=ROOT/'outputs/rgbd-stream-ui-validation';image_root.mkdir(parents=True,exist_ok=True)
        app.processEvents();w.grab().save(str(image_root/('hardware-recording.png' if args.hardware else 'recording-ui.png')))
        wait(lambda:not s.busy,args.duration+25)
        assert s.summary and s.summary['status']=='completed',s.summary
        seq=RGBDSequence(s.pending);assert len(seq.frames)>1
        assert not recording_active(s.pending) and w.mode_tabs.isTabEnabled(0)
        assert s.name.isEnabled()
        seq.read(0)
        if args.hardware:seq.read(0,registered=True)
        print('STREAM_GUI_OK',s.pending,s.summary['saved_frames'],s.summary['actual_saved_fps'],flush=True)
        if args.hardware:
            # Explicit early stop must retain committed frames and release the device.
            s.name.setText('验证_提前结束_非动作数据');s.duration.setValue(10);s.delay.setValue(0)
            s.start_preview();wait(lambda:s.state=='preview' and s.frames>0);s.begin_record()
            wait(lambda:s.state=='recording');wait(lambda:'已写' in s.stats.text() and '已写 0 帧' not in s.stats.text())
            s.stop();wait(lambda:not s.busy,20)
            assert s.summary['status']=='stopped' and s.summary['saved_frames']>0,s.summary
            RGBDSequence(s.pending).read(0)
            print('STREAM_EARLY_STOP_OK',s.pending,s.summary['saved_frames'],flush=True)
        assert 'torch' not in sys.modules and 'pyorbbecsdk' not in sys.modules
    finally:w.close();app.processEvents()
