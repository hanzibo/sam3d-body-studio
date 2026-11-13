# SPDX-License-Identifier: GPL-3.0-only
"""Real INT8 startup warmup with camera off, then immediate live replay."""
from pathlib import Path
import base64
import json
import os
import sys
import tempfile
import time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'gui'),str(ROOT)]
from PyQt5 import QtCore,QtGui
from app import MainWindow,create_application
from hand_sequence import read_sequence

OUT=Path(os.environ.get('HAND_STARTUP_OUT','/run/media/hzb/Data/experiments/hand-startup-warmup-20261113'));OUT.mkdir(parents=True,exist_ok=True)
def wait(app,predicate,seconds=300):
 end=time.monotonic()+seconds
 while not predicate():
  app.processEvents();time.sleep(.002)
  if time.monotonic()>end:raise AssertionError('timeout')
 app.processEvents()

with tempfile.TemporaryDirectory() as settings:
 QtCore.QSettings.setPath(QtCore.QSettings.NativeFormat,QtCore.QSettings.UserScope,settings)
 app=create_application();w=MainWindow();h=w.hand;l=h.live
 w.resize(1600,1050);w.workspace_tabs.setCurrentWidget(h);h.tabs.setCurrentWidget(l);w.show()
 timer=QtCore.QTimer();timer.setTimerType(QtCore.Qt.PreciseTimer)
 try:
  assert h.startup_warmup_check.isChecked() and h.startup_warmup_source.isVisible()
  assert not w.camera.active and w.camera_state=='off'
  (h.accelerated_btn if os.environ.get('HAND_STARTUP_FP16') else h.int8_btn).click()
  if os.environ.get('HAND_STARTUP_FILTER_TEST'):h.pose_filter.setChecked(True)
  count=int(os.environ.get('HAND_STARTUP_PREDICTIONS','5'))
  h.prediction_count.setCurrentIndex(h.prediction_count.findData(count))
  began=time.monotonic();h.start()
  wait(app,lambda:h.state in ('ready','off'));assert h.state=='ready',h.info.text()
  assert h.startup_warmup['status']=='ready',h.startup_warmup
  assert h.startup_warmup['batch_sizes']==[1,2]
  assert not h.prediction_count.isEnabled() and not h.intermediate_compile.isEnabled()
  assert [1280,800] in h.startup_warmup['image_sizes']
  assert not w.camera.active and w.camera_state=='off'
  w.grab().save(str(OUT/'offline-ready.png'))
  startup=dict(h.startup_warmup,total_gui_seconds=time.monotonic()-began)
  streams={}
  for name in ('20261112_173053_single_hand_d4f1fc','20261112_173116_two_hands_acdf29'):
   source=Path('/run/media/hzb/Data/experiments/wilor/datasets')/name
   manifest,rows,k=read_sequence(source);raw=(source/rows[150]['file']).read_bytes()
   image=QtGui.QImage.fromData(raw,'JPG');index=[0]
   def feed():
    i=index[0];index[0]+=1;now=time.monotonic()
    w.camera.latest_frame=dict(index=i,jpeg=base64.b64encode(raw).decode(),_jpeg_bytes=raw,_image=image,received_monotonic=now)
    w.camera.frame.emit(image)
   timer.timeout.connect(feed);timer.start(33);w.camera_state='preview';w.capture_metadata=manifest['profile'];feed()
   # Deliberately do not call online warmup: test the first live result.
   h.hand_selection.setCurrentIndex(h.hand_selection.findData('both'))
   l.begin();wait(app,lambda:len(l.samples)>=8,60)
   if os.environ.get('HAND_STARTUP_FILTER_TEST'):
    assert list(l.samples)[-1]['one_euro']['pose']
    assert list(l.samples)[-1]['one_euro']['raw_preserved']
    assert all(n>=2 for n in list(l.samples)[-1]['one_euro']['history_frames'])
    h.filter_parameters['min_cutoff'].setValue(.8);h.position_filter.setChecked(True)
    old=len(l.samples);wait(app,lambda:len(l.samples)>=old+5,60)
    last=list(l.samples)[-1]['one_euro'];assert last['position'] and last['min_cutoff']==.8,last
    h.pose_filter.setChecked(False);h.position_filter.setChecked(False)
    old=len(l.samples);wait(app,lambda:len(l.samples)>=old+4,60)
    assert not list(l.samples)[-1]['one_euro']['pose']
    h.pose_filter.setChecked(True)
   if os.environ.get('HAND_STARTUP_SELECTION_TEST'):
    for selection,threshold in [('left',.4),('right',.5),('both',.4)]:
     h.hand_selection.setCurrentIndex(h.hand_selection.findData(selection));h.confidence.setValue(threshold)
     old=len(l.samples);wait(app,lambda:len(l.samples)>=old+5,60)
     for sample in list(l.samples)[-2:]:
      assert sample['detection_options']==dict(confidence=threshold,hand_selection=selection),sample
      assert sample['hands']==(1 if selection!='both' or 'single_hand' in name else 2),sample
   l.end('test');wait(app,lambda:not l.running)
   streams[name]=l.last_summary
   assert all(s['hands']==(1 if 'single_hand' in name or s.get('detection_options',{}).get('hand_selection','both')!='both' else 2) for s in l.samples)
   assert l.runtime['execution']['intermediate_compile'] is True
   assert l.runtime['execution']['intermediate_predictions']==count
   assert l.runtime['feedback_runtime']['last']['intermediate_keypoint_calls']==count
   assert l.samples[0]['worker_ms']<600,l.samples[0]
   timer.stop();timer.timeout.disconnect(feed)
   w.camera_state='off';w.camera.latest_frame=None
  (OUT/'validation.json').write_text(json.dumps(dict(startup=startup,streams=streams),indent=2))
  print('OFFLINE_STARTUP_WARMUP_OK',json.dumps(dict(startup=startup,first_live_ms={k:v['samples'][0]['worker_ms'] for k,v in streams.items()})),flush=True)
 finally:
  timer.stop();h.shutdown();w.camera_state='off';w.close();app.processEvents()
