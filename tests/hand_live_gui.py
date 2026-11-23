# SPDX-License-Identifier: GPL-3.0-only
"""Real worker + recorded RGB live injection + physical Gemini preview smoke."""
from pathlib import Path
import sys,time,json,base64,tempfile,os
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'gui'),str(ROOT)]
from PyQt5 import QtCore,QtGui
from app import MainWindow,create_application
from studio.hand.hand_sequence import read_sequence
OUT=Path(os.environ.get('HAND_LIVE_OUT','/run/media/hzb/Data/experiments/hand-live-20261113'));OUT=OUT/('fp16' if os.environ.get('HAND_LIVE_FP16') else 'int8');OUT.mkdir(parents=True,exist_ok=True)
def wait(app,predicate,seconds=180):
 end=time.monotonic()+seconds
 while not predicate():
  app.processEvents();time.sleep(.005)
  if time.monotonic()>end:raise AssertionError('timeout')
 app.processEvents()
with tempfile.TemporaryDirectory() as settings:
 QtCore.QSettings.setPath(QtCore.QSettings.NativeFormat,QtCore.QSettings.UserScope,settings)
 app=create_application();w=MainWindow();h=w.hand;l=h.live;w.resize(1600,1050);w.workspace_tabs.setCurrentWidget(h);h.tabs.setCurrentWidget(l);w.show()
 try:
  print('GUI_CREATED',flush=True)
  (h.accelerated_btn if os.environ.get('HAND_LIVE_FP16') else h.int8_btn).click();h.start();wait(app,lambda:h.state in ('ready','off'));assert h.state=='ready',h.info.text()
  source=Path(os.environ.get('HAND_LIVE_SOURCE','/run/media/hzb/Data/experiments/wilor/datasets/20261112_173116_two_hands_acdf29'));expected_hands=1 if 'single_hand' in source.name else 2;manifest,records,k=read_sequence(source)
  w.camera_state='preview';w.capture_metadata=manifest['profile'];index=[0]
  def feed():
   i=index[0];index[0]+=1
   row=records[(150+i)%len(records)];raw=(source/row['file']).read_bytes()
   received=time.monotonic();image=QtGui.QImage.fromData(raw,'JPG')
   w.camera.latest_frame=dict(index=i,jpeg=base64.b64encode(raw).decode(),received_monotonic=received,client_receive_monotonic=received,_image=image,_jpeg_bytes=raw)
   w.camera.frame.emit(image)
  timer=QtCore.QTimer();timer.timeout.connect(feed);timer.start(33);feed();l.warmup()
  wait(app,lambda:not l.running,300);assert '预热完成' in l.status.text(),l.status.text();print('WARMED',flush=True)
  l.begin();assert l.running;assert not h.run.isEnabled() and not h.sequence.start.isEnabled()
  wait(app,lambda:len(l.samples)>=25,90);w.grab().save(str(OUT/'recorded-overlay.png'));l.overlay.grabFramebuffer().save(str(OUT/'overlay-frame.png'))
  assert all(x['hands']==expected_hands for x in l.samples),l.status.text()
  assert l.runtime['execution']==h.execution_config(),l.runtime
  assert l.runtime['execution_counts']['body']==0
  if not os.environ.get('HAND_LIVE_FP16'):assert l.runtime['backbone_engine_executions']==expected_hands
  l.end('test stop');wait(app,lambda:not l.running);overlay=l.last_summary;assert l.temp is None
  print('OVERLAY',json.dumps({k:v for k,v in overlay.items() if k not in ('samples','camera')}),flush=True)
  l.view.setCurrentIndex(1);l.begin();wait(app,lambda:len(l.samples)>=8,60);w.grab().save(str(OUT/'recorded-3d.png'));l.end('test stop');wait(app,lambda:not l.running);three=l.last_summary
  timer.stop();w.camera_state='off';w.camera.latest_frame=None
  l.open_camera();wait(app,lambda:w.camera_state=='preview',30);wait(app,lambda:w.camera.latest_frame is not None,15)
  print('PHYSICAL_CAMERA_READY',json.dumps(w.capture_metadata),flush=True)
  l.view.setCurrentIndex(0);l.begin();wait(app,lambda:len(l.samples)>=8,90);w.grab().save(str(OUT/'physical-camera.png'));l.end('physical done');wait(app,lambda:not l.running);physical=l.last_summary
  l.close_camera();wait(app,lambda:not w.camera.active,15)
  # Restart after a drained stop, then release during a real in-flight request.
  w.camera_state='preview';w.capture_metadata=manifest['profile'];timer.start(33);feed();l.begin()
  wait(app,lambda:l.pending is not None);temporary=Path(l.temp.name);h.shutdown();timer.stop()
  assert not temporary.exists() and not l.running and h.state=='off'
  w.camera_state='off'
  (OUT/'validation.json').write_text(json.dumps(dict(recorded_overlay=overlay,recorded_3d=three,physical_camera=physical),indent=2))
  print('HAND_LIVE_GUI_OK',flush=True)
 finally:w.close();app.processEvents()
