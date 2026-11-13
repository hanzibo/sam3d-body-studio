# SPDX-License-Identifier: GPL-3.0-only
"""Real auto hand detection, bimanual batch and sequence playback, off desktop."""
from pathlib import Path
import json,sys,tempfile,time,os
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'gui'),str(ROOT)]
from PyQt5 import QtCore
from app import create_application,MainWindow
from studio_paths import path
OUT=Path(os.environ.get('HAND_GUI_OUTPUT','/run/media/hzb/Data/experiments/hand-auto-20261112'));OUT.mkdir(parents=True,exist_ok=True)
DATA=Path('/run/media/hzb/Data/experiments/wilor/datasets')

def wait(app,predicate,seconds=120):
 end=time.monotonic()+seconds
 while not predicate():
  app.processEvents();time.sleep(.01)
  if time.monotonic()>end:raise AssertionError('timeout')
 app.processEvents()

def pause(app):
 for _ in range(20):app.processEvents();time.sleep(.02)

with tempfile.TemporaryDirectory() as settings:
 QtCore.QSettings.setPath(QtCore.QSettings.NativeFormat,QtCore.QSettings.UserScope,settings)
 app=create_application();w=MainWindow();h=w.hand;w.resize(1600,1100);w.workspace_tabs.setCurrentWidget(h);w.show();pause(app)
 try:
  source=DATA/'20261112_173116_two_hands_acdf29'
  h.accelerated_btn.click();assert h.feedback_cache.isChecked() and h.intermediate_keypoints.isChecked() and h.final_hand_vertices.isChecked()
  h.int8_btn.click();assert h.backbone_choice.currentData()=='trt_int8' and h.final_hand_vertices.isChecked()
  h.baseline_btn.click();assert not h.feedback_cache.isChecked() and not h.intermediate_keypoints.isChecked() and not h.final_hand_vertices.isChecked()
  h.load_image(source/'frames/000150.jpg');h.automatic.setChecked(True)
  execution=json.loads(os.environ.get('HAND_GUI_EXECUTION','{}'))
  if execution:
   h.detector_choice.setCurrentIndex(h.detector_choice.findData(execution.get('detector','native')))
   h.backbone_choice.setCurrentIndex(h.backbone_choice.findData(execution.get('backbone','native')))
   for key in ('decoder_compile','mhr_compile','mhr_graph','feedback_cache','intermediate_keypoints','final_hand_vertices'):getattr(h,key).setChecked(execution.get(key,False))
   h.compile_mode.setCurrentIndex(h.compile_mode.findData(execution.get('compile_mode','default')))
  h.start();wait(app,lambda:h.state in ('ready','off'));assert h.state=='ready',h.info.text()
  h.submit('auto_detect');wait(app,lambda:h.state=='ready');assert len(h.canvas.detected)==2,h.info.text()
  assert not h.viewer._hand_active
  h.submit('warmup');wait(app,lambda:h.state=='ready');assert '预热完成' in h.info.text(),h.info.text()
  h.submit('run');wait(app,lambda:h.state=='ready');report=json.loads((h.result_dir/'report.json').read_text())
  assert report['hand_reports'][0]['execution']==h.execution_config()
  assert not h.backbone_choice.isEnabled() and not h.int8_btn.isEnabled()
  if execution.get('backbone')=='trt_int8':
   runtime=report['hand_reports'][0]['backbone_runtime'];assert runtime['max_batch']==1 and runtime['precision']=='INT8/FP16 mixed'
   assert report['hand_reports'][0]['backbone']=='TensorRT INT8/FP16 mixed'
   assert report['hand_reports'][0]['backbone_engine_executions']==2
  assert report['batch_size']==2 and report['execution_counts']==dict(body=0,hand=1,backbone=1)
  h.tabs.setCurrentIndex(2);pause(app);w.grab().save(str(OUT/'automatic-dual-overlay.png'));h.overlay.grabFramebuffer().save(str(OUT/'automatic-dual-overlay-frame.png'))
  h.tabs.setCurrentIndex(1);pause(app);w.grab().save(str(OUT/'automatic-dual-3d.png'))
  s=h.sequence;h.tabs.setCurrentWidget(s);s.load_source(source);s.stride.setValue(120);s.run()
  wait(app,lambda:h.state=='ready',180)
  assert s.directory and (s.directory/'sequence.json').is_file(),h.info.text()
  info=json.loads((s.directory/'sequence.json').read_text());assert info['complete'] and 'video_error' not in info,info
  assert len(s.rows)==4 and (s.directory/'overlay.mp4').stat().st_size>10000
  s.slider.setValue(3);s.view.setCurrentIndex(1);pause(app);w.grab().save(str(OUT/'sequence-3d.png'))
  s.view.setCurrentIndex(0);pause(app);w.grab().save(str(OUT/'sequence-overlay.png'))
  s.slider.setValue(0);s.toggle_play();assert s.timer.isActive();s.advance();assert s.slider.value()==1;s.timer.stop()
  (OUT/'gui-validation.json').write_text(json.dumps(dict(automatic_report=report,sequence=str(s.directory),info=info),indent=2))
  print('AUTO_HAND_GUI_OK: detector, dual batch, calibration, sequence export and playback')
 finally:w.close();app.processEvents()
