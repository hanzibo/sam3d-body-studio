# SPDX-License-Identifier: GPL-3.0-only
"""Bounded rendering keeps each result paired with its frozen RGB snapshot."""
from collections import deque
from pathlib import Path
import sys
import time
from types import SimpleNamespace as NS
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'gui'))
from hand_live import HandLivePanel


class Label:
 def setText(self,*args):pass
 def setToolTip(self,*args):pass


class Harness:
 handle_event=HandLivePanel.handle_event
 submit_render=HandLivePanel.submit_render
 painted=HandLivePanel.painted
 end=HandLivePanel.end
 @property
 def running(self):return self.active or self.pending is not None or self.paint_pending is not None
 def __init__(self):
  self.active=True;self.warming=False;self.pending=None;self.paint_pending=None;self.render_waiting=None
  self.render_dropped=0;self.result_count=self.errors=self.successes=0;self.mode='overlay'
  self.runtime=None
  self.samples=deque();self.next_due=0;self.status=self.latency=Label();self.draws=[];self.finished=False
  self.viewer=NS(clear_mesh=lambda:None);self.overlay=NS(update=lambda:None,clear_frame=lambda image:None,painted_frame_index=None,paint_cpu_ms=.1)
  self.hand=NS(state='ready',job=None,info=Label(),update_controls=lambda:None,display_scene=self.display)
  self.poll=lambda:None;self.abort=lambda message:(_ for _ in ()).throw(AssertionError(message))
  self.finish=lambda:setattr(self,'finished',True)
 def display(self,image,data,report,*args,**kwargs):
  self.draws.append((image.pixelColor(0,0).name(),float(data['vertices'][0,0])))
 def result(self,directory,index,color):
  self.pending=str(index);self.last_index=index
  self.last_image=QtGui.QImage(2,2,QtGui.QImage.Format_RGB888);self.last_image.fill(QtGui.QColor(color))
  self.received_at=self.client_received_at=self.sent_at=self.queued_at=time.monotonic()
  self.client_decode_ms=self.gui_delivery_ms=0
  file=directory/f'{index}.npz'
  np.savez(file,vertices=np.full((3,3),index),faces=[[0,1,2]],joints=np.zeros((1,3)),edges=np.empty((0,2)),camera_matrix=np.eye(3))
  event=dict(type='hand_stream_result',job=str(index),frame_index=index,path=str(file),worker_ms=1,
             report=dict(hand_count=1,detect_ms=.1,reconstruct_ms=.9))
  assert self.handle_event(event)


def test_slow_paint_replaces_only_waiting_result_and_preserves_rgb(tmp_path):
 app=QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])
 h=Harness();h.result(tmp_path,1,'red');h.result(tmp_path,2,'green');h.result(tmp_path,3,'blue')
 assert h.draws==[('#ff0000',1.)] and h.render_dropped==1
 assert h.paint_pending['frame_index']==1 and h.render_waiting[-1]['frame_index']==3
 # A swap unrelated to this frame must not count it as displayed.
 h.painted('overlay');assert not h.samples
 h.overlay.painted_frame_index=1;h.painted('overlay')
 assert h.draws==[('#ff0000',1.),('#0000ff',3.)]
 h.overlay.painted_frame_index=3;h.painted('overlay')
 assert [s['frame_index'] for s in h.samples]==[1,3]
 assert h.render_waiting is None
 app.processEvents()


def test_stop_drains_inflight_and_drops_queued_render(tmp_path):
 app=QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])
 h=Harness();h.result(tmp_path,1,'red');h.result(tmp_path,2,'blue');h.pending='inflight'
 h.end('stopped')
 assert not h.finished and h.paint_pending is None and h.render_waiting is None
 assert h.handle_event(dict(type='error',job='inflight',message='cancelled'))
 assert h.finished and h.pending is None


def test_poll_does_not_wait_for_frame_swap():
 # With no new camera packet, poll still reaches the camera while a paint is pending.
 touched=[]
 class Host:
  camera_state='preview'
  @property
  def camera(self):touched.append(True);return NS(latest_frame=None)
 h=NS(pending=None,paint_pending={'frame_index':1},active=True,warming=False,next_due=0,
      host=Host(),isVisible=lambda:True,window=lambda:NS(isMinimized=lambda:False))
 HandLivePanel.poll(h);assert touched
 h.window=lambda:NS(isMinimized=lambda:True);touched.clear()
 HandLivePanel.poll(h);assert not touched


def test_failed_inference_does_not_leave_a_stale_paint_blocker(tmp_path):
 app=QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])
 h=Harness();h.result(tmp_path,1,'red');h.result(tmp_path,2,'blue');h.pending='failed'
 assert h.handle_event(dict(type='error',job='failed',message='bad frame'))
 assert h.errors==1 and h.paint_pending is None and h.render_waiting is None
 assert h.render_dropped==2 and not h.finished
