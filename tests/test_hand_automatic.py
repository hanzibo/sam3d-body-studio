# SPDX-License-Identifier: GPL-3.0-only
import json
from pathlib import Path
import numpy as np
import pytest
from studio.hand.hand_automatic import HandSideTracker,choose_hands,pack_scene
from studio.hand.hand_sequence import read_sequence


def hand(side,box,score=.9):return dict(side=side,raw_side=side,bbox=box,score=score)

def test_slots_do_not_retain_stale_mesh_and_class_correction_requires_proximity():
 t=HandSideTracker();a=hand('left',[0,0,40,80]);b=hand('right',[150,0,190,80])
 assert [r['side'] for r in t.update([a,b],0)]==['left','right']
 assert t.update([],0.1)==[]
 result=t.update([hand('right',[1,0,41,80])],.2)
 assert len(result)==1 and result[0]['side']=='left' and result[0]['raw_side']=='right'
 result=t.update([hand('right',[1,0,41,80])],1)
 assert len(result)==1 and result[0]['side']=='right'

def test_never_invent_opposite_side_for_single_detection():
 assert [x['side'] for x in choose_hands([hand('right',[0,0,40,80])])]==['right']
 assert HandSideTracker().update([hand('right',[0,0,40,80])],0)[0]['side']=='right'


@pytest.mark.parametrize('stabilize',[False,True])
def test_forced_side_and_confidence_apply_before_reconstruction(monkeypatch,stabilize):
 from types import SimpleNamespace as NS
 from studio.hand.hand_automatic import AutomaticHands
 import studio.hand.hand_automatic as hand_automatic
 candidates=[hand('right',[0,0,40,80],.91),hand('left',[0,0,40,80],.84)]
 calls=[];thresholds=[]
 def detect(rgb,confidence):thresholds.append(confidence);return candidates,1.
 def predict(rgb,hands,padding,matrix):
  calls.append([h['side'] for h in hands])
  return [({}, {}, None,dict(inference_ms=2,timings={},execution_counts={})) for h in hands]
 obj=AutomaticHands.__new__(AutomaticHands)
 obj.detector=NS(detect=detect,metadata={});obj.tracker=HandSideTracker();obj.engine=NS(predict_many=predict)
 monkeypatch.setattr(hand_automatic,'pack_scene',lambda hands,outputs,matrix:{})
 rgb=np.zeros((100,100,3),np.uint8)
 for choice,expected in [('both',['left','right']),('right',['right']),('left',['left'])]:
  _,_,report=obj.process(rgb,stabilize=stabilize,hand_selection=choice)
  assert calls[-1]==expected
  assert [h['side'] for h in report['hands']]==expected
  assert report['detection_options']['hand_selection']==choice
  assert report['missing']==[]
  if choice!='both':
   assert report['hands'][0]['raw_side']=='right'
   assert report['hands'][0]['forced_side']==choice
 _,_,report=obj.process(rgb,confidence=.9,stabilize=stabilize,hand_selection='both')
 assert calls[-1]==['right'] and thresholds[-1]==.9
 before=len(calls)
 _,_,report=obj.process(rgb,confidence=.9,stabilize=stabilize,hand_selection='left')
 assert calls[-1]==['left'] and len(calls)==before+1
 assert report['hands'][0]['raw_side']=='right' and report['missing']==[]
 _,_,report=obj.process(rgb,confidence=.95,stabilize=stabilize,hand_selection='left')
 assert report['hands']==[] and len(calls)==before+1
 with pytest.raises(ValueError):obj.process(rgb,hand_selection='invalid')


def test_single_side_tracker_cannot_relabel_to_forbidden_slot():
 tracker=HandSideTracker()
 tracker.update([hand('left',[0,0,40,80])],0)
 result=tracker.update([hand('right',[1,0,41,80])],.1,sides=('right',))
 assert [h['side'] for h in result]==['right']


def test_forced_slot_accepts_flipped_detector_label_without_previous_track():
 tracker=HandSideTracker()
 candidate=dict(hand('left',[0,0,40,80]),side='right',forced_side='right')
 for timestamp in (0,.1,1):
  result=tracker.update([candidate],timestamp,sides=('right',))
  assert len(result)==1 and result[0]['side']=='right'
  assert result[0]['raw_side']=='left' and 'user forced' in result[0]['identity_source']


def test_empty_scene_is_valid_and_clears_previous_geometry():
 data=pack_scene([],[],np.eye(3))
 for key in ('vertices','faces','joints'):assert data[key].shape==(0,3)
 assert data['sides'].shape==(0,)


def make_sequence(root):
 (root/'frames').mkdir();(root/'frames/a.jpg').write_bytes(b'fixture')
 manifest=dict(complete=True,frames=1,profile=dict(width=1280,height=720,fps=30,intrinsics=dict(fx=600,fy=600,cx=640,cy=360)))
 (root/'manifest.json').write_text(json.dumps(manifest))
 (root/'timestamps.jsonl').write_text(json.dumps(dict(file='frames/a.jpg',relative_seconds=0,sequence=0))+'\n')
 return manifest

def test_manifest_camera_and_path_boundary(tmp_path):
 make_sequence(tmp_path);manifest,rows,k=read_sequence(tmp_path)
 assert len(rows)==1;np.testing.assert_array_equal(k,[[600,0,640],[0,600,360],[0,0,1]])
 (tmp_path/'timestamps.jsonl').write_text(json.dumps(dict(file='../outside.jpg',relative_seconds=0,sequence=0))+'\n')
 with pytest.raises(ValueError):read_sequence(tmp_path)

def test_reject_incomplete_and_invalid_stride(tmp_path):
 m=make_sequence(tmp_path);m['complete']=False;(tmp_path/'manifest.json').write_text(json.dumps(m))
 with pytest.raises(ValueError):read_sequence(tmp_path)
 with pytest.raises(ValueError):read_sequence(tmp_path,0)
