# SPDX-License-Identifier: GPL-3.0-only
import base64
from types import SimpleNamespace
import cv2
import numpy as np
import pytest
from studio.hand.hand_stream import process_hand_frame
class Automatic:
 def __init__(self,hands=1):
  self.resets=0;self.calls=[];self.batches=[];self.hands=hands
  self.tracker=SimpleNamespace(reset=self.reset)
  self.engine=SimpleNamespace(predict_many=lambda rgb,samples,**kw:self.batches.append(len(samples)))
 def reset(self):self.resets+=1
 def process(self,rgb,**kw):
  self.calls.append(kw)
  return {'vertices':np.zeros((self.hands,3))},[],{'hands':[{'side':'left'}]*self.hands,'hand_count':self.hands}
def command(tmp_path):
 ok,jpeg=cv2.imencode('.jpg',np.zeros((48,64,3),np.uint8));assert ok
 return dict(jpeg=base64.b64encode(jpeg).decode(),camera_matrix=[[100,0,32],[0,100,24],[0,0,1]],stream_id='a',received_monotonic=1,frame_index=1,output=str(tmp_path))
def test_latest_scene_overwrites_and_stream_resets(tmp_path):
 a=Automatic();cmd=command(tmp_path)
 for i in range(4):cmd['frame_index']=i;result=process_hand_frame(a,cmd)
 assert a.resets==1 and len(list(tmp_path.iterdir()))==1 and result['frame_index']==3
 with np.load(result['path'],allow_pickle=False) as d:assert d['vertices'].shape==(1,3)
 cmd['stream_id']='b';process_hand_frame(a,cmd);assert a.resets==2
@pytest.mark.parametrize('hands,other',[(1,2),(2,1),(0,None)])
def test_warm_both_batches_without_inventing_visible_hands(tmp_path,hands,other):
 a=Automatic(hands);cmd=command(tmp_path);cmd['warmup']=True;r=process_hand_frame(a,cmd)
 assert len(a.calls)==3 and all(not c['stabilize'] for c in a.calls)
 assert a.batches==([] if other is None else [other]*3)
 assert r['report']['hand_count']==hands
@pytest.mark.parametrize('key,value',[('jpeg','invalid'),('camera_matrix',[[float('nan'),0,0],[0,1,0],[0,0,1]]),('camera_matrix',np.zeros((3,3)).tolist())])
def test_reject_bad_frame_and_intrinsics(tmp_path,key,value):
 cmd=command(tmp_path);cmd[key]=value
 with pytest.raises(ValueError):process_hand_frame(Automatic(),cmd)
 assert not list(tmp_path.iterdir())


def test_transient_file_transport_and_path_boundary(tmp_path):
 a=Automatic();cmd=command(tmp_path);raw=base64.b64decode(cmd.pop('jpeg'))
 source=tmp_path/'input.jpg';source.write_bytes(raw);cmd['image']=str(source)
 process_hand_frame(a,cmd);assert len(list(tmp_path.iterdir()))==2
 cmd['image']=str(tmp_path.parent/'elsewhere.jpg')
 with pytest.raises(ValueError):process_hand_frame(a,cmd)


def test_compact_stream_retains_exact_render_arrays_and_sends_runtime_once(tmp_path):
 class RichAutomatic(Automatic):
  def process(self,rgb,**kw):
   data,outputs,report=super().process(rgb,**kw)
   data.update(faces=np.array([[0,1,2]],np.int32),joints=np.zeros((2,3),np.float32),
               edges=np.array([[0,1]],np.int32),camera_matrix=np.eye(3,dtype=np.float32),
               native_debug=np.arange(321,dtype=np.float32))
   report.update(hand_reports=[{'execution':{'backbone':'native'},'backbone_runtime':None,'execution_counts':{'hand':1}}],detector={'backend':'native'},
                 detect_ms=1,reconstruct_ms=2,raw_detections=[{'bbox':[0,0,10,10]}])
   return data,outputs,report
 a=RichAutomatic();cmd=command(tmp_path)
 full=process_hand_frame(a,cmd)
 with np.load(full['path']) as archive:expected={k:archive[k] for k in archive.files}
 cmd['compact_result']=True;compact=process_hand_frame(a,cmd)
 with np.load(compact['path']) as archive:
  assert set(archive.files)=={'vertices','faces','joints','edges','camera_matrix'}
  for key in archive.files:np.testing.assert_array_equal(archive[key],expected[key])
 assert compact['report']['hand_reports']==full['report']['hand_reports']
 assert compact['report']['frame_execution']['execution_counts']=={'hand':1}
 assert 'raw_detections' not in compact['report']
 assert 'hand_reports' not in process_hand_frame(a,cmd)['report']
 cmd['stream_id']='new';assert 'hand_reports' in process_hand_frame(a,cmd)['report']
