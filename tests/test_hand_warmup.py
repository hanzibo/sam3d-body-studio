# SPDX-License-Identifier: GPL-3.0-only
import json
from types import SimpleNamespace as NS
import cv2
import numpy as np
import pytest
from hand_warmup import camera_variant,load_sample,warmup_local_sample


def sample(tmp_path,height=720):
 directory=tmp_path/'frames';directory.mkdir()
 image=directory/'sample.png'
 rgb=np.zeros((height,1280,3),np.uint8);rgb[:,:,0]=123
 assert cv2.imwrite(str(image),rgb[:,:,::-1])
 intr=dict(fx=608,fy=609,cx=638,cy=height/2+1,width=1280,height=height)
 (tmp_path/'manifest.json').write_text(json.dumps({'profile':{'intrinsics':intr}}))
 return image,rgb


def test_camera_layout_preserves_pixels_and_adjusts_intrinsics(tmp_path):
 filename,expected=sample(tmp_path);rgb,k,source=load_sample(filename)
 np.testing.assert_array_equal(rgb,expected)
 padded,pk=camera_variant(rgb,k)
 assert padded.shape==(800,1280,3)
 np.testing.assert_array_equal(padded[40:760],rgb)
 assert not padded[:40].any() and pk[1,2]==k[1,2]+40
 np.testing.assert_array_equal(pk[[0,1],[0,1]],k[[0,1],[0,1]])


@pytest.mark.parametrize('hands',[1,2])
def test_both_batches_are_warmed_and_tracker_cleared(tmp_path,hands):
 filename,_=sample(tmp_path);calls=[];batches=[];resets=[]
 def process(rgb,**kwargs):
  calls.append((rgb.shape,kwargs))
  return {},[],{'hands':[dict(side='right',bbox=[1,2,30,40]) for _ in range(hands)]}
 automatic=NS(process=process,tracker=NS(reset=lambda:resets.append(True)),
              engine=NS(predict_many=lambda rgb,samples,**kwargs:batches.append(len(samples))),
              detector=NS(metadata={'backend':'test'}))
 report=warmup_local_sample(automatic,filename)
 assert report['status']=='ready' and report['batch_sizes']==[1,2]
 assert len(calls)==6 and len(batches)==6 and set(batches)==({2} if hands==1 else {1})
 assert report['duplicated_crop_for_batch2']==(hands==1)
 assert len(resets)==2 and automatic._hand_stream_id is None
 assert all(not kwargs['stabilize'] for _,kwargs in calls)
 assert not list(tmp_path.rglob('*.npz'))


def test_missing_hand_is_not_reported_as_success(tmp_path):
 filename,_=sample(tmp_path);resets=[]
 automatic=NS(process=lambda *args,**kwargs:({},[],{'hands':[]}),tracker=NS(reset=lambda:resets.append(True)))
 with pytest.raises(ValueError,match='未检测到手'):warmup_local_sample(automatic,filename)
 assert len(resets)==2


def test_bad_calibration_rejected(tmp_path):
 filename,_=sample(tmp_path)
 (tmp_path/'manifest.json').write_text(json.dumps({'profile':{'intrinsics':dict(fx=0,fy=1,cx=0,cy=0,width=1280,height=720)}}))
 with pytest.raises(ValueError,match='内参'):load_sample(filename)
