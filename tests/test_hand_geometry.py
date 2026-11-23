# SPDX-License-Identifier: GPL-3.0-only
import numpy as np
import pytest
from studio.hand.hand_geometry import validate_bbox,mirror_geometry,hand_topology,validate_hand_outputs

@pytest.mark.parametrize('box',[[1,2,1,30],[-1,2,40,60],[0,0,11,50],[0,0,101,50],[0,0,float('nan'),50]])
def test_invalid_hand_roi(box):
    with pytest.raises(ValueError):validate_bbox(box,100,100)

def test_mirror_preserves_projection_and_orientation():
    v=np.array([[.1,.2,.3],[-.2,.1,.4]],np.float32);t=np.array([.2,0,1.],np.float32)
    r=np.array([[[0.,-1,0],[1,0,0],[0,0,1]]]);f=np.array([[0,1,0]])
    mv,mj,mr,mt,mf=mirror_geometry(v,v,r,t,f)
    k=np.array([[600.,0,620],[0,610.,360],[0,0,1.]])
    mirrored_k=k.copy();mirrored_k[0,2]=1279-k[0,2]
    pix=(v+t)@k.T;pix=pix[:,:2]/pix[:,2:]
    new=(mv+mt)@mirrored_k.T;new=new[:,:2]/new[:,2:]
    np.testing.assert_allclose(new[:,0],1279-pix[:,0],atol=1e-4)
    np.testing.assert_allclose(new[:,1],pix[:,1],atol=1e-4)
    np.testing.assert_allclose(np.linalg.det(mr),1,atol=1e-6)
    restored=mirror_geometry(mv,mj,mr,mt,mf)
    for actual,expected in zip(restored,(v,v,r,t,f)):np.testing.assert_allclose(actual,expected)

def test_hand_skin_selection_excludes_forearm():
    ids,faces,joints,edges,wrist=hand_topology(['root','r_wrist','finger'],[-1,0,1],np.array([[0,1,2],[0,2,3]]),np.arange(4),np.array([1,2,2,0]),np.ones(4),4)
    np.testing.assert_array_equal(ids,[0,1,2]);np.testing.assert_array_equal(faces,[[0,1,2]])
    np.testing.assert_array_equal(joints,[1,2]);np.testing.assert_array_equal(edges,[[0,1]]);assert wrist==1


def test_projection_at_camera_plane_is_allowed_only_for_unused_surface():
    raw=dict(pred_vertices=np.array([[0.,0,1],[0,0,-1]],np.float32),pred_cam_t=np.array([0.,0,1],np.float32),pred_keypoints_2d_verts=np.array([[2.,3.],[np.inf,np.nan]],np.float32))
    valid,bad=validate_hand_outputs(raw,np.array([0]))
    assert valid.tolist()==[True,False] and bad.tolist()==[1]
    assert np.isinf(raw['pred_keypoints_2d_verts'][1,0])
    with pytest.raises(ValueError,match='手部顶点投影'):validate_hand_outputs(raw,np.array([1]))
    raw['pred_vertices'][1,2]=-.5
    with pytest.raises(ValueError,match='不在相机平面'):validate_hand_outputs(raw,np.array([0]))
    raw['pred_vertices'][1,2]=np.nan
    with pytest.raises(ValueError,match='pred_vertices'):validate_hand_outputs(raw,np.array([0]))
