# SPDX-License-Identifier: GPL-3.0-only
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from studio.hand.hand_temporal import HandTemporalFilter,RotationOneEuro,filter_options

OPTIONS=dict(pose=True,position=False,min_cutoff=1.5,beta=.5,d_cutoff=1.)
def inputs(n=1):
    return dict(global_rot=np.zeros((n,3),np.float32),hand=np.zeros((n,108),np.float32),pred_cam_t=np.ones((n,3),np.float32))
def hands(*sides):return [dict(side=s,bbox=[0,0,100,100]) for s in sides]
def test_stationary_jitter_reduces_without_mutation_and_position_is_independent():
    stage=HandTemporalFilter();raw=[];filtered=[]
    for i in range(100):
        a=inputs();a['hand'][:]=.03*(-1)**i;a['global_rot'][:,0]=.03*(-1)**i;a['pred_cam_t'][:,0]=i
        saved={k:v.copy() for k,v in a.items()}
        out,_=stage.parameters(a,hands('right'),i/30,OPTIONS)
        raw.append(a['hand'][0,0]);filtered.append(out['hand'][0,0])
        for key in a:np.testing.assert_array_equal(a[key],saved[key])
        np.testing.assert_array_equal(out['pred_cam_t'],a['pred_cam_t'])
    assert np.std(filtered[20:])<np.std(raw[20:])*.4

def test_so3_short_path_across_wrap_and_valid_rotations():
    f=RotationOneEuro(1.,0.,1.)
    f(Rotation.from_euler('z',179,degrees=True).as_matrix(),0.)
    out=f(Rotation.from_euler('z',-179,degrees=True).as_matrix(),.03)
    assert abs(Rotation.from_matrix(out).as_euler('xyz',degrees=True)[2])>178
    np.testing.assert_allclose(out.T@out,np.eye(3),atol=1e-12)
    assert np.linalg.det(out)==pytest.approx(1.)

@pytest.mark.parametrize('stamp',[0.,-.1,1.])
def test_resets_time_reversal_or_large_gap(stamp):
    stage=HandTemporalFilter();stage.parameters(inputs(),hands('right'),0.,OPTIONS)
    a=inputs();a['hand'][:]=.5
    out,history=stage.parameters(a,hands('right'),stamp,OPTIONS)
    assert history==[1];np.testing.assert_array_equal(out['hand'],a['hand'])

def test_side_slots_missing_and_changed_configuration_reset():
    stage=HandTemporalFilter();a=inputs(2)
    stage.parameters(a,hands('left','right'),0.,OPTIONS)
    a['hand'][:]=1
    out,history=stage.parameters(a,hands('right','left'),.03,OPTIONS)
    assert history==[2,2]
    stage.parameters(inputs(),hands('left'),.06,OPTIONS)
    _,history=stage.parameters(inputs(2),hands('left','right'),.09,OPTIONS)
    assert history==[4,1]
    _,history=stage.parameters(inputs(2),hands('left','right'),.12,dict(OPTIONS,beta=1.))
    assert history==[1,1]

def test_position_only_and_large_box_jump():
    stage=HandTemporalFilter();options=dict(OPTIONS,pose=False,position=True)
    stage.parameters(inputs(),hands('right'),0,options)
    a=inputs();a['pred_cam_t'][:]=2;a['hand'][:]=3
    out,_=stage.parameters(a,hands('right'),.03,options)
    assert (out['pred_cam_t']<2).all();np.testing.assert_array_equal(out['hand'],a['hand'])
    changed=[dict(side='right',bbox=[1000,1000,1100,1100])]
    out,history=stage.parameters(a,changed,.06,options);assert history==[1]
    np.testing.assert_array_equal(out['pred_cam_t'],a['pred_cam_t'])

@pytest.mark.parametrize('options',[dict(min_cutoff=0),dict(beta=-1),dict(d_cutoff=float('nan'))])
def test_bad_options(options):
    with pytest.raises(ValueError):filter_options(options)
