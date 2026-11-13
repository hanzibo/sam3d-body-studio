# SPDX-License-Identifier: GPL-3.0-only
"""Check hand feedback against actual upstream callbacks, with immutable inputs."""
import ast
from types import SimpleNamespace as NS
import pytest
torch=pytest.importorskip('torch')
pytest.importorskip('roma')
from torch.nn import functional as F
from hand_feedback import HandFeedbackCall
from studio_paths import path


@pytest.mark.parametrize('batch',[1,2])
def test_hand_feedback_matches_upstream_and_does_not_cross_calls(batch):
    source=path('official_repo')/'sam_3d_body/models/meta_arch/sam3d_body.py'
    if not source.is_file():pytest.skip('Optional official checkout unavailable')
    cls=next(x for x in ast.parse(source.read_text()).body if isinstance(x,ast.ClassDef) and x.name=='SAM3DBody')
    names=('keypoint_token_update_fn_hand','keypoint3d_token_update_fn_hand')
    methods=[x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name in names]
    scope=dict(torch=torch,F=F)
    exec(compile(ast.Module(body=methods,type_ignores=[]),str(source),'exec'),scope)
    torch.manual_seed(51)
    model=NS(pelvis_idx=[9,10],decoder_hand=NS(layers=[None]*6),cfg=NS(MODEL=NS(BACKBONE=NS(TYPE='dinov3_vith16plus'))))
    for key in ('keypoint_embedding','keypoint3d_embedding'):setattr(model,key+'_hand',NS(weight=torch.zeros(70,16)))
    for key in ('keypoint_embedding_idxs','keypoint3d_embedding_idxs'):setattr(model,key+'_hand',list(range(70)))
    model.keypoint_posemb_linear_hand=torch.nn.Linear(2,16)
    model.keypoint3d_posemb_linear_hand=torch.nn.Linear(3,16)
    model.keypoint_feat_linear_hand=torch.nn.Linear(24,16)
    image=torch.randn(batch,24,8,8);call=HandFeedbackCall(model,image,143)
    pose=dict(pred_keypoints_2d_cropped=torch.rand(batch,70,2)*2-1,pred_keypoints_2d_depth=torch.randn(batch,70),pred_keypoints_3d=torch.randn(batch,70,3))
    with torch.inference_mode():
        for layer in range(5):
            tokens=torch.randn(batch,143,16);augment=torch.randn_like(tokens)
            originals=[tokens,augment,image,*pose.values()];snapshots=[x.clone() for x in originals]
            reference=scope[names[0]](model,3,image,tokens,augment,pose,layer)
            reference=scope[names[1]](model,73,*reference)
            result=call(tokens,augment,pose,layer)
            for x,y in zip(result[:2],reference[:2]):torch.testing.assert_close(x,y,rtol=0,atol=0)
            for x,y in zip(originals,snapshots):torch.testing.assert_close(x,y,rtol=0,atol=0)
        assert (call.computations,call.hits)==(1,4)
        second=HandFeedbackCall(model,image+1,143)
        changed=second(tokens,augment,pose,0)
        assert not torch.equal(changed[0],result[0])
        assert (second.computations,second.hits)==(1,0)
