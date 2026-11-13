# SPDX-License-Identifier: GPL-3.0-only
"""Schedule parity with upstream and mandatory final prediction coverage."""
import ast
from typing import Optional
from types import SimpleNamespace
import pytest
torch=pytest.importorskip('torch')
from hand_decoder import scheduled_forward
from studio_paths import path


class Layer:
    def __init__(self,index):self.index=index;self.calls=0
    def __call__(self,tokens,image,augment,image_augment,mask):
        self.calls+=1
        return tokens+augment*.1+image.mean()*0.01+self.index,image+0.02


def run(count,upstream=False):
    decoder=SimpleNamespace(layers=[Layer(i) for i in range(6)],norm_final=lambda t:t*.9,do_interm_preds=True,keypoint_token_update=True)
    torch.manual_seed(17)
    tokens=torch.randn(2,7,8);image=torch.randn(2,8,4,4);augment=torch.randn_like(tokens)
    snapshots=[x.clone() for x in (tokens,image,augment)]
    predictions=[];feedback=[]
    def pose(tokens,prev_pose_output,layer_idx):
        predictions.append(layer_idx);return tokens[:,0].clone()
    def update(tokens,augment,pose,index):
        feedback.append(index);return tokens+pose[:,None]*.1,augment+pose[:,None]*.01,pose,index
    kw=dict(token_augment=augment,token_to_pose_output_fn=pose,keypoint_token_update_fn=update)
    if upstream:
        source=path('official_repo')/'sam_3d_body/models/decoders/promptable_decoder.py'
        if not source.is_file():pytest.skip('Optional upstream checkout missing')
        cls=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='PromptableDecoder')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
        scope=dict(torch=torch,Optional=Optional)
        exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),scope)
        result=scope['forward'](decoder,tokens,image,**kw)
    else:result=scheduled_forward(decoder,set(range(count)),tokens,image,**kw)
    for actual,expected in zip((tokens,image,augment),snapshots):torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert all(layer.calls==1 for layer in decoder.layers)
    return result,predictions,feedback


def test_all_predictions_exactly_match_upstream():
    actual,p,f=run(5);expected,p_ref,f_ref=run(5,True)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert p==p_ref==list(range(6));assert f==f_ref==list(range(5))


@pytest.mark.parametrize('count',range(6))
def test_schedule_keeps_all_layers_and_final_prediction(count):
    result,p,f=run(count)
    assert p==list(range(count))+[5]
    assert f==list(range(count))
    assert len(result[1])==count+1
