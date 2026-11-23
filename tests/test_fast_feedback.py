# SPDX-License-Identifier: GPL-3.0-only
"""Compare against actual upstream methods without loading its model or weights."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
try:
    import torch
    from torch.nn import functional as F
except ImportError:
    torch = None
from studio.body.fast_feedback import FeedbackCall, install_fast_feedback_cache


@unittest.skipIf(torch is None, 'requires PyTorch')
class FeedbackTests(unittest.TestCase):
    def setUp(self):
        from studio_paths import path
        source = path('fast_repo')/'sam_3d_body/models/meta_arch/sam3d_body.py'
        if not source.is_file():
            self.skipTest('optional upstream checkout is unavailable')
        tree = ast.parse(source.read_text())
        cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'SAM3DBody')
        names = ('keypoint_token_update_fn', 'keypoint3d_token_update_fn')
        methods = [x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name in names]
        scope = dict(torch=torch, F=F)
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), 'exec'), scope)
        self.original2 = scope[names[0]]
        self.original3 = scope[names[1]]
        torch.manual_seed(42)
        m = NS(keypoint_embedding_idxs=list(range(70)), keypoint3d_embedding_idxs=list(range(70)),
               keypoint_embedding=NS(weight=torch.zeros(70,16)),
               keypoint3d_embedding=NS(weight=torch.zeros(70,16)),pelvis_idx=[0,1],
               keypoint_posemb_linear=torch.nn.Sequential(torch.nn.Linear(2,16),torch.nn.ReLU(),torch.nn.Linear(16,16)),
               keypoint_feat_linear=torch.nn.Linear(24,16),
               keypoint3d_posemb_linear=torch.nn.Sequential(torch.nn.Linear(3,16),torch.nn.ReLU(),torch.nn.Linear(16,16)),
               decoder=NS(layers=[None]*6),cfg=NS(MODEL=NS(BACKBONE=NS(TYPE='dinov3_vith16plus'))))
        self.m=m

    def pose(self):
        p=dict(pred_keypoints_2d_cropped=torch.rand(1,70,2)*1.5-.75,
               pred_keypoints_2d_depth=torch.randn(1,70),pred_keypoints_3d=torch.randn(1,70,3))
        p['pred_keypoints_2d_cropped'][0,:3]=torch.tensor([[-.5,-.5],[.5,.5],[0,0]])
        p['pred_keypoints_2d_depth'][0,:3]=1
        return p

    def test_exact_values_cache_reuse_and_input_ownership(self):
        m=self.m;image=torch.randn(1,24,8,8)
        call=FeedbackCall(m,image,145);p0,p1,p2=self.pose(),self.pose(),self.pose()
        with torch.inference_mode():
            for layer,p in enumerate([p0,p1,p2,p2,p2]):
                tokens=torch.randn(1,145,16);augment=torch.randn_like(tokens)
                inputs=[tokens,augment,image,*p.values()];snapshots=[v.clone() for v in inputs]
                ref=self.original2(m,5,image,tokens,augment,p,layer)
                ref=self.original3(m,75,*ref)
                got=call(tokens,augment,p,layer)
                for i in (0,1):torch.testing.assert_close(got[i],ref[i],rtol=0,atol=0)
                for v,s in zip(inputs,snapshots):torch.testing.assert_close(v,s,rtol=0,atol=0)
            self.assertEqual((call.computations,call.hits),(3,2))
            # A new image/call must recompute even if the pose object is reused.
            second=FeedbackCall(m,image+1,145)
            different=second(tokens,augment,p2,0)
            self.assertEqual((second.computations,second.hits),(1,0))
            self.assertFalse(torch.equal(different[0],got[0]))

    def test_adapter_scopes_each_call_and_falls_back_with_gradients(self):
        m=self.m;m.training=False
        m.cfg.MODEL.DECODER={'DO_KEYPOINT_TOKENS':True,'DO_KEYPOINT3D_TOKENS':True}
        p=self.pose()
        def original(tokens,image,augment,keypoint_token_update_fn=None,**kwargs):
            for i in range(5):tokens,augment,_,_=keypoint_token_update_fn(tokens,augment,p,i)
            return tokens,augment
        m.decoder.forward=original
        controller=install_fast_feedback_cache(m)
        self.assertIs(controller,install_fast_feedback_cache(m))
        tokens=torch.randn(1,145,16);aug=torch.zeros_like(tokens);image=torch.randn(1,24,8,8)
        def sentinel(*args):raise RuntimeError('original callback')
        with torch.inference_mode():
            m.decoder.forward(tokens,image,aug,keypoint_token_update_fn=sentinel)
            self.assertEqual(controller['last'],{'computations':1,'hits':4})
            m.decoder.forward(tokens,image+1,aug,keypoint_token_update_fn=sentinel)
            self.assertEqual(controller['last'],{'computations':1,'hits':4})
        with self.assertRaisesRegex(RuntimeError,'original callback'):
            m.decoder.forward(tokens,image,aug,keypoint_token_update_fn=sentinel)
        self.assertIsNone(controller['last'])

    def test_batch_extra_prompts_and_alternate_crop_coordinates(self):
        m=self.m
        m.cfg.MODEL.BACKBONE.TYPE='vit_hmr'
        image=torch.randn(2,24,8,8)
        p0,p1=self.pose(),self.pose()
        pose={k:torch.cat([p0[k],p1[k]],dim=0) for k in p0}
        # Two extra prompt rows shift both keypoint blocks by two rows.
        tokens=torch.randn(2,147,16);augment=torch.randn_like(tokens)
        with torch.inference_mode():
            ref=self.original2(m,7,image,tokens,augment,pose,0)
            ref=self.original3(m,77,*ref)
            got=FeedbackCall(m,image,147)(tokens,augment,pose,0)
            for i in (0,1):torch.testing.assert_close(got[i],ref[i],rtol=0,atol=0)

if __name__=='__main__':unittest.main()
