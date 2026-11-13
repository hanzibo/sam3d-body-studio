# SPDX-License-Identifier: GPL-3.0-only
"""Precision boundaries and immutable original assets; no GPU required."""
from types import SimpleNamespace
import unittest
import torch
from studio_paths import path
from mhr_precision import MixedMHR, HalfMHR, install_mhr_precision
from fast_keypoints import SelectedMHR70


class PrecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        p=path('fast_checkpoint_dir')
        if not (p/'assets/mhr_model.pt').exists():raise unittest.SkipTest('local MHR asset required')
        cls.mhr=torch.jit.load(str(p/'assets/mhr_model.pt'),map_location='cpu').eval()
        state=torch.load(p/'model.ckpt',weights_only=True,map_location='cpu',mmap=True)
        cls.mapping=state.get('state_dict',state)['head_pose.keypoint_mapping']

    def test_reject_unsupported_and_noop_default(self):
        self.assertEqual(install_mhr_precision(None,'fp32')['mode'],'fp32')
        with self.assertRaises(ValueError):install_mhr_precision(None,'int8')
        for options in [dict(apply_correctives=True,enable_hand_model=False),dict(apply_correctives=False,enable_hand_model=True)]:
            with self.assertRaises(ValueError):install_mhr_precision(SimpleNamespace(head_pose=SimpleNamespace(**options)),'all_mixed')

    def inputs(self):
        generator=torch.Generator().manual_seed(17)
        return (torch.randn(1,45,generator=generator)*.2,
                torch.randn(1,204,generator=generator)*.02,
                torch.randn(1,72,generator=generator)*.1)

    def test_mixed_preserves_fk_and_original_buffers(self):
        original=self.mhr.character_torch.linear_blend_skinning
        saved=original.skin_weights_flattened.clone()
        mixed=MixedMHR(self.mhr).eval()
        self.assertEqual(mixed.skinning_half.skin_weights_flattened.dtype,torch.float16)
        self.assertEqual(mixed.skinning_half.skin_indices_flattened.dtype,original.skin_indices_flattened.dtype)
        self.assertEqual(self.mhr.character_torch.blend_shape.base_shape.dtype,torch.float32)
        with torch.inference_mode():
            expected=self.mhr(*self.inputs(),False);actual=mixed(*self.inputs(),False)
        torch.testing.assert_close(expected[1],actual[1],atol=0,rtol=0)
        self.assertEqual(actual[0].dtype,torch.float32)
        self.assertTrue(torch.isfinite(actual[0]).all())
        torch.testing.assert_close(saved,original.skin_weights_flattened,atol=0,rtol=0)

    def test_selected_mixed_preserves_joints_and_output_ownership(self):
        compact=SelectedMHR70(self.mhr,self.mapping).eval()
        head=SimpleNamespace(mhr=self.mhr,_studio_selected_mhr70=compact,apply_correctives=False,enable_hand_model=False)
        with torch.inference_mode():expected=compact(*self.inputs())
        status=install_mhr_precision(SimpleNamespace(head_pose=head),'all_mixed')
        self.assertEqual(status['fk_dtype'],'float32')
        with torch.inference_mode():
            actual=compact(*self.inputs());saved=actual[0].clone()
            compact(*(x+.01 for x in self.inputs()))
        torch.testing.assert_close(expected[1],actual[1],atol=0,rtol=0)
        torch.testing.assert_close(saved,actual[0],atol=0,rtol=0)
        self.assertEqual(actual[0].dtype,torch.float32)

    def test_full_half_keeps_original_and_public_fp32(self):
        half=HalfMHR(self.mhr).eval()
        with torch.inference_mode():out=half(*self.inputs(),False)
        self.assertTrue(all(x.dtype==torch.float32 and torch.isfinite(x).all() for x in out))
        self.assertEqual(self.mhr.character_torch.blend_shape.base_shape.dtype,torch.float32)
        self.assertEqual(half.half_model.character_torch.blend_shape.base_shape.dtype,torch.float16)


if __name__=='__main__':unittest.main()
