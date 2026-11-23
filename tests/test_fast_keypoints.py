# SPDX-License-Identifier: GPL-3.0-only
"""Check compact geometry against locally installed, unchanged upstream MHR."""
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'requires optional PyTorch environment')
class KeypointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from studio_paths import path
        root = path('fast_checkpoint_dir')
        if not (root/'assets/mhr_model.pt').is_file():
            raise unittest.SkipTest('requires locally installed MHR asset')
        from studio.body.fast_keypoints import SelectedMHR70
        torch.set_num_threads(4)
        cls.mhr = torch.jit.load(str(root/'assets/mhr_model.pt'), map_location='cpu').eval()
        state = torch.load(root/'model.ckpt', weights_only=True, mmap=True, map_location='cpu')
        cls.mapping = state.get('state_dict', state)['head_pose.keypoint_mapping']
        cls.compact = SelectedMHR70(cls.mhr, cls.mapping).eval()

    def test_support_and_original_skinning_unchanged(self):
        c = self.compact
        self.assertEqual(c.spec['vertices'], 468)
        self.assertEqual(c.spec['joints'], 50)
        self.assertEqual(c.spec['influences'], 1347)
        self.assertEqual(self.mhr.character_torch.linear_blend_skinning.num_vertices, 18439)
        self.assertEqual(self.mhr.character_torch.linear_blend_skinning.vert_indices_flattened.numel(), 51337)
        support = (self.mapping[:70] != 0).any(dim=0)
        self.assertEqual(c.vertices.tolist(), support[:18439].nonzero().flatten().tolist())
        self.assertEqual(c.joints.tolist(), support[18439:].nonzero().flatten().tolist())

    def test_batch_shape_expression_and_pose_against_full_geometry(self):
        torch.manual_seed(19)
        with torch.inference_mode():
            for batch in (1, 2):
                shape = torch.randn(batch, 45)*.4
                params = torch.randn(batch, 204)*.04
                params[:, :3] *= 50
                expr = torch.randn(batch, 72)*.1
                vertices, skeleton = self.mhr(shape, params, expr, False)
                expected = torch.einsum('kv,bvc->bkc', self.mapping,
                                        torch.cat((vertices, skeleton[..., :3]), dim=1)*.01)[:, :70]
                actual, joints = self.compact(shape, params, expr)
                self.assertEqual(tuple(joints.shape), (batch, 127, 3))
                torch.testing.assert_close(joints, skeleton[..., :3]*.01, atol=0, rtol=0)
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_final_training_gradients_correctives_and_hand_fallback(self):
        from studio.body.fast_keypoints import install_fast_keypoint_path
        sentinel = object()
        calls = []
        def original(*args, **kwargs):
            calls.append(kwargs)
            return sentinel
        head = SimpleNamespace(mhr=self.mhr, keypoint_mapping=self.mapping,
            training=False, enable_hand_model=False, apply_correctives=False,
            _mhr_forward_core=original)
        model = SimpleNamespace(head_pose=head)
        control = install_fast_keypoint_path(model)
        self.assertIs(control, install_fast_keypoint_path(model))
        args = [None]*7
        with torch.inference_mode():
            self.assertIs(head._mhr_forward_core(*args, slim_mode=False), sentinel)
            control['enabled'] = False
            self.assertIs(head._mhr_forward_core(*args, slim_mode=True), sentinel)
            control['enabled'] = True
            for name in ('training', 'enable_hand_model', 'apply_correctives'):
                setattr(head, name, True)
                self.assertIs(head._mhr_forward_core(*args, slim_mode=True), sentinel)
                setattr(head, name, False)
        with torch.enable_grad():
            self.assertIs(head._mhr_forward_core(*args, slim_mode=True), sentinel)
        self.assertEqual(len(calls), 6)

if __name__ == '__main__':
    unittest.main()
