# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Independent rotation composition checks for graph-safe MHR features."""
import unittest
try:
    import torch
    from studio.rgbd.depth_pose_accelerated import corrective_features
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch dependencies unavailable')
class CorrectiveFeatureTests(unittest.TestCase):
    def test_rotation_columns_and_derivatives(self):
        torch.manual_seed(61)
        jp = torch.randn(1, 889, dtype=torch.float64, requires_grad=True)
        angles = jp.reshape(1, 127, 7)[:, 2:, 3:6]
        x, y, z = angles.unbind(-1)
        one, zero = torch.ones_like(x), torch.zeros_like(x)
        rx = torch.stack((one, zero, zero, zero, x.cos(), -x.sin(), zero, x.sin(), x.cos()), -1).reshape(1, 125, 3, 3)
        ry = torch.stack((y.cos(), zero, y.sin(), zero, one, zero, -y.sin(), zero, y.cos()), -1).reshape(1, 125, 3, 3)
        rz = torch.stack((z.cos(), -z.sin(), zero, z.sin(), z.cos(), zero, zero, zero, one), -1).reshape(1, 125, 3, 3)
        rotation = rz @ ry @ rx
        expected = torch.cat((rotation[..., 0], rotation[..., 1]), -1)
        expected = expected-torch.tensor([1., 0., 0., 0., 1., 0.])
        actual = corrective_features(jp)
        torch.testing.assert_close(actual, expected.flatten(1))
        weights = torch.randn_like(actual)
        a = torch.autograd.grad((actual*weights).sum(), jp, retain_graph=True)[0]
        b = torch.autograd.grad((expected.flatten(1)*weights).sum(), jp)[0]
        torch.testing.assert_close(a, b)

    def test_identity_pose_has_zero_corrective_features(self):
        torch.testing.assert_close(corrective_features(torch.zeros(1, 889)), torch.zeros(1, 750))


if __name__ == '__main__':
    unittest.main()
