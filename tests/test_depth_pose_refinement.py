# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Stage guards, bounded calibration and exclusion of guard pixels from fitting."""
import unittest
try:
    import numpy as np
    from depth_pose_refinement import (PoseDepthConfig, SurfaceFit, acceptance,
                                       delta_bounds, score_depth)
    from depth_refinement import unproject, split_pixels
except ImportError:
    np = None


@unittest.skipIf(np is None, 'numerical dependencies unavailable')
class PoseDepthTests(unittest.TestCase):
    def test_scale_and_translation_recovery_with_fixed_matches(self):
        rng = np.random.default_rng(17)
        local = rng.normal(size=(1000, 3))*.35
        normals = local/np.linalg.norm(local, axis=1, keepdims=True)
        translation = np.array([.02, .01, 2.])
        target = local*1.025+translation+[.008, -.004, .012]
        surface = SurfaceFit.__new__(SurfaceFit); surface.c = PoseDepthConfig()
        surface.pairs = lambda _: (np.arange(len(local)), target, normals)
        scale, t, fit = surface.fit_scale(local.astype(np.float32), translation)
        self.assertTrue(fit['sufficient'])
        self.assertAlmostEqual(scale, 1.025, delta=.005)
        self.assertLess(np.linalg.norm(t-translation-[.008, -.004, .012]), .003)

    def test_missing_model_cannot_improve_guard_by_dropping_pixels(self):
        observed = np.ones((80, 100))*2
        mask = np.ones(observed.shape, bool); guard = split_pixels(observed.shape)
        before = score_depth(observed, mask, observed+.03, guard)
        missing = observed.copy(); missing[guard] = 0
        after = score_depth(observed, mask, missing, guard)
        self.assertEqual(acceptance(before, after), 'no_guard_improvement')

    def test_guard_depth_not_used_for_normals_or_correspondence_tree(self):
        z = np.ones((80, 100))*2; k = dict(fx=100, fy=100, cx=50, cy=40)
        mask = np.ones(z.shape, bool)
        first = SurfaceFit(z, unproject(z, k), mask, k, lambda _: z)
        changed = z.copy(); changed[first.guard] += .4
        second = SurfaceFit(changed, unproject(changed, k), mask, k, lambda _: z)
        np.testing.assert_array_equal(first.target, second.target)
        np.testing.assert_array_equal(first.normals, second.normals)

    def test_preexisting_limit_violation_is_not_increased_or_snapped(self):
        low, high = delta_bounds(np.array([2., .5, -.5]), np.array([[0, 1], [0, 1], [0, 1]]), .2)
        np.testing.assert_allclose(low, [-.2, -.2, 0])
        np.testing.assert_allclose(high, [0, .2, .2])

    def test_config_bounds(self):
        for config in [PoseDepthConfig(steps=0), PoseDepthConfig(max_angle_deg=90),
                       PoseDepthConfig(max_scale_fraction=float('nan'))]:
            with self.assertRaises(ValueError):
                config.validate()


if __name__ == '__main__':
    unittest.main()
