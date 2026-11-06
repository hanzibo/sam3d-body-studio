# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Geometry checks for observable translation, degeneracy and guard fallback."""
import unittest
try:
    import numpy as np
    from depth_refinement import DepthConfig, solve_translation, refine_translation, unproject, split_pixels
except ImportError:
    np=None


@unittest.skipIf(np is None,'numerical dependencies unavailable')
class TranslationTests(unittest.TestCase):
    def test_curved_translation_with_outliers(self):
        rng=np.random.default_rng(1);normals=rng.normal(size=(1800,3));normals/=np.linalg.norm(normals,axis=1)[:,None]
        target=normals*.5+[0,0,2];delta=np.array([.012,-.017,.026]);source=target-delta
        source[:50]+=.5
        fitted,history=solve_translation(source,target,normals,DepthConfig(iterations=15,prior_strength=.001))
        self.assertLess(np.linalg.norm(fitted-delta),.001)
        self.assertTrue(history)

    def test_plane_cannot_invent_lateral_motion(self):
        x,y=np.meshgrid(np.linspace(-.5,.5,40),np.linspace(-.5,.5,40));target=np.column_stack((x.ravel(),y.ravel(),np.ones(x.size)*2))
        normals=np.tile([0.,0.,1.],(len(target),1));source=target-[.01,.02,.03]
        fitted,history=solve_translation(source,target,normals,DepthConfig())
        self.assertTrue(np.allclose(fitted[:2],0));self.assertAlmostEqual(fitted[2],.03,delta=.002)
        self.assertEqual(history[0]['observable_dimensions'],1)

    def test_missing_depth_preserves_raw(self):
        z=np.zeros((80,100));mask=np.ones(z.shape,bool);k=dict(fx=100,fy=100,cx=50,cy=40)
        raw=np.ones_like(z)*2
        delta,final,report=refine_translation(z,unproject(z,k),mask,raw,k,lambda _:self.fail('must not render'))
        self.assertFalse(report['accepted']);self.assertTrue(np.array_equal(raw,final));self.assertTrue(np.all(delta==0))

    def test_guard_rejects_wrong_render(self):
        z=np.ones((100,120))*2;k=dict(fx=100,fy=100,cx=60,cy=50);raw=z+.03
        delta,final,report=refine_translation(z,unproject(z,k),np.ones(z.shape,bool),raw,k,lambda _:z+.08)
        self.assertFalse(report['accepted']);self.assertEqual(report['reason'],'no_guard_improvement');self.assertTrue(np.all(delta==0))

    def test_plane_refinement_and_input_invariance(self):
        z=np.ones((100,120))*2;k=dict(fx=100,fy=100,cx=60,cy=50);raw=z+.03;xyz=unproject(z,k);original=xyz.copy()
        delta,final,report=refine_translation(z,xyz,np.ones(z.shape,bool),raw,k,lambda d:raw+d[2])
        self.assertTrue(report['accepted']);self.assertLess(np.abs(final-z).mean(),.001)
        self.assertTrue(np.array_equal(xyz,original));self.assertTrue(np.allclose(delta[:2],0))
        guard=split_pixels(z.shape);self.assertTrue(guard.any() and (~guard).any())

if __name__=='__main__':unittest.main()
