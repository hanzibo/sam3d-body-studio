# SPDX-License-Identifier: GPL-3.0-only
"""Verify translation removal and camera-space units independently of models."""
import unittest
import numpy as np
from scripts.report_standard_reference import compare_prediction


class ReferenceMetrics(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(42)
        self.reference=dict(pred_vertices=rng.normal(size=(30,3)),pred_joint_coords=rng.normal(size=(127,3)),
                            pred_keypoints_3d=rng.normal(size=(70,3)),pred_cam_t=np.array([0.,0.,3.]))

    def test_self_reference_is_zero(self):
        result=compare_prediction(self.reference,self.reference)
        for key,value in result.items():
            if key=='angles_difference_deg':self.assertTrue(all(v==0 for v in value.values()))
            else:self.assertAlmostEqual(value,0,places=9,msg=key)

    def test_only_translation_removed_no_rigid_fit(self):
        offset=np.array([.03,.04,0.])
        candidate={key:value.copy() for key,value in self.reference.items()}
        for key in ('pred_vertices','pred_joint_coords','pred_keypoints_3d'):candidate[key]+=offset
        result=compare_prediction(self.reference,candidate)
        self.assertAlmostEqual(result['mesh_pelvis_aligned_mean_mm'],0,places=9)
        self.assertAlmostEqual(result['joints127_pelvis_aligned_mean_mm'],0,places=9)
        self.assertAlmostEqual(result['camera_mesh_mean_mm'],50,places=9)
        self.assertAlmostEqual(result['camera_pelvis_difference_mm'],50,places=9)
        candidate['pred_cam_t']-=offset
        result=compare_prediction(self.reference,candidate)
        self.assertAlmostEqual(result['camera_mesh_mean_mm'],0,places=9)
        self.assertAlmostEqual(result['camera_translation_difference_mm'],50,places=9)


if __name__=='__main__':unittest.main()
