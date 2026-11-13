# SPDX-License-Identifier: GPL-3.0-only
import unittest
import numpy as np
from rgbd_comparison_report import metrics_for,summarize

class MetricTests(unittest.TestCase):
    def test_common_frames_and_own_valid_errors_expose_survivor_bias(self):
        from rgbd_comparison_data import ALL_METHODS
        rows=[]
        for method in ALL_METHODS:
            for frame in (0,1):
                rows.append(dict(method=method,frame=frame,time_s=frame*10.,valid=not(method=='first_init' and frame==1),reference_segmented=True,observed_depth_pixels=100,reference_needs_review=False,mae_mm=10.+frame*90.,point_surface_mae_mm=5.+frame*45.))
        summary=summarize(rows,ALL_METHODS)
        self.assertTrue(all(s['common_quality_frames']==1 for s in summary))
        self.assertEqual(summary[0]['mae_mm'],10.)
        self.assertEqual(summary[0]['own_valid_mae_mm'],55.)
        self.assertEqual(summary[3]['valid_frames'],1)
        self.assertEqual(summary[3]['time_windows'][2]['valid_frames'],0)
    def test_missing_surface_penalty_is_not_hidden_by_overlap_mae(self):
        observed=np.full((4,4),2.,np.float32);mask=np.ones((4,4),bool);rendered=observed.copy();rendered[:2]=0
        verts=np.array([[0,0,2],[1,0,2],[0,1,2]],np.float32);faces=np.array([[0,1,2]]);points=np.array([[.2,.2,2]],np.float32)
        result=metrics_for(observed,np.zeros((4,4,3)),mask,rendered,verts,faces,points)
        self.assertEqual(result['mae_mm'],0.);self.assertAlmostEqual(result['capped_mae_100mm'],50.,places=4);self.assertEqual(result['model_coverage_fraction'],.5)
        self.assertLess(result['point_surface_mae_mm'],1e-4)
    def test_no_overlap_retains_missing_surface_failure(self):
        observed=np.full((4,4),2.,np.float32);mask=np.ones((4,4),bool)
        result=metrics_for(observed,np.zeros((4,4,3)),mask,np.zeros((4,4)),np.zeros((3,3)),np.array([[0,1,2]]),np.empty((0,3)))
        self.assertIsNone(result['mae_mm']);self.assertAlmostEqual(result['capped_mae_100mm'],100.);self.assertEqual(result['model_coverage_fraction'],0.)

if __name__=='__main__':unittest.main()
