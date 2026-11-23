# SPDX-License-Identifier: GPL-3.0-only
import unittest
import numpy as np
from studio.rgbd.rgbd_tracking_points import point_membership,sampled_indices,ProjectiveSurface
from studio.rgbd.rgbd_tracking_report import membership_metrics
from studio.rgbd.depth_pose_refinement import PoseDepthConfig

class TrackingPointTests(unittest.TestCase):
    def test_membership_uses_measurement_uv_and_bounds(self):
        mask=np.zeros((20,20),bool);mask[5:15,5:15]=True
        uv=np.array([[10,10],[1,1],[-1,10],[25,10]])
        np.testing.assert_array_equal(point_membership(uv,mask),[True,False,False,False])
        m=membership_metrics(uv,mask);self.assertEqual(m['inside_fraction'],.25)
        self.assertIsNone(membership_metrics([],mask)['inside_fraction'])

    def test_sampling_matches_real_time_not_every_third_assumption(self):
        rows=[dict(relative_time_s=t) for t in [0,.03,.06,.1,.14,.205,.21,.31]]
        self.assertEqual(sampled_indices(rows,10),[0,3,5,7])

    def test_candidate_does_not_need_mask_and_logs_target_location(self):
        h=w=32;k=dict(fx=30.,fy=30.,cx=16.,cy=16.,height=h,width=w)
        cloud=np.zeros((h,w,3),np.float32);normals=np.zeros_like(cloud);pixels=np.zeros((h,w,2),np.float32)
        cloud[16,17]=[1/30,0,1];normals[16,17]=[0,0,-1];pixels[16,17]=[17,16]
        surface=ProjectiveSurface(cloud,normals,pixels,k,k,lambda v:np.ones((h,w),np.float32),[0,0,31,31],PoseDepthConfig())
        ids,target,n=surface.pairs(np.array([[0.,0.,1.]]))
        np.testing.assert_array_equal(ids,[0]);np.testing.assert_array_equal(surface.batches[0]['pixel_uv'],[[17,16]])
        # Model projection is (16,16); real target pixel is (17,16).
        mask=np.zeros((h,w),bool);mask[16,16]=True
        self.assertFalse(point_membership(surface.batches[0]['pixel_uv'],mask)[0])
        oracle=ProjectiveSurface(cloud,normals,pixels,k,k,surface.render,[0,0,31,31],PoseDepthConfig(),mask)
        self.assertEqual(len(oracle.pairs(np.array([[0.,0.,1.]]))[0]),0)

    def test_no_valid_depth_returns_empty_not_background(self):
        k=dict(fx=20.,fy=20.,cx=5.,cy=5.,height=10,width=10)
        surface=ProjectiveSurface(np.zeros((10,10,3)),np.zeros((10,10,3)),np.zeros((10,10,2)),k,k,lambda v:np.ones((10,10)),[0,0,9,9],PoseDepthConfig())
        self.assertEqual(len(surface.pairs(np.array([[0.,0.,1.]]))[0]),0)

if __name__=='__main__':unittest.main()
