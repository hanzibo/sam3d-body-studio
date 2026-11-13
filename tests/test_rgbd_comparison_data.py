# SPDX-License-Identifier: GPL-3.0-only
import unittest
import numpy as np
from rgbd_comparison_data import FullGeometry,static_jitter,measured_cloud
from rgbd_geometry import register_depth

class ComparisonTests(unittest.TestCase):
    def test_full_geometry_matches_independent_calibrated_registration(self):
        k=dict(width=32,height=24,fx=42.,fy=41.,cx=16.,cy=12.)
        calibration=dict(color_intrinsics=k,depth_intrinsics=k,depth_to_color_R=np.eye(3).tolist(),depth_to_color_t_m=[.013,-.004,.02],color_distortion={},depth_distortion={})
        raw=np.full((24,32),2000,np.uint16);raw[:,16:]=2300;raw[4:6,5:8]=0
        class Source:
            camera=dict(raw_calibration=calibration)
            def read(self,index):return np.zeros((24,32,3),np.uint8),raw,dict(depth_scale_mm=1.)
        image,actual,_=FullGeometry(Source()).read(0)
        _,expected=register_depth(raw.astype(float)*.001,k,{},k,np.eye(3),calibration['depth_to_color_t_m'])
        np.testing.assert_allclose(actual,expected,atol=3e-7)
        self.assertTrue((actual[:,:,2]==0).any())
    def test_display_downsample_keeps_measured_values_and_reference_flags(self):
        xyz=np.arange(90,dtype=np.float32).reshape(5,6,3)*.001+1;rgb=np.zeros((5,6,3),np.uint8);rgb[:,:,0]=150
        mask=np.indices((5,6))[0]%2==0;r=measured_cloud(xyz,rgb,mask,max_points=7)
        self.assertEqual(len(r['cloud']),7);self.assertTrue(np.all(r['colors'][:,2]==150))
        for p,person in zip(r['cloud'],r['person']):
            ids=np.argwhere((xyz==p).all(2));self.assertEqual(len(ids),1);self.assertEqual(person,mask[tuple(ids[0])])
    def test_static_jitter_does_not_call_linear_motion_noise(self):
        rows=[]
        for i,t in enumerate(np.linspace(.25,1.75,15)):
            rows.append(dict(method='tracking',valid=True,time_s=float(t),root_camera_m=[0,0,2+.01*t],wrist_camera_m=[[.2+t*.01,0,2],[-.2,0,2]]))
        out=static_jitter(rows,'tracking');self.assertLess(out['root_detrended_rms_mm'],1e-9);self.assertLess(out['wrist_detrended_rms_mm'],1e-9)
        self.assertFalse(static_jitter(rows,'cuda60')['available'])

if __name__=='__main__':unittest.main()
