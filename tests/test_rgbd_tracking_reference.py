# SPDX-License-Identifier: GPL-3.0-only
import json
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np
from studio.video.video_project import SCHEMA,segment_folder,segment_key,save_json
from studio.rgbd.rgbd_tracking_reference import export_reviewed_reference,reference_directory
from studio.rgbd.rgbd_tracking_points import SequenceGeometry,project

class ReferenceTests(unittest.TestCase):
    def test_manual_revisions_preserve_previous_masks_and_source_indices(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);segment=dict(start=0,end=0,points=[[8,8,1]])
            project=dict(schema=SCHEMA,frame_count=2,pts=[0,.1],width=16,height=16,source_kind='rgbd_tracking',source_sha256='test',tracks=[dict(id=1,segments=[segment])])
            save_json(root/'project.json',project);save_json(root/'inputs.json',dict(frames=[dict(source_index=0,time_s=0),dict(source_index=3,time_s=.1)]))
            def write_mask():
                folder=segment_folder(root,project,project['tracks'][0],segment);folder.mkdir(parents=True)
                mask=np.zeros((16,16),np.uint8);mask[4:12,4:12]=255;cv2.imwrite(str(folder/'000000.png'),mask)
                save_json(folder/'complete.json',dict(key=segment_key(project,project['tracks'][0],segment)))
            write_mask();first=export_reviewed_reference(root);before=(first/'000000.png').read_bytes()
            meta=json.loads((first/'manifest.json').read_text());self.assertEqual(meta['frames'][1]['source_index'],3);self.assertFalse(meta['frames'][1]['segmented'])
            self.assertEqual(reference_directory(root),first);self.assertEqual(export_reviewed_reference(root),first)
            segment['points'].append([2,2,0]);save_json(root/'project.json',project);write_mask();second=export_reviewed_reference(root)
            self.assertNotEqual(first,second);self.assertEqual(reference_directory(root),second);self.assertEqual((first/'000000.png').read_bytes(),before)

    def test_raw_normals_and_pixel_coordinates_have_metric_provenance(self):
        k=dict(width=32,height=24,fx=40.,fy=40.,cx=16.,cy=12.)
        class Source:
            camera=dict(raw_calibration=dict(color_intrinsics=k,depth_intrinsics=k,color_distortion={},depth_distortion={},depth_to_color_R=np.eye(3).tolist(),depth_to_color_t_m=[.01,0,0]))
            def read(self,i):return np.zeros((24,32,3),np.uint8),np.full((24,32),2000 if i==0 else 0,np.uint16),dict(depth_scale_mm=1.)
        geometry=SequenceGeometry(Source());_,p,n,uv,_=geometry.read(0);valid=p[:,:,2]>0
        self.assertGreater(valid.sum(),10);np.testing.assert_allclose(p[:,:,2][valid],2.)
        np.testing.assert_allclose(uv[valid],project(p[valid],k),atol=1e-5);np.testing.assert_allclose(np.abs(n[valid,2]),1.)
        self.assertEqual(np.count_nonzero(geometry.read(1)[1]),0)

if __name__=='__main__':unittest.main()
