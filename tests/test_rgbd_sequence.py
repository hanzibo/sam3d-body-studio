# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Raw sequence fidelity, timestamp integrity and active deletion protection."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from studio.rgbd.rgbd_sequence import SequenceWriter, RGBDSequence, safe_name, action_plan
from studio.rgbd.recorder_session import recording_active
from studio.core.data_catalog import scan, validate_delete


class SequenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.output=self.root/'outputs/rgbd-sequences/test'
        k=dict(width=4,height=3,fx=5.,fy=5.,cx=1.5,cy=1.)
        self.camera=dict(raw_calibration=dict(color_intrinsics=k,depth_intrinsics=k,
            color_distortion={},depth_distortion={},depth_to_color_R=np.eye(3).tolist(),depth_to_color_t_m=[0,0,0]))
        self.rgb=np.arange(36,dtype=np.uint8).reshape(3,4,3);self.depth=np.full((3,4),2000,np.uint16)
        self.writer=SequenceWriter(self.output,'测试双臂',self.camera,dict(max_skew_ms=20),queue_size=4)
        self.addCleanup(self.writer.close,'stopped')

    def metadata(self,number):
        return dict(color_timestamp_ms=1000+number*100,depth_timestamp_ms=1001+number*100,
            host_monotonic_ns=10**9+number*10**8,relative_time_s=number*.1,skew_ms=1.,depth_scale_mm=1.)

    def test_lossless_roundtrip_timestamps_calibration_and_register(self):
        for i in range(3):self.assertTrue(self.writer.submit(self.rgb,self.depth,self.metadata(i)))
        m=self.writer.close('completed');self.assertEqual(m['saved_frames'],3)
        self.assertAlmostEqual(m['actual_saved_fps'],10.)
        seq=RGBDSequence(self.output);rgb,raw,row=seq.read(1)
        np.testing.assert_array_equal(rgb,self.rgb);np.testing.assert_array_equal(raw,self.depth)
        self.assertEqual(row['color_timestamp_ms'],1100)
        rgb,z,xyz,row=seq.read(0,registered=True)
        self.assertAlmostEqual(z[1,1],2.);self.assertAlmostEqual(xyz[1,1,2],2.)
        self.assertEqual(seq.meta['name'],'测试双臂')

    def test_timestamp_rejection_and_partial_status(self):
        self.writer.submit(self.rgb,self.depth,self.metadata(1))
        with self.assertRaises(ValueError):self.writer.submit(self.rgb,self.depth,self.metadata(1))
        bad=self.metadata(2);bad['skew_ms']=30
        with self.assertRaises(ValueError):self.writer.submit(self.rgb,self.depth,bad)
        self.writer.close('interrupted')
        with self.assertRaises(ValueError):RGBDSequence(self.output)
        self.assertEqual(len(RGBDSequence(self.output,allow_partial=True).frames),1)

    def test_active_deletion_block_and_no_overwrite(self):
        self.assertTrue(recording_active(self.output))
        entry=next(e for e in scan(self.root) if e.path==self.output)
        self.assertEqual(entry.category,'连续 RGB-D 序列');self.assertFalse(entry.deletable)
        with self.assertRaises(ValueError):validate_delete(self.root,entry)
        with self.assertRaises(FileExistsError):SequenceWriter(self.output,'overwrite',{},dict(max_skew_ms=20))
        self.writer.close('stopped');self.assertFalse(recording_active(self.output))

    def test_write_failure_does_not_commit_pair(self):
        import cv2
        with patch.object(cv2,'imwrite',return_value=False):
            self.writer.submit(self.rgb,self.depth,self.metadata(0));self.writer.frames.join()
        m=self.writer.close();self.assertEqual(m['status'],'error');self.assertEqual(m['saved_frames'],0)
        self.assertEqual((self.output/'frames.jsonl').read_text(),'')

    def test_names_and_plan(self):
        self.assertEqual(safe_name('../双臂 / 测试'),'双臂-测试')
        with self.assertRaises(ValueError):safe_name('../')
        for duration in (5,20,300):
            plan=action_plan(duration)
            self.assertEqual(plan[0]['start_s'],0);self.assertEqual(plan[-1]['end_s'],duration)
            for a,b in zip(plan,plan[1:]):self.assertEqual(a['end_s'],b['start_s'])

    def test_native_jpeg_passthrough_without_reencoding(self):
        import cv2
        _,encoded=cv2.imencode('.jpg',self.rgb)
        encoded=encoded.tobytes()
        self.writer.submit(None,self.depth,self.metadata(0),rgb_encoded=encoded)
        self.writer.close()
        seq=RGBDSequence(self.output);row=seq.frames[0]
        self.assertEqual((self.output/row['rgb']).read_bytes(),encoded)
        self.assertEqual(row['rgb_encoding'],'camera_mjpg_passthrough')
        self.assertEqual(seq.read(0)[0].shape,self.rgb.shape)

    def test_bounded_queue_records_drop_without_overwriting(self):
        import cv2
        import threading
        entered=threading.Event();release=threading.Event();original=cv2.imwrite
        def slow_write(*args,**kwargs):
            entered.set();release.wait(3)
            return original(*args,**kwargs)
        with patch.object(cv2,'imwrite',side_effect=slow_write):
            try:
                self.writer.submit(self.rgb,self.depth,self.metadata(0))
                self.assertTrue(entered.wait(3))
                for i in range(1,5):self.assertTrue(self.writer.submit(self.rgb,self.depth,self.metadata(i)))
                self.assertFalse(self.writer.submit(self.rgb,self.depth,self.metadata(5)))
            finally:release.set();self.writer.frames.join()
        m=self.writer.close()
        self.assertEqual(m['dropped_queue'],1);self.assertEqual(m['saved_frames'],5)
        seq=RGBDSequence(self.output);self.assertEqual([r['frame'] for r in seq.frames],list(range(5)))

if __name__=='__main__':unittest.main()
