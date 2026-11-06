# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Segment correction invalidates only affected identities; GPU lease is cross-process."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import shutil
try:
    import numpy as np
    from video_project import add_track,put_segment,segment_key,run_key,save_json,require_masks,mask_at,segment_folder
    from launch_config import LaunchConfig
    from fast_selection import SinglePersonDetector
except ImportError:
    np=None


@unittest.skipIf(np is None,'requires GUI dependencies')
class VideoTests(unittest.TestCase):
    def project(self):return dict(source_sha256='test',width=100,height=80,frame_count=20,tracks=[])

    def test_correction_and_independent_tracks(self):
        p=self.project();a=add_track(p,'a');b=add_track(p,'b')
        put_segment(p,a['id'],0,19,[[20,20,1]]);put_segment(p,b['id'],3,19,[[30,20,1]])
        bkey=segment_key(p,b,b['segments'][0]);old=run_key(p,LaunchConfig(),0)
        put_segment(p,a['id'],10,19,[[25,20,1]])
        self.assertEqual([(s['start'],s['end']) for s in a['segments']],[(0,9),(10,19)])
        self.assertEqual(bkey,segment_key(p,b,b['segments'][0]));self.assertNotEqual(old,run_key(p,LaunchConfig(),0))
        self.assertNotEqual(run_key(p,LaunchConfig(),0),run_key(p,LaunchConfig(),1))
        with self.assertRaises(ValueError):put_segment(p,a['id'],0,19,[[20,20,0]])
        with self.assertRaises(ValueError):put_segment(p,a['id'],0,20,[[20,20,1]])

    def test_incomplete_masks_never_count_as_complete(self):
        p=self.project();t=add_track(p,'a');put_segment(p,1,2,3,[[10,10,1]])
        with tempfile.TemporaryDirectory() as folder:
            dest=segment_folder(folder,p,t,t['segments'][0]);dest.mkdir(parents=True)
            self.assertIsNone(mask_at(folder,p,t,2))
            save_json(dest/'complete.json',dict(key=segment_key(p,t,t['segments'][0])))
            with self.assertRaises(ValueError):require_masks(folder,p)
            for i in (2,3):(dest/f'{i:06d}.png').write_bytes(b'test')
            self.assertTrue(require_masks(folder,p))
            put_segment(p,1,2,3,[[15,15,1]])
            self.assertIsNone(mask_at(folder,p,t,2))

    def test_detection_cache_assignment_and_ambiguity(self):
        class Detector:
            calls=0
            def run_human_detection(self,image,**kwargs):
                self.calls+=1
                return dict(boxes=np.array([[0,0,40,80],[60,0,100,80]]),keypoints=np.ones((2,17,3)))
        raw=Detector();d=SinglePersonDetector(raw);image=np.zeros((80,100,3));d.video_key='a'
        d.selection_mask=np.zeros((80,100));d.selection_mask[:,0:40]=1
        self.assertEqual(d.run_human_detection(image)['boxes'][0,0],0)
        d.selection_mask[:]=0;d.selection_mask[:,60:100]=1
        self.assertEqual(d.run_human_detection(image)['boxes'][0,0],60);self.assertEqual(raw.calls,1)
        with self.assertRaises(ValueError):d.run_human_detection(image)
        d.video_key='b';d.selection_mask[:]=1
        with self.assertRaises(ValueError):d.run_human_detection(image)
        self.assertEqual(raw.calls,2)
        d.video_key='c';d.selection_mask[:]=0;d.selection_mask[:,:40]=1;d.reserved_boxes=[[0,0,40,80]]
        with self.assertRaises(ValueError):d.run_human_detection(image)

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'requires FFmpeg')
    def test_vfr_frame_pts_and_last_duration_survive_export(self):
        import cv2
        from unittest.mock import patch
        from video_project import annotation_key
        from video_render import render_video
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'frames').mkdir();r=p/'runs/test';r.mkdir(parents=True)
            project=dict(schema='studio_video_v1',source_sha256='test',source='unused',width=100,height=80,
                frame_count=3,pts=[0.,.05,.2],duration=.24,has_audio=False,tracks=[])
            save_json(p/'project.json',project)
            save_json(r/'run.json',dict(status='complete',source_sha256='test',annotations=annotation_key(project),successful=0,records=[]))
            for i in range(3):cv2.imwrite(str(p/'frames'/f'{i:06d}.png'),np.zeros((80,100,3),np.uint8))
            with patch('gpu_lease.acquire_gpu_lease'):
                render_video(p,r,'skeleton',lambda *a,**k:None)
            report=json.loads((r/'render-skeleton/complete.json').read_text())
            self.assertEqual(report['frames'],3)
            self.assertAlmostEqual(float(report['probe']['streams'][0]['duration']),.24,places=5)

    def test_exclusive_video_lease_rejects_body_until_exit(self):
        import gpu_lease
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'isolated.lock'
            prefix=f'import gpu_lease; from pathlib import Path; gpu_lease.LOCK_PATH=Path({str(target)!r}); '
            with patch.object(gpu_lease,'LOCK_PATH',target):
                lease=gpu_lease.acquire_gpu_lease('exclusive')
                try:
                    result=subprocess.run([sys.executable,'-c',prefix+"gpu_lease.acquire_gpu_lease('body')"],capture_output=True,text=True)
                    self.assertNotEqual(result.returncode,0);self.assertIn('显存互斥',result.stderr)
                finally:lease.close()
                result=subprocess.run([sys.executable,'-c',prefix+"gpu_lease.acquire_gpu_lease('sam3')"],capture_output=True)
                self.assertEqual(result.returncode,0)
