# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Recorder integrity and cross-process cleanup boundaries; model-free."""
import json
from pathlib import Path
import tempfile
import unittest
from recorder_session import RecordingSession, recording_active
from rgbd_batch import PoseSet
from data_catalog import scan, validate_delete


class RecorderSessionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.directory=self.root/'outputs/rgbd-sets/recording-test'
        self.session=RecordingSession.create(self.directory,2);self.addCleanup(self.session.close)

    def sample(self,name='first',skew=1):
        folder=self.directory/'samples'/name;folder.mkdir(parents=True)
        for name in ('rgb.png','rgb_raw.png','depth_raw.png','rgbd.npz'):(folder/name).write_bytes(b'fixture')
        (folder/'camera.json').write_text(json.dumps(dict(schema='studio_rgbd_v1',units='metres',rgb_rectified=True,timestamps=dict(skew_ms=skew),max_timestamp_skew_ms=20)))
        return folder

    def test_attach_and_retake_preserve_versions_and_studio_format(self):
        first=self.sample();second=self.sample('second')
        self.session.attach(0,first);self.session.attach(0,second)
        poses=PoseSet(self.directory)
        self.assertEqual(poses.sample_path(poses.poses[0]),second)
        self.assertEqual(poses.poses[0]['previous_samples'],['samples/first'])
        self.assertTrue(first.is_dir());self.assertEqual(self.session.completed,1)
        self.session.close();resumed=RecordingSession(self.directory)
        self.assertEqual(resumed.completed,1);resumed.close()

    def test_incomplete_or_unsynchronized_samples_do_not_replace_current(self):
        valid=self.sample();self.session.attach(0,valid)
        invalid=self.sample('invalid',skew=40)
        with self.assertRaises(ValueError):self.session.attach(0,invalid)
        (invalid/'rgbd.npz').unlink()
        with self.assertRaises(ValueError):self.session.attach(0,invalid)
        self.assertEqual(PoseSet(self.directory).sample_path(self.session.poses.poses[0]),valid)

    def test_exclusive_session_and_live_deletion_revalidation(self):
        self.assertTrue(recording_active(self.directory))
        with self.assertRaises(ValueError):RecordingSession(self.directory)
        entry=next(e for e in scan(self.root) if e.path==self.directory)
        self.assertFalse(entry.deletable)
        with self.assertRaises(ValueError):validate_delete(self.root,entry)
        self.session.close();self.assertFalse(recording_active(self.directory))
        stale=next(e for e in scan(self.root) if e.path==self.directory)
        self.assertTrue(stale.deletable)
        reopened=RecordingSession(self.directory)
        try:
            with self.assertRaises(ValueError):validate_delete(self.root,stale)
        finally:reopened.close()
        self.assertEqual(validate_delete(self.root,stale),self.directory)

if __name__=='__main__':unittest.main()
