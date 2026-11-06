# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Preview cache correctness under file replacement and memory bounds."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'gui'))
try:
    import numpy as np
    from display_cache import DisplayCache
except ImportError:np=None


@unittest.skipIf(np is None,'NumPy unavailable')
class DisplayCacheTests(unittest.TestCase):
    def test_hash_cache_detects_replacement_with_same_size_and_mtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/'input';file.write_bytes(b'old');cache=DisplayCache();original=cache.digest(file)
            with patch.object(Path,'read_bytes',side_effect=AssertionError('unchanged file reread')):
                self.assertEqual(cache.digest(file),original)
            stamp=file.stat().st_mtime_ns;replacement=Path(tmp)/'new';replacement.write_bytes(b'new');os.utime(replacement,ns=(stamp,stamp));replacement.replace(file)
            self.assertNotEqual(cache.digest(file),original)

    def test_prediction_cache_is_immutable_and_invalidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/'person.npz';np.savez(file,pred_vertices=np.zeros((3,3)),pred_cam_t=np.zeros(3),faces=np.array([[0,1,2]]))
            cache=DisplayCache();key,a=cache.prediction(file)
            with patch.object(np,'load',side_effect=AssertionError('unchanged prediction reloaded')):
                self.assertIs(cache.prediction(file)[1],a)
            with self.assertRaises(ValueError):a['pred_vertices'][0,0]=1
            replacement=Path(tmp)/'new.npz';np.savez(replacement,pred_vertices=np.ones((3,3)),pred_cam_t=np.zeros(3),faces=np.array([[0,1,2]]));replacement.replace(file)
            new,b=cache.prediction(file);self.assertNotEqual(key,new);self.assertTrue(np.all(b['pred_vertices']==1))

    def test_bounds_and_clear(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache=DisplayCache()
            for i in range(70):
                file=Path(tmp)/f'{i}.npz';np.savez(file,pred_vertices=np.zeros((3,3)));cache.digest(file);cache.prediction(file)
            self.assertEqual(len(cache.hashes),64);self.assertEqual(len(cache.predictions),6)
            cache.clear();self.assertFalse(cache.hashes or cache.predictions)

if __name__=='__main__':unittest.main()
