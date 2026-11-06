# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""CPU checks for mask binding and foreground preservation."""
import tempfile
import unittest
from pathlib import Path
try:
    import cv2
    import numpy as np
    from mask_prompt import morph_mask, read_prompt, sha256, FixedMaskSegmentor
except ImportError:
    cv2 = None


@unittest.skipIf(cv2 is None, 'requires GUI OpenCV/NumPy dependencies')
class MaskTests(unittest.TestCase):
    def test_morphology(self):
        raw = np.zeros((60, 80), np.uint8); raw[15:45, 20:60] = 1
        self.assertLess(morph_mask(raw, -3).sum(), raw.sum())
        self.assertGreater(morph_mask(raw, 3).sum(), raw.sum())
        np.testing.assert_array_equal(morph_mask(raw, 0), raw)
        self.assertEqual(morph_mask(raw, -40).sum(), 0)
        with self.assertRaises(ValueError): morph_mask(raw, 41)

    def test_image_and_mask_binding(self):
        with tempfile.TemporaryDirectory() as folder:
            image=Path(folder)/'rgb.png'; mask=Path(folder)/'mask.png'
            cv2.imwrite(str(image), np.zeros((20, 30, 3), np.uint8))
            raw=np.zeros((20,30),np.uint8);raw[5:15,10:20]=255;cv2.imwrite(str(mask),raw)
            command=dict(type='run',image=str(image), mask_prompt=dict(path=str(mask),image_sha256=sha256(image),mask_sha256=sha256(mask)))
            actual,info=read_prompt(command,(20,30))
            self.assertTrue(info['enabled']); self.assertEqual(actual.max(),1)
            self.assertEqual(actual.sum(),100)
            with self.assertRaises(ValueError): read_prompt(command,(30,20))
            command['type']='stream'
            with self.assertRaises(ValueError): read_prompt(command,(20,30))
            command['type']='run';image.write_bytes(b'changed')
            with self.assertRaises(ValueError): read_prompt(command,(20,30))

    def test_fast_adapter_preserves_single_person_contract(self):
        adapter=FixedMaskSegmentor(np.ones((20,30),np.uint8))
        masks,scores=adapter.run_sam(np.zeros((20,30,3)),np.zeros((1,4)))
        self.assertTrue(adapter.used);self.assertEqual(masks.shape,(1,20,30,1))
        self.assertEqual(scores.tolist(),[1.0])
        with self.assertRaises(ValueError): adapter.run_sam(np.zeros((20,30,3)),np.zeros((2,4)))
