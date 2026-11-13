# SPDX-License-Identifier: GPL-3.0-only
"""INT8 compatibility boundaries and GUI profile persistence, without a GPU."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from launch_config import LaunchConfig, PRESETS, backbone_engine, required_resources
from studio_paths import path

INT8='Fast INT8 512 身体（Embedl）'


class Int8ConfigTests(unittest.TestCase):
    def test_legacy_profiles_default_to_fp16(self):
        legacy=LaunchConfig().to_dict();legacy.pop('backbone_precision')
        self.assertEqual(LaunchConfig.from_dict(legacy),LaunchConfig())
        self.assertNotEqual(PRESETS[INT8].config_id,PRESETS['身体 512 对照'].config_id)

    def test_unsupported_engine_shapes_and_paths_blocked(self):
        base=PRESETS[INT8].validate()
        for updates in [dict(image_size=256),dict(mode='full'),dict(trt_backbone=False),
                        dict(backend='standard'),dict(backbone_precision='int4')]:
            with self.subTest(updates=updates),self.assertRaises(ValueError):replace(base,**updates).validate()
        self.assertEqual(backbone_engine(base),path('backbone_engine_int8_512'))
        self.assertIn(path('backbone_engine_int8_512'),required_resources(base))
        self.assertNotIn(path('backbone_engine_512'),required_resources(base))
        small=replace(base,image_size=384).validate()
        self.assertEqual(backbone_engine(small),path('backbone_engine_int8_384'))
        self.assertNotEqual(backbone_engine(small),backbone_engine(base))

    def test_existing_tuning_and_roundtrip_still_available(self):
        base=PRESETS[INT8]
        for count in (0,1,2,3):
            cfg=replace(base,body_interm_predictions=count).validate()
            self.assertEqual(len(cfg.body_interm_layers),count)
        for name in PRESETS:PRESETS[name].validate()
        self.assertEqual(LaunchConfig.from_dict(json.loads(json.dumps(base.to_dict()))),base)


class Int8PanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from PyQt5 import QtWidgets
        except ImportError:raise unittest.SkipTest('PyQt5 unavailable')
        cls.app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_select_int8_normalizes_only_required_controls(self):
        from gui.profile_panel import ProfilePanel
        panel=ProfilePanel(PRESETS['演示最快'])
        panel.precision.setCurrentIndex(panel.precision.findData('int8'))
        # Existing panel behavior normalizes the inactive SMPL filter target to MHR on edits.
        self.assertEqual(panel.config(),replace(PRESETS[INT8],image_size=384,one_euro_target='mhr'))
        self.assertFalse(panel.hand.isEnabled())
        self.assertTrue(panel.size.isEnabled())
        panel.size.setCurrentIndex(panel.size.findData(512))
        self.assertEqual(panel.config().image_size,512)
        self.assertFalse(panel.backbone.isEnabled())
        self.assertTrue(panel.feedback.isEnabled())
        panel.precision.setCurrentIndex(panel.precision.findData('fp16'))
        self.assertTrue(panel.hand.isEnabled())
        self.assertTrue(panel.size.isEnabled())
        self.assertTrue(panel.backbone.isEnabled())
        panel.close()

    def test_presets_and_saved_config_restore_precision(self):
        from gui.profile_panel import ProfilePanel
        panel=ProfilePanel(PRESETS['标准版身体'])
        self.assertFalse(panel.precision.isEnabled())
        panel.presets.setCurrentIndex(panel.presets.findText(INT8))
        self.assertEqual(panel.config(),PRESETS[INT8])
        with tempfile.TemporaryDirectory() as temp:
            file=Path(temp)/'profile.json';panel.config().save(file)
            panel.set_config(PRESETS['演示最快'])
            panel.set_config(LaunchConfig.from_dict(json.loads(file.read_text())))
        self.assertEqual(panel.config(),PRESETS[INT8])
        self.assertIn('INT8',panel.summary.text())
        self.assertFalse(panel.hand.isChecked())
        panel.close()


if __name__=='__main__':unittest.main()
