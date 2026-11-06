# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Deletion boundary regressions using disposable data, never user recordings."""
from pathlib import Path
import tempfile
import unittest

from data_catalog import scan, validate_delete, checked_path


class DataCatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ('outputs/rgbd/sample/rgb.png', 'outputs/rgbd/sample/predictions/run/person.npz',
                     'outputs/rgbd-sets/set/samples/pose-001/rgbd.npz', 'outputs/gui/run/inference.json',
                     'outputs/gui/run-history.json', 'outputs/gui/desktop.log', 'outputs/yolo-config/settings.json',
                     'outputs/studio-model-gpu.lock', 'models/checkpoint.pt', 'config/local.json', 'backups/source.py', 'setup.log'):
            path = self.root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('abc')

    def entry(self, name):
        return next(e for e in scan(self.root) if e.path == self.root / name)

    def test_atomic_groups_and_model_exclusion(self):
        entries = scan(self.root)
        sample = self.entry('outputs/rgbd/sample')
        self.assertEqual((sample.size, sample.files), (6, 2))
        self.assertEqual(self.entry('outputs/rgbd-sets/set').files, 1)
        self.assertFalse(any('models' in e.path.parts or 'config' in e.path.parts for e in entries))
        self.assertFalse(any(e.path.name == 'predictions' for e in entries))
        self.assertEqual(validate_delete(self.root, sample), sample.path)

    def test_protected_indices_backups_and_active_data(self):
        for name in ('outputs/gui/run-history.json', 'outputs/gui/desktop.log', 'outputs/yolo-config', 'backups', 'outputs/studio-model-gpu.lock'):
            with self.assertRaises(ValueError): validate_delete(self.root, self.entry(name))
        sample = self.entry('outputs/rgbd/sample')
        with self.assertRaises(ValueError): validate_delete(self.root, sample, [sample.path/'rgb.png'])
        with self.assertRaises(ValueError): validate_delete(self.root, sample, [sample.path.parent])
        validate_delete(self.root, sample, [self.root/'outputs/rgbd/other'])

    def test_paths_symlinks_and_stale_entries(self):
        with self.assertRaises(ValueError): checked_path(self.root, self.root)
        with self.assertRaises(ValueError): checked_path(self.root, self.root/'../outside')
        (self.root/'outputs/rgbd/link').symlink_to(self.root/'models', target_is_directory=True)
        linked = self.entry('outputs/rgbd/link')
        self.assertFalse(linked.deletable)
        self.assertEqual(linked.size, 0)
        with self.assertRaises(ValueError): checked_path(self.root, linked.path/'checkpoint.pt')
        stale = self.entry('setup.log'); stale.path.unlink()
        with self.assertRaises(ValueError): validate_delete(self.root, stale)

    def test_revalidate_directory_link_swap(self):
        entry = self.entry('outputs/gui/run')
        entry.path.rename(self.root/'moved-run')
        entry.path.symlink_to(self.root/'models', target_is_directory=True)
        with self.assertRaises(ValueError): validate_delete(self.root, entry)
        self.assertTrue((self.root/'models/checkpoint.pt').is_file())

    def test_nested_link_is_not_counted_or_previewed(self):
        sample = self.entry('outputs/rgbd/sample')
        (sample.path/'link').symlink_to(self.root/'models', target_is_directory=True)
        updated = self.entry('outputs/rgbd/sample')
        self.assertEqual(updated.size, sample.size)
        self.assertIn('符号链接', updated.warning)
        with self.assertRaises(ValueError): checked_path(self.root, sample.path/'link/checkpoint.pt')


if __name__ == '__main__': unittest.main()
