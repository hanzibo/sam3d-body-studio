# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""No model packages required. Validate machine config and engine trust boundaries."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

class Portability(unittest.TestCase):
    def invoke(self,source,config):
        return subprocess.run([sys.executable,'-c',source],cwd=ROOT,text=True,capture_output=True,
                              env=dict(os.environ,SAM3D_STUDIO_PATHS=str(config)))

    def test_unconfigured_paths_are_relative_to_checkout(self):
        with tempfile.TemporaryDirectory() as temp:
            result=self.invoke("import studio_paths as s; print(s.path('fast_repo')); print(s.path('fast_checkpoint_dir'))",Path(temp)/'missing.json')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(result.stdout.splitlines(),[str(ROOT/'third_party/Fast-SAM-3D-Body'),str(ROOT/'third_party/Fast-SAM-3D-Body/checkpoints/sam-3d-body-dinov3')])

    def test_custom_root_propagates_to_checkpoints(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg=Path(temp)/'paths.json'; cfg.write_text(json.dumps({'fast_repo':temp}))
            result=self.invoke("import studio_paths as s; print(s.path('yolo_engine'))",cfg)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(result.stdout.strip(),temp+'/checkpoints/yolo/yolo11m-pose.engine')

    def test_config_typo_and_empty_value_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg=Path(temp)/'paths.json'
            for data in ({'fast_reop':'anything'},{'fast_repo':''},[]):
                cfg.write_text(json.dumps(data))
                self.assertNotEqual(self.invoke('import studio_paths',cfg).returncode,0)

    def test_engine_requires_explicit_registration_and_matches_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg=Path(temp)/'paths.json';registry=Path(temp)/'registry.json';engine=Path(temp)/'test.engine'
            cfg.write_text(json.dumps({'engine_registry':str(registry)}));engine.write_bytes(b'local test engine')
            digest=hashlib.sha256(engine.read_bytes()).hexdigest()
            source=f'from engine_registry import verify_engine; verify_engine({str(engine)!r},{digest!r})'
            self.assertNotEqual(self.invoke(source,cfg).returncode,0)
            registry.write_text(json.dumps({str(engine):{'sha256':digest}}))
            self.assertEqual(self.invoke(source,cfg).returncode,0)
            registry.write_text(json.dumps({str(engine):{'sha256':'invalid'}}))
            self.assertNotEqual(self.invoke(source,cfg).returncode,0)

    def test_standard_profile_rejects_fast_features(self):
        from dataclasses import replace
        from launch_config import LaunchConfig
        with self.assertRaises(ValueError):replace(LaunchConfig(),backend='standard').validate()
        base=LaunchConfig(backend='standard',image_size=512,trt_backbone=False,trt_detector=False,
                          decoder_compile=False,mhr_compile=False,interm_compile=False)
        base.validate()
        self.assertEqual(base.config_id,LaunchConfig.from_dict(base.to_dict()).config_id)
        self.assertNotEqual(base.config_id,replace(base,warmup_runs=2).config_id)

    def test_worker_environments_are_isolated(self):
        import studio_paths as s
        for kind in ('standard','fast','camera','gui','render'):
            value=s.environment(kind)
            self.assertNotIn('QT_QPA_PLATFORM_PLUGIN_PATH',value)
            if kind in ('gui','camera'):self.assertNotIn('PYTHONPATH',value)
        self.assertEqual(s.environment('fast')['PYTHONPATH'],str(s.path('fast_repo')))

if __name__=='__main__':unittest.main()
