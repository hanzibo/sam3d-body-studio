# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Studio backend resources; inference packages stay in isolated processes."""
from pathlib import Path
from studio_paths import path

ROOT = Path(__file__).resolve().parent
FAST_REPO = path('fast_repo')
FAST_CHECKPOINT = path('fast_checkpoint_dir')
BACKBONE_ENGINE = path('backbone_engine_512')
YOLO_ENGINE = path('yolo_engine')
BACKENDS = {
    'standard': dict(label='标准版', launcher=ROOT/'launch_model.sh',
        python=path('standard_python'),
        resources=[path('standard_checkpoint')]),
    'fast': dict(label='Fast', launcher=ROOT/'launch_fast_model.sh',
        python=path('fast_python'),
        resources=[FAST_CHECKPOINT/'model.ckpt', FAST_CHECKPOINT/'model_config.yaml',
                   FAST_CHECKPOINT/'assets/mhr_model.pt', path('yolo_checkpoint'),
                   FAST_REPO/'notebook/images/dancing.jpg', BACKBONE_ENGINE, YOLO_ENGINE]),
}

def missing_resources(backend, config=None):
    if config is not None:
        from launch_config import required_resources
        return [str(p) for p in required_resources(config) if not p.is_file()]
    spec = BACKENDS[backend]
    return [str(p) for p in [spec['python'], spec['launcher'], *spec['resources']] if not p.is_file()]
