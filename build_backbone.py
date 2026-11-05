# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Build only the DINOv3 backbone, keeping upstream code and full MHR unloaded."""
import argparse
import gc
import hashlib
import importlib.util
import os
from pathlib import Path

from studio_paths import path
REPO=path('fast_repo')
OUT=path('backbone_engine_512').parent
os.environ.update(USE_COMPILE='0',USE_COMPILE_BACKBONE='0',USE_TRT_BACKBONE='0')
import torch
from sam_3d_body.models.backbones.dinov3 import Dinov3Backbone
from sam_3d_body.utils.config import get_config

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--export',action='store_true')
p.add_argument('--build',action='store_true')
p.add_argument('--image-size',type=int,choices=[384,512],default=512)
p.add_argument('--output-dir',type=Path,default=OUT)
p.add_argument('--body-only',action='store_true',help='Build batch-1 engine for body-only inference')
a=p.parse_args()
if not a.export and not a.build:p.error('Specify --export and/or --build')
OUT=a.output_dir
batch_sizes=[1] if a.body_only else [1,3,3]
torch.set_num_threads(4)
OUT.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location('upstream_converter',REPO/'convert_backbone_tensorrt.py')
converter=importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)
converter.IMAGE_SIZE=(a.image_size,a.image_size)
converter.OUTPUT_SIZE=(a.image_size//16,a.image_size//16)
converter.TRT_OUTPUT_DIR=str(OUT)
converter.ONNX_PATH=str(OUT/'backbone_dinov3.onnx')
converter.TRT_PATH=str(OUT/'backbone_dinov3_fp16.engine')
if a.export:
    checkpoint=path('fast_checkpoint_dir')/'model.ckpt'
    with checkpoint.open('rb') as f:
        assert hashlib.file_digest(f,'sha256').hexdigest()=='b5a2f9d305dd02626b967aa2e86021fba07065df66ce7a7e00ffb9664f150abf'
    cfg=get_config(str(checkpoint.parent/'model_config.yaml'))
    from offline_dinov3 import local_dinov3_source
    with local_dinov3_source():
        backbone=Dinov3Backbone(name=cfg.MODEL.BACKBONE.TYPE,cfg=cfg).eval()
    state=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    state=state.get('state_dict',state)
    selected={k.removeprefix('backbone.'):v for k,v in state.items() if k.startswith('backbone.')}
    missing,unexpected=backbone.load_state_dict(selected,strict=False)
    if missing!=['encoder.mask_token'] or unexpected:
        raise RuntimeError((missing,unexpected))
    with torch.no_grad():backbone.encoder.mask_token.zero_()
    del selected,state
    gc.collect()
    # Upstream exporter uses legacy dynamic_axes; torch 2.10 defaults to Dynamo.
    original_export=torch.onnx.export
    def legacy_export(*args,**kwargs):
        kwargs['dynamo']=False
        return original_export(*args,**kwargs)
    torch.onnx.export=legacy_export
    try:
        with torch.no_grad():converter.step1_export_onnx(backbone,batch_sizes)
    finally:torch.onnx.export=original_export
    del backbone
    gc.collect()
    torch.cuda.empty_cache()
if a.build:
    if not converter.step2_convert_tensorrt(batch_sizes):
        raise RuntimeError('TensorRT engine build failed')
print('BACKBONE_CONVERSION_OK',flush=True)
