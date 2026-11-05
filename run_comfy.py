# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors, Studio modifications.
# Loader adapted from ComfyUI contributors' SAM3DBody_Loader.execute in
# comfy_extras/nodes_sam3d_body.py, revision
# 858fd58dcd47b1407a6778e0920f02a9b40fd538; upstream GPLv3, see LICENSE.
# Modified: command-line single-image workflow, strict loading, timings/export.
"""Single-image SAM 3D Body inference with the ComfyUI adaptation."""
import argparse
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
from studio_paths import path
REPO = path('official_repo')
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--image', type=Path, default=REPO / 'notebook/images/dancing.jpg')
parser.add_argument('--checkpoint', type=Path, default=path('standard_checkpoint'))
parser.add_argument('--output', type=Path, default=ROOT / 'outputs/comfy-full')
parser.add_argument('--bbox', type=float, nargs=4)
parser.add_argument('--inference-type', choices=['body', 'full'], default='full')
a = parser.parse_args()
if a.checkpoint.suffix != '.safetensors':
    parser.error('This runner accepts safetensors only.')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

import cv2
import numpy as np
import torch
import trimesh
import comfy.model_management as mm
import comfy.model_patcher
import comfy.storage
import comfy.ops
import comfy.utils
from comfy.ldm.sam3d_body.model.model import SAM3DBody
from comfy_extras.sam3d_body.utils import run_batched_frames

image = cv2.imread(str(a.image))
if image is None:
    raise FileNotFoundError(a.image)
h, w = image.shape[:2]
bbox = a.bbox or [0, 0, w, h]
x1, y1, x2, y2 = bbox
if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
    raise ValueError('bbox must be inside the input image (xyxy pixels)')
a.output.mkdir(parents=True, exist_ok=True)
with a.checkpoint.open('rb') as checkpoint_file:
    checkpoint_sha256 = hashlib.file_digest(checkpoint_file, 'sha256').hexdigest()
start = time.perf_counter()
logging.info('Loading safetensors: %s', a.checkpoint)
sd = comfy.utils.load_torch_file(str(a.checkpoint), safe_load=True)
sd = {k.replace('.layers.0.0.', '.layers.0.'): v for k, v in sd.items()}
device = mm.get_torch_device()
weight_dtype = comfy.utils.weight_dtype(sd)
# Preserve BF16 backbone and FP32 rig/decoder tensors from this mixed-dtype checkpoint.
dtype = weight_dtype
operations = comfy.ops.manual_cast
model = SAM3DBody(dtype=dtype, operations=operations)
sd.pop('hand_cls_embed.weight', None)
sd.pop('hand_cls_embed.bias', None)
missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
if missing or unexpected:
    raise RuntimeError(f'Checkpoint mismatch: missing={missing}, unexpected={unexpected}')
model.backbone_dtype = dtype
model.eval()
patcher = comfy.model_patcher.CoreModelPatcher(
    model, load_device=device, offload_device=mm.unet_offload_device(),
    size=mm.module_size(model), fast_disk=comfy.storage.state_dict_fast_disk(sd))
del sd
gc.collect()
load_seconds = time.perf_counter() - start
logging.info('All checkpoint keys matched; loading GPU model (%s)', dtype)
with torch.inference_mode():
    mm.load_models_gpu([patcher], memory_required=model.memory_used_forward(1, a.inference_type == 'full'))
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    frames = run_batched_frames(model,
        [torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).contiguous()],
        [torch.tensor([bbox], dtype=torch.float32)], None, model.image_size,
        a.inference_type, crops_per_chunk=1)
    torch.cuda.synchronize()
    infer_seconds = time.perf_counter() - start
    peak_allocated = torch.cuda.max_memory_allocated() / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
    faces = model.head_pose.faces_np().copy()
outputs = frames[0]
if not outputs:
    raise RuntimeError('No person returned')
for i, person in enumerate(outputs):
    for k, v in person.items():
        if isinstance(v, np.ndarray) and not np.isfinite(v).all():
            raise RuntimeError(f'Non-finite output: {k}')
    vertices = person['pred_vertices']
    assert vertices.shape == (18439, 3)
    assert faces.min() >= 0 and faces.max() < len(vertices)
    np.savez_compressed(a.output / f'person_{i:02d}.npz', **{k:v for k,v in person.items() if v is not None}, faces=faces)
    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(a.output / f'person_{i:02d}.ply')
logging.info('Inference succeeded; rendering mesh and keypoints')
mm.unload_all_models()
del patcher, model
gc.collect()
from tools.vis_utils import visualize_sample_together
combined = visualize_sample_together(image, outputs, faces).clip(0, 255).astype(np.uint8)
if not cv2.imwrite(str(a.output / 'reconstruction.jpg'), combined):
    raise RuntimeError('Failed to save reconstruction')
preview = cv2.resize(combined, (1800, int(combined.shape[0] * 1800 / combined.shape[1])))
cv2.imwrite(str(a.output / 'preview.jpg'), preview)
report = {
    'model_inference_completed': True, 'implementation': 'ComfyUI SAM3DBody',
    'comfy_commit': subprocess.check_output(['git','-C',str(path('comfy_repo')),'rev-parse','HEAD'], text=True).strip(),
    'checkpoint': str(a.checkpoint), 'checkpoint_sha256': checkpoint_sha256, 'checkpoint_dtype': str(weight_dtype), 'runtime_dtype': str(dtype),
    'image': str(a.image), 'input_hw': [h,w], 'bbox_xyxy': bbox,
    'inference_type': a.inference_type, 'person_count': len(outputs),
    'load_seconds': load_seconds, 'inference_seconds_single_cold_run': infer_seconds,
    'peak_cuda_allocated_gib_inference': peak_allocated, 'peak_cuda_reserved_gib_inference': peak_reserved,
    'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
    'vertex_shapes': [list(p['pred_vertices'].shape) for p in outputs],
    'keypoint_shapes': [list(p['pred_keypoints_3d'].shape) for p in outputs],
    'faces_shape': list(faces.shape), 'all_outputs_finite': True,
    'checkpoint_missing_keys': missing, 'checkpoint_unexpected_keys': unexpected,
    'detector': None, 'camera': 'default diagonal-focal uncalibrated intrinsics',
    'validation_scope': 'single-image demo; no accuracy benchmark or reference parity validation',
}
(a.output / 'inference.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(report, ensure_ascii=False, indent=2))
