# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors, Studio modifications.
# Loader adapted from ComfyUI contributors' SAM3DBody_Loader.execute in
# comfy_extras/nodes_sam3d_body.py, revision
# 858fd58dcd47b1407a6778e0920f02a9b40fd538; upstream GPLv3, see LICENSE.
# Modified: persistent process protocol, strict loading, dtype selection,
# warmup, timing, camera calibration and asynchronous result export.
"""Persistent SAM model. JSON commands on stdin; prefixed events on stdout."""
from gpu_lease import acquire_gpu_lease
import os
_GPU_LEASE = acquire_gpu_lease(os.environ.get('BODYSTUDIO_GPU_KIND','body'))
import gc
import os
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys
import time
import traceback

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
ROOT = Path(__file__).resolve().parent
from launch_config import worker_config
from worker_benchmark import measure
from stream_protocol import read_image, emit_mesh
from camera_intrinsics import resolve_intrinsics
CONFIG=worker_config('standard')
from studio_paths import path
CHECKPOINT = path('standard_checkpoint')


def emit(kind, **values):
    print('@@SAM3D ' + json.dumps(dict(type=kind, backend_id='standard',
          session_id=os.environ.get('BODYSTUDIO_SESSION', ''), config_id=CONFIG.config_id, **values)), flush=True)


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


class Engine:
    def __init__(self):
        began = time.perf_counter()
        with CHECKPOINT.open('rb') as file:
            self.sha = hashlib.file_digest(file, 'sha256').hexdigest()
        sd = comfy.utils.load_torch_file(str(CHECKPOINT), safe_load=True)
        sd = {k.replace('.layers.0.0.', '.layers.0.'): v for k,v in sd.items()}
        self.dtype = comfy.utils.weight_dtype(sd)
        self.model = SAM3DBody(dtype=self.dtype, operations=comfy.ops.manual_cast)
        sd.pop('hand_cls_embed.weight', None)
        sd.pop('hand_cls_embed.bias', None)
        missing, unexpected = self.model.load_state_dict(sd, strict=False, assign=True)
        if missing or unexpected:
            raise RuntimeError(f'Checkpoint mismatch: {missing}, {unexpected}')
        self.model.backbone_dtype = self.dtype
        self.model.eval()
        self.patcher = comfy.model_patcher.CoreModelPatcher(
            self.model, load_device=mm.get_torch_device(), offload_device=mm.unet_offload_device(),
            size=mm.module_size(self.model), fast_disk=comfy.storage.state_dict_fast_disk(sd))
        del sd
        gc.collect()
        from smpl_stage import SmplStage
        self.smpl_stage=SmplStage(CONFIG) if CONFIG.smpl_enabled else None
        from mhr_filter import MhrFilterStage
        self.mhr_stage=MhrFilterStage(CONFIG) if CONFIG.mhr_one_euro_enabled else None
        with torch.inference_mode():
            mm.load_models_gpu([self.patcher], memory_required=self.model.memory_used_forward(1, True))
            image=cv2.imread(str(path('official_repo')/'notebook/images/dancing.jpg'))
            self.warm_times=[]
            layouts=[]
            if CONFIG.warmup_layout in ('sample','both'): layouts.append(('上传示例',image))
            if CONFIG.warmup_layout in ('camera','both'): layouts.append(('相机尺寸',cv2.resize(image,(1280,800))))
            for name,warm in layouts:
                h,w=warm.shape[:2]; oh,ow=image.shape[:2]
                bbox=[732*w/ow,114*h/oh,1523*w/ow,1451*h/oh]
                for index in range(CONFIG.warmup_runs):
                    emit('stage',message=f'正在预热 {name} · {CONFIG.mode} · {index+1}/{CONFIG.warmup_runs}')
                    warm_outputs,elapsed=self.forward(warm,bbox,CONFIG.mode)
                    if self.smpl_stage:self.smpl_stage.process(warm_outputs[0])
                    self.warm_times.append(dict(layout=name,mode=CONFIG.mode,seconds=elapsed))
        if self.smpl_stage:self.smpl_stage.reset('warmup_complete')
        self.load_seconds = time.perf_counter()-began
        self.commit = subprocess.check_output(['git','-C',str(path('comfy_repo')),'rev-parse','HEAD'], text=True).strip()
        emit('ready', actual_config=CONFIG.to_dict(), warmup=self.warm_times, load_seconds=self.load_seconds, allocated_gib=torch.cuda.memory_allocated()/1024**3,smpl_assets=self.smpl_stage.asset_hashes if self.smpl_stage else None)

    @torch.inference_mode()
    def forward(self, image, bbox, mode, cam_int=None, mask=None):
        mm.load_models_gpu([self.patcher], memory_required=self.model.memory_used_forward(1, mode == 'full'))
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        began = time.perf_counter()
        frames = run_batched_frames(self.model,
            [torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).contiguous()],
            [torch.tensor([bbox], dtype=torch.float32)],
            [torch.from_numpy(mask[None, :, :, None])] if mask is not None else None, self.model.image_size,
            mode, crops_per_chunk=1, cam_int=cam_int)
        torch.cuda.synchronize()
        return frames[0], time.perf_counter()-began

    def run(self, command):
        began = time.perf_counter()
        image = read_image(command)
        if image is None:
            raise ValueError('无法读取输入图片')
        h,w = image.shape[:2]
        bbox = command.get('bbox') or [0,0,w,h]
        x1,y1,x2,y2 = bbox
        if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
            raise ValueError('人物框超出图片边界')
        mode = command['inference_type']
        if mode != CONFIG.mode:
            raise ValueError('Unknown inference type')
        matrix, camera_info = resolve_intrinsics(command.get('camera_intrinsics'), w, h)
        cam_int = torch.from_numpy(matrix).unsqueeze(0) if matrix is not None else None
        from mask_prompt import read_prompt
        mask, mask_info = read_prompt(command, (h,w))
        benchmark=None
        if command['type']=='benchmark':
            outputs,seconds,benchmark=measure(command,lambda:self.forward(image,bbox,mode,cam_int,mask),emit)
        else:
            outputs,seconds=self.forward(image,bbox,mode,cam_int,mask)
        if not outputs:
            raise ValueError('未生成人体')
        smpl_info=self.smpl_stage.process(outputs[0],command) if self.smpl_stage else None
        mhr_info=self.mhr_stage.process(outputs[0],command,camera_info) if self.mhr_stage else None
        faces = self.model.head_pose.faces_np().copy()
        if command['type'] == 'video':
            from video_protocol import emit_video
            emit_video(command,outputs[0],faces,seconds,camera_info,emit,bbox);return
        if command['type'] == 'stream':
            emit_mesh(command, outputs[0], faces, seconds, camera_info, began, emit,smpl_info,mhr_info)
            return
        out = Path(command['output'])
        out.mkdir(parents=True, exist_ok=True)
        for i, person in enumerate(outputs):
            for value in person.values():
                if isinstance(value, np.ndarray) and not np.isfinite(value).all():
                    raise ValueError('网格输出有无效数值')
            vertices = person['pred_vertices']
            if vertices.shape != (18439,3) or faces.min() < 0 or faces.max() >= len(vertices):
                raise ValueError('网格拓扑无效')
            np.savez_compressed(out/f'person_{i:02d}.npz', **{k:v for k,v in person.items() if v is not None}, faces=faces)
            trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(out/f'person_{i:02d}.ply')
        report = dict(model_inference_completed=True, implementation='ComfyUI SAM3DBody persistent worker',
            backend_id='standard', session_id=os.environ.get('BODYSTUDIO_SESSION', ''),
            comfy_commit=self.commit, checkpoint=str(CHECKPOINT), checkpoint_sha256=self.sha,
            checkpoint_dtype=str(self.dtype), runtime_dtype=str(self.dtype), image=command['image'],
            mask_prompt=mask_info,
            input_hw=[h,w], bbox_xyxy=bbox, inference_type=mode, person_count=len(outputs),
            load_seconds=self.load_seconds, inference_seconds_single_cold_run=seconds,
            inference_seconds=seconds, worker_seconds_to_mesh=time.perf_counter()-began,
            model_reused=True, comparison_status='pending',smpl_info=smpl_info,mhr_info=mhr_info,
            smpl_assets=self.smpl_stage.asset_hashes if self.smpl_stage else None,
            launch_config=CONFIG.to_dict(), actual_config=CONFIG.to_dict(),config_id=CONFIG.config_id,
            benchmark=benchmark,warmup=self.warm_times,compile_enabled=False,tensorrt_enabled=False,
            image_sha256=hashlib.sha256(Path(command['image']).read_bytes()).hexdigest(),
            peak_cuda_allocated_gib_inference=torch.cuda.max_memory_allocated()/1024**3,
            peak_cuda_reserved_gib_inference=torch.cuda.max_memory_reserved()/1024**3,
            gpu=torch.cuda.get_device_name(), torch=torch.__version__,
            vertex_shapes=[list(p['pred_vertices'].shape) for p in outputs],
            keypoint_shapes=[list(p['pred_keypoints_3d'].shape) for p in outputs], faces_shape=list(faces.shape),
            all_outputs_finite=True, checkpoint_missing_keys=[], checkpoint_unexpected_keys=[], detector=None,
            camera=camera_info['source'], camera_intrinsics_used=camera_info,
            camera_intrinsics_origin=command.get('camera_intrinsics_origin'),
            validation_scope='single-image demo; no accuracy benchmark or physical scale validation')
        (out/'inference.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
        emit('mesh', job=command['job'], output=str(out), report=report)


try:
    engine = Engine()
    for line in sys.stdin:
        command = json.loads(line)
        if command['type'] == 'stop':
            break
        if command['type'] in ('run','benchmark','stream','video'):
            try:
                if command.get('backend_id', 'standard') != 'standard' or command.get('session_id', os.environ.get('BODYSTUDIO_SESSION', '')) != os.environ.get('BODYSTUDIO_SESSION', ''):
                    raise ValueError('后端会话已失效')
                if command.get('config_id') != CONFIG.config_id:
                    raise ValueError('启动配置已失效')
                engine.run(command)
            except Exception as exc:
                if engine.smpl_stage:engine.smpl_stage.reset('frame_error')
                if engine.mhr_stage:engine.mhr_stage.reset('frame_error')
                if command.get('type') != 'stream':
                    traceback.print_exc()
                emit('error', job=command['job'], message=str(exc))
except Exception as exc:
    traceback.print_exc()
    emit('fatal', message=str(exc))
    sys.exit(1)
