# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Compact offline frame export without PLY, GUI history or comparison subprocesses."""
from pathlib import Path
import cv2
import numpy as np


def emit_video(command,person,faces,seconds,camera_info,emit,bbox):
    output=Path(command['output']);output.mkdir(parents=True,exist_ok=True)
    vertices=np.asarray(person['pred_vertices'])
    if vertices.shape!=(18439,3) or not np.isfinite(vertices).all():raise ValueError('视频网格无效')
    data={k:v for k,v in person.items() if isinstance(v,np.ndarray)}
    # Only native numerical arrays, never object pickles.
    data={k:v for k,v in data.items() if v.dtype.kind in 'biufc'}
    for value in data.values():
        if not np.isfinite(value).all():raise ValueError('视频输出包含非有限值')
    temporary=output/'prediction.partial.npz'
    np.savez_compressed(temporary,**data);temporary.replace(output/'prediction.npz')
    faces_path=Path(command['topology'])
    if not faces_path.exists():np.save(faces_path,np.asarray(faces,dtype=np.int32))
    emit('video_mesh',job=command['job'],path=str(output/'prediction.npz'),inference_seconds=seconds,
         bbox_xyxy=bbox,camera_intrinsics_used=camera_info,mask_used=command.get('mask_prompt') is not None)
