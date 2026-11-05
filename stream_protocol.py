"""Bounded per-frame protocol shared by both resident model workers."""
import base64
from pathlib import Path
import time
import cv2
import numpy as np
from pose_data import extract_pose


def read_image(command):
    if command['type'] == 'stream':
        raw = base64.b64decode(command['jpeg'], validate=True)
        if len(raw) > 8 * 1024 * 1024:
            raise ValueError('视频帧过大')
        return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    return cv2.imread(command['image'])


def emit_mesh(command, person, faces, seconds, camera_info, began, emit, smpl_info=None, mhr_info=None):
    vertices = np.asarray(person['pred_vertices'], dtype=np.float32)
    faces = np.asarray(faces, dtype=np.uint32)
    if vertices.shape != (18439, 3) or not np.isfinite(vertices).all() or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError('视频网格无效')
    # Only one in-flight request; GUI consumes this file before dispatching again.
    path = Path(command['output']) / 'mesh.npz'
    translation = np.asarray(person['pred_cam_t'], dtype=np.float32).reshape(3)
    if not np.isfinite(translation).all():
        raise ValueError('相机平移无效')
    pose = extract_pose(person)
    np.savez(path, pred_vertices=vertices, faces=faces, **pose)
    emit('stream_mesh', job=command['job'], path=str(path), inference_seconds=seconds,
         worker_seconds=time.perf_counter()-began, camera_intrinsics_used=camera_info,
         frame_index=command['frame_index'], smpl_info=smpl_info, mhr_info=mhr_info)
