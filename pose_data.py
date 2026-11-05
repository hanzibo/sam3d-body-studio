# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""MHR70 body landmarks; geometric angles are not anatomical joint rotations."""
import ast
from pathlib import Path
import numpy as np

# Copy the official names without importing the model or its GPU dependencies.
from studio_paths import path
_SOURCE = path('official_repo')/'sam_3d_body/metadata/mhr70.py'
NAMES = (next(ast.literal_eval(n.value) for n in ast.parse(_SOURCE.read_text()).body
             if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'mhr_names' for t in n.targets))
         if _SOURCE.is_file() else [f'keypoint_{i}' for i in range(70)])
EDGES = np.array([(69,0),(69,5),(69,6),(5,7),(7,62),(6,8),(8,41),
                  (5,9),(6,10),(9,10),(5,6),(9,11),(11,13),(10,12),(12,14),
                  (13,17),(17,15),(13,16),(14,20),(20,18),(14,19)], dtype=np.int32)
LEFT = {5,7,9,11,13,15,16,17,62}
RIGHT = {6,8,10,12,14,18,19,20,41}
COLORS = np.array([(0.27,1,.65,1) if a in LEFT or b in LEFT else
                   (1,.66,.3,1) if a in RIGHT or b in RIGHT else (.85,.9,1,1)
                   for a,b in EDGES], dtype=np.float32)
ANGLE_TRIPLES = {'left_elbow': (5,7,62), 'right_elbow': (6,8,41),
                 'left_knee': (9,11,13), 'right_knee': (10,12,14)}
ANGLE_LABELS = {'left_elbow':'左肘', 'right_elbow':'右肘', 'left_knee':'左膝', 'right_knee':'右膝'}
POSE_SHAPES = {'pred_keypoints_2d':(70,2), 'pred_keypoints_3d':(70,3),
               'pred_joint_coords':(127,3), 'pred_global_rots':(127,3,3), 'pred_cam_t':(3,)}
PARAM_KEYS = ('body_pose_params','hand_pose_params','shape_params','scale_params','global_rot','mhr_model_params')


def extract_pose(data):
    result = {}
    for key, shape in POSE_SHAPES.items():
        array = np.asarray(data[key], dtype=np.float32)
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError('骨架数据无效：'+key)
        result[key] = array.copy()
    for key in PARAM_KEYS:
        if key in data:
            array = np.asarray(data[key], dtype=np.float32)
            if array.ndim != 1 or not np.isfinite(array).all():
                raise ValueError('姿态参数无效：'+key)
            result[key] = array.copy()
    from smpl_stage import SMPL_SHAPES
    for key,shape in SMPL_SHAPES.items():
        if key in data:
            array=np.asarray(data[key],dtype=np.float32)
            if array.shape!=shape or not np.isfinite(array).all():raise ValueError('SMPL 数据无效：'+key)
            result[key]=array.copy()
    for key in ('smpl_postprocess_ms','smpl_smooth_source_frame_index','smpl_smooth_source_monotonic','smpl_smooth_lag_ms','smpl_smooth_history_frames','smpl_one_euro_ms','smpl_one_euro_source_monotonic','smpl_one_euro_source_frame_index','smpl_one_euro_history_frames'):
        if key in data:
            array=np.asarray(data[key])
            if array.shape!=() or not np.isfinite(array).all():raise ValueError('SMPL 时间信息无效：'+key)
            result[key]=array.copy()
    from mhr_filter import MHR_FILTER_SHAPES, MHR_FILTER_SCALARS
    for key,shape in MHR_FILTER_SHAPES.items():
        if key in data:
            array=np.asarray(data[key],dtype=np.float32)
            if array.shape!=shape or not np.isfinite(array).all():raise ValueError('MHR 滤波数据无效：'+key)
            result[key]=array.copy()
    if 'mhr_projection_valid' in data:
        mask=np.asarray(data['mhr_projection_valid'])
        if mask.shape!=(70,) or mask.dtype!=np.bool_:raise ValueError('MHR 投影掩码无效')
        result['mhr_projection_valid']=mask.copy()
    for key in MHR_FILTER_SCALARS:
        if key in data:
            array=np.asarray(data[key])
            if array.shape!=() or not np.isfinite(array).all():raise ValueError('MHR 滤波时间信息无效：'+key)
            result[key]=array.copy()
    return result


def joint_angles(points, triples=ANGLE_TRIPLES):
    points = np.asarray(points)
    result = {}
    for key, (a,b,c) in triples.items():
        u,v = points[a]-points[b], points[c]-points[b]
        denominator = np.linalg.norm(u)*np.linalg.norm(v)
        result[key] = float(np.degrees(np.arccos(np.clip(np.dot(u,v)/denominator,-1,1)))) if denominator>1e-8 else None
    return result


def angle_text(angles):
    return '  ·  '.join(ANGLE_LABELS[k]+' '+(f'{v:.1f}°' if v is not None else '—') for k,v in angles.items())


def schema():
    from smpl_stage import schema as smpl_schema
    return dict(**smpl_schema(),schema_version=1, keypoint_format='mhr70', keypoint_names=NAMES, body_edges=EDGES.tolist(),
        angle_definition='3D three-point included angle in degrees; straight limb about 180; not anatomical flexion or local Euler angles',
        angle_triples=ANGLE_TRIPLES, coordinates='model output x right/y down/z forward, model-scale metres; camera translation stored separately; not calibrated world coordinates',
        joint_format='MHR 127 joint indices, pred_global_rots are global 3x3 rotations, not local joint angles',
        timing='host RGB receipt monotonic and Unix estimate; excludes exposure/USB; variable frame intervals, no interpolation',
        mhr_one_euro='Separate filtered MHR70 and MHR127 positions in model CV frame; raw fields unchanged; 2D reprojected with current raw camera translation and effective K. No rotation or mesh filtering; no fixed lookahead, frequency-dependent phase lag, no exact bone-length constraints. Reset on new stream, non-increasing timestamp, >0.5s gap or frame failure; not identity tracking.',
        scope='single selected person, MHR raw per-frame predictions; optional SMPL raw parameters and separately timestamped smoothed positions; no identity tracking, RGB or depth recording')
