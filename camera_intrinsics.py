"""Validated RGB pinhole intrinsics in original, uncropped image pixels."""
import math
import numpy as np


def resolve_intrinsics(value, width, height):
    if value is None:
        focal = math.hypot(width, height)
        matrix = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float32)
        return None, dict(source='default_diagonal', width=width, height=height, matrix=matrix.tolist())
    if not isinstance(value, dict):
        raise ValueError('camera_intrinsics 必须是包含 width/height/fx/fy/cx/cy 的对象')
    if value.get('width') != width or value.get('height') != height:
        raise ValueError('RGB 内参分辨率与输入照片不一致；请先按图像变换更新内参')
    try:
        fx, fy, cx, cy = [float(value[key]) for key in ('fx', 'fy', 'cx', 'cy')]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('RGB 内参字段缺失或无效') from exc
    if not all(math.isfinite(x) for x in (fx, fy, cx, cy)) or min(fx, fy) <= 0:
        raise ValueError('RGB 焦距必须为有限正数，主点必须为有限数值')
    matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    return matrix, dict(source='provided_rgb', width=width, height=height, matrix=matrix.tolist(),
                        distortion_applied=False)
