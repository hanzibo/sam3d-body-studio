# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Image-bound mask snapshots shared by GUI and model workers (no CUDA imports)."""
import hashlib
from pathlib import Path
import cv2
import numpy as np


def sha256(file):
    return hashlib.sha256(Path(file).read_bytes()).hexdigest()


def morph_mask(raw, radius):
    if type(radius) is not int or not -40 <= radius <= 40:
        raise ValueError('掩膜半径须为 -40…40 个原图像素')
    mask = (np.asarray(raw) > 0).astype(np.uint8)
    if mask.ndim != 2:
        raise ValueError('掩膜必须是二维')
    if radius:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*abs(radius)+1,)*2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE if radius > 0 else cv2.MORPH_ERODE,
                                kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return mask


def read_prompt(command, hw):
    """Require both immutable image and mask digests; binary [0,1], full RGB size."""
    info = command.get('mask_prompt')
    if info is None:
        return None, dict(enabled=False)
    if command.get('type') == 'stream':
        raise ValueError('静态掩膜不能用于实时流')
    if sha256(command['image']) != info['image_sha256']:
        raise ValueError('掩膜对应的 RGB 已改变')
    if sha256(info['path']) != info['mask_sha256']:
        raise ValueError('掩膜文件已改变')
    raw = cv2.imread(str(info['path']), cv2.IMREAD_GRAYSCALE)
    if raw is None or raw.shape != tuple(hw) or not raw.any():
        raise ValueError('掩膜为空或与 RGB 尺寸不一致')
    if not np.isin(raw, [0, 255]).all():
        raise ValueError('掩膜 PNG 必须为二值 0/255')
    mask = (raw > 0).astype(np.uint8)
    return mask, dict(info, enabled=True, encoding='binary 0/1 before crop',
                     foreground_pixels=int(mask.sum()), rgb_background_unchanged=True)


class FixedMaskSegmentor:
    """Supply an existing mask through the estimator's segmentor API, preserving YOLO."""
    def __init__(self, mask):
        self.mask = mask
        self.used = False

    def run_sam(self, image, boxes):
        if len(boxes) != 1 or image.shape[:2] != self.mask.shape:
            raise ValueError('掩膜提示仅支持当前图像的单个人体')
        self.used = True
        return self.mask[None, :, :, None].copy(), np.ones(1, dtype=np.float32)
