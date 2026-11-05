# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Per-image camera selection; independent of the resident model launch profile."""
import json
from pathlib import Path
from camera_intrinsics import resolve_intrinsics

PRESET = Path(__file__).resolve().parent/'config/gemini336l_rgb.json'


def select_camera_intrinsics(enabled, width, height, metadata=None, metadata_error=None):
    if not enabled:
        return None, dict(source='disabled', label='开关已关闭 · 使用默认估计内参')
    if metadata_error:
        raise ValueError('照片 camera.json 无效：'+metadata_error)
    if metadata is not None:
        value = metadata.get('intrinsics')
        if value is None:
            raise ValueError('照片 camera.json 缺少 RGB 内参')
        source = 'photo_metadata'
        label = '照片元数据'
        serial = metadata.get('serial')
    else:
        if not PRESET.is_file():
            return None, dict(source='no_fixed_preset', label='未配置固定相机内参 · 使用默认估计；相机照片优先使用元数据')
        preset = json.loads(PRESET.read_text())
        value = preset['intrinsics']
        if (value['width'], value['height']) != (width, height):
            return None, dict(source='size_not_applicable', label=f'当前 {width}×{height} 不适用固定 Gemini 内参 · 使用默认估计')
        source = 'fixed_gemini336l'
        label = '固定 Gemini 内参（请确认照片来自此相机）'
        serial = preset.get('serial')
    resolve_intrinsics(value, width, height)
    value = dict(value, **{key:float(value[key]) for key in ('fx','fy','cx','cy')})
    return value, dict(source=source, serial=serial,
                       label=f"{label} · {width}×{height} · fx/fy {value['fx']:.1f}/{value['fy']:.1f}")
