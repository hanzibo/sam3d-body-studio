# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Orbbec frame decoding for isolated acquisition workers (no GUI imports)."""
import cv2
import numpy as np
from pyorbbecsdk import OBFormat


def color_frame_to_bgr(frame):
    width, height = frame.get_width(), frame.get_height()
    fmt = frame.get_format()
    data = np.asanyarray(frame.get_data(), dtype=np.uint8)
    if fmt == OBFormat.MJPG:
        result = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if result is None:
            raise ValueError('Invalid MJPG color frame')
        return result
    if fmt in (OBFormat.RGB, OBFormat.BGR):
        image = data.reshape(height, width, 3)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if fmt == OBFormat.RGB else image.copy()
    conversions = [(OBFormat.YUYV, cv2.COLOR_YUV2BGR_YUY2, (height,width,2)),
                   (OBFormat.UYVY, cv2.COLOR_YUV2BGR_UYVY, (height,width,2)),
                   (OBFormat.I420, cv2.COLOR_YUV2BGR_I420, (height*3//2,width)),
                   (OBFormat.NV12, cv2.COLOR_YUV2BGR_NV12, (height*3//2,width)),
                   (OBFormat.NV21, cv2.COLOR_YUV2BGR_NV21, (height*3//2,width))]
    for format_id, conversion, shape in conversions:
        if fmt == format_id:
            return cv2.cvtColor(data.reshape(shape), conversion)
    raise ValueError(f'Unsupported color format: {fmt}')
