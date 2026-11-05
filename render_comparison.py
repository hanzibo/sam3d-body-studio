# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Generate comparison after the GUI has already received the mesh."""
import json
from pathlib import Path
import sys
import time
import traceback
import cv2
import numpy as np

out = Path(sys.argv[1])
began = time.perf_counter()
report = json.loads((out/'inference.json').read_text())
try:
    from tools.vis_utils import visualize_sample_together
    image = cv2.imread(str(out/'input.png'))
    outputs = []
    for i in range(report['person_count']):
        with np.load(out/f'person_{i:02d}.npz', allow_pickle=False) as data:
            faces = data['faces'].copy()
            outputs.append({k:data[k].copy() for k in data.files if k != 'faces'})
    camera_info = report.get('camera_intrinsics_used')
    if camera_info:
        from calibrated_visualization import visualize_with_intrinsics
        combined = visualize_with_intrinsics(image, outputs, faces, camera_info)
        report['comparison_camera'] = camera_info
    else:
        combined = visualize_sample_together(image, outputs, faces).clip(0,255).astype(np.uint8)
    if not cv2.imwrite(str(out/'reconstruction.jpg'), combined):
        raise OSError('无法保存图像对照')
    preview = cv2.resize(combined, (1800, int(combined.shape[0]*1800/combined.shape[1])))
    if not cv2.imwrite(str(out/'preview.jpg'), preview):
        raise OSError('无法保存预览图')
    report['comparison_status'] = 'complete'
except Exception as exc:
    report['comparison_status'] = 'failed'
    report['comparison_error'] = str(exc)
    traceback.print_exc()
    raise
finally:
    report['comparison_seconds'] = time.perf_counter()-began
    temporary = out/'inference.render.tmp'
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    temporary.replace(out/'inference.json')
