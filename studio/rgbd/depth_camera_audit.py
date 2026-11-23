# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Check frozen camera/crop metadata and test the actual Fast projection method."""
import argparse
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from studio_paths import path


def audit(source):
    import torch
    camera_source = path('fast_repo')/'sam_3d_body/models/heads/camera_head.py'
    tree = ast.parse(camera_source.read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'perspective_projection')
    method.name = 'camera_projection_under_test'
    def project(points, k):
        p = points @ k.transpose(1, 2)
        return p[..., :2]/p[..., 2:]
    namespace = dict(torch=torch, perspective_projection=project)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(camera_source), 'exec'), namespace)
    fn = namespace['camera_projection_under_test']
    common = dict(points_3d=torch.tensor([[[.1, -.2, 0.]]]), pred_cam=torch.tensor([[-.8, .1, -.1]]),
                  bbox_center=torch.tensor([[630., 365.]]), bbox_size=torch.tensor([500.]),
                  img_size=torch.tensor([[1280., 720.]]), cam_int=torch.tensor([[[608., 0, 638.], [0, 608., 361.], [0, 0, 1.]]]))
    outputs = [fn(SimpleNamespace(img_size=(size, size), default_scale_factor=1.), **common) for size in (384, 512)]
    delta = float(torch.max(torch.abs(outputs[0]['pred_cam_t']-outputs[1]['pred_cam_t'])))
    records = []
    for folder in sorted((source/'fast384').glob('pose-*')):
        pose = folder.name
        small = json.loads((folder/'prediction/inference.json').read_text())
        large = json.loads((source/'fast512'/pose/'prediction/inference.json').read_text())
        with np.load(folder/'prediction/person_00.npz') as a, np.load(source/'fast512'/pose/'prediction/person_00.npz') as b:
            translation_delta = a['pred_cam_t']-b['pred_cam_t']
        records.append(dict(pose=pose, same_detected_bbox=small['bbox_xyxy'] == large['bbox_xyxy'],
                            same_intrinsics=small['camera_intrinsics_used'] == large['camera_intrinsics_used'],
                            fast384_minus_fast512_translation_mm=(translation_delta*1000).tolist()))
    return dict(projection_source=str(camera_source), same_predicted_camera_different_network_size_max_translation_delta_m=delta,
                all_recorded_boxes_equal=all(r['same_detected_bbox'] for r in records),
                all_recorded_intrinsics_equal=all(r['same_intrinsics'] for r in records), records=records,
                conclusion='No direct 384/512 multiplier in the tested projection method; frozen Fast crops and K checked separately. This does not prove all upstream feature/crop code or training-resolution behaviour is correct.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--source', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); result = audit(Path(args.source))
    Path(args.output).write_text(json.dumps(result, indent=2)); print(json.dumps(result, indent=2))
