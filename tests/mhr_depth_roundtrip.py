# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Local asset integration: mesh, MHR127, MHR70, bounds and editor roundtrip."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from depth_pose_refinement import MhrSurfaceModel
from depth_pose_experiment import load_arrays, derived_arrays

root = Path(sys.argv[1])
model = MhrSurfaceModel()
for profile in ('standard512', 'fast512', 'fast384'):
    initial = load_arrays(root/profile/'pose-001/translation/person_00.npz')
    model.seed(initial)
    k = dict(fx=608., fy=608., cx=638., cy=361.)
    regenerated = derived_arrays(model, model.base.numpy(), 1., initial['pred_cam_t'], initial, k)
    for key in ('pred_vertices', 'pred_keypoints_3d', 'pred_joint_coords', 'pred_global_rots'):
        np.testing.assert_allclose(regenerated[key], initial[key], atol=2e-6, rtol=0)
    final = load_arrays(root/profile/'pose-001/pose/person_00.npz')
    state, result = model.engine.seed(final)
    assert float(result['import_max_error_mm']) < .01
    assert not state['overrides']
    np.testing.assert_allclose(result['joints'], final['pred_joint_coords'], atol=2e-6, rtol=0)
    # Uniform metric calibration must roundtrip independently of native scale PCs.
    scaled = derived_arrays(model, model.base.numpy(), 1.025, initial['pred_cam_t'], initial, k)
    state, result = model.engine.seed(scaled)
    assert abs(state['metric_scale']-1.025) < 1e-7
    assert float(result['import_max_error_mm']) < .01
    np.testing.assert_array_equal(initial['shape_params'], final['shape_params'])
    np.testing.assert_array_equal(initial['hand_pose_params'], final['hand_pose_params'])
    np.testing.assert_array_equal(initial['scale_params'], final['scale_params'])
    assert 'pred_pose_raw' not in final
print('MHR_DEPTH_ROUNDTRIP_OK: CV skeleton, native rotations, MHR70, immutable shape/hands, metric scale')
