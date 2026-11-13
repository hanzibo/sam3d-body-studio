# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Opt-in local GPU/assets test: cached MHR, actual deformations and graph reuse.

Run separately from performance collection; requires the local v2 fixtures.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from gpu_lease import acquire_gpu_lease
from depth_pose_refinement import MhrSurfaceModel
from depth_pose_accelerated import CudaPoseFitter, corrective_features
from depth_pose_render import FittingDepthRenderer
from rgbd_evaluate import DepthRenderer
from depth_pose_experiment import load_arrays


def main(root):
    lease = acquire_gpu_lease('exclusive')
    root = Path(root); model = MhrSurfaceModel()
    for profile in ('fast384', 'standard512'):
        arrays = load_arrays(root/profile/'pose-001/rigid/person_00.npz')
        model.seed(arrays)
        jp = torch.randn(1, 889, requires_grad=True)
        a = corrective_features(jp)
        b = model.engine.model.pose_correctives_model._pose_features_from_joint_params(jp)
        torch.testing.assert_close(a, b)
        weight = torch.randn_like(a)
        ga = torch.autograd.grad((a*weight).sum(), jp, retain_graph=True)[0]
        gb = torch.autograd.grad((b*weight).sum(), jp)[0]
        torch.testing.assert_close(ga, gb, rtol=2e-5, atol=2e-5)
        fitter = CudaPoseFitter(model)
        rng = np.random.default_rng(13)
        k = dict(fx=608., fy=608., cx=638., cy=361., width=1280, height=720)
        original = DepthRenderer(arrays['faces'], k)
        fast = FittingDepthRenderer(arrays['faces'], k, original.renderer)
        try:
            for _ in range(4):
                p = model.base.numpy().copy()
                p[model.indices] += rng.uniform(-.04, .04, len(model.indices))
                with torch.no_grad():
                    fitter.base.copy_(torch.tensor(p, device='cuda')); fitter.angle.zero_()
                    _, vertices, _ = fitter.forward()
                    expected = model.forward(torch.tensor(p))[0].numpy()
                error = np.linalg.norm(vertices.cpu().numpy()-expected, axis=1).max()*1000
                assert error < .01, (profile, error)
                camera = expected+arrays['pred_cam_t']
                old = original.render(camera); new = fast.render(camera)
                mask = (old > .05) & (new > .05)
                assert np.max(np.abs(old[mask]-new[mask])) < .0001
                assert np.count_nonzero((old > .05) != (new > .05)) <= 5
                shifted = fast.render(camera+[0, 0, .1])
                assert np.abs(new-shifted).max() > .05, 'vertex buffer was not updated'
            # Restore all optimizer state and parameters, including its step
            # counter; replaying a graph must not retain a previous photograph.
            for _ in range(2):
                fitter.prepare(model, k, 1., arrays['pred_cam_t'], None)
                assert not fitter.angle.any() and not fitter.shift.any()
                assert all(not value.any() for state in fitter.optimizer.state.values()
                           for value in state.values() if isinstance(value, torch.Tensor))
        finally:
            original.close(); fast.close()
        print(profile, 'PARITY_OK', fitter.parity_mm, 'mm', flush=True)
        del fitter
    print('DEPTH_SPEED_INTEGRATION_OK')
    lease.close()


if __name__ == '__main__':
    main(sys.argv[1])
