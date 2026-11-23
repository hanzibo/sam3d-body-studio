# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Bounded, offline MHR surface fitting. Sensor surfaces are not skeletal truth."""
from dataclasses import asdict, dataclass
import time
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from studio.rgbd.depth_refinement import split_pixels


@dataclass(frozen=True)
class PoseDepthConfig:
    steps: int = 60
    correspond_every: int = 10
    max_points: int = 1200
    max_angle_deg: float = 12.
    max_root_deg: float = 5.
    max_translation_m: float = .05
    max_scale_fraction: float = .05
    min_points: int = 100
    max_correspondence_m: float = .12
    reprojection_sigma_px: float = 12.

    def validate(self):
        for key, low, high in [('steps', 1, 300), ('correspond_every', 1, 30),
                               ('max_points', 100, 5000), ('min_points', 20, 1000)]:
            value = getattr(self, key)
            if type(value) is not int or not low <= value <= high:
                raise ValueError('invalid '+key)
        for key, low, high in [('max_angle_deg', .1, 20), ('max_root_deg', .1, 10),
                               ('max_translation_m', .001, .1), ('max_scale_fraction', .001, .1),
                               ('max_correspondence_m', .005, .2), ('reprojection_sigma_px', 1, 30)]:
            value = getattr(self, key)
            if not np.isfinite(value) or not low <= value <= high:
                raise ValueError('invalid '+key)
        return self


def project(points, k):
    return points[:, :2]/np.maximum(points[:, 2:], .01)*[k['fx'], k['fy']]+[k['cx'], k['cy']]


def delta_bounds(base, limits, radius):
    """Do not increase any pre-existing official-limit violation or snap the seed."""
    low = np.maximum(-radius, np.minimum(limits[:, 0]-base, 0))
    high = np.minimum(radius, np.maximum(limits[:, 1]-base, 0))
    return low, high


def score_depth(observed, mask, rendered, guard):
    fixed = mask & np.isfinite(observed) & (observed > .05) & guard
    if not fixed.any():
        raise ValueError('no valid guard depth')
    model = np.isfinite(rendered) & (rendered > .05)
    error = np.minimum(np.abs(rendered[fixed]-observed[fixed]), .1)
    error[~model[fixed]] = .1
    return dict(capped_mae_mm=float(error.mean()*1000),
                coverage=float((mask & model).sum()/max(1, mask.sum())),
                iou=float((mask & model).sum()/max(1, (mask | model).sum())))


def acceptance(before, after):
    if after['capped_mae_mm'] > before['capped_mae_mm']-.5:
        return 'no_guard_improvement'
    if after['iou'] < before['iou']-.015:
        return 'silhouette_degraded'
    if after['coverage'] < before['coverage']-.02:
        return 'coverage_degraded'
    return 'accepted'


class SurfaceFit:
    """Only fit pixels enter the target tree and normal estimation."""
    def __init__(self, observed, xyz, mask, k, render, config=None):
        self.c = (config or PoseDepthConfig()).validate()
        self.observed, self.mask, self.k, self.render = observed, np.asarray(mask, bool), k, render
        if observed.shape != self.mask.shape or xyz.shape != (*observed.shape, 3):
            raise ValueError('RGB-D dimensions differ')
        self.guard = split_pixels(self.mask.shape)
        interior = cv2.erode(self.mask.astype('uint8'), np.ones((11, 11), np.uint8)) > 0
        valid = np.isfinite(observed) & (observed > .05) & np.isfinite(xyz).all(2)
        self.fit_mask = interior & valid & ~self.guard
        if self.fit_mask.sum() < self.c.min_points:
            raise ValueError('insufficient fitting depth')
        points = xyz[self.fit_mask].astype(float)
        rng = np.random.default_rng(2)
        if len(points) > 10000:
            points = points[rng.choice(len(points), 10000, replace=False)]
        dist, ids = cKDTree(points).query(points, k=12)
        near = points[ids]; near = near-near.mean(1, keepdims=True)
        eig, vec = np.linalg.eigh(np.einsum('nki,nkj->nij', near, near)/12)
        valid = (dist[:, -1] < .05) & (eig[:, 0]/np.maximum(eig.sum(1), 1e-12) < .08)
        self.target, self.normals = points[valid], vec[valid, :, 0]
        if len(self.target) < self.c.min_points:
            raise ValueError('insufficient reliable normals')
        self.tree = cKDTree(self.target)

    def pairs(self, vertices):
        z = self.render(vertices); uv = np.rint(project(vertices, self.k)).astype(int)
        h, w = z.shape
        good = (vertices[:, 2] > .05) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        ids = np.flatnonzero(good); x, y = uv[ids].T
        ids = ids[self.fit_mask[y, x] & (z[y, x] > .05) & (np.abs(vertices[ids, 2]-z[y, x]) < .015)]
        if len(ids) > self.c.max_points:
            ids = ids[np.linspace(0, len(ids)-1, self.c.max_points).astype(int)]
        distance, matches = self.tree.query(vertices[ids])
        good = distance < self.c.max_correspondence_m
        ids, matches = ids[good], matches[good]
        return ids, self.target[matches], self.normals[matches]

    def score(self, vertices):
        z = self.render(vertices)
        return score_depth(self.observed, self.mask, z, self.guard), z

    def fit_scale(self, local, translation, fixed_scale=None):
        """Fit global metric multiplier + XYZ, with zero-centred bounded prior."""
        local = np.asarray(local, dtype=np.float64)
        translation = np.asarray(translation, dtype=np.float64)
        c = self.c; free = fixed_scale is None
        x = np.zeros(4 if free else 3)
        def unpack(v):
            return (1+v[3] if free else fixed_scale), translation+v[:3]
        for _ in range(5):
            scale, t = unpack(x)
            ids, target, normals = self.pairs(local*scale+t)
            if len(ids) < c.min_points:
                break
            def residual(v):
                s, tt = unpack(v); d = local[ids]*s+tt-target
                values = [np.einsum('ij,ij->i', d, normals)/.02,
                          (d*.15/.04).ravel(), v[:3]/.05*np.sqrt(len(ids)*.02)]
                if free:
                    values.append(np.array([v[3]/.03*np.sqrt(len(ids)*.02)]))
                return np.concatenate(values)
            bounds = np.full(len(x), c.max_translation_m)
            if free:
                bounds[3] = c.max_scale_fraction
            fit = least_squares(residual, x, bounds=(-bounds, bounds), loss='huber', f_scale=1., max_nfev=30)
            if np.linalg.norm(fit.x-x) < 1e-5:
                x = fit.x; break
            x = fit.x
        scale, t = unpack(x)
        return float(scale), t, dict(points=len(ids), delta_translation_m=x[:3].tolist(),
                                    scale=float(scale), sufficient=len(ids) >= c.min_points)


class MhrSurfaceModel:
    """Small CPU-only differentiable MHR; neither DINO nor SAM3 is loaded."""
    def __init__(self):
        import torch
        from studio.mhr.mhr_lab_engine import LabEngine
        from studio.core.backends import FAST_CHECKPOINT
        self.engine = LabEngine('cpu')
        state = torch.load(FAST_CHECKPOINT/'model.ckpt', weights_only=True, mmap=True, map_location='cpu')
        state = state.get('state_dict', state)
        self.mapping = state['head_pose.keypoint_mapping'][:70].float().clone()
        self.cv = torch.tensor([1., -1., -1.])
        # Root + trunk + arm direction/elbow + hip direction/knee. Axial limb
        # twists, hands, face and per-bone scales are deliberately frozen.
        self.indices = np.array([3, 4, 5, 7, 9, 11, 13, 15, 17, 31, 32, 34, 35, 36,
                                 41, 42, 44, 45, 46, 51, 52, 53, 60, 61, 62])

    def seed(self, arrays):
        import torch
        self.state, result = self.engine.seed(arrays)
        error = float(result['import_max_error_mm'])
        if error > .1:
            raise ValueError(f'MHR reconstruction mismatch: {error:.4f} mm')
        self.base = torch.tensor(self.state['model'])
        self.identity = torch.tensor(self.state['shape'])[None]
        self.expression = torch.tensor(self.state['expression'])[None]
        with torch.no_grad():
            v, j, keypoints, _ = self.forward(self.base)
        kp_error = float(np.linalg.norm(keypoints.numpy()-arrays['pred_keypoints_3d'], axis=1).max()*1000)
        if kp_error > .1:
            raise ValueError(f'MHR keypoint reconstruction mismatch: {kp_error:.4f} mm')
        return dict(mesh_max_error_mm=error, keypoint_max_error_mm=kp_error)

    def forward(self, params):
        import torch
        v, skel = self.engine.model(self.identity, params[None], self.expression, self.state['correctives'])
        v = v[0]*.01; j = skel[0, :, :3]*.01
        kp = self.mapping @ torch.cat((v, j), dim=0)
        return v*self.cv, j*self.cv, kp*self.cv, skel[0, :, 3:7]

    def fit_pose(self, surface, arrays, scale, translation, body=True):
        import torch
        from torch.nn.functional import huber_loss
        began = time.perf_counter(); c = surface.c
        selected = self.indices if body else self.indices[:3]
        indices = torch.tensor(selected)
        angle = torch.zeros(len(indices), requires_grad=True)
        shift = torch.zeros(3, requires_grad=True)
        optimizer = torch.optim.Adam([angle, shift], lr=.008)
        radius = np.deg2rad(np.where(selected < 6, c.max_root_deg, c.max_angle_deg))
        lo, hi = delta_bounds(self.base[selected].numpy(), self.engine.limits[selected], radius)
        lo, hi = torch.tensor(lo, dtype=torch.float32), torch.tensor(hi, dtype=torch.float32)
        t = torch.tensor(translation, dtype=torch.float32)
        def project_t(p):
            return p[:, :2]/p[:, 2:].clamp(min=.05)*torch.tensor([surface.k['fx'], surface.k['fy']])+torch.tensor([surface.k['cx'], surface.k['cy']])
        with torch.no_grad():
            _, _, kp, _ = self.forward(self.base)
            baseline_uv = project_t(kp*scale+t)
        # Body landmarks only; hands and face should not dominate the objective.
        body_ids = torch.tensor([0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 41, 62, 69])
        history = []
        for step in range(c.steps):
            params = self.base.index_add(0, indices, angle)
            verts, _, kp, _ = self.forward(params)
            camera = verts*scale+t+shift
            if step % c.correspond_every == 0:
                ids, target, normals = surface.pairs(camera.detach().numpy())
                if len(ids) < c.min_points:
                    break
                ids = torch.tensor(ids); target = torch.tensor(target, dtype=torch.float32)
                normals = torch.tensor(normals, dtype=torch.float32)
            d = camera[ids]-target
            plane = (d*normals).sum(1)/.02
            uv = (project_t(kp*scale+t+shift)-baseline_uv)[body_ids]/c.reprojection_sigma_px
            loss = (huber_loss(plane, torch.zeros_like(plane))
                    + .15*huber_loss(d/.04, torch.zeros_like(d))
                    + .5*uv.square().mean() + .05*(angle/.15).square().mean()
                    + .02*(shift/.05).square().mean())
            if not torch.isfinite(loss):
                raise ValueError('non-finite pose objective')
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            with torch.no_grad():
                angle.copy_(torch.maximum(lo, torch.minimum(hi, angle)))
                shift.clamp_(-c.max_translation_m, c.max_translation_m)
            if step % c.correspond_every == 0:
                history.append(dict(step=step, loss=float(loss.detach()), points=len(ids)))
        with torch.no_grad():
            params = self.base.index_add(0, indices, angle)
            v, j, kp, quat = self.forward(params)
        changes = [dict(index=int(i), name=self.engine.names[i], delta_deg=float(np.rad2deg(a)))
                   for i, a in zip(selected, angle.detach().numpy())]
        return params.numpy(), v.numpy(), j.numpy(), kp.numpy(), quat.numpy(), (t+shift).detach().numpy(), dict(
            solver_seconds=time.perf_counter()-began, history=history, changes=changes,
            delta_translation_m=shift.detach().tolist(), config=asdict(c),
            points=len(ids), sufficient=len(ids) >= c.min_points,
            max_keypoint_projection_change_px=float(torch.linalg.norm(project_t(kp*scale+t+shift.detach())-baseline_uv, dim=1)[body_ids].max()))
