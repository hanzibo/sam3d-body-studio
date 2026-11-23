# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Reusable CUDA-graph MHR fitting; same loss, bounds and correctives as v2.

Graph construction is cold setup, never silently charged as warm-frame time.
The caller must hold the Studio GPU lease. No network or SAM3 is loaded here.
"""
import time
import numpy as np
import torch
from dataclasses import asdict
from studio.rgbd.depth_pose_refinement import delta_bounds, MhrSurfaceModel

BODY_IDS = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 41, 62, 69]


def corrective_features(jp):
    """MHR's first two XYZ rotation columns, without a CPU temporary."""
    angles = jp.reshape(1, 127, 7)[:, 2:, 3:6]
    cx, cy, cz = angles.cos().unbind(-1)
    sx, sy, sz = angles.sin().unbind(-1)
    return torch.stack((cy*cz-1, cy*sz, -sy,
                        -cx*sz+sx*sy*cz, cx*cz+sx*sy*sz-1, sx*cy), -1).flatten(1)


class CudaPoseFitter:
    def __init__(self, reference, body=True):
        from studio.core.backends import FAST_CHECKPOINT
        began = time.perf_counter()
        self.device = torch.device('cuda')
        self.model = torch.jit.load(str(FAST_CHECKPOINT/'assets/mhr_model.pt'), map_location=self.device).eval()
        self.mapping = reference.mapping[BODY_IDS].to(self.device)
        self.selected = reference.indices if body else reference.indices[:3]
        self.indices = torch.tensor(self.selected, device=self.device)
        self.limits = reference.engine.limits[self.selected]
        self.correctives = bool(reference.state['correctives'])
        self.corrective_dense = None
        if self.correctives:
            # TorchScript reconstructs a COO matrix on every forward, which
            # synchronizes CUDA and cannot be captured. Materialize its small
            # first-layer matrix once; retain the full pose-corrective output.
            sparse = getattr(self.model.pose_correctives_model.pose_dirs_predictor, '0')
            self.corrective_dense = torch.sparse_coo_tensor(
                sparse.sparse_indices, sparse.sparse_weight, tuple(sparse.sparse_shape)).to_dense()
        self.cv = torch.tensor([1., -1., -1.], device=self.device)
        self.base = reference.base.to(self.device).clone()
        self.identity = reference.identity.to(self.device).clone()
        self.expression = reference.expression.to(self.device).clone()
        self.angle = torch.zeros(len(self.selected), device=self.device, requires_grad=True)
        self.shift = torch.zeros(3, device=self.device, requires_grad=True)
        self.rest = torch.zeros(1, 18439, 3, device=self.device)
        self.target = torch.zeros(1200, 3, device=self.device)
        self.normals = torch.zeros_like(self.target)
        self.ids = torch.zeros(1200, device=self.device, dtype=torch.long)
        self.valid = torch.zeros(1200, device=self.device)
        self.count = torch.ones((), device=self.device)
        self.t = torch.zeros(3, device=self.device)
        self.scale = torch.ones((), device=self.device)
        self.focal = torch.ones(2, device=self.device)
        self.center = torch.zeros(2, device=self.device)
        self.baseline_uv = torch.zeros(len(BODY_IDS), 2, device=self.device)
        self.sigma = torch.full((), 12., device=self.device)
        self.lo = torch.full_like(self.angle, -.1)
        self.hi = torch.full_like(self.angle, .1)
        self.shift_bound = torch.full((), .05, device=self.device)
        self.optimizer = torch.optim.Adam([self.angle, self.shift], lr=.008, capturable=True)
        self.prepare(reference, dict(fx=600, fy=600, cx=640, cy=360), 1., np.array([0, 0, 2.]), None)
        with torch.no_grad():
            _, vertex, _ = self.forward()
            expected = reference.forward(reference.base)[0].numpy()
            self.parity_mm = float(np.linalg.norm(vertex.cpu().numpy()-expected, axis=1).max()*1000)
        if self.parity_mm > .01:
            raise ValueError(f'cached MHR parity failed: {self.parity_mm:.6f} mm')
        # Warm up on a side stream before capture; do not capture JIT setup.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(5):
                self.step()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.step()
        torch.cuda.synchronize()
        self.setup_ms = (time.perf_counter()-began)*1000

    def project(self, p):
        return p[:, :2]/p[:, 2:].clamp(min=.05)*self.focal+self.center

    def forward(self):
        params = self.base.index_add(0, self.indices, self.angle)
        ct = self.model.character_torch
        jp = ct.model_parameters_to_joint_parameters(torch.cat((params[None], torch.zeros_like(self.identity)), 1))
        skel = ct.joint_parameters_to_skeleton_state(jp)
        rest = self.rest
        if self.correctives:
            pc = self.model.pose_correctives_model
            features = corrective_features(jp)
            hidden = torch.nn.functional.linear(features, self.corrective_dense)
            hidden = getattr(pc.pose_dirs_predictor, '1')(hidden)
            delta = getattr(pc.pose_dirs_predictor, '2')(hidden).reshape(1, -1, 3)
            rest = rest+delta
        v = ct.skin_points(skel, rest)[0]*.01*self.cv
        j = skel[0, :, :3]*.01*self.cv
        kp = self.mapping @ torch.cat((v, j), 0)
        return params, v, kp

    def step(self):
        from torch.nn.functional import huber_loss
        _, v, kp = self.forward()
        d = v[self.ids]*self.scale+self.t+self.shift-self.target
        plane = (d*self.normals).sum(1)/.02
        uv = (self.project(kp*self.scale+self.t+self.shift)-self.baseline_uv)/self.sigma
        # Padding has zero weight. Denominators match v2's mean reductions.
        loss = ((huber_loss(plane, torch.zeros_like(plane), reduction='none')*self.valid).sum()/self.count
                + .15*(huber_loss(d/.04, torch.zeros_like(d), reduction='none')*self.valid[:, None]).sum()/(3*self.count)
                + .5*uv.square().mean()+.05*(self.angle/.15).square().mean()
                + .02*(self.shift/.05).square().mean())
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            self.angle.copy_(torch.maximum(self.lo, torch.minimum(self.hi, self.angle)))
            self.shift.copy_(torch.maximum(-self.shift_bound, torch.minimum(self.shift_bound, self.shift)))
        self.loss = loss.detach()

    @torch.no_grad()
    def prepare(self, reference, k, scale, translation, config):
        if bool(reference.state['correctives']) != self.correctives:
            raise ValueError('correctives mode needs its own graph')
        self.base.copy_(reference.base); self.identity.copy_(reference.identity)
        self.expression.copy_(reference.expression)
        self.rest.copy_(self.model.character_torch.blend_shape(self.identity)+self.model.face_expressions_model(self.expression))
        self.t.copy_(torch.as_tensor(translation, device=self.device))
        self.scale.fill_(scale)
        self.focal.copy_(torch.tensor([k['fx'], k['fy']], device=self.device))
        self.center.copy_(torch.tensor([k['cx'], k['cy']], device=self.device))
        self.angle.zero_(); self.shift.zero_()
        for state in self.optimizer.state.values():
            for value in state.values():
                if isinstance(value, torch.Tensor):
                    value.zero_()
        if config is not None:
            if config.max_points > len(self.ids):
                raise ValueError('graph supports max_points <= 1200')
            radius = np.deg2rad(np.where(self.selected < 6, config.max_root_deg, config.max_angle_deg))
            lo, hi = delta_bounds(reference.base[self.selected].numpy(), self.limits, radius)
            self.lo.copy_(torch.as_tensor(lo, device=self.device)); self.hi.copy_(torch.as_tensor(hi, device=self.device))
            self.sigma.fill_(config.reprojection_sigma_px); self.shift_bound.fill_(config.max_translation_m)
        _, _, kp = self.forward()
        self.baseline_uv.copy_(self.project(kp*self.scale+self.t))

    def fit(self, reference, surface, scale, translation, detailed=False):
        began = time.perf_counter(); stages = {}; c = surface.c
        def stamp(name, start):
            torch.cuda.synchronize()
            stages[name] = stages.get(name, 0.)+(time.perf_counter()-start)*1000
        start = time.perf_counter()
        self.prepare(reference, surface.k, scale, translation, c)
        stamp('prepare_ms', start)
        history = []; count = 0; actual_steps = 0
        for step in range(c.steps):
            if step % c.correspond_every == 0:
                start = time.perf_counter()
                with torch.no_grad():
                    _, v, _ = self.forward()
                    camera = (v*self.scale+self.t+self.shift).cpu().numpy()
                stamp('correspondence_forward_download_ms', start)
                start = time.perf_counter()
                ids, target, normals = surface.pairs(camera); count = len(ids)
                stamp('correspondence_ms', start)
                if count < c.min_points:
                    break
                with torch.no_grad():
                    self.valid.zero_(); self.valid[:count] = 1; self.count.fill_(count)
                    self.ids[:count].copy_(torch.as_tensor(ids, device=self.device))
                    self.target[:count].copy_(torch.as_tensor(target, device=self.device))
                    self.normals[:count].copy_(torch.as_tensor(normals, device=self.device))
            start = time.perf_counter()
            self.graph.replay()
            actual_steps += 1
            if hasattr(surface, 'batches') and surface.batches:
                batch = surface.batches[-1]
                batch.setdefault('first_step', step)
                batch['used_steps'] = batch.get('used_steps', 0)+1
            if detailed:
                stamp('graph_step_ms', start)
            if step % c.correspond_every == 0:
                history.append(dict(step=step, loss=float(self.loss), points=count))
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            params, _, _ = self.forward()
            params = params.cpu().numpy().copy(); t = (self.t+self.shift).cpu().numpy().copy()
        stamp('final_download_ms', start)
        if not np.isfinite(params).all() or not np.isfinite(t).all() or not np.isfinite(float(self.loss)):
            raise ValueError('non-finite CUDA fit')
        report = dict(solver_seconds=time.perf_counter()-began, timings=stages, steps=actual_steps, requested_steps=c.steps,
                      points=count, sufficient=count >= c.min_points and actual_steps > 0, history=history,
                      implementation='cuda_graph_cached_rest_v1', cold_setup_ms=self.setup_ms,
                      correctives=self.correctives)
        return params, t, report


class AcceleratedMhrSurfaceModel(MhrSurfaceModel):
    """Drop-in offline experiment backend; original CPU backend stays default."""
    def __init__(self):
        super().__init__()
        self.fitters = {}

    def fit_pose(self, surface, arrays, scale, translation, body=True):
        began = time.perf_counter()
        key = (bool(self.state['correctives']), bool(body))
        cold = key not in self.fitters
        if cold:
            self.fitters[key] = CudaPoseFitter(self, body=body)
        params, t, report = self.fitters[key].fit(self, surface, scale, translation)
        with torch.no_grad():
            v, j, kp, quat = self.forward(torch.tensor(params))
            _, _, original_kp, _ = self.forward(self.base)
        selected = self.indices if body else self.indices[:3]
        from studio.rgbd.depth_pose_refinement import project
        delta_uv = project(kp.numpy()*scale+t, surface.k)-project(original_kp.numpy()*scale+translation, surface.k)
        report.update(solver_seconds=time.perf_counter()-began, cold_graph_built=cold,
                      config=asdict(surface.c),
                      changes=[dict(index=int(i), name=self.engine.names[i],
                                    delta_deg=float(np.rad2deg(params[i]-self.base[i].item()))) for i in selected],
                      delta_translation_m=(t-translation).tolist(),
                      max_keypoint_projection_change_px=float(np.linalg.norm(delta_uv[BODY_IDS], axis=1).max()))
        return params, v.numpy(), j.numpy(), kp.numpy(), quat.numpy(), t, report
