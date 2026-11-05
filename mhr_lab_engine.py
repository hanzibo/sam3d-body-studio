# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""MHR-only editor. Native units internally; preview arrays use Studio camera axes/metres."""
from pathlib import Path
import importlib.util
import json
import time
import numpy as np
import torch
import roma
from backends import FAST_CHECKPOINT, FAST_REPO

ASSET = FAST_CHECKPOINT/'assets/mhr_model.pt'


def vector(value, length, name):
    a = np.asarray(value, dtype=np.float32)
    if a.shape != (length,) or not np.isfinite(a).all():
        raise ValueError(f'{name} 必须是 {length} 个有限数值')
    return a.copy()


class LabEngine:
    def __init__(self, device='auto'):
        torch.set_num_threads(4)
        self.device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device)
        self.model = torch.jit.load(str(ASSET), map_location=self.device).eval()
        self.names = list(self.model.get_parameter_names())[:204]
        self.joint_names = list(self.model.get_joint_names())
        self.faces = self.model.character_torch.mesh.faces.cpu().numpy().astype(np.uint32)
        self.parents = self.model.character_torch.skeleton.joint_parents.cpu().numpy().astype(np.int64)
        self.transform = self.model.get_parameter_transform().cpu().numpy()[:, :204].reshape(127, 7, 204)
        self.limits = self.model.get_parameter_limits().cpu().numpy()[:204]
        self.hand_buffers = None
        self.base = None
        self.baseline = None
        self.source = ''
        with torch.inference_mode():
            for _ in range(3):
                self.model(torch.zeros(1,45,device=self.device),torch.zeros(1,204,device=self.device),torch.zeros(1,72,device=self.device))
        if self.device.type=='cuda':torch.cuda.synchronize()

    def metadata(self):
        kinds = []
        for i in range(204):
            col = self.transform[:, :, i]
            kinds.append('angle' if np.any(col[:, 3:6]) and not np.any(col[:, :3]) and not np.any(col[:, 6]) else 'value')
        return dict(names=self.names, joint_names=self.joint_names, parents=self.parents.tolist(),
                    limits=self.limits.tolist(), kinds=kinds, device=str(self.device),
                    joint_parameters=[np.flatnonzero(np.any(self.transform[j] != 0, axis=0)).tolist() for j in range(127)])

    def recover_fast_parameters(self, data):
        # Only mmap the trusted local checkpoint; no backbone construction or GPU upload.
        if self.hand_buffers is None:
            sd = torch.load(FAST_CHECKPOINT/'model.ckpt', map_location='cpu', weights_only=True, mmap=True)
            sd = sd.get('state_dict', sd)
            wanted = ['scale_mean', 'scale_comps', 'hand_pose_mean', 'hand_pose_comps', 'hand_joint_idxs_left', 'hand_joint_idxs_right']
            self.hand_buffers = {k: sd['head_pose.'+k].detach().float().clone() for k in wanted}
        b = self.hand_buffers
        out = torch.zeros(1, 204)
        out[0, 3:6] = torch.from_numpy(vector(data['global_rot'], 3, 'global_rot'))
        body = np.asarray(data['body_pose'], dtype=np.float32).reshape(-1)
        if len(body) not in (130, 133) or not np.isfinite(body).all():
            raise ValueError('body_pose 尺寸或数值无效')
        out[0, 6:136] = torch.from_numpy(body[:130].copy())
        hand = torch.from_numpy(vector(data['hand'], 108, 'hand')).reshape(2, 54)
        spec = importlib.util.spec_from_file_location('lab_mhr_utils', FAST_REPO/'sam_3d_body/models/modules/mhr_utils.py')
        utils = importlib.util.module_from_spec(spec); spec.loader.exec_module(utils)
        hp = utils.compact_cont_to_model_params_hand_fast(b['hand_pose_mean'] + hand @ b['hand_pose_comps'])
        out[0, b['hand_joint_idxs_left'].long()] = hp[0]
        out[0, b['hand_joint_idxs_right'].long()] = hp[1]
        scale = torch.from_numpy(vector(data['scale'], 28, 'scale'))
        out[0, 136:] = b['scale_mean'] + scale @ b['scale_comps']
        return out[0].numpy()

    def seed(self, data=None, source='默认中性人体'):
        if data is not None:
            data = dict(data)
            for short, stored in [('body_pose','body_pose_params'),('hand','hand_pose_params'),('scale','scale_params'),('shape','shape_params'),('face','expr_params')]:
                if short not in data and stored in data:data[short] = data[stored]
        if data is None:
            state = dict(model=np.zeros(204, np.float32).tolist(), shape=np.zeros(45, np.float32).tolist(),
                         expression=np.zeros(72, np.float32).tolist(), overrides={}, correctives=True)
        elif 'lab_state_json' in data:
            state = json.loads(str(data['lab_state_json']))
        else:
            model = vector(data['mhr_model_params'], 204, 'mhr_model_params') if 'mhr_model_params' in data else self.recover_fast_parameters(data)
            state = dict(model=model.tolist(), shape=vector(data['shape'], 45, 'shape').tolist(),
                         expression=vector(data.get('face', np.zeros(72)), 72, 'face').tolist(), overrides={}, correctives=True)
        # Commit only after the candidate successfully generates a finite mesh.
        result = self.render(state)
        if data is not None and 'pred_vertices' in data and 'lab_state_json' not in data:
            original = np.asarray(data['pred_vertices'], dtype=np.float32)
            if original.shape != result['vertices'].shape or not np.isfinite(original).all():
                raise ValueError('导入网格尺寸或数值无效')
            alternate = dict(state, correctives=not state['correctives'])
            candidate = self.render(alternate)
            if np.mean((candidate['vertices']-original)**2) < np.mean((result['vertices']-original)**2):
                state, result = alternate, candidate
            errors = np.linalg.norm(result['vertices']-original, axis=1)*1000
            result['import_mean_error_mm'] = np.asarray(errors.mean())
            result['import_max_error_mm'] = np.asarray(errors.max())
        base_state = json.loads(str(data['lab_base_json'])) if data is not None and 'lab_base_json' in data else state
        baseline_result = self.render(base_state) if base_state != state else result
        self.base = json.loads(json.dumps(base_state))
        self.baseline = baseline_result['vertices'].copy()
        self.source = source
        return state, result

    @torch.inference_mode()
    def render(self, state):
        began = time.perf_counter()
        mp = torch.as_tensor(vector(state['model'], 204, 'model'), device=self.device)[None]
        identity = torch.as_tensor(vector(state['shape'], 45, 'shape'), device=self.device)[None]
        expr = torch.as_tensor(vector(state['expression'], 72, 'expression'), device=self.device)[None]
        overrides = state.get('overrides', {})
        jp = self.model.character_torch.model_parameters_to_joint_parameters(torch.cat([mp, torch.zeros_like(identity)], 1))
        jp = jp.reshape(1, 127, 7).clone()
        for key, values in overrides.items():
            j = int(key)
            if not 0 <= j < 127:
                raise ValueError('关节编号必须介于 0–126')
            jp[0, j, :6] = torch.as_tensor(vector(values, 6, '关节局部变换'), device=self.device)
        flat = jp.reshape(1, -1)
        ct = self.model.character_torch
        local = ct.joint_parameters_to_local_skeleton_state(flat)
        skel = ct.local_skeleton_state_to_skeleton_state(local)
        rest = ct.blend_shape(identity) + self.model.face_expressions_model(expr)
        if state.get('correctives', True):
            rest = rest + self.model.pose_correctives_model(flat)
        vertices = ct.skin_points(skel, rest)
        # Native +Y up, cm -> existing Studio camera-space metres. Do not mix raw SAM rotations with camera positions.
        cv = torch.tensor([1, -1, -1], device=self.device, dtype=vertices.dtype)
        result = dict(vertices=(vertices[0]*.01*cv).cpu().numpy(),
                      joints=(skel[0, :, :3]*.01*cv).cpu().numpy(),
                      joint_parameters=jp[0].cpu().numpy(),
                      local_positions=local[0, :, :3].cpu().numpy(),
                      global_positions=skel[0, :, :3].cpu().numpy(),
                      local_rotations=roma.unitquat_to_rotmat(local[0, :, 3:7]).cpu().numpy(),
                      global_rotations=roma.unitquat_to_rotmat(skel[0, :, 3:7]).cpu().numpy())
        if not all(np.isfinite(a).all() for a in result.values()):
            raise ValueError('当前参数产生了非有限值，请恢复参数')
        result['compute_ms'] = np.asarray((time.perf_counter()-began)*1000)
        return result
