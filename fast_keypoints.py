# SPDX-License-Identifier: LicenseRef-SAM-3D-Body
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modifications Copyright (c) 2026 hanzibo and contributors.
# Parameter assembly adapted from Fast MHRHead._mhr_forward_core at
# d72aa36913a1673ace029d345f437561a85ec9e2. See LICENSES/SAM-3D-Body.txt.
"""Exact-support MHR70 geometry for intermediate body predictions only.

Select every nonzero vertex/joint dependency, without thresholding regression
weights. Keep all 127 skeleton transforms and the original TorchScript skinning
operator; restrict identity/expression bases and skin influences to needed
vertices. Final predictions, gradients and correctives use upstream geometry.
"""
import copy
from functools import wraps

import torch
from torch import nn


class SelectedMHR70(nn.Module):
    def __init__(self, mhr, mapping):
        super().__init__()
        ct = mhr.character_torch
        nverts = ct.blend_shape.base_shape.shape[0]
        njoints = ct.skeleton.joint_translation_offsets.shape[0]
        if tuple(mapping.shape) != (308, nverts + njoints):
            raise ValueError('Unsupported MHR keypoint mapping layout')
        w = mapping[:70].detach()
        support = (w != 0).any(dim=0)
        vertices = support[:nverts].nonzero().flatten()
        joints = support[nverts:].nonzero().flatten()
        if not vertices.numel() or not joints.numel():
            raise ValueError('MHR70 mapping must include surface and joint support')
        self.register_buffer('vertices', vertices)
        self.register_buffer('joints', joints)
        self.register_buffer('mapping', torch.cat((w[:, vertices], w[:, nverts + joints]), dim=1).contiguous())
        self.register_buffer('base_shape', ct.blend_shape.base_shape[vertices].detach().contiguous())
        self.register_buffer('shape_vectors', ct.blend_shape.shape_vectors[:, vertices].detach().contiguous())
        self.register_buffer('expression_vectors', mhr.face_expressions_model.shape_vectors[:, vertices].detach().contiguous())
        # Share immutable skeleton and parameter transform; never trim ancestors.
        self.character = ct
        lbs = copy.deepcopy(ct.linear_blend_skinning)
        remap = torch.full((nverts,), -1, dtype=torch.long, device=vertices.device)
        remap[vertices] = torch.arange(len(vertices), device=vertices.device)
        compact = remap[lbs.vert_indices_flattened]
        keep = compact >= 0
        lbs.skin_indices_flattened = lbs.skin_indices_flattened[keep].clone()
        lbs.skin_weights_flattened = lbs.skin_weights_flattened[keep].clone()
        lbs.vert_indices_flattened = compact[keep].clone()
        lbs.num_vertices = len(vertices)
        self.skinning = lbs
        self.spec = dict(vertices=len(vertices), joints=len(joints),
                         influences=int(keep.sum().item()), keypoints=70,
                         original_vertices=nverts, original_keypoints=308)

    def forward(self, shape, model_params, expression):
        rest = torch.einsum('nvd,bn->bvd', self.shape_vectors, shape) + self.base_shape
        rest = rest + torch.einsum('nvd,bn->bvd', self.expression_vectors, expression)
        joint_params = self.character.model_parameters_to_joint_parameters(
            torch.cat((model_params, torch.zeros_like(shape)), dim=1))
        skeleton = self.character.joint_parameters_to_skeleton_state(joint_params)
        vertices = self.skinning(skeleton, rest) * .01
        joints = skeleton[..., :3] * .01
        kps = torch.einsum('kv,bvc->bkc', self.mapping,
                           torch.cat((vertices, joints[:, self.joints]), dim=1))
        return kps, joints


    @torch.compiler.disable
    def inference(self, shape, model_params, expression):
        """Replay static single-person geometry; return owned outputs each time.

        TorchScript FK otherwise dispatches many tiny CUDA kernels from the host.
        Capture only this geometry block, not decoder feedback or final outputs.
        CPU and other batches retain the same ordinary tensor implementation.
        """
        if shape.device.type != 'cuda' or shape.shape[0] != 1:
            return self(shape, model_params, expression)
        signature = tuple((tuple(x.shape), x.dtype, x.device) for x in (shape, model_params, expression))
        cached = getattr(self, '_replay', None)
        if cached is None or cached[0] != signature:
            inputs = tuple(x.clone() for x in (shape, model_params, expression))
            stream = torch.cuda.Stream(device=shape.device)
            stream.wait_stream(torch.cuda.current_stream(shape.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self(*inputs)
            torch.cuda.current_stream(shape.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                outputs = self(*inputs)
            cached = (signature, inputs, graph, outputs)
            self._replay = cached
        _, inputs, graph, outputs = cached
        for target, source in zip(inputs, (shape, model_params, expression)):
            target.copy_(source)
        graph.replay()
        # Feedback from earlier layers must survive the next replay.
        return tuple(x.clone() for x in outputs)


def install_fast_keypoint_path(model):
    """Adapt only slim, no-correctives, inference-mode body MHR calls.

    Install after checkpoint loading but before warmup. Controller ``enabled``
    is a compile guard: toggle only outside timing, then warm the selected path.
    """
    if hasattr(model, '_studio_keypoint_path'):
        return model._studio_keypoint_path
    head = model.head_pose
    head._studio_selected_mhr70 = SelectedMHR70(head.mhr, head.keypoint_mapping).eval()
    original = head._mhr_forward_core
    controller = dict(enabled=True, version='mhr70_exact_support_graph_v2',
                      single_person_cuda_graph=True,
                      **head._studio_selected_mhr70.spec)

    @wraps(original)
    def mhr_core(global_trans, global_rot, body_pose_params, hand_pose_params,
                 scale_params, shape_params, expr_params,
                 return_keypoints=True, slim_mode=False):
        eligible = (slim_mode and return_keypoints and controller['enabled']
                    and not head.training and not torch.is_grad_enabled()
                    and not head.enable_hand_model
                    and not getattr(head, 'apply_correctives', True))
        if not eligible:
            return original(global_trans, global_rot, body_pose_params, hand_pose_params,
                            scale_params, shape_params, expr_params,
                            return_keypoints=return_keypoints, slim_mode=slim_mode)
        scales = head.scale_mean[None, :] + scale_params @ head.scale_comps
        pose = torch.cat((global_trans * 10, global_rot, body_pose_params[..., :130]), dim=1)
        if hand_pose_params is not None:
            pose = head.replace_hands_in_pose(pose, hand_pose_params)
        params = torch.cat((pose, scales), dim=1)
        kps, joints = head._studio_selected_mhr70.inference(shape_params, params, expr_params)
        return None, kps, joints, params, None

    head._mhr_forward_core = mhr_core
    model._studio_keypoint_path = controller
    return controller
