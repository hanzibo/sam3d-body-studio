# SPDX-License-Identifier: GPL-3.0-only
"""Opt-in MHR geometry precision experiments. Decoder and parameter assembly stay FP32.

Install before compilation/graph capture. Original assets are never modified.
Mixed modes keep FK and public outputs in FP32; only blend/skin work uses FP16.
"""
import copy
from types import MethodType
import torch
from torch import nn

MODES = ('fp32', 'final_mixed', 'all_mixed', 'all_fp16')


class MixedMHR(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.original = original
        ct = original.character_torch
        self.register_buffer('base_half', ct.blend_shape.base_shape.detach().half().clone())
        self.register_buffer('shape_half', ct.blend_shape.shape_vectors.detach().half().clone())
        self.register_buffer('expression_half', original.face_expressions_model.shape_vectors.detach().half().clone())
        self.skinning_half = copy.deepcopy(ct.linear_blend_skinning).half()

    @property
    def character_torch(self):
        return self.original.character_torch

    @property
    def face_expressions_model(self):
        return self.original.face_expressions_model

    def forward(self, shape, params, expression, apply_correctives=False):
        if apply_correctives:
            raise ValueError('MHR precision experiments require correctives disabled')
        ct = self.original.character_torch
        joint_params = ct.model_parameters_to_joint_parameters(torch.cat((params.float(), torch.zeros_like(shape).float()), 1))
        skeleton = ct.joint_parameters_to_skeleton_state(joint_params)
        rest = torch.einsum('nvd,bn->bvd', self.shape_half, shape.half()) + self.base_half
        rest = rest + torch.einsum('nvd,bn->bvd', self.expression_half, expression.half())
        vertices = self.skinning_half(skeleton.half(), rest)
        return vertices.float(), skeleton


class HalfMHR(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.original = original
        self.half_model = copy.deepcopy(original).half()

    @property
    def character_torch(self):
        return self.original.character_torch

    @property
    def face_expressions_model(self):
        return self.original.face_expressions_model

    def forward(self, shape, params, expression, apply_correctives=False):
        if apply_correctives:
            raise ValueError('MHR precision experiments require correctives disabled')
        vertices, skeleton = self.half_model(shape.half(), params.half(), expression.half(), False)
        return vertices.float(), skeleton.float()


def selected_mixed_forward(self, shape, params, expression):
    ct = self.character
    joint_params = ct.model_parameters_to_joint_parameters(torch.cat((params.float(), torch.zeros_like(shape).float()), 1))
    skeleton = ct.joint_parameters_to_skeleton_state(joint_params)
    rest = torch.einsum('nvd,bn->bvd', self.shape_half, shape.half()) + self.base_half
    rest = rest + torch.einsum('nvd,bn->bvd', self.expression_half, expression.half())
    vertices = self.skinning_half(skeleton.half(), rest).float() * .01
    joints = skeleton[..., :3] * .01
    kps = torch.einsum('kv,bvc->bkc', self.mapping, torch.cat((vertices, joints[:, self.joints]), 1))
    return kps, joints


def selected_half_forward(self, shape, params, expression):
    # A private copy preserves the original module and its FP32 buffers.
    kps, joints = self.half_geometry(shape.half(), params.half(), expression.half())
    return kps.float(), joints.float()


def install_mhr_precision(model, mode):
    if mode not in MODES:
        raise ValueError(f'Unknown MHR precision experiment: {mode}')
    status = dict(mode=mode, decoder_dtype='float32', public_dtype='float32',
                  fk_dtype='float16' if mode=='all_fp16' else 'float32',
                  final_geometry='float32' if mode=='fp32' else 'mixed' if mode!='all_fp16' else 'float16',
                  intermediate_geometry='mixed' if mode=='all_mixed' else 'float16' if mode=='all_fp16' else 'float32')
    if mode == 'fp32':
        return status
    head = model.head_pose
    if head.apply_correctives or head.enable_hand_model:
        raise ValueError('Precision experiments support body/no-correctives only')
    selected = head._studio_selected_mhr70
    if mode=='all_mixed':
        for name, source in [('base',selected.base_shape),('shape',selected.shape_vectors),('expression',selected.expression_vectors)]:
            selected.register_buffer(name+'_half',source.detach().half().clone())
        selected.skinning_half=copy.deepcopy(selected.skinning).half()
        selected.forward=MethodType(selected_mixed_forward,selected)
    elif mode=='all_fp16':
        selected.half_geometry=copy.deepcopy(selected).half()
        selected.forward=MethodType(selected_half_forward,selected)
    head.mhr=(HalfMHR(head.mhr) if mode=='all_fp16' else MixedMHR(head.mhr)).eval()
    return status
