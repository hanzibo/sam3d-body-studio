# SPDX-License-Identifier: LicenseRef-SAM-3D-Body
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modifications Copyright (c) 2026 hanzibo and contributors.
# Feedback arithmetic adapted from SAM3DBody keypoint update methods, via Fast
# d72aa36913a1673ace029d345f437561a85ec9e2. Governed by LICENSES/SAM-3D-Body.txt.
"""Per-decoder-call Fast body feedback cache, without changing upstream code.

Only inference, one-way, channel-first body decoders use this adapter. Cached
values depend on this call's conditioned image and exact pose-output object.
Each layer still adds the feedback to its *new* token tensor. No state crosses
images, people, prompt rounds, or decoder calls; source tensors stay immutable.
"""
from functools import wraps


def feedback_values(model, image, pose):
    """Compute immutable 2D feature delta and 2D/3D position augmentations."""
    import torch
    from torch.nn import functional as F

    xy = pose['pred_keypoints_2d_cropped'][:, model.keypoint_embedding_idxs]
    depth = pose['pred_keypoints_2d_depth'][:, model.keypoint_embedding_idxs]
    xy01 = xy + 0.5
    valid = ~((xy01[..., 0] < 0) | (xy01[..., 0] > 1)
              | (xy01[..., 1] < 0) | (xy01[..., 1] > 1) | (depth < 1e-5))
    augment2d = model.keypoint_posemb_linear(xy) * valid[..., None]
    sample = xy * 2
    if model.cfg.MODEL.BACKBONE.TYPE in (
            'vit_hmr', 'vit', 'vit_b', 'vit_l', 'vit_hmr_512_384'):
        sample[..., 0] = sample[..., 0] / 12 * 16
    features = F.grid_sample(image, sample[:, :, None, :], mode='bilinear',
                             padding_mode='zeros', align_corners=False)
    features = features.squeeze(3).permute(0, 2, 1) * valid[..., None]
    delta = model.keypoint_feat_linear(features)
    xyz = pose['pred_keypoints_3d']
    xyz = xyz - (xyz[:, [model.pelvis_idx[0]]] + xyz[:, [model.pelvis_idx[1]]]) / 2
    augment3d = model.keypoint3d_posemb_linear(xyz[:, model.keypoint3d_embedding_idxs])
    return delta, augment2d, augment3d


class FeedbackCall:
    """One invocation only; pose identity is safe because its reference is kept."""
    def __init__(self, model, image, token_count):
        self.model = model
        self.image = image
        self.n2 = len(model.keypoint_embedding_idxs)
        self.n3 = len(model.keypoint3d_embedding_idxs)
        # Upstream appends 2D then 3D keypoint embeddings at the sequence end.
        self.start3 = token_count - self.n3
        self.start2 = self.start3 - self.n2
        self.pose = None
        self.values = None
        self.computations = 0
        self.hits = 0

    def __call__(self, tokens, augment, pose, layer_idx):
        if layer_idx == len(self.model.decoder.layers) - 1:
            return tokens, augment, pose, layer_idx
        if pose is not self.pose:
            self.values = feedback_values(self.model, self.image, pose)
            self.pose = pose
            self.computations += 1
        else:
            self.hits += 1
        delta, aug2, aug3 = self.values
        # Own exactly one copy of each mutable output. Do not mutate inputs:
        # prior decoder states or cached values may be retained by callers.
        out = tokens.clone()
        out[:, self.start2:self.start3, :] += delta
        out_aug = augment.clone()
        out_aug[:, self.start2:self.start3, :] = aug2
        out_aug[:, self.start3:self.start3 + self.n3, :] = aug3
        return out, out_aug, pose, layer_idx


def install_fast_feedback_cache(model):
    """Install an inference adapter and return its toggleable controller.

    Unsupported layouts keep the original callback. ``enabled=False`` is the
    untouched upstream feedback path, useful for paired output/timing checks.
    """
    if hasattr(model, '_studio_feedback_cache'):
        return model._studio_feedback_cache
    import torch
    decoder = model.decoder
    original = decoder.forward
    controller = {'enabled': True, 'last': None, 'version': 'per_call_v1'}

    @wraps(original)
    def forward(*args, **kwargs):
        controller['last'] = None
        tokens = args[0] if args else kwargs.get('token_embedding')
        image = args[1] if len(args) > 1 else kwargs.get('image_embedding')
        augment = args[2] if len(args) > 2 else kwargs.get('token_augment')
        channel_first = args[5] if len(args) > 5 else kwargs.get('channel_first', True)
        cfg = model.cfg.MODEL.DECODER
        supported = (
            controller['enabled'] and not model.training and not torch.is_grad_enabled()
            and kwargs.get('keypoint_token_update_fn') is not None
            and cfg.get('DO_KEYPOINT_TOKENS', False)
            and cfg.get('DO_KEYPOINT3D_TOKENS', False)
            and not cfg.get('ENABLE_TWOWAY', False)
            and channel_first and image is not None and image.ndim == 4
            and tokens is not None and augment is not None
            and tokens.shape == augment.shape
            and tokens.shape[1] >= len(model.keypoint_embedding_idxs)
                                    + len(model.keypoint3d_embedding_idxs) + 1
            and kwargs.get('hand_embeddings') is None
        )
        if not supported:
            return original(*args, **kwargs)
        call = FeedbackCall(model, image, tokens.shape[1])
        kwargs['keypoint_token_update_fn'] = call
        try:
            return original(*args, **kwargs)
        finally:
            controller['last'] = {'computations': call.computations,
                                  'hits': call.hits}
            # The closure and its GPU tensors die on return, including errors.
    decoder.forward = forward
    model._studio_feedback_cache = controller
    return controller
