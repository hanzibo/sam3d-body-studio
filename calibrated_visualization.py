# SPDX-License-Identifier: LicenseRef-SAM-3D-Body
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modifications Copyright (c) 2026 hanzibo and contributors.
# Adapted from SAM 3D Body renderer.py and tools/vis_utils.py at
# b5c765a0d89d789985e186d396315e7590887b94 (see docs/LICENSE_AUDIT.md).
# Modified: exact fx/fy/cx/cy, renderer cleanup, single-person panel composition.
# This file is governed by LICENSES/SAM-3D-Body.txt, not the root GPL grant.
"""Comparison panels with the exact RGB pinhole matrix used for inference."""
import cv2
import numpy as np
import pyrender
import trimesh
from sam_3d_body.visualization.renderer import create_raymond_lights
from tools.vis_utils import visualizer, LIGHT_BLUE


def render_mesh(image, vertices, translation, faces, matrix, side=False):
    height, width = image.shape[:2]
    mesh = trimesh.Trimesh(vertices.copy(), faces.copy(), process=False)
    if side:
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
    material = pyrender.MetallicRoughnessMaterial(metallicFactor=0, alphaMode='OPAQUE',
                                                baseColorFactor=(*LIGHT_BLUE[::-1], 1))
    scene = pyrender.Scene(bg_color=[1, 1, 1, 0], ambient_light=(.3, .3, .3))
    scene.add(pyrender.Mesh.from_trimesh(mesh, material=material))
    pose = np.eye(4);pose[:3, 3] = translation;pose[0, 3] *= -1
    scene.add(pyrender.IntrinsicsCamera(fx=matrix[0, 0], fy=matrix[1, 1],
              cx=matrix[0, 2], cy=matrix[1, 2], zfar=1e12), pose=pose)
    for light in create_raymond_lights():scene.add_node(light)
    renderer = pyrender.OffscreenRenderer(width, height)
    try:
        rgba, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    finally:
        renderer.delete()
    alpha = rgba[..., 3:4].astype(np.float32) / 255
    return (rgba[..., :3] * alpha + image * (1 - alpha)).clip(0, 255).astype(np.uint8)


def visualize_with_intrinsics(image, outputs, faces, camera_info):
    if len(outputs) != 1:
        raise ValueError('精确内参对照目前仅支持单人')
    person = outputs[0]
    matrix = np.asarray(camera_info['matrix'], dtype=np.float32)
    points = np.concatenate([person['pred_keypoints_2d'], np.ones((70, 1))], axis=1)
    skeleton = visualizer.draw_skeleton(image.copy(), points)
    # Same centering convention as upstream visualization; no refit to image edges.
    vertices = person['pred_vertices'] + person['pred_cam_t']
    center = (vertices.max(axis=0) + vertices.min(axis=0)) / 2
    centered = vertices - center
    overlay = render_mesh(image, centered, center, faces, matrix)
    side = render_mesh(np.full_like(image, 255), centered, center, faces, matrix, side=True)
    return np.concatenate([image, skeleton, overlay, side], axis=1)
