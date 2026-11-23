# SPDX-License-Identifier: GPL-3.0-only
"""Hand ROI validation, skinning-based topology and explicit mirror transforms."""
import numpy as np


def validate_bbox(box, width, height):
    a = np.asarray(box, dtype=np.float32)
    if a.shape != (4,) or not np.isfinite(a).all():
        raise ValueError('请先框选整只手')
    x1, y1, x2, y2 = a
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError('手框必须位于原图内')
    if min(x2-x1, y2-y1) < 12:
        raise ValueError('手框过小，宽高至少 12 像素')
    return a


def hand_topology(names, parents, faces, vertex_indices, joint_indices, weights, count):
    """Select wrist descendants by >=50% accumulated skin weight; keep full triangles."""
    wrist = list(names).index('r_wrist')
    descendants = {wrist}
    for _ in names:
        descendants.update(i for i, p in enumerate(parents) if int(p) in descendants)
    joint_ids = np.array(sorted(descendants), dtype=np.int32)
    mass = np.zeros(count, np.float64)
    selected = np.isin(joint_indices, joint_ids)
    np.add.at(mass, np.asarray(vertex_indices)[selected], np.asarray(weights)[selected])
    selected_vertices = mass >= .5
    chosen = np.asarray(faces)[selected_vertices[np.asarray(faces)].all(axis=1)]
    if not len(chosen):
        raise ValueError('MHR 手部拓扑提取失败')
    vertex_ids = np.unique(chosen)
    remap = np.full(count, -1, np.int32); remap[vertex_ids] = np.arange(len(vertex_ids))
    lookup = {int(j): i for i, j in enumerate(joint_ids)}
    edges = np.array([(lookup[int(p)], lookup[int(j)]) for j in joint_ids
                      if (p := int(parents[j])) in lookup], dtype=np.int32)
    return vertex_ids, remap[chosen], joint_ids, edges, wrist


def mirror_geometry(vertices, joints, rotations, translation, faces):
    """Reflect camera coordinates; reverse winding and conjugate global rotations."""
    s = np.diag([-1., 1., 1.]).astype(np.float32)
    return (vertices @ s, joints @ s, s @ rotations @ s,
            np.asarray(translation) @ s, faces[:, [0, 2, 1]])


def validate_hand_outputs(raw, vertex_ids):
    """Reject invalid 3D/hand data; identify undefined irrelevant projections.

    The native hand head also projects its unused full-body surface. A point
    exactly on the camera plane has no finite perspective projection. Allow
    that only outside the exported hand AND within 1e-6 m of that plane. Keep
    the original raw projection and attach a validity mask, never invent pixels.
    """
    projection=raw['pred_keypoints_2d_verts']
    for key,value in raw.items():
        if key!='pred_keypoints_2d_verts' and not np.isfinite(value).all():
            raise ValueError('手部输出非有限值：'+key)
    valid=np.isfinite(projection).all(axis=-1)
    bad=np.flatnonzero(~valid)
    if not valid[vertex_ids].all():raise ValueError('手部顶点投影非有限值')
    if len(bad):
        z=raw['pred_vertices'][bad,2]+raw['pred_cam_t'][2]
        if np.any(np.abs(z)>1e-6):raise ValueError(f'非手部投影异常，且不在相机平面：ids={bad.tolist()}, z={z.tolist()}')
    return valid,bad
