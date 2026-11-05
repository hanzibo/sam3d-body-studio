# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Metric RGB-D geometry. All arrays use colour camera CV axes, metres.

No alignment or shape fitting is performed. Projection is into a rectified colour
image with the supplied effective K. Raw depth remains available for auditing.
"""
import cv2
import numpy as np


def matrix(k):
    return np.array([[k['fx'], 0, k['cx']], [0, k['fy'], k['cy']], [0, 0, 1]], dtype=np.float64)


def coefficients(d):
    return np.array([d.get(n, 0) for n in ('k1','k2','p1','p2','k3','k4','k5','k6')], np.float64)


def register_depth(raw_m, depth_k, depth_d, color_k, rotation, translation_m):
    """Calibrated deprojection -> rigid D2C -> nearest-surface colour Z buffer.

    Returns sparse, undilated colour-camera depth and exact measured XYZ per
    occupied pixel. Never fills holes or fabricates depth on discontinuities.
    """
    h, w = int(color_k['height']), int(color_k['width'])
    yy, xx = np.nonzero(np.isfinite(raw_m) & (raw_m > .05) & (raw_m < 20))
    uv = np.column_stack((xx, yy)).astype(np.float64).reshape(-1, 1, 2)
    rays = cv2.undistortPoints(uv, matrix(depth_k), coefficients(depth_d)).reshape(-1, 2)
    xyz = np.column_stack((rays, np.ones(len(rays)))) * raw_m[yy, xx, None]
    xyz = xyz @ np.asarray(rotation).reshape(3, 3).T + np.asarray(translation_m).reshape(3)
    good = np.isfinite(xyz).all(1) & (xyz[:, 2] > .05)
    xyz = xyz[good]
    uv = np.rint(xyz[:, :2] / xyz[:, 2:] * [color_k['fx'], color_k['fy']] +
                 [color_k['cx'], color_k['cy']]).astype(np.int64)
    good = (uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
    xyz, uv = xyz[good], uv[good]
    # Stable nearest-Z winner at each target pixel, independent of input order.
    ids = uv[:,1]*w+uv[:,0]
    order = np.lexsort((xyz[:,2], ids))
    ids, xyz = ids[order], xyz[order]
    first = np.r_[True, ids[1:] != ids[:-1]] if len(ids) else np.zeros(0, bool)
    cloud = np.zeros((h*w,3), np.float32)
    cloud[ids[first]] = xyz[first]
    cloud = cloud.reshape(h,w,3)
    return cloud[:,:,2].copy(), cloud


def write_ply(path, xyz, rgb=None, faces=None):
    xyz = np.asarray(xyz, np.float32)
    fields = [('x','<f4'),('y','<f4'),('z','<f4')]
    if rgb is not None: fields += [('red','u1'),('green','u1'),('blue','u1')]
    data = np.empty(len(xyz), dtype=fields)
    for i, n in enumerate(('x','y','z')): data[n] = xyz[:,i]
    if rgb is not None:
        for i,n in enumerate(('red','green','blue')): data[n] = rgb[:,i]
    header = 'ply\nformat binary_little_endian 1.0\ncomment camera_x_right_y_down_z_forward_metres\n'
    header += f'element vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\n'
    if rgb is not None: header += 'property uchar red\nproperty uchar green\nproperty uchar blue\n'
    if faces is not None: header += f'element face {len(faces)}\nproperty list uchar int vertex_indices\n'
    with open(path,'wb') as f:
        f.write((header+'end_header\n').encode());data.tofile(f)
        if faces is not None:
            fd=np.empty(len(faces),dtype=[('count','u1'),('indices','<i4',(3,))])
            fd['count']=3;fd['indices']=faces;fd.tofile(f)


def depth_metrics(observed, rendered, mask):
    mask = np.asarray(mask, bool)
    if not mask.any(): raise ValueError('请先独立圈选人体区域或载入人体掩膜')
    valid = np.isfinite(observed)&(observed>.05)
    model = np.isfinite(rendered)&(rendered>.05)
    overlap = mask&valid&model
    if not overlap.any(): raise ValueError('人体区域内没有模型与有效深度的重合，请检查坐标、人物和掩膜')
    residual = rendered[overlap]-observed[overlap]
    absolute = np.abs(residual)
    # Silhouette scores do not discard invalid depth pixels or missing model.
    result = dict(mask_pixels=int(mask.sum()), overlap_pixels=int(overlap.sum()),
        valid_depth_fraction=float((mask&valid).sum()/mask.sum()),
        model_coverage_fraction=float((mask&model).sum()/mask.sum()),
        silhouette_iou=float((mask&model).sum()/(mask|model).sum()),
        signed_mean_mm=float(residual.mean()*1000), median_signed_mm=float(np.median(residual)*1000),
        mae_mm=float(absolute.mean()*1000), median_absolute_mm=float(np.median(absolute)*1000),
        p90_mm=float(np.percentile(absolute,90)*1000),
        within_10mm=float((absolute<.01).mean()),within_20mm=float((absolute<.02).mean()),
        within_50mm=float((absolute<.05).mean()),
        definition='raw metric camera coordinates; model Z minus observed Z; no fitting; visible surface only')
    return result, overlap, residual


def surface_distances(points, vertices, faces):
    """Exact point-to-triangle distance, accelerated without optional rtree.

    A nearest vertex is an upper bound on distance. Every potentially closer
    triangle has its centroid within bound + maximum centroid-to-corner radius.
    Hence the KD-tree ball search cannot omit the nearest triangle.
    """
    import trimesh
    from scipy.spatial import cKDTree
    tri = np.asarray(vertices)[np.asarray(faces)]
    centres = tri.mean(1)
    radius = np.linalg.norm(tri-centres[:,None,:],axis=2).max()
    bound = cKDTree(np.asarray(vertices)[np.unique(faces)]).query(points)[0]
    tree = cKDTree(centres)
    distances = np.empty(len(points))
    for start in range(0,len(points),64):
        ps=points[start:start+64]
        candidates=tree.query_ball_point(ps,bound[start:start+64]+radius+1e-8)
        for offset,(p,indices) in enumerate(zip(ps,candidates)):
            ids=np.asarray(indices,dtype=np.int64)
            closest=trimesh.triangles.closest_point(tri[ids],np.repeat(p[None],len(ids),axis=0))
            distances[start+offset]=np.linalg.norm(closest-p,axis=1).min()
    return distances
