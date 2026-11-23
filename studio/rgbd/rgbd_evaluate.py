# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Offline visible-surface evaluation, isolated EGL process, no model loading."""
import os
os.environ['PYOPENGL_PLATFORM']='egl'
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import pyrender
import trimesh
from studio.rgbd.rgbd_geometry import depth_metrics, surface_distances, write_ply


class DepthRenderer:
    """One EGL context per evaluation, reused by the raw and corrected surfaces."""
    def __init__(self,faces,k):
        self.faces=faces;self.k=k
        self.renderer=pyrender.OffscreenRenderer(int(k['width']),int(k['height']))

    def render(self,vertices):
        vertices=np.array(vertices,copy=True);vertices[:,1:]*=-1
        scene=pyrender.Scene()
        scene.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(vertices,self.faces,process=False)))
        k=self.k
        scene.add(pyrender.IntrinsicsCamera(k['fx'],k['fy'],k['cx'],k['cy'],znear=.05,zfar=20))
        return self.renderer.render(scene,flags=pyrender.RenderFlags.SKIP_CULL_FACES)[1]

    def close(self):self.renderer.delete()


def render_depth(vertices,faces,k):
    renderer=DepthRenderer(faces,k)
    try:return renderer.render(vertices)
    finally:renderer.close()


def evaluate(sample,prediction,mask_path,output,refine=False):
    sample,prediction,output=Path(sample),Path(prediction),Path(output)
    output.mkdir(parents=True,exist_ok=False)
    meta=json.loads((sample/'camera.json').read_text());k=meta['intrinsics']
    with np.load(sample/'rgbd.npz',allow_pickle=False) as f:observed,xyz=f['depth_m'],f['xyz_m']
    with np.load(prediction,allow_pickle=False) as f:
        vertices=np.asarray(f['pred_vertices'])+np.asarray(f['pred_cam_t']).reshape(1,3);faces=f['faces']
    if not np.isfinite(vertices).all():raise ValueError('模型坐标包含无效数值')
    mask=cv2.imread(str(mask_path),cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape!=observed.shape:raise ValueError('人体掩膜尺寸不匹配')
    mask=mask>0
    renderer=DepthRenderer(faces,k)
    try:
        rendered=renderer.render(vertices)
        if refine:
            from studio.rgbd.depth_refinement import refine_translation
            delta,corrected,refinement=refine_translation(observed,xyz,mask,rendered,k,lambda d:renderer.render(vertices+d))
    finally:renderer.close()
    metrics,overlap,residual=depth_metrics(observed,rendered,mask)
    # Fixed seed, uniform sample of independently masked measured points.
    points=xyz[mask&(observed>.05)]
    rng=np.random.default_rng(0)
    if len(points)>2000:points=points[rng.choice(len(points),2000,replace=False)]
    distance=surface_distances(points,vertices,faces)
    metrics.update(point_surface_count=len(points),point_surface_mae_mm=float(distance.mean()*1000),
        point_surface_median_mm=float(np.median(distance)*1000),point_surface_p90_mm=float(np.percentile(distance,90)*1000),
        point_surface_definition='observed masked points to full triangle surface, uniform sample max 2000, seed 0; no back-surface penalty',
        sample=str(sample.resolve()),prediction=str(prediction.resolve()),mask_sha256=hashlib.sha256(Path(mask_path).read_bytes()).hexdigest())
    source_path=sample/'mask_source.json'
    if source_path.is_file():
        source=json.loads(source_path.read_text())
        if source.get('mask_sha256')==metrics['mask_sha256']:
            metrics['mask_source']=source
            (output/'mask_source.json').write_text(json.dumps(source,ensure_ascii=False,indent=2))
    metrics.setdefault('mask_source',dict(method='external_or_manual',meaning='independent image mask; not anatomical ground truth'))
    image=cv2.imread(str(sample/'rgb.png'))
    signed=np.zeros_like(observed);signed[overlap]=residual
    heat=cv2.applyColorMap(np.uint8(np.clip((signed+.05)/.10,0,1)*255),cv2.COLORMAP_COOLWARM if hasattr(cv2,'COLORMAP_COOLWARM') else cv2.COLORMAP_JET)
    heat[~overlap]=image[~overlap]//3
    cv2.imwrite(str(output/'depth_error.png'),heat)
    cv2.imwrite(str(output/'mask.png'),np.uint8(mask)*255)
    np.savez_compressed(output/'depth_comparison.npz',observed_m=observed,rendered_m=rendered,overlap=overlap,
                        point_surface_points_m=points,point_surface_distances_m=distance)
    write_ply(output/'body_camera.ply',vertices,faces=faces)
    write_ply(output/'person_points.ply',xyz[mask&(observed>.05)],image[:,:,::-1][mask&(observed>.05)])
    if refine:
        refined_metrics,refined_overlap,refined_residual=depth_metrics(observed,corrected,mask)
        refined_metrics['definition']='translation-refined camera coordinates; visible measured surface consistency'
        refined_distance=surface_distances(points,vertices+delta,faces)
        refined_metrics.update(point_surface_count=len(points),point_surface_mae_mm=float(refined_distance.mean()*1000),
            point_surface_median_mm=float(np.median(refined_distance)*1000),point_surface_p90_mm=float(np.percentile(refined_distance,90)*1000))
        refinement.update(metrics=refined_metrics,prediction_sha256=hashlib.sha256(prediction.read_bytes()).hexdigest(),
            rgbd_sha256=hashlib.sha256((sample/'rgbd.npz').read_bytes()).hexdigest(),
            camera_sha256=hashlib.sha256((sample/'camera.json').read_bytes()).hexdigest(),
            mask_sha256=metrics['mask_sha256'],rgb_sha256=hashlib.sha256((sample/'rgb.png').read_bytes()).hexdigest())
        metrics['depth_refinement']=refinement
        with np.load(prediction,allow_pickle=False) as source:
            arrays={key:source[key].copy() for key in source.files}
        arrays['pred_cam_t']=arrays['pred_cam_t']+delta.astype(arrays['pred_cam_t'].dtype)
        # Local mesh/joints/rotations are intentionally identical; only camera translation changes.
        np.savez_compressed(output/'refined_prediction.npz',**arrays)
        np.savez_compressed(output/'refined_depth.npz',rendered_m=corrected,delta_translation_m=delta)
        write_ply(output/'body_camera_refined.ply',vertices+delta,faces=faces)
        signed=np.zeros_like(observed);signed[refined_overlap]=refined_residual
        heat=cv2.applyColorMap(np.uint8(np.clip((signed+.05)/.10,0,1)*255),cv2.COLORMAP_JET)
        heat[~refined_overlap]=image[~refined_overlap]//3
        cv2.imwrite(str(output/'depth_error_refined.png'),heat)
        (output/'refinement.json').write_text(json.dumps(refinement,ensure_ascii=False,indent=2))
    (output/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2))
    print('@@RGBD '+json.dumps(dict(type='evaluated',output=str(output),metrics=metrics)),flush=True)
    return metrics


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sample',required=True);p.add_argument('--prediction',required=True)
    p.add_argument('--refine',action='store_true');p.add_argument('--mask',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    try:evaluate(args.sample,args.prediction,args.mask,args.output,args.refine)
    except Exception as exc:
        import traceback;traceback.print_exc()
        print('@@RGBD '+json.dumps(dict(type='error',message=str(exc))),flush=True);raise SystemExit(1)
