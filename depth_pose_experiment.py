# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Offline, resumable staged RGB-D experiment using frozen network predictions."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
from depth_pose_refinement import MhrSurfaceModel, SurfaceFit, PoseDepthConfig, acceptance, project
from depth_experiment import save
from rgbd_evaluate import DepthRenderer
from rgbd_geometry import depth_metrics, surface_distances, write_ply

STAGES = {'translation': '仅平移', 'scale': '固定尺度', 'rigid': '整体朝向', 'pose': '身体姿态'}


def digest(file):
    with Path(file).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_arrays(file):
    with np.load(file, allow_pickle=False) as source:
        return {key: source[key].copy() for key in source.files}


def inputs(row, source, profile):
    sample = Path(row['sample']); folder = source/profile/f"pose-{row['pose']:03d}"
    mask_path = Path(row['mask']).resolve()
    if digest(sample/'rgb.png') != row['rgb_sha256'] or digest(mask_path) != row['mask_sha256']:
        raise ValueError('frozen RGB/mask changed')
    reference = json.loads((folder/'evaluation/refinement.json').read_text())
    for field, file in [('rgbd_sha256', sample/'rgbd.npz'), ('camera_sha256', sample/'camera.json'),
                        ('prediction_sha256', folder/'prediction/person_00.npz')]:
        if reference[field] != digest(file):
            raise ValueError('v1 input changed: '+str(file))
    arrays = load_arrays(folder/'evaluation/refined_prediction.npz')
    with np.load(sample/'rgbd.npz') as cloud:
        depth, xyz = cloud['depth_m'], cloud['xyz_m']
    k = json.loads((sample/'camera.json').read_text())['intrinsics']
    mask = cv2.imread(str(mask_path), 0) > 0
    return sample, folder, arrays, depth, xyz, k, mask


def derived_arrays(model, params, scale, translation, original, k):
    import torch
    import roma
    with torch.no_grad():
        v, j, kp, quat = model.forward(torch.tensor(params, dtype=torch.float32))
        rotations = roma.unitquat_to_rotmat(quat).numpy()
    out = {key: value.copy() for key, value in original.items() if key != 'pred_pose_raw'}
    out.update(pred_vertices=v.numpy()*scale, pred_cam_t=np.asarray(translation, np.float32),
               pred_keypoints_3d=kp.numpy()*scale, pred_keypoints_2d=project(kp.numpy()*scale+translation, k).astype(np.float32),
               pred_joint_coords=j.numpy()*scale,
               pred_global_rots=rotations, mhr_model_params=np.asarray(params, np.float32),
               global_rot=np.asarray(params[3:6], np.float32), depth_metric_scale=np.asarray(scale),
               prediction_origin=np.asarray('depth_surface_postfit_v2'))
    body = out['body_pose_params'].copy(); body[:130] = params[6:136]; out['body_pose_params'] = body
    return out


def stage_metrics(depth, xyz, mask, vertices, faces, rendered):
    metrics, overlap, residual = depth_metrics(depth, rendered, mask)
    points = xyz[mask & np.isfinite(xyz).all(2) & (depth > .05)]
    if len(points) > 2000:
        points = points[np.random.default_rng(0).choice(len(points), 2000, replace=False)]
    dist = surface_distances(points, vertices, faces)
    metrics.update(point_surface_count=len(points), point_surface_mae_mm=float(dist.mean()*1000),
                   point_surface_p90_mm=float(np.percentile(dist, 90)*1000),
                   definition='visible sensor surface consistency; not anatomical pose accuracy')
    return metrics, overlap, residual


def save_stage(dest, arrays, surface, depth, xyz, k, sample, mask_hash, report, original_report):
    dest.mkdir(parents=True, exist_ok=False)
    vertices = arrays['pred_vertices']+arrays['pred_cam_t']
    score, rendered = surface.score(vertices)
    metrics, overlap, residual = stage_metrics(depth, xyz, surface.mask, vertices, arrays['faces'], rendered)
    metrics.update(mask_sha256=mask_hash, sample=str(sample), depth_fit=report)
    report['applied_guard'] = score
    np.savez_compressed(dest/'person_00.npz', **arrays)
    save(dest/'metrics.json', metrics); save(dest/'fit.json', report)
    save(dest/'inference.json', dict(original_report, depth_fit=report,
         note='Network time belongs to the original inference; fitting is separate postprocessing.'))
    np.savez_compressed(dest/'depth_comparison.npz', rendered_m=rendered)
    rgb = cv2.imread(str(sample/'rgb.png'))
    heat = cv2.applyColorMap(np.uint8(np.clip((residual+.05)/.1, 0, 1)*255), cv2.COLORMAP_JET)
    image = rgb//3; image[overlap] = heat.reshape(-1, 3)
    cv2.imwrite(str(dest/'depth_error.png'), image)
    contour, _ = cv2.findContours(np.uint8(rendered > .05)*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rgb, contour, -1, (80, 240, 120), 2)
    cv2.imwrite(str(dest/'overlay.png'), rgb)
    write_ply(dest/'body_camera.ply', vertices, faces=arrays['faces'])
    return metrics


def calibrate(rows, source, profile, config):
    estimates = []; evidence = []
    for row in rows:
        _, _, a, z, xyz, k, mask = inputs(row, source, profile)
        renderer = DepthRenderer(a['faces'], k)
        try:
            surface = SurfaceFit(z, xyz, mask, k, renderer.render, config)
            before, _ = surface.score(a['pred_vertices']+a['pred_cam_t'])
            scale, t, fit = surface.fit_scale(a['pred_vertices'], a['pred_cam_t'])
            after, _ = surface.score(a['pred_vertices']*scale+t)
            reason = acceptance(before, after)
            # The scale bound is a diagnostic, not evidence of a valid optimum.
            bounded = abs(scale-1) >= config.max_scale_fraction*.999
            reliable = fit['sufficient'] and reason == 'accepted' and not bounded
            if reliable:
                estimates.append(scale)
            evidence.append(dict(pose=row['pose'], scale=scale, reliable=reliable, bound_reached=bounded,
                                 before=before, candidate=after, reason=reason, fit=fit))
        finally:
            renderer.close()
    # Require at least two consistent calibration poses; do not fit validation poses.
    valid = len(estimates) >= 2 and np.ptp(estimates) < .03
    scale = float(np.median(estimates)) if valid else 1.
    return dict(scale=scale, accepted=valid, calibration_poses=[r['pose'] for r in rows],
                reason='consistent_calibration' if valid else 'insufficient_consistent_scale_evidence',
                meaning='one per-profile metric multiplier; shape remains frame-dependent, not a true fixed identity model',
                evidence=evidence)


def process(row, source, out, profile, model, calibration, config, refresh_derived=False, fit_backend='cpu'):
    sample, folder, original, depth, xyz, k, mask = inputs(row, source, profile)
    dest = out/profile/f"pose-{row['pose']:03d}"
    manifest = dict(schema='studio_depth_pose_experiment_v2', profile=profile, pose=row['pose'],
                    config=asdict(config), calibration=calibration,
                    source_sha256={str(f.resolve()): digest(f) for f in [sample/'rgb.png', sample/'rgbd.npz', sample/'camera.json',
                        Path(row['mask']), folder/'prediction/person_00.npz', folder/'evaluation/refined_prediction.npz']})
    if fit_backend != 'cpu':
        manifest['fit_backend'] = fit_backend
    if (dest/'complete.json').exists():
        if json.loads((dest/'complete.json').read_text()) != manifest:
            raise ValueError('resume input/config mismatch: '+str(dest))
        if refresh_derived:
            model.seed(original)
            for stage in STAGES:
                folder_stage=dest/stage
                fit=json.loads((folder_stage/'fit.json').read_text())
                refreshed=derived_arrays(model, np.asarray(fit['native_mhr_model_params'],np.float32),
                                         fit['scale'],np.asarray(fit['translation_m'],np.float32),original,k)
                previous=load_arrays(folder_stage/'person_00.npz')
                if not np.allclose(previous['pred_vertices'],refreshed['pred_vertices'],atol=1e-7,rtol=0):
                    raise ValueError('derived refresh would change evaluated geometry')
                temp=folder_stage/'person_00.refresh.npz'
                np.savez_compressed(temp,**refreshed);temp.replace(folder_stage/'person_00.npz')
            print(f'{profile} pose {row["pose"]:02d}: derived arrays refreshed; evaluated vertices unchanged',flush=True)
        return
    if dest.exists():
        raise ValueError('incomplete experiment directory; preserve it and choose another output: '+str(dest))
    dest.mkdir(parents=True)
    renderer = DepthRenderer(original['faces'], k)
    try:
        fitting_renderer = renderer
        if fit_backend == 'cuda-graph':
            from depth_pose_render import FittingDepthRenderer
            fitting_renderer = FittingDepthRenderer(original['faces'], k, renderer.renderer)
        surface = SurfaceFit(depth, xyz, mask, k, fitting_renderer.render, config)
        original_report = json.loads((folder/'prediction/inference.json').read_text())
        parity = model.seed(original); params = model.base.numpy().copy()
        scale = 1.; translation = original['pred_cam_t'].copy(); arrays = original
        for stage in STAGES:
            began = time.perf_counter(); before, _ = surface.score(arrays['pred_vertices']+arrays['pred_cam_t'])
            report = dict(stage=stage, stage_label=STAGES[stage], config=asdict(config), parity=parity,
                          scope='surface consistency, not skeletal truth; guard used for acceptance, not an independent test',
                          calibration=calibration, accepted=False, reason='v1_translation_baseline',
                          candidate_guard=before, before_guard=before)
            if stage == 'scale':
                if calibration['accepted']:
                    s, t, fit = surface.fit_scale(original['pred_vertices'], translation, calibration['scale'])
                    candidate, _ = surface.score(original['pred_vertices']*s+t)
                    reason = acceptance(before, candidate) if fit['sufficient'] else 'insufficient_correspondences'
                    report.update(fit=fit, candidate_guard=candidate, reason=reason, accepted=reason == 'accepted')
                    if report['accepted']:
                        scale, translation = s, t
                        arrays = derived_arrays(model, params, scale, translation, original, k)
                else:
                    report['reason'] = calibration['reason']
            elif stage in ('rigid', 'pose'):
                # Keep the scaled camera mesh separate from the native MHR parameters.
                model.base = __import__('torch').tensor(params, dtype=__import__('torch').float32)
                p, _, _, _, _, t, fit = model.fit_pose(surface, arrays, scale, translation, body=stage == 'pose')
                report.update(fit=fit, reason='insufficient_correspondences', trials=[])
                if fit['sufficient']:
                    # Fixed, predeclared backtracking; guard remains a selection check.
                    for fraction in (1., .5, .25):
                        candidate_params = params+(p-params)*fraction
                        candidate_t = translation+(t-translation)*fraction
                        candidate_arrays = derived_arrays(model, candidate_params, scale, candidate_t, original, k)
                        candidate, _ = surface.score(candidate_arrays['pred_vertices']+candidate_t)
                        reason = acceptance(before, candidate)
                        report['trials'].append(dict(fraction=fraction, guard=candidate, reason=reason))
                        if reason == 'accepted':
                            params, translation, arrays = candidate_params, candidate_t, candidate_arrays
                            report.update(accepted=True, reason=reason, applied_fraction=fraction, candidate_guard=candidate)
                            break
                    if not report['accepted']:
                        report['reason'] = report['trials'][-1]['reason']
            report.update(scale=float(scale), total_seconds=time.perf_counter()-began,
                          native_mhr_model_params=params.tolist(), translation_m=translation.tolist())
            # The network prediction remains unmodified in the source directory.
            if stage == 'translation':
                arrays = derived_arrays(model, params, scale, translation, original, k)
            # Evaluate exports with the original renderer independently of the
            # fitting buffer optimization; share its context to avoid EGL teardown.
            surface.render = renderer.render
            try:
                metrics = save_stage(dest/stage, arrays, surface, depth, xyz, k, sample, row['mask_sha256'], report, original_report)
            finally:
                surface.render = fitting_renderer.render
            print(f"{profile} pose {row['pose']:02d} {stage}: {metrics['mae_mm']:.1f}mm / {report['reason']}", flush=True)
        save(dest/'complete.json', manifest)
    finally:
        renderer.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True); p.add_argument('--masks', required=True); p.add_argument('--output', required=True)
    p.add_argument('--profiles', nargs='+', choices=['standard512', 'fast512', 'fast384'], default=['standard512', 'fast512', 'fast384'])
    p.add_argument('--calibration-poses', nargs='+', type=int, default=[1, 2, 3])
    p.add_argument('--poses', nargs='+', type=int)
    p.add_argument('--fit-backend', choices=['cpu', 'cuda-graph'], default='cpu',
                   help='Opt-in cached CUDA fitting; CPU v2 remains the reference default')
    p.add_argument('--pose-steps', type=int, default=60, help='Steps per rigid/pose stage; reduced budgets need quality comparison')
    p.add_argument('--refresh-derived', action='store_true', help='Regenerate derived arrays from saved fitted parameters; abort if evaluated vertices change')
    args = p.parse_args()
    from gpu_lease import acquire_gpu_lease
    lease = acquire_gpu_lease('exclusive')
    source = Path(args.source).resolve(); out = Path(args.output).resolve(); out.mkdir(parents=True, exist_ok=True)
    rows = json.loads(Path(args.masks).read_text()); config = PoseDepthConfig(steps=args.pose_steps).validate()
    if args.fit_backend == 'cuda-graph':
        from depth_pose_accelerated import AcceleratedMhrSurfaceModel
        model = AcceleratedMhrSurfaceModel()
    else:
        model = MhrSurfaceModel()
    for profile in args.profiles:
        folder = out/profile; folder.mkdir(exist_ok=True)
        calibration_file = folder/'calibration.json'
        calibration_rows = [r for r in rows if r['pose'] in args.calibration_poses]
        if len(calibration_rows) != len(args.calibration_poses):
            raise ValueError('calibration pose missing')
        # Recheck source and calibration membership even on resume.
        calibration = calibrate(calibration_rows, source, profile, config)
        if calibration_file.exists() and json.loads(calibration_file.read_text()) != calibration:
            raise ValueError('calibration changed on resume')
        save(calibration_file, calibration)
        print(profile+' calibration '+str(calibration['scale'])+' / '+calibration['reason'], flush=True)
        for row in rows:
            if args.poses and row['pose'] not in args.poses:
                continue
            process(row, source, out, profile, model, calibration, config, args.refresh_derived, args.fit_backend)
    print('COMPLETE: raw predictions preserved; report generation is a separate step', flush=True)
    lease.close()


if __name__ == '__main__':
    main()
