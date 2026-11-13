# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Frozen-input, synchronized single-frame pose fitting benchmark.

The pose-only experiment starts at the SAME saved v2 rigid result for all
variants. Pipeline mode instead starts at v1 translation and runs both stages.
No new network inference, source overwrite, GUI publishing or camera access.
"""
import argparse
import cProfile
from dataclasses import replace, asdict
import hashlib
import json
from pathlib import Path
import pstats
import time
import cv2
import numpy as np

from depth_pose_refinement import MhrSurfaceModel, SurfaceFit, PoseDepthConfig, acceptance
from depth_pose_experiment import inputs, load_arrays, digest
from rgbd_evaluate import DepthRenderer
from rgbd_geometry import depth_metrics
from depth_pose_render import FittingDepthRenderer

ROOT = Path(__file__).resolve().parent
VARIANTS = {'cpu60': (False, 60), 'cuda60': (True, 60), 'cuda30': (True, 30),
            'cuda15': (True, 15), 'cuda10': (True, 10)}


def write(file, value):
    file.parent.mkdir(parents=True, exist_ok=True)
    temp = file.with_suffix(file.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)); temp.replace(file)


def summarize(records):
    out = []
    for profile, variant, mode in sorted({(r['profile'], r['variant'], r['mode']) for r in records}):
        group = [r for r in records if (r['profile'], r['variant'], r['mode']) == (profile, variant, mode)]
        row = dict(profile=profile, variant=variant, mode=mode, count=len(group))
        for name in ('solver_ms', 'warm_path_ms', 'prep_ms', 'accept_ms', 'mae_mm', 'silhouette_iou', 'guard_mae_mm'):
            values = [r[name] for r in group]
            row[name+'_mean'] = float(np.mean(values))
        for name in ('solver_ms', 'warm_path_ms'):
            values = np.concatenate([r['repeated_'+name] for r in group])
            row[name+'_p50'] = float(np.median(values)); row[name+'_p95'] = float(np.percentile(values, 95))
        row['accepted'] = sum(r['accepted'] for r in group)
        row['same_initial_rigid'] = mode == 'pose'
        row['estimated_network_plus_path_ms'] = float(np.mean([r['network_ms']+r['warm_path_ms'] for r in group]))
        out.append(row)
    return out


def run(args):
    import torch
    from gpu_lease import acquire_gpu_lease
    lease = acquire_gpu_lease('exclusive')
    from depth_pose_accelerated import CudaPoseFitter
    out = Path(args.output).resolve(); out.mkdir(parents=True, exist_ok=False)
    source = ROOT/'outputs/depth-fusion/experiment-v1'
    v2 = ROOT/'outputs/depth-fusion/experiment-v2'
    masks = ROOT/'outputs/depth-fusion/masks/masks.json'
    rows = json.loads(masks.read_text())
    if args.poses:
        rows = [r for r in rows if r['pose'] in args.poses]
    config = PoseDepthConfig()
    model = MhrSurfaceModel(); graphs = {}; records = []; parity = []; cold = []
    write(out/'manifest.json', dict(schema='studio_depth_speed_v1', arguments=vars(args),
        torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), timing_definition='In-memory RGB-D + recovered native MHR to accepted camera mesh. Includes fresh normals/tree, solver, guard rendering/backtracking. Excludes IO, native-parameter recovery/parity, SAM inference, GUI, exports, dense point-triangle evaluation, cold model/graph/GL setup.',
        quality_definition='Visible sensor surface consistency only; guard selects acceptance and is not independent ground truth.',
        source_sha256={str(f): digest(f) for f in [masks, ROOT/'depth_pose_refinement.py', ROOT/'depth_pose_accelerated.py', ROOT/'depth_pose_render.py', ROOT/'depth_pose_benchmark.py']},
        network_time_note='Estimated sum uses previously synchronized frozen-network benchmark, not a new end-to-end measurement.'))
    snapshot = out/'source-snapshot'; snapshot.mkdir()
    for name in json.loads((out/'manifest.json').read_text())['source_sha256']:
        file = Path(name)
        (snapshot/file.name).write_bytes(file.read_bytes())
    for profile in args.profiles:
        for row in rows:
            sample, folder, translation_arrays, z, xyz, k, mask = inputs(row, source, profile)
            seed_file = v2/profile/f"pose-{row['pose']:03d}"/'rigid/person_00.npz'
            initial = load_arrays(seed_file) if args.mode == 'pose' else translation_arrays
            model.seed(initial); initial_params = model.base.numpy().copy()
            initial_t = initial['pred_cam_t'].copy()
            original_render = DepthRenderer(initial['faces'], k)
            fast_render = FittingDepthRenderer(initial['faces'], k, original_render.renderer)
            initial_mesh = initial['pred_vertices']+initial_t
            # Warm both EGL paths and verify depth before any fitting.
            old_z = original_render.render(initial_mesh); new_z = fast_render.render(initial_mesh)
            overlap = (old_z > .05) & (new_z > .05)
            check = dict(profile=profile, pose=row['pose'], correctives=model.state['correctives'],
                         depth_max_error_mm=float(np.abs(old_z[overlap]-new_z[overlap]).max()*1000),
                         silhouette_disagreement=int(np.count_nonzero((old_z > .05) != (new_z > .05))),
                         seed_sha256=digest(seed_file) if args.mode == 'pose' else digest(folder/'evaluation/refined_prediction.npz'))
            if check['depth_max_error_mm'] > .1 or check['silhouette_disagreement'] > 5:
                raise ValueError('depth renderer parity failed: '+str(check))
            parity.append(check)
            try:
                for variant in args.variants:
                    accelerated, steps = VARIANTS[variant]
                    body_modes = [False, True] if args.mode == 'pipeline' else [True]
                    if accelerated:
                        for body in body_modes:
                            key = (bool(model.state['correctives']), body)
                            if key not in graphs:
                                graphs[key] = CudaPoseFitter(model, body)
                                cold.append(dict(correctives=key[0], body=body, setup_ms=graphs[key].setup_ms))
                    renderer = fast_render if accelerated else original_render
                    repeated = []
                    for repeat in range(args.warmups+args.repeats):
                        model.base = torch.tensor(initial_params)
                        t = initial_t.copy(); params = initial_params.copy(); arrays = initial
                        torch.cuda.synchronize(); start = time.perf_counter()
                        conf = replace(config, steps=steps)
                        surface = SurfaceFit(z, xyz, mask, k, renderer.render, conf)
                        prep_ms = (time.perf_counter()-start)*1000
                        solver_ms = 0.; accept_ms = 0.; reports = []; accepted = False
                        for body in body_modes:
                            model.base = torch.tensor(params)
                            before, _ = surface.score(arrays['pred_vertices']+t)
                            if accelerated:
                                p, candidate_t, report = graphs[(bool(model.state['correctives']), body)].fit(model, surface, 1., t)
                            else:
                                profiler = cProfile.Profile() if not records and repeat == args.warmups else None
                                if profiler:
                                    profiler.enable()
                                p, _, _, _, _, candidate_t, report = model.fit_pose(surface, arrays, 1., t, body)
                                if profiler:
                                    profiler.disable(); profiler.dump_stats(str(out/'cpu-fit.prof'))
                                    with (out/'cpu-profile.txt').open('w') as stream:
                                        pstats.Stats(profiler, stream=stream).sort_stats('cumtime').print_stats(35)
                            solver_ms += report['solver_seconds']*1000
                            check_start = time.perf_counter(); trials = []
                            accepted = False
                            if report['sufficient']:
                                for fraction in (1., .5, .25):
                                    candidate_params = params+(p-params)*fraction
                                    shift = t+(candidate_t-t)*fraction
                                    with torch.no_grad():
                                        v = model.forward(torch.tensor(candidate_params))[0].numpy()
                                    after, _ = surface.score(v+shift)
                                    reason = acceptance(before, after)
                                    trials.append(dict(fraction=fraction, reason=reason, guard=after))
                                    if reason == 'accepted':
                                        params, t = candidate_params, shift
                                        arrays = dict(arrays, pred_vertices=v)
                                        accepted = True
                                        break
                            accept_ms += (time.perf_counter()-check_start)*1000
                            reports.append(dict(fit=report, trials=trials, accepted=accepted, body=body))
                        torch.cuda.synchronize(); warm_ms = (time.perf_counter()-start)*1000
                        repeated.append(dict(solver_ms=solver_ms, warm_path_ms=warm_ms,
                                             prep_ms=prep_ms, accept_ms=accept_ms))
                        # Keep the FIRST measured result, never cherry-pick quality or runtime.
                        if repeat == args.warmups:
                            measured_arrays = arrays; measured_t = t.copy(); measured_params = params.copy()
                            measured_reports = reports; measured_accepted = accepted
                    measured = repeated[args.warmups:]
                    mesh = measured_arrays['pred_vertices']+measured_t
                    rendered = original_render.render(mesh)  # independent original renderer for reported metrics
                    metrics, _, _ = depth_metrics(z, rendered, mask)
                    guard, _ = surface.score(mesh)
                    network_ms = json.loads((folder/'prediction/inference.json').read_text())['benchmark']['mean_ms']
                    result = dict(profile=profile, pose=row['pose'], variant=variant, mode=args.mode,
                                  accepted=measured_accepted, reports=measured_reports,
                                  network_ms=network_ms, mae_mm=metrics['mae_mm'],
                                  silhouette_iou=metrics['silhouette_iou'], guard_mae_mm=guard['capped_mae_mm'],
                                  repeated_solver_ms=[r['solver_ms'] for r in measured],
                                  repeated_warm_path_ms=[r['warm_path_ms'] for r in measured])
                    result.update({name: float(np.median([r[name] for r in measured])) for name in measured[0]})
                    dest = out/profile/f"pose-{row['pose']:03d}"/variant; dest.mkdir(parents=True)
                    np.savez_compressed(dest/'mesh.npz', vertices=measured_arrays['pred_vertices'], translation=measured_t,
                                        model_params=measured_params, faces=initial['faces'])
                    write(dest/'result.json', result)
                    records.append(result)
                    write(out/'records.json', records)
                    write(out/'summary.json', summarize(records))
                    write(out/'parity.json', parity); write(out/'cold-setup.json', cold)
                    print(f"{profile} {row['pose']:02d} {variant}: solver {result['solver_ms']:.1f} / path {result['warm_path_ms']:.1f} ms / {result['mae_mm']:.2f} mm", flush=True)
            finally:
                original_render.close(); fast_render.close()
    write(out/'complete.json', dict(records=len(records), done=True))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--profiles', nargs='+', choices=['fast384', 'fast512', 'standard512'], default=['fast384', 'fast512', 'standard512'])
    p.add_argument('--poses', type=int, nargs='+')
    p.add_argument('--variants', nargs='+', choices=list(VARIANTS), default=list(VARIANTS))
    p.add_argument('--mode', choices=['pose', 'pipeline'], default='pose')
    p.add_argument('--warmups', type=int, default=1); p.add_argument('--repeats', type=int, default=3)
    args = p.parse_args()
    if args.repeats < 1 or args.warmups < 0:
        p.error('repeats >= 1, warmups >= 0 required')
    run(args)


if __name__ == '__main__':
    main()
