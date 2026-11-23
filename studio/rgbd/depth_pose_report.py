# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Staged surface-fit evidence and optional non-destructive Studio publication."""
import argparse
import csv
import html
import json
from pathlib import Path
import shutil
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw
from studio.rgbd.depth_experiment import save
from studio.rgbd.depth_pose_experiment import STAGES, digest, load_arrays


def publish(sample, result_dir, raw_dir, profile, stage, info, metrics, report_file, mask_file):
    """Add rows without using the original batch cache identity for fitted data."""
    from studio.rgbd.recorder_session import recording_active
    if recording_active(sample.parent.parent.parent):
        raise ValueError('capture is active; cannot publish')
    if not (sample/'mask.png').exists():
        # Add the same independent mask used for fitting; never replace a user's mask.
        shutil.copy2(mask_file,sample/'mask.png')
    if digest(sample/'mask.png') != metrics['mask_sha256']:
        raise ValueError('current mask differs from experiment; refusing publication')
    raw_label = 'depth-v2-'+profile+'-raw'
    label = 'depth-v2-'+profile+'-'+stage
    source = sample/'predictions'/raw_label
    target = sample/'predictions'/label
    evaluation = sample/'evaluations'/label
    if not source.exists():
        shutil.copytree(raw_dir, source)
    if digest(source/'person_00.npz') != digest(raw_dir/'person_00.npz'):
        raise ValueError('published source mismatch')
    for dest in (target, evaluation):
        if not dest.exists():
            shutil.copytree(result_dir, dest)
        elif digest(dest/'person_00.npz') != digest(result_dir/'person_00.npz'):
            raise ValueError('published stage mismatch')
    request=raw_dir.parent/'request.json'
    if request.is_file():
        for dest in (source,target):
            if not (dest/'request.json').exists():shutil.copy2(request,dest/'request.json')
    report = json.loads((result_dir/'fit.json').read_text())
    report.update(source_prediction=str(source/'person_00.npz'),
                  source_prediction_sha256=digest(source/'person_00.npz'),
                  experiment_report=str(report_file),
                  fit_inputs={name: digest(sample/name) for name in ('rgb.png', 'rgbd.npz', 'camera.json', 'mask.png')})
    result = dict(label=label, config=info['actual_config'], prediction=str(target/'person_00.npz'),
                  evaluation=str(evaluation), inference_ms=info['benchmark']['mean_ms'],
                  rgb_sha256=digest(sample/'rgb.png'), person_bbox_xyxy=info['bbox_xyxy'],
                  input_selection='mask_bbox', depth_fit=report, metrics=metrics,
                  note='Separate post-fit row; original network time reused, CPU fitting reported separately.')
    index = sample/'results.json'
    previous = json.loads(index.read_text()) if index.exists() else []
    backup = sample/'history-trash'/('before-depth-v2-'+time.strftime('%Y%m%d-%H%M%S')+'.json')
    if not any(r.get('depth_fit') for r in previous):
        backup.parent.mkdir(exist_ok=True); save(backup, previous)
    save(index, [r for r in previous if r.get('label') != label]+[result])


def camera_planes(file, observed, meshes):
    canvas = Image.new('RGB', (1200, 500), '#0b1220'); draw = ImageDraw.Draw(canvas)
    colours = ['#f6ad55', '#68b7ff', '#b591ff', '#42e68a']
    # Same coordinate limits across variants in each panel, no individual alignment.
    for panel, (a, b, name) in enumerate([(0, 1, 'XY'), (0, 2, 'XZ'), (2, 1, 'ZY')]):
        sets = [p[:, [a, b]] for p in [observed]+meshes]
        bounds = np.concatenate(sets); lo = bounds.min(0); hi = bounds.max(0)
        factor = min(350/max(hi[0]-lo[0], .01), 390/max(hi[1]-lo[1], .01))
        for points, colour in zip(sets, colours):
            xy = (points-lo)*factor+[panel*400+25, 55]
            draw.point([tuple(p) for p in xy], fill=colour)
        draw.text((panel*400+25, 12), name+' / metres', fill='white')
    draw.text((20, 465), 'ORANGE measured | BLUE translation | PURPLE rigid | GREEN pose (same camera coordinates)', fill='white')
    canvas.save(file)


def build(root, source, masks, do_publish=False):
    rows = json.loads(Path(masks).read_text()); records = []; sections = []
    for profile in ('standard512', 'fast512', 'fast384'):
        for row in rows:
            folder = root/profile/f"pose-{row['pose']:03d}"
            if not (folder/'complete.json').exists():
                continue
            manifest = json.loads((folder/'complete.json').read_text())
            for name, expected in manifest['source_sha256'].items():
                if digest(name) != expected:
                    raise ValueError('source changed since experiment: '+name)
            sample = Path(row['sample']); tiles = []; scene_meshes = []; pose_records = []
            for stage, title in STAGES.items():
                directory = folder/stage; metrics = json.loads((directory/'metrics.json').read_text())
                fit = json.loads((directory/'fit.json').read_text())
                inference = json.loads((directory/'inference.json').read_text())
                record = dict(profile=profile, pose=row['pose'], stage=stage,
                              group='calibration' if row['pose'] in manifest['calibration']['calibration_poses'] else 'validation',
                              accepted=fit['accepted'], reason=fit['reason'], scale=fit['scale'],
                              postprocess_ms=fit['total_seconds']*1000,
                              network_ms=inference['benchmark']['mean_ms'],
                              guard_capped_mae_mm=fit['applied_guard']['capped_mae_mm'])
                for key in ('mae_mm', 'p90_mm', 'silhouette_iou', 'model_coverage_fraction', 'point_surface_mae_mm'):
                    record[key] = metrics[key]
                arrays = load_arrays(directory/'person_00.npz')
                initial = load_arrays(folder/'translation/person_00.npz')
                joints = arrays['pred_keypoints_3d']+arrays['pred_cam_t']
                initial_joints = initial['pred_keypoints_3d']+initial['pred_cam_t']
                changes = np.linalg.norm(joints-initial_joints, axis=1)*1000
                record.update(left_elbow_change_mm=float(changes[7]), right_elbow_change_mm=float(changes[8]),
                              left_wrist_change_mm=float(changes[62]), right_wrist_change_mm=float(changes[41]))
                save(directory/'joint_changes.json', dict(
                    meaning='predicted landmark movement relative to translation baseline, NOT errors against ground truth',
                    keypoint_movement_mm=changes.tolist()))
                records.append(record); pose_records.append(record)
                image = Image.open(directory/'overlay.png'); image.thumbnail((480, 270))
                tile = Image.new('RGB', (480, 300), '#172033'); tile.paste(image, (0, 30))
                ImageDraw.Draw(tile).text((8, 8), f'{stage}: {metrics["mae_mm"]:.1f} mm | {fit["reason"]}', fill='white')
                tiles.append(tile)
                if stage in ('translation', 'rigid', 'pose'):
                    scene_meshes.append((arrays['pred_vertices']+arrays['pred_cam_t'])[::3])
                if do_publish:
                    publish(sample, directory, source/profile/f"pose-{row['pose']:03d}"/'prediction', profile, stage,
                            inference, metrics, root/'report.html',Path(row['mask']))
            contact = Image.new('RGB', (960, 600))
            for i, tile in enumerate(tiles):
                contact.paste(tile, ((i % 2)*480, (i//2)*300))
            contact.save(folder/'stages.png')
            with np.load(sample/'rgbd.npz') as cloud:
                observed = cloud['xyz_m'][cv2.imread(row['mask'], 0) > 0]
            observed = observed[np.isfinite(observed).all(1) & (observed[:, 2] > .05)][::5]
            camera_planes(folder/'camera_planes.png', observed, scene_meshes)
            rel = folder.relative_to(root).as_posix()
            table = ''.join('<tr>'+''.join(f'<td>{html.escape(str(r[k]))}</td>' for k in ('stage', 'mae_mm', 'point_surface_mae_mm', 'silhouette_iou', 'reason'))+'</tr>' for r in pose_records)
            sections.append(f'<details class="pose" data-profile="{profile}"><summary>{profile} · 姿态 {row["pose"]} · {pose_records[0]["group"]}</summary><table><tr><th>阶段</th><th>深度 MAE mm</th><th>点面 MAE mm</th><th>IoU</th><th>接受/回退原因</th></tr>{table}</table><img loading="lazy" src="{rel}/stages.png"><img loading="lazy" src="{rel}/camera_planes.png"><p>橙色为实测表面；蓝色仅平移，紫色整体朝向，绿色身体姿态。三者使用同一相机坐标，没有单独对齐。</p><a href="{rel}/pose/fit.json">姿态参数变化与完整拟合记录</a></details>')
    summaries = []
    for profile in ('standard512', 'fast512', 'fast384'):
        for group in ('all', 'validation'):
            for stage in STAGES:
                items = [r for r in records if r['profile'] == profile and r['stage'] == stage and (group == 'all' or r['group'] == group)]
                if not items:
                    continue
                result = dict(profile=profile, group=group, stage=stage, count=len(items), accepted=sum(r['accepted'] for r in items))
                for key in ('mae_mm', 'point_surface_mae_mm', 'silhouette_iou', 'postprocess_ms', 'guard_capped_mae_mm'):
                    result[key] = float(np.mean([r[key] for r in items]))
                summaries.append(result)
    save(root/'summary.json', dict(summary=summaries, records=records))
    with (root/'per-pose.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    table = ''.join(f'<tr><td>{r["profile"]}</td><td>{r["group"]} ({r["count"]})</td><td>{STAGES[r["stage"]]}</td><td>{r["accepted"]}</td><td>{r["mae_mm"]:.2f}</td><td>{r["point_surface_mae_mm"]:.2f}</td><td>{r["silhouette_iou"]:.3f}</td><td>{r["postprocess_ms"]:.0f}</td></tr>' for r in summaries)
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RGB-D 受约束姿态拟合实验</title><style>body{background:#0b1220;color:#e2e8f0;font:18px/1.65 system-ui;max-width:1300px;margin:32px auto;padding:20px}td,th{padding:10px;border-bottom:1px solid #334155;text-align:left}table{border-collapse:collapse;width:100%;font-size:15px}details{background:#172033;padding:18px;margin:12px 0;border-radius:10px}summary{cursor:pointer}img{max-width:100%;display:block;margin:16px auto}p{color:#c0cede}a{color:#74bfff}select{font:inherit;padding:8px;background:#172033;color:white;border:1px solid #526079;border-radius:6px}.scroll{overflow:auto}</style><h1>RGB-D：从位置到身体姿态</h1><p>使用同一批冻结的网络输出，依次验证：仅平移 → 固定尺度 → 整体朝向 → 身体姿态。保留全部回退项，逐姿态等权统计。原始预测、RGB、深度、内参与掩膜不覆盖。</p><p>尺度仅由姿态 1–3 估计一个每配置共用的米制乘数，证据不足时使用 1；其余 9 姿态不参与尺度校准。整体朝向调整 3 个根旋转参数；姿态阶段开放根旋转与 22 个身体参数。固定手指、表情、身份形状和肢体尺度系数。每阶段根欧拉参数每分量 ±5°（两阶段累计可达 ±10°）、身体参数每个 ±12°、平移每轴 ±50 mm；RGB 投影先验防止过拟合深度。</p><p>仅可见表面参与匹配。深度按空间块 80% 拟合 / 20% 接受检查，检查点不进入目标点云或法线。每阶段固定尝试完整、1/2、1/4 更新，通过误差、轮廓与覆盖率检查才应用，否则保留上阶段。检查点用于选择，因此不能称为独立测试集。validation 仅表示不参与尺度校准，仍做逐图姿态拟合。</p><p><strong>本实验验证衣物表面与模型的一致性，不能证明真实关节角度更准确。</strong>关节变化量只是预测修改幅度。不是实时性能测试，表中的 CPU 后处理时间不含加载、导出及精确三角面评估，且是各阶段独立耗时，完整流程需要累加。仅平移复用 V1 输出，此列时间仅为基线复核；网络时间沿用第一阶段记录。</p><h2>逐姿态等权平均（含回退）</h2><div class="scroll"><table><tr><th>配置</th><th>范围</th><th>阶段</th><th>接受数</th><th>深度 MAE mm</th><th>点面 MAE mm</th><th>IoU</th><th>阶段后处理 ms</th></tr>'''+table+'''</table></div><h2>逐图对照</h2><label>配置筛选 <select id="profile"><option value="">全部</option><option>standard512</option><option>fast512</option><option>fast384</option></select></label>'''+''.join(sections)+'''<script>document.getElementById('profile').onchange=function(){document.querySelectorAll('.pose').forEach(e=>e.hidden=this.value&&e.dataset.profile!==this.value)}</script></html>'''
    (root/'report.html').write_text(page)
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True); parser.add_argument('--source', required=True); parser.add_argument('--masks', required=True)
    parser.add_argument('--publish', action='store_true'); args = parser.parse_args()
    build(Path(args.experiment).resolve(), Path(args.source).resolve(), args.masks, args.publish)
