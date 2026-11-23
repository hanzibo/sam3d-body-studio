# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Build a portable report from completed frozen-input speed benchmarks."""
import argparse
import csv
import html
import json
from pathlib import Path
import cv2
import numpy as np
from studio.rgbd.depth_pose_experiment import inputs, load_arrays
from studio.rgbd.rgbd_evaluate import DepthRenderer
from studio.rgbd.rgbd_geometry import depth_metrics


def esc(value):
    return html.escape(str(value))


def build(root, pipeline=None):
    root = Path(root).resolve()
    if not (root/'complete.json').is_file():
        raise ValueError('benchmark incomplete; refusing a final report')
    records = json.loads((root/'records.json').read_text())
    summary = json.loads((root/'summary.json').read_text())
    comparisons = []
    for row in records:
        baseline = next(r for r in records if r['profile'] == row['profile'] and r['pose'] == row['pose'] and r['variant'] == 'cpu60')
        comparisons.append(dict(profile=row['profile'], pose=row['pose'], variant=row['variant'],
                                mae_delta_mm=row['mae_mm']-baseline['mae_mm'],
                                iou_delta=row['silhouette_iou']-baseline['silhouette_iou'],
                                solver_ms=row['solver_ms'], path_ms=row['warm_path_ms']))
    def table(rows):
        head = '<tr><th>配置</th><th>算法</th><th>求解均值 / P95 ms</th><th>后处理均值 / P95 ms</th><th>深度 MAE mm</th><th>轮廓 IoU</th><th>接受数</th><th>网络＋后处理估算 ms</th></tr>'
        cells = []
        for r in rows:
            cells.append('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in [r['profile'], r['variant'],
                f"{r['solver_ms_mean']:.1f} / {r['solver_ms_p95']:.1f}",
                f"{r['warm_path_ms_mean']:.1f} / {r['warm_path_ms_p95']:.1f}",
                f"{r['mae_mm_mean']:.2f}", f"{r['silhouette_iou_mean']:.3f}",
                f"{r['accepted']}/{r['count']}", f"{r['estimated_network_plus_path_ms']:.1f}"] )+'</tr>')
        return '<table>'+head+''.join(cells)+'</table>'
    extra = ''
    if pipeline:
        p = Path(pipeline).resolve()
        if not (p/'complete.json').is_file():
            raise ValueError('pipeline benchmark incomplete')
        extra = '<h2>完整 rigid＋pose 两阶段</h2><p>从 v1 平移结果开始。下表后处理包含 fresh 深度法线/树、rigid、pose、各阶段原有接受检查；仍不含 v1 平移和原始 SAM 推理。</p>'+table(json.loads((p/'summary.json').read_text()))
    quality = []
    for s in summary:
        delta = [r for r in comparisons if r['profile'] == s['profile'] and r['variant'] == s['variant']]
        quality.append('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in [s['profile'], s['variant'],
            f"{np.mean([r['mae_delta_mm'] for r in delta]):+.3f}",
            f"{max(r['mae_delta_mm'] for r in delta):+.3f}",
            sum(r['mae_delta_mm'] > 2 for r in delta)])+'</tr>')
    # Same captured RGB and same camera: no per-model alignment or recropping.
    evidence = []; repo = Path(__file__).resolve().parents[2]
    masks = json.loads((repo/'outputs/depth-fusion/masks/masks.json').read_text())
    profiles = sorted({r['profile'] for r in records})
    for profile in profiles:
        worst = sorted([r for r in comparisons if r['profile'] == profile and r['variant'] == 'cuda15'],
                       key=lambda r: r['mae_delta_mm'], reverse=True)
        if not worst:
            continue
        pose = worst[0]['pose']; row = next(r for r in masks if r['pose'] == pose)
        sample, _, arrays, z, xyz, k, mask = inputs(row, repo/'outputs/depth-fusion/experiment-v1', profile)
        renderer = DepthRenderer(arrays['faces'], k)
        tiles = []
        try:
            for variant in ('cpu60', 'cuda60', 'cuda30', 'cuda15', 'cuda10'):
                item = next((r for r in records if (r['profile'], r['pose'], r['variant']) == (profile, pose, variant)), None)
                if item is None:
                    continue
                path = root/profile/f'pose-{pose:03d}'/variant
                a = load_arrays(path/'mesh.npz'); rendered = renderer.render(a['vertices']+a['translation'])
                rgb = cv2.imread(str(sample/'rgb.png')); overlay = rgb.copy()
                visible = rendered > .05
                overlay[visible] = (.35*rgb[visible]+.65*np.array([210, 215, 225])).astype('uint8')
                image = cv2.resize(overlay, (448, 252))
                cv2.rectangle(image, (0, 0), (448, 27), (20, 20, 20), -1)
                cv2.putText(image, f"{variant} / {item['mae_mm']:.2f}mm", (8, 19), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1)
                tiles.append(image)
            dest = root/f'comparison-{profile}-pose-{pose:03d}.jpg'
            cv2.imwrite(str(dest), np.concatenate(tiles, axis=1))
            evidence.append(f'<h3>{esc(profile)} / 姿态 {pose}：15 步相对 CPU60 退化最大的样本</h3><img src="{esc(dest.name)}">')
        finally:
            renderer.close()
    page = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>单帧深度拟合加速基线</title>
<style>body{background:#0b1220;color:#d9e4f2;font:17px/1.7 sans-serif;margin:30px;max-width:1800px}table{border-collapse:collapse;width:100%;font-size:15px}td,th{border:1px solid #344155;padding:9px;text-align:left}th{background:#1c3047}h1,h2{color:#80c7ff}img{width:100%;height:auto}a{color:#80c7ff}.note{padding:16px;border-left:4px solid #e8aa55;background:#1b2434}code{color:#b9e9bd}</style>
<h1>SAM 3D Body：单帧深度拟合加速基线</h1>
<p class="note">这是静态 RGB-D、冻结预测上的离线测量，不是相机端到端 FPS。主表所有方案从同一个已完成 rigid 优化的结果开始；不能把 pose-only 耗时当成完整后处理。预热、图构建、磁盘读写、GUI 与精确三角面评估不计入热帧；每帧重新估计深度法线和树，不缓存同一照片预处理来美化延迟。</p>
<h2>同初值 pose 阶段对照</h2>'''+table(summary)+extra+'''
<h2>质量相对 CPU60 的差异</h2><p>平均值包含所有姿态及回退。检查点用于接受判定，不是独立真值。数值浮点和对应点边界会使 CUDA60 与 CPU60 有细微差别；少步数属于算法预算改变。</p>
<table><tr><th>配置</th><th>算法</th><th>平均 MAE 差 mm</th><th>最差单图 MAE 差 mm</th><th>退化超过 2mm 的图数</th></tr>'''+''.join(quality)+'''</table>
<h2>按规则选取的最差样本</h2><p>以下选择 cuda15 相对 CPU60 的最大误差增加项，避免只展示好看的结果。覆盖图仅用于肉眼参考；量化指标由原始全分辨率渲染器评估。</p>'''+''.join(evidence)+'''
<h2>计时与复现范围</h2><ul><li>cpu60：原始 CPU MHR、原始渲染、60 步 Adam。</li><li>cuda60：缓存固定体形与表情，CUDA Graph，60 步相同目标函数、边界与接受阈值，深度专用持久缓冲渲染。</li><li>cuda30/15/10：同上，但减少迭代次数。Standard 保留姿态形变修正；Fast 使用其原预测匹配的修正模式。</li><li>均值：每张图重复计时的中位数，再对姿态等权平均；P95：所有热帧重复值的合并分位数。</li><li>网络＋后处理为不同实验结果相加的估算，且未包括 v1 平移、采集、图像传输或显示，不是完整实测延迟。</li></ul>
<p><a href="summary.json">汇总 JSON</a> · <a href="records.json">逐图与阶段记录</a> · <a href="quality-deltas.json">逐图质量差</a> · <a href="cpu-profile.txt">CPU 计时剖析</a> · <a href="parity.json">深度渲染一致性</a> · <a href="cold-setup.json">冷启动图构建</a> · <a href="manifest.json">环境与来源指纹</a></p></html>'''
    (root/'report.html').write_text(page)
    (root/'quality-deltas.json').write_text(json.dumps(comparisons, indent=2))
    with (root/'summary.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    print(root/'report.html')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('experiment'); p.add_argument('--pipeline')
    a = p.parse_args(); build(a.experiment, a.pipeline)
