#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Report measured 384 INT8 results, optionally render calibrated overlays."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('experiment',type=Path)
    p.add_argument('--render',action='store_true')
    a=p.parse_args();root=a.experiment
    summary=json.loads((root/'summary.json').read_text())
    manifest=json.loads((root/'manifest.json').read_text())
    reference=Path(summary['reference'])
    old=json.loads((reference/'summary.json').read_text())
    names={'fast384':'Fast FP16 384','int8384':'Fast INT8 384'}
    text=['# 本地 INT8 384 实验','',
          '从 Embedl 512 静态 ONNX 改图：修改 7 个形状常量，重新生成 32 份 24×24 位置角度表，保留全部权重、Q/DQ 和量化尺度。',
          '输入 [1,3,384,384]，输出 [1,1280,24,24]。位置表恢复源文件中的 BF16 周期值，先在 512 上逐表重建验证，再计算 384 网格；没有插值旧角度。',
          '这不是发布者提供或重新标定的 384 量化模型，仍需独立评估。','',
          '## 完整推理时间','',
          'RTX 4060 Laptop 8GB，正式 worker，YOLO 每次运行；3 次反馈，解码/MHR/中间预测编译，反馈缓存、关键点专用路径、最终图重放及 CPU IO 优化开启。',
          '各 worker 启动预热 10 次，各姿态再预热 3 次、计时 20 次，共 240 次测量。CUDA 同步，包含检测/裁剪/网络/CPU 输出，不含磁盘读写/导出/显示。',
          '两个 384 方案为本轮顺序复测；512 列引用前轮同数据同框结果，仅供参照，未做四方案本轮交替测速。','',
          '| 方案 | 测量批次 | 平均 ms | P95 ms |','|---|---|---:|---:|']
    for label in ['fast384','int8384']:
        t=summary['timings'][label];text.append(f'| {names[label]} | 本轮 | {t["mean_ms"]:.2f} | {t["p95_ms"]:.2f} |')
    for label,name in [('int8512','Fast INT8 512'),('std512','Standard512')]:
        t=old['timings'][label];text.append(f'| {name} | 前轮参照 | {t["mean_ms"]:.2f} | {t["p95_ms"]:.2f} |')
    text+=['','## 相对 Standard512 的输出差异','',
           '12 张相同 RGB 与相机 K；YOLO 实际框逐张检查与前轮一致。Standard512 是冻结的网络预测参照，不是真值。',
           '骨盆对齐仅移除整体平移，不拟合旋转或尺度；相机系指标不做对齐。','',
           '| 平均差异 | FP16 384 | INT8 384 | INT8 512（前轮） |','|---|---:|---:|---:|']
    for key,title in [('mesh_pelvis_aligned_mean_mm','对齐网格 mm'),('joints127_pelvis_aligned_mean_mm','对齐127关节 mm'),
                      ('arm_keypoints_pelvis_aligned_mean_mm','对齐双臂关键点 mm'),('camera_mesh_mean_mm','相机系网格 mm'),
                      ('camera_pelvis_difference_mm','相机系骨盆 mm'),('camera_pelvis_z_difference_mm','骨盆深度有符号差 mm'),
                      ('elbow_mean_difference_deg','双肘角度 °')]:
        values=[summary['aggregates'][label][key] for label in ['fast384','int8384']]+[old['aggregates']['int8512'][key]]
        text.append('| '+title+' | '+' | '.join(f'{value:.3f}' for value in values)+' |')
    text+=['','## 与 FP16 384 直接对比','']
    delta=summary['aggregates']['int8384_vs_fp16384']
    text.append(f'平均骨盆对齐网格距离 {delta["mesh_pelvis_aligned_mean_mm"]:.3f} mm；双肘角度差 {delta["elbow_mean_difference_deg"]:.3f}°；相机系骨盆距离 {delta["camera_pelvis_difference_mm"]:.3f} mm。')
    text+=['','逐姿态指标及所有计时见 `summary.json`，原始输出见 `fast384/pose-*` 与 `int8384/pose-*`。',
           '未验证实时相机端到端帧率、深度真值误差或连续动作抖动；INT8 384 保留降分辨率与量化两种误差来源。','']
    (root/'REPORT.md').write_text('\n'.join(text))
    if not a.render:return
    from studio_paths import environment,path
    from studio.core.gpu_lease import acquire_gpu_lease
    lease=acquire_gpu_lease('exclusive');os.environ.update(environment('fast'))
    sys.path.insert(0,str(path('fast_repo')))
    import cv2
    from studio.body.calibrated_visualization import render_mesh
    out=root/'visuals';out.mkdir(exist_ok=True);montage=[]
    for case in manifest['cases']:
        image=cv2.imread(case['image']);k=case['intrinsics']
        matrix=np.array([[k['fx'],0,k['cx']],[0,k['fy'],k['cy']],[0,0,1]],np.float32)
        panels=[('RGB',image)]
        for folder,name in [(reference/'std512','Standard512'),(root/'fast384','FP16 384'),
                             (root/'int8384','INT8 384'),(reference/'int8512','INT8 512')]:
            with np.load(folder/f'pose-{case["id"]:02d}/person_00.npz',allow_pickle=False) as d:
                panels.append((name,render_mesh(image,d['pred_vertices'],d['pred_cam_t'],d['faces'],matrix)))
        tiles=[]
        for name,panel in panels:
            tile=cv2.resize(panel,(400,250));bar=np.full((38,400,3),22,np.uint8)
            cv2.putText(bar,f'Pose {case["id"]:02d} {name}',(8,26),cv2.FONT_HERSHEY_SIMPLEX,.55,(240,240,240),1,cv2.LINE_AA)
            tiles.append(np.concatenate([bar,tile]))
        combined=np.concatenate(tiles,axis=1)
        assert cv2.imwrite(str(out/f'pose-{case["id"]:02d}.jpg'),combined)
        if case['id'] in [1,4,11,12]:montage.append(combined)
    assert cv2.imwrite(str(out/'comparison.jpg'),np.concatenate(montage))
    print('REPORT_AND_VISUALS_OK',out,flush=True)


if __name__=='__main__':main()
