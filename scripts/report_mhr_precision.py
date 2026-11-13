#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Report MHR precision stages without confusing local GPU time with end-to-end speed."""
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
    geometry=json.loads((root/'geometry/geometry.json').read_text())
    full=json.loads((root/'full/full.json').read_text())
    names=dict(fp32='MHR FP32',final_mixed='仅最终混合精度',all_mixed='中间＋最终混合精度',all_fp16='中间＋最终全 FP16')
    aggregates={}
    for mode,rows in full['differences'].items():
        if not rows:continue
        keys=['mesh_pelvis_aligned_mean_mm','camera_mesh_mean_mm','joints127_pelvis_aligned_mean_mm',
              'camera_pelvis_difference_mm','elbow_mean_difference_deg','pixel_mean','rotation_max_abs']
        aggregates[mode]={k:float(np.mean([row[k] for row in rows])) for k in keys}
        aggregates[mode]['pixel_max']=max(row['pixel_max'] for row in rows)
        aggregates[mode]['worst_pose_mesh_mean_mm']=max(row['camera_mesh_mean_mm'] for row in rows)
        worst_vertex=0.;worst_joint=0.;orthogonality=0.;worst_rotation=0.
        for row in rows:
            folder=f'pose-{row["pose"]:02d}'
            with np.load(root/'full/fp32'/folder/'person_00.npz',allow_pickle=False) as r,np.load(root/'full'/mode/folder/'person_00.npz',allow_pickle=False) as c:
                worst_vertex=max(worst_vertex,float(np.linalg.norm((c['pred_vertices']+c['pred_cam_t'])-(r['pred_vertices']+r['pred_cam_t']),axis=-1).max()*1000))
                worst_joint=max(worst_joint,float(np.linalg.norm((c['pred_joint_coords']+c['pred_cam_t'])-(r['pred_joint_coords']+r['pred_cam_t']),axis=-1).max()*1000))
                rotations=c['pred_global_rots'].astype(np.float64)
                orthogonality=max(orthogonality,float(np.abs(rotations@rotations.transpose(0,2,1)-np.eye(3)).max()))
                worst_rotation=max(worst_rotation,float(np.abs(rotations-r['pred_global_rots']).max()))
        aggregates[mode].update(worst_camera_vertex_mm=worst_vertex,worst_camera_joint_mm=worst_joint,
                                rotation_orthogonality_max_abs=orthogonality,worst_rotation_element_delta=worst_rotation)
    summary=dict(timings=full['timings'],aggregates=aggregates,errors=full['errors'])
    repeat=root/'fp32-repeat/full.json'
    if repeat.is_file():summary['fp32_repeat']=json.loads(repeat.read_text())['timings']['fp32']
    (root/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    text=['# MHR 精度逐步实验','',
          '在现有 FP32 TorchScript 权重上进行浮点转换，未重新训练、未 INT8 标定、未修改上游和原始权重。','',
          '## 方案与范围','',
          '- FP32：原有 MHR 几何、127 骨架和关键点路径。',
          '- 仅最终混合精度：最终网格的体型/表情混合与蒙皮 FP16；FK、参数映射、关键点回归及输出 FP32。中间反馈保持原样。',
          '- 中间＋最终混合精度：中间468顶点关键点专用路径也使用上述混合策略；FK及回归保持 FP32。',
          '- 中间＋最终全 FP16：两条 MHR 几何路径的浮点资产与输入转 FP16，包括 FK；返回网络接口时转回 FP32。',
          '“全 FP16”仅指 MHR 几何内部，不包含 519 维参数预测 head、6D旋转转换、解码器和相机 head，这些保持 FP32。',
          '所有方案禁用姿态修正网络。索引/父子关系/面片等整数数据保留原整数类型。','',
          '## 固定参数的几何测试','',
          '相同12组真实预测参数；单独捕获几何 CUDA Graph，预热后每组20次，各240次；不包含网络/检测/文件/渲染。顺序测速，局部差异不能直接视作整帧收益。','',
          '| 几何阶段 | 平均 ms | 几何位置平均差 mm | 最大差 mm |','|---|---:|---:|---:|']
    for label,title in [('final_fp32','最终 FP32'),('final_mixed','最终混合'),('mid_fp32','中间 FP32'),('mid_mixed','中间混合'),('final_fp16','最终 FP16'),('mid_fp16','中间 FP16')]:
        row=geometry[label]
        if not row['ok']:
            text.append(f'| {title} | 失败 | — | — |');continue
        diffs=[v for v in row['differences'] if v['output']==0]
        mean=np.mean([v['mean_mm'] for v in diffs]) if diffs else 0
        worst=max(v['max_mm'] for v in diffs) if diffs else 0
        text.append(f'| {title} | {row["mean_ms"]:.4f} | {mean:.3f} | {worst:.3f} |')
    text+=['','最终阶段的几何位置指标是18439顶点；中间阶段是70关键点。混合方案固定相同参数时，127骨架坐标与FP32完全一致。','',
           '## 完整推理','',
           'RTX4060 Laptop 8GB；INT8 384 编码器、YOLO TensorRT 每次执行、FP32解码器、3次反馈和所有现有推理优化。每方案单独加载worker，启动预热10次、各姿态再预热3次、正式20次，共240次。',
           'CUDA同步；包含检测/裁剪/网络/CPU输出，不含加载/编译/磁盘读写/导出/显示。方案顺序运行，未做同进程ABBA；不足1ms的差异不宜直接解释为收益。','',
           '| 方案 | 平均 ms | P95 ms |','|---|---:|---:|']
    for mode,row in full['timings'].items():text.append(f'| {names[mode]} | {row["mean_ms"]:.3f} | {row["p95_ms"]:.3f} |')
    if 'fp32_repeat' in summary:
        row=summary['fp32_repeat'];text.append(f'| FP32 末轮复测 | {row["mean_ms"]:.3f} | {row["p95_ms"]:.3f} |')
    text+=['','本轮未观察到混合或全FP16的完整推理加速，保留FP32默认。最终几何局部仅省约0.17ms，中间几何没有变快；不能据此宣称完整网络加速。']
    text+=['','## 相对本轮 FP32 的完整输出差异','',
           '这里衡量数值变化，不是真实人体精度。中间关键点变化会反馈至后续解码器，因此可能引起最终姿态和相机位置变化。','',
           '| 方案 | 对齐网格 mm | 相机系网格 mm | 对齐127关节 mm | 双肘角 ° | 投影均值 px | 最远顶点 mm |','|---|---:|---:|---:|---:|---:|---:|']
    for mode,row in aggregates.items():
        keys=['mesh_pelvis_aligned_mean_mm','camera_mesh_mean_mm','joints127_pelvis_aligned_mean_mm','elbow_mean_difference_deg','pixel_mean','worst_camera_vertex_mm']
        text.append('| '+names[mode]+' | '+' | '.join(f'{row[k]:.4f}' for k in keys)+' |')
    text+=['','全FP16骨架输出的旋转矩阵正交性也有所下降：本轮最大 `abs(R @ R.T - I)` 为 '+f'{aggregates.get("all_fp16",{}).get("rotation_orthogonality_max_abs",float("nan")):.6f}'+
           '；混合方案保持FP32骨架，约为4e-7。原head直接将四元数转换为矩阵，本实验未额外归一化来改变原计算语义。']
    if full['errors']:text+=['','失败项：'+json.dumps(full['errors'],ensure_ascii=False)]
    text+=['','## 使用与限制','',
           '默认继续使用 FP32。诊断环境变量 `BODYSTUDIO_MHR_PRECISION` 可取 `fp32/final_mixed/all_mixed/all_fp16`；未设置时行为不变。',
           '精度适配在编译与图捕获之前安装。实验仅支持身体模式、无correctives、已启用中间关键点专用路径；切换后必须重启worker重新预热。',
           '本轮不增加GUI选项，避免将未见明确整帧收益的实验误当作推荐启动方案。',
           '配置ID仍对应原LaunchConfig；实验精度单独记录在ready及inference.json的execution.mhr_precision字段，不应只按配置ID合并这些实验记录。',
           '本轮未验收实际相机端到端帧率或连续动作抖动。','']
    (root/'REPORT.md').write_text('\n'.join(text))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    if not a.render:return
    from studio_paths import environment,path
    from gpu_lease import acquire_gpu_lease
    lease=acquire_gpu_lease('exclusive');os.environ.update(environment('fast'))
    sys.path.insert(0,str(path('fast_repo')))
    import cv2
    from calibrated_visualization import render_mesh
    manifest=json.loads((root/'full/manifest.json').read_text())
    out=root/'visuals';out.mkdir(exist_ok=True);montage=[]
    labels=[k for k in names if k in full['timings']]
    for case in manifest['cases']:
        image=cv2.imread(case['image']);k=case['intrinsics']
        matrix=np.array([[k['fx'],0,k['cx']],[0,k['fy'],k['cy']],[0,0,1]],np.float32)
        panels=[('RGB',image)]
        for mode in labels:
            with np.load(root/'full'/mode/f'pose-{case["id"]:02d}/person_00.npz',allow_pickle=False) as d:
                panels.append((mode,render_mesh(image,d['pred_vertices'],d['pred_cam_t'],d['faces'],matrix)))
        tiles=[]
        for name,panel in panels:
            tile=cv2.resize(panel,(400,250));bar=np.full((38,400,3),22,np.uint8)
            cv2.putText(bar,f'Pose {case["id"]:02d} {name}',(8,26),cv2.FONT_HERSHEY_SIMPLEX,.55,(240,240,240),1,cv2.LINE_AA)
            tiles.append(np.concatenate([bar,tile]))
        combined=np.concatenate(tiles,axis=1)
        assert cv2.imwrite(str(out/f'pose-{case["id"]:02d}.jpg'),combined)
        if case['id'] in [1,4,11,12]:montage.append(combined)
    assert cv2.imwrite(str(out/'comparison.jpg'),np.concatenate(montage))
    print('VISUALS_OK',out,flush=True)


if __name__=='__main__':main()
