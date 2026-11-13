#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Summarize the paired INT8 experiment and render saved mesh comparisons."""
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
    a=p.parse_args()
    r=json.loads((a.experiment/'full/full.json').read_text())
    enc=json.loads((a.experiment/'encoder/encoder.json').read_text())
    from pose_data import joint_angles
    rows=[]
    for i,c in enumerate(r['checks']):
        pair={name:np.load(a.experiment/f'full/case-{i:02d}-{name}.npz',allow_pickle=False) for name in ['fp16','int8']}
        angles={name:joint_angles(d['p0_pred_keypoints_3d']) for name,d in pair.items()}
        camera=np.linalg.norm(pair['int8']['p0_pred_cam_t']-pair['fp16']['p0_pred_cam_t'])*1000
        rows.append(dict(case=i,image=c['image'],mesh_mean_mm=c['differences']['pred_vertices']['mean_l2']*1000,
            mesh_max_mm=c['differences']['pred_vertices']['max_l2']*1000,
            skeleton_mean_mm=c['differences']['pred_joint_coords']['mean_l2']*1000,
            camera_translation_l2_mm=float(camera),camera_space_mesh_mean_mm=c['camera_space_mesh_delta_mm']['mean'],
            angle_delta_deg={k:abs(angles['int8'][k]-angles['fp16'][k]) for k in angles['fp16']},
            projection_mean_px=c['differences']['pred_keypoints_2d']['mean_abs']))
    summary=dict(encoder=enc['timing'],full=r['timing'],cases=rows,
        accuracy_scope='Difference against FP16 predictions, not ground-truth error; default image-diagonal focal length; no SAM3 mask, no depth refinement, no temporal smoothing',
        mesh_mean_mm=float(np.mean([x['mesh_mean_mm'] for x in rows])),
        mesh_max_mm=max(x['mesh_max_mm'] for x in rows),
        camera_space_mesh_mean_mm=float(np.mean([x['camera_space_mesh_mean_mm'] for x in rows])),
        camera_translation_l2_mean_mm=float(np.mean([x['camera_translation_l2_mm'] for x in rows])),
        camera_translation_l2_max_mm=max(x['camera_translation_l2_mm'] for x in rows),
        elbow_angle_delta_mean_deg=float(np.mean([x['angle_delta_deg'][k] for x in rows for k in ['left_elbow','right_elbow']])),
        elbow_angle_delta_max_deg=max(x['angle_delta_deg'][k] for x in rows for k in ['left_elbow','right_elbow']))
    (a.experiment/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    text=['# Embedl INT8 编码器实测','', '2026-11-09 · RTX 4060 Laptop 8GB · TensorRT 10.16.1 · Fast512身体模式','',
          '双方保留YOLO TensorRT、FP32解码器/MHR编译、三次中间反馈、反馈缓存、MHR70图重放、最终图重放及身体IO优化。',
          '各预热10次，ABBA/BAAB交替正式测量各240次。完整计时包括检测、裁剪、网络及CPU输出，排除加载、编译、文件读写和渲染。','',
          '| 阶段 | FP16均值 | INT8均值 | 节省 | 降幅 |','|---|---:|---:|---:|---:|']
    for label,t in [('编码器',enc['timing']),('完整推理',r['timing'])]:
        old,new=t['fp16']['mean_ms'],t['int8']['mean_ms']
        text.append(f'| {label} | {old:.3f} ms | {new:.3f} ms | {old-new:.3f} ms | {(1-new/old)*100:.2f}% |')
    text += ['',f"13张输入的对应顶点平均距离 {summary['mesh_mean_mm']:.3f} mm，最远顶点 {summary['mesh_max_mm']:.3f} mm。",
             f"相机平移三维距离平均 {summary['camera_translation_l2_mean_mm']:.3f} mm，最大 {summary['camera_translation_l2_max_mm']:.3f} mm。",
             f"双肘几何夹角差平均 {summary['elbow_angle_delta_mean_deg']:.3f}°，最大 {summary['elbow_angle_delta_max_deg']:.3f}°。",
             '上述均为相对FP16的预测差异，不能称为真实人体精度损失；没有用深度真值验收。',
             '尚未验证连续流抖动、真实相机帧率、SAM3掩膜提示、显式相机内参或手部精修；仅支持此次固定512单人实验。','',
             '| 输入 | 网格平均差 mm | 骨架平均差 mm | 相机平移差 mm | 左肘差 ° | 右肘差 ° |',
             '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        text.append(f"| {'历史单图' if row['case']==0 else '姿态'+str(row['case'])} | {row['mesh_mean_mm']:.3f} | {row['skeleton_mean_mm']:.3f} | {row['camera_translation_l2_mm']:.3f} | {row['angle_delta_deg']['left_elbow']:.3f} | {row['angle_delta_deg']['right_elbow']:.3f} |")
    text += ['', '原始数据：`encoder/encoder.json`、`full/full.json`、`full/case-*.npz`。',
             '来源与低内存构建：`build/build.json`、`build/prepare-fp16.log`、`build/build-half.log`。',
             '原始FP32外部权重3.37GB。先折叠128个恒定权重量化节点，全部通过ONNX参考实现逐值检查；再按作者路径将剩余浮点权重及IO转FP16，保留激活Q/DQ节点。',
             '原始构建曾发生全局OOM；随后6GiB受限构建触及上限中止。最终采用强类型FP16/INT8构建、7GiB物理内存上限；构建中允许最多4GiB已有交换空间。没有修改全局交换配置。',
             '正式Studio配置和旧FP16引擎保持不变；所有大文件位于Data盘。','']
    (a.experiment/'REPORT.md').write_text('\n'.join(text))
    print(json.dumps({k:v for k,v in summary.items() if k!='cases'},ensure_ascii=False,indent=2),flush=True)
    if not a.render:return
    from studio_paths import environment,path
    from gpu_lease import acquire_gpu_lease
    lease=acquire_gpu_lease('exclusive')
    os.environ.update(environment('fast'))
    sys.path.insert(0,str(path('fast_repo')))
    import cv2
    from sam_3d_body.visualization.renderer import Renderer
    # Use verified existing MHR topology rather than loading model weights again.
    faces=np.load(ROOT/'outputs/release-validation/20261108-194307-628158-fast/prediction/person_00.npz',allow_pickle=False)['faces']
    assert faces.shape==(36874,3) and faces.max()<18439
    np.save(a.experiment/'full/faces.npy',faces)
    out=a.experiment/'visuals';out.mkdir(exist_ok=True)
    montage=[]
    for row in rows:
        i=row['case'];image=cv2.imread(row['image']);h,w=image.shape[:2]
        panels=[image]
        for name in ['fp16','int8']:
            d=np.load(a.experiment/f'full/case-{i:02d}-{name}.npz',allow_pickle=False)
            renderer=Renderer(focal_length=float(np.hypot(h,w)),faces=faces)
            overlay=renderer(d['p0_pred_vertices'],d['p0_pred_cam_t'],image.copy(),mesh_base_color=(.65,.74,.86),scene_bg_color=(1,1,1))
            panels.append(np.clip(overlay*255,0,255).astype(np.uint8))
        tiles=[]
        for name,panel in zip(['RGB','FP16','INT8 mixed'],panels):
            tile=cv2.resize(panel,(480,round(h*480/w)))
            bar=np.full((42,480,3),22,np.uint8)
            cv2.putText(bar,f'Case {i:02d}  {name}',(12,27),cv2.FONT_HERSHEY_SIMPLEX,.65,(240,240,240),1,cv2.LINE_AA)
            tiles.append(np.concatenate([bar,tile]))
        combined=np.concatenate(tiles,axis=1)
        assert cv2.imwrite(str(out/f'case-{i:02d}.jpg'),combined)
        if i in [1,4,11,12]:montage.append(combined)
    assert cv2.imwrite(str(out/'comparison.jpg'),np.concatenate(montage,axis=0))
    print('VISUALS_OK',out,flush=True)


if __name__=='__main__':main()
