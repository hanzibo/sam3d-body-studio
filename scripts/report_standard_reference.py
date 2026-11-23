#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Numerical and image comparisons against Standard512, not ground truth."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from studio.body.pose_data import joint_angles


def compare_prediction(reference, candidate):
    """Correspondence distances; remove pelvis translation only, no rigid/scale fit."""
    points={name:np.asarray(d['pred_keypoints_3d'],np.float64) for name,d in [('r',reference),('c',candidate)]}
    pelvis={name:p[[9,10]].mean(0) for name,p in points.items()}
    delta=pelvis['c']-pelvis['r']
    rv=np.asarray(reference['pred_vertices'],np.float64)
    cv=np.asarray(candidate['pred_vertices'],np.float64)
    rc=np.asarray(reference['pred_cam_t'],np.float64).reshape(3)
    cc=np.asarray(candidate['pred_cam_t'],np.float64).reshape(3)
    mesh=np.linalg.norm(cv-rv-delta,axis=1)*1000
    joints=np.linalg.norm(np.asarray(candidate['pred_joint_coords'],np.float64)-reference['pred_joint_coords']-delta,axis=1)*1000
    landmarks=np.linalg.norm(points['c']-points['r']-delta,axis=1)*1000
    arms=landmarks[[5,6,7,8,41,62]]
    a,b=joint_angles(points['r']),joint_angles(points['c'])
    angles={k:abs(a[k]-b[k]) for k in a if a[k] is not None and b[k] is not None}
    camera_mesh=np.linalg.norm(cv+cc-rv-rc,axis=1)*1000
    return dict(mesh_pelvis_aligned_mean_mm=float(mesh.mean()),mesh_pelvis_aligned_p95_mm=float(np.percentile(mesh,95)),
        joints127_pelvis_aligned_mean_mm=float(joints.mean()),keypoints70_pelvis_aligned_mean_mm=float(landmarks.mean()),
        arm_keypoints_pelvis_aligned_mean_mm=float(arms.mean()),camera_mesh_mean_mm=float(camera_mesh.mean()),
        camera_translation_difference_mm=float(np.linalg.norm(cc-rc)*1000),
        camera_pelvis_difference_mm=float(np.linalg.norm(pelvis['c']+cc-pelvis['r']-rc)*1000),
        camera_pelvis_z_difference_mm=float((pelvis['c']+cc-pelvis['r']-rc)[2]*1000),
        elbow_mean_difference_deg=float(np.mean([angles[k] for k in ['left_elbow','right_elbow']])),
        angles_difference_deg=angles)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('experiment',type=Path);p.add_argument('--render',action='store_true')
    a=p.parse_args();root=a.experiment
    manifest=json.loads((root/'manifest.json').read_text())
    labels=['std512','fast384','int8512'];names=['Standard 512','Fast FP16 384','Fast INT8 512']
    timings={};per_case=[]
    for label in labels:
        rows=json.loads((root/label/'cases.json').read_text())
        values=np.array([s*1000 for row in rows for s in row['benchmark']['pipeline_seconds']])
        timings[label]=dict(n=len(values),mean_ms=float(values.mean()),median_ms=float(np.median(values)),p95_ms=float(np.percentile(values,95)))
    for case in manifest['cases']:
        arrays={}
        for label in labels:
            with np.load(root/label/f'pose-{case["id"]:02d}/person_00.npz',allow_pickle=False) as d:
                arrays[label]={k:d[k].copy() for k in ['pred_vertices','pred_keypoints_3d','pred_joint_coords','pred_cam_t','faces']}
        for label in labels[1:]:
            assert np.array_equal(arrays['std512']['faces'],arrays[label]['faces'])
            metrics=compare_prediction(arrays['std512'],arrays[label])
            per_case.append(dict(id=case['id'],variant=label,**metrics))
    aggregates={}
    for label in labels[1:]:
        rows=[x for x in per_case if x['variant']==label]
        aggregates[label]={k:float(np.mean([x[k] for x in rows])) for k in rows[0] if k not in ['id','variant','angles_difference_deg']}
        aggregates[label]['worst_pose_mesh_aligned_mean_mm']=max(x['mesh_pelvis_aligned_mean_mm'] for x in rows)
        aggregates[label]['worst_pose_elbow_delta_deg']=max(v for x in rows for k,v in x['angles_difference_deg'].items() if 'elbow' in k)
    counts={}
    for metric in ['mesh_pelvis_aligned_mean_mm','joints127_pelvis_aligned_mean_mm','camera_mesh_mean_mm','elbow_mean_difference_deg']:
        counts[metric]=sum(per_case[2*i+1][metric]<per_case[2*i][metric] for i in range(len(manifest['cases'])))
    summary=dict(reference='Studio Standard512 Comfy BF16 body mode; not ground truth',timings=timings,
        aggregates=aggregates,int8_closer_pose_counts=counts,cases=per_case,protocol=manifest['scope'])
    (root/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    with (root/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        keys=[k for k in per_case[0] if k!='angles_difference_deg']
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        writer.writerows({k:row[k] for k in keys} for row in per_case)
    text=['# Fast384 与 INT8 512：以 Standard512 为参照','',
        '12 个已录制 Gemini 全身姿态；Fast 每次执行 YOLO，两种 Fast 的实际检测框须逐姿态一致，Standard 使用这些检测框；三方案使用相同 RGB 与标定内参。无掩膜提示。',
        '均为身体模式。Fast 保留三次反馈及所有现有加速；关闭 SMPL、滤波、深度细化。Standard 为 Studio 的 Comfy BF16 512 路径。',
        '逐一加载三个正式 worker，各预热 10 次，每个姿态再预热 3 次、测量 20 次，总计每方案 240 次。',
        'Fast 计时包含每次 YOLO、裁剪、网络和 CPU 输出；Standard 消费给定框、不含检测。三者都不含加载、文件读写、导出和渲染。顺序运行，未做 ABBA 交替。',
        'Fast 每个 worker：10 次启动预热＋12×(3次预热＋20次测量)=286 次 YOLO，计数逐姿态验收。','',
        '| 方案 | 平均 ms | 中位数 ms | P95 ms |','|---|---:|---:|---:|']
    for label,name in zip(labels,names):
        t=timings[label];text.append(f'| {name} | {t["mean_ms"]:.2f} | {t["median_ms"]:.2f} | {t["p95_ms"]:.2f} |')
    text+=['','下表是相对 Standard 的距离，不是真实人体误差。骨盆对齐只去除整体平移，保留朝向、尺寸和姿态差异；没有旋转／尺度拟合。','',
        '| 相对 Standard512 的平均差异 | Fast384 | INT8 512 |','|---|---:|---:|']
    for metric,title in [('mesh_pelvis_aligned_mean_mm','骨盆对齐后网格 mm'),('joints127_pelvis_aligned_mean_mm','骨盆对齐后127关节 mm'),
        ('arm_keypoints_pelvis_aligned_mean_mm','骨盆对齐后双臂6关键点 mm'),('camera_mesh_mean_mm','相机系网格 mm'),
        ('camera_pelvis_difference_mm','相机系骨盆位置 mm'),('camera_pelvis_z_difference_mm','骨盆深度有符号差 mm（正为更远）'),
        ('elbow_mean_difference_deg','双肘几何夹角 °')]:
        text.append(f'| {title} | {aggregates["fast384"][metric]:.3f} | {aggregates["int8512"][metric]:.3f} |')
    text+=['',f'INT8 512 的骨盆对齐网格距离在 {counts["mesh_pelvis_aligned_mean_mm"]}/12 个姿态中更接近 Standard。',
        '不代表每个姿态、每个关节都会改善，也没有验证连续动作抖动。Standard 本身不是光学动捕或人体真值。','',
        '| 姿态 | Fast384 网格 mm | INT8 网格 mm | Fast384 肘角 ° | INT8 肘角 ° |','|---|---:|---:|---:|---:|']
    for index,case in enumerate(manifest['cases']):
        r,s=per_case[2*index:2*index+2]
        text.append(f'| {case["name"]} | {r["mesh_pelvis_aligned_mean_mm"]:.2f} | {s["mesh_pelvis_aligned_mean_mm"]:.2f} | {r["elbow_mean_difference_deg"]:.2f} | {s["elbow_mean_difference_deg"]:.2f} |')
    text+=['','`manifest.json` 保存输入哈希、共同框、内参；各方案目录保存配置、worker 日志、原始网格与逐次计时。',
           '逐姿态差异见 `comparison.csv` 与 `summary.json`。图像对照使用同一标定 K，不通过重新平移对齐修饰投影。','']
    (root/'REPORT.md').write_text('\n'.join(text))
    print(json.dumps({k:v for k,v in summary.items() if k not in ['cases']},ensure_ascii=False,indent=2),flush=True)
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
        panels=[image]
        for label in labels:
            with np.load(root/label/f'pose-{case["id"]:02d}/person_00.npz',allow_pickle=False) as d:
                panels.append(render_mesh(image,d['pred_vertices'],d['pred_cam_t'],d['faces'],matrix))
        tiles=[]
        for name,panel in zip(['RGB',*names],panels):
            tile=cv2.resize(panel,(400,225));bar=np.full((38,400,3),22,np.uint8)
            cv2.putText(bar,f'Pose {case["id"]:02d}  {name}',(10,26),cv2.FONT_HERSHEY_SIMPLEX,.55,(240,240,240),1,cv2.LINE_AA)
            tiles.append(np.concatenate([bar,tile]))
        combined=np.concatenate(tiles,axis=1)
        assert cv2.imwrite(str(out/f'pose-{case["id"]:02d}.jpg'),combined)
        if case['id'] in [1,4,11,12]:montage.append(combined)
    assert cv2.imwrite(str(out/'comparison.jpg'),np.concatenate(montage,axis=0))
    print('VISUALS_OK',out,flush=True)


if __name__=='__main__':main()
