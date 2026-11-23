# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Build per-pose raw/refined evidence and optionally publish copies into Studio."""
import argparse
import base64
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',required=True);p.add_argument('--masks',required=True);p.add_argument('--pose-set');p.add_argument('--publish',action='store_true');p.add_argument('--masks-reviewed',action='store_true');args=p.parse_args()
    if args.publish and not args.masks_reviewed:raise ValueError('review the segmentation contact sheet before publication')
    session=None
    if args.pose_set:
        from studio.rgbd.rgbd_batch import PoseSet
        from studio.rgbd.recorder_session import recording_active
        if recording_active(args.pose_set):raise ValueError('recording session active')
        session=PoseSet(args.pose_set)
    root=Path(args.experiment).resolve();masks=json.loads(Path(args.masks).read_text());records=[];sections=[]
    def embedded(file):return 'data:image/png;base64,'+base64.b64encode(Path(file).read_bytes()).decode()
    for name in ('standard512','fast512','fast384'):
        for row in masks:
            folder=root/name/f"pose-{row['pose']:03d}";file=folder/'evaluation'/'metrics.json'
            if not file.is_file():continue
            raw=json.loads(file.read_text());ref=raw['depth_refinement'];after=ref['metrics']
            inference=json.loads((folder/'prediction'/'inference.json').read_text())
            record=dict(profile=name,pose=row['pose'],accepted=ref['accepted'],reason=ref['reason'],inference_ms=inference['benchmark']['mean_ms'],
                solver_ms=ref.get('solver_seconds',0)*1000,refinement_ms=ref['total_seconds']*1000,
                delta_x_mm=ref['delta_translation_m'][0]*1000,delta_y_mm=ref['delta_translation_m'][1]*1000,delta_z_mm=ref['delta_translation_m'][2]*1000)
            for key in ('mae_mm','p90_mm','silhouette_iou','model_coverage_fraction','point_surface_mae_mm'):
                record['raw_'+key]=raw[key];record['refined_'+key]=after[key]
            records.append(record)
            # Same RGB and error scale, contours for location only (not a texture render).
            rgb=cv2.imread(str(Path(row['sample'])/'rgb.png'));raw_z=np.load(folder/'evaluation'/'depth_comparison.npz')['rendered_m'];new_z=np.load(folder/'evaluation'/'refined_depth.npz')['rendered_m']
            tiles=[]
            for label,z,color in [('RAW',raw_z,(255,180,100)),('DEPTH',new_z,(80,240,120))]:
                image=rgb.copy();contours,_=cv2.findContours(np.uint8(z>.05)*255,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image,contours,-1,color,2);tile=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB));tile.thumbnail((480,270));ImageDraw.Draw(tile).text((8,8),label,fill='white');tiles.append(tile)
            for heat in ('depth_error.png','depth_error_refined.png'):
                tile=Image.open(folder/'evaluation'/heat);tile.thumbnail((480,270));tiles.append(tile)
            contact=Image.new('RGB',(960,540))
            for i,tile in enumerate(tiles):contact.paste(tile,((i%2)*480,(i//2)*270))
            contact.save(folder/'comparison.png')
            sections.append(f'<details><summary>{name} · 姿态 {row["pose"]} · MAE {raw["mae_mm"]:.1f} → {after["mae_mm"]:.1f} mm · {ref["reason"]}</summary><p>上：原始 / 已应用位置轮廓；下：原始 / 已应用深度误差，±50 mm 同色标。回退时右侧与原始一致。</p><img src="{embedded(folder/"comparison.png")}"/><pre>{html.escape(json.dumps(record,ensure_ascii=False,indent=2))}</pre></details>')
            if row['pose'] in (1,6,12):
                with np.load(Path(row['sample'])/'rgbd.npz') as cloud:
                    observed=cloud['xyz_m'][cv2.imread(str(row['mask']),0)>0]
                observed=observed[np.isfinite(observed).all(1)&(observed[:,2]>.05)][::5]
                with np.load(folder/'prediction'/'person_00.npz') as pred:vertices=pred['pred_vertices']+pred['pred_cam_t'].reshape(1,3)
                corrected=vertices+np.array(ref['delta_translation_m']);scene=Image.new('RGB',(1200,520),'#0b1220');draw=ImageDraw.Draw(scene)
                for panel,(a,b,label) in enumerate([(0,1,'XY / image plane'),(0,2,'XZ / top view'),(2,1,'ZY / side view')]):
                    stacks=[points[:,[a,b]] for points in (observed,vertices,corrected)];all_points=np.concatenate(stacks);lo=all_points.min(0);hi=all_points.max(0);scale=min(350/max(hi[0]-lo[0],.01),425/max(hi[1]-lo[1],.01))
                    for points,color in zip(stacks,['#f6ad55','#68b7ff','#42e68a']):
                        xy=(points-lo)*scale+[panel*400+25,50]
                        draw.point([tuple(p) for p in xy],fill=color)
                    draw.text((panel*400+25,10),label+' / metres',fill='white')
                draw.text((25,490),'ORANGE: measured surface | BLUE: raw | GREEN: applied depth correction (same pose / shape)',fill='white')
                scene.save(folder/'camera_planes.png');sections.append(f'<details><summary>{name} · 姿态 {row["pose"]} · 相机系三视图</summary><p>固定米制尺度正交投影：橙色为实测表面，蓝色为原始网格采样点，绿色为已应用校正。仅用于观察位置与外形差异。</p><img src="{embedded(folder/"camera_planes.png")}"/></details>')
            if args.publish:
                sample=Path(row['sample']);mask=Path(row['mask']);target=sample/'mask.png'
                if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()!=row['mask_sha256']:raise ValueError('current sample has another mask; refusing overwrite')
                if not target.exists():shutil.copy2(mask,target)
                source=dict(row);source['visually_reviewed']=True;source['review_scope']='12-pose RGB contact sheet; one full-body primary person, no manual ground truth';(sample/'mask_source.json').write_text(json.dumps(source,indent=2))
                label='depth-v1-'+name;pred=sample/'predictions'/label;ev=sample/'evaluations'/label
                if not pred.exists():shutil.copytree(folder/'prediction',pred);shutil.copy2(folder/'request.json',pred/'request.json')
                if not ev.exists():shutil.copytree(folder/'evaluation',ev)
                r=dict(label=label,config=inference['actual_config'],prediction=str(pred/'person_00.npz'),evaluation=str(ev),metrics=raw,
                    inference_ms=inference['benchmark']['mean_ms'],rgb_sha256=row['rgb_sha256'],person_bbox_xyxy=inference['bbox_xyxy'],
                    input_selection='mask_bbox',crop_note='same independent SAM3 person-mask bounding ROI; Fast retains its YOLO detected crop',experiment=str(root))
                if session:
                    index=row['pose']-1
                    if session.sample_path(session.poses[index])!=sample.resolve():raise ValueError('pose-set sample mismatch')
                    r.update(batch_config_id=inference['config_id'],batch_input=session.fingerprint(index))
                    run_id='depth-v1-'+name
                    if not any(item.get('id')==run_id for item in session.data['runs']):
                        session.data['runs'].append(dict(id=run_id,kind='reconstruct',status='complete',config_id=inference['config_id'],config=inference['actual_config'],selection='mask_bbox',items=[],origin=str(root)))
                previous=json.loads((sample/'results.json').read_text()) if (sample/'results.json').exists() else []
                previous=[item for item in previous if item.get('label')!=label]+[r]
                temp=sample/'results.tmp';temp.write_text(json.dumps(previous,ensure_ascii=False,indent=2));temp.replace(sample/'results.json')
    if session and args.publish:session.save();session.export_csv(session.directory)
    summary={}
    for name in ('standard512','fast512','fast384'):
        group=[r for r in records if r['profile']==name]
        if not group:continue
        summary[name]=dict(poses=len(group),accepted=sum(r['accepted'] for r in group))
        for key in group[0]:
            if key not in ('pose','accepted') and type(group[0][key]) in (int,float):summary[name][key]=float(np.mean([r[key] for r in group]))
    (root/'summary.json').write_text(json.dumps(dict(summary=summary,records=records,meaning='equal-weight per-pose means; acceptance guard is not independent anatomical test'),indent=2))
    with (root/'per-pose.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    table=''
    for name,s in summary.items():
        table+=f'<tr><td>{name}</td><td>{s["accepted"]}/{s["poses"]}</td><td>{s["raw_mae_mm"]:.2f} → {s["refined_mae_mm"]:.2f}</td><td>{s["raw_point_surface_mae_mm"]:.2f} → {s["refined_point_surface_mae_mm"]:.2f}</td><td>{s["inference_ms"]:.1f}</td><td>{s["solver_ms"]:.1f} / {s["refinement_ms"]:.1f}</td></tr>'
    page='''<!doctype html><meta charset="utf-8"><title>RGB-D 深度位置校正实验</title><style>body{background:#0b1220;color:#e2e8f0;font:18px/1.65 system-ui;max-width:1300px;margin:40px auto;padding:20px}td,th{padding:12px;border-bottom:1px solid #334155;text-align:left}table{border-collapse:collapse;width:100%}details{background:#172033;padding:18px;margin:12px 0;border-radius:10px}summary{cursor:pointer}img{max-width:100%}pre{white-space:pre-wrap;font-size:14px}p{color:#b7c8db}</style><h1>12 姿态 × 3 方案：深度位置校正</h1><p>冻结网络输出，仅读取实测 RGB-D 与内参，拟合整体 ΔXYZ；不改变关节姿态、体形、尺度和旋转。原始 NPZ 始终保留。SAM3 先单独分割并退出，随后顺序运行人体模型。没有用深度掩膜提示网络，隔离后处理效果。</p><p>指标为逐姿态等权平均，回退结果也计入；不是只汇报成功项。自动掩膜已检查主体，但不是人工真值。衣物表面、传感器噪声、遮挡和形状误差仍存在。空间块 80% 拟合 / 20% 接受检查；该检查集用于回退决策，不能视为独立测试集。</p><table><tr><th>方案</th><th>接受</th><th>深度 MAE mm</th><th>点面 MAE mm</th><th>网络 ms</th><th>求解 / 校正总 ms</th></tr>'''+table+'</table><p>网络：每姿态额外预热 2 次、测量 5 次，CUDA 同步，排除加载、解码、导出和渲染。校正总耗时包含点处理、法线、求解及一次候选深度渲染，排除原始渲染、精确三角面评估和磁盘导出；当前离线 EGL 实现不是实时性能证明。Standard 使用掩膜外接框，Fast 在相同选择 ROI 内仍使用 YOLO 实际裁剪；因此主要证据是同一预测前后的配对变化，跨后端比较包含检测差异。</p><h2>逐姿态证据</h2>'+''.join(sections)
    (root/'report.html').write_text(page)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
