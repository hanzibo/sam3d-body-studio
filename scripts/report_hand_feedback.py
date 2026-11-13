# SPDX-License-Identifier: GPL-3.0-only
"""Render measured hand feedback ablations and full-sequence validation."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--baseline-full',type=Path,required=True);p.add_argument('--full-directory',default='full');a=p.parse_args();base=a.directory
paired=json.loads((base/'paired/paired.json').read_text());names=('baseline','feedback','keypoints','combined')
lines=['# 手部反馈与中间关键点路径加速验证','','## 同进程交替对照','','4 个固定手框代表帧，每种模式预热 4 次，20 轮按正序/逆序交替测试。编码器 TRT FP16 512；解码器/MHR FP32，保留编译和 Graph。baseline 为上一轮 512 加速方案。以下计时含裁剪、H2D、网络、MHR、同步，不含检测、D2H、导出及渲染。','','| 模式 | 单手重建 ms | 双手重建 ms | 单手解码含 MHR ms | 双手解码含 MHR ms |','|---|---:|---:|---:|---:|']
summary={}
for name in names:
 values=[]
 for key in ('inference_ms','hand_decoder_including_mhr_gpu_ms'):
  for n in (1,2):values.append(float(np.mean([c['stats'][name][key]['mean'] for c in paired['cases'] if c['hands']==n])))
 summary[name]=dict(zip(('single_ms','dual_ms','single_decoder_ms','dual_decoder_ms'),values))
 lines.append(f"| {name} | "+' | '.join(f'{v:.2f}' for v in values)+' |')
lines+=['','feedback=合并反馈写入/减少 clone，并编译反馈计算；keypoints=中间预测仅算实际依赖的关键点；combined=两项组合。所有模式保留 6 次预测、5 次反馈及最终完整 MHR。实际每层姿态不同，反馈缓存命中为 0；不将收益说成重复姿态缓存复用。','','## 固定框输出差异','','相对同进程 baseline；不是三维真值误差。','','| 模式 | 相机系顶点最大差 mm | 关节最大差 mm | 投影最大差 px | 旋转矩阵分量最大差 | MHR 参数分量最大差 |','|---|---:|---:|---:|---:|---:|']
for name in names[1:]:
 metrics=[max(c['differences'][name][key]['max'] for c in paired['cases']) for key in ('camera_vertices_mm','joints_mm','projection_px','global_rotations','mhr_params')]
 lines.append(f'| {name} | '+' | '.join(f'{v:.8f}' for v in metrics)+' |')
lines+=['','GPU 首次批量检查：与原始 MHR 的关键点输出 assert_close(rtol=1e-5, atol=1e-6 米)。当前权重的右手 21 个关键点依赖 21 个骨架节点、0 个表面顶点。中间保持全部 127 节点 FK，只省掉对反馈无贡献的表面计算和全顶点投影；最终网格及姿态修正保持原路径。','']
if (base/'profile/profile.json').is_file() and (base/'profile-after/profile.json').is_file():
 before=json.loads((base/'profile/profile.json').read_text())['batches'];after=json.loads((base/'profile-after/profile.json').read_text())['batches']
 lines+=['## 代表帧计时剖析','','同一双手帧预热后测量 20 次。CUDA stream 区间包含 CPU 派发间隙，下面是嵌套统计，不能相加；有额外插桩开销，以交替对照表衡量最终收益。','','| 双手阶段 | 上一轮 ms | 本轮 ms | 每帧调用次数 |','|---|---:|---:|---:|']
 for key,label in [('pose_callback','姿态预测回调总计'),('mhr_total','其中：MHR 参数及几何'),('projection','其中：投影'),('feedback','关键点反馈')]:
  b=before['2']['components'][key];c=after['2']['components'][key]
  lines.append(f"| {label} | {b['mean_ms']:.2f} | {c['mean_ms']:.2f} | {c['calls']:.0f} |")
 lines+=['']
lines+=['## 连续 900 帧','','每种手数排除最初 3 次。检测器仍逐帧执行，模型、阈值和后处理相同。两次完整序列非交替运行，频率/温度可能影响结果。','','| 序列/手数 | 帧数 | 原方案重建 ms | 新方案重建 ms | 检测 ms | 检测+重建 ms | 含压缩读写 ms |','|---|---:|---:|---:|---:|---:|---:|']
videos=[]
for d in sorted((base/a.full_directory).glob('*')):
 file=d/'sequence.json';metrics=d/'timing-groups.json';old=a.baseline_full/d.name/'timing-groups.json'
 if not file.is_file() or not metrics.is_file() or not old.is_file():continue
 info=json.loads(file.read_text())
 if not info.get('complete'):continue
 before=json.loads(old.read_text())['groups_by_detected_hand_count'];after=json.loads(metrics.read_text())['groups_by_detected_hand_count']
 for count,c in after.items():
  if not c['measured_frames']:continue
  recon=c['reconstruct_ms']['mean'];detect=c['detect_ms']['mean']
  lines.append(f"| {d.name}/{count} | {c['frames']} | {before[count]['reconstruct_ms']['mean']:.2f} | {recon:.2f} | {detect:.2f} | {recon+detect:.2f} | {c['frame_with_io_ms']['mean']:.2f} |")
 videos.append(f'- {d.name}：[叠加视频]({a.full_directory}/{d.name}/overlay.mp4) · [逐帧计时]({a.full_directory}/{d.name}/timings.csv)')
lines+=['']+videos+['']
diff=base/'full-output-differences.json'
if diff.is_file():
 r=json.loads(diff.read_text());summary['full_output_differences']=r
 lines+=['## 全序列数值对照','','同一检测器与设置，按帧及左右手标签配对；仍不是实际三维误差。','','| 序列 | 左右集合变化帧数 | 手框平均 IoU | 相机系顶点平均差 mm | 最大差 mm | 投影最大差 px |','|---|---:|---:|---:|---:|---:|']
 for name,c in r['sequences'].items():
  lines.append(f"| {name} | {c['frames_with_different_side_sets']} | {c['box_iou']['mean']:.6f} | {c['camera_vertices_mm']['mean']:.6f} | {c['camera_vertices_mm']['max']:.6f} | {c['projection_px']['max']:.6f} |")
validation=base/'sequence-validation.json'
if validation.is_file():
 r=json.loads(validation.read_text());events=[dict(sequence=name,**event) for name,v in r.items() for event in v['undefined_nonhand_camera_plane_projections']]
 lines+=['','## 全序列检查和投影边界','', '900 帧均检查：5 次中间关键点路径、1 次完整 MHR；全部手部输出有限、投影回验通过。两个视频各 450 帧、960×540、30 FPS。详见 sequence-validation.json。','', '首次完整运行在第二组 source_index=324 发生原生全网格投影非有限值中止，未纳入正式计时。单帧复跑未复现；独立完整重跑确认该帧右手分支的非手部顶点 11479 落在相机平面（绝对深度 ≤1e-6 米）。只允许这类无关顶点投影无定义并记录 valid mask；全部 3D、相机和手部投影仍严格检查。未对像素值做替换，最终完整人体表面计算保留。', '', f'正式重跑记录的边界事件：`{json.dumps(events,ensure_ascii=False)}`。原始中止日志 full-worker.log、排查日志 frame324-debug.log 保留。','']
lines+=['','## 使用和边界','','Studio → 手部实验 → 采用 512 加速方案 → 预加载 → 预热。可单独取消“反馈复制优化”或“中间预测仅计算手部关键点”进行对照；方案加载后锁定，须释放后再改。恢复原生基线会关闭所有加速项。','', '这轮没有 INT8、没有 384、没有跳过手框检测，也没有减少反馈次数。测试使用离线录制，不等于相机实时流吞吐。视频按录制帧率播放，不是推理帧率。','', '原始录制和上一轮结果保留，新结果位于 Data 盘独立目录。GUI、回放验证日志同目录保存。']
(base/'REPORT.md').write_text('\n'.join(lines)+'\n');(base/'summary.json').write_text(json.dumps(summary,indent=2));print(base/'REPORT.md')
