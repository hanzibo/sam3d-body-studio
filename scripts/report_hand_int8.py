# SPDX-License-Identifier: GPL-3.0-only
"""Combine paired hand encoder timing and full-sequence quantization evidence."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from studio.hand.hand_sequence_metrics import summarize
p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);a=p.parse_args();root=a.directory
paired=json.loads((root/'paired/paired.json').read_text());quality=json.loads((root/'comparison/hand-quality.json').read_text());differences=json.loads((root/'comparison/full-output-differences.json').read_text())
lines=['# 手部 INT8 512 验证','', '基线：FP16 TRT 512 编码器、WiLoR FP16 TRT 640 检测器、FP32 解码器/MHR 编译 + CUDA Graph、反馈复制优化、中间关键点路径、最终手部表面。候选仅替换为既有 Embedl INT8/FP16 混合精度 512 编码器；没有重新量化/校准。六次预测和五次反馈不变。','', '## 引擎接口','', 'INT8 引擎固定 B=1，输入 `[1,3,512,512]`，输出 `[1,1280,32,32]`，IO FP16；双手分别编码后合为 B=2 送解码器。FP16 引擎原生支持 B=2。没有覆盖或新建引擎。','', '## 同进程配对计时','', '六个固定手框案例，每个方案预热 4 次，然后正反顺序交替各 30 次。完整推理含裁剪、H2D、网络、同步，不含检测、D2H、IO 和渲染；GPU 子项嵌套且包含 CPU 派发间隙。','', '| 批量 | FP16 推理 ms | INT8 推理 ms | 节省 | 降幅 | FP16 编码 ms | INT8 编码 ms |','|---|---:|---:|---:|---:|---:|---:|']
aggregate={}
for n in (1,2):
 cs=[c for c in paired['cases'] if c['hands']==n];stat={mode:{k:float(np.mean([c['stats'][mode][k]['mean'] for c in cs])) for k in ('inference_ms','backbone_gpu_ms','hand_decoder_including_mhr_gpu_ms')} for mode in ('fp16','int8')};aggregate[str(n)]=stat
 f,i=stat['fp16'],stat['int8'];old,new=f['inference_ms'],i['inference_ms'];lines.append(f"| {n} 手 | {old:.2f} | {new:.2f} | {old-new:.2f} ms | {(old-new)/old*100:.1f}% | {f['backbone_gpu_ms']:.2f} | {i['backbone_gpu_ms']:.2f} |")
lines+=['','## 完整序列与质量差异','','每段 450 帧，合计 900 帧。与上一轮 FP16 完整序列相同检测器、相同框与标签配对。下列是方案间差异，没有真实手部三维真值，不能将差异等同精度误差。','','| 序列 | 相机顶点均差 / 最大 mm | 手腕对齐顶点均差 mm | 投影均差 / 最大 px | 关节全局旋转均差 ° |','|---|---:|---:|---:|---:|']
for name,q in quality['sequences'].items():
 d=differences['sequences'][name];assert d['frames_with_different_side_sets']==0 and d['box_iou']['mean']==1
 v=d['camera_vertices_mm'];px=d['projection_px'];qd=q['differences']
 lines.append(f"| {name} | {v['mean']:.2f} / {v['max']:.2f} | {qd['wrist_aligned_vertices_mm']['mean']:.2f} | {px['mean']:.2f} / {px['max']:.2f} | {qd['global_rotation_angle_deg']['mean']:.2f} |")
lines+=['','旋转均差按每个手部关节全局旋转矩阵的相对旋转角统计，含手腕朝向，不等同纯手指屈伸误差。手腕对齐仅移除平移，不移除旋转或体型差。','','## 连续变化代理','','二阶差分单位 mm/source-frame²；30 FPS 源帧上统计连续存在的同侧手部，包含真实动作变化和噪声，只作为相对稳定性线索，不能直接称作抖动真值。','','| 序列 | FP16 手腕二阶均值 / P95 mm | INT8 手腕二阶均值 / P95 mm | FP16 局部关节二阶均值 / P95 mm | INT8 局部关节二阶均值 / P95 mm |','|---|---:|---:|---:|---:|']
for name,q in quality['sequences'].items():
 vals=[]
 for mode,key in [('fp16','wrist_second_difference_mm'),('int8','wrist_second_difference_mm'),('fp16','local_joint_second_difference_mm'),('int8','local_joint_second_difference_mm')]:
  x=q['temporal'][mode][key];vals.append(f"{x['mean']:.3f} / {x['p95']:.3f}")
 lines.append('| '+name+' | '+' | '.join(vals)+' |')
lines+=['','## 完整序列预热后计时','','每种手数剔除前三次出现；同一段内逐帧重建不是配对交替计时，跨次运行受 GPU 状态影响。检测耗时约另加 10 ms，视频渲染另计。','','| 序列 | 手数 | INT8 重建 ms | 检测 ms | 含读写整帧 ms |','|---|---:|---:|---:|---:|']
full={}
for d in sorted((root/'full').iterdir()):
 info=json.loads((d/'sequence.json').read_text());assert info['complete'] and info['frames']==450 and 'video_error' not in info
 metrics=summarize(d);full[d.name]=metrics
 rows=[json.loads(l) for l in (d/'frames.jsonl').read_text().splitlines()]
 for row in rows:
  for h in row['report']['hand_reports']:
   assert h['backbone_engine_executions']==row['report']['hand_count']
   assert h['execution']['backbone']=='trt_int8' and h['feedback_runtime']['last']['compact_surface_calls']==1
 for n,x in metrics['groups_by_detected_hand_count'].items():
  if x['measured_frames']:lines.append(f"| {d.name} | {n} | {x['reconstruct_ms']['mean']:.2f} | {x['detect_ms']['mean']:.2f} | {x['frame_with_io_ms']['mean']:.2f} |")
lines+=['','## 使用及对照','','Studio 重启 → 手部实验 → 试用 INT8 512 → 预加载/预热。也可从编码器下拉选择实验项。FP16 加速预设仍保留；加载后锁定配置，需先释放模型才能切换。','', 'INT8 有真实数值变化，因此作为可选实验方案，不将本轮速度收益解释为无精度损失。','']
for name in quality['sequences']:
 file=root/'videos'/(name+'-fp16-vs-int8.mp4');lines.append(f'- {name}：[FP16 左 / INT8 右并排视频]({file})。')
lines+=['','视频按源 30 FPS 播放，不能将播放速度当成实际推理帧率。原始视频、基线、候选 NPZ/JSONL、逐帧 CSV 与统计均保留。']
if (root/'validation.json').is_file():
 validation=json.loads((root/'validation.json').read_text())
 lines+=['','## GUI 与检查','',f"{validation['cpu_tests']} 项 CPU 测试通过；900 帧完整序列通过。Xvfb 下实际 GPU GUI 的预设切换、模型加载、单/双手预热、检测、重建、短序列导出和回放通过。短 GUI 代表帧双手重建 {validation['gui_representative_reconstruction_ms']:.2f} ms、检测 {validation['gui_representative_detection_ms']:.2f} ms，不替代配对计时或现场相机帧率。两段视频已全文件解码。"]
(root/'REPORT.md').write_text('\n'.join(lines)+'\n');(root/'summary.json').write_text(json.dumps(dict(baseline=str(a.baseline),paired=aggregate,quality=quality,full_timings=full),indent=2));print('\n'.join(lines[:20]))
