# SPDX-License-Identifier: GPL-3.0-only
"""Summarize final-surface paired timing, parity and complete-sequence checks."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from studio.hand.hand_sequence_metrics import summarize
p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args();root=a.directory
paired=json.loads((root/'paired/paired.json').read_text());rows=[]
for batch in (1,2):
 cases=[c for c in paired['cases'] if c['hands']==batch]
 old=np.mean([c['stats']['baseline']['inference_ms']['mean'] for c in cases]);new=np.mean([c['stats']['compact']['inference_ms']['mean'] for c in cases])
 rows.append(f'| {batch} 手 | {old:.3f} | {new:.3f} | {old-new:.3f} | {(old-new)/old*100:.2f}% |')
validation={};seqrows=[]
for d in sorted((root/'full').iterdir()):
 info=json.loads((d/'sequence.json').read_text());assert info['complete'] and info['frames']==450 and 'video_error' not in info,info
 frames=[json.loads(l) for l in (d/'frames.jsonl').read_text().splitlines()];assert len(frames)==450
 max_roundtrip=0.;hand_count=0
 for r in frames:
  for h in r['report']['hand_reports']:
   assert h['raw_vertex_layout']=='hand_only' and h['vertices']==2321 and h['triangles']==4613
   assert h['feedback_runtime']['last']['full_mhr_calls']==0
   assert h['feedback_runtime']['last']['compact_surface_calls']==1
   assert h['feedback_runtime']['last']['intermediate_keypoint_calls']==5
   assert h['projection_roundtrip_max_px']<.05
   max_roundtrip=max(max_roundtrip,h['projection_roundtrip_max_px']);hand_count+=1
  with np.load(d/r['scene'],allow_pickle=False) as data:
   for key in data.files:
    if np.issubdtype(data[key].dtype,np.number):assert np.isfinite(data[key]).all(),(d,r['index'],key)
 metrics=summarize(d)['groups_by_detected_hand_count']
 for count,item in metrics.items():
  if item['measured_frames']:
   seqrows.append(f'| {d.name} | {count} | {item["measured_frames"]} | {item["reconstruct_ms"]["mean"]:.2f} | {item["detect_ms"]["mean"]:.2f} | {item["frame_with_io_ms"]["mean"]:.2f} |')
 validation[d.name]=dict(frames=len(frames),hand_instances=hand_count,projection_roundtrip_max_px=max_roundtrip,all_arrays_finite=True,all_compact=True,steady_full_surface_calls=0)
(root/'sequence-validation.json').write_text(json.dumps(validation,indent=2))
diffs=json.loads((root/'comparison/full-output-differences.json').read_text())
text='''# 最终手部顶点专用路径验证\n\n## 方法\n\n基线为此前 TRT FP16 512 编码器 + FP32 解码器/MHR 编译与 Graph + 反馈复制优化 + 中间关键点专用路径。候选只增加最终手部表面计算。检测器不变，六次预测、五次反馈保留。五个固定手框案例每种方案预热四次，随后同进程正反顺序交替各 30 次。网络时间包括裁剪、H2D、网络和同步，不含检测、D2H、磁盘和渲染。\n\n## 配对计时\n\n| 批量 | 完整表面 ms | 手部表面 ms | 节省 ms | 降幅 |\n|---|---:|---:|---:|---:|\n'''+ '\n'.join(rows)
text+='\n\n## 数值一致性\n\n'
for label in ('camera_vertices_mm','local_vertices_mm','joints_mm','projection_px','global_rotations','mhr_params'):
 maximum=max(c['differences']['compact'][label]['max'] for c in paired['cases']);text+=f'- `{label}` 最大绝对/点距离差：{maximum:.10g}。\n'
text+='''\n这是与已有输出的数值一致性，不是真实手部三维精度验证。稳态表面为 2321 顶点、4613 三角形；完整 127 节点 FK、形状、表情、姿态修正保留。仅裁剪姿态修正末层输出行、形状/表情基和蒙皮影响列表。首次预热为校验会调用原始完整 MHR。\n\n## 完整序列\n\n每个批量剔除前三次出现；以下重建耗时不含检测，IO 耗时含检测与压缩读写、视频渲染另计。\n\n| 序列 | 手数 | 样本 | 重建 ms | 检测 ms | 帧含 IO ms |\n|---|---:|---:|---:|---:|---:|\n'''+ '\n'.join(seqrows)
text+='\n\n输出一致性（与上一轮完整序列比较）：\n\n```json\n'+json.dumps(diffs,indent=2)+'\n```\n'
text+='\n\nGUI：手部实验 → 采用 512 加速方案，包含“最终仅生成手部网格”；关闭该项可回到上一轮输出。raw NPZ 顶点由 18439 行改为 2321 行，需读取 pred_vertex_ids；展示/导出用手部网格拓扑不变。\n'
(root/'REPORT.md').write_text(text);print(text[:1600]);print('SEQUENCE_VALIDATION_OK',json.dumps(validation))
