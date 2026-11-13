# SPDX-License-Identifier: GPL-3.0-only
"""CPU report of fixed-box numeric parity and full automatic sequence timings."""
import argparse,json,sys
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--baseline-full',type=Path);a=p.parse_args();base=a.directory
benchmarks={}
for variant in ('native','trt','decoder','all','reduced','tf32'):
 f=base/variant/'benchmark.json'
 if f.is_file():benchmarks[variant]=json.loads(f.read_text())
lines=['# 手部 512 推理加速验证','', '固定手框对照：同一照片、同一手框/左右标签、同一相机内参；4 个代表帧，每帧预热 3 次后重复 10 次。网络计时含裁剪、H2D、编码器、手部解码器、MHR、同步，不含检测、导出或渲染。','', '| 配置 | 单手均值 ms | 双手均值 ms | 相机空间顶点最大差 mm | 投影最大差 px |','|---|---:|---:|---:|---:|']
metrics={}
for name,b in benchmarks.items():
 diffs=[];pixels=[]
 if name!='native':
  for case in b['cases']:
   with np.load(base/'native'/(case['case']+'.npz')) as ref,np.load(base/name/(case['case']+'.npz')) as candidate:
    for i in range(case['hands']):
     left=candidate[f'{i}_vertices']+candidate[f'{i}_camera_translation'];right=ref[f'{i}_vertices']+ref[f'{i}_camera_translation'];diffs.extend((np.linalg.norm(left-right,axis=-1)*1000).tolist());pixels.extend(np.linalg.norm(candidate[f'{i}_vertices_2d']-ref[f'{i}_vertices_2d'],axis=-1).tolist())
 times={str(n):float(np.mean([c['stats']['inference_ms']['mean'] for c in b['cases'] if c['hands']==n])) for n in (1,2)}
 metrics[name]=dict(network_ms_by_hand_count=times,camera_vertices_difference_mm=dict(mean=float(np.mean(diffs)),max=float(max(diffs))) if diffs else None,projection_difference_px=dict(mean=float(np.mean(pixels)),max=float(max(pixels))) if pixels else None)
 lines.append(f"| {name} | {times['1']:.2f} | {times['2']:.2f} | {max(diffs) if diffs else 0:.4f} | {max(pixels) if pixels else 0:.4f} |")
lines+=['','native=原生基线；trt=仅替换编码器；decoder=TRT+解码层编译；all=TRT+解码层编译+MHR 前后处理编译+MHR CUDA Graph，default 模式；reduced=同 all，reduce-overhead；tf32=同 all，额外开启 TF32（相机投影保持完整 FP32）。','', 'MHR 姿态修正保留。Graph 模块只将固定稀疏首层权重一次展开、旋转特征在 GPU 构造；每种批量首次捕获前自动检查与原始 MHR 的输出一致性。','', 'TF32 与 reduce-overhead 未取得稳定附加收益，推荐 all / default；TF32 不暴露为 GUI 推荐项。差异是相对原生输出，非真实重建误差。', '', '## 完整自动序列','']
for d in sorted((base/'full').glob('*')) if (base/'full').exists() else []:
 f=d/'sequence.json'
 if not f.exists():continue
 info=json.loads(f.read_text())
 if not info.get('complete'):continue
 lines += [f"### {d.name}",'',f"{info['frames']} 帧；[叠加视频](full/{d.name}/overlay.mp4)；[逐帧计时](full/{d.name}/timings.csv)。",'']
 groups=json.loads((d/'timing-groups.json').read_text())['groups_by_detected_hand_count']
 lines += ['| 检测手数 | 帧数 | 预热后测量帧数 | 检测 ms | 重建 ms | 含读写 ms | 含读写 P95 ms |','|---|---:|---:|---:|---:|---:|---:|']
 for count,g in groups.items():
  if not g['measured_frames']:continue
  lines.append(f"| {count} | {g['frames']} | {g['measured_frames']} | {g['detect_ms']['mean']:.2f} | {g['reconstruct_ms']['mean']:.2f} | {g['frame_with_io_ms']['mean']:.2f} | {g['frame_with_io_ms']['p95']:.2f} |")
 lines += ['','分组排除每种手数最初 3 次；IO 包含帧读取、输出复制、压缩 NPZ 保存，不含后续视频合成。视频回放 30 FPS 不是推理 FPS。','']
f=base/'detector/detector-comparison.json'
if f.is_file():
 r=json.loads(f.read_text());det={n:float(np.mean([c['backends'][n]['mean_ms'] for c in r['cases']])) for n in ('native','trt_fp16')};metrics['detector_mean_ms']=det
 lines += ['## 手框检测对照','',f"8 帧 × 每帧 10 次：原生 {det['native']:.2f} ms，TensorRT {det['trt_fp16']:.2f} ms，包含预处理/NMS/CPU 提框。原生矩形自动 padding 与引擎固定 640 正方形不同，框或标签可能改变；详见 detector/detector-comparison.json。",'']
if a.baseline_full:
 lines += ['## 连续序列相对原生基线提速','','同一录制、同一手数分组，分别去掉该手数最初 3 次；两次运行并非交替配对，温度/频率可能影响计时。','','| 序列/手数 | 原生重建 ms | 加速重建 ms | 重建耗时下降 | 原生检测+重建 ms | 加速检测+重建 ms |','|---|---:|---:|---:|---:|---:|']
 for d in sorted((base/'full').iterdir()):
  old=a.baseline_full/d.name/'timing-groups.json';new=d/'timing-groups.json'
  if not old.is_file() or not new.is_file():continue
  ref=json.loads(old.read_text())['groups_by_detected_hand_count'];cur=json.loads(new.read_text())['groups_by_detected_hand_count']
  for count,c in cur.items():
   r=ref.get(count,{})
   if not c.get('measured_frames') or not r.get('measured_frames'):continue
   before=r['reconstruct_ms']['mean'];after=c['reconstruct_ms']['mean']
   lines.append(f"| {d.name}/{count} | {before:.2f} | {after:.2f} | {(1-after/before)*100:.1f}% | {before+r['detect_ms']['mean']:.2f} | {after+c['detect_ms']['mean']:.2f} |")
 lines += ['','检测+重建不含磁盘读写、视频合成、GUI 渲染；包含每帧自动检测，没有跳过检测器。','']
f=base/'full-output-differences.json'
if f.is_file():
 r=json.loads(f.read_text())
 lines += ['## 全序列输出对照','','相对原生自动检测方案，按源帧和左右标签配对；检测器 padding/手框也有变化，以下不是三维真实误差。','','| 序列 | 左右手集合变化帧数 | 手框平均 IoU | 局部顶点平均差 mm | 相机系顶点平均差 mm | 相机系顶点最大差 mm | 投影平均差 px |','|---|---:|---:|---:|---:|---:|---:|']
 for name,g in r['sequences'].items():
  lines.append(f"| {name} | {g['frames_with_different_side_sets']} | {g['box_iou']['mean']:.4f} | {g['local_vertices_mm']['mean']:.3f} | {g['camera_vertices_mm']['mean']:.3f} | {g['camera_vertices_mm']['max']:.3f} | {g['projection_px']['mean']:.3f} |")
 lines += ['','详细逐帧结果：full-output-differences.csv；统计：full-output-differences.json。局部极值仍应结合原图检查，不能从平均差推导所有帧都等价。','']
 metrics['full_output_comparison']=r
lines += ['## 使用与验证边界','','重启 Studio → 手部实验 → 采用 512 加速方案 → 预加载 → 预热。模型加载后配置锁定，换方案须释放。单/双手分别缓存预热，身体和 SAM3 仍互斥。','', '静态/离线测试，尚未验证相机实时流；没有手部三维真值，也没有逐帧手框标注。完整自动版本还包含检测器 padding/框变化，不能将所有输出差异归因于 SAM 数值精度。','', '原始录制保持不变，输出均在 Data 盘；权重及引擎不纳入 Git。当前不宣称双手达到 100 ms / 10 Hz。']
(base/'REPORT.md').write_text('\n'.join(lines)+'\n');(base/'comparison.json').write_text(json.dumps(metrics,indent=2));print(base/'REPORT.md')
