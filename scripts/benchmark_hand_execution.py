# SPDX-License-Identifier: GPL-3.0-only
"""Bounded GPU hand execution benchmark, run via launch env in a memory scope."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from studio_paths import path
sys.path.insert(0,str(path('official_repo')))
import cv2,numpy as np
from hand_engine import HandEngine
from hand_acceleration import HandExecution
from gpu_lease import acquire_gpu_lease

p=argparse.ArgumentParser();p.add_argument('--variant',choices=['native','trt','decoder','mhr','graph','all','reduced','tf32','feedback','keypoints','feedback_keypoints'],required=True)
p.add_argument('--output',type=Path,required=True);p.add_argument('--reference',type=Path);p.add_argument('--repeats',type=int,default=10);a=p.parse_args()
extra=a.variant in ('feedback','keypoints','feedback_keypoints')
config=HandExecution(backbone='native' if a.variant=='native' else 'trt_fp16',decoder_compile=extra or a.variant in ('decoder','all','reduced','tf32'),mhr_compile=extra or a.variant in ('mhr','all','reduced','tf32'),mhr_graph=extra or a.variant in ('graph','all','reduced','tf32'),tf32=a.variant=='tf32',compile_mode='reduce-overhead' if a.variant=='reduced' else 'default',feedback_cache=a.variant in ('feedback','feedback_keypoints'),intermediate_keypoints=a.variant in ('keypoints','feedback_keypoints'))
a.output.mkdir(parents=True,exist_ok=True)
if (a.output/'benchmark.json').exists():raise FileExistsError('Refusing to overwrite completed benchmark')
lease=acquire_gpu_lease('exclusive');engine=HandEngine(lambda s:print(s,flush=True),execution=config)
source=Path('/run/media/hzb/Data/experiments/hand-auto-20261112/full');cases=[]
for name,index in [('20261112_173053_single_hand_d4f1fc',0),('20261112_173053_single_hand_d4f1fc',150),('20261112_173116_two_hands_acdf29',150),('20261112_173116_two_hands_acdf29',449)]:
 info=json.loads((source/name/'sequence.json').read_text());rows=[json.loads(x) for x in (source/name/'frames.jsonl').read_text().splitlines()];r=rows[index]
 image=cv2.cvtColor(cv2.imread(str(Path(info['source'])/r['input_file'])),cv2.COLOR_BGR2RGB);hands=[dict(bbox=x['bbox'],side=x['side']) for x in r['report']['hands']]
 label=name+f'-{index:06d}';print('CASE '+label,flush=True)
 warm_start=time.perf_counter()
 for _ in range(3):result=engine.predict_many(image,hands,padding=1.1,matrix=info['camera_matrix'])
 warm_seconds=time.perf_counter()-warm_start;measurements=[]
 for _ in range(a.repeats):
  start=time.perf_counter();result=engine.predict_many(image,hands,padding=1.1,matrix=info['camera_matrix']);wall=(time.perf_counter()-start)*1000
  measurements.append(dict(wall_ms=wall,inference_ms=result[0][3]['inference_ms'],**result[0][3]['timings']))
 arrays={}
 for i,(data,raw,crop,report) in enumerate(result):
  for key in ('vertices','joints','camera_translation','vertices_2d'):arrays[f'{i}_{key}']=data[key]
  for key in ('hand','shape','scale','mhr_model_params'):arrays[f'{i}_raw_{key}']=raw[key]
 np.savez_compressed(a.output/(label+'.npz'),**arrays)
 diff={}
 if a.reference:
  with np.load(a.reference/(label+'.npz'),allow_pickle=False) as ref:
   for i in range(len(hands)):
    for key,unit in [('vertices',1000),('joints',1000),('camera_translation',1000),('vertices_2d',1)]:
     d=np.linalg.norm(arrays[f'{i}_{key}']-ref[f'{i}_{key}'],axis=-1);diff[f'{i}_{key}']=dict(mean=float(d.mean()*unit),max=float(d.max()*unit))
 stats={k:dict(mean=float(np.mean([v[k] for v in measurements])),p95=float(np.percentile([v[k] for v in measurements],95))) for k in measurements[0]}
 case=dict(case=label,hands=len(hands),warmup_seconds=warm_seconds,stats=stats,difference=diff,report=result[0][3]);cases.append(case)
 (a.output/'benchmark.json').write_text(json.dumps(dict(variant=a.variant,execution=config.to_dict(),cases=cases),indent=2));print(json.dumps(dict(case=label,stats=stats,difference=diff)),flush=True)
print('BENCHMARK_DONE',flush=True)
