# SPDX-License-Identifier: GPL-3.0-only
"""Profile warmed hand callbacks; CUDA intervals include host dispatch gaps."""
import argparse,json,sys,time
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from studio_paths import path
sys.path.insert(0,str(path('official_repo')))
from gpu_lease import acquire_gpu_lease
from hand_engine import HandEngine
from hand_config import HandExecution
import cv2,numpy as np,torch

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--runs',type=int,default=20);p.add_argument('--optimized',action='store_true');a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
if (a.output/'profile.json').exists():raise FileExistsError('Completed profile exists')
lease=acquire_gpu_lease('exclusive')
engine=HandEngine(lambda s:print(s,flush=True),HandExecution(backbone='trt_fp16',detector='trt_fp16',decoder_compile=True,mhr_compile=True,mhr_graph=True,feedback_cache=a.optimized,intermediate_keypoints=a.optimized))
source=Path('/run/media/hzb/Data/experiments/hand-auto-20261112/full/20261112_173116_two_hands_acdf29')
info=json.loads((source/'sequence.json').read_text());row=json.loads((source/'frames.jsonl').read_text().splitlines()[150])
rgb=cv2.cvtColor(cv2.imread(str(Path(info['source'])/row['input_file'])),cv2.COLOR_BGR2RGB)
hands=[dict(bbox=h['bbox'],side=h['side']) for h in row['report']['hands']]
current=None;restores=[]
def timed(name,fn):
 @torch.compiler.disable
 def call(*args,**kwargs):
  if current is None:return fn(*args,**kwargs)
  start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
  t=time.perf_counter();out=fn(*args,**kwargs);cpu=(time.perf_counter()-t)*1000;end.record()
  current.append((name,start,end,cpu));return out
 return call
def patch(obj,name,label):
 original=getattr(obj,name);setattr(obj,name,timed(label,original));restores.append((obj,name,original))
def predict(n):return engine.predict_many(rgb,hands[:n],1.1,info['camera_matrix'])
results={}
for n in (1,2):
 for _ in range(3):ref=predict(n)
 before=[]
 for _ in range(a.runs):before.append(predict(n)[0][3]['inference_ms'])
 model=engine.model;decoder=model.decoder_hand;original=decoder.forward
 def forward(*args,**kwargs):
  for key,label in [('token_to_pose_output_fn','pose_callback'),('keypoint_token_update_fn','feedback')]:
   if kwargs.get(key):kwargs[key]=timed(label,kwargs[key])
  return original(*args,**kwargs)
 decoder.forward=forward;restores.append((decoder,'forward',original))
 if a.optimized:
  from hand_feedback import HandFeedbackCall
  patch(HandFeedbackCall,'__call__','feedback')
 for i,layer in enumerate(decoder.layers):patch(layer,'_call_impl',f'layer_{i}')
 patch(model.head_pose_hand,'mhr_forward','mhr_total')
 patch(model,'camera_project_hand','projection')
 patch(model,'_full_to_crop','crop_projection')
 for _ in range(3):predict(n)
 rows=[]
 for _ in range(a.runs):
  current=[];out=predict(n);torch.cuda.synchronize()
  rows.append(dict(inference_ms=out[0][3]['inference_ms'],intervals=[dict(name=name,gpu_ms=s.elapsed_time(e),cpu_ms=c) for name,s,e,c in current]));current=None
 for obj,name,fn in reversed(restores):setattr(obj,name,fn)
 restores.clear()
 after=[]
 for _ in range(3):predict(n)
 for _ in range(a.runs):after.append(predict(n)[0][3]['inference_ms'])
 aggregate=defaultdict(list);counts=defaultdict(list)
 for r in rows:
  sums=defaultdict(float);calls=defaultdict(int)
  for interval in r['intervals']:sums[interval['name']]+=interval['gpu_ms'];calls[interval['name']]+=1
  for name,value in sums.items():aggregate[name].append(value);counts[name].append(calls[name])
 delta=max(float(np.max(np.abs(x[0]['vertices']-y[0]['vertices']))) for x,y in zip(ref,out))
 assert delta<1e-5,delta
 results[n]=dict(before_mean_ms=float(np.mean(before)),after_mean_ms=float(np.mean(after)),instrumented_mean_ms=float(np.mean([r['inference_ms'] for r in rows])),components={name:dict(mean_ms=float(np.mean(v)),calls=float(np.mean(counts[name]))) for name,v in aggregate.items()},vertices_max_difference_m=delta,rows=rows)
 print(json.dumps({k:v for k,v in results[n].items() if k!='rows'}),flush=True)
(a.output/'profile.json').write_text(json.dumps(dict(scope='Nested CUDA stream intervals, not additive pure kernel time; profiler overhead bracketed by uninstrumented timings',batches=results),indent=2))
