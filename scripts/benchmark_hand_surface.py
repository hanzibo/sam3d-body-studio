# SPDX-License-Identifier: GPL-3.0-only
"""Paired same-process ablation, with unchanged TRT512 encoder and detector."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from studio_paths import path
sys.path.insert(0,str(path('official_repo')))
from gpu_lease import acquire_gpu_lease
from hand_engine import HandEngine
from hand_config import HandExecution
import cv2,numpy as np

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--repeats',type=int,default=20);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
if (a.output/'paired.json').exists():raise FileExistsError('Completed benchmark exists')
lease=acquire_gpu_lease('exclusive')
config=HandExecution(backbone='trt_fp16',detector='trt_fp16',decoder_compile=True,mhr_compile=True,mhr_graph=True,feedback_cache=True,intermediate_keypoints=True,final_hand_vertices=True)
engine=HandEngine(lambda s:print(s,flush=True),config);control=engine.model._studio_hand_feedback
def select(name):
 control['enabled']=True;control['feedback']=True;control['keypoints']=True;control['final_vertices']=name=='compact'
source=Path('/run/media/hzb/Data/experiments/hand-auto-20261112/full');cases=[]
for name,index in [('20261112_173053_single_hand_d4f1fc',0),('20261112_173053_single_hand_d4f1fc',150),('20261112_173116_two_hands_acdf29',150),('20261112_173116_two_hands_acdf29',449),('20261112_173053_single_hand_d4f1fc',324)]:
 info=json.loads((source/name/'sequence.json').read_text());r=json.loads((source/name/'frames.jsonl').read_text().splitlines()[index])
 image=cv2.cvtColor(cv2.imread(str(Path(info['source'])/r['input_file'])),cv2.COLOR_BGR2RGB)
 hands=[dict(bbox=h['bbox'],side=h['side']) for h in r['report']['hands']]
 def predict():return engine.predict_many(image,hands,1.1,info['camera_matrix'])
 names=['baseline','compact'];measurements={n:[] for n in names};results={}
 for n in names:
  select(n)
  for _ in range(4):results[n]=predict()
 for i in range(a.repeats):
  for n in names if i%2==0 else names[::-1]:
   select(n);start=time.perf_counter();result=predict();wall=(time.perf_counter()-start)*1000
   measurements[n].append(dict(wall_ms=wall,inference_ms=result[0][3]['inference_ms'],**result[0][3]['timings']));results[n]=result
 stats={n:{key:dict(mean=float(np.mean([r[key] for r in values])),p95=float(np.percentile([r[key] for r in values],95))) for key in values[0]} for n,values in measurements.items()}
 diffs={}
 for n in names[1:]:
  values={key:[] for key in ('camera_vertices_mm','local_vertices_mm','joints_mm','projection_px','global_rotations','mhr_params')}
  arrays={}
  for h,(ref,cur) in enumerate(zip(results['baseline'],results[n])):
   values['camera_vertices_mm'].extend(np.linalg.norm(cur[0]['vertices']+cur[0]['camera_translation']-ref[0]['vertices']-ref[0]['camera_translation'],axis=-1)*1000)
   for key,label,scale in [('vertices','local_vertices_mm',1000),('joints','joints_mm',1000),('vertices_2d','projection_px',1)]:
    values[label].extend(np.linalg.norm(cur[0][key]-ref[0][key],axis=-1)*scale)
   values['global_rotations'].extend(np.abs(cur[0]['global_rotations']-ref[0]['global_rotations']).flatten())
   values['mhr_params'].extend(np.abs(cur[1]['mhr_model_params']-ref[1]['mhr_model_params']).flatten())
   for key in ('vertices','joints','vertices_2d','global_rotations','camera_translation'):
    arrays[f'{h}_ref_{key}']=ref[0][key];arrays[f'{h}_candidate_{key}']=cur[0][key]
  diffs[n]={key:dict(mean=float(np.mean(v)),max=float(np.max(v))) for key,v in values.items()}
  assert diffs[n]['camera_vertices_mm']['max']<.05,diffs[n]
  assert diffs[n]['projection_px']['max']<.02,diffs[n]
  np.savez_compressed(a.output/(name+f'-{index:06d}-{n}.npz'),**arrays)
 case=dict(sequence=name,index=index,hands=len(hands),stats=stats,differences=diffs,selected_geometry=engine.model.head_pose_hand._studio_selected_hand.spec,intermediate_parity_mm=engine.model.head_pose_hand._studio_selected_hand.parity_mm,final_geometry=engine.model.head_pose_hand._studio_final_hand.spec,final_parity_mm=engine.model.head_pose_hand._studio_final_hand.parity_mm,layouts={n:results[n][0][3]['raw_vertex_layout'] for n in names},counts=results['compact'][0][3]['execution_counts'],feedback=results['compact'][0][3]['feedback_runtime'])
 cases.append(case)
 (a.output/'paired.json').write_text(json.dumps(dict(scope='Same process, 5 fixed-box cases, full surface vs hand-only surface interleaved in forward/reverse order after warmup; network timing excludes detector/export/render',execution=config.to_dict(),cases=cases),indent=2))
 print(json.dumps(dict(sequence=name,index=index,hands=len(hands),inference_ms={n:v['inference_ms']['mean'] for n,v in stats.items()},differences=diffs)),flush=True)
print('PAIRED_DONE',flush=True)
