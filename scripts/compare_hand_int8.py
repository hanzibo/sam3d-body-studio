# SPDX-License-Identifier: GPL-3.0-only
"""Quantization output differences and temporal proxies, without 3D ground truth."""
import argparse,json,sys
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
def stats(values):
 v=np.asarray(values);return dict(mean=float(v.mean()),median=float(np.median(v)),p95=float(np.percentile(v,95)),max=float(v.max()),samples=int(v.size)) if v.size else None
summaries={}
for d in sorted(a.candidate.iterdir()):
 if not (d/'sequence.json').is_file():continue
 baseline=a.baseline/d.name;ci=json.loads((d/'sequence.json').read_text());ri=json.loads((baseline/'sequence.json').read_text())
 assert ci['complete'] and ri['complete'];cr=[json.loads(l) for l in (d/'frames.jsonl').read_text().splitlines()];rr=[json.loads(l) for l in (baseline/'frames.jsonl').read_text().splitlines()];assert len(cr)==len(rr)
 values={k:[] for k in ('wrist_aligned_vertices_mm','wrist_camera_mm','joint_camera_mm','global_rotation_angle_deg')};temporal={mode:{key:[] for key in ('wrist_step_mm','local_joint_step_mm','wrist_second_difference_mm','local_joint_second_difference_mm')} for mode in ('fp16','int8')};history={};max_roundtrip=0.;instances=0
 for c,r in zip(cr,rr):
  assert c['source_index']==r['source_index'] and c['timestamp']==r['timestamp'];assert {h['side'] for h in c['report']['hands']}=={h['side'] for h in r['report']['hands']}
  for h in c['report']['hand_reports']:
   assert h['execution']['backbone']=='trt_int8' and h['decoder']==h['mhr']=='FP32' and h['raw_vertex_layout']=='hand_only'
   assert h['feedback_runtime']['last']['full_mhr_calls']==0
   max_roundtrip=max(max_roundtrip,h['projection_roundtrip_max_px']);instances+=1
  with np.load(d/c['scene'],allow_pickle=False) as cn,np.load(baseline/r['scene'],allow_pickle=False) as rn:
   for data in (cn,rn):
    for key in data.files:
     if np.issubdtype(data[key].dtype,np.number):assert np.isfinite(data[key]).all(),(d,c['index'],key)
   for side in c['report']['hands']:
    side=side['side'];key=side+'_';names=cn[key+'joint_names'].tolist();wi=next(i for i,n in enumerate(names) if n.endswith('wrist'))
    cv=cn[key+'vertices'];rv=rn[key+'vertices'];cj=cn[key+'joints'];rj=rn[key+'joints'];ct=cn[key+'camera_translation'];rt=rn[key+'camera_translation']
    values['wrist_aligned_vertices_mm'].extend(np.linalg.norm((cv-cj[wi])-(rv-rj[wi]),axis=-1)*1000)
    values['wrist_camera_mm'].append(np.linalg.norm(cj[wi]+ct-rj[wi]-rt)*1000)
    values['joint_camera_mm'].extend(np.linalg.norm(cj+ct-rj-rt,axis=-1)*1000)
    relative=cn[key+'global_rotations']@rn[key+'global_rotations'].transpose(0,2,1)
    values['global_rotation_angle_deg'].extend(np.degrees(np.arccos(np.clip((np.trace(relative,axis1=1,axis2=2)-1)/2,-1,1))))
    for mode,j,t in [('fp16',rj,rt),('int8',cj,ct)]:
     hk=(mode,side);cur=(c['source_index'],j[wi]+t,j-j[wi]);hist=history.setdefault(hk,[])
     if hist and hist[-1][0]+1==cur[0]:
      temporal[mode]['wrist_step_mm'].append(np.linalg.norm(cur[1]-hist[-1][1])*1000)
      temporal[mode]['local_joint_step_mm'].extend(np.linalg.norm(cur[2]-hist[-1][2],axis=-1)*1000)
      if len(hist)>=2 and hist[-2][0]+2==cur[0]:
       temporal[mode]['wrist_second_difference_mm'].append(np.linalg.norm(cur[1]-2*hist[-1][1]+hist[-2][1])*1000)
       temporal[mode]['local_joint_second_difference_mm'].extend(np.linalg.norm(cur[2]-2*hist[-1][2]+hist[-2][2],axis=-1)*1000)
     hist.append(cur);history[hk]=hist[-2:]
 summaries[d.name]=dict(frames=len(cr),hand_instances=instances,all_outputs_finite=True,projection_roundtrip_max_px=max_roundtrip,differences={k:stats(v) for k,v in values.items()},temporal={m:{k:stats(v) for k,v in x.items()} for m,x in temporal.items()})
result=dict(sequences=summaries,scope='Same frames, detected boxes and labels; no 3D ground truth. Wrist-aligned vertices remove translation but retain orientation/shape. Temporal first/second differences in mm per source frame include real motion and are only stability proxies, not measured jitter or accuracy.')
(a.output/'hand-quality.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
