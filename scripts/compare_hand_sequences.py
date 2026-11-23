# SPDX-License-Identifier: GPL-3.0-only
"""Compare paired automatic outputs to baseline, not to physical ground truth."""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from studio.hand.hand_automatic import iou
p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();rows=[];summaries={}
for candidate in sorted(a.candidate.iterdir()):
 reference=a.baseline/candidate.name
 if not (candidate/'sequence.json').is_file() or not reference.is_dir():continue
 ci=json.loads((candidate/'sequence.json').read_text());ri=json.loads((reference/'sequence.json').read_text())
 if not ci.get('complete') or not ri.get('complete'):raise ValueError('Cannot compare incomplete outputs')
 cr=[json.loads(x) for x in (candidate/'frames.jsonl').read_text().splitlines()];rr=[json.loads(x) for x in (reference/'frames.jsonl').read_text().splitlines()]
 if len(cr)!=len(rr):raise ValueError('Source frame count mismatch')
 local=[];camera=[];pixels=[];overlaps=[];different_sides=0;paired=0
 for c,r in zip(cr,rr):
  if c['source_index']!=r['source_index'] or c['timestamp']!=r['timestamp']:raise ValueError('Source alignment mismatch')
  ch={h['side']:h for h in c['report']['hands']};rh={h['side']:h for h in r['report']['hands']}
  different_sides+=set(ch)!=set(rh)
  with np.load(candidate/c['scene'],allow_pickle=False) as cn,np.load(reference/r['scene'],allow_pickle=False) as rn:
   for side in sorted(ch.keys()&rh.keys()):
    paired+=1;overlap=iou(ch[side]['bbox'],rh[side]['bbox']);overlaps.append(overlap)
    dv=np.linalg.norm(cn[side+'_vertices']-rn[side+'_vertices'],axis=-1)*1000
    dc=np.linalg.norm(cn[side+'_vertices']+cn[side+'_camera_translation']-rn[side+'_vertices']-rn[side+'_camera_translation'],axis=-1)*1000
    dp=np.linalg.norm(cn[side+'_vertices_2d']-rn[side+'_vertices_2d'],axis=-1)
    local.extend(dv);camera.extend(dc);pixels.extend(dp)
    rows.append(dict(sequence=candidate.name,frame=c['source_index'],side=side,box_iou=overlap,local_vertex_mean_mm=float(dv.mean()),camera_vertex_mean_mm=float(dc.mean()),camera_vertex_p95_mm=float(np.percentile(dc,95)),projection_mean_px=float(dp.mean())))
 def stats(values):return dict(mean=float(np.mean(values)),p95=float(np.percentile(values,95)),max=float(np.max(values))) if len(values) else None
 summaries[candidate.name]=dict(frames=len(cr),frames_with_different_side_sets=int(different_sides),paired_hands=paired,box_iou=stats(overlaps),local_vertices_mm=stats(local),camera_vertices_mm=stats(camera),projection_px=stats(pixels))
a.output.mkdir(parents=True,exist_ok=True)
with (a.output/'full-output-differences.csv').open('w',newline='') as stream:
 writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
(a.output/'full-output-differences.json').write_text(json.dumps(dict(sequences=summaries,scope='Paired by source frame and side label only; no ground truth; changed detector padding/boxes affect 3D results; low IoU or incorrect identity can invalidate physical correspondence'),indent=2));print(json.dumps(summaries,indent=2))
