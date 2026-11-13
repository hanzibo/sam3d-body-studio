# SPDX-License-Identifier: GPL-3.0-only
"""Bounded real GPU same-frame / consecutive-frame hand smoothing checks."""
import json,sys,time
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from gpu_lease import acquire_gpu_lease
from hand_engine import HandEngine
from hand_config import HandExecution
from hand_temporal import HandTemporalFilter
lease=acquire_gpu_lease('exclusive')
config=HandExecution(backbone='trt_int8',detector='trt_fp16',decoder_compile=True,mhr_compile=True,mhr_graph=True,feedback_cache=True,intermediate_keypoints=True,final_hand_vertices=True,intermediate_compile=True,intermediate_predictions=3)
e=HandEngine(lambda s:print(s,flush=True),config)
out=Path('/run/media/hzb/Data/experiments/hand-one-euro-20261113');out.mkdir(parents=True,exist_ok=True)
source=Path('/run/media/hzb/Data/experiments/hand-auto-20261112/full')
measurements=[]
for name in ('20261112_173053_single_hand_d4f1fc','20261112_173116_two_hands_acdf29'):
    meta=json.loads((source/name/'sequence.json').read_text());rows=(source/name/'frames.jsonl').read_text().splitlines()
    row=json.loads(rows[150]);image=cv2.cvtColor(cv2.imread(str(Path(meta['source'])/row['input_file'])),cv2.COLOR_BGR2RGB)
    hands=[dict(bbox=h['bbox'],side=h['side']) for h in row['report']['hands']]
    for _ in range(4):baseline=e.predict_many(image,hands,1.1,meta['camera_matrix'])
    options=dict(pose=True,position=False,min_cutoff=1.5,beta=.5,d_cutoff=1.)
    e._temporal_filter=HandTemporalFilter();e.temporal_request=(0.,options)
    filtered=e.predict_many(image,hands,1.1,meta['camera_matrix'])
    for base,cur in zip(baseline,filtered):
        for key in ('vertices','joints','global_rotations','camera_translation'):
            np.testing.assert_allclose(base[0][key],cur[0][key],atol=2e-5,rtol=1e-4)
            np.testing.assert_allclose(base[0][key],cur[0]['raw_'+key],atol=1e-6,rtol=1e-5)
        for key in base[1]:np.testing.assert_allclose(base[1][key],cur[1][key],atol=1e-6,rtol=1e-5)
        assert cur[3]['execution_counts']==dict(body=0,hand=1,backbone=1)
    # consecutive observed frames, use fixed representative boxes to isolate filter cost
    costs=[];walls=[];deltas=[];raw_paths=[];filtered_paths=[]
    for i in range(1,26):
        row=json.loads(rows[150+i]);image=cv2.cvtColor(cv2.imread(str(Path(meta['source'])/row['input_file'])),cv2.COLOR_BGR2RGB)
        e.temporal_request=(i/15,options);start=time.perf_counter();result=e.predict_many(image,hands,1.1,meta['camera_matrix'])
        walls.append((time.perf_counter()-start)*1000);costs.append(result[0][3]['one_euro']['processing_ms'])
        raw_paths.append(np.concatenate([h[0]['raw_joints']-h[0]['raw_joints'][0] for h in result]))
        filtered_paths.append(np.concatenate([h[0]['joints']-h[0]['joints'][0] for h in result]))
        deltas.append(float(np.mean([np.linalg.norm(h[0]['vertices']-h[0]['raw_vertices'],axis=1).mean()*1000 for h in result])))
        if i in (1,12,25):np.savez(out/f'{name}-{i:03d}.npz',**{str(n)+'_'+k:v for n,h in enumerate(result) for k,v in h[0].items()})
    measurements.append(dict(sequence=name,hands=len(hands),filter_mean_ms=float(np.mean(costs[5:])),filter_p95_ms=float(np.percentile(costs[5:],95)),total_mean_ms=float(np.mean(walls[5:])),raw_frame_delta_mm=float(np.mean(np.linalg.norm(np.diff(raw_paths,axis=0),axis=-1))*1000),filtered_frame_delta_mm=float(np.mean(np.linalg.norm(np.diff(filtered_paths,axis=0),axis=-1))*1000),mean_change_mm=float(np.mean(deltas)),options=options))
    e.temporal_request=None
(out/'validation.json').write_text(json.dumps(measurements,indent=2));print('HAND_TEMPORAL_OK',json.dumps(measurements),flush=True)
