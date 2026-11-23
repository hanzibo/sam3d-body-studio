# SPDX-License-Identifier: GPL-3.0-only
"""Paired full detector timings, exact input tensor and output parity."""
import sys,os,json,time,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from studio_paths import environment
os.environ.update(environment('fast'))
from studio.core.gpu_lease import acquire_gpu_lease
lease=acquire_gpu_lease('exclusive')
import torch,cv2,numpy as np
from studio.hand.hand_automatic import HandDetector
p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4)
detectors={key:HandDetector('trt_fp16',optimized_preprocess=key=='candidate') for key in ('baseline','candidate')}
images=[]
for directory in sorted(Path('/run/media/hzb/Data/experiments/wilor/datasets').iterdir()):
 if not (directory/'manifest.json').is_file():continue
 for index in (0,150,330,449):images.append((directory.name+f'-{index}',cv2.cvtColor(cv2.imread(str(directory/f'frames/{index:06d}.jpg')),cv2.COLOR_BGR2RGB)))
# Include actual camera aspect ratio, unusual odd dimensions and channels.
rng=np.random.default_rng(2)
for size in ((800,1280),(479,641)):images.append((f'random-{size}',rng.integers(0,256,(*size,3),dtype=np.uint8)))
for d in detectors.values():
 for _ in range(8):d.detect(images[0][1])
rows=[]
for label,image in images:
 d0=detectors['baseline'].model.predictor;d1=detectors['candidate'].model.predictor
 with torch.inference_mode():
  expected=d0.preprocess([image[:,:,::-1].copy()]);actual=d1.preprocess([image]);torch.cuda.synchronize()
  assert torch.equal(expected,actual),label
 ref,_=detectors['baseline'].detect(image);candidate,_=detectors['candidate'].detect(image)
 assert len(ref)==len(candidate),(label,ref,candidate)
 for x,y in zip(ref,candidate):
  assert x['side']==y['side'];np.testing.assert_allclose(x['bbox'],y['bbox'],rtol=0,atol=0);assert x['score']==y['score']
 samples={key:[] for key in detectors};stages={key:[] for key in detectors}
 for i in range(30):
  for key in (list(detectors) if i%2==0 else list(detectors)[::-1]):
   _,ms=detectors[key].detect(image);samples[key].append(ms);stages[key].append(detectors[key].last_timings.copy())
 rows.append(dict(case=label,input_bitwise_equal=True,boxes_scores_exact=True,times=samples,stages=stages))
report=dict(rows=rows,scope='Alternating same-image TRT detector; RGB caller to CPU boxes including preprocessing/TRT/NMS/copies; 8 warmups per backend, 30 pairs per case; unchanged engine and letterbox; recorded frames and random shape cases, no GT accuracy')
for key in detectors:
 all_values=[x for row in rows for x in row['times'][key]]
 report[key]=dict(mean_ms=float(np.mean(all_values)),p95_ms=float(np.percentile(all_values,95)),stages={stage:float(np.mean([x[stage] for row in rows for x in row['stages'][key]])) for stage in ('preprocess','inference','postprocess','other')})
(a.output/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
