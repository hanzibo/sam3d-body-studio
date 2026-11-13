# SPDX-License-Identifier: GPL-3.0-only
"""Compare deployed native and TRT hand-box paths on identical RGB samples."""
import argparse,sys,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from studio_paths import environment
os.environ.update(environment('fast'))
from gpu_lease import acquire_gpu_lease
from hand_automatic import HandDetector,choose_hands,iou
import cv2,numpy as np
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
lease=acquire_gpu_lease('exclusive');detectors={key:HandDetector(key) for key in ('native','trt_fp16')};images=[]
for d in sorted(Path('/run/media/hzb/Data/experiments/wilor/datasets').iterdir()):
 if not (d/'manifest.json').is_file():continue
 for index in (0,150,330,449):images.append((d.name+f'-{index:06d}',cv2.cvtColor(cv2.imread(str(d/f'frames/{index:06d}.jpg')),cv2.COLOR_BGR2RGB)))
results=[]
for detector in detectors.values():
 for _ in range(5):detector.detect(images[0][1])
for label,image in images:
 row=dict(case=label,backends={})
 for name,detector in detectors.items():
  times=[]
  for _ in range(10):hands,ms=detector.detect(image);times.append(ms)
  row['backends'][name]=dict(mean_ms=float(np.mean(times)),p95_ms=float(np.percentile(times,95)),hands=choose_hands(hands))
 native={h['side']:h for h in row['backends']['native']['hands']};trt={h['side']:h for h in row['backends']['trt_fp16']['hands']}
 row['common_sides']=sorted(native.keys()&trt.keys());row['missing_from_trt']=sorted(native.keys()-trt.keys());row['extra_in_trt']=sorted(trt.keys()-native.keys());row['box_iou']={side:iou(native[side]['bbox'],trt[side]['bbox']) for side in row['common_sides']};results.append(row)
report=dict(cases=results,metadata={k:v.metadata for k,v in detectors.items()},scope='Wall including preprocessing, inference, NMS and CPU box extraction; 5 warmups; 10 calls per image; native auto rectangular letterbox versus TRT fixed square640; same source weights but not identical padding; no GT accuracy')
(a.output/'detector-comparison.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
