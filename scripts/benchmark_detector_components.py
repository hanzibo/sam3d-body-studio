#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Measure the existing detector and INT8/384 backbone without changing engines."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from studio_paths import environment, path
from studio.core.gpu_lease import acquire_gpu_lease
from benchmark_embedl_encoder import memory_guard


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    guard = memory_guard()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(environment('fast'))
    sys.path.insert(0, str(path('fast_repo')))
    lease = acquire_gpu_lease('exclusive')
    import cv2
    import numpy as np
    import torch
    from tools.build_detector import HumanDetector
    from studio.body.embedl_backbone import EmbedlTensorRTBackbone
    torch.set_num_threads(4)
    cases = json.loads(Path('/run/media/hzb/Data/experiments/standard-reference-yolo-20261109/manifest.json').read_text())['cases']
    images = [cv2.imread(case['image']) for case in cases]
    detector = HumanDetector(name='yolo_pose', device='cuda', model=str(path('yolo_engine')))
    speeds = []
    original = detector.detector.predict
    def predict(*a, **kw):
        results = original(*a, **kw)
        speeds.append(dict(results[0].speed))
        return results
    detector.detector.predict = predict
    elapsed = []
    with contextlib.redirect_stdout(io.StringIO()), torch.inference_mode():
        for _ in range(15):
            detector.run_human_detection(images[0], default_to_full_image=False)
        speeds.clear()
        for image in images:
            for _ in range(20):
                torch.cuda.synchronize()
                began = time.perf_counter()
                result = detector.run_human_detection(image, default_to_full_image=False)
                torch.cuda.synchronize()
                elapsed.append((time.perf_counter()-began)*1000)
                assert len(result['boxes']) and np.isfinite(result['keypoints']).all()
    def stats(values):
        return dict(n=len(values), mean_ms=float(np.mean(values)), median_ms=float(np.median(values)), p95_ms=float(np.percentile(values,95)))
    backend = detector.detector.predictor.model
    bindings = {key: dict(shape=list(value.shape), dtype=str(value.dtype)) for key,value in backend.bindings.items()}
    report = dict(memory_guard=guard, gpu=torch.cuda.get_device_name(), yolo_engine=str(path('yolo_engine')),
                  image_shapes=[list(image.shape) for image in images], yolo_bindings=bindings,
                  yolo_full=stats(elapsed), yolo_stages={key:stats([row[key] for row in speeds]) for key in speeds[0]},
                  scope='Sequential isolated components; 15 warmups and 240 timed calls on 12 RGB-D poses. YOLO full wall time includes preprocess, TRT, postprocess and CPU result extraction. Ultralytics stage timers synchronize CUDA.')
    backbone = EmbedlTensorRTBackbone(Path('/run/media/hzb/Data/models/embedl-sam-3d-body/embedl_sam3dbody_int8_384_trt10_fp16.engine'),384).eval()
    image = cv2.resize(cv2.cvtColor(images[0],cv2.COLOR_BGR2RGB),(384,384)).astype(np.float32)/255
    image = (image-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
    x = torch.from_numpy(image.transpose(2,0,1)[None].copy()).cuda()
    values = []
    with torch.inference_mode():
        for _ in range(15): backbone(x)
        torch.cuda.synchronize()
        for _ in range(240):
            start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record(); output=backbone(x); end.record(); end.synchronize()
            values.append(start.elapsed_time(end))
        assert torch.isfinite(output).all()
    report['dino_int8_384'] = stats(values)
    report['dino_scope'] = 'CUDA events; production adapter input FP16 cast, TRT and FP16 output; no crop, disk, CPU transfers or decoder.'
    (args.output/'results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__': main()
