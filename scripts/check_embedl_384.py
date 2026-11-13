#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Compare the experimental 384 graph through real Studio workers.

Reuses frozen Standard512 outputs for geometric reference, not fresh timings.
Fresh FP16/INT8 384 timings both include YOLO on every iteration.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from launch_config import PRESETS
from benchmark_embedl_encoder import memory_guard
from compare_backbones_standard import run, sha256
from report_standard_reference import compare_prediction


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,default=Path('/run/media/hzb/Data/experiments/standard-reference-yolo-20261109'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--runs',type=int,default=20)
    a=p.parse_args();guard=memory_guard()
    if not 1<=a.runs<=200:raise ValueError('runs must be 1–200')
    a.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((a.reference/'manifest.json').read_text())
    manifest.update(reference=str(a.reference),memory_guard=guard)
    expected={case['id']:np.array(case['bbox'],copy=True) for case in manifest['cases']}
    for case in manifest['cases']:
        assert sha256(case['image'])==case['image_sha256']
    (a.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    variants=[('fast384','演示最快'),('int8384','实验：Fast INT8 384 身体（Embedl）')]
    timing={}
    for label,name in variants:
        config=replace(PRESETS[name],warmup_runs=10,warmup_layout='camera').validate()
        rows=run(config,label,manifest['cases'],a.output,a.runs)
        for case in manifest['cases']:
            np.testing.assert_allclose(case['bbox'],expected[case['id']],rtol=0,atol=1e-4)
            report=json.loads((a.output/label/f'pose-{case["id"]:02d}/inference.json').read_text())
            assert report['compile_actual']==dict(decoder=True,mhr=True,interm=True)
            assert report['feedback_cache']['enabled'] and report['feedback_cache']['last']['hits']==2
            assert report['keypoint_path']['graph_captured']
            assert report['execution']['final_replay']['graph_captured']
            assert report['execution']['body_io']['enabled']
            if label=='int8384':
                assert report['backbone_runtime']['input_shape']==[1,3,384,384]
                assert report['backbone_runtime']['output_shape']==[1,1280,24,24]
        values=np.array([s*1000 for row in rows for s in row['benchmark']['pipeline_seconds']])
        timing[label]=dict(n=len(values),mean_ms=float(values.mean()),median_ms=float(np.median(values)),p95_ms=float(np.percentile(values,95)))
        (a.output/'timings.json').write_text(json.dumps(timing,indent=2))
    per_pose=[]
    for case in manifest['cases']:
        def load(folder):
            with np.load(folder/f'pose-{case["id"]:02d}/person_00.npz',allow_pickle=False) as d:
                return {k:d[k].copy() for k in ['pred_vertices','pred_keypoints_3d','pred_joint_coords','pred_cam_t','faces']}
        predictions={label:load(a.output/label) for label,_ in variants}
        reference=load(a.reference/'std512')
        for label,prediction in predictions.items():
            assert np.array_equal(reference['faces'],prediction['faces'])
            per_pose.append(dict(id=case['id'],variant=label,**compare_prediction(reference,prediction)))
        per_pose.append(dict(id=case['id'],variant='int8384_vs_fp16384',**compare_prediction(predictions['fast384'],predictions['int8384'])))
    aggregates={}
    for label in ['fast384','int8384','int8384_vs_fp16384']:
        rows=[row for row in per_pose if row['variant']==label]
        aggregates[label]={key:float(np.mean([row[key] for row in rows])) for key in rows[0] if key not in ('id','variant','angles_difference_deg')}
        aggregates[label]['worst_pose_aligned_mesh_mm']=max(row['mesh_pelvis_aligned_mean_mm'] for row in rows)
        aggregates[label]['worst_pose_elbow_deg']=max(value for row in rows for key,value in row['angles_difference_deg'].items() if 'elbow' in key)
    report=dict(timings=timing,aggregates=aggregates,cases=per_pose,
                reference=str(a.reference),warning='Standard512 is a frozen prediction reference, not ground truth. 384 uses unchanged 512 quantization scales. No live-camera/temporal validation.',
                protocol=f'Fresh 384 FP16 and INT8 actual workers, sequential; 10 initial warmups + 3 per pose + {a.runs} timed runs per pose. YOLO executes every inference; synchronized complete inference excluding disk/load/export/render. All existing Fast optimizations enabled.')
    (a.output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='cases'},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
