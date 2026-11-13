#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Bounded, isolated MHR precision checks and real-worker end-to-end tests."""
import argparse
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from benchmark_embedl_encoder import memory_guard


def geometry(a):
    import torch
    from gpu_lease import acquire_gpu_lease
    from studio_paths import path,environment
    from fast_keypoints import SelectedMHR70
    from mhr_precision import MixedMHR,HalfMHR,install_mhr_precision
    lease=acquire_gpu_lease('exclusive');os.environ.update(environment('fast'))
    torch.set_num_threads(4)
    checkpoint=path('fast_checkpoint_dir')
    mhr=torch.jit.load(str(checkpoint/'assets/mhr_model.pt'),map_location='cuda').eval()
    state=torch.load(checkpoint/'model.ckpt',map_location='cpu',weights_only=True,mmap=True)
    state=state.get('state_dict',state)
    mapping=state['head_pose.keypoint_mapping'].cuda()
    spec=importlib.util.spec_from_file_location('mhr_precision_utils',path('fast_repo')/'sam_3d_body/models/modules/mhr_utils.py')
    utils=importlib.util.module_from_spec(spec);spec.loader.exec_module(utils)
    b={key:state['head_pose.'+key].cuda() for key in ['scale_mean','scale_comps','hand_pose_mean','hand_pose_comps','hand_joint_idxs_left','hand_joint_idxs_right']}
    inputs=[]
    for p in sorted(a.reference.glob('int8384/pose-*/person_00.npz')):
        with np.load(p,allow_pickle=False) as d:
            tensor=lambda k:torch.from_numpy(d[k].copy()).cuda().float().reshape(1,-1)
            shape=tensor('shape_params');expr=tensor('expr_params')
            params=torch.zeros(1,204,device='cuda')
            params[:,3:6]=tensor('global_rot');params[:,6:136]=tensor('body_pose_params')[:,:130]
            hands=b['hand_pose_mean']+tensor('hand_pose_params').reshape(2,54)@b['hand_pose_comps']
            hands=utils.compact_cont_to_model_params_hand_fast(hands)
            params[:,b['hand_joint_idxs_left'].long()]=hands[0]
            params[:,b['hand_joint_idxs_right'].long()]=hands[1]
            params[:,136:]=b['scale_mean']+tensor('scale_params')@b['scale_comps']
            inputs.append((shape,params,expr))
    assert len(inputs)==12
    results={};references={}
    for label in ['final_fp32','final_mixed','mid_fp32','mid_mixed','final_fp16','mid_fp16']:
        try:
            if label.startswith('final'):
                module=mhr if label.endswith('fp32') else MixedMHR(mhr).eval() if label.endswith('mixed') else HalfMHR(mhr).eval()
                call=lambda *args:module(*args,False)
            else:
                selected=SelectedMHR70(mhr,mapping).eval()
                head=SimpleNamespace(mhr=mhr,_studio_selected_mhr70=selected,apply_correctives=False,enable_hand_model=False)
                install_mhr_precision(SimpleNamespace(head_pose=head),'fp32' if label.endswith('fp32') else 'all_mixed' if label.endswith('mixed') else 'all_fp16')
                module=selected;call=module
            with torch.inference_mode():
                fixed=tuple(t.clone() for t in inputs[0])
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(5):call(*fixed)
                torch.cuda.current_stream().wait_stream(stream)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):out=call(*fixed)
                values=[];deltas=[];saved=[]
                for args in inputs:
                    for dst,src in zip(fixed,args):dst.copy_(src)
                    for _ in range(3):graph.replay()
                    for _ in range(a.runs):
                        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        start.record();graph.replay();end.record();end.synchronize();values.append(start.elapsed_time(end))
                    arrays=tuple(t.float().cpu().numpy().copy() for t in out)
                    if not all(np.isfinite(x).all() for x in arrays):raise ValueError('Non-finite geometry')
                    saved.append(arrays)
                kind=label.split('_')[0]
                if label.endswith('fp32'):references[kind]=saved
                else:
                    for ref,candidate in zip(references[kind],saved):
                        # Full MHR returns cm, compact MHR returns meters.
                        unit=10 if kind=='final' else 1000
                        for index,(r,c) in enumerate(zip(ref,candidate)):
                            dims=3 if index==1 else c.shape[-1]
                            distance=np.linalg.norm(c[...,:dims]-r[...,:dims],axis=-1)*unit
                            deltas.append(dict(output=index,mean_mm=float(distance.mean()),max_mm=float(distance.max())))
                results[label]=dict(ok=True,n=len(values),mean_ms=float(np.mean(values)),p95_ms=float(np.percentile(values,95)),differences=deltas)
            print(label,json.dumps(results[label]),flush=True)
        except Exception as exc:
            import traceback
            results[label]=dict(ok=False,error=str(exc),traceback=traceback.format_exc())
            print('FAILED',label,str(exc)[-500:],flush=True)
            if label.endswith('fp32'):raise
        (a.output/'geometry.json').write_text(json.dumps(results,indent=2))
    return results


def full(a):
    from compare_backbones_standard import run,sha256
    from report_standard_reference import compare_prediction
    from launch_config import PRESETS
    manifest=json.loads((a.reference/'manifest.json').read_text())
    for case in manifest['cases']:assert sha256(case['image'])==case['image_sha256']
    (a.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    config=replace(PRESETS['实验：Fast INT8 384 身体（Embedl）'],warmup_runs=10,warmup_layout='camera').validate()
    timings={};errors={};diffs={}
    for mode in a.modes:
        os.environ['BODYSTUDIO_MHR_PRECISION']=mode
        try:
            rows=run(config,mode,manifest['cases'],a.output,a.runs)
            times=np.array([v*1000 for row in rows for v in row['benchmark']['pipeline_seconds']])
            timings[mode]=dict(n=len(times),mean_ms=float(times.mean()),median_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95)))
            comparisons=[]
            for case in manifest['cases']:
                folder=f'pose-{case["id"]:02d}'
                report=json.loads((a.output/mode/folder/'inference.json').read_text())
                assert report['execution']['mhr_precision']['mode']==mode
                assert report['execution']['final_replay']['graph_captured']
                assert report['keypoint_path']['graph_captured']
                assert report['compile_actual']==dict(decoder=True,mhr=True,interm=True)
                assert report['feedback_cache']['last']['hits']==2
                if mode=='fp32':continue
                with np.load(a.output/'fp32'/folder/'person_00.npz',allow_pickle=False) as r,np.load(a.output/mode/folder/'person_00.npz',allow_pickle=False) as c:
                    row=compare_prediction(r,c)
                    row['rotation_max_abs']=float(np.abs(c['pred_global_rots']-r['pred_global_rots']).max())
                    row['pixel_mean']=float(np.linalg.norm(c['pred_keypoints_2d']-r['pred_keypoints_2d'],axis=-1).mean())
                    row['pixel_max']=float(np.linalg.norm(c['pred_keypoints_2d']-r['pred_keypoints_2d'],axis=-1).max())
                    row['pose']=case['id'];comparisons.append(row)
            diffs[mode]=comparisons
        except Exception as exc:
            errors[mode]=str(exc);print('VARIANT_FAILED',mode,str(exc),flush=True)
            if mode=='fp32':raise
        report=dict(timings=timings,differences=diffs,errors=errors)
        (a.output/'full.json').write_text(json.dumps(report,indent=2))
    os.environ.pop('BODYSTUDIO_MHR_PRECISION',None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['geometry','full'],required=True)
    p.add_argument('--reference',type=Path,default=Path('/run/media/hzb/Data/experiments/embedl384-20261109/full'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--runs',type=int,default=20)
    p.add_argument('--modes',nargs='+',default=['fp32','final_mixed','all_mixed','all_fp16'])
    a=p.parse_args();guard=memory_guard()
    if not 1<=a.runs<=200:raise ValueError('runs must be 1–200')
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'protocol.json').write_text(json.dumps(dict(stage=a.stage,memory_guard=guard,modes=a.modes,runs=a.runs,reference=str(a.reference)),indent=2))
    for file in [Path(__file__),ROOT/'mhr_precision.py']:(a.output/file.name).write_bytes(file.read_bytes())
    (geometry if a.stage=='geometry' else full)(a)


if __name__=='__main__':main()
