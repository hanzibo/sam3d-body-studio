#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Paired, warmed full-inference and decoder timing; original vs feedback cache."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from studio_paths import environment,path
from gpu_lease import acquire_gpu_lease


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--image-size',type=int,choices=[384,512],default=384)
    p.add_argument('--runs',type=int,default=30)
    p.add_argument('--warmup',type=int,default=10)
    p.add_argument('--poses',type=Path)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    lease=acquire_gpu_lease('exclusive')
    os.environ.update(environment('fast'))
    sys.path.insert(0,str(path('fast_repo')))
    loader=Path('/home/hzb/Work/fast-sam3d-body-deploy/run.py')
    sys.argv=[str(loader),'--image',str(a.image),'--output',str(a.output),
              '--compile','--trt-backbone',str(path(f'backbone_engine_{a.image_size}')),
              '--trt-yolo',str(path('yolo_engine')),'--image-size',str(a.image_size),
              '--inference-type','body','--warmup',str(a.warmup),'--runs','3']
    from offline_dinov3 import local_dinov3_source
    ns={'__name__':'__main__','__file__':str(loader)}
    with local_dinov3_source():
        exec(compile(loader.read_text().split('faces=estimator.faces.copy()')[0],str(loader),'exec'),ns)
    torch,np,cv2=ns['torch'],ns['np'],ns['cv2']
    from fast_feedback import install_fast_feedback_cache,FeedbackCall
    model=ns['model'];control=install_fast_feedback_cache(model);forward=ns['forward']
    report=dict(schema='fast_feedback_benchmark_v1',image_size=a.image_size,
                gpu=torch.cuda.get_device_name(),settings=ns['settings'],
                warmup=a.warmup,runs_per_block=a.runs,
                scope='body, TRT backbone+YOLO, decoder/MHR/intermediate compile; full inference includes detector/crop/network/CPU outputs; excludes IO/load/render/export',
                adapter_sha256=hashlib.sha256((ROOT/'fast_feedback.py').read_bytes()).hexdigest(),
                source_revision=__import__('subprocess').check_output(['git','-C',str(path('fast_repo')),'rev-parse','HEAD'],text=True).strip())
    (a.output/'fast_feedback-tested.py').write_bytes((ROOT/'fast_feedback.py').read_bytes())
    def summary(values):
        return dict(mean_ms=float(np.mean(values)),median_ms=float(np.median(values)),p95_ms=float(np.percentile(values,95)),n=len(values))
    def copy_outputs(outputs):
        return [{k:v.copy() if isinstance(v,np.ndarray) else v for k,v in row.items()} for row in outputs]
    def differences(ref,got):
        assert len(ref)==len(got),'person count changed'
        result={}
        for i,(x,y) in enumerate(zip(ref,got)):
            for key in x:
                v,w=x[key],y[key]
                if not isinstance(v,np.ndarray) or not np.issubdtype(v.dtype,np.number):continue
                assert v.shape==w.shape,(key,v.shape,w.shape)
                assert np.isfinite(w).all(),key
                d=np.abs(v.astype(np.float64)-w.astype(np.float64))
                result[f'{i}:{key}']=dict(max_abs=float(d.max()) if d.size else 0.,mean_abs=float(d.mean()) if d.size else 0.,exact=bool(np.array_equal(v,w)))
        return result
    null=open(os.devnull,'w')
    # All CUDA work finishes at each normal forward boundary. Alternate order
    # ABBA/BAAB to reduce clock/temperature drift; never time model construction.
    blocks=[];values={'original':[],'cached':[]}
    with torch.inference_mode(),contextlib.redirect_stdout(null):
        for enabled in (False,True):
            control['enabled']=enabled
            for _ in range(a.warmup):forward()
        for enabled in (False,True,True,False,True,False,False,True):
            control['enabled']=enabled
            for _ in range(3):forward()
            times=[]
            for _ in range(a.runs):
                out,t=forward();times.append(t*1000)
            label='cached' if enabled else 'original'
            values[label]+=times
            blocks.append(dict(variant=label,**summary(times)))
    report['full_inference']={k:summary(v) for k,v in values.items()}
    report['timing_blocks']=blocks
    print('PAIRED_FULL_TIMING',json.dumps(report['full_inference']),flush=True)
    # Boundary instrumentation is isolated from the full-inference timings.
    decoder_forward=model.decoder.forward;call_forward=FeedbackCall.__call__
    active=None
    def timed(name,fn):
        def run(*args,**kwargs):
            if active is None:return fn(*args,**kwargs)
            s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            s.record();r=fn(*args,**kwargs);e.record();active.append((name,s,e));return r
        return run
    def wrapper(*args,**kwargs):
        if not control['enabled'] and kwargs.get('keypoint_token_update_fn') is not None:
            kwargs['keypoint_token_update_fn']=timed('feedback',kwargs['keypoint_token_update_fn'])
        return decoder_forward(*args,**kwargs)
    model.decoder.forward=timed('decoder',wrapper)
    FeedbackCall.__call__=timed('feedback',call_forward)
    rows=[]
    with torch.inference_mode(),contextlib.redirect_stdout(null):
        for enabled in (False,True,True,False):
            control['enabled']=enabled
            for _ in range(3):forward()
            for _ in range(a.runs//2):
                active=[];forward();torch.cuda.synchronize()
                sums={'feedback':0.,'decoder':0.}
                for name,s,e in active:sums[name]+=s.elapsed_time(e)
                rows.append(dict(variant='cached' if enabled else 'original',**sums))
                active=None
    model.decoder.forward=decoder_forward;FeedbackCall.__call__=call_forward
    report['boundaries']={variant:{key:summary([r[key] for r in rows if r['variant']==variant]) for key in ('feedback','decoder')} for variant in values}
    report['boundary_note']='CUDA event intervals include CPU dispatch/stream gaps; instrumented separately, not subtracted from uninstrumented full times'
    # Across-image, mask and schedule tests check cache invalidation as well as
    # exact geometry; no cached original outputs are used to initialize the model.
    samples=[a.image]
    if a.poses:
        manifest=json.loads((a.poses/'pose_set.json').read_text())
        for row in manifest['poses']:
            if row.get('sample'):
                samples.append(a.poses/row['sample']/'rgb.png')
        assert len(samples)>1, 'pose set contains no captures'
    original_image=ns['rgb'];checks=[]
    with torch.inference_mode(),contextlib.redirect_stdout(null):
        for image_path in samples:
            bgr=cv2.imread(str(image_path));assert bgr is not None,str(image_path)
            ns['rgb']=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            control['enabled']=False;ref=copy_outputs(forward()[0])
            control['enabled']=True;got=copy_outputs(forward()[0])
            diff=differences(ref,got)
            checks.append(dict(image=str(image_path),image_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),differences=diff,cache=control['last']))
            # Save paired arrays to allow independent later checks.
            for variant,data in [('original',ref),('cached',got)]:
                np.savez_compressed(a.output/f'case-{len(checks):02d}-{variant}.npz',**{f'p{i}_{k}':v for i,row in enumerate(data) for k,v in row.items() if isinstance(v,np.ndarray)})
        ns['rgb']=original_image
        schedules=[]
        for layers in ('0,1,2','0,2','2','-1'):
            os.environ['BODY_INTERM_PRED_LAYERS']=layers
            control['enabled']=False;ref=copy_outputs(forward()[0])
            control['enabled']=True;got=copy_outputs(forward()[0])
            schedules.append(dict(layers=layers,differences=differences(ref,got),cache=control['last']))
        os.environ['BODY_INTERM_PRED_LAYERS']='0,1,2'
        # Different masks and camera intrinsics in consecutive calls.
        mask_checks=[]
        from mask_prompt import FixedMaskSegmentor
        h,w=ns['rgb'].shape[:2]
        for factor in (1.,.8):
            mask=np.zeros((h,w),np.uint8);mask[int(h*.1):int(h*.95),int(w*.2):int(w*.8)]=1
            if factor<1:mask[:h//3]=0
            k=torch.tensor([[[900*factor,0,w/2],[0,900*factor,h/2],[0,0,1]]],device='cuda',dtype=torch.float32)
            pair=[]
            for enabled in (False,True):
                control['enabled']=enabled
                segmentor=FixedMaskSegmentor(mask);ns['estimator'].sam=segmentor
                try:
                    result=ns['estimator'].process_one_image(ns['rgb'].copy(),inference_type='body',hand_box_source='yolo_pose',cam_int=k,use_mask=True)
                    torch.cuda.synchronize();assert segmentor.used
                    pair.append(copy_outputs(result))
                finally:ns['estimator'].sam=None
            mask_checks.append(dict(focal_factor=factor,differences=differences(*pair),cache=control['last']))
    null.close()
    report.update(output_checks=checks,schedule_checks=schedules,mask_intrinsic_checks=mask_checks)
    all_diffs=[d for case in checks+schedules+mask_checks for d in case['differences'].values()]
    report['all_output_arrays_exact']=all(d['exact'] for d in all_diffs)
    report['max_abs_all_fields']=max(d['max_abs'] for d in all_diffs)
    ref_ms=report['full_inference']['original']['mean_ms'];new_ms=report['full_inference']['cached']['mean_ms']
    report['saving_ms']=ref_ms-new_ms;report['saving_percent']=(ref_ms-new_ms)/ref_ms*100
    (a.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print('BENCHMARK_COMPLETE',a.output/'report.json',flush=True)
    print(json.dumps({k:report[k] for k in ('full_inference','boundaries','all_output_arrays_exact','max_abs_all_fields','saving_ms','saving_percent')},ensure_ascii=False,indent=2),flush=True)
    geometry=[d for case in checks+schedules+mask_checks for k,d in case['differences'].items() if any(name in k for name in ('vertices','keypoints_3d','cam_t','joint_coords'))]
    assert max(d['max_abs'] for d in geometry)<1e-5,'geometry changed by >=0.01mm'

if __name__=='__main__':main()
