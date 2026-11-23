#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Paired, warmed full-inference and decoder timing; final-core replay and body synchronization/transfer ablation."""
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
from studio.core.gpu_lease import acquire_gpu_lease


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
    from studio.core.offline_dinov3 import local_dinov3_source
    ns={'__name__':'__main__','__file__':str(loader)}
    with local_dinov3_source():
        exec(compile(loader.read_text().split('faces=estimator.faces.copy()')[0],str(loader),'exec'),ns)
    torch,np,cv2=ns['torch'],ns['np'],ns['cv2']
    from studio.body.fast_feedback import install_fast_feedback_cache,FeedbackCall
    model=ns['model'];feedback=install_fast_feedback_cache(model);forward=ns['forward']
    from studio.body.fast_keypoints import install_fast_keypoint_path
    keypoints=install_fast_keypoint_path(model)
    from studio.body.fast_execution import install_final_body_replay,install_body_io
    final=install_final_body_replay(model);io=install_body_io(ns['estimator'])
    class Switch(dict):
        def __setitem__(self,key,value):
            super().__setitem__(key,value)
            if key=='enabled':final.enabled=value;io['enabled']=value
    control=Switch(version='final_and_io_v1')
    report=dict(schema='fast_execution_benchmark_v1',image_size=a.image_size,
                gpu=torch.cuda.get_device_name(),settings=ns['settings'],
                warmup=a.warmup,runs_per_block=a.runs,
                scope='body, TRT backbone+YOLO, decoder/MHR/intermediate compile; full inference includes detector/crop/network/CPU outputs; excludes IO/load/render/export',
                adapter_sha256=hashlib.sha256((ROOT/'studio/body/fast_execution.py').read_bytes()).hexdigest(),
                source_revision=__import__('subprocess').check_output(['git','-C',str(path('fast_repo')),'rev-parse','HEAD'],text=True).strip())
    (a.output/'fast_execution-tested.py').write_bytes((ROOT/'studio/body/fast_execution.py').read_bytes())
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
    blocks=[];values={'baseline':[],'optimized':[]}
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
            label='optimized' if enabled else 'baseline'
            values[label]+=times
            blocks.append(dict(variant=label,**summary(times)))
    report['keypoint_path']=dict(control)
    report['feedback_cache']=dict(feedback)
    report['feedback_adapter_sha256']=hashlib.sha256((ROOT/'studio/body/fast_feedback.py').read_bytes()).hexdigest()
    (a.output/'fast_feedback-tested.py').write_bytes((ROOT/'studio/body/fast_feedback.py').read_bytes())
    report['baseline_note']='Both variants retain feedback cache and MHR70 graph. Baseline=previous optimized version; optimized=final full-core graph plus quiet body IO.'
    report['full_inference']={k:summary(v) for k,v in values.items()}
    report['timing_blocks']=blocks
    print('PAIRED_FULL_TIMING',json.dumps(report['full_inference']),flush=True)
    ablations={name:[] for name in ('baseline','final_only','io_only','both')}
    with torch.inference_mode(),contextlib.redirect_stdout(null):
        for name,use_final,use_io in [('baseline',False,False),('final_only',True,False),('io_only',False,True),('both',True,True),('both',True,True),('io_only',False,True),('final_only',True,False),('baseline',False,False)]:
            final.enabled=use_final;io['enabled']=use_io
            for _ in range(3):forward()
            for _ in range(a.runs):ablations[name].append(forward()[1]*1000)
    report['ablations']={name:summary(v) for name,v in ablations.items()}
    report['final_replay']=final.status()
    report['body_io']=dict(io)
    print('ABLATIONS',json.dumps(report['ablations']),flush=True)

    # Boundary instrumentation is isolated from the full-inference timings.
    decoder_forward=model.decoder.forward;slim_forward=model._compiled_interm_pred_body
    active=None
    def timed(name,fn):
        def run(*args,**kwargs):
            if active is None:return fn(*args,**kwargs)
            s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            s.record();r=fn(*args,**kwargs);e.record();active.append((name,s,e));return r
        return run
    model.decoder.forward=timed('decoder',decoder_forward)
    model._compiled_interm_pred_body=timed('final_core',slim_forward)
    rows=[]
    with torch.inference_mode(),contextlib.redirect_stdout(null):
        for enabled in (False,True,True,False):
            control['enabled']=enabled
            for _ in range(3):forward()
            for _ in range(a.runs//2):
                active=[];forward();torch.cuda.synchronize()
                sums={'final_core':0.,'decoder':0.}
                for name,s,e in active:sums[name]+=s.elapsed_time(e)
                rows.append(dict(variant='optimized' if enabled else 'baseline',**sums))
                active=None
    model.decoder.forward=decoder_forward;model._compiled_interm_pred_body=slim_forward
    report['boundaries']={variant:{key:summary([r[key] for r in rows if r['variant']==variant]) for key in ('final_core','decoder')} for variant in values}
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
            checks.append(dict(image=str(image_path),image_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),differences=diff,path=dict(control)))
            # Save paired arrays to allow independent later checks.
            for variant,data in [('baseline',ref),('optimized',got)]:
                np.savez_compressed(a.output/f'case-{len(checks):02d}-{variant}.npz',**{f'p{i}_{k}':v for i,row in enumerate(data) for k,v in row.items() if isinstance(v,np.ndarray)})
        ns['rgb']=original_image
        schedules=[]
        for layers in ('0,1,2','0,2','2','-1'):
            os.environ['BODY_INTERM_PRED_LAYERS']=layers
            control['enabled']=False;ref=copy_outputs(forward()[0])
            control['enabled']=True;got=copy_outputs(forward()[0])
            schedules.append(dict(layers=layers,differences=differences(ref,got),path=dict(control)))
        os.environ['BODY_INTERM_PRED_LAYERS']='0,1,2'
        # Different masks and camera intrinsics in consecutive calls.
        mask_checks=[]
        from studio.segmentation.mask_prompt import FixedMaskSegmentor
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
            projection=[]
            for variant,data in zip(('baseline','optimized'),pair):
                np.savez_compressed(a.output/f'mask-{factor}-{variant}.npz',**{f'p{i}_{k}':v for i,row in enumerate(data) for k,v in row.items() if isinstance(v,np.ndarray)})
                xy=data[0]['pred_keypoints_2d'];z=data[0]['pred_keypoints_3d'][:,2]+data[0]['pred_cam_t'][2]
                projection.append(dict(variant=variant,xy_min=xy.min(axis=0).tolist(),xy_max=xy.max(axis=0).tolist(),camera_z_min=float(z.min()),camera_z_max=float(z.max())))
            mask_checks.append(dict(focal_factor=factor,differences=differences(*pair),path=dict(control),projection=projection))
    # Graph replay must consume fresh parameters and preserve earlier outputs.
    selected=model.head_pose._studio_selected_mhr70
    with torch.inference_mode():
        torch.manual_seed(19)
        shape=torch.randn(1,45,device='cuda')*.2
        params=torch.randn(1,204,device='cuda')*.03
        expression=torch.randn(1,72,device='cuda')*.1
        first=selected.inference(shape,params,expression)
        saved=tuple(x.clone() for x in first)
        second=selected.inference(shape+.1,params+.01,expression-.1)
        expected=selected(shape+.1,params+.01,expression-.1)
        for old,snapshot in zip(first,saved):
            torch.testing.assert_close(old,snapshot,atol=0,rtol=0)
        replay_error=max((x-y).abs().max().item() for x,y in zip(second,expected))
        assert replay_error<1e-5, replay_error
        assert not torch.equal(first[0],second[0]), 'stale graph inputs'
        report['replay_check']=dict(previous_outputs_preserved=True,fresh_inputs=True,max_abs_error_m=replay_error)
    with torch.inference_mode():
        final.enabled=True
        args=tuple(x.clone() if isinstance(x,torch.Tensor) else x for x in final.cache[1])
        previous=final(*args);saved=tuple(x.clone() for x in previous)
        changed=(args[0]+.001,)+args[1:]
        current=final(*changed);expected=final.original(*changed)
        for old,snapshot in zip(previous,saved):torch.testing.assert_close(old,snapshot,atol=0,rtol=0)
        errors=[float((x-y).abs().max()) for x,y in zip(current,expected)]
        assert errors[8]<1e-5 and not torch.equal(current[8],previous[8])
        report['final_replay_check']=dict(fresh_inputs=True,previous_outputs_preserved=True,max_abs_per_output=errors)
        from studio.body.fast_execution import packed_to_cpu
        from sam_3d_body.utils import recursive_to
        raw=dict(a=torch.arange(24,device='cuda',dtype=torch.float32).reshape(2,12),b=torch.arange(6,device='cuda',dtype=torch.int64),c=None)
        first=packed_to_cpu(raw,recursive_to,set(raw));saved_cpu=first['a'].clone()
        second=packed_to_cpu(dict(raw,a=raw['a']+9),recursive_to,set(raw))
        torch.testing.assert_close(first['a'],saved_cpu,atol=0,rtol=0)
        for k in ('a','b'):torch.testing.assert_close(first[k],raw[k].cpu(),atol=0,rtol=0)
        torch.testing.assert_close(second['a'],raw['a'].cpu()+9,atol=0,rtol=0)
        report['packed_transfer_check']=dict(mixed_dtypes_exact=True,previous_outputs_preserved=True)
    report['final_replay']=final.status()
    report['body_io']=dict(io)
    null.close()
    report.update(output_checks=checks,schedule_checks=schedules,mask_intrinsic_checks=mask_checks)
    all_diffs=[d for case in checks+schedules+mask_checks for d in case['differences'].values()]
    report['all_output_arrays_exact']=all(d['exact'] for d in all_diffs)
    report['max_abs_all_fields']=max(d['max_abs'] for d in all_diffs)
    ref_ms=report['full_inference']['baseline']['mean_ms'];new_ms=report['full_inference']['optimized']['mean_ms']
    report['saving_ms']=ref_ms-new_ms;report['saving_percent']=(ref_ms-new_ms)/ref_ms*100
    (a.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print('BENCHMARK_COMPLETE',a.output/'report.json',flush=True)
    print(json.dumps({k:report[k] for k in ('full_inference','boundaries','all_output_arrays_exact','max_abs_all_fields','saving_ms','saving_percent')},ensure_ascii=False,indent=2),flush=True)
    geometry=[d for case in checks+schedules+mask_checks for k,d in case['differences'].items() if any(name in k for name in ('vertices','keypoints_3d','cam_t','joint_coords'))]
    assert max(d['max_abs'] for d in geometry)<1e-5,'geometry changed by >=0.01mm'

if __name__=='__main__':main()
