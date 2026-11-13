#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Profile existing compiled Fast decoder boundaries without editing upstream.
CUDA-event intervals include stream idle/host dispatch; not pure kernel time.
The compiled intermediate core fuses pose head, MHR and camera projection.
"""
import argparse, contextlib, hashlib, io, json, os, statistics, sys, time
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from studio_paths import environment, path
from gpu_lease import acquire_gpu_lease

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--runs',type=int,default=40)
    p.add_argument('--warmup',type=int,default=10)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    lease=acquire_gpu_lease('exclusive')
    os.environ.update(environment('fast'))
    sys.path.insert(0,str(path('fast_repo')))
    # Execute the verified loader and its normal warmup/baseline, stopping before
    # exports/sys.exit. No source checkout or compiler flags are modified.
    loader=Path('/home/hzb/Work/fast-sam3d-body-deploy/run.py')
    sys.argv=[str(loader),'--image',str(a.image),'--output',str(a.output),
              '--compile','--trt-backbone',str(path('backbone_engine_384')),
              '--trt-yolo',str(path('yolo_engine')),'--image-size','384',
              '--inference-type','body','--warmup',str(a.warmup),'--runs',str(a.runs)]
    code=loader.read_text().split('faces=estimator.faces.copy()')[0]
    ns={'__name__':'__main__','__file__':str(loader)}
    from offline_dinov3 import local_dinov3_source
    with local_dinov3_source():
        exec(compile(code,str(loader),'exec'),ns)
    torch=ns['torch'];np=ns['np'];model=ns['model'];estimator=ns['estimator']
    forward=ns['forward'];baseline_outputs=ns['outputs']
    def summarize(v):
        return dict(mean_ms=float(np.mean(v)),median_ms=float(np.median(v)),
                    p95_ms=float(np.percentile(v,95)),n=len(v))
    report=dict(schema='compiled_fast_decoder_profile_v1',image=str(a.image),
        image_sha256=hashlib.sha256(a.image.read_bytes()).hexdigest(),
        loader=str(loader),loader_sha256=hashlib.sha256(loader.read_bytes()).hexdigest(),
        gpu=torch.cuda.get_device_name(),config=ns['settings'],
        baseline_before=summarize(np.array(ns['times'])*1000),
        boundaries='CUDA stream event intervals plus CPU call wall time; no inner synchronize',
        caveat='Compiled intermediate/final cores fuse pose head + MHR + camera projection; categories are not pure MHR kernel times')
    rows=[];current=None;restores=[]
    def timed(name,fn):
        def wrapped(*args,**kwargs):
            if current is None:return fn(*args,**kwargs)
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            start.record();t=time.perf_counter()
            result=fn(*args,**kwargs)
            cpu=(time.perf_counter()-t)*1000;end.record()
            current.append((name,start,end,cpu))
            return result
        return wrapped
    def patch(obj,name,label):
        original=getattr(obj,name);setattr(obj,name,timed(label,original));restores.append((obj,name,original))
    patch(model,'forward_decoder','body_total')
    patch(model,'_compiled_interm_pred_body_slim','intermediate_core')
    patch(model,'_compiled_interm_pred_body','final_core')
    for i,layer in enumerate(model.decoder.layers):patch(layer,'_call_impl',f'transformer_layer_{i}')
    decoder_forward=model.decoder.forward
    def decoder_wrapper(*args,**kwargs):
        for name,label in [('token_to_pose_output_fn','pose_callback'),('keypoint_token_update_fn','keypoint_feedback')]:
            if kwargs.get(name) is not None:kwargs[name]=timed(label,kwargs[name])
        return decoder_forward(*args,**kwargs)
    model.decoder.forward=decoder_wrapper
    restores.append((model.decoder,'forward',decoder_forward))
    # Permit any changed guards to settle before recording instrumentation.
    with torch.inference_mode(),contextlib.redirect_stdout(io.StringIO()):
        for _ in range(a.warmup):forward()
        for i in range(a.runs):
            current=[];outputs,seconds=forward();torch.cuda.synchronize()
            row=dict(pipeline_ms=seconds*1000,intervals=[dict(name=n,stream_ms=s.elapsed_time(e),cpu_call_ms=c) for n,s,e,c in current])
            rows.append(row);current=None
    for obj,name,original in reversed(restores):setattr(obj,name,original)
    max_diff=float(np.max(np.abs(outputs[0]['pred_vertices']-baseline_outputs[0]['pred_vertices'])))
    assert max_diff<1e-4,('Profiling changed output',max_diff)
    aggregate=defaultdict(list);calls=defaultdict(list)
    for row in rows:
        totals=defaultdict(float);counts=defaultdict(int)
        for interval in row['intervals']:
            totals[interval['name']]+=interval['stream_ms'];counts[interval['name']]+=1
        for n,v in totals.items():aggregate[n].append(v);calls[n].append(counts[n])
    report.update(instrumented=summarize([r['pipeline_ms'] for r in rows]),
        components={n:dict(**summarize(v),calls_per_frame=statistics.mean(calls[n])) for n,v in aggregate.items()},
        max_vertex_difference_m=max_diff,frames=rows)
    after=[]
    with torch.inference_mode(),contextlib.redirect_stdout(io.StringIO()):
        for _ in range(5):forward()
        for _ in range(a.runs):after.append(forward()[1]*1000)
    report['baseline_after']=summarize(after)
    c=report['components'];layer_total=sum(v['mean_ms'] for k,v in c.items() if k.startswith('transformer_layer_'))
    report['derived']=dict(transformer_layers_total_ms=layer_total,
        callback_unpack_clone_ms=c['pose_callback']['mean_ms']-c['intermediate_core']['mean_ms']-c['final_core']['mean_ms'],
        other_decoder_ms=c['body_total']['mean_ms']-layer_total-c['pose_callback']['mean_ms']-c['keypoint_feedback']['mean_ms'],
        pipeline_instrumentation_delta_vs_bracketing_baseline_ms=report['instrumented']['mean_ms']-(report['baseline_before']['mean_ms']+report['baseline_after']['mean_ms'])/2)
    # Actual end-to-end ablation of the already-supported intermediate settings.
    # These measurements do not include instrumentation.
    ablations={}
    with torch.inference_mode(),contextlib.redirect_stdout(io.StringIO()):
        for count,layers in [(3,'0,1,2'),(2,'0,2'),(1,'2'),(0,'-1')]:
            os.environ['BODY_INTERM_PRED_LAYERS']=layers
            for _ in range(a.warmup):forward()
            values=[]
            for _ in range(a.runs):result,elapsed=forward();values.append(elapsed*1000)
            delta=np.linalg.norm(result[0]['pred_vertices']-baseline_outputs[0]['pred_vertices'],axis=1)*1000
            ablations[str(count)]=dict(**summarize(values),layers=layers,mean_vertex_difference_mm=float(delta.mean()),max_vertex_difference_mm=float(delta.max()),note='Output change versus 3-feedback reference, not accuracy error')
    os.environ['BODY_INTERM_PRED_LAYERS']='0,1,2'
    report['intermediate_ablations']=ablations
    f=a.output/'profile.json';f.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print('PROFILE_COMPLETE',f,flush=True)
    print(json.dumps({k:v for k,v in report.items() if k!='frames'},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
