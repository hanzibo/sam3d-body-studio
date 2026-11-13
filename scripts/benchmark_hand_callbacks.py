# SPDX-License-Identifier: GPL-3.0-only
"""Paired callback compilation and feedback schedule benchmark, fixed boxes."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from studio_paths import path
sys.path.insert(0,str(path('official_repo')))
from hand_config import HandExecution
from hand_engine import HandEngine
from gpu_lease import acquire_gpu_lease
import cv2,numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=20)
    parser.add_argument('--cases',type=int,default=4)
    parser.add_argument('--backbone',choices=['trt_fp16','trt_int8'],default='trt_fp16')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    report=args.output/'paired.json'
    if report.exists():raise FileExistsError(report)
    lease=acquire_gpu_lease('exclusive')
    config=HandExecution(backbone=args.backbone,detector='trt_fp16',decoder_compile=True,mhr_compile=True,mhr_graph=True,feedback_cache=True,intermediate_keypoints=True,final_hand_vertices=True,intermediate_compile=True)
    engine=HandEngine(lambda message:print(message,flush=True),config)
    control=engine.model._studio_hand_feedback
    variants={'baseline5':(False,5),**{f'compiled{n}':(True,n) for n in (5,4,3,2,1,0)}}
    def select(name):
        compile,count=variants[name];control['callback_compile']=compile;control['intermediate_layers']=list(range(count))
    source=Path('/run/media/hzb/Data/experiments/hand-auto-20261112/full')
    cases=[]
    for sequence,index in [('20261112_173053_single_hand_d4f1fc',150),('20261112_173116_two_hands_acdf29',150),('20261112_173053_single_hand_d4f1fc',324),('20261112_173116_two_hands_acdf29',449)][:args.cases]:
        info=json.loads((source/sequence/'sequence.json').read_text())
        row=json.loads((source/sequence/'frames.jsonl').read_text().splitlines()[index])
        rgb=cv2.cvtColor(cv2.imread(str(Path(info['source'])/row['input_file'])),cv2.COLOR_BGR2RGB)
        hands=[dict(bbox=h['bbox'],side=h['side']) for h in row['report']['hands']]
        def predict():return engine.predict_many(rgb,hands,1.1,info['camera_matrix'])
        results={};samples={name:[] for name in variants}
        for name in variants:
            select(name);began=time.monotonic()
            for _ in range(4):results[name]=predict()
            print('WARM',sequence,index,name,round(time.monotonic()-began,2),results[name][0][3]['feedback_runtime']['last'],flush=True)
        for iteration in range(args.repeats):
            for name in list(variants) if iteration%2==0 else list(variants)[::-1]:
                select(name);results[name]=predict()
                r=results[name][0][3];samples[name].append(dict(inference_ms=r['inference_ms'],**r['timings']))
        stats={name:{k:dict(mean=float(np.mean([v[k] for v in rows])),p95=float(np.percentile([v[k] for v in rows],95))) for k in rows[0]} for name,rows in samples.items()}
        differences={};arrays={}
        for name,result in results.items():
            errors={k:[] for k in ('camera_vertices_mm','joints_mm','projection_px','mhr_params')}
            for h,(ref,cur) in enumerate(zip(results['baseline5'],result)):
                for key in ('vertices','vertices_2d','joints','camera_translation'):
                    arrays[f'{name}_{h}_{key}']=cur[0][key]
                errors['camera_vertices_mm'].extend(np.linalg.norm(cur[0]['vertices']+cur[0]['camera_translation']-ref[0]['vertices']-ref[0]['camera_translation'],axis=-1)*1000)
                for key,label,scale in [('joints','joints_mm',1000),('vertices_2d','projection_px',1)]:
                    errors[label].extend(np.linalg.norm(cur[0][key]-ref[0][key],axis=-1)*scale)
                errors['mhr_params'].extend(np.abs(cur[1]['mhr_model_params']-ref[1]['mhr_model_params']).ravel())
            differences[name]={k:dict(mean=float(np.mean(v)),max=float(np.max(v))) for k,v in errors.items()}
            count=variants[name][1];runtime=result[0][3]['feedback_runtime']['last']
            assert runtime['intermediate_keypoint_calls']==count,runtime
            assert runtime['compact_surface_calls']==1,runtime
            assert runtime['feedback_computations']==count,runtime
            if name=='compiled5':
                assert differences[name]['camera_vertices_mm']['max']<.05,differences[name]
                assert differences[name]['projection_px']['max']<.02,differences[name]
        np.savez_compressed(args.output/f'{sequence}-{index}.npz',**arrays)
        case=dict(sequence=sequence,index=index,hands=len(hands),stats=stats,differences=differences,runtimes={k:v[0][3]['feedback_runtime']['last'] for k,v in results.items()})
        cases.append(case)
        report.write_text(json.dumps(dict(scope='Fixed hand boxes; paired forward/reverse after warmup. Includes crop/H2D/network/MHR/sync; excludes detection/D2H/export/render. Differences are relative to the same-encoder five-feedback baseline, not true accuracy.',execution=config.to_dict(),variants={k:dict(intermediate_compile=c,intermediate_predictions=n,layers=list(range(n))) for k,(c,n) in variants.items()},cases=cases),indent=2))
        print('CASE_DONE',sequence,index,{k:round(v['inference_ms']['mean'],3) for k,v in stats.items()},flush=True)
    print('CALLBACK_BENCHMARK_OK',flush=True)


if __name__=='__main__':main()
