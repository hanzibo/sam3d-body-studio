"""Synchronized inference timing; caller owns CUDA boundaries and output validation."""
import json
from pathlib import Path
import time
import numpy as np

def measure(command, forward, emit):
    count=command.get('runs',50)
    warm=command.get('warmup_runs',10)
    if type(count) is not int or not 1 <= count <= 200 or type(warm) is not int or not 1 <= warm <= 30:
        raise ValueError('测速次数无效（测量 1–200，预热 1–30）')
    out=Path(command['output']);out.mkdir(parents=True,exist_ok=True)
    report={'complete':False,'warmup_runs':warm,'requested_runs':count,'completed_runs':0,
            'pipeline_seconds':[],'scope':'RGB in memory -> detection/crop/model/CPU outputs; CUDA synchronized; excludes load/decode/export/render'}
    began=time.perf_counter()
    def save():
        temp=out/'benchmark.tmp'
        temp.write_text(json.dumps(report,ensure_ascii=False,indent=2))
        temp.replace(out/'benchmark.json')
    save()
    for index in range(warm):
        forward()
        emit('progress',job=command['job'],message=f'同图测速 · 预热 {index+1}/{warm}')
    for index in range(count):
        outputs,seconds=forward()
        report['pipeline_seconds'].append(seconds)
        report['completed_runs']=index+1
        save()
        emit('progress',job=command['job'],message=f'同图测速 · 测量 {index+1}/{count}')
    values=np.array(report['pipeline_seconds'])*1000
    report.update(complete=True,mean_ms=float(values.mean()),median_ms=float(np.median(values)),
                  p95_ms=float(np.percentile(values,95)),min_ms=float(values.min()),max_ms=float(values.max()),
                  wall_seconds=time.perf_counter()-began)
    save()
    return outputs,seconds,report
