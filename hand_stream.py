# SPDX-License-Identifier: GPL-3.0-only
"""Bounded hand stream request: RGB in memory, one transient scene on disk."""
import base64
import os
from pathlib import Path
import time
import cv2
import numpy as np


def process_hand_frame(automatic,command):
    began=time.perf_counter();worker_start=time.monotonic()
    if 'jpeg' in command:
        if len(command['jpeg'])>12*1024*1024:raise ValueError('相机帧过大')
        encoded=base64.b64decode(command['jpeg'],validate=True)
    else:
        # A single in-flight request owns the transient JPEG until result/stop.
        directory=Path(command['output']).resolve()
        source=Path(command['image']).resolve()
        if source.parent!=directory or source.name!='input.jpg':raise ValueError('实时图像路径越界')
        if source.stat().st_size>8*1024*1024:raise ValueError('相机帧过大')
        encoded=source.read_bytes()
    if len(encoded)>8*1024*1024:raise ValueError('相机帧过大')
    image=cv2.imdecode(np.frombuffer(encoded,np.uint8),cv2.IMREAD_COLOR)
    if image is None or image.ndim!=3:raise ValueError('无法解码相机彩色帧')
    h,w=image.shape[:2]
    if w*h>4096*2160:raise ValueError('相机分辨率过大')
    matrix=np.asarray(command['camera_matrix'],np.float32)
    if matrix.shape!=(3,3) or not np.isfinite(matrix).all() or min(matrix[0,0],matrix[1,1])<=0:raise ValueError('相机内参无效')
    stream_id=command['stream_id']
    if getattr(automatic,'_hand_stream_id',None)!=stream_id:
        automatic.tracker.reset();automatic._hand_stream_id=stream_id
        automatic._hand_stream_runtime_sent=False
        if hasattr(automatic.engine,'_temporal_filter'):automatic.engine._temporal_filter.reset()
    rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB);warm=bool(command.get('warmup',False))
    decode_done=time.monotonic()
    count=3 if warm else 1
    from hand_temporal import filter_options
    options=filter_options(command.get('one_euro'))
    signature=(command.get('hand_selection','both'),tuple(matrix.ravel()),(w,h))
    if getattr(automatic,'_hand_filter_signature',None)!=signature:
        if hasattr(automatic.engine,'_temporal_filter'):automatic.engine._temporal_filter.reset()
        automatic._hand_filter_signature=signature
    automatic.engine.temporal_request=(command['received_monotonic'],options) if not warm and (options['pose'] or options['position']) else None
    if automatic.engine.temporal_request is None and hasattr(automatic.engine,'_temporal_filter'):automatic.engine._temporal_filter.reset()
    try:
        for _ in range(count):
            data,outputs,report=automatic.process(rgb,confidence=command.get('confidence',.4),padding=command.get('padding',1.1),matrix=matrix,timestamp=command['received_monotonic'],stabilize=not warm,hand_selection=command.get('hand_selection','both'))
        if not outputs and hasattr(automatic.engine,'_temporal_filter'):automatic.engine._temporal_filter.reset()
    except Exception:
        if hasattr(automatic.engine,'_temporal_filter'):automatic.engine._temporal_filter.reset()
        raise
    finally:automatic.engine.temporal_request=None
    report['one_euro']=outputs[0][3].get('one_euro',dict(**options,processing_ms=0.,history_frames=[])) if outputs else dict(**options,processing_ms=0.,history_frames=[])
    if warm and report['hands']:
        # Warm both actual tensor batch sizes even when only one hand is visible.
        # Repeated crop is used only for graph/compiler warmup, never displayed.
        other=1 if len(report['hands'])==2 else 2
        samples=report['hands'][:1]*other
        for _ in range(3):automatic.engine.predict_many(rgb,samples,padding=command.get('padding',1.1),matrix=matrix)
        report['warmup_batch_sizes']=[1,2]
    inference_done=time.monotonic()
    directory=Path(command['output']);directory.mkdir(parents=True,exist_ok=True)
    filename=directory/'scene.npz';temporary=directory/'scene.pending.npz'
    if command.get('compact_result',False):
        # Live display consumes these arrays only. Full photo/sequence exports
        # retain native parameters and all per-hand diagnostics unchanged.
        data={key:value for key,value in data.items() if key in ('vertices','faces','joints','edges','camera_matrix') or ('_raw_' in key and report.get('one_euro',{}).get('raw_preserved'))}
        fields=('task','hand_count','hands','detect_ms','detector_timings','reconstruct_ms','warmup_batch_sizes','detection_options','one_euro')
        compact={key:report[key] for key in fields if key in report}
        if report.get('hand_reports'):
            first=report['hand_reports'][0]
            compact['frame_execution']={key:first[key] for key in ('execution_counts','backbone_engine_executions') if key in first}
            if 'executions' in (first.get('backbone_runtime') or {}):
                compact['frame_execution']['total_backbone_executions']=first['backbone_runtime']['executions']
        if not automatic._hand_stream_runtime_sent and report.get('hand_reports'):
            compact['hand_reports']=report['hand_reports'][:1]
            compact['detector']=report.get('detector')
            automatic._hand_stream_runtime_sent=True
        report=compact
    np.savez(temporary,**data);os.replace(temporary,filename)
    report['camera_source']='live_camera_metadata';report['image_size']=[w,h]
    finished=time.monotonic()
    stage_timings=dict(read_decode_ms=(decode_done-worker_start)*1000,inference_wall_ms=(inference_done-decode_done)*1000,scene_write_ms=(finished-inference_done)*1000)
    return dict(worker_start_monotonic=worker_start,worker_finished_monotonic=finished,stage_timings=stage_timings,path=str(filename),frame_index=command['frame_index'],stream_id=stream_id,warmup=warm,report=report,worker_ms=(time.perf_counter()-began)*1000)
