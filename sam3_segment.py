"""One-shot point-prompt SAM3 segmentation. Exits to fully release GPU memory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from gpu_lease import acquire_gpu_lease

from studio_paths import path
REPO=path('sam3_repo')
CHECKPOINT=path('sam3_checkpoint')


def emit(kind,**kwargs):print('@@RGBD '+json.dumps(dict(type=kind,**kwargs),ensure_ascii=False),flush=True)


def segment(request_path,engine=None):
    request=request_path if isinstance(request_path,dict) else json.loads(Path(request_path).read_text())
    output=Path(request['output']);output.mkdir(parents=True,exist_ok=False)
    # Acquire before importing a GPU framework or touching the checkpoint.
    global _lease
    if '_lease' not in globals():_lease=acquire_gpu_lease('sam3')
    lease=_lease
    emit('stage',message='人体模型互斥检查通过，'+('正在加载 SAM3 点提示分割模型…' if engine is None else '复用已加载的 SAM3 处理下一姿态…'))
    os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
    sys.path.insert(0,str(REPO))
    import cv2
    import numpy as np
    from PIL import Image
    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    if not torch.cuda.is_available():raise RuntimeError('SAM3 分割需要可用 CUDA；未更改现有环境')
    free,total=torch.cuda.mem_get_info()
    if engine is None and free<5*1024**3:raise RuntimeError(f'显存不足：当前空闲 {free/1024**3:.2f} GiB，至少需要 5 GiB；请关闭其他占用 GPU 的程序')
    image=Image.open(request['image']).convert('RGB');width,height=image.size
    points=np.asarray(request['points_pixels'],np.float32);labels=np.asarray(request['point_labels'],np.int32)
    if points.ndim!=2 or points.shape[1]!=2 or len(points)!=len(labels) or not len(points) or len(points)>128:
        raise ValueError('分割提示点格式无效，最多 128 点')
    if not np.isfinite(points).all() or np.any(points<0) or np.any(points>=[width,height]) or not np.isin(labels,[0,1]).all() or not (labels==1).any():
        raise ValueError('提示点必须在图片内，且至少需要一个人体前景点')
    rgb_sha=hashlib.sha256(Path(request['image']).read_bytes()).hexdigest()
    if rgb_sha!=request['rgb_sha256']:raise ValueError('原始样本图像已改变，请重新标点')
    torch.cuda.reset_peak_memory_stats();began=time.perf_counter()
    if engine is None:
        model=build_sam3_image_model(checkpoint_path=str(CHECKPOINT),
            bpe_path=str(REPO/'sam3/assets/bpe_simple_vocab_16e6.txt.gz'),load_from_HF=False,
            device='cuda',compile=False,enable_inst_interactivity=True)
        processor=Sam3Processor(model)
        torch.cuda.synchronize();load_seconds=time.perf_counter()-began
    else:model,processor,load_seconds=engine
    emit('stage',message='SAM3 已加载，正在根据前景 / 背景点生成掩膜…')
    began=time.perf_counter()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        state=processor.set_image(image)
        masks,scores,_=model.predict_inst(state,point_coords=points,point_labels=labels,
            multimask_output=True,normalize_coords=True)
    torch.cuda.synchronize();inference_seconds=time.perf_counter()-began
    masks=np.asarray(masks).reshape(-1,height,width)>0;scores=np.asarray(scores).reshape(-1)
    # Prefer masks satisfying prompts, then SAM's predicted quality score.
    xy=np.rint(points).astype(int);xy[:,0]=np.clip(xy[:,0],0,width-1);xy[:,1]=np.clip(xy[:,1],0,height-1)
    satisfied=(masks[:,xy[:,1],xy[:,0]]==(labels[None,:]>0)).sum(axis=1)
    rank=np.lexsort((-np.nan_to_num(scores,nan=-1e6),-satisfied))
    candidates=[]
    for i in range(len(masks)):
        mask=np.uint8(masks[i])*255
        if not cv2.imwrite(str(output/f'mask_{i:02d}.png'),mask):raise OSError('掩膜保存失败')
        candidates.append(dict(index=i,score=float(scores[i]),prompt_matches=int(satisfied[i]),pixels=int(masks[i].sum())))
    selected=int(rank[0])
    if not masks[selected].any():raise ValueError('SAM3 返回空掩膜，请调整提示点')
    report=dict(schema='studio_sam3_points_v1',image=str(Path(request['image']).resolve()),rgb_sha256=rgb_sha,
        points_pixels=points.tolist(),point_labels=labels.tolist(),selected=selected,candidates=candidates,
        load_seconds=load_seconds,inference_seconds=inference_seconds,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
        checkpoint=str(CHECKPOINT),checkpoint_bytes=CHECKPOINT.stat().st_size,
        checkpoint_mtime_ns=CHECKPOINT.stat().st_mtime_ns,
        precision='BF16 autocast',implementation='SAM3 image model predict_inst with foreground/background points',
        meaning='SAM-predicted mask and quality score; not manual ground truth or measured segmentation IoU')
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    emit('segmented',output=str(output),report=report)
    # Keep lease alive through process exit / CUDA destructor; GUI applies only after exit.
    return model,processor,load_seconds


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--request',required=True);args=parser.parse_args()
    try:
        request=json.loads(Path(args.request).read_text())
        if 'requests' in request:
            engine=None;success=0
            for i,item in enumerate(request['requests']):
                emit('stage',message=f"SAM3 批量分割 {i+1}/{len(request['requests'])}")
                try:engine=segment(item,engine);success+=1
                except Exception as exc:emit('segment_error',image=item.get('image'),message=str(exc))
            emit('batch_complete',success=success,total=len(request['requests']))
        else:segment(request)
    except Exception as exc:
        import traceback;traceback.print_exc()
        emit('error',message=str(exc));raise SystemExit(1)
