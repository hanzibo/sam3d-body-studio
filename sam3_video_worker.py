# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Bounded SAM3 point-initialized temporal instance propagation; exclusive GPU job."""
import gc
import json
import os
from pathlib import Path
import sys
import time
from gpu_lease import acquire_gpu_lease
from studio_paths import path
from video_project import load_project,frame_path,segment_folder,segment_key,save_json


def segment_video(directory,emit):
    lease=acquire_gpu_lease('sam3')
    os.environ['HF_HUB_OFFLINE']='1';sys.path.insert(0,str(path('sam3_repo')))
    import torch
    import cv2
    import numpy as np
    from sam3.model_builder import build_tracker
    directory,project=load_project(directory)
    total=sum(s['end']-s['start']+1 for t in project['tracks'] for s in t['segments'])
    if not total:raise ValueError('请先保存人物分段提示')
    emit('stage',message='正在加载 SAM3 视频跟踪组件（CPU 卸载、有界历史）')
    model=build_tracker(apply_temporal_disambiguation=False,with_backbone=True,compile_mode=None)
    checkpoint=torch.load(str(path('sam3_checkpoint')),map_location='cpu',weights_only=True,mmap=True)
    checkpoint=checkpoint.get('model',checkpoint)
    weights={}
    for name,value in checkpoint.items():
        if name.startswith('tracker.'):weights[name[len('tracker.'):]]=value
        elif name.startswith('detector.backbone.vision_backbone.'):weights[name[len('detector.'):]]=value
    model.load_state_dict(weights,strict=True);del weights,checkpoint;gc.collect()
    model.cuda().eval();torch.cuda.reset_peak_memory_stats();completed=0
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for track in project['tracks']:
            for segment in track['segments']:
                folder=segment_folder(directory,project,track,segment);folder.mkdir(parents=True,exist_ok=True)
                indices=range(segment['start'],segment['end']+1)
                if (folder/'complete.json').is_file() and all((folder/f'{i:06d}.png').is_file() for i in indices):
                    completed+=len(indices);emit('progress',message=f'复用人物 {track["id"]} 分段',current=completed,total=total);continue
                state=model.init_state(video_height=project['height'],video_width=project['width'],num_frames=1,offload_state_to_cpu=True)
                statuses=[];began=time.monotonic()
                for local,index in enumerate(indices):
                    image=cv2.imread(str(frame_path(directory,index)))
                    if image is None:raise ValueError(f'RGB 帧缺失 {index}')
                    rgb=cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),(model.image_size,model.image_size))
                    tensor=torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).cuda().float().div_(127.5).sub_(1)
                    features=model.forward_image(tensor)
                    state['num_frames']=local+1;state['cached_features']={local:(tensor,features)}
                    if local==0:
                        points=[[p[0]/project['width'],p[1]/project['height']] for p in segment['points']]
                        model.add_new_points_or_box(state,frame_idx=0,obj_id=track['id'],points=points,
                            labels=[p[2] for p in segment['points']],rel_coordinates=True)
                        model.propagate_in_video_preflight(state)
                    result=None
                    for result in model.propagate_in_video(state,start_frame_idx=local,max_frame_num_to_track=0,reverse=False,tqdm_disable=True):pass
                    if result is None or result[0]!=local or result[1]!=[track['id']]:raise RuntimeError('SAM3 帧 / 人物 ID 不一致')
                    logits=result[3][0,0];score=float(result[4].reshape(-1)[0].float().cpu())
                    mask=(logits>0).cpu().numpy() if score>0 else np.zeros((project['height'],project['width']),bool)
                    if not cv2.imwrite(str(folder/f'{index:06d}.png'),mask.astype(np.uint8)*255):raise OSError('保存掩膜失败')
                    statuses.append(dict(frame=index,visible=bool(mask.any()),pixels=int(mask.sum()),score_logit=score))
                    # Retain the conditioning frame plus 32 recent non-conditioning frames.
                    mappings=[state['output_dict']['non_cond_frame_outputs'],state['frames_already_tracked']]
                    mappings += [v['non_cond_frame_outputs'] for v in state['output_dict_per_obj'].values()]
                    for mapping in mappings:
                        for old in list(mapping):
                            if old!=0 and old<local-31:del mapping[old]
                    completed+=1
                    if local%5==0:emit('progress',message=f'SAM3 人物 {track["id"]} · 帧 {index}',current=completed,total=total)
                save_json(folder/'complete.json',dict(key=segment_key(project,track,segment),frames=statuses,
                    seconds=time.monotonic()-began,peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3))
                del state,result,features,tensor;gc.collect();torch.cuda.empty_cache()
    emit('segmented',directory=str(directory),current=completed,total=total)
