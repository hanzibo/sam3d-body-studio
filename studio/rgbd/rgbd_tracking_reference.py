# SPDX-License-Identifier: GPL-3.0-only
"""Immutable manually reviewed reference revisions for RGB-D tracking."""
import json
from pathlib import Path
import cv2
import numpy as np
from studio.video.video_project import load_project,mask_at,require_masks,annotation_key,file_hash,save_json


def reference_directory(directory):
    directory=Path(directory).resolve();pointer=directory/'active-reference.json'
    if not pointer.exists():return directory/'reference_masks'
    entry=json.loads(pointer.read_text());folder=(directory/entry['directory']).resolve();folder.relative_to(directory)
    if file_hash(folder/'manifest.json')!=entry['manifest_sha256']:raise ValueError('active reference has changed')
    return folder


def export_reviewed_reference(directory):
    directory,project=load_project(directory)
    if project.get('source_kind')!='rgbd_tracking':raise ValueError('仅用于 RGB-D 跟踪分割任务')
    if len(project['tracks'])!=1:raise ValueError('当前实验为单人物，请只保留一个人物 ID')
    require_masks(directory,project)
    source=json.loads((directory/'inputs.json').read_text())
    annotation=annotation_key(project);revision=directory/'references'/('reviewed-'+annotation[:20]);manifest=revision/'manifest.json'
    if not manifest.exists():
        revision.mkdir(parents=True,exist_ok=False);rows=[];track=project['tracks'][0]
        for i in range(project['frame_count']):
            maskfile=mask_at(directory,project,track,i)
            mask=cv2.imread(str(maskfile),0) if maskfile else np.zeros((project['height'],project['width']),np.uint8)
            if mask is None or mask.shape!=(project['height'],project['width']):raise ValueError('mask dimensions differ')
            target=revision/f'{i:06d}.png'
            if not cv2.imwrite(str(target),mask):raise OSError(str(target))
            rows.append(dict(frame=i,source_index=source['frames'][i]['source_index'],time_s=source['frames'][i]['time_s'],pixels=int((mask>0).sum()),
                             sha256=file_hash(target),segmented=maskfile is not None,source=str(maskfile) if maskfile else 'outside manually configured segments; not independently segmented'))
        save_json(manifest,dict(grid='undistorted colour',source='user-confirmed point-prompt SAM3 reference, not anatomical ground truth',
                               annotations=annotation,inputs_sha256=file_hash(directory/'inputs.json'),project=project,frames=rows))
    pointer=dict(directory=str(revision.relative_to(directory)),manifest_sha256=file_hash(manifest),annotations=annotation)
    save_json(directory/'active-reference.json',pointer)
    return revision
