# SPDX-License-Identifier: GPL-3.0-only
"""Offline startup warmup, using a local photo without opening any camera."""
import json
from pathlib import Path
import time


def load_sample(filename):
    import cv2
    import numpy as np
    filename=Path(filename).expanduser().resolve()
    image=cv2.imread(str(filename))
    if image is None:raise ValueError('无法读取离线预热图片：'+str(filename))
    h,w=image.shape[:2]
    if w*h>4096*2160:raise ValueError('离线预热图片过大')
    matrix=None;source='estimated'
    for candidate in (filename.parent/'camera.json',filename.parent.parent/'manifest.json'):
        if not candidate.is_file():continue
        data=json.loads(candidate.read_text())
        profile=data.get('profile',data);intr=profile.get('intrinsics')
        if not intr:continue
        iw,ih=int(intr.get('width',w)),int(intr.get('height',h))
        if iw!=w or ih!=h:raise ValueError('预热图片尺寸与内参不一致')
        matrix=np.array([[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1]],np.float32)
        source=str(candidate);break
    if matrix is None:matrix=np.array([[np.hypot(w,h),0,w/2],[0,np.hypot(w,h),h/2],[0,0,1]],np.float32)
    if not np.isfinite(matrix).all() or min(matrix[0,0],matrix[1,1])<=0:raise ValueError('离线预热图片内参无效')
    return cv2.cvtColor(image,cv2.COLOR_BGR2RGB),matrix,source


def camera_variant(rgb,matrix,width=1280,height=800):
    """Cover the Studio's Gemini layout, adjusting intrinsics with the image."""
    import cv2
    h,w=rgb.shape[:2]
    if (w,h)==(width,height):return rgb,matrix.copy()
    scale=min(width/w,height/h);rw,rh=round(w*scale),round(h*scale)
    resized=cv2.resize(rgb,(rw,rh)) if (rw,rh)!=(w,h) else rgb
    left,top=(width-rw)//2,(height-rh)//2
    result=cv2.copyMakeBorder(resized,top,height-rh-top,left,width-rw-left,cv2.BORDER_CONSTANT,value=0)
    k=matrix.copy();k[0]*=rw/w;k[1]*=rh/h;k[0,2]+=left;k[1,2]+=top
    return result,k


def warmup_local_sample(automatic,filename,stage=lambda message:None):
    """Warm real detector + B1/B2 hand paths. Never save predicted sample data."""
    began=time.perf_counter();rgb,matrix,origin=load_sample(filename)
    variants=[(rgb,matrix)]
    if rgb.shape[:2]!=(800,1280):variants.append(camera_variant(rgb,matrix))
    layouts=[];detected=[];synthetic=False
    automatic.tracker.reset()
    engine=getattr(automatic,'engine',None)
    previous_request=getattr(engine,'temporal_request',None)
    # Exercise the optional final-MHR signature before the camera starts.
    if engine is not None:engine.temporal_request=(0.,dict(pose=True,position=True,min_cutoff=1.5,beta=.5,d_cutoff=1.))
    try:
        for image,k in variants:
            h,w=image.shape[:2];layouts.append([w,h])
            stage(f'离线预热：{w}×{h}，检测器与实际手部批量 ×3')
            for _ in range(3):
                _,_,report=automatic.process(image,confidence=.4,padding=1.1,matrix=k,stabilize=False)
            hands=report['hands']
            if not hands:raise ValueError('预热样本未检测到手，请换一张手部清晰的图片')
            detected.append(len(hands))
            for batch in (1,2):
                if len(hands)==batch:continue
                samples=hands[:batch]
                if len(samples)<batch:
                    samples=[dict(hands[0]),dict(hands[0])];synthetic=True
                stage(f'离线预热：{w}×{h}，补充 B{batch} ×3')
                for _ in range(3):automatic.engine.predict_many(image,samples,padding=1.1,matrix=k)
    finally:
        if engine is not None:engine.temporal_request=previous_request
        if hasattr(engine,'_temporal_filter'):engine._temporal_filter.reset()
        automatic.tracker.reset()
        automatic._hand_stream_id=None
    return dict(status='ready',one_euro_prewarmed=True,source=str(Path(filename).expanduser().resolve()),
                seconds=time.perf_counter()-began,batch_sizes=[1,2],image_sizes=layouts,
                detected_hands=detected,duplicated_crop_for_batch2=synthetic,
                intrinsics_source=origin,detector=automatic.detector.metadata,
                scope='offline sample startup warmup; no camera access or sample mesh export')
