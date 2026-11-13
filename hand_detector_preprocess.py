# SPDX-License-Identifier: GPL-3.0-only
"""Exact fixed-TRT letterbox with RGB input and reusable upload buffers."""
import numpy as np
import torch
from ultralytics.models.yolo.pose.predict import PosePredictor


class HandRGBPredictor(PosePredictor):
    """For HandDetector's one-RGB-image contract; preserve LetterBox and /255."""
    def preprocess(self,images):
        if isinstance(images,torch.Tensor):return super().preprocess(images)
        if len(images)!=1:raise ValueError('手部检测预处理仅支持单幅 RGB 图像')
        resized=self.pre_transform(images)[0]
        shape=(1,3,*resized.shape[:2]);dtype=torch.float16 if self.model.fp16 else torch.float32
        signature=(shape,dtype,self.device)
        if getattr(self,'_hand_signature',None)!=signature:
            self._hand_host=torch.empty(shape,dtype=torch.uint8,pin_memory=True)
            self._hand_host_array=self._hand_host.numpy()
            self._hand_bytes=torch.empty(shape,dtype=torch.uint8,device=self.device)
            self._hand_float=torch.empty(shape,dtype=dtype,device=self.device)
            self._hand_copy_done=None;self._hand_signature=signature
        # Do not overwrite pinned CPU data while an earlier asynchronous DMA uses it.
        if self._hand_copy_done is not None and not self._hand_copy_done.query():self._hand_copy_done.synchronize()
        np.copyto(self._hand_host_array[0],resized.transpose(2,0,1))
        self._hand_bytes.copy_(self._hand_host,non_blocking=True)
        if self._hand_copy_done is None:self._hand_copy_done=torch.cuda.Event()
        self._hand_copy_done.record()
        self._hand_float.copy_(self._hand_bytes);self._hand_float.div_(255)
        return self._hand_float
