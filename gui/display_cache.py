# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Bounded read-only preview caches, invalidated by file identity and stat changes."""
from collections import OrderedDict
import hashlib
from pathlib import Path
import numpy as np


def file_token(file):
    file=Path(file);st=file.stat()
    return (str(file.resolve()),st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)


class DisplayCache:
    def __init__(self):self.hashes=OrderedDict();self.predictions=OrderedDict()
    def clear(self):self.hashes.clear();self.predictions.clear()
    @staticmethod
    def remember(cache,key,value,limit):
        cache[key]=value;cache.move_to_end(key)
        while len(cache)>limit:cache.popitem(last=False)
        return value
    def digest(self,file):
        token=file_token(file)
        if token in self.hashes:self.hashes.move_to_end(token);return self.hashes[token]
        value=hashlib.sha256(Path(file).read_bytes()).hexdigest()
        if file_token(file)!=token:raise OSError('文件正在改变，请稍后重试')
        return self.remember(self.hashes,token,value,64)
    def prediction(self,file):
        token=file_token(file)
        if token in self.predictions:self.predictions.move_to_end(token);return token,self.predictions[token]
        with np.load(file,allow_pickle=False) as source:
            arrays={key:source[key] for key in ('pred_vertices','pred_cam_t','faces','pred_keypoints_3d') if key in source}
        if file_token(file)!=token:raise OSError('预测文件正在改变，请稍后重试')
        for array in arrays.values():array.setflags(write=False)
        return token,self.remember(self.predictions,token,arrays,6)
