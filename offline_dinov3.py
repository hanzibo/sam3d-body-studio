"""Adapt Fast's unpinned torch.hub request to the existing local DINOv3 source.

HF_HUB_OFFLINE does not cover torch.hub's GitHub default-branch probe. Keep this
adapter scoped to construction, without editing the read-only upstream clone.
"""
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def local_dinov3_source():
    import torch.hub
    from studio_paths import path
    directory=path('dinov3_repo')
    if not (directory/'hubconf.py').is_file():
        raise FileNotFoundError('本地 DINOv3 源码缓存缺失：'+str(directory)+
                                '；请检查 dinov3_repo 配置并获取固定版本源码，不会自动联网下载。')
    original=torch.hub.load
    def load(repo_or_dir,*args,**kwargs):
        if str(repo_or_dir) in ('facebookresearch/dinov3','facebookresearch/dinov3:main') and kwargs.get('source','github')=='github':
            kwargs['source']='local'
            print('DINOv3 离线源码：'+str(directory),flush=True)
            return original(str(directory),*args,**kwargs)
        return original(repo_or_dir,*args,**kwargs)
    torch.hub.load=load
    try:yield directory
    finally:torch.hub.load=original
