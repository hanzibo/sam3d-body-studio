# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""OS-enforced Studio model lease: SAM3 excludes all body/MHR GPU models.

No CUDA imports. Leases survive Python object cleanup and close only explicitly
or on process exit. Forked compiler helpers do not retain the parent's lease.
Unrelated applications which do not use this lock are outside this mechanism.
"""
import fcntl
import os
from pathlib import Path

LOCK_PATH=Path(__file__).resolve().parent/'outputs/studio-model-gpu.lock'
_DESCRIPTORS=set()


def _after_fork():
    for fd in tuple(_DESCRIPTORS):
        try:os.close(fd)
        except OSError:pass
    _DESCRIPTORS.clear()

os.register_at_fork(after_in_child=_after_fork)


class GpuLease:
    def __init__(self,fd):self.fd=fd
    def close(self):
        if self.fd is not None:
            _DESCRIPTORS.discard(self.fd);os.close(self.fd);self.fd=None
    # Intentionally no destructor: retain until real process exit, even after
    # exceptions/module teardown while CUDA objects are still being destroyed.


def acquire_gpu_lease(kind):
    LOCK_PATH.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(LOCK_PATH,os.O_RDWR|os.O_CREAT|os.O_CLOEXEC,0o600)
    operation=fcntl.LOCK_EX if kind in ('sam3','exclusive') else fcntl.LOCK_SH
    try:fcntl.flock(fd,operation|fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise RuntimeError('显存互斥保护：SAM3 / 离线视频独占任务与其他 SAM 3D Body / MHR GPU 模型不能同时加载。请等待其他 Studio 窗口释放模型后重试。') from None
    _DESCRIPTORS.add(fd)
    return GpuLease(fd)
