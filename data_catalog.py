# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Local artifact inventory and deletion boundaries; no GUI/model imports."""
from dataclasses import dataclass
from pathlib import Path
import os
import stat


OUTPUT_TYPES = {
    'video': ('视频重建任务', '原视频引用、解码帧、人物分段、掩膜、各配置预测与合成视频；整套管理。'),
    'rgbd': ('RGB-D 单张样本', '照片、深度/点云、内参、掩膜、预测与评估作为一个整体。'),
    'rgbd-sets': ('多姿态样本集', '整套姿态照片、分割、批次记录与对比表；整套管理。'),
    'gui': ('重建结果', '输入照片、启动配置、预测参数/网格、对照图和报告。'),
    'segmentation': ('单图 SAM3 分割', '原始 RGB、提示点、候选掩膜与分割日志。'),
    'camera': ('彩色相机照片', '相机彩色照片及拍摄元数据。'),
    'motions': ('动作录制', '帧序列 JSONL、关节/角度 CSV、录制元数据。'),
    'streams': ('实时流摘要', '实时流统计；逐帧临时文件在停止后自动清理。'),
    'sessions': ('模型会话日志', '模型加载/预热日志及会话配置。'),
    'mhr-lab': ('MHR 实验导出', '同一次导出的参数 NPZ / JSON 与网格 PLY。'),
    'yolo-config': ('运行配置（只读）', '检测器自动生成配置，不作为实验数据清理。'),
}
LEGACY = ('camera-analysis', 'fast-analysis', 'community-review', 'Log', 'logs', 'backups')
PROTECTED_GUI_FILES = {'desktop.log', 'run-history.json', 'run-history.tmp'}


@dataclass(frozen=True)
class Entry:
    path: Path
    category: str
    description: str
    size: int
    files: int
    modified: float
    deletable: bool
    warning: str = ''


def checked_path(root, candidate):
    """Reject outside paths and symlinks in every component, including the leaf."""
    root = Path(root).resolve()
    candidate = Path(os.path.abspath(candidate))
    if not candidate.is_relative_to(root) or candidate == root:
        raise ValueError('路径不在本应用数据边界内')
    current = root
    for part in candidate.relative_to(root).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError('符号链接仅列出，不允许跟随、预览或删除')
    if not candidate.resolve().is_relative_to(root):
        raise ValueError('路径已改变或越界')
    return candidate


def measure(path, cancelled=lambda: False):
    """Logical bytes; skip link targets and report unreadable/disappeared items."""
    total = count = 0
    modified = 0.
    warnings = []
    pending = [path]
    while pending:
        if cancelled():
            break
        item = pending.pop()
        try:
            info = item.lstat()
            modified = max(modified, info.st_mtime)
            if stat.S_ISLNK(info.st_mode):
                warnings.append('含符号链接，未统计目标')
            elif stat.S_ISDIR(info.st_mode):
                with os.scandir(item) as children:
                    pending.extend(Path(child.path) for child in children)
            elif stat.S_ISREG(info.st_mode):
                total += info.st_size
                count += 1
        except OSError:
            warnings.append('部分文件无法读取或扫描中发生变化')
    return total, count, modified, '；'.join(sorted(set(warnings)))


def candidates(root):
    root = Path(root)
    outputs = root / 'outputs'
    if outputs.is_dir() and not outputs.is_symlink():
        for folder in sorted(outputs.iterdir()):
            label, desc = OUTPUT_TYPES.get(folder.name, ('测试 / 分析产物', '历史验证、性能分析及临时实验产物。'))
            if folder.name in OUTPUT_TYPES and folder.is_dir() and not folder.is_symlink():
                if folder.name == 'yolo-config':
                    yield folder, label, desc, False
                else:
                    for item in sorted(folder.iterdir()):
                        locked = folder.name == 'gui' and item.name in PROTECTED_GUI_FILES
                        yield item, '运行索引 / 桌面日志（只读）' if locked else label, desc, not locked
            else:
                locked = folder.name == 'studio-model-gpu.lock'
                yield folder, '模型互斥锁（只读）' if locked else label if folder.is_dir() else '实验室 / 其他导出', desc, not locked
    for name in LEGACY:
        folder = root / name
        if folder.exists() or folder.is_symlink():
            yield folder, '备份（只读）' if name == 'backups' else '历史日志 / 分析', '早期部署或测试产物；不自动迁移。', name != 'backups'
    for item in sorted(root.glob('*.log')):
        yield item, '部署日志', '早期部署日志；当前桌面日志在 outputs/gui/desktop.log。', True


def scan(root, cancelled=lambda: False):
    entries = []
    for path, category, description, deletable in candidates(root):
        if cancelled():
            break
        try:
            checked_path(root, path)
            size, files, modified, warning = measure(path, cancelled)
        except ValueError as exc:
            size, files, modified, warning, deletable = 0, 0, 0., str(exc), False
        entries.append(Entry(path, category, description, size, files, modified, deletable, warning))
    return entries


def validate_delete(root, entry, protected=()):
    """Rebuild allowlist immediately before deletion; never trust a UI row alone."""
    path = checked_path(root, entry.path)
    allowed = {p: enabled for p, _, _, enabled in candidates(root)}
    if not entry.deletable or not allowed.get(path, False):
        raise ValueError('此项只读，或已不在允许清理的目录中')
    if not path.exists():
        raise ValueError('数据已不存在，请刷新')
    for active in protected:
        if active is None or not str(active):
            continue
        active = Path(active).resolve()
        if active == path or active.is_relative_to(path) or path.is_relative_to(active):
            raise ValueError('数据正在当前页面 / 模型会话中使用，请先切换到其他数据或关闭后重新启动应用')
    return path


def human_size(value):
    for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB'):
        if value < 1024 or unit == 'TiB':
            return f'{value:.1f} {unit}'
        value /= 1024
