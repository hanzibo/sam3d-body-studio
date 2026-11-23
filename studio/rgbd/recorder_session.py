# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Resumable RGB-D pose recording, independent of camera/GUI/model packages."""
import fcntl
import json
from pathlib import Path
from studio.rgbd.rgbd_batch import PoseSet, save_json

SUGGESTIONS = (
    '正面自然站立，双臂放松', '正面双臂平举', '正面双臂前伸',
    '正面双肘弯曲，手臂离开躯干', '左转约45度，双肘弯曲',
    '右转约45度，双臂展开', '一臂抬高、一臂放低', '侧身，手臂与躯干保持间隙',
)


def recording_active(directory):
    """A persistent empty file is harmless: the OS lock expires on process exit."""
    lock = Path(directory)/'.recorder.lock'
    if not lock.is_file():return False
    try:
        with lock.open('rb') as stream:
            try:fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:return True
    except OSError:return True  # Do not permit cleanup of an unreadable active marker.
    return False


class RecordingSession:
    def __init__(self, directory):
        self.poses = PoseSet(directory)
        self.lock = (self.poses.directory/'.recorder.lock').open('a+b')
        try:fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise ValueError('这组数据正在另一个记录窗口中使用')

    @classmethod
    def create(cls, directory, count):
        poses = PoseSet.create(directory, count)
        for i, pose in enumerate(poses.poses):
            pose['notes'] = SUGGESTIONS[i] if i < len(SUGGESTIONS) else '自定义姿态，请保持静止'
        poses.data['recorder'] = dict(schema='studio_rgbd_recorder_v1', kind='independent_static_rgbd_snapshots')
        poses.save()
        return cls(directory)

    @property
    def directory(self):return self.poses.directory

    @property
    def completed(self):return sum(bool(p.get('sample')) for p in self.poses.poses)

    def attach(self, index, sample):
        """Only completed snapshots enter the manifest. Retakes never delete data."""
        sample = Path(sample).resolve()
        sample.relative_to(self.directory)
        required = ('rgb.png', 'rgb_raw.png', 'depth_raw.png', 'rgbd.npz', 'camera.json')
        if any(not (sample/name).is_file() for name in required):raise ValueError('采集不完整，未替换当前姿态')
        meta = json.loads((sample/'camera.json').read_text())
        if meta.get('schema') != 'studio_rgbd_v1' or meta.get('units') != 'metres' or not meta.get('rgb_rectified'):
            raise ValueError('不是标定配准后的 Studio RGB-D 样本')
        skew = meta.get('timestamps', {}).get('skew_ms')
        if not isinstance(skew, (int, float)) or not 0 <= skew <= meta.get('max_timestamp_skew_ms', 20):
            raise ValueError('RGB-D 时间差不合格，未替换当前姿态')
        pose = self.poses.poses[index]
        previous = pose.get('sample')
        updated = dict(pose, sample=str(sample.relative_to(self.directory)))
        if previous and previous != updated['sample']:
            updated['previous_samples'] = [*pose.get('previous_samples', []), previous]
        # Persist before replacing in-memory state, so failed saves remain recoverable.
        data = dict(self.poses.data, poses=[dict(p) for p in self.poses.poses])
        data['poses'][index] = updated
        save_json(self.poses.path, data)
        self.poses.data = data
        return meta

    def close(self):
        if not self.lock.closed:
            fcntl.flock(self.lock, fcntl.LOCK_UN)
            self.lock.close()
