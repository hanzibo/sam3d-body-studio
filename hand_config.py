# SPDX-License-Identifier: GPL-3.0-only
"""Dependency-free hand launch options shared by Qt and GPU worker."""
from dataclasses import dataclass,asdict

@dataclass(frozen=True)
class HandExecution:
    backbone: str = 'native'
    detector: str = 'native'
    decoder_compile: bool = False
    mhr_compile: bool = False
    mhr_graph: bool = False
    tf32: bool = False
    compile_mode: str = 'default'
    feedback_cache: bool = False
    intermediate_keypoints: bool = False
    final_hand_vertices: bool = False
    intermediate_compile: bool = False
    intermediate_predictions: int = 5

    def __post_init__(self):
        if self.detector not in ('native','trt_fp16'):raise ValueError('无效的手部检测器方案')
        if self.backbone not in ('native','trt_fp16','trt_int8'):raise ValueError('无效的手部编码器方案')
        if self.compile_mode not in ('default','reduce-overhead'):raise ValueError('无效的编译模式')
        if type(self.intermediate_predictions) is not int or not 0 <= self.intermediate_predictions <= 5:
            raise ValueError('手部中间预测次数须为 0–5 的整数')
        for key in ('decoder_compile','mhr_compile','mhr_graph','tf32','feedback_cache','intermediate_keypoints','final_hand_vertices','intermediate_compile'):
            if type(getattr(self,key)) is not bool:raise ValueError('编译开关须为布尔值')

    def to_dict(self):return asdict(self)
