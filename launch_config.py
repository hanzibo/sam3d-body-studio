"""Versioned launch configuration shared by the GUI and isolated workers."""
from dataclasses import dataclass, asdict, replace
import hashlib
import json
import os
import math
from pathlib import Path
from studio_paths import path

ROOT = Path(__file__).resolve().parent
ENGINE_384 = path('backbone_engine_384')

@dataclass(frozen=True)
class LaunchConfig:
    schema_version: int = 1
    backend: str = 'fast'
    mode: str = 'body'
    image_size: int = 384
    trt_backbone: bool = True
    trt_detector: bool = True
    decoder_compile: bool = True
    mhr_compile: bool = True
    interm_compile: bool = True
    warmup_runs: int = 10
    warmup_layout: str = 'both'
    smpl_enabled: bool = False
    smpl_smoothing: bool = False
    smpl_model_path: str = ''
    body_interm_predictions: int = 3
    one_euro_enabled: bool = False
    one_euro_target: str = 'smpl'  # Legacy profiles retain their SMPL-only meaning.
    one_euro_min_cutoff: float = 1.5
    one_euro_beta: float = 5.0
    one_euro_d_cutoff: float = 1.0

    def validate(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError('不支持的方案版本')
        if self.backend not in ('fast', 'standard') or self.mode not in ('body', 'full'):
            raise ValueError('实现或模式无效')
        if type(self.image_size) is not int or self.image_size not in (384, 512):
            raise ValueError('网络分辨率只支持 384 或 512')
        for key in ('trt_backbone','trt_detector','decoder_compile','mhr_compile','interm_compile','smpl_enabled','smpl_smoothing','one_euro_enabled'):
            if type(getattr(self,key)) is not bool:
                raise ValueError(f'{key} 必须是布尔值')
        if type(self.body_interm_predictions) is not int or self.body_interm_predictions not in (0,1,2,3):
            raise ValueError('身体中间预测次数必须为 0–3')
        if self.body_interm_predictions!=3 and (self.backend!='fast' or self.mode!='body'):
            raise ValueError('中间预测次数实验仅支持 Fast 身体模式')
        if self.one_euro_target not in ('mhr','smpl','both'):
            raise ValueError('One Euro 滤波目标无效')
        if self.smpl_one_euro_enabled and not self.smpl_enabled:
            raise ValueError('SMPL One Euro 需要启用 SMPL 转换；MHR 骨架滤波无需 SMPL')
        for key,low,high in [('one_euro_min_cutoff',.01,30.),('one_euro_beta',0.,100.),('one_euro_d_cutoff',.01,30.)]:
            value=getattr(self,key)
            if type(value) not in (int,float) or not math.isfinite(value) or not low<=value<=high:
                raise ValueError(key+' 超出允许范围')
        if type(self.smpl_model_path) is not str:
            raise ValueError('SMPL 模型路径必须是字符串')
        if self.smpl_smoothing and not self.smpl_enabled:
            raise ValueError('MLP 平滑需要先启用 SMPL 转换')
        if type(self.warmup_runs) is not int or not 1 <= self.warmup_runs <= 30:
            raise ValueError('预热次数范围为 1–30')
        if self.warmup_layout not in ('sample','camera','both'):
            raise ValueError('预热输入无效')
        if self.backend == 'standard':
            if self.image_size != 512 or any((self.trt_backbone,self.trt_detector,self.decoder_compile,self.mhr_compile,self.interm_compile)):
                raise ValueError('标准版仅支持 512，TensorRT 和编译选项不适用')
        elif self.mode == 'full' and self.image_size == 384:
            raise ValueError('384 当前仅开放身体模式；完整模式请使用 512（现有 384 TRT 引擎仅支持 batch=1）')
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data,dict):
            raise ValueError('方案必须是 JSON 对象')
        try:
            return cls(**data).validate()
        except TypeError as exc:
            raise ValueError(f'方案字段无效：{exc}') from exc

    @property
    def mhr_one_euro_enabled(self):
        return self.one_euro_enabled and self.one_euro_target in ('mhr','both')

    @property
    def smpl_one_euro_enabled(self):
        return self.one_euro_enabled and self.one_euro_target in ('smpl','both')

    @property
    def config_id(self):
        return hashlib.sha256(json.dumps(self.to_dict(),sort_keys=True).encode()).hexdigest()[:16]

    @property
    def body_interm_layers(self):
        return {3:(0,1,2),2:(0,2),1:(0,),0:()}[self.body_interm_predictions]

    @property
    def compile_enabled(self):
        return any((self.decoder_compile,self.mhr_compile,self.interm_compile))

    def summary(self):
        mode = '身体' if self.mode == 'body' else '身体＋手部精修'
        extra = ' / SMPL'+('＋MLP（中心帧）' if self.smpl_smoothing else '原始') if self.smpl_enabled else ''
        if self.one_euro_enabled:extra+=' / One Euro：'+{'mhr':'MHR 骨架','smpl':'SMPL 骨架','both':'MHR＋SMPL'}[self.one_euro_target]
        if self.backend == 'standard':
            return f'标准版 / {mode} / 512'+extra
        labels = '/'.join('开' if x else '关' for x in (self.decoder_compile,self.mhr_compile,self.interm_compile))
        return (f'Fast / {mode} / {self.image_size} / 骨干 {"TRT" if self.trt_backbone else "PyTorch"}'
                f' / 检测 {"TRT" if self.trt_detector else "PyTorch"} / 编译(解码/MHR/中间) {labels} / 身体反馈 {self.body_interm_predictions}次＋最终'+extra)

    def save(self,path):
        Path(path).write_text(json.dumps(self.to_dict(),ensure_ascii=False,indent=2))

PRESETS = {
    '演示最快': LaunchConfig(),
    'MHR 骨架＋One Euro': replace(LaunchConfig(),one_euro_enabled=True,one_euro_target='mhr'),
    'SMPL 身体低延迟': replace(LaunchConfig(),smpl_enabled=True),
    'SMPL 身体＋One Euro': replace(LaunchConfig(),smpl_enabled=True,one_euro_enabled=True),
    '实验：身体反馈2次＋One Euro': replace(LaunchConfig(),smpl_enabled=True,one_euro_enabled=True,body_interm_predictions=2),
    '实验：身体反馈1次＋One Euro': replace(LaunchConfig(),smpl_enabled=True,one_euro_enabled=True,body_interm_predictions=1),
    'SMPL 身体＋MLP平滑': replace(LaunchConfig(),smpl_enabled=True,smpl_smoothing=True),
    '身体 512 对照': replace(LaunchConfig(),image_size=512),
    '完整人体 512': replace(LaunchConfig(),image_size=512,mode='full'),
    '关闭编译对照': replace(LaunchConfig(),decoder_compile=False,mhr_compile=False,interm_compile=False),
    '关闭 TensorRT 对照': replace(LaunchConfig(),trt_backbone=False,trt_detector=False),
    '标准版身体': LaunchConfig(backend='standard',image_size=512,trt_backbone=False,trt_detector=False,
                              decoder_compile=False,mhr_compile=False,interm_compile=False),
}

def worker_config(backend):
    path = os.environ.get('BODYSTUDIO_CONFIG')
    config = LaunchConfig.from_dict(json.loads(Path(path).read_text())) if path else (
        PRESETS['标准版身体'] if backend == 'standard' else PRESETS['演示最快'])
    if config.backend != backend:
        raise ValueError('启动方案与后端不一致')
    return config

def backbone_engine(config):
    from backends import BACKBONE_ENGINE
    return ENGINE_384 if config.image_size == 384 else BACKBONE_ENGINE

def required_resources(config):
    from backends import BACKENDS, FAST_CHECKPOINT, FAST_REPO, YOLO_ENGINE
    spec = BACKENDS[config.backend]
    resources = [spec['python'],spec['launcher']]
    if config.backend == 'standard':
        resources += spec['resources']
    else:
        resources += [FAST_CHECKPOINT/'model.ckpt',FAST_CHECKPOINT/'model_config.yaml',
                      FAST_CHECKPOINT/'assets/mhr_model.pt', path('yolo_checkpoint'),
                      FAST_REPO/'notebook/images/dancing.jpg']
        if config.trt_backbone:
            resources.append(backbone_engine(config))
        if config.trt_detector:
            resources.append(YOLO_ENGINE)
    if config.smpl_enabled:
        from smpl_stage import resources as smpl_resources
        resources += smpl_resources(config)
    return resources
