# SPDX-License-Identifier: GPL-3.0-only
"""Static Embedl encoder adapter. Imported only by the isolated Fast worker.

The downloaded/locally built engine retains its publisher's license. This
adapter neither downloads nor builds weights, and never changes input RGB.
"""
from pathlib import Path
import torch
import tensorrt as trt


class EmbedlTensorRTBackbone(torch.nn.Module):
    """Use Torch-owned device buffers on the calling CUDA stream, without PyCUDA."""

    def __init__(self, engine_path, image_size=512):
        super().__init__()
        if image_size not in (384, 512):
            raise ValueError('Embedl 仅支持已验证的 384 / 512 静态引擎')
        self.embed_dim = self.embed_dims = 1280
        self.patch_size = 16
        self.calls = 0
        self.filename = Path(engine_path)
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, '')
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(self.filename.read_bytes())
        if self.engine is None:
            raise RuntimeError('无法加载 Embedl TensorRT 引擎，请核对 GPU 和 TensorRT 版本')
        inputs, outputs = [], []
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            (inputs if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else outputs).append(name)
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError('Embedl 引擎必须恰有一个图像输入和一个特征输出')
        self.input_name, self.output_name = inputs[0], outputs[0]
        self.input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        self.output_shape = tuple(self.engine.get_tensor_shape(self.output_name))
        if self.input_shape != (1, 3, image_size, image_size) or self.output_shape != (1, 1280, image_size//16, image_size//16):
            raise ValueError(f'Embedl 引擎尺寸不兼容：{self.input_shape} → {self.output_shape}')
        if any(self.engine.get_tensor_dtype(name) != trt.float16 for name in (self.input_name, self.output_name)):
            raise ValueError('此适配器需要经过验证的 FP16 IO / INT8 混合精度引擎')
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError('无法创建 Embedl TensorRT 执行上下文')

    def forward(self, x, extra_embed=None):
        if extra_embed is not None:
            raise ValueError('Embedl TensorRT 不支持额外骨干嵌入')
        if tuple(x.shape) != self.input_shape or x.device.type != 'cuda':
            raise ValueError(f'Embedl INT8 输入须为 CUDA 上的 {self.input_shape}，仅支持单人身体模式')
        with torch.cuda.device(x.device):
            inp = x.to(torch.float16).contiguous()
            if not self.context.set_input_shape(self.input_name, tuple(inp.shape)):
                raise RuntimeError('无法设置 Embedl 输入尺寸')
            # A fresh output preserves prior predictions when later frames run.
            out = torch.empty(self.output_shape, device=x.device, dtype=torch.float16)
            for name, tensor in ((self.input_name, inp), (self.output_name, out)):
                if not self.context.set_tensor_address(name, tensor.data_ptr()):
                    raise RuntimeError('无法绑定 Embedl CUDA 缓冲区')
            if not self.context.execute_async_v3(torch.cuda.current_stream(x.device).cuda_stream):
                raise RuntimeError('Embedl TensorRT 执行失败')
        self.calls += 1
        return out

    def status(self):
        return dict(implementation='EmbedlTensorRTBackbone', precision='INT8/FP16 mixed',
                    io_dtype='float16', input_shape=list(self.input_shape),
                    output_shape=list(self.output_shape), calls=self.calls,
                    engine=str(self.filename.resolve()))
