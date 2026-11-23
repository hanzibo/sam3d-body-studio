# SPDX-License-Identifier: GPL-3.0-only
"""Hand-only execution options; no changes to body workers or upstream files."""
import hashlib
import torch
from studio_paths import path
from engine_registry import verify_engine

from studio.hand.hand_config import HandExecution

class HandTRTBackbone(torch.nn.Module):
    """Reuse a pinned 512 FP16 or mixed INT8/FP16 engine; respect actual IO dtype and profiles.

    B=2 uses its supported profile or sequential B=1; never a silent shape override.
    Fresh output storage keeps previous results intact across frames.
    """
    def __init__(self,precision='trt_fp16'):
        super().__init__()
        import tensorrt as trt
        if precision not in ('trt_fp16','trt_int8'):raise ValueError('Unsupported hand encoder precision')
        self.precision=precision
        self.filename=path('backbone_engine_int8_512' if precision=='trt_int8' else 'backbone_engine_512')
        with self.filename.open('rb') as stream:self.sha256=hashlib.file_digest(stream,'sha256').hexdigest()
        verify_engine(self.filename,self.sha256)
        self.logger=trt.Logger(trt.Logger.WARNING);trt.init_libnvinfer_plugins(self.logger,'');self.runtime=trt.Runtime(self.logger)
        self.engine=self.runtime.deserialize_cuda_engine(self.filename.read_bytes())
        if self.engine is None:raise RuntimeError('手部 TensorRT 引擎无法加载')
        names=[self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        ins=[n for n in names if self.engine.get_tensor_mode(n)==trt.TensorIOMode.INPUT]
        outs=[n for n in names if self.engine.get_tensor_mode(n)==trt.TensorIOMode.OUTPUT]
        if len(ins)!=1 or len(outs)!=1:raise ValueError('手部编码器须为单输入单输出')
        self.input,self.output=ins[0],outs[0]
        self.input_shape=tuple(self.engine.get_tensor_shape(self.input));self.output_shape=tuple(self.engine.get_tensor_shape(self.output))
        if self.input_shape[1:]!=(3,512,512) or self.output_shape[1:]!=(1280,32,32):raise ValueError('手部编码器须为 512/1280 特征')
        dtype={trt.float16:torch.float16,trt.float32:torch.float32}
        self.input_dtype=dtype[self.engine.get_tensor_dtype(self.input)];self.output_dtype=dtype[self.engine.get_tensor_dtype(self.output)]
        self.max_batch=self.input_shape[0]
        if self.max_batch<0:
            low,opt,high=self.engine.get_tensor_profile_shape(self.input,0)
            if low[0]!=1 or high[0]<1:raise ValueError('引擎批量配置不兼容')
            self.max_batch=high[0]
        if self.max_batch<1:raise ValueError('无有效 TensorRT 批量')
        self.context=self.engine.create_execution_context()
        if self.context is None:raise RuntimeError('手部 TensorRT 上下文创建失败')
        self.embed_dim=self.embed_dims=1280;self.patch_size=16;self.executions=0

    def forward(self,x,extra_embed=None):
        if extra_embed is not None:raise ValueError('手部 TRT 不支持额外骨干嵌入')
        if x.device.type!='cuda' or x.shape[0] not in (1,2) or tuple(x.shape[1:])!=(3,512,512):raise ValueError('手部 TRT 输入须为 CUDA B=1/2,3,512,512')
        inp=x.to(self.input_dtype).contiguous();result=torch.empty((len(x),1280,32,32),device=x.device,dtype=self.output_dtype)
        step=min(len(x),self.max_batch)
        for i in range(0,len(x),step):
            part=inp[i:i+step]
            if not self.context.set_input_shape(self.input,tuple(part.shape)):raise RuntimeError('手部 TRT 输入批量设置失败')
            if tuple(self.context.get_tensor_shape(self.output))!=(len(part),1280,32,32):raise RuntimeError('手部 TRT 输出形状错误')
            if not self.context.set_tensor_address(self.input,part.data_ptr()) or not self.context.set_tensor_address(self.output,result[i:i+step].data_ptr()):raise RuntimeError('手部 TRT 绑定失败')
            if not self.context.execute_async_v3(torch.cuda.current_stream(x.device).cuda_stream):raise RuntimeError('手部 TRT 执行失败')
            self.executions+=1
        return result.to(torch.float16)

    def status(self):return dict(engine=str(self.filename),sha256=self.sha256,precision='INT8/FP16 mixed' if self.precision=='trt_int8' else 'FP16 compute',input_dtype=str(self.input_dtype),output_dtype=str(self.output_dtype),max_batch=self.max_batch,input_shape=self.input_shape,output_shape=self.output_shape,dual_hand_policy='sequential B1 + B1, batched decoder' if self.max_batch==1 else 'native B2 encoder and decoder',executions=self.executions)

class DeviceMHR(torch.nn.Module):
    """Equivalent MHR composition retaining learned pose correctives.

    Pose-feature construction uses device tensors throughout; the fixed sparse
    first layer is materialized once as a dense matrix with identical weights,
    avoiding temporary CPU transfers and sparse construction during graph capture.
    """
    def __init__(self,module):
        super().__init__();self.module=module
        first=getattr(module.pose_correctives_model.pose_dirs_predictor,'0')
        weight=torch.sparse_coo_tensor(first.sparse_indices,first.sparse_weight,tuple(first.sparse_shape)).to_dense()
        self.register_buffer('corrective_weight',weight)
        self.register_buffer('corrective_bias',first.bias)

    def forward(self,shape,params,expression):
        import roma
        m=self.module;ct=m.character_torch
        rest=ct.blend_shape(shape)
        joint_params=ct.model_parameters_to_joint_parameters(torch.cat((params,torch.zeros_like(shape)),dim=1))
        state=ct.joint_parameters_to_skeleton_state(joint_params)
        angles=joint_params.reshape(len(shape),-1,7)[:,2:,3:6]
        rotation=roma.euler_to_rotmat('xyz',angles)
        features=torch.cat((rotation[..., :,0],rotation[..., :,1]),dim=-1)
        features[:,:,0]-=1;features[:,:,4]-=1
        hidden=torch.nn.functional.linear(features.flatten(1,2),self.corrective_weight,self.corrective_bias)
        hidden=getattr(m.pose_correctives_model.pose_dirs_predictor,'1')(hidden)
        corrective=getattr(m.pose_correctives_model.pose_dirs_predictor,'2')(hidden).reshape(len(shape),-1,3)
        rest=rest+m.face_expressions_model(expression)+corrective
        return ct.skin_points(state,rest),state

class GraphMHR(torch.nn.Module):
    """Optional exact TorchScript geometry replay with separate B=1/B=2 caches."""
    def __init__(self,module):
        super().__init__();self.module=DeviceMHR(module);self.cache={};self.parity_mm={}

    @torch.compiler.disable
    def forward(self,*args):
        signature=tuple((tuple(x.shape),x.dtype,x.device) for x in args)
        cached=self.cache.get(signature)
        if cached is None:
            reference=self.module.module(*args);candidate=self.module(*args)
            for expected,actual in zip(reference,candidate):
                torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-3)
            self.parity_mm[len(args[0])]=float((reference[0]-candidate[0]).abs().max().item())*10
            del reference,candidate
            inputs=tuple(x.clone() for x in args);stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):self.module(*inputs)
            torch.cuda.current_stream().wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):outputs=self.module(*inputs)
            cached=(inputs,graph,outputs);self.cache[signature]=cached
        inputs,graph,outputs=cached
        for target,source in zip(inputs,args):target.copy_(source)
        graph.replay()
        return tuple(x.clone() for x in outputs)


def install_execution(engine,config,stage=lambda message:None):
    """Install before warmup; errors remain visible, no silent eager fallback."""
    model=engine.model
    torch.set_float32_matmul_precision('high' if config.tf32 else 'highest')
    if config.tf32:
        stage('启用 TF32 矩阵加速（权重和张量仍为 FP32，投影保持完整 FP32）')
        original_project=model.camera_project_hand
        def precise_project(*args,**kwargs):
            previous=torch.backends.cuda.matmul.allow_tf32
            try:
                torch.backends.cuda.matmul.allow_tf32=False
                return original_project(*args,**kwargs)
            finally:torch.backends.cuda.matmul.allow_tf32=previous
        model.camera_project_hand=precise_project
    if config.backbone in ('trt_fp16','trt_int8'):
        stage('加载已登记的 512 '+('INT8/FP16 混合精度' if config.backbone=='trt_int8' else 'FP16')+' TensorRT 编码器（按实际批量配置执行）')
        # Drop eager encoder before deserializing to bound GPU memory.
        model.backbone.cpu();del model.backbone
        import gc
        gc.collect();torch.cuda.empty_cache()
        model.backbone=HandTRTBackbone(config.backbone).eval()
    if config.mhr_graph:
        stage('启用 MHR CUDA Graph（单 / 双手分别预热）')
        model.head_pose_hand.mhr=GraphMHR(model.head_pose_hand.mhr)
    if config.decoder_compile:
        stage('启用手部解码层 torch.compile；首次预热可能较慢')
        for layer in model.decoder_hand.layers:
            layer.forward=torch.compile(layer.forward,mode=config.compile_mode,dynamic=False,fullgraph=True)
    if config.mhr_compile:
        stage('启用手部 MHR 参数 / 投影前处理编译')
        model.head_pose_hand.mhr_forward=torch.compile(model.head_pose_hand.mhr_forward,mode=config.compile_mode,dynamic=False,fullgraph=False)
    if config.feedback_cache or config.intermediate_keypoints or config.final_hand_vertices or config.intermediate_compile or config.intermediate_predictions!=5:
        stage('启用手部反馈复制优化 / 中间关键点专用路径（保留姿态修正）')
        from studio.hand.hand_feedback import install_hand_feedback
        install_hand_feedback(model,config,engine.ids)
    stage(f'手部中间预测 {config.intermediate_predictions} 次 + 最终预测；回调整体编译：'+('开' if config.intermediate_compile else '关'))
    return config.to_dict()
