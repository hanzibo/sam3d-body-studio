#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Isolated Embedl INT8 experiment; never changes production engine settings."""
import argparse
import contextlib
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gpu_lease import acquire_gpu_lease
from studio_paths import environment, path


def digest(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def memory_guard():
    """Refuse unbounded runs on this workstation, including engine builds."""
    group=next(line.split(':',2)[2] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
    location=Path('/sys/fs/cgroup')/group.lstrip('/')
    limits=[]
    for folder in [location,*location.parents]:
        f=folder/'memory.max'
        if f.exists() and f.read_text().strip().isdigit():limits.append(int(f.read_text().strip()))
    if not limits or min(limits)>8*1024**3:
        raise RuntimeError('Run inside a bounded scope: systemd-run --user --scope -p MemoryMax=7G -p MemorySwapMax=2G ...')
    return dict(cgroup=str(location),memory_max_bytes=min(limits))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['build', 'encoder', 'full'], required=True)
    p.add_argument('--models', type=Path, default=Path('/run/media/hzb/Data/models/embedl-sam-3d-body'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--runs', type=int, default=60)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--onnx-name', default='embedl_sam3dbody_int8.onnx')
    p.add_argument('--engine-name', default='embedl_sam3dbody_int8_trt10_fp16.engine')
    p.add_argument('--image-size', type=int, choices=(384,512), default=512)
    p.add_argument('--strong-fp16',action='store_true',help='Build strongly typed from prepared FP16 Q/DQ ONNX')
    p.add_argument('--image', type=Path, default=ROOT/'outputs/gui/20261103-174926-349375/input.png')
    p.add_argument('--poses', type=Path, default=ROOT/'outputs/rgbd-sets/recording-20261106-135615-851252')
    a = p.parse_args()
    if a.stage == 'full' and a.image_size != 512:
        p.error('Legacy full harness is 512-only; use check_embedl_384.py for 384 production workers')
    guard=memory_guard()
    a.output.mkdir(parents=True, exist_ok=True)
    a.models.mkdir(parents=True, exist_ok=True)
    os.environ.update(environment('fast'))
    tmp = a.models/'tmp'
    tmp.mkdir(exist_ok=True)
    os.environ['TMPDIR'] = str(tmp)
    lease = acquire_gpu_lease('exclusive')
    import torch
    import tensorrt as trt
    import numpy as np
    torch.set_num_threads(4)
    engine_path = a.models/a.engine_name
    report = dict(schema='embedl_encoder_experiment_v1', stage=a.stage,
                  torch=torch.__version__, tensorrt=trt.__version__, gpu=torch.cuda.get_device_name(),
                  engine=str(engine_path), date=time.strftime('%Y-%m-%dT%H:%M:%S%z'),memory_guard=guard)
    (a.output/'benchmark-tested.py').write_bytes(Path(__file__).read_bytes())
    logger = trt.Logger(trt.Logger.INFO if a.stage == 'build' else trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, '')
    if a.stage == 'build':
        if engine_path.exists():
            raise FileExistsError(f'Refusing to replace existing engine: {engine_path}')
        torch.cuda.init()
        onnx_path = a.models/a.onnx_name
        import onnx
        metadata = onnx.load(onnx_path, load_external_data=False)
        external_files = sorted({entry.value for tensor in metadata.graph.initializer
                                 for entry in tensor.external_data if entry.key == 'location'})
        report['sources'] = {p.name: dict(bytes=p.stat().st_size, sha256=digest(p))
                             for p in [onnx_path, *[onnx_path.parent/name for name in external_files], a.models/'infer_trt.py']}
        del metadata
        builder = trt.Builder(logger)
        network = builder.create_network((1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)) if a.strong_fp16 else 0)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(onnx_path)):
            raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1024**3)
        config.builder_optimization_level = 3
        if not a.strong_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            config.set_flag(trt.BuilderFlag.INT8)
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        began = time.perf_counter()
        plan = builder.build_serialized_network(network, config)
        if plan is None:
            raise RuntimeError('TensorRT INT8 engine build failed')
        temp_path = engine_path.with_suffix('.engine.partial')
        with temp_path.open('wb') as f:
            f.write(memoryview(plan))
        temp_path.replace(engine_path)
        report.update(build_seconds=time.perf_counter()-began, engine_bytes=engine_path.stat().st_size,
                      engine_sha256=digest(engine_path), workspace_bytes=1024**3,
                      precision='Q/DQ INT8; strongly typed FP16 nonquantized ops' if a.strong_fp16 else 'Q/DQ INT8 plus FP16 fallback; FP32 ONNX IO', builder_optimization_level=3)
        (a.output/'build.json').write_text(json.dumps(report, indent=2))
        print('BUILD_OK', json.dumps(report), flush=True)
        return

    class Encoder(torch.nn.Module):
        """Bind Torch-owned CUDA buffers to an ONNX engine on the current stream."""
        def __init__(self, filename):
            super().__init__()
            self.embed_dim = self.embed_dims = 1280
            self.patch_size = 16
            self.runtime = trt.Runtime(logger)
            self.engine = self.runtime.deserialize_cuda_engine(filename.read_bytes())
            if self.engine is None:
                raise RuntimeError(f'Cannot deserialize {filename}')
            self.context = self.engine.create_execution_context()
            self.inputs = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
                           if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT]
            self.outputs = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
                            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]
            assert len(self.inputs) == len(self.outputs) == 1
            self.input_name, self.output_name = self.inputs[0], self.outputs[0]
            if tuple(self.engine.get_tensor_shape(self.input_name)) != (1,3,a.image_size,a.image_size):
                raise ValueError(f'Engine input does not match requested size {a.image_size}: {filename}')
            self.input_dtype = torch.from_numpy(np.empty((), dtype=trt.nptype(self.engine.get_tensor_dtype(self.input_name)))).dtype
            self.output_dtype = torch.from_numpy(np.empty((), dtype=trt.nptype(self.engine.get_tensor_dtype(self.output_name)))).dtype
            self.calls = 0
        def forward(self, x, extra_embed=None):
            assert extra_embed is None and tuple(x.shape) == (1, 3, a.image_size, a.image_size)
            inp = x.to(self.input_dtype).contiguous()
            self.context.set_input_shape(self.input_name, tuple(inp.shape))
            out = torch.empty(tuple(self.context.get_tensor_shape(self.output_name)), device=x.device, dtype=self.output_dtype)
            self.context.set_tensor_address(self.input_name, inp.data_ptr())
            self.context.set_tensor_address(self.output_name, out.data_ptr())
            if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
                raise RuntimeError('TensorRT execute failed')
            self.calls += 1
            # Match the existing Fast FP16-backbone public interface.
            return out.to(torch.float16)

    def summary(v):
        return dict(n=len(v), mean_ms=float(np.mean(v)), median_ms=float(np.median(v)), p95_ms=float(np.percentile(v, 95)))

    if a.stage == 'encoder':
        import cv2
        im = cv2.imread(str(a.models/'sample_input.png'))[:, :, ::-1].copy()
        im = cv2.resize(im, (a.image_size, a.image_size)).astype(np.float32)/255
        im = (im-np.array([.485,.456,.406], np.float32))/np.array([.229,.224,.225], np.float32)
        x = torch.from_numpy(im.transpose(2,0,1)[None].copy()).cuda()
        encoders = {name: Encoder(filename).eval() for name, filename in
                    [('fp16', path(f'backbone_engine_{a.image_size}')), ('int8', engine_path)]}
        report['io'] = {k: dict(input_name=v.input_name, output_name=v.output_name,
                               input_dtype=str(v.input_dtype), output_dtype=str(v.output_dtype)) for k,v in encoders.items()}
        values = {k: [] for k in encoders}
        with torch.inference_mode():
            for encoder in encoders.values():
                for _ in range(a.warmup): encoder(x)
            torch.cuda.synchronize()
            for label in ['fp16','int8','int8','fp16','int8','fp16','fp16','int8']:
                encoder = encoders[label]
                for _ in range(3): encoder(x)
                torch.cuda.synchronize()
                for _ in range(a.runs):
                    s,e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    s.record(); out=encoder(x); e.record(); e.synchronize()
                    values[label].append(s.elapsed_time(e))
            f = {k:v(x).float().cpu().numpy() for k,v in encoders.items()}
        report['timing'] = {k:summary(v) for k,v in values.items()}
        report['feature_relative_l2'] = float(np.linalg.norm(f['int8']-f['fp16'])/np.linalg.norm(f['fp16']))
        report['scope'] = 'Torch-input cast + TensorRT engine + FP16 output cast, CUDA events; no host transfers; identical normalized input'
        (a.output/'encoder.json').write_text(json.dumps(report, indent=2))
        for name, enc in encoders.items():
            (a.output/f'{name}-layers.json').write_text(enc.engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))
        print('ENCODER_RESULT', json.dumps(report), flush=True)
        return

    sys.path.insert(0, str(path('fast_repo')))
    loader = Path('/home/hzb/Work/fast-sam3d-body-deploy/run.py')
    sys.argv = [str(loader), '--image', str(a.image), '--output', str(a.output), '--compile',
                '--trt-backbone', str(path('backbone_engine_512')), '--trt-yolo', str(path('yolo_engine')),
                '--image-size', '512', '--inference-type', 'body', '--warmup', '1', '--runs', '1']
    from offline_dinov3 import local_dinov3_source
    ns = {'__name__':'__main__', '__file__':str(loader)}
    with local_dinov3_source():
        exec(compile(loader.read_text().split('faces=estimator.faces.copy()')[0],str(loader),'exec'),ns)
    from fast_feedback import install_fast_feedback_cache
    from fast_keypoints import install_fast_keypoint_path
    from fast_execution import install_final_body_replay, install_body_io
    model=ns['model']
    feedback=install_fast_feedback_cache(model)
    keypoints=install_fast_keypoint_path(model)
    final=install_final_body_replay(model)
    io=install_body_io(ns['estimator'])
    baseline=model.backbone
    quantized=Encoder(engine_path).eval()
    variants={'fp16':baseline, 'int8':quantized}
    values={k:[] for k in variants}
    null=open(os.devnull,'w')
    with torch.inference_mode(), contextlib.redirect_stdout(null):
        for enc in variants.values():
            model.backbone=enc
            for _ in range(a.warmup): ns['forward']()
        for label in ['fp16','int8','int8','fp16','int8','fp16','fp16','int8']:
            model.backbone=variants[label]
            for _ in range(3): ns['forward']()
            for _ in range(a.runs): values[label].append(ns['forward']()[1]*1000)
    report['timing']={k:summary(v) for k,v in values.items()}
    report['scope']='512 single-person body; YOLO TRT + crop + DINO + compiled FP32 decoder/MHR + CPU outputs; excludes load/compile/render/disk IO; same feedback/keypoint/final-graph/IO optimizations'
    report['optimizations']=dict(feedback=dict(feedback),keypoints=dict(keypoints),final=final.status(),io=dict(io))
    report['adapter_sha256']={name:digest(ROOT/name) for name in ['fast_feedback.py','fast_keypoints.py','fast_execution.py']}
    report['settings']=ns['settings']
    report['int8_output_adapter']='ONNX native output converted to FP16 to match the existing backbone contract'
    report['engine_sha256']={name:digest(filename) for name,filename in [('fp16',path('backbone_engine_512')),('int8',engine_path)]}
    (a.output/'full.json').write_text(json.dumps(report,indent=2))
    print('FULL_TIMING',json.dumps(report['timing']),flush=True)
    images=[a.image]
    if a.poses.exists():
        manifest=json.loads((a.poses/'pose_set.json').read_text())
        images += [a.poses/row['sample']/'rgb.png' for row in manifest['poses'] if row.get('sample')]
    checks=[]
    import cv2
    with torch.inference_mode(),contextlib.redirect_stdout(null):
        for i,image in enumerate(images):
            rgb=cv2.cvtColor(cv2.imread(str(image)),cv2.COLOR_BGR2RGB)
            ns['rgb']=rgb
            predictions={}
            for name,encoder in variants.items():
                model.backbone=encoder
                rows,_=ns['forward']()
                predictions[name]=[{k:v.copy() if isinstance(v,np.ndarray) else v for k,v in row.items()} for row in rows]
                np.savez_compressed(a.output/f'case-{i:02d}-{name}.npz',**{f'p{j}_{k}':v for j,row in enumerate(rows) for k,v in row.items() if isinstance(v,np.ndarray)})
            assert len(predictions['fp16'])==len(predictions['int8'])==1
            diffs={}
            for k,v in predictions['fp16'][0].items():
                w=predictions['int8'][0].get(k)
                if not isinstance(v,np.ndarray) or not np.issubdtype(v.dtype,np.number): continue
                assert v.shape==w.shape and np.isfinite(w).all(),k
                delta=w.astype(np.float64)-v.astype(np.float64)
                diffs[k]=dict(mean_abs=float(np.mean(abs(delta))),max_abs=float(np.max(abs(delta))))
                if delta.ndim==2 and delta.shape[-1]==3:
                    d=np.linalg.norm(delta,axis=-1)
                    diffs[k].update(mean_l2=float(d.mean()),p95_l2=float(np.percentile(d,95)),max_l2=float(d.max()))
            ref,got=predictions['fp16'][0],predictions['int8'][0]
            extra={}
            if 'pred_vertices' in ref and 'pred_cam_t' in ref:
                r=ref['pred_vertices'].astype(np.float64)+ref['pred_cam_t'].reshape(1,3)
                g=got['pred_vertices'].astype(np.float64)+got['pred_cam_t'].reshape(1,3)
                d=np.linalg.norm(g-r,axis=-1)
                extra['camera_space_mesh_delta_mm']=dict(mean=float(d.mean()*1000),p95=float(np.percentile(d,95)*1000),max=float(d.max()*1000))
                d=np.linalg.norm((g-g.mean(0))-(r-r.mean(0)),axis=-1)
                extra['centroid_aligned_mesh_delta_mm']=dict(mean=float(d.mean()*1000),p95=float(np.percentile(d,95)*1000),max=float(d.max()*1000))
            checks.append(dict(image=str(image),image_sha256=digest(image),differences=diffs,**extra))
            report['checks']=checks
            (a.output/'full.json').write_text(json.dumps(report,indent=2))
    report['int8_calls']=quantized.calls
    report['faces_shape']=list(ns['estimator'].faces.shape)
    (a.output/'full.json').write_text(json.dumps(report,indent=2))
    print('FULL_CHECKS_OK',len(checks),flush=True)


if __name__=='__main__':
    main()
