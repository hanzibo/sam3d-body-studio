"""Persistent configurable Fast worker using the verified loader and Studio protocol."""
from gpu_lease import acquire_gpu_lease
_GPU_LEASE = acquire_gpu_lease('body')
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
from backends import FAST_REPO, FAST_CHECKPOINT, BACKBONE_ENGINE, YOLO_ENGINE
from studio_paths import path as resource_path
from engine_registry import verify_engine
from fast_selection import SinglePersonDetector
from launch_config import worker_config, backbone_engine
from worker_benchmark import measure
from stream_protocol import read_image, emit_mesh
from camera_intrinsics import resolve_intrinsics

BACKEND = 'fast'
SESSION = os.environ.get('BODYSTUDIO_SESSION', '')
CONFIG = worker_config('fast')
startup = time.perf_counter()
def emit(kind, **values):
    print('@@SAM3D ' + json.dumps(dict(type=kind, backend_id=BACKEND, session_id=SESSION, config_id=CONFIG.config_id, **values)), flush=True)

def stage(message):
    emit('stage', message=message)

REPO = FAST_REPO
ROOT = Path(__file__).resolve().parent
a = SimpleNamespace(compile=CONFIG.compile_enabled, trt_backbone=backbone_engine(CONFIG) if CONFIG.trt_backbone else None, trt_yolo=YOLO_ENGINE if CONFIG.trt_detector else None,
                    backbone_dtype='float16', image=REPO/'notebook/images/dancing.jpg')
stage('正在校验权重并加载 · '+CONFIG.summary())

try:
    settings = {
        'GPU_HAND_PREP':'1', 'LAYER_DTYPE':'fp32', 'SKIP_KEYPOINT_PROMPT':'1', 'IMG_SIZE':str(CONFIG.image_size),
        'USE_COMPILE':'1' if a.compile else '0', 'USE_COMPILE_BACKBONE':'0',
        'DECODER_COMPILE':'1' if CONFIG.decoder_compile else '0', 'COMPILE_MODE':'default',
        'INTERM_COMPILE':'1' if CONFIG.interm_compile else '0',
        'COMPILE_WARMUP_BATCH_SIZES':'', 'MHR_USE_CUDA_GRAPH':'0',
        'KEYPOINT_PROMPT_INTERM_INTERVAL':'999', 'BODY_INTERM_PRED_LAYERS':','.join(map(str,CONFIG.body_interm_layers)) or '-1',
        'HAND_INTERM_PRED_LAYERS':'0,1', 'MHR_NO_CORRECTIVES':'1', 'PARALLEL_DECODERS':'1',
        'FOV_TRT':'0', 'USE_TRT_BACKBONE':'0', 'DEBUG_NAN':'0',
    }
    os.environ.update(settings)

    import cv2
    import numpy as np
    import torch
    import trimesh
    from sam_3d_body.models.meta_arch import SAM3DBody
    from sam_3d_body.utils.config import get_config
    from sam_3d_body import SAM3DBodyEstimator
    from tools.build_detector import HumanDetector

    checkpoint_dir = FAST_CHECKPOINT
    yolo_path = resource_path('yolo_checkpoint')
    expected = {
        checkpoint_dir/'model.ckpt':'b5a2f9d305dd02626b967aa2e86021fba07065df66ce7a7e00ffb9664f150abf',
        checkpoint_dir/'assets/mhr_model.pt':'352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc',
        yolo_path:'29b17eaf3a3117cbea906090dbedf9159f7c6a49db58ec8b99ed2dfde1cf6eb2',
    }
    verification={}
    for path, wanted in expected.items():
        with path.open('rb') as f:
            digest=hashlib.file_digest(f,'sha256').hexdigest()
        if digest != wanted:
            raise RuntimeError(f'Checkpoint hash mismatch: {path}')
        verification[str(path)]={'size':path.stat().st_size,'sha256':digest}
    engine_verification={}
    for path in (a.trt_backbone,a.trt_yolo):
        if path is not None:
            with path.open('rb') as f:
                digest=hashlib.file_digest(f,'sha256').hexdigest()
            verify_engine(path, digest)
            engine_verification[str(path.resolve())]={'size':path.stat().st_size,'sha256':digest}
    print('All three binary checkpoint hashes verified',flush=True)
    image=cv2.imread(str(a.image))
    if image is None:
        raise FileNotFoundError(a.image)
    rgb=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    torch.set_num_threads(4)
    start=time.perf_counter()
    cfg=get_config(str(checkpoint_dir/'model_config.yaml'))
    cfg.defrost()
    cfg.MODEL.IMAGE_SIZE=[CONFIG.image_size,CONFIG.image_size]
    cfg.MODEL.MHR_HEAD.MHR_MODEL_PATH=str(checkpoint_dir/'assets/mhr_model.pt')
    cfg.TRAIN.FP16_TYPE=a.backbone_dtype
    cfg.freeze()
    from offline_dinov3 import local_dinov3_source
    with local_dinov3_source():
        model=SAM3DBody(cfg)
    state=torch.load(checkpoint_dir/'model.ckpt',map_location='cpu',weights_only=True,mmap=True)
    state=state.get('state_dict',state)
    missing, unexpected=model.load_state_dict(state, strict=False)
    # MHR tensors are loaded separately from the verified TorchScript asset.
    derived_missing = ['backbone.encoder.mask_token', 'head_pose.hand_pose_comps_ori',
                       'head_pose_hand.hand_pose_comps_ori', 'prompt_encoder.all_point_embeddings.weight']
    invalid_missing=[k for k in missing if not k.startswith(('head_pose.mhr.','head_pose_hand.mhr.')) and k not in derived_missing]
    invalid_unexpected=[k for k in unexpected if k not in ['hand_cls_embed.weight','hand_cls_embed.bias']]
    if invalid_missing or invalid_unexpected:
        raise RuntimeError(f'Model mismatch: missing={invalid_missing}, unexpected={invalid_unexpected}')
    # Fast adds these derived/cache parameters to the original checkpoint layout.
    with torch.no_grad():
        model.backbone.encoder.mask_token.zero_()
        model.head_pose.hand_pose_comps_ori.copy_(model.head_pose.hand_pose_comps)
        model.head_pose_hand.hand_pose_comps_ori.copy_(model.head_pose_hand.hand_pose_comps)
        model.prompt_encoder._sync_embeddings()
    del state
    gc.collect()
    model=model.cuda().eval()
    if a.trt_backbone:
        # Replace only after strict checkpoint validation and derived-cache setup.
        # Avoid constructing the model without encoder parameters before loading.
        from sam_3d_body.models.backbones.dinov3_tensorrt import create_tensorrt_backbone
        trt_backbone=create_tensorrt_backbone(engine_path=str(a.trt_backbone.resolve()),image_size=(CONFIG.image_size,CONFIG.image_size))
        trt_backbone._init_engine()
        if tuple(trt_backbone._engine.get_tensor_shape('input'))[-2:] != (CONFIG.image_size,CONFIG.image_size):
            raise ValueError('TensorRT 输入尺寸与方案不一致')
        model.backbone=trt_backbone.eval()
        gc.collect()
        torch.cuda.empty_cache()
    # Native backbone is explicitly eager; no hidden backbone compile.
    if a.compile:
        # Upstream combines three compilers. Suppress only MHR calls when unchecked.
        heads=(model.head_pose,model.head_pose_hand)
        if not CONFIG.mhr_compile:
            for head in heads:
                head.apply_compile=lambda *args,**kwargs: None
        try:
            model.apply_compile(mode=settings['COMPILE_MODE'],dtype=None)
        finally:
            if not CONFIG.mhr_compile:
                for head in heads:
                    del head.apply_compile
    actual_compile = dict(decoder=bool(getattr(model.decoder,'_compiled', False)),
                          mhr=bool(model.head_pose._compiled),
                          interm=bool(getattr(model,'_interm_compiled',False)))
    if actual_compile != dict(decoder=CONFIG.decoder_compile,mhr=CONFIG.mhr_compile,interm=CONFIG.interm_compile):
        raise ValueError(f'实际编译状态与方案不一致：{actual_compile}')
    detector_path=a.trt_yolo.resolve() if a.trt_yolo else yolo_path
    detector=SinglePersonDetector(HumanDetector(name='yolo_pose',device='cuda',model=str(detector_path)))
    estimator=SAM3DBodyEstimator(model,cfg,human_detector=detector,fov_estimator=None)
    torch.cuda.synchronize()
    load_seconds=time.perf_counter()-start


    def check(outputs):
        if len(outputs) != 1:
            raise ValueError('Fast 仅支持选择一个人物')
        for person in outputs:
            for value in person.values():
                if isinstance(value, np.ndarray) and not np.isfinite(value).all():
                    raise ValueError('模型输出包含无效数值')
            if person['pred_vertices'].shape != (18439,3):
                raise ValueError('模型顶点形状不符合预期')

    @torch.inference_mode()
    def forward(rgb, mode, roi=None, cam_int=None):
        detector.roi = roi
        torch.cuda.synchronize()
        began = time.perf_counter()
        outputs = estimator.process_one_image(rgb.copy(), inference_type=mode, hand_box_source='yolo_pose', cam_int=cam_int)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - began
        check(outputs)
        return outputs, elapsed

    observed_shapes={}
    def observe_backbone(module,args,output):
        observed_shapes.update(input=list(args[0].shape),output=list(output.shape))
        if tuple(args[0].shape[-2:]) != (CONFIG.image_size,CONFIG.image_size) or tuple(output.shape[-2:]) != (CONFIG.image_size//16,CONFIG.image_size//16):
            raise ValueError('实际骨干尺寸与方案不一致')
    model.backbone.register_forward_hook(observe_backbone)
    body_depth=len(model.decoder.layers)
    if any(i>=body_depth-1 for i in CONFIG.body_interm_layers):raise ValueError('所选身体反馈层超出当前解码器范围')
    faces = estimator.faces.copy()
    if faces.shape != (36874,3) or faces.min() < 0 or faces.max() >= 18439:
        raise ValueError('模型拓扑不符合预期')
    from smpl_stage import SmplStage
    smpl_stage = SmplStage(CONFIG) if CONFIG.smpl_enabled else None
    from mhr_filter import MhrFilterStage
    mhr_stage=MhrFilterStage(CONFIG) if CONFIG.mhr_one_euro_enabled else None
    warm_times = []
    layouts=[]
    if CONFIG.warmup_layout in ('sample','both'):
        layouts.append(('上传示例',rgb))
    if CONFIG.warmup_layout in ('camera','both'):
        layouts.append(('相机尺寸',cv2.resize(rgb,(1280,800))))
    for layout,warm_rgb in layouts:
        for index in range(CONFIG.warmup_runs):
            stage(f'正在预热 {layout} · {CONFIG.mode} · {index+1}/{CONFIG.warmup_runs}')
            warm_outputs,elapsed=forward(warm_rgb,CONFIG.mode)
            if smpl_stage:smpl_stage.process(warm_outputs[0])
            warm_times.append(dict(layout=layout,mode=CONFIG.mode,seconds=elapsed))
    if smpl_stage:smpl_stage.reset('warmup_complete')
    load_seconds=time.perf_counter()-startup
    commit=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
    emit('ready',load_seconds=load_seconds,warmup=warm_times,actual_config=CONFIG.to_dict(),
         allocated_gib=torch.cuda.memory_allocated()/1024**3,smpl_assets=smpl_stage.asset_hashes if smpl_stage else None)

    def run(command):
        began = time.perf_counter()
        image = read_image(command)
        if image is None:
            raise ValueError('无法读取输入图片')
        h,w = image.shape[:2]
        roi = command.get('bbox')
        if roi is not None:
            if len(roi) != 4 or not np.isfinite(roi).all():
                raise ValueError('人物框无效')
            x1,y1,x2,y2 = roi
            if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
                raise ValueError('人物框超出图片边界')
        mode = command['inference_type']
        if mode != CONFIG.mode:
            raise ValueError('Unknown inference type')
        rgb_input=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        matrix, camera_info = resolve_intrinsics(command.get('camera_intrinsics'), w, h)
        cam_int = torch.from_numpy(matrix).unsqueeze(0).cuda() if matrix is not None else None
        benchmark=None
        if command['type']=='benchmark':
            outputs,seconds,benchmark=measure(command,lambda:forward(rgb_input,mode,roi,cam_int),emit)
        else:
            outputs,seconds=forward(rgb_input,mode,roi,cam_int)
        smpl_info=smpl_stage.process(outputs[0],command) if smpl_stage else None
        mhr_info=mhr_stage.process(outputs[0],command,camera_info) if mhr_stage else None
        if command['type'] == 'stream':
            emit_mesh(command, outputs[0], faces, seconds, camera_info, began, emit,smpl_info,mhr_info)
            return
        out = Path(command['output'])
        out.mkdir(parents=True, exist_ok=True)
        person = outputs[0]
        np.savez_compressed(out/'person_00.npz', **{k:v for k,v in person.items() if v is not None}, faces=faces)
        trimesh.Trimesh(vertices=person['pred_vertices'],faces=faces,process=False).export(out/'person_00.ply')
        report = dict(model_inference_completed=True, implementation='Fast-SAM-3D-Body persistent worker',
            backend_id=BACKEND, session_id=SESSION, repo_commit=commit,
            checkpoint_verification=verification, engine_verification=engine_verification,
            settings=settings, body_interm_layers=list(CONFIG.body_interm_layers), body_decoder_depth=body_depth, compile_enabled=CONFIG.compile_enabled, tensorrt_enabled=CONFIG.trt_backbone or CONFIG.trt_detector,
            launch_config=CONFIG.to_dict(), actual_config=CONFIG.to_dict(), config_id=CONFIG.config_id,
            observed_backbone_shapes=observed_shapes, benchmark=benchmark,
            image_sha256=hashlib.sha256(Path(command['image']).read_bytes()).hexdigest(),
            backbone_execution='TensorRT' if CONFIG.trt_backbone else 'PyTorch eager',
            compile_requested=dict(decoder=CONFIG.decoder_compile,mhr=CONFIG.mhr_compile,interm=CONFIG.interm_compile),
            compile_actual=actual_compile,
            checkpoint_dtype='float16 backbone / float32 decoder',
            detector='YOLO11m-Pose '+('TensorRT' if CONFIG.trt_detector else 'PyTorch'), hand_box_source='yolo_pose',
            selection_roi=roi, bbox_xyxy=detector.selected_bbox, detector_keypoints=detector.selected_keypoints,
            hand_warning='手腕关键点置信度较低，建议检查手部结果' if detector.hand_warning else None,
            image=command['image'], input_hw=[h,w], inference_type=mode, person_count=1,
            load_seconds=load_seconds, warmup=warm_times,
            inference_seconds=seconds, worker_seconds_to_mesh=time.perf_counter()-began,
            model_reused=True, comparison_status='pending',smpl_info=smpl_info,mhr_info=mhr_info,
            smpl_assets=smpl_stage.asset_hashes if smpl_stage else None,
            gpu=torch.cuda.get_device_name(), torch=torch.__version__,
            vertex_shapes=[list(person['pred_vertices'].shape)], faces_shape=list(faces.shape),
            keypoint_shapes=[list(person['pred_keypoints_3d'].shape)], all_outputs_finite=True,
            camera=camera_info['source']+'; MoGe disabled', camera_intrinsics_used=camera_info,
            camera_intrinsics_origin=command.get('camera_intrinsics_origin'),
            validation_scope='single selected person; no physical scale validation')
        (out/'inference.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        emit('mesh',job=command['job'],output=str(out),report=report)

    for line in sys.stdin:
        command = None
        try:
            command = json.loads(line)
            if command.get('session_id', SESSION) != SESSION or command.get('backend_id', BACKEND) != BACKEND:
                raise ValueError('后端会话已失效')
            if command['type'] == 'stop':
                break
            if command.get('config_id') != CONFIG.config_id:
                raise ValueError('启动配置已失效')
            if command['type'] not in ('run','benchmark','stream'):
                raise ValueError('Unknown command')
            run(command)
        except Exception as exc:
            if smpl_stage:smpl_stage.reset('frame_error')
            if mhr_stage:mhr_stage.reset('frame_error')
            if not isinstance(command, dict) or command.get('type') != 'stream':
                traceback.print_exc()
            emit('error',job=command.get('job') if isinstance(command,dict) else None,message=str(exc))
except Exception as exc:
    traceback.print_exc()
    emit('fatal', message=str(exc))
    sys.exit(1)
