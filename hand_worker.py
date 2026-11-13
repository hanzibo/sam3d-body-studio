# SPDX-License-Identifier: GPL-3.0-only
"""Exclusive hand-only worker; newline JSON stdin and @@HAND events stdout."""
import json
import hashlib
import sys
import time
import traceback
from pathlib import Path
from gpu_lease import acquire_gpu_lease


def emit(event_type, **values):
    print('@@HAND '+json.dumps(dict(type=event_type,**values),ensure_ascii=False),flush=True)


def main():
    lease = acquire_gpu_lease('exclusive')
    import cv2
    import numpy as np
    from hand_engine import HandEngine
    start = time.perf_counter()
    import argparse
    from hand_config import HandExecution
    parser=argparse.ArgumentParser();parser.add_argument('--execution',default='{}')
    parser.add_argument('--startup-warmup-image')
    parser.add_argument('--no-startup-warmup',action='store_true');args=parser.parse_args()
    execution=HandExecution(**json.loads(args.execution))
    engine = HandEngine(lambda message:emit('stage',message=message),execution=execution)
    load_seconds=time.perf_counter()-start
    automatic=None
    warmup=dict(status='disabled')
    if not args.no_startup_warmup:
        from studio_paths import path
        source=Path(args.startup_warmup_image).expanduser() if args.startup_warmup_image else path('hand_warmup_image')
        if not source.is_file():
            warmup=dict(status='skipped',source=str(source),message='未找到离线预热图片；可在启动方案中选择样本，或使用在线补充预热')
        else:
            warm_start=time.perf_counter()
            try:
                emit('stage',message='模型已加载，正在使用本地照片自动预热；无需打开相机或摆手…')
                from hand_automatic import AutomaticHands
                from hand_warmup import warmup_local_sample
                automatic=AutomaticHands(engine)
                warmup=warmup_local_sample(automatic,source,lambda message:emit('stage',message=message))
                warmup['seconds']=time.perf_counter()-warm_start
            except Exception as exc:
                traceback.print_exc()
                warmup=dict(status='failed',source=str(source),seconds=time.perf_counter()-warm_start,message=str(exc))
    emit('ready',load_seconds=load_seconds,startup_seconds=time.perf_counter()-start,
         execution=engine.execution_actual,startup_warmup=warmup)
    for line in sys.stdin:
        command = {}
        try:
            command = json.loads(line)
            kind = command['type']
            if kind == 'stop': break
            job = command['job']; job_start = time.perf_counter()
            if kind in ('auto_detect','auto_run','auto_warmup','sequence','hand_stream'):
                if automatic is None:
                    emit('stage',job=job,message='加载专用左右手检测器（WiLoR YOLOv8m-Pose）')
                    from hand_automatic import AutomaticHands
                    automatic=AutomaticHands(engine)
                if kind=='hand_stream':
                    from hand_stream import process_hand_frame
                    emit('hand_stream_result',job=job,**process_hand_frame(automatic,command));continue
                if kind=='sequence':
                    from hand_sequence import run_sequence
                    run_sequence(automatic,command,emit);continue
                image_path=Path(command['image']);image=cv2.imread(str(image_path))
                if image is None:raise ValueError('无法读取输入图片')
                automatic.tracker.reset();outputs=[]
                for _ in range(3 if kind=='auto_warmup' else 1):
                    data,outputs,report=automatic.process(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),
                        confidence=command.get('confidence',.4),padding=command.get('padding',1.1),
                        matrix=command.get('camera_matrix'),stabilize=False,detect_only=kind=='auto_detect',hand_selection=command.get('hand_selection','both'))
                if kind=='auto_warmup' and len(report['hands'])==2:
                    emit('stage',job=job,message='补充预热单手批量，减少双手进出画面时首次卡顿')
                    for _ in range(3):
                        engine.predict_many(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),report['hands'][:1],padding=command.get('padding',1.1),matrix=command.get('camera_matrix'))
                    report['warmup_batch_sizes']=[1,2]
                output=Path(command['output']);output.mkdir(parents=True,exist_ok=True)
                if (output/'hands.npz').exists():raise ValueError('拒绝覆盖已有结果')
                cv2.imwrite(str(output/'input.png'),image)
                np.savez_compressed(output/'hands.npz',**data)
                for hand,(_,raw,crop,_) in zip(report['hands'],outputs):
                    cv2.imwrite(str(output/(hand['side']+'-crop.png')),cv2.cvtColor(crop,cv2.COLOR_RGB2BGR))
                    np.savez_compressed(output/(hand['side']+'-raw.npz'),**raw)
                report.update(camera_source=command.get('camera_source','estimated'),total_seconds=time.perf_counter()-job_start)
                (output/'request.json').write_text(json.dumps(command,ensure_ascii=False,indent=2))
                (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
                if len(data['vertices']):
                    import trimesh
                    trimesh.Trimesh(data['vertices'],data['faces'],process=False).export(output/'hands-camera.ply')
                emit('auto_result',job=job,output=str(output),report=report,kind=kind);continue
            if kind not in ('warmup','run'): raise ValueError('未知手部请求')
            image_path = Path(command['image']); image = cv2.imread(str(image_path))
            if image is None: raise ValueError('无法读取输入图片')
            rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            params = dict(bbox=command['hand_bbox'], side=command['side'], padding=command['padding'],matrix=command.get('camera_matrix'))
            count = 3 if kind == 'warmup' else 1
            for index in range(count):
                if kind == 'warmup':emit('stage',job=job,message=f'手部分支预热 {index+1}/{count}')
                data,raw,crop,report = engine.predict(rgb,**params)
            if kind == 'warmup':
                if command.get('output'):
                    warm_dir=Path(command['output']);warm_dir.mkdir(parents=True,exist_ok=True)
                    (warm_dir/'request.json').write_text(json.dumps(command,ensure_ascii=False,indent=2))
                    (warm_dir/'warmup.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
                emit('warmed',job=job,report=report); continue
            output = Path(command['output']); output.mkdir(parents=True,exist_ok=True)
            if (output/'hand.npz').exists(): raise ValueError('拒绝覆盖已有手部结果')
            cv2.imwrite(str(output/'input.png'),image)
            cv2.imwrite(str(output/'crop.png'),cv2.cvtColor(crop,cv2.COLOR_RGB2BGR))
            np.savez_compressed(output/'hand.npz',**data)
            np.savez_compressed(output/'raw_hand.npz',**raw)
            import trimesh
            trimesh.Trimesh(data['vertices'],data['faces'],process=False).export(output/'hand.ply')
            with image_path.open('rb') as stream:report['image_sha256']=hashlib.file_digest(stream,'sha256').hexdigest()
            report.update(source_image=str(image_path), camera_source=command.get('camera_source','estimated'),
                          total_seconds=time.perf_counter()-job_start)
            (output/'request.json').write_text(json.dumps(command,ensure_ascii=False,indent=2))
            (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
            emit('result',job=job,output=str(output),report=report)
        except Exception as exc:
            traceback.print_exc()
            emit('error',job=command.get('job'),message=str(exc))


if __name__ == '__main__':
    try:main()
    except Exception as exc:
        traceback.print_exc();emit('fatal',message=str(exc));sys.exit(1)
