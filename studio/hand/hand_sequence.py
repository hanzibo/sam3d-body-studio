# SPDX-License-Identifier: GPL-3.0-only
"""Offline automatic single/bimanual reconstruction and calibrated video export."""
import json
from pathlib import Path
import time
import numpy as np


def read_sequence(directory,stride=1):
    directory=Path(directory).resolve()
    if not 1<=stride<=120:raise ValueError('抽帧间隔应为 1–120')
    manifest=json.loads((directory/'manifest.json').read_text())
    if not manifest.get('complete'):raise ValueError('录制尚未完成，不读取进行中的序列')
    records=[json.loads(line) for line in (directory/'timestamps.jsonl').read_text().splitlines() if line.strip()]
    if not records or len(records)!=manifest['frames']:raise ValueError('录制帧数与时间戳不一致')
    previous=-1
    for row in records:
        image=(directory/row['file']).resolve()
        if not image.is_relative_to(directory) or not image.is_file():raise ValueError('序列帧缺失或路径越界')
        timestamp=float(row['relative_seconds'])
        if not np.isfinite(timestamp) or timestamp<=previous:raise ValueError('帧时间戳无效或不递增')
        previous=timestamp
    profile=manifest['profile']
    if not np.isfinite(float(profile['fps'])) or not 0<float(profile['fps'])<=240:raise ValueError('采集帧率无效')
    if not all(0<int(profile[key])<=16384 for key in ('width','height')):raise ValueError('图像尺寸无效')
    intr=profile['intrinsics'];matrix=np.array([[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1]],np.float32)
    if not np.isfinite(matrix).all() or min(matrix[0,0],matrix[1,1])<=0:raise ValueError('相机内参无效')
    return manifest,records[::stride],matrix


def render_video(directory,emit):
    import cv2
    import pyrender
    import trimesh
    directory=Path(directory);manifest=json.loads((directory/'sequence.json').read_text())
    rows=[json.loads(line) for line in (directory/'frames.jsonl').read_text().splitlines()]
    width=960;height=round(manifest['source_height']*width/manifest['source_width']);height+=height%2
    k=np.asarray(manifest['camera_matrix'],np.float32).copy();k[0]*=width/manifest['source_width'];k[1]*=height/manifest['source_height']
    renderer=pyrender.OffscreenRenderer(width,height)
    writer=cv2.VideoWriter(str(directory/'overlay.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),manifest['playback_fps'],(width,height))
    if not writer.isOpened():renderer.delete();raise ValueError('MP4 编码器无法启动')
    scene=pyrender.Scene(bg_color=[0,0,0,0],ambient_light=[.65,.65,.65])
    scene.add(pyrender.IntrinsicsCamera(k[0,0],k[1,1],k[0,2],k[1,2],znear=.01,zfar=100),pose=np.eye(4))
    scene.add(pyrender.DirectionalLight(color=np.ones(3),intensity=2),pose=np.eye(4))
    colors={'left':(.3,.9,.65,1.),'right':(.95,.65,.3,1.)}
    try:
        for index,row in enumerate(rows):
            image=cv2.imread(str(Path(manifest['source'])/row['input_file']))
            if image is None:raise ValueError('合成时源帧缺失')
            image=cv2.resize(image,(width,height));nodes=[]
            with np.load(directory/row['scene'],allow_pickle=False) as data:
                for side in data['sides']:
                    vertices=data[side+'_vertices']+data[side+'_camera_translation'];vertices=vertices.copy();vertices[:,1:]*=-1
                    mesh=trimesh.Trimesh(vertices,data[side+'_faces'],process=False)
                    material=pyrender.MetallicRoughnessMaterial(baseColorFactor=colors[str(side)],metallicFactor=0,roughnessFactor=.8)
                    nodes.append(scene.add(pyrender.Mesh.from_trimesh(mesh,material=material,smooth=True)))
                if nodes:
                    rgba,depth=renderer.render(scene,flags=pyrender.RenderFlags.RGBA)
                    mask=depth>0;image[mask]=(.85*rgba[:,:,:3][:,:,::-1][mask]+.15*image[mask]).astype(np.uint8)
                for hand in row['report']['hands']:
                    # Scale each xy pair once (four coordinates).
                    box=np.asarray(hand['bbox'])*np.tile([width/manifest['source_width'],height/manifest['source_height']],2)
                    x1,y1,x2,y2=box.astype(int);color=(100,230,90) if hand['side']=='left' else (60,170,250)
                    cv2.rectangle(image,(x1,y1),(x2,y2),color,1);cv2.putText(image,f"{hand['side']} {hand['score']:.2f}",(x1,max(15,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1,cv2.LINE_AA)
            missing=','.join(row['report']['missing'])
            cv2.putText(image,f"frame {row['source_index']} | {row['report']['elapsed_ms']:.0f} ms | missing: {missing or '-'}",(12,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1,cv2.LINE_AA)
            writer.write(image)
            for node in nodes:scene.remove_node(node)
            if index%15==0 or index==len(rows)-1:emit('stage',message=f'合成叠加视频 {index+1}/{len(rows)}')
    finally:writer.release();renderer.delete()
    return str(directory/'overlay.mp4')


def run_sequence(automatic,command,emit):
    import cv2
    source=Path(command['source']).resolve();directory=Path(command['output']).resolve()
    stride=int(command.get('stride',1));manifest,records,matrix=read_sequence(source,stride)
    if directory.exists() and any(directory.iterdir()):raise ValueError('拒绝覆盖已有序列结果')
    directory.mkdir(parents=True,exist_ok=True);(directory/'frames').mkdir()
    profile=manifest['profile'];source_width=profile['width'];source_height=profile['height']
    info=dict(schema=1,execution=automatic.engine.execution_actual,complete=False,source=str(source),source_frames=manifest['frames'],frames=len(records),stride=stride,
              source_width=source_width,source_height=source_height,camera_matrix=matrix.tolist(),source_manifest=manifest,
              playback_fps=float(profile['fps'])/stride,detector=automatic.detector.metadata,
              policy='detect each sampled frame; at most one left and one right; no stale mesh when absent',
              options={key:command.get(key) for key in ('confidence','padding','stabilize','render_video','hand_selection')})
    (directory/'sequence.json').write_text(json.dumps(info,indent=2,ensure_ascii=False));automatic.tracker.reset();rows=[]
    started=time.perf_counter()
    with (directory/'frames.jsonl').open('w') as stream:
        for index,record in enumerate(records):
            frame_start=time.perf_counter();image=cv2.imread(str(source/record['file']))
            if image is None or image.shape[:2]!=(source_height,source_width):raise ValueError('帧无法读取或尺寸与标定不符')
            rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            data,outputs,report=automatic.process(rgb,confidence=command.get('confidence',.4),padding=command.get('padding',1.1),
                matrix=matrix,timestamp=record['relative_seconds'],stabilize=command.get('stabilize',True),hand_selection=command.get('hand_selection','both'))
            file=f'frames/{index:06d}.npz';np.savez_compressed(directory/file,**data)
            report['frame_with_io_ms']=(time.perf_counter()-frame_start)*1000
            row=dict(index=index,source_index=record['sequence'],timestamp=record['relative_seconds'],input_file=record['file'],scene=file,report=report)
            stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush();rows.append(row)
            emit('frame',job=command['job'],output=str(directory),row=row,total=len(records))
    network=np.array([r['report']['reconstruct_ms'] for r in rows]);detect=np.array([r['report']['detect_ms'] for r in rows]);total=np.array([r['report']['frame_with_io_ms'] for r in rows])
    summary=dict(frames=len(rows),processing_seconds=time.perf_counter()-started,
                 detection_mean_ms=float(detect.mean()),reconstruction_mean_ms=float(network.mean()),frame_with_io_mean_ms=float(total.mean()),
                 frame_with_io_p95_ms=float(np.percentile(total,95)),
                 hand_counts={str(n):sum(r['report']['hand_count']==n for r in rows) for n in (0,1,2)},
                 left_present=sum('left' not in r['report']['missing'] for r in rows),right_present=sum('right' not in r['report']['missing'] for r in rows),
                 label_corrections=sum(h['raw_side']!=h['side'] for r in rows for h in r['report']['hands']),
                 timing_scope='offline wall; includes first-use overhead; no rendering; not live camera FPS or labeled accuracy')
    info.update(complete=True,summary=summary);(directory/'sequence.json').write_text(json.dumps(info,ensure_ascii=False,indent=2))
    (directory/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    if command.get('render_video',True):
        try:
            video=render_video(directory,lambda kind,**kw:emit(kind,job=command['job'],**kw));info['video']=video
        except Exception as exc:info['video_error']=str(exc)
        (directory/'sequence.json').write_text(json.dumps(info,ensure_ascii=False,indent=2))
    from studio.hand.hand_sequence_metrics import summarize
    summarize(directory)
    emit('sequence_done',job=command['job'],output=str(directory),report=info)
