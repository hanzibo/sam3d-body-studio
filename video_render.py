# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Persistent multi-mesh rasterization and timestamp-aware video composition."""
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import cv2
from video_project import load_project,frame_path,annotation_key,save_json,digest
from pose_data import EDGES

PALETTE=[(112,190,255),(100,240,150),(255,160,100),(220,140,245)]


def render_video(directory,run,layers,emit):
    from gpu_lease import acquire_gpu_lease
    lease=acquire_gpu_lease('exclusive')
    directory,project=load_project(directory);run=Path(run).resolve()
    if not run.is_relative_to(directory/'runs'):raise ValueError('运行记录不在此视频任务中')
    summary=json.loads((run/'run.json').read_text())
    if summary['status']!='complete' or summary['source_sha256']!=project['source_sha256'] or summary['annotations']!=annotation_key(project):
        raise ValueError('请选择与当前分段提示匹配、已完成的重建记录')
    w,h=project['width'],project['height'];records={}
    for record in summary['records']:records.setdefault(record['frame'],[]).append(record)
    cache=run/('render-'+layers);cache.mkdir(exist_ok=True)
    renderer=None
    if layers in ('mesh','both') and summary['successful']:
        os.environ.setdefault('PYOPENGL_PLATFORM','egl')
        import pyrender
        import trimesh
        renderer=pyrender.OffscreenRenderer(w,h)
        faces=np.load(run/'faces.npy',allow_pickle=False)
    try:
        for index in range(project['frame_count']):
            image=cv2.imread(str(frame_path(directory,index)))
            if image is None:raise ValueError('输入视频帧缺失')
            items=records.get(index,[]);scene=None;camera_key=None
            if renderer:
                scene=pyrender.Scene(bg_color=[0,0,0,0],ambient_light=[.65,.65,.65])
                light=pyrender.DirectionalLight(color=np.ones(3),intensity=2.)
                scene.add(light,pose=np.eye(4))
            annotations=[]
            for record in items:
                ident=record['person_id'];color=PALETTE[(ident-1)%len(PALETTE)]
                if record['status']!='ok':continue
                file=run/'predictions'/f'{index:06d}'/f'person-{ident:03d}'/'prediction.npz'
                with np.load(file,allow_pickle=False) as data:
                    points=np.asarray(data['pred_keypoints_2d']).copy()
                    if scene is not None:
                        matrix=record['camera_intrinsics_used']['matrix']
                        if camera_key is None:
                            camera_key=matrix;K=np.array(matrix)
                            scene.add(pyrender.IntrinsicsCamera(K[0,0],K[1,1],K[0,2],K[1,2],znear=.01,zfar=100.),pose=np.eye(4))
                        elif matrix!=camera_key:raise ValueError('同一帧人物的相机内参不一致')
                        vertices=(data['pred_vertices']+data['pred_cam_t'])*[1,-1,-1]
                        if not np.isfinite(vertices).all():raise ValueError('渲染顶点无效')
                        material=pyrender.MetallicRoughnessMaterial(baseColorFactor=[v/255 for v in color]+[1.],metallicFactor=0.,roughnessFactor=.8,doubleSided=True)
                        scene.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(vertices=vertices,faces=faces,process=False),material=material,smooth=True))
                    annotations.append((ident,color,points))
            if scene is not None and camera_key is not None:
                rgba,depth=renderer.render(scene,flags=pyrender.RenderFlags.RGBA)
                visible=depth>0;mesh_bgr=rgba[:,:,:3][:,:,::-1]
                image[visible]=(image[visible]*.18+mesh_bgr[visible]*.82).astype(np.uint8)
            for ident,color,points in annotations:
                if layers in ('skeleton','both'):
                    for a,b in EDGES:
                        if np.isfinite(points[[a,b]]).all():
                            # Clip off-image projections before conversion to C integer coordinates.
                            p=np.clip(points[[a,b]],-10000,10000).astype(int)
                            cv2.line(image,tuple(p[0]),tuple(p[1]),color[::-1],2,cv2.LINE_AA)
                valid=points[np.isfinite(points).all(axis=1)]
                if len(valid):
                    at=(int(np.clip(valid[:,0].mean(),0,w-90)),int(np.clip(valid[:,1].min()-12,24,h-1)))
                    cv2.putText(image,f'ID {ident}',at,cv2.FONT_HERSHEY_SIMPLEX,.65,color[::-1],2,cv2.LINE_AA)
            failures=[str(r['person_id']) for r in items if r['status']=='failed']
            if failures:cv2.putText(image,'Failed ID: '+','.join(failures),(12,28),cv2.FONT_HERSHEY_SIMPLEX,.6,(40,40,255),2)
            if not cv2.imwrite(str(cache/f'{index:06d}.png'),image):raise OSError('渲染帧写入失败')
            if index%10==0:emit('progress',message=f'渲染 {index+1}/{project["frame_count"]}',current=index+1,total=project['frame_count'])
    finally:
        if renderer:renderer.delete()
    emit('stage',message='正在合成视频并保留原音轨')
    source=Path(project['source'])
    if project['has_audio'] and not source.is_file():raise ValueError('保留音频需要原始视频仍位于导入路径')
    pts=np.asarray(project['pts']);rate=(len(pts)-1)/(pts[-1]-pts[0]) if len(pts)>1 else 1/project['duration']
    regular=len(pts)<3 or np.max(np.abs(np.diff(pts)-1/rate))<1e-5
    command=['ffmpeg','-y','-v','error']
    if regular:command+=['-framerate',str(rate),'-i',str(cache/'%06d.png')]
    else:
        # An ffconcat image demuxer defaults to 25 Hz; override its time base.
        listing=cache/'frames.ffconcat';lines=['ffconcat version 1.0']
        for i in range(len(pts)):
            duration=(pts[i+1]-pts[i]) if i+1<len(pts) else project['duration']-pts[i]
            lines += [f"file '{i:06d}.png'",'option framerate 1000000',f'duration {duration:.9f}']
        lines += [f"file '{len(pts)-1:06d}.png'",'option framerate 1000000']
        listing.write_text('\n'.join(lines)+'\n');command+=['-safe','0','-f','concat','-i',str(listing)]
    # VFR needs a sentinel frame at the end to establish the last sample's
    # duration. A lossless remux below removes that sentinel, preserving PTS.
    command+=['-an','-frames:v',str(project['frame_count']+(not regular)),'-c:v','libx264','-preset','fast','-bf','0',
              '-crf','18','-pix_fmt','yuv420p','-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2','-fps_mode','vfr',
              '-enc_time_base','1:1000000','-video_track_timescale','1000000']
    output=run/f'overlay-{layers}.mp4';temporary=run/f'overlay-{layers}.partial.mp4'
    encoded=run/f'overlay-{layers}.encoded.mp4'
    subprocess.run(command+[str(encoded)],check=True)
    mux=['ffmpeg','-y','-v','error','-i',str(encoded)]
    if project['has_audio']:mux+=['-ss',str(project['source_time_origin']),'-i',str(source)]
    mux+=['-map','0:v:0','-c:v','copy']
    if project['has_audio']:mux+=['-map','1:a:0?','-c:a','aac','-b:a','128k']
    mux+=['-t',str(project['duration']),'-video_track_timescale','1000000','-movflags','+faststart',str(temporary)]
    subprocess.run(mux,check=True)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams','-show_format','-of','json',str(temporary)]))
    stream=next(s for s in probe['streams'] if s['codec_type']=='video')
    if int(stream['nb_read_frames'])!=project['frame_count']:raise ValueError('合成视频帧数不一致')
    if abs(float(stream['duration'])-project['duration'])>max(.05,1/rate+.005):raise ValueError('合成视频时间轴不一致')
    if project['has_audio'] and not any(s['codec_type']=='audio' for s in probe['streams']):raise ValueError('音轨未成功保留')
    timeline=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','frame=best_effort_timestamp_time','-of','json',str(temporary)]))
    actual=np.array([float(f['best_effort_timestamp_time']) for f in timeline['frames']])
    if len(actual)!=len(pts) or np.max(np.abs(actual-pts))>.001:raise ValueError('合成视频帧时间戳与原始输入不一致')
    temporary.replace(output)
    encoded.unlink()
    save_json(cache/'complete.json',dict(layers=layers,frames=project['frame_count'],output=str(output),probe=probe,
        prediction_revision=summary.get('prediction_revision')))
    emit('rendered',video=str(output))
