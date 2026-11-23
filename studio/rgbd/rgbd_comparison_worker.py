# SPDX-License-Identifier: GPL-3.0-only
"""Three-method frozen-sequence benchmark inside the exclusive Fast worker."""
import json,time
from pathlib import Path
import cv2
import numpy as np
import torch
from studio.rgbd.rgbd_comparison_data import FullGeometry,atomic_npz,load_npz,METHODS
from studio.rgbd.rgbd_sequence import RGBDSequence
from studio.video.video_project import file_hash,save_json
from studio.rgbd.rgbd_geometry import matrix
from studio.rgbd.depth_pose_refinement import MhrSurfaceModel,SurfaceFit,PoseDepthConfig,acceptance
from studio.rgbd.depth_pose_accelerated import CudaPoseFitter
from studio.rgbd.depth_pose_render import FittingDepthRenderer
from studio.rgbd.depth_refinement import refine_translation


def run_comparison(command,forward,faces,emit):
    src=Path(command['directory']);dest=Path(command['output']);source=json.loads((src/'inputs.json').read_text())
    manifest=json.loads((dest/'manifest.json').read_text())
    if manifest['inputs_sha256']!=file_hash(src/'inputs.json'):raise ValueError('inputs changed')
    seq=RGBDSequence(source['source']);geometry=FullGeometry(seq);k=geometry.k
    if file_hash(seq.directory/'camera.json')!=source['camera_sha256'] or file_hash(seq.directory/'frames.jsonl')!=source['index_sha256']:raise ValueError('raw source changed')
    refs=dest/'reference';ref=json.loads((refs/'manifest.json').read_text())
    model=MhrSurfaceModel();graphs={};renderer=FittingDepthRenderer(faces,k)
    (dest/'frames').mkdir(exist_ok=True);(dest/'network').mkdir(exist_ok=True)
    np.savez_compressed(dest/'topology.npz',faces=faces)
    trackrows=[json.loads(r) for r in (src/'candidate-15/frames.jsonl').read_text().splitlines()]
    cold=[]
    try:
        for i,entry in enumerate(source['frames']):
            file=dest/'frames'/f'{i:06d}.npz';info=file.with_suffix('.json')
            if file.exists() and info.exists():
                saved=json.loads(info.read_text())
                if saved['artifact_sha256']!=file_hash(file):raise ValueError('changed completed frame')
                continue
            started=time.perf_counter();image,xyz,_=geometry.read(entry['source_index']);geometry_ms=(time.perf_counter()-started)*1000
            maskfile=refs/f'{i:06d}.png'
            if file_hash(maskfile)!=ref['frames'][i]['sha256']:raise ValueError('reference changed')
            mask=cv2.imread(str(maskfile),0)>0
            meshes=np.zeros((3,18439,3),np.float32);joints=np.zeros((3,127,3),np.float32);keypoints=np.zeros((3,70,3),np.float32)
            valid=np.zeros(3,bool);rows=[];fitreports=[]
            if entry['bbox'] is not None:
                network=dest/'network'/f'{i:06d}.npz';networkmeta=network.with_suffix('.json')
                old=src/'network_keyframes'/f'{i:06d}.npz'
                if network.exists() and networkmeta.exists():
                    netinfo=json.loads(networkmeta.read_text());arrays=load_npz(network)
                    if netinfo['rgb_sha256']!=entry['rgb_sha256'] or netinfo['artifact_sha256']!=file_hash(network):raise ValueError('network cache changed')
                elif old.exists():
                    netinfo=json.loads(old.with_suffix('.json').read_text())
                    if netinfo['rgb_sha256']!=entry['rgb_sha256']:raise ValueError('old RGB cache mismatch')
                    arrays=load_npz(old);atomic_npz(network,**arrays)
                    netinfo.update(origin='existing same-profile keyframe; original measured time retained',artifact_sha256=file_hash(network))
                    save_json(networkmeta,netinfo)
                else:
                    cam=torch.from_numpy(matrix(k).astype(np.float32)).unsqueeze(0).cuda()
                    output,seconds=forward(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),'body',entry['bbox'],cam)
                    arrays={key:value for key,value in output[0].items() if value is not None};arrays['faces']=faces
                    atomic_npz(network,**arrays);netinfo=dict(network_seconds=seconds,rgb_sha256=entry['rgb_sha256'],artifact_sha256=file_hash(network),origin='new synchronized same-profile inference')
                    save_json(networkmeta,netinfo)
                detector=entry['detector_seconds']*1000;network_ms=netinfo['network_seconds']*1000
                seedtime=time.perf_counter();model.seed(arrays);seed_ms=(time.perf_counter()-seedtime)*1000
                if model.state['correctives']:raise ValueError('Fast body comparison expected correctives=False')
                initial=model.base.numpy().copy();translation=arrays['pred_cam_t'].copy()
                meshes[0]=arrays['pred_vertices']+translation;joints[0]=arrays['pred_joint_coords']+translation;keypoints[0]=arrays['pred_keypoints_3d']+translation;valid[0]=True
                rows.append(dict(method='independent',valid=True,status='PREDICTED',network_ms=network_ms,detector_ms=detector,core_ms=network_ms+detector,estimated_input_ms=geometry_ms+network_ms+detector))
                # Full established baseline: translation-only seed, rigid60, pose60.
                for body in (False,True):
                    if body not in graphs:
                        graphs[body]=CudaPoseFitter(model,body);cold.append(dict(body=body,setup_ms=graphs[body].setup_ms))
                torch.cuda.synchronize();begin=time.perf_counter();params=initial.copy();t=translation.copy();v=arrays['pred_vertices'].copy();fitstatus='NO_REFERENCE_FALLBACK_RAW'
                if ref['frames'][i].get('segmented',True) and mask.sum()>300:
                    delta,_,tr=refine_translation(xyz[:,:,2],xyz,mask,renderer.render(v+t),k,lambda d:renderer.render(v+t+d))
                    t=t+delta;fitreports.append(dict(stage='translation',report=tr));conf=PoseDepthConfig(steps=60)
                    try:surface=SurfaceFit(xyz[:,:,2],xyz,mask,k,renderer.render,conf)
                    except ValueError as e:
                        fitreports.append(dict(stage='surface',reason=str(e)))
                        surface=None;fitstatus='INSUFFICIENT_DEPTH_FALLBACK'
                    if surface is not None:
                        fitstatus='REFINED'
                        for body in (False,True):
                            model.base=torch.tensor(params);before,_=surface.score(v+t)
                            p,tt,report=graphs[body].fit(model,surface,1.,t)
                            accepted=False;trials=[]
                            if report['sufficient']:
                                for fraction in (1.,.5,.25):
                                    pp=params+(p-params)*fraction;pt=t+(tt-t)*fraction
                                    with torch.no_grad():vv=model.forward(torch.tensor(pp))[0].numpy()
                                    after,_=surface.score(vv+pt);reason=acceptance(before,after);trials.append(dict(fraction=fraction,reason=reason,guard=after))
                                    if reason=='accepted':params,t,v=pp,pt,vv;accepted=True;break
                            fitreports.append(dict(stage='rigid60' if not body else 'pose60',report=report,accepted=accepted,trials=trials))
                with torch.no_grad():vv,jj,kk,_=model.forward(torch.tensor(params))
                meshes[1]=vv.numpy()+t;joints[1]=jj.numpy()+t;keypoints[1]=kk.numpy()+t;valid[1]=True
                torch.cuda.synchronize();post_ms=(time.perf_counter()-begin)*1000+seed_ms
                rows.append(dict(method='cuda60',valid=True,status=fitstatus,network_ms=network_ms,detector_ms=detector,postprocess_ms=post_ms,core_ms=network_ms+detector+post_ms,estimated_input_ms=geometry_ms+network_ms+detector+post_ms))
                # Restore exact prior saved tracking state, not a rerun or independently fitted mesh.
                tr=trackrows[i];statefile=src/'candidate-15/states'/f'{i:06d}.npz'
                if statefile.exists():
                    st=load_npz(statefile);model.identity=torch.tensor(st['identity']);model.expression=torch.tensor(st['expression'])
                    with torch.no_grad():vv,jj,kk,_=model.forward(torch.tensor(st['params']))
                    meshes[2]=vv.numpy()+st['translation'];joints[2]=jj.numpy()+st['translation'];keypoints[2]=kk.numpy()+st['translation'];valid[2]=bool(st['valid'])
            else:
                rows=[dict(method=m,valid=False,status='LOST',core_ms=entry['detector_seconds']*1000,estimated_input_ms=geometry_ms+entry['detector_seconds']*1000) for m in METHODS[:2]]
            tr=trackrows[i];rows.append(dict(method='tracking',valid=bool(valid[2]),status=tr['state'],keyframe=tr['keyframe'],core_ms=sum(tr.get('stages_ms',{}).get(key,0) for key in ('keyframe_setup_ms','fit_ms','accept_ms'))+tr['detector_ms']+(tr.get('network_ms',0) if tr.get('network_cached') else 0),estimated_input_ms=tr['estimated_observed_input_ms'],recorded_previous_run=True))
            for row in rows:row.update(frame=i,time_s=entry['time_s'],source_index=entry['source_index'])
            if not np.isfinite(meshes).all():raise ValueError('nonfinite mesh')
            atomic_npz(file,vertices=meshes,joints=joints,keypoints=keypoints,valid=valid)
            save_json(info,dict(frame=i,rgb_sha256=entry['rgb_sha256'],geometry_ms=geometry_ms,methods=rows,fit_reports=fitreports,artifact_sha256=file_hash(file)))
            if i%10==0:emit('stage',message=f'comparison {src.name} {i}/{len(source["frames"])}')
        save_json(dest/'complete.json',dict(frames=len(source['frames']),cold_graphs=cold,manifest_sha256=file_hash(dest/'manifest.json'),timing='Offline replay with actual measured/cached same-profile network cost; exports/evaluation/video excluded from core; not online throughput. Tracking recorded in previous run.'))
        emit('comparison_complete',directory=str(dest))
    finally:renderer.close()
