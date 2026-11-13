# SPDX-License-Identifier: GPL-3.0-only
"""Full-resolution visible-surface metrics, synchronized playback and replay assets."""
import os
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import argparse,csv,html,json,subprocess,time
from pathlib import Path
import cv2
import numpy as np
import pyrender,trimesh
from PIL import Image,ImageDraw,ImageFont
from rgbd_sequence import RGBDSequence
from rgbd_comparison_data import FullGeometry,METHODS,LABELS,METHOD_LABELS,comparison_methods,load_npz,atomic_npz,measured_cloud,static_jitter
from rgbd_geometry import depth_metrics,surface_distances
from rgbd_evaluate import DepthRenderer
from rgbd_tracking_points import point_membership
from video_project import file_hash,save_json

FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def text(image,lines,size=19):
    p=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB));draw=ImageDraw.Draw(p);font=ImageFont.truetype(FONT,size)
    for line,xy,color in lines:draw.text(xy,line,font=font,fill=color)
    return cv2.cvtColor(np.asarray(p),cv2.COLOR_RGB2BGR)


def look_at(eye,target):
    z=(eye-target);z=z/np.linalg.norm(z);x=np.cross([0.,1.,0.],z);x=x/np.linalg.norm(x);y=np.cross(z,x)
    pose=np.eye(4);pose[:3,:3]=np.column_stack((x,y,z));pose[:3,3]=eye;return pose


class SceneRenderer:
    def __init__(self,faces,center,shared_renderer=None):
        self.faces=faces;self.owns_renderer=shared_renderer is None;self.renderer=shared_renderer if shared_renderer is not None else pyrender.OffscreenRenderer(640,360,point_size=2)
        self.pose=look_at(center+np.array([2.,.35,2.]),center)
    def render(self,vertices,cloud,colors):
        self.renderer.viewport_width=640;self.renderer.viewport_height=360;self.renderer.point_size=2
        scene=pyrender.Scene(bg_color=[.035,.06,.10,1],ambient_light=[.65,.65,.65])
        scene.add(pyrender.IntrinsicsCamera(420,420,320,180,znear=.05,zfar=20),pose=self.pose)
        p=cloud.copy();p[:,1:]*=-1
        scene.add(pyrender.Mesh.from_points(p,colors=colors))
        if vertices is not None:
            v=vertices.copy();v[:,1:]*=-1
            material=pyrender.MetallicRoughnessMaterial(baseColorFactor=[.70,.84,1.,.62],metallicFactor=0.,roughnessFactor=.8,alphaMode='BLEND')
            scene.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(v,self.faces,process=False),material=material,smooth=True))
        scene.add(pyrender.DirectionalLight(color=np.ones(3),intensity=2.),pose=self.pose)
        return self.renderer.render(scene,flags=pyrender.RenderFlags.SKIP_CULL_FACES)[0][:,:,::-1]
    def close(self):
        if self.owns_renderer:self.renderer.delete()


def overlay(image,z,k):
    out=image.copy();valid=z>.05
    # Depth-derived shading makes articulation visible without changing geometry.
    smooth=cv2.GaussianBlur(z,(3,3),0);dx=cv2.Sobel(smooth,cv2.CV_32F,1,0)/8;dy=cv2.Sobel(smooth,cv2.CV_32F,0,1)/8
    nx=-dx*k['fx']/np.maximum(z,.05);ny=-dy*k['fy']/np.maximum(z,.05)
    light=np.clip((1+.25*nx-.35*ny)/np.sqrt(nx*nx+ny*ny+1),0,1)
    body=np.stack((210+35*light,180+55*light,160+70*light),-1)
    out[valid]=(.35*out[valid]+.65*body[valid]).astype(np.uint8)
    return out


def metrics_for(depth,xyz,mask,z,vertices,faces,points):
    valid=mask&(depth>.05)&np.isfinite(depth);covered=valid&(z>.05)
    # Penalize missing surface rather than reporting an attractive MAE on tiny overlap.
    capped=np.full(int(valid.sum()),.1)
    if valid.any():capped=np.minimum(np.where(z[valid]>.05,np.abs(z[valid]-depth[valid]),.1),.1)
    result=dict(capped_mae_100mm=float(capped.mean()*1000) if len(capped) else None,observed_depth_pixels=int(valid.sum()),evaluated_point_count=len(points))
    if covered.any():result.update(depth_metrics(depth,z,mask)[0])
    else:result.update(mae_mm=None,p90_mm=None,silhouette_iou=float((mask&(z>.05)).sum()/max(1,(mask|(z>.05)).sum())),model_coverage_fraction=0.)
    if len(points):
        dist=surface_distances(points,vertices,faces)*1000
        result.update(point_surface_mae_mm=float(dist.mean()),point_surface_p90_mm=float(np.percentile(dist,90)))
    else:result.update(point_surface_mae_mm=None,point_surface_p90_mm=None)
    return result


def summarize(rows,methods=METHODS):
    summary=[];common=set.intersection(*[{r['frame'] for r in rows if r['method']==method and r['valid'] and r['reference_segmented'] and r['observed_depth_pixels']>0} for method in methods])
    for method in methods:
        allr=[r for r in rows if r['method']==method];group=[r for r in allr if r['frame'] in common and r['valid']]
        s=dict(method=method,frames=len(allr),valid_frames=sum(r['valid'] for r in allr),common_quality_frames=len(group),
               reference_suspect_frames=sorted({r['frame'] for r in group if r['reference_needs_review']}),jitter=static_jitter(rows,method))
        for key in ('mae_mm','p90_mm','silhouette_iou','model_coverage_fraction','point_surface_mae_mm','point_surface_p90_mm','capped_mae_100mm','yolo_reprojection_px','landmark_agreement_cuda60_mm','root_relative_landmark_agreement_cuda60_mm'):
            vals=[r[key] for r in group if r.get(key) is not None];s[key]=float(np.mean(vals)) if vals else None
        clean=[r for r in group if not r['reference_needs_review']]
        s['reference_screened_frames']=len(clean);s['reference_screened_mae_mm']=float(np.mean([r['mae_mm'] for r in clean if r.get('mae_mm') is not None])) if any(r.get('mae_mm') is not None for r in clean) else None
        for side in ('left','right'):
            angles=[(r['time_s'],r['elbow_'+side+'_deg']) for r in allr if r['valid'] and .25<=r['time_s']<=1.75 and 'elbow_'+side+'_deg' in r]
            if len(angles)>=5:
                t,a=np.asarray(angles).T;x=np.column_stack((t-t.mean(),np.ones(len(t))));s['static_elbow_'+side+'_detrended_std_deg']=float(np.std(a-x@np.linalg.lstsq(x,a,rcond=None)[0]))
        for key in ('core_ms','estimated_input_ms'):
            for scope,rr in [('all',allr),('visible',[r for r in allr if r['frame'] in common])]:
                values=[r[key] for r in rr if r.get(key) is not None];s[f'{scope}_{key}_median']=float(np.median(values)) if values else None;s[f'{scope}_{key}_p95']=float(np.percentile(values,95)) if values else None
        own=[r for r in allr if r['valid'] and r['reference_segmented'] and r['observed_depth_pixels']>0]
        s['own_valid_quality_frames']=len(own)
        for key in ('mae_mm','point_surface_mae_mm'):
            values=[r[key] for r in own if r.get(key) is not None];s['own_valid_'+key]=float(np.mean(values)) if values else None
        s['time_windows']=[]
        for start in (0,5,10,15):
            window=[r for r in allr if start<=r['time_s']<start+5];valid=[r for r in window if r['valid']]
            item=dict(start_s=start,end_s=start+5,frames=len(window),valid_frames=len(valid))
            for key in ('mae_mm','point_surface_mae_mm'):
                values=[r[key] for r in valid if r.get(key) is not None];item[key]=float(np.mean(values)) if values else None
            s['time_windows'].append(item)
        summary.append(s)
    return summary


def evaluate(directory):
    directory=Path(directory).resolve();complete=directory/'report/complete.json'
    if complete.exists():
        previous=json.loads(complete.read_text())
        if previous['evaluation_source_sha256']!=file_hash(Path(__file__)):raise ValueError('report code changed; preserve old report before rerun')
        print('REPORT_COMPLETE',directory.name,flush=True);return
    methods=comparison_methods(directory)
    manifest=json.loads((directory/'manifest.json').read_text());source=json.loads((directory/'inputs.json').read_text())
    if not (directory/'complete.json').exists():raise ValueError('inference not complete')
    base_reader=None;base_rows=None
    if manifest.get('base_comparison'):
        base=Path(manifest['base_comparison'])
        for file,key in ((base/'report/rows.json','base_rows_sha256'),(base/'report/comparison.mp4','base_video_sha256')):
            if file_hash(file)!=manifest[key]:raise ValueError('frozen base report changed')
        if file_hash(base/'reference/manifest.json')!=file_hash(directory/'reference/manifest.json') or file_hash(base/'inputs.json')!=file_hash(directory/'inputs.json'):raise ValueError('reference/input mismatch')
        base_rows=json.loads((base/'report/rows.json').read_text());base_reader=cv2.VideoCapture(str(base/'report/comparison.mp4'))
    ref=json.loads((directory/'reference/manifest.json').read_text());faces=load_npz(directory/'topology.npz')['faces']
    geometry=FullGeometry(RGBDSequence(source['source']));k=geometry.k
    report=directory/'report';report.mkdir(exist_ok=True);(report/'replay').mkdir(exist_ok=True);(report/'snapshots').mkdir(exist_ok=True)
    renderer=DepthRenderer(faces,k);first=load_npz(directory/'frames/000000.npz');center=(first['vertices'][2].min(0)+first['vertices'][2].max(0))*.5*[1,-1,-1]
    scene=SceneRenderer(faces,center,renderer.renderer);rows=[];elapsed=time.perf_counter()
    span=source['frames'][-1]['time_s']-source['frames'][0]['time_s'];fps=(len(source['frames'])-1)/span
    video=report/'comparison.mp4';width,height=640*len(methods),840
    process=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{width}x{height}','-r',f'{fps:.9f}','-i','-','-an','-c:v','libx264','-threads','4','-preset','fast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(video.with_suffix('.partial.mp4'))],stdin=subprocess.PIPE)
    try:
        for i,entry in enumerate(source['frames']):
            f=directory/'frames'/f'{i:06d}.npz';info=json.loads(f.with_suffix('.json').read_text())
            if file_hash(f)!=info['artifact_sha256']:raise ValueError('mesh artifact changed')
            arrays=load_npz(f);image,xyz,_=geometry.read(entry['source_index']);maskfile=directory/'reference'/f'{i:06d}.png'
            if file_hash(maskfile)!=ref['frames'][i]['sha256']:raise ValueError('reference changed')
            mask=cv2.imread(str(maskfile),0)>0;rawpoints=xyz[mask&(xyz[:,:,2]>.05)]
            if len(rawpoints)>256:rawpoints=rawpoints[np.random.default_rng(i).choice(len(rawpoints),256,replace=False)]
            observed=entry.get('keypoints');suspect=False
            if observed is not None:
                kp=np.asarray(observed)[[0,5,6,7,8,11,12,13,14]];kp=kp[(kp[:,2]>.7)&(kp[:,0]>=0)&(kp[:,0]<mask.shape[1])&(kp[:,1]>=0)&(kp[:,1]<mask.shape[0])]
                if len(kp)>=3:suspect=float(point_membership(kp[:,:2],cv2.dilate(mask.astype(np.uint8),np.ones((7,7),np.uint8))>0).mean())<.75
            cloud=measured_cloud(xyz,image,mask);atomic_npz(report/'replay'/f'{i:06d}.npz',**cloud)
            cv2.imwrite(str(report/'replay'/f'{i:06d}.jpg'),cv2.resize(image,(640,360)),[cv2.IMWRITE_JPEG_QUALITY,88])
            canvas=np.full((height,width,3),(28,19,12),np.uint8)
            if base_reader is not None:
                ok,previous=base_reader.read()
                if not ok or previous.shape!=(840,1920,3):raise ValueError('base video incomplete')
                canvas[:,:1920]=previous
            canvas=text(canvas,[(f'{source["name"]}  |  {entry["time_s"]:.3f}s  |  帧 {i} / 源帧 {entry["source_index"]}   相机系米制 · 未额外对齐',(18,6),'#e2e8f0')],20)
            for m,method in enumerate(methods):
                if base_rows is not None and m<3:
                    row=dict(base_rows[3*i+m])
                    if row['frame']!=i or row['method']!=method:raise ValueError('base row order mismatch')
                    rows.append(row);continue
                row=dict(info['methods'][m]);row.update(reference_needs_review=suspect,reference_segmented=ref['frames'][i].get('segmented',True),observed_depth_pixels=int((mask&(xyz[:,:,2]>.05)).sum()))
                vertices=arrays['vertices'][m] if arrays['valid'][m] else None
                renderer.renderer.viewport_width=k['width'];renderer.renderer.viewport_height=k['height']
                z=renderer.render(vertices) if vertices is not None else np.zeros(mask.shape,np.float32)
                if vertices is not None:
                    row.update(metrics_for(xyz[:,:,2],xyz,mask,z,vertices,faces,rawpoints))
                    row['root_camera_m']=arrays['joints'][m,0].tolist();row['wrist_camera_m']=arrays['keypoints'][m,[62,41]].tolist()
                    BODY_IDS=[0,5,6,7,8,9,10,11,12,13,14,41,62,69]
                    if arrays['valid'][1]:
                        a=arrays['keypoints'][m,BODY_IDS];b=arrays['keypoints'][1,BODY_IDS]
                        row['landmark_agreement_cuda60_mm']=float(np.linalg.norm(a-b,axis=1).mean()*1000)
                        row['root_relative_landmark_agreement_cuda60_mm']=float(np.linalg.norm((a-arrays['joints'][m,0])-(b-arrays['joints'][1,0]),axis=1).mean()*1000)
                    for side,ids in [('left',(5,7,62)),('right',(6,8,41))]:
                        shoulder,elbow,wrist=arrays['keypoints'][m,list(ids)];u=shoulder-elbow;v=wrist-elbow
                        row['elbow_'+side+'_deg']=float(np.rad2deg(np.arccos(np.clip(np.dot(u,v)/max(1e-8,np.linalg.norm(u)*np.linalg.norm(v)),-1,1))))
                    if observed is not None:
                        from rgbd_tracking_points import COCO_TO_MHR,project
                        uv=project(arrays['keypoints'][m],k);obs=np.asarray(observed);ids=[j for j in (5,6,7,8,9,10,11,12,13,14) if obs[j,2]>.5]
                        row['yolo_reprojection_px']=float(np.linalg.norm(uv[np.asarray(COCO_TO_MHR)[ids]]-obs[ids,:2],axis=1).mean()) if ids else None
                else:row.update(mae_mm=None,p90_mm=None,point_surface_mae_mm=None,silhouette_iou=0.,model_coverage_fraction=0.,capped_mae_100mm=100. if mask.any() else None)
                rows.append(row)
                x=m*640;rgb=overlay(image,z,k)
                contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(rgb,contours,-1,(80,220,80),1)
                canvas[82:442,x:x+640]=cv2.resize(rgb,(640,360));canvas[480:840,x:x+640]=scene.render(vertices,cloud['cloud'][cloud['person']] if cloud['person'].any() else cloud['cloud'],cloud['colors'][cloud['person']] if cloud['person'].any() else cloud['colors'])
                mae=f'{row["mae_mm"]:.1f}mm' if row.get('mae_mm') is not None else '—'
                canvas=text(canvas,[(METHOD_LABELS[method],(x+12,42),'#8ec5ff'),(f'{row["status"]}  MAE {mae}  IoU {row["silhouette_iou"]:.3f}',(x+12,446),'#e2e8f0')],18)
            if suspect:canvas=text(canvas,[('参考掩膜待核查；此帧指标为暂定',(1050,8),'#ffba75')],17)
            process.stdin.write(canvas.tobytes())
            if i in (0,20,50,100,155,170,199):cv2.imwrite(str(report/'snapshots'/f'{i:06d}.jpg'),canvas,[cv2.IMWRITE_JPEG_QUALITY,90])
            save_json(report/'rows.json',rows)
            if i%10==0:print('evaluate',directory.name,i,len(source['frames']),f'{time.perf_counter()-elapsed:.1f}s',flush=True)
    finally:
        process.stdin.close();code=process.wait();renderer.close();scene.close()
        if base_reader is not None:base_reader.release()
    if code:raise RuntimeError('ffmpeg failed')
    video.with_suffix('.partial.mp4').replace(video)
    keys=sorted({k for r in rows for k in r if k not in ('root_camera_m','wrist_camera_m')})
    with (report/'frames.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    summary=summarize(rows,methods);save_json(report/'summary.json',dict(name=source['name'],methods=summary,reference_manifest_sha256=file_hash(directory/'reference/manifest.json'),
        point_surface_sample='Uniform deterministic max256 observed masked points per frame, common for methods; exact triangle surface distance; no correspondence filtering',
        reference='SAM3 reference, not anatomical ground truth; CUDA60 fits using this mask; tracking candidate never does',
        evaluation='full-resolution visible surface, equal frame mean over shared valid frames; no extra alignment; failures/validity separately reported; YOLO residual is in-objective diagnostic only',
        timing='offline measured plus cached input costs, excludes export and display; previous tracking run; no live throughput claim'))
    write_page(directory,source['name'],summary)
    save_json(complete,dict(frames=len(source['frames']),rows=len(rows),evaluation_source_sha256=file_hash(Path(__file__)),input_manifest_sha256=file_hash(directory/'manifest.json'),video_sha256=file_hash(video)))
    print('EVALUATED',directory.name,flush=True)


def n(value,digits=2):return '—' if value is None else f'{value:.{digits}f}'


def table(summaries):
    result='<table><tr><th>方案</th><th>共同有效帧</th><th>深度MAE mm</th><th>P90 mm</th><th>点面MAE / P90 mm</th><th>轮廓IoU</th><th>覆盖率</th><th>可见帧核心中位/P95 ms</th><th>静止腕抖动 mm</th><th>有效帧</th><th>各自有效帧深度 / 点面 MAE mm</th></tr>'
    for i,s in enumerate(summaries):
        vals=[METHOD_LABELS[s['method']],s['common_quality_frames'],n(s['mae_mm']),n(s['p90_mm']),n(s['point_surface_mae_mm'])+' / '+n(s['point_surface_p90_mm']),n(s['silhouette_iou'],3),n(s['model_coverage_fraction'],3),n(s['visible_core_ms_median'],1)+' / '+n(s['visible_core_ms_p95'],1),n(s['jitter'].get('wrist_detrended_rms_mm')),s['valid_frames'],n(s.get('own_valid_mae_mm'))+' / '+n(s.get('own_valid_point_surface_mae_mm'))]
        result+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in vals)+'</tr>'
    return result+'</table>'

STYLE='<style>body{font:18px sans-serif;background:#0c1422;color:#e2e8f0;max-width:1600px;margin:30px auto;padding:20px}p,li{line-height:1.8}a{color:#7dbbff}video,img{width:100%}table{width:100%;border-collapse:collapse}th,td{padding:10px;border:1px solid #3d5069}.note{background:#25364d;padding:18px} .warning{background:#493719;padding:18px}</style>'
NOTICE='''<p class="note">前三列：独立网络、独立网络＋CUDA60、周期初始化＋连续15步。若有第四列，则为仅首帧初始化＋连续15步；不再调用网络，失效和出画后不重置，只能尝试从最后接受的状态继续。上排为RGB叠加，下排为固定斜视角人体参考点云叠加。实际相机坐标、米制尺度，不额外对齐。视频按输入时间跨度播放；播放速度不代表处理速度。交互式三维回放在 Studio 的“RGB-D 跟踪对照”页。</p><p>全部为Fast384身体模式。CUDA60保留原完整平移＋刚体60＋姿态60，使用离线SAM3掩膜；连续方案为既有无掩膜候选版，每秒或失败时SAM＋60步，普通帧15步，固定体型。差异不只是迭代次数。所有列共同有效帧用于质量均值，失败帧数另列；同时给出各自有效帧误差，避免第四方案失败导致主表只剩容易帧。CSV保留全部帧，JSON另按每5秒分窗统计漂移。衣服深度及SAM3掩膜不是骨架真值。YOLO重投影残差已参与跟踪拟合，只作诊断。JSON/CSV另含相对CUDA60的身体关键点差异、去根位置后的姿态差异与肘角；这些是算法间一致性，不是关节真值误差。静止抖动取预设0.25–1.75秒并去线性趋势，仍包含自然晃动，不说明动作延迟。</p><p>点到面使用每帧同一组最多256个均匀真实测量点，计算到完整三角面的距离；深度MAE/P90使用全分辨率重叠区域，并附覆盖率及缺失表面按100mm计的截断误差。时间是离线核心及缓存输入成本估计，排除视频、文件导出及GUI，不代表在线帧率；三种方案运行批次不同。</p>'''


def write_page(directory,name,summary):
    suspects=summary[0]['reference_suspect_frames']
    warning='<p class="warning">参考掩膜待核查采样帧：'+str(suspects)+'。保留其统计；JSON另给参考筛查后的MAE。尚不能当作最终验收。</p>' if suspects else ''
    body='<!doctype html><meta charset="utf-8"><title>'+html.escape(name)+'：多方案对照</title>'+STYLE+'<h1>'+html.escape(name)+'：多方案连续对照</h1>'+NOTICE+warning+table(summary)
    body+='<p><a href="frames.csv">逐帧CSV</a> · <a href="summary.json">指标与定义</a> · <a href="../manifest.json">配置与来源</a> · <a href="../../report.html">四组合览</a></p><video controls preload="metadata" src="comparison.mp4"></video>'
    body+='<p>绿色参考轮廓不参与当前显示自动对齐。若人体无有效结果，画面保留点云并明确标记LOST/DEGRADED。</p>'
    (directory/'report/report.html').write_text(body)


def aggregate(root):
    cases=[];body='<!doctype html><meta charset="utf-8"><title>四组RGB-D多方案对照</title>'+STYLE+'<h1>四组 RGB-D：独立重建 / CUDA60 / 连续细化对照</h1>'+NOTICE
    body+='<p><a href="playlist.mp4">四组顺序连续播放视频</a> · <a href="summary.json">汇总JSON</a></p>'
    for case in ('arms','cross','turn','recover'):
        file=root/case/'report/summary.json'
        if not file.exists():continue
        s=json.loads(file.read_text());cases.append(dict(case=case,**s));body+='<h2>'+html.escape(s['name'])+'</h2>'+table(s['methods'])
        if s['methods'][0]['reference_suspect_frames']:body+='<p class="warning">参考待核查：'+str(s['methods'][0]['reference_suspect_frames'])+'。当前为暂定统计。</p>'
        body+=f'<p><a href="{case}/report/report.html">单组详情与CSV</a></p><video controls preload="none" src="{case}/report/comparison.mp4"></video>'
    save_json(root/'summary.json',cases);(root/'report.html').write_text(body)
    from rgbd_comparison_playlist import make_playlist
    if cases:make_playlist(root,[root/s['case']/'report/comparison.mp4' for s in cases])
    print('AGGREGATED',root,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--aggregate',action='store_true');a=p.parse_args();aggregate(a.directory) if a.aggregate else evaluate(a.directory)
