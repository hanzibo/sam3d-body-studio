# SPDX-License-Identifier: GPL-3.0-only
"""Post-hoc SAM3 membership audit. Kept out of candidate inference worker."""
import csv
import html
import json
from pathlib import Path
import subprocess
import cv2
import numpy as np
from studio.rgbd.rgbd_tracking_points import point_membership
from studio.video.video_project import save_json,file_hash


def membership_metrics(uv,mask):
    uv=np.asarray(uv).reshape(-1,2)
    if not len(uv):return dict(count=0,inside_fraction=None,boundary_fraction=None,outside_beyond_3px_fraction=None)
    kernel=np.ones((7,7),np.uint8)
    inner=cv2.erode(mask.astype(np.uint8),kernel)>0;outer=cv2.dilate(mask.astype(np.uint8),kernel)>0
    exact=point_membership(uv,mask);interior=point_membership(uv,inner);expanded=point_membership(uv,outer)
    return dict(count=len(uv),inside_fraction=float(exact.mean()),boundary_fraction=float((expanded&~interior).mean()),
                outside_beyond_3px_fraction=float((~expanded).mean()))


def draw_points(image,uv,mask):
    inside=point_membership(uv,mask)
    for p,good in zip(uv,inside):cv2.circle(image,tuple(np.rint(p).astype(int)),2,(80,230,80) if good else (30,30,250),-1)
    contour,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image,contour,-1,(255,220,0),1)
    return image


def evaluate(directory,steps):
    from studio.rgbd.rgbd_tracking_reference import reference_directory
    reference_folder=reference_directory(directory)
    source=json.loads((directory/'inputs.json').read_text());reference=json.loads((reference_folder/'manifest.json').read_text())
    modes=['candidate','oracle'];results={};summaries=[];all_rows=[]
    suffix='' if reference_folder.name=='reference_masks' else '-'+file_hash(reference_folder/'manifest.json')[:10]
    dest=directory/f'comparison-{steps}{suffix}';dest.mkdir(exist_ok=False);(dest/'overlays').mkdir()
    for mode in modes:
        folder=directory/f'{mode}-{steps}'
        complete=json.loads((folder/'complete.json').read_text())
        if complete['inputs_sha256']!=file_hash(directory/'inputs.json'):raise ValueError('input changed')
        frames=[json.loads(line) for line in (folder/'frames.jsonl').read_text().splitlines()]
        if len(frames)!=len(source['frames']) or len(frames)!=complete['frames']:raise ValueError('incomplete frame index')
        results[mode]=[]
        for i,row in enumerate(frames):
            if row['frame']!=i:raise ValueError('misordered frame')
            maskfile=reference_folder/f'{i:06d}.png'
            if file_hash(maskfile)!=reference['frames'][i]['sha256']:raise ValueError('reference changed')
            mask=cv2.imread(str(maskfile),0)>0
            with np.load(folder/'points'/f'{i:06d}.npz',allow_pickle=False) as f:
                uv=f['pixel_uv'];batch=f['batch'];candidate=f['candidate_uv']
                # All actual optimizer batches, plus per-frame unique measured pixels.
                unique=np.unique(np.rint(uv).astype(int),axis=0)
                last=uv[batch==batch.max()] if len(batch) else uv
                metrics=membership_metrics(uv,mask);unique_metrics=membership_metrics(unique,mask)
                last_metrics=membership_metrics(last,mask)
                candidate_metrics=membership_metrics(candidate,mask)
            current_keypoints=source['frames'][i].get('keypoints')
            reference_keypoint_fraction=None;reference_quality='unverified'
            if current_keypoints is not None:
                kp=np.asarray(current_keypoints)[[0,5,6,7,8,11,12,13,14]]
                kp=kp[(kp[:,2]>.7)&(kp[:,0]>=0)&(kp[:,0]<mask.shape[1])&(kp[:,1]>=0)&(kp[:,1]<mask.shape[0])]
                if len(kp)>=3:
                    expanded=cv2.dilate(mask.astype(np.uint8),np.ones((7,7),np.uint8))>0
                    reference_keypoint_fraction=float(point_membership(kp[:,:2],expanded).mean())
                    reference_quality='needs_review' if reference_keypoint_fraction<.75 else 'keypoints_consistent_not_ground_truth'
            record=dict(reference_keypoint_fraction=reference_keypoint_fraction,reference_quality=reference_quality,mode=mode,frame=i,time_s=row['time_s'],state=row['state'],valid=row['valid'],keyframe=row['keyframe'],
                        reference_segmented=reference['frames'][i].get('segmented',reference['frames'][i]['source']!='no unambiguous YOLO person; NOT independently SAM3-tested'),
                        reference_pixels=int(mask.sum()),used_pairs=metrics['count'],unique_pixels=unique_metrics['count'],
                        point_inside_fraction=metrics['inside_fraction'],unique_inside_fraction=unique_metrics['inside_fraction'],
                        last_inside_fraction=last_metrics['inside_fraction'],outside_beyond_3px_fraction=metrics['outside_beyond_3px_fraction'],
                        boundary_fraction=metrics['boundary_fraction'],candidate_projection_inside_fraction=candidate_metrics['inside_fraction'],
                        fit_ms=row.get('stages_ms',{}).get('fit_ms'),estimated_input_ms=row['estimated_observed_input_ms'],
                        keypoint_residual_px=row.get('after_keypoints_px'),fit_point_distance_mm=row.get('after_depth_mm'))
            all_rows.append(record);results[mode].append((record,last))
        default_reference=directory/'reference_masks/manifest.json'
        fit_reference_hash=complete.get('reference_manifest_sha256') or (file_hash(default_reference) if default_reference.exists() else None)
        stats=[x[0] for x in results[mode] if x[0]['used_pairs'] and x[0]['reference_segmented']]
        total=sum(x['used_pairs'] for x in stats)
        nonkey=[x for x in stats if not x['keyframe']]
        nonkey_pairs=sum(x['used_pairs'] for x in nonkey)
        summary=dict(mode=mode,frames=len(frames),frames_with_points=len(stats),valid_frames=sum(r['valid'] for r in frames),lost_frames=sum(r['state']=='LOST' for r in frames),
                     reference_suspect_frames=[x['frame'] for x in stats if x['reference_quality']=='needs_review'],
                     total_pairs=total,tracking_only_pairs=nonkey_pairs,tracking_only_inside_fraction=sum(x['point_inside_fraction']*x['used_pairs'] for x in nonkey)/nonkey_pairs if nonkey_pairs else None,
                     tracking_only_worst_frame_inside_fraction=min([x['point_inside_fraction'] for x in nonkey],default=None),
                     keyframes=sum(r['keyframe'] for r in frames),
                     total_estimated_median_ms=float(np.median([r['estimated_observed_input_ms'] for r in frames])),
                     total_estimated_p95_ms=float(np.percentile([r['estimated_observed_input_ms'] for r in frames],95)),
                     weighted_inside_fraction=sum(x['point_inside_fraction']*x['used_pairs'] for x in stats)/total if total else None,
                     mean_frame_inside_fraction=float(np.mean([x['point_inside_fraction'] for x in stats])) if stats else None,
                     worst_frame_inside_fraction=min([x['point_inside_fraction'] for x in stats],default=None),
                     weighted_outside_beyond_3px=sum(x['outside_beyond_3px_fraction']*x['used_pairs'] for x in stats)/total if total else None,
                     median_tracking_fit_ms=float(np.median([x['fit_ms'] for x in nonkey])) if nonkey else None,
                     median_tracking_estimated_input_ms=float(np.median([x['estimated_input_ms'] for x in nonkey])) if nonkey else None,
                     fit_reference_matches_evaluation=(mode=='candidate' or fit_reference_hash==file_hash(reference_folder/'manifest.json')),
                     timing_note='offline replay, cached YOLO measured cost included; not measured live-camera throughput')
        summaries.append(summary)
    save_json(dest/'summary.json',dict(name=source['name'],modes=summaries,definition='SAM3 consistency, not anatomical accuracy. Used points include rejected fits; candidates are model projections, not observations.',
                                     reference_note=reference.get('source',source['prompt_source']),reference_manifest=str(reference_folder/'manifest.json'),reference_manifest_sha256=file_hash(reference_folder/'manifest.json'),rows=all_rows))
    with (dest/'frames.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
    # One full-length, timestamp-sampled comparison video; points in final optimizer batch.
    w,h=source['k']['width'],source['k']['height'];vw,vh=w,h//2+56
    process=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{vw}x{vh}',
        '-r',str(source['hz']),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','22','-pix_fmt','yuv420p',str(dest/'point_comparison.mp4')],stdin=subprocess.PIPE)
    # Ensure even dimensions for H.264.
    chosen=set(range(5,len(source['frames']),10))|{0,len(source['frames'])-1}
    ranked=sorted(results['candidate'],key=lambda x:x[0]['point_inside_fraction'] if x[0]['point_inside_fraction'] is not None else 2)
    chosen.update(x[0]['frame'] for x in ranked[:10])
    cards=[]
    try:
        for i in range(len(source['frames'])):
            image=cv2.imread(str(directory/'frames'/f'{i:06d}.png'));mask=cv2.imread(str(reference_folder/f'{i:06d}.png'),0)>0
            panels=[]
            for mode in modes:
                record,uv=results[mode][i];panel=draw_points(image.copy(),uv,mask)
                panel=cv2.resize(panel,(w//2,h//2));panel=cv2.copyMakeBorder(panel,56,0,0,0,cv2.BORDER_CONSTANT,value=(20,25,35))
                fraction=record['last_inside_fraction'];text=f'{mode} {record["state"]} points={len(uv)} inside={fraction:.1%}' if fraction is not None else f'{mode} {record["state"]} NO POINTS'
                cv2.putText(panel,text,(8,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(240,240,240),1,cv2.LINE_AA)
                cv2.putText(panel,f't={record["time_s"]:.2f}s frame={i} green=in SAM3 / red=outside',(8,43),cv2.FONT_HERSHEY_SIMPLEX,.45,(220,220,220),1,cv2.LINE_AA)
                panels.append(panel)
            joined=np.hstack(panels);process.stdin.write(joined.tobytes())
            if i in chosen:
                cv2.imwrite(str(dest/'overlays'/f'{i:06d}.jpg'),joined,[cv2.IMWRITE_JPEG_QUALITY,90]);cards.append((i,source['frames'][i]['time_s']))
    finally:
        process.stdin.close();code=process.wait()
    if code:raise RuntimeError('video encode failed')
    reference_warning='' if summaries[1]['fit_reference_matches_evaluation'] else '<p class="warning">当前参考掩膜已修订：右侧 oracle 是在旧掩膜上拟合的固定结果，本报告仅重新检查其落点，不能作为使用新掩膜重新拟合的对照。</p>'
    table=''.join('<tr>'+''.join('<td>'+html.escape(str(s.get(k)))+'</td>' for k in ('mode','frames_with_points','weighted_inside_fraction','worst_frame_inside_fraction','weighted_outside_beyond_3px','median_tracking_fit_ms'))+'</tr>' for s in summaries)
    images=''.join(f'<figure><img loading="lazy" src="overlays/{i:06d}.jpg"><figcaption>帧 {i} / {t:.2f}s</figcaption></figure>' for i,t in cards)
    document='''<!doctype html><meta charset="utf-8"><title>RGB-D 跟踪选点对照</title><style>body{background:#0c1422;color:#e8edf5;font:17px sans-serif;max-width:1400px;margin:32px auto;padding:20px}a{color:#70baff}video,img{width:100%}table{border-collapse:collapse}td,th{border:1px solid #526277;padding:12px}figure{margin:25px 0}p{line-height:1.7} .warning{background:#493719;padding:16px}</style>'''
    document+=f'<h1>{html.escape(source["name"])}：实际优化点与 SAM3 对照</h1><p>左侧：无 SAM3 的候选版；右侧：SAM3 限制对应点的离线版。绿点位于参考人体掩膜内，红点位于掩膜外，青色为掩膜轮廓。视频显示每帧最后一批实际优化点；CSV 统计该帧所有批次，包含未接受的优化尝试。</p>'
    document+='<p class="warning">SAM3 使用前景/背景点沿时间传播，参考来源见 JSON；自动提示结果需肉眼核查，不是人工真值。两组初始化/关键帧都使用相同无掩膜 60 步拟合；只有非关键帧改变选点方式。点在人体内不代表覆盖完整，也不证明姿态正确。离线耗时包含导出及缓存成本估计，不能视为在线帧率。</p>'
    document+=reference_warning
    suspects=summaries[0]['reference_suspect_frames']
    if suspects:document+='<p class="warning">参考掩膜在部分帧未覆盖高置信度身体关键点，建议手动核查这些采样帧：'+html.escape(str(suspects))+ '。此检查只用于提示参考质量；未据此改变候选选点或删除低分统计。</p>'
    document+='<table><tr><th>版本</th><th>有点帧</th><th>按对应点加权人体内比例</th><th>最差帧</th><th>距掩膜&gt;3px外点比例</th><th>普通帧拟合中位ms</th></tr>'+table+'</table>'
    document+='<p><a href="frames.csv">逐帧 CSV</a> · <a href="summary.json">完整指标</a> · <a href="../inputs.json">时间戳/源帧/标定索引</a>。原始逐批点坐标在 ../candidate-15/points/ 与 ../oracle-15/points/，NPZ 中 pixel_uv 为去畸变 RGB 像素坐标。</p><video controls preload="metadata" src="point_comparison.mp4"></video>'+images
    (dest/'report.html').write_text(document)
    print(json.dumps(summaries,ensure_ascii=False,indent=2),flush=True)
    return summaries
