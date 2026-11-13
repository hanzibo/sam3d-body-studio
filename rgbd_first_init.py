# SPDX-License-Identifier: GPL-3.0-only
"""Add a strict first-frame-only initialization ablation to frozen comparisons."""
import argparse,json,os,shutil,subprocess
from pathlib import Path
from studio_paths import path,environment
from video_project import file_hash,save_json
from rgbd_comparison_data import ALL_METHODS,load_npz,atomic_npz
import numpy as np

ROOT=Path(__file__).resolve().parent
CASES=('arms','cross','turn','recover')

def link(source,dest):
    # Immutable inputs only. Never hard-link writable outputs or manifests.
    try:os.link(source,dest)
    except OSError:shutil.copy2(source,dest)

def prepare(base,dest):
    previous=json.loads((base/'manifest.json').read_text());original=Path(previous['source'])
    frozen=dict(schema='studio_rgbd_four_method_v1',source=str(original),base_comparison=str(base),
        base_complete_sha256=file_hash(base/'complete.json'),base_manifest_sha256=file_hash(base/'manifest.json'),
        base_rows_sha256=file_hash(base/'report/rows.json'),base_video_sha256=file_hash(base/'report/comparison.mp4'),
        inputs_sha256=file_hash(base/'inputs.json'),method_order=list(ALL_METHODS),
        first_init='same mask-free first60, then15 steps only; no periodic/failure/return network reinitialization; retain last accepted state across missing detections; current-frame YOLO still used',
        source_sha256={n:file_hash(ROOT/n) for n in ('rgbd_first_init.py','rgbd_tracking_worker.py','rgbd_tracking_points.py','rgbd_comparison_data.py','depth_pose_accelerated.py','fast_worker.py')})
    manifest=dest/'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text())!=frozen:raise ValueError('changed inputs/code: use a new --output')
        return
    dest.mkdir(parents=True,exist_ok=False)
    for name in ('inputs.json','launch.json','topology.npz'):shutil.copy2(base/name,dest/name)
    shutil.copytree(base/'reference',dest/'reference',copy_function=link)
    run=dest/'first_init_run';run.mkdir()
    for name in ('inputs.json','launch.json'):shutil.copy2(base/name,run/name)
    shutil.copytree(original/'frames',run/'frames',copy_function=link)
    (run/'network_keyframes').mkdir()
    # Cache just the very first usable network prediction; further calls are
    # forbidden by the worker policy, not merely avoided by a full-frame cache.
    inputs=json.loads((base/'inputs.json').read_text())
    first=next(i for i,r in enumerate(inputs['frames']) if r['bbox'] is not None)
    for ext in ('.npz','.json'):
        shutil.copy2(original/'network_keyframes'/f'{first:06d}{ext}',run/'network_keyframes'/f'{first:06d}{ext}')
    save_json(manifest,frozen)

def track(output,cases):
    from launch_config import LaunchConfig
    commands=[]
    for case in cases:
        directory=output/case/'first_init_run'
        if (directory/'first-init-15/complete.json').exists():continue
        config=LaunchConfig.from_dict(json.loads((directory/'launch.json').read_text()))
        commands.append(dict(type='rgbd_track',job='first-init-'+case,config_id=config.config_id,directory=str(directory),mode='candidate',steps=15,policy='first_init'))
    if not commands:return
    env=environment('fast');env.update(BODYSTUDIO_GPU_KIND='exclusive',BODYSTUDIO_CONFIG=str(output/cases[0]/'launch.json'))
    with (output/'first-init-inference.log').open('a') as log:
        code=subprocess.run([str(path('fast_python')),str(ROOT/'fast_worker.py')],input=''.join(json.dumps(c)+'\n' for c in commands+[dict(type='stop')]),text=True,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=ROOT).returncode
    if code:raise RuntimeError('first-init worker exited; inspect first-init-inference.log')
    for case in cases:
        if not (output/case/'first_init_run/first-init-15/complete.json').exists():raise RuntimeError('incomplete '+case+'; inspect first-init-inference.log')

def merge(base,dest):
    if (dest/'complete.json').exists():return
    run=dest/'first_init_run/first-init-15';complete=json.loads((run/'complete.json').read_text())
    source=json.loads((base/'inputs.json').read_text());rows=[json.loads(s) for s in (run/'frames.jsonl').read_text().splitlines()]
    assert len(rows)==len(source['frames'])
    keys=[r for r in rows if r['keyframe']]
    if len(keys)!=1 or keys[0]['frame']!=0:raise ValueError('strict first-init policy violated')
    (dest/'frames').mkdir(exist_ok=True)
    for i,(entry,row) in enumerate(zip(source['frames'],rows)):
        original=base/'frames'/f'{i:06d}.npz';info=json.loads(original.with_suffix('.json').read_text())
        if file_hash(original)!=info['artifact_sha256']:raise ValueError('baseline artifact changed')
        old=load_npz(original);a={}
        for name in ('vertices','joints','keypoints'):a[name]=np.concatenate((old[name],np.zeros_like(old[name][:1])),axis=0)
        a['valid']=np.r_[old['valid'],bool(row['valid'])]
        if row['valid']:
            actual=load_npz(run/'meshes'/f'{i:06d}.npz')
            for name in ('vertices','joints','keypoints'):a[name][3]=actual[name]
        core=sum(row.get('stages_ms',{}).get(k,0) for k in ('keyframe_setup_ms','fit_ms','accept_ms'))+row['detector_ms']+(row.get('network_ms',0) if row.get('network_cached') else 0)
        info['methods'].append(dict(method='first_init',valid=bool(row['valid']),status=row['state'],keyframe=row['keyframe'],frame=i,source_index=entry['source_index'],time_s=entry['time_s'],core_ms=core,estimated_input_ms=row['estimated_observed_input_ms'],network_ms=row.get('network_ms',0),initialization_policy='first_frame_only'))
        file=dest/'frames'/f'{i:06d}.npz';atomic_npz(file,**a)
        info.update(base_artifact_sha256=file_hash(original),artifact_sha256=file_hash(file));save_json(file.with_suffix('.json'),info)
    save_json(dest/'complete.json',dict(frames=len(rows),methods=list(ALL_METHODS),first_init_complete_sha256=file_hash(run/'complete.json'),cold_graph_setup_ms=complete['cold_graph_setup_ms'],manifest_sha256=file_hash(dest/'manifest.json'),base='first three columns copied exactly; historical measured timing retained'))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=ROOT/'outputs/depth-fusion/tracking-comparison-v1')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/depth-fusion/tracking-comparison-four-v1')
    parser.add_argument('--stage',choices=('all','infer','evaluate'),default='all')
    args=parser.parse_args();args.base=args.base.resolve();args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
    for case in CASES:prepare(args.base/case,args.output/case)
    if args.stage in ('all','infer'):
        track(args.output,CASES)
        for case in CASES:merge(args.base/case,args.output/case)
    if args.stage in ('all','evaluate'):
        for case in CASES:subprocess.run([str(path('standard_python')),str(ROOT/'rgbd_comparison_report.py'),str(args.output/case)],env=environment('standard'),cwd=ROOT,check=True)
        subprocess.run([str(path('standard_python')),str(ROOT/'rgbd_comparison_report.py'),str(args.output),'--aggregate'],env=environment('standard'),cwd=ROOT,check=True)

if __name__=='__main__':main()
