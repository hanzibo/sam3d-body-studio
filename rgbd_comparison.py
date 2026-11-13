# SPDX-License-Identifier: GPL-3.0-only
"""Reproducible three-method comparison of the four frozen tracking recordings."""
import argparse,json,subprocess,shutil
from pathlib import Path
from studio_paths import path,environment
from video_project import file_hash,save_json
from rgbd_tracking_reference import reference_directory

ROOT=Path(__file__).resolve().parent
CASES=('arms','cross','turn','recover')


def prepare_case(source,dest):
    source=source.resolve();ref=reference_directory(source)
    manifest=dict(schema='studio_rgbd_three_method_v1',source=str(source),inputs_sha256=file_hash(source/'inputs.json'),
        tracking_complete_sha256=file_hash(source/'candidate-15/complete.json'),tracking_rows_sha256=file_hash(source/'candidate-15/frames.jsonl'),
        launch_sha256=file_hash(source/'launch.json'),reference_manifest_sha256=file_hash(ref/'manifest.json'),
        method_order=['independent','cuda60','tracking'],
        methods=dict(independent='Fast384 body, independent network each frame, supplied calibrated K and same detector box; original metric translation',
          cuda60='Same independent prediction + established full-resolution translation refinement + rigid60 + pose60 CUDA Graph, same default bounds/guard/backtracking; OFFLINE SAM3 mask assistance; no shape/scale optimization',
          tracking='Existing frozen candidate-15 states, periodic 1s SAM+60 keyframe and 15-step continuation, no SAM3 fitting mask; identity frozen per segment'),
        evaluation='Full-resolution calibrated raw measured depth, common SAM3 reference, no additional alignment; observed surface consistency, not anatomical ground truth')
    manifest['source_sha256']={n:file_hash(ROOT/n) for n in ('rgbd_comparison.py','rgbd_comparison_worker.py','rgbd_comparison_data.py','depth_pose_accelerated.py','depth_pose_refinement.py','depth_refinement.py','fast_worker.py')}
    if (dest/'manifest.json').exists():
        if json.loads((dest/'manifest.json').read_text())!=manifest:raise ValueError('resume source/config mismatch; use a new output directory')
        return
    dest.mkdir(parents=True,exist_ok=False);shutil.copytree(ref,dest/'reference');shutil.copy2(source/'inputs.json',dest/'inputs.json');shutil.copy2(source/'launch.json',dest/'launch.json')
    save_json(dest/'manifest.json',manifest)


def run(args):
    output=Path(args.output).resolve();source=Path(args.source).resolve();output.mkdir(parents=True,exist_ok=True)
    for case in args.cases:prepare_case(source/case,output/case)
    if args.stage in ('all','infer'):
        configs=[json.loads((output/c/'launch.json').read_text()) for c in args.cases]
        if any(c!=configs[0] for c in configs):raise ValueError('mixed launch profiles')
        env=environment('fast');env.update(BODYSTUDIO_GPU_KIND='exclusive',BODYSTUDIO_CONFIG=str(output/args.cases[0]/'launch.json'))
        commands=[dict(type='rgbd_compare',directory=str(source/c),output=str(output/c),config_id=__import__('launch_config').LaunchConfig.from_dict(configs[0]).config_id) for c in args.cases if not (output/c/'complete.json').exists()]
        if commands:
            with (output/'inference.log').open('a') as log:
                result=subprocess.run([str(path('fast_python')),str(ROOT/'fast_worker.py')],input=''.join(json.dumps(c)+'\n' for c in commands+[dict(type='stop')]),text=True,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=ROOT)
            if result.returncode:raise RuntimeError('worker failed: '+str(output/'inference.log'))
        for c in args.cases:
            if not (output/c/'complete.json').exists():raise RuntimeError('incomplete '+c+'; inspect inference.log')
    if args.stage in ('all','evaluate'):
        for c in args.cases:
            subprocess.run([str(path('standard_python')),str(ROOT/'rgbd_comparison_report.py'),str(output/c)],env=environment('standard'),cwd=ROOT,check=True)
        subprocess.run([str(path('standard_python')),str(ROOT/'rgbd_comparison_report.py'),str(output),'--aggregate'],env=environment('standard'),cwd=ROOT,check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',default=str(ROOT/'outputs/depth-fusion/tracking-v1'));p.add_argument('--output',default=str(ROOT/'outputs/depth-fusion/tracking-comparison-v1'))
    p.add_argument('--stage',choices=('all','infer','evaluate'),default='all');p.add_argument('--cases',nargs='+',choices=CASES,default=list(CASES));run(p.parse_args())

if __name__=='__main__':main()
