# SPDX-License-Identifier: GPL-3.0-only
"""Opt-in audit of real network/mesh/replay/video outputs; does not run inference."""
import json,subprocess,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rgbd_comparison_data import load_npz,METHODS,comparison_methods
from video_project import file_hash


def audit(root):
    for case in ('arms','cross','turn','recover'):
        directory=root/case;source=json.loads((directory/'inputs.json').read_text());manifest=json.loads((directory/'manifest.json').read_text());original=Path(manifest['source'])
        methods=comparison_methods(directory);count=len(methods)
        frames=source['frames'];assert len(frames)==200
        rows=json.loads((directory/'report/rows.json').read_text());assert len(rows)==len(frames)*count
        faces=load_npz(directory/'topology.npz')['faces'];assert faces.shape==(36874,3)
        for i,entry in enumerate(frames):
            f=directory/'frames'/f'{i:06d}.npz';info=json.loads(f.with_suffix('.json').read_text());assert file_hash(f)==info['artifact_sha256']
            a=load_npz(f);assert a['vertices'].shape==(count,18439,3) and np.isfinite(a['vertices']).all()
            assert a['joints'].shape==(count,127,3) and a['keypoints'].shape==(count,70,3)
            assert [r['method'] for r in info['methods']]==list(methods)
            if count==4:
                base=load_npz(Path(manifest['base_comparison'])/'frames'/f'{i:06d}.npz')
                for name in ('vertices','joints','keypoints','valid'):np.testing.assert_array_equal(a[name][:3],base[name])
                tr=directory/'first_init_run/first-init-15/states'/f'{i:06d}.npz'
                assert bool(a['valid'][3])==(bool(load_npz(tr)['valid']) if tr.exists() else False)
            if entry['bbox'] is not None and count==3:
                net=load_npz(directory/'network'/f'{i:06d}.npz')
                np.testing.assert_allclose(a['vertices'][0],net['pred_vertices']+net['pred_cam_t'],atol=1e-7,rtol=0)
                assert bool(a['valid'][0]) and bool(a['valid'][1])
            elif entry['bbox'] is None:assert not a['valid'].any()
            st=original/'candidate-15/states'/f'{i:06d}.npz'
            if st.exists():assert bool(load_npz(st)['valid'])==bool(a['valid'][2])
            cloud=load_npz(directory/'report/replay'/f'{i:06d}.npz')
            assert cloud['cloud'].shape==(len(cloud['colors']),3) and len(cloud['person'])==len(cloud['cloud'])
            assert len(cloud['cloud'])<=24000 and np.isfinite(cloud['cloud']).all()
            for row in rows[count*i:count*i+count]:assert row['frame']==i and row['source_index']==entry['source_index'] and row['time_s']==entry['time_s']
        video=directory/'report/comparison.mp4'
        stream=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames,width,height','-of','json',str(video)]))['streams'][0]
        assert int(stream['nb_frames'])==200 and stream['width']==640*count and stream['height']==840
        if count==4:
            run=directory/'first_init_run';timings=[json.loads(s) for s in (run/'first-init-15/frames.jsonl').read_text().splitlines()]
            assert [r['frame'] for r in timings if r['keyframe']]==[0]
            assert all(r.get('network_ms',0)==0 for r in timings[1:])
            assert len(list((run/'network_keyframes').glob('*.npz')))==1
        print(case,'REAL_COMPARISON_ARTIFACTS_OK',len(rows),'rows',flush=True)
    stream=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames','-of','json',str(root/'playlist.mp4')]))['streams'][0]
    assert int(stream['nb_frames'])==800
    playlist=json.loads((root/'playlist.json').read_text());assert abs(playlist['actual_duration_s']-playlist['expected_duration_s'])<.2
    assert playlist['frames']==800 and file_hash(root/'playlist.mp4')==playlist['sha256']
    timestamps=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','frame=best_effort_timestamp_time','-of','json',str(root/'playlist.mp4')]))['frames']
    pts=np.array([float(f['best_effort_timestamp_time']) for f in timestamps]);assert len(pts)==800 and np.all(np.diff(pts)>0) and np.max(np.diff(pts))<.2
    print('FOUR_CASE_PLAYLIST_OK',flush=True)

if __name__=='__main__':audit(Path(sys.argv[1]))
