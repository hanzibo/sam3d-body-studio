# SPDX-License-Identifier: GPL-3.0-only
"""Concatenate per-case videos with independently derived rational time bases."""
import argparse,hashlib,json,subprocess
from pathlib import Path

def digest(file):return hashlib.sha256(Path(file).read_bytes()).hexdigest()

def probe(file):
    return json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames,width,height:format=duration','-of','json',str(file)]))

def make_playlist(root,videos):
    root=Path(root);videos=list(map(Path,videos));inputs=[];args=['ffmpeg','-hide_banner','-loglevel','error','-y','-filter_complex_threads','2']
    for file in videos:
        data=probe(file);inputs.append(dict(path=str(file.resolve()),sha256=digest(file),duration_s=float(data['format']['duration']),frames=int(data['streams'][0]['nb_frames'])));args+=['-threads','2','-i',str(file)]
    if not videos:raise ValueError('no comparison videos')
    # concat demuxer + stream copy is invalid when the independently measured
    # per-case frame rates produce different MP4 time bases. Normalize PTS in
    # a filter graph, retaining source-time playback and every sampled frame.
    filters=[f'[{i}:v]settb=AVTB,setpts=PTS-STARTPTS[v{i}]' for i in range(len(videos))]
    filters.append(''.join(f'[v{i}]' for i in range(len(videos)))+f'concat=n={len(videos)}:v=1:a=0[out]')
    partial=root/'playlist.partial.mp4';target=root/'playlist.mp4'
    args+=['-filter_complex',';'.join(filters),'-map','[out]','-an','-fps_mode','vfr','-enc_time_base','1:90000','-c:v','libx264','-threads','4','-preset','fast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(partial)]
    subprocess.run(args,check=True)
    actual=probe(partial);expected_duration=sum(v['duration_s'] for v in inputs);expected_frames=sum(v['frames'] for v in inputs)
    if int(actual['streams'][0]['nb_frames'])!=expected_frames or abs(float(actual['format']['duration'])-expected_duration)>.2:raise ValueError('playlist timestamp/frame-count validation failed')
    partial.replace(target)
    (root/'playlist.json').write_text(json.dumps(dict(inputs=inputs,expected_duration_s=expected_duration,actual_duration_s=float(actual['format']['duration']),frames=expected_frames,sha256=digest(target),generator_sha256=digest(__file__)),ensure_ascii=False,indent=2))
    print('PLAYLIST_TIME_VERIFIED',expected_frames,actual['format']['duration'],flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    make_playlist(args.root,[args.root/c/'report/comparison.mp4' for c in ('arms','cross','turn','recover')])
