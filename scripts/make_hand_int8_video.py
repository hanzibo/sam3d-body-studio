# SPDX-License-Identifier: GPL-3.0-only
"""Side-by-side FP16/INT8 overlays, preserving source videos and timing scope."""
import argparse,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
plan=[]
for d in sorted(a.candidate.iterdir()):
 if not (d/'overlay.mp4').exists():continue
 old=a.baseline/d.name/'overlay.mp4';new=d/'overlay.mp4';out=a.output/(d.name+'-fp16-vs-int8.mp4')
 def probe(file):return json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,nb_frames,r_frame_rate','-of','json',str(file)]))['streams'][0]
 x,y=probe(old),probe(new);assert x==y and int(x['nb_frames'])==450,(x,y)
 filt="[0:v]pad=iw:ih+40:0:40:black,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='FP16 512':x=12:y=8:fontsize=24:fontcolor=white[l];[1:v]pad=iw:ih+40:0:40:black,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='INT8-FP16 512 - experimental':x=12:y=8:fontsize=24:fontcolor=white[r];[l][r]hstack=inputs=2[v]"
 argv=['ffmpeg','-hide_banner','-loglevel','error','-n','-threads','2','-i',str(old),'-threads','2','-i',str(new),'-filter_complex_threads','2','-filter_complex',filt,'-map','[v]','-an','-c:v','libx264','-threads','4','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(out)]
 plan.append(dict(baseline=str(old),candidate=str(new),output=str(out),layout='left FP16, right INT8; identical 30 FPS source playback, not measured inference FPS',argv=argv))
 (a.output/'edit-plan.json').write_text(json.dumps(plan,indent=2));subprocess.run(argv,check=True)
 assert probe(out)['nb_frames']=='450';subprocess.run(['ffmpeg','-v','error','-threads','2','-i',str(out),'-f','null','-'],check=True)
 for frame in (150,330):
  subprocess.run(['ffmpeg','-v','error','-n','-ss',str(frame/30),'-i',str(out),'-frames:v','1','-threads','2',str(a.output/(d.name+f'-{frame:06d}.png'))],check=True)
 print('VIDEO_OK',out,flush=True)
