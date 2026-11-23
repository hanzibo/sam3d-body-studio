# SPDX-License-Identifier: GPL-3.0-only
"""Build the optional WiLoR hand detector engine into a separate Data directory."""
import os,sys,hashlib,json,shutil,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from studio_paths import path,environment
from studio.core.gpu_lease import acquire_gpu_lease
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
os.environ.update(environment('fast'));os.environ['YOLO_AUTOINSTALL']='false'
lease=acquire_gpu_lease('exclusive')
from studio.hand.hand_automatic import DETECTOR_SHA256
source=path('hand_detector_checkpoint')
with source.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
if digest!=DETECTOR_SHA256:raise ValueError('Unexpected hand detector checkpoint')
a.output.mkdir(parents=True,exist_ok=True);checkpoint=a.output/'detector_fp16.pt'
if (a.output/'detector_fp16.engine').exists():raise FileExistsError('Refusing to overwrite existing engine')
if not checkpoint.exists():shutil.copyfile(source,checkpoint)
with checkpoint.open('rb') as f:
 if hashlib.file_digest(f,'sha256').hexdigest()!=digest:raise ValueError('Export copy hash mismatch')
from ultralytics import YOLO
model=YOLO(str(checkpoint))
engine=model.export(format='engine',imgsz=640,batch=1,dynamic=False,half=True,device=0,workspace=1,simplify=True,nms=False)
from engine_registry import verify_engine
# Registration covers only this explicitly requested local build.
registry=path('engine_registry');entries=json.loads(registry.read_text()) if registry.is_file() else {}
engine=Path(engine)
with engine.open('rb') as f:engine_digest=hashlib.file_digest(f,'sha256').hexdigest()
entries[str(engine.resolve())]=dict(sha256=engine_digest,size=engine.stat().st_size)
registry.parent.mkdir(parents=True,exist_ok=True);tmp=registry.with_suffix('.tmp');tmp.write_text(json.dumps(entries,indent=2)+'\n');tmp.replace(registry)
(a.output/'build.json').write_text(json.dumps(dict(source=str(source),source_sha256=digest,engine=str(engine),engine_sha256=engine_digest,input=[1,3,640,640],precision='FP16',role='WiLoR hand detector; SAM reconstruction remains independent'),indent=2))
print('HAND_DETECTOR_BUILD_DONE',engine,flush=True)
