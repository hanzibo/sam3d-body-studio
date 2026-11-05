"""Portable pose-set manifest and comparable-result lookup; no GUI or GPU imports."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

def digest(path):
    path=Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None

def save_json(path,data):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2));tmp.replace(path)

class PoseSet:
    def __init__(self,directory):
        self.directory=Path(directory).resolve();self.path=self.directory/'pose_set.json'
        self._fingerprints={}
        self.data=json.loads(self.path.read_text())
        if self.data.get('schema')!='studio_pose_set_v1':raise ValueError('不是 Studio 姿态样本集')
        if not 1<=len(self.poses)<=100:raise ValueError('姿态数量必须在 1–100 之间')
        for p in self.poses:
            if p.get('sample'):self.sample_path(p)
        # A crashed queue is resumable, never silently marked successful.
        for run in self.data.get('runs',[]):
            if run.get('status')=='running':run['status']='interrupted'
    @classmethod
    def create(cls,directory,count):
        if not 1<=count<=100:raise ValueError('姿态数量必须在 1–100 之间')
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
        save_json(directory/'pose_set.json',dict(schema='studio_pose_set_v1',poses=[dict(id=i+1,name=f'姿态{i+1}',sample=None) for i in range(count)],runs=[]))
        return cls(directory)
    @property
    def poses(self):return self.data['poses']
    def save(self):save_json(self.path,self.data)
    def sample_path(self,pose):
        if not pose.get('sample'):return None
        path=(self.directory/pose['sample']).resolve()
        if not path.is_relative_to(self.directory):raise ValueError('样本路径不能位于样本集之外')
        return path
    def attach(self,index,sample):
        sample=Path(sample).resolve()
        if not (sample/'camera.json').is_file() or not (sample/'rgbd.npz').is_file():raise ValueError('样本采集不完整')
        self.poses[index]['sample']=str(sample.relative_to(self.directory));self.save()
    def fingerprint(self,index):
        path=self.sample_path(self.poses[index])
        if path is None:return None
        # Include actual geometry and calibration, not only the RGB image.
        names=('rgb.png','rgbd.npz','camera.json','mask.png');signature=[]
        for name in names:
            file=path/name;stat=file.stat() if file.is_file() else None
            signature.append((stat.st_mtime_ns,stat.st_ctime_ns,stat.st_size) if stat else None)
        cached=self._fingerprints.get(path)
        if cached and cached[0]==signature:return cached[1].copy()
        value={name:digest(path/name) for name in names};self._fingerprints[path]=(signature,value)
        return value.copy()
    def results(self,index):
        path=self.sample_path(self.poses[index])
        if path is None or not (path/'results.json').is_file():return []
        data=json.loads((path/'results.json').read_text())
        fingerprint=self.fingerprint(index)
        return [r for r in data if r.get('batch_input')==fingerprint]
    def latest(self,index,config_id,selection):
        path=self.sample_path(self.poses[index])
        for r in reversed(self.results(index)):
            if r.get('batch_config_id')!=config_id or r.get('input_selection')!=selection:continue
            prediction=path/'predictions'/Path(r['prediction']).parent.name/'person_00.npz'
            if not prediction.is_file():continue
            mask=self.fingerprint(index)['mask.png']
            if mask and (r.get('metrics',{}).get('mask_sha256')!=mask or not r.get('evaluation')):continue
            return r
        return None
    def export_csv(self,directory):
        directory=Path(directory);details=[]
        keys={(r['config_id'],r['selection']) for r in self.data.get('runs',[])}
        for i,p in enumerate(self.poses):
            for key in sorted(keys):
                r=self.latest(i,*key);m=r.get('metrics',{}) if r else {};r=r or {}
                details.append(dict(pose=p['name'],config_id=key[0],selection=key[1],status='complete' if r else 'missing_failed_or_input_changed',inference_ms=r.get('inference_ms'),
                    depth_mae_mm=m.get('mae_mm'),point_surface_mae_mm=m.get('point_surface_mae_mm'),
                    silhouette_iou=m.get('silhouette_iou'),coverage=m.get('model_coverage_fraction'),mask_sha256=m.get('mask_sha256')))
        columns=['pose','config_id','selection','status','inference_ms','depth_mae_mm','point_surface_mae_mm','silhouette_iou','coverage','mask_sha256']
        with (directory/'pose_details.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(details)
        # All configurations in each selection group use the SAME valid pose subset per metric.
        summary=[]
        runs=self.data.get('runs',[])
        for selection in sorted({r['selection'] for r in runs}|{d['selection'] for d in details}):
            configs=sorted({r['config_id'] for r in runs if r['selection']==selection}|{d['config_id'] for d in details if d['selection']==selection})
            for metric in columns[4:9]:
                values={c:{d['pose']:d[metric] for d in details if d['config_id']==c and d['selection']==selection and d[metric] is not None} for c in configs}
                common=set.intersection(*(set(v) for v in values.values())) if values else set()
                for c in configs:
                    vals=[values[c][p] for p in sorted(common)]
                    summary.append(dict(config_id=c,selection=selection,metric=metric,common_pose_count=len(vals),total_pose_count=len(self.poses),
                        mean=statistics.mean(vals) if vals else None,median=statistics.median(vals) if vals else None,
                        worst=(min(vals) if metric in ('silhouette_iou','coverage') else max(vals)) if vals else None))
        with (directory/'pose_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=['config_id','selection','metric','common_pose_count','total_pose_count','mean','median','worst']);w.writeheader();w.writerows(summary)
        return details,summary
