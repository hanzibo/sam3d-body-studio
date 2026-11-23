# SPDX-License-Identifier: GPL-3.0-only
"""Calibrated full-resolution measurements and immutable comparison artifacts."""
import json
from pathlib import Path
import cv2
import numpy as np
from studio.rgbd.rgbd_geometry import matrix,coefficients

METHODS=('independent','cuda60','tracking')
LABELS=('独立网络','独立网络＋CUDA60','初始化＋连续细化')
FIRST_INIT='first_init'
ALL_METHODS=METHODS+(FIRST_INIT,)
METHOD_LABELS=dict(zip(ALL_METHODS,('独立网络','独立网络＋CUDA60','周期初始化＋连续细化','仅首帧初始化＋连续细化')))

def comparison_methods(directory):
    manifest=json.loads((Path(directory)/'manifest.json').read_text())
    methods=tuple(manifest.get('method_order',METHODS))
    if methods not in (METHODS,ALL_METHODS):raise ValueError('unsupported comparison method order')
    return methods

class FullGeometry:
    """All measured depth samples, no fit normal/ROI rejection and no hole filling."""
    def __init__(self,sequence):
        self.sequence=sequence;c=sequence.camera['raw_calibration'];self.k=c['color_intrinsics'];dk=c['depth_intrinsics']
        yy,xx=np.indices((dk['height'],dk['width']))
        rays=cv2.undistortPoints(np.stack((xx,yy),-1).astype(np.float32).reshape(-1,1,2),matrix(dk),coefficients(c['depth_distortion'])).reshape(-1,2)
        self.rays=np.column_stack((rays,np.ones(len(rays),np.float32)))
        self.rotation=np.asarray(c['depth_to_color_R'],np.float32).reshape(3,3);self.translation=np.asarray(c['depth_to_color_t_m'],np.float32)
        self.maps=cv2.initUndistortRectifyMap(matrix(self.k),coefficients(c['color_distortion']),None,matrix(self.k),(self.k['width'],self.k['height']),cv2.CV_32FC1)
    def read(self,index):
        image,raw,row=self.sequence.read(index);image=cv2.remap(image,*self.maps,cv2.INTER_LINEAR)
        z=raw.ravel().astype(np.float32)*row['depth_scale_mm']/1000
        p=(self.rays*z[:,None])@self.rotation.T+self.translation
        good=(z>.05)&(z<20)&(p[:,2]>.05)&np.isfinite(p).all(1);p=p[good]
        k=self.k;h,w=k['height'],k['width']
        uv=np.rint(p[:,:2]/p[:,2:]*[k['fx'],k['fy']]+[k['cx'],k['cy']]).astype(int)
        good=(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h);p=p[good];uv=uv[good];slots=uv[:,1]*w+uv[:,0]
        nearest=np.full(h*w,np.inf,np.float32);np.minimum.at(nearest,slots,p[:,2])
        winners=np.flatnonzero(p[:,2]==nearest[slots]);first=np.full(h*w,len(p),np.int64);np.minimum.at(first,slots[winners],winners)
        cells=np.flatnonzero(first<len(p));cloud=np.zeros((h*w,3),np.float32);cloud[cells]=p[first[cells]]
        return image,cloud.reshape(h,w,3),row


def atomic_npz(file,**arrays):
    file=Path(file);temp=file.with_suffix('.tmp.npz');np.savez_compressed(temp,**arrays);temp.replace(file)


def load_npz(file):
    with np.load(file,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}


def measured_cloud(xyz,rgb,mask,max_points=24000):
    good=np.isfinite(xyz).all(2)&(xyz[:,:,2]>.05)&(xyz[:,:,2]<10)
    ids=np.flatnonzero(good)
    if len(ids)>max_points:ids=ids[np.linspace(0,len(ids)-1,max_points).astype(int)]
    return dict(cloud=xyz.reshape(-1,3)[ids],colors=rgb[:,:,::-1].reshape(-1,3)[ids],person=mask.ravel()[ids])


def static_jitter(rows,method,interval=(.25,1.75)):
    """Detrend declared initial standing interval; excludes gaps/reinitialization jumps."""
    rows=[r for r in rows if r['method']==method and r['valid'] and interval[0]<=r['time_s']<=interval[1]]
    if len(rows)<5:return dict(frames=len(rows),available=False)
    t=np.array([r['time_s'] for r in rows]);p=np.array([r['wrist_camera_m'] for r in rows]);root=np.array([r['root_camera_m'] for r in rows])
    x=np.column_stack((t-t.mean(),np.ones(len(t))))
    wrist=p.reshape(len(t),-1);wr=wrist-x@np.linalg.lstsq(x,wrist,rcond=None)[0]
    rr=root-x@np.linalg.lstsq(x,root,rcond=None)[0]
    return dict(frames=len(rows),available=True,interval_s=interval,
                wrist_detrended_rms_mm=float(np.sqrt(np.mean(wr.reshape(-1,2,3)**2,axis=(0,1)).sum())*1000),
                root_detrended_rms_mm=float(np.sqrt(np.mean(rr**2,axis=0).sum())*1000),
                definition='declared initial still interval, linear detrend; natural human sway remains, not skeletal truth')
