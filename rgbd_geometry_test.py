"""Geometry acceptance: analytic depth/registration, occlusion, metric scoring."""
import numpy as np
from rgbd_geometry import register_depth,depth_metrics,surface_distances
from rgbd_evaluate import render_depth

k=dict(width=64,height=48,fx=60.,fy=60.,cx=31.5,cy=23.5)
d=np.full((48,64),2.,np.float32)
z,xyz=register_depth(d,k,{},k,np.eye(3),[0,0,0])
assert np.allclose(z,d)
assert np.allclose(xyz[20,30],[(-1.5)*2/60,(-3.5)*2/60,2])
# An absolute translation must remain observable, never be silently fitted away.
v=np.array([[-2,-2,2],[2,-2,2],[2,2,2],[-2,2,2]],np.float32)
f=np.array([[0,1,2],[0,2,3]])
r=render_depth(v,f,k)
assert np.allclose(r,2,atol=1e-5)
m,_,_=depth_metrics(d,r,np.ones_like(d,bool));assert m['mae_mm']<.01 and m['silhouette_iou']==1
r=render_depth(v+[0,0,.1],f,k)
m,_,_=depth_metrics(d,r,np.ones_like(d,bool));assert abs(m['mae_mm']-100)<.01 and m['signed_mean_mm']>99
p=np.array([[0,0,1.8],[.1,.2,2.1],[3,0,2]],float)
a=surface_distances(p,v,f)
assert np.allclose(a,[.2,.1,1])
# Nearest-surface z-buffer keeps the foreground, with no hole filling.
k2=dict(k,width=1,height=1,cx=0,cy=0,fx=.01,fy=.01)
z,_=register_depth(np.array([[3.,1.]],np.float32),dict(k,width=2,height=1,cx=0,cy=0),{},k2,np.eye(3),[0,0,0])
assert z[0,0]==1
mask=np.ones_like(d,bool);rendered=np.zeros_like(d);rendered[:24]=2
m,_,_=depth_metrics(d,rendered,mask);assert m['model_coverage_fraction']==.5 and m['mae_mm']==0
print('PASS calibrated deprojection, nearest-Z, absolute camera depth, exact triangle distances, missing coverage')
