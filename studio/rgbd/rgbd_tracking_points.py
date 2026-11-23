# SPDX-License-Identifier: GPL-3.0-only
"""Measured RGB-D correspondences, with pixel provenance and no mask dependency."""
import cv2
import numpy as np
from studio.rgbd.rgbd_geometry import matrix, coefficients
def project(points,k):
    return points[:,:2]/np.maximum(points[:,2:],.01)*[k['fx'],k['fy']]+[k['cx'],k['cy']]

# COCO17 -> MHR70, checked against upstream metadata/mhr70.py.
COCO_TO_MHR = [0, 1, 2, 3, 4, 5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14]


def sampled_indices(frames, hz):
    if not np.isfinite(hz) or hz <= 0:raise ValueError('hz must be positive')
    result=[]; next_time=0.
    for i,row in enumerate(frames):
        t=row['relative_time_s']
        if t+1e-6 >= next_time:
            result.append(i);next_time=t+1/hz-.001
    return result


class SequenceGeometry:
    """Cache calibration; normals from raw organized depth, never RGB splat holes."""
    def __init__(self, sequence, factor=4, raw_stride=2):
        self.sequence=sequence; self.factor=factor;self.raw_stride=raw_stride
        c=sequence.camera['raw_calibration'];self.k=c['color_intrinsics'];self.dk=c['depth_intrinsics']
        self.rotation=np.asarray(c['depth_to_color_R'],np.float32).reshape(3,3)
        self.translation=np.asarray(c['depth_to_color_t_m'],np.float32)
        yy,xx=np.indices((self.dk['height'],self.dk['width']))
        rays=cv2.undistortPoints(np.stack((xx,yy),-1).astype(np.float32).reshape(-1,1,2),matrix(self.dk),coefficients(c['depth_distortion'])).reshape(*xx.shape,2)
        self.rays=np.concatenate((rays,np.ones((*xx.shape,1),np.float32)),2)[::raw_stride,::raw_stride]
        self.maps=cv2.initUndistortRectifyMap(matrix(self.k),coefficients(c['color_distortion']),None,matrix(self.k),(self.k['width'],self.k['height']),cv2.CV_32FC1)
        self.small_k={key:(value/factor if key in ('fx','fy','cx','cy') else value//factor if key in ('width','height') else value) for key,value in self.k.items()}

    def read(self,index):
        image,raw,row=self.sequence.read(index)
        image=cv2.remap(image,*self.maps,cv2.INTER_LINEAR)
        z=raw[::self.raw_stride,::self.raw_stride].astype(np.float32)*row['depth_scale_mm']/1000
        xyz=(self.rays*z[...,None])@self.rotation.T+self.translation
        dx=xyz[1:-1,2:]-xyz[1:-1,:-2];dy=xyz[2:,1:-1]-xyz[:-2,1:-1]
        normals=np.zeros_like(xyz);n=np.cross(dx,dy);length=np.linalg.norm(n,axis=2)
        normals[1:-1,1:-1]=n/np.maximum(length[...,None],1e-12)
        valid=np.isfinite(xyz).all(2)&(z>.1)&(z<10)
        good=np.zeros_like(valid)
        good[1:-1,1:-1]=(valid[1:-1,1:-1]&valid[1:-1,:-2]&valid[1:-1,2:]&valid[:-2,1:-1]&valid[2:,1:-1]&
            (np.abs(z[1:-1,2:]-z[1:-1,:-2])<.04)&(np.abs(z[2:,1:-1]-z[:-2,1:-1])<.04)&(length>1e-8))
        p=xyz[good].astype(np.float32);normals=normals[good].astype(np.float32)
        rgb_uv=project(p,self.k).astype(np.float32)
        uv=np.rint(rgb_uv/self.factor).astype(int);h,w=self.small_k['height'],self.small_k['width']
        inside=(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)&(p[:,2]>.05)
        p,normals,rgb_uv,uv=p[inside],normals[inside],rgb_uv[inside],uv[inside]
        linear=uv[:,1]*w+uv[:,0]
        # Linear-time nearest measured Z, deterministic first winner on ties.
        nearest=np.full(h*w,np.inf,np.float32);np.minimum.at(nearest,linear,p[:,2])
        winners=np.flatnonzero(p[:,2]==nearest[linear])
        first=np.full(h*w,len(p),np.int64);np.minimum.at(first,linear[winners],winners)
        slots=np.flatnonzero(first<len(p));keep=first[slots]
        cloud=np.zeros((h*w,3),np.float32);ns=np.zeros_like(cloud);pixels=np.zeros((h*w,2),np.float32)
        cloud[slots]=p[keep];ns[slots]=normals[keep];pixels[slots]=rgb_uv[keep]
        return image,cloud.reshape(h,w,3),ns.reshape(h,w,3),pixels.reshape(h,w,2),row


class ProjectiveSurface:
    """Select current measurements around visible model projections; optional oracle mask.

    Every returned pair is logged before optimizer use. `pixel_uv` belongs to
    the measured point, NOT merely the source model vertex's projected pixel.
    """
    def __init__(self,cloud,normals,pixels,k,small_k,render,bbox,config,oracle_mask=None):
        self.cloud,self.normals,self.pixels=cloud,normals,pixels
        self.k,self.small_k,self.render,self.c=k,small_k,render,config
        self.bbox=np.asarray(bbox);self.oracle_mask=oracle_mask
        self.batches=[];self.phase='track';self.candidates=[]

    def pairs(self,vertices):
        h,w=self.cloud.shape[:2];uv=project(vertices,self.small_k)
        center=np.rint(uv).astype(int)
        visible_z=self.render(vertices)
        inside=(vertices[:,2]>.05)&(center[:,0]>=0)&(center[:,0]<w)&(center[:,1]>=0)&(center[:,1]<h)
        ids=np.flatnonzero(inside);x,y=center[ids].T
        ids=ids[(visible_z[y,x]>.05)&(np.abs(visible_z[y,x]-vertices[ids,2])<.025)]
        # Uniform image tiles keep torso density from consuming the entire budget.
        if len(ids):
            tile=(center[ids,1]//8)*(w//8+1)+center[ids,0]//8
            order=np.lexsort((ids,tile));groups=tile[order]
            starts=np.maximum.accumulate(np.where(np.r_[True,groups[1:]!=groups[:-1]],np.arange(len(groups)),0))
            rank=np.arange(len(groups))-starts
            ids=ids[order[np.lexsort((groups,rank))]][:self.c.max_points]
        offsets=np.array([(x,y) for y in range(-2,3) for x in range(-2,3)])
        query=center[ids,None,:]+offsets[None]
        ok=(query[:,:,0]>=0)&(query[:,:,0]<w)&(query[:,:,1]>=0)&(query[:,:,1]<h)
        qx=np.clip(query[:,:,0],0,w-1);qy=np.clip(query[:,:,1],0,h-1)
        targets=self.cloud[qy,qx];ns=self.normals[qy,qx];pixels=self.pixels[qy,qx]
        view=targets/np.maximum(np.linalg.norm(targets,axis=2,keepdims=True),1e-8)
        ok &= (targets[:,:,2]>.05)&(np.abs((ns*view).sum(2))>.15)
        ok &= ((pixels>=self.bbox[:2])&(pixels<=self.bbox[2:])).all(2)
        if self.oracle_mask is not None:
            px=np.clip(np.rint(pixels[:,:,0]).astype(int),0,self.oracle_mask.shape[1]-1)
            py=np.clip(np.rint(pixels[:,:,1]).astype(int),0,self.oracle_mask.shape[0]-1)
            ok &= self.oracle_mask[py,px]
        distance=np.linalg.norm(targets-vertices[ids,None,:],axis=2)
        distance[~ok]=np.inf
        best=distance.argmin(1);good=distance[np.arange(len(ids)),best]<self.c.max_correspondence_m
        rows=np.arange(len(ids))[good];chosen=best[good]
        self.candidates.append(dict(vertex_ids=ids.copy(),model_uv=project(vertices[ids],self.k).astype(np.float32)))
        ids=ids[good];target=targets[rows,chosen];normal=ns[rows,chosen];pixel=pixels[rows,chosen]
        self.batches.append(dict(vertex_ids=ids.copy(),pixel_uv=pixel.copy(),target_xyz=target.copy(),normal_xyz=normal.copy(),phase=self.phase))
        return ids,target,normal


def point_membership(uv,mask):
    uv=np.asarray(uv).reshape(-1,2);h,w=mask.shape
    xy=np.rint(uv).astype(int);inside=(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
    result=np.zeros(len(uv),bool);result[inside]=mask[xy[inside,1],xy[inside,0]]
    return result
