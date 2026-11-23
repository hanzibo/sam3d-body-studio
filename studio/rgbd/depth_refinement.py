# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Conservative translation-only visible-surface RGB-D refinement (metres, CV axes).

No rotations, shape, scale, network tensors or joint angles are fitted. A spatial
block split keeps guard points out of fitting; guard is an acceptance check, not
an independent anatomical accuracy benchmark.
"""
from dataclasses import asdict, dataclass
import time
import cv2
import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class DepthConfig:
    iterations: int = 5
    max_points: int = 1500
    max_translation_m: float = .35
    max_correspondence_m: float = .12
    huber_m: float = .02
    erosion_px: int = 5
    eigen_ratio: float = .03
    prior_strength: float = .02
    min_points: int = 100
    block_px: int = 24
    max_silhouette_drop: float = .015
    max_coverage_drop: float = .02

    def validate(self):
        for key in ('iterations','max_points','erosion_px','min_points','block_px'):
            if type(getattr(self,key)) is not int:raise ValueError(key+' must be an integer')
        if not 1<=self.iterations<=20 or not 100<=self.max_points<=10000 or not 0<=self.erosion_px<=30 or not 20<=self.min_points<=1000 or not 8<=self.block_px<=128:raise ValueError('invalid refinement size')
        for key,lo,hi in [('max_translation_m',.001,1.),('max_correspondence_m',.005,.5),('huber_m',.001,.1),('eigen_ratio',.001,.5),('prior_strength',.001,1.),('max_silhouette_drop',0.,.1),('max_coverage_drop',0.,.1)]:
            value=getattr(self,key)
            if not np.isfinite(value) or not lo<=value<=hi:raise ValueError('invalid '+key)
        return self


def split_pixels(shape,block=24):
    yy,xx=np.indices(shape)
    return ((xx//block)*73856093+(yy//block)*19349663)%5==0


def unproject(depth,k):
    yy,xx=np.indices(depth.shape)
    return np.stack(((xx-k['cx'])/k['fx']*depth,(yy-k['cy'])/k['fy']*depth,depth),axis=-1)


def solve_translation(source,target,normals,config):
    """Robust point-to-plane ICP with weak eigen-directions frozen and zero prior."""
    c=config.validate();tree=cKDTree(target);delta=np.zeros(3);history=[]
    for _ in range(c.iterations):
        moved=source+delta;distance,index=tree.query(moved)
        valid=distance<c.max_correspondence_m
        if valid.sum()<c.min_points:break
        n=normals[index[valid]];residual=np.einsum('ij,ij->i',n,target[index[valid]]-moved[valid])
        weights=np.minimum(1.,c.huber_m/np.maximum(np.abs(residual),1e-9))
        h=n.T@(weights[:,None]*n)/len(n);b=n.T@(weights*residual)/len(n)-c.prior_strength*delta
        eig,q=np.linalg.eigh(h);keep=eig>max(eig[-1]*c.eigen_ratio,1e-8)
        step=q[:,keep]@((q[:,keep].T@b)/(eig[keep]+c.prior_strength))
        if np.linalg.norm(step)>.05:step*=.05/np.linalg.norm(step)
        delta+=step
        if np.linalg.norm(delta)>c.max_translation_m:delta*=c.max_translation_m/np.linalg.norm(delta)
        history.append(dict(points=int(valid.sum()),eigenvalues=eig.tolist(),observable_dimensions=int(keep.sum()),step_mm=(step*1000).tolist()))
        if np.linalg.norm(step)<.0002:break
    return delta,history


def refine_translation(observed,xyz,mask,raw_depth,k,render,config=None):
    """Return applied delta, final Z buffer, JSON report. render(delta) reuses context."""
    began=time.perf_counter();c=(config or DepthConfig()).validate();mask=np.asarray(mask,bool)
    if observed.shape!=mask.shape or raw_depth.shape!=mask.shape or xyz.shape!=(*mask.shape,3):raise ValueError('RGB-D shape mismatch')
    guard=split_pixels(mask.shape,c.block_px)
    valid=mask&np.isfinite(observed)&(observed>.05)&np.isfinite(xyz).all(2)
    interior=cv2.erode(mask.astype('uint8'),np.ones((2*c.erosion_px+1,)*2,np.uint8))>0
    fit=valid&interior&~guard
    report=dict(schema='studio_depth_translation_v1',config=asdict(c),accepted=False,reason='insufficient_depth',
        fit_pixels=int(fit.sum()),guard_pixels=int((valid&guard).sum()),delta_translation_m=[0.,0.,0.],candidate_translation_m=[0.,0.,0.],
        fixed='pose, identity, scale, joint rotations and local positions',
        split='24px-style spatial block hash modulo 5; guard excluded from correspondences and initial Z; guard used only to accept/reject',
        scope='sensor surface consistency, not anatomical ground truth; clothing and segmentation remain confounders')
    def finish(delta,depth):
        report['total_seconds']=time.perf_counter()-began
        return delta,depth,report
    if fit.sum()<c.min_points or (valid&guard).sum()<c.min_points:return finish(np.zeros(3),raw_depth)
    overlap=fit&(raw_depth>.05)
    if overlap.sum()<c.min_points:
        report['reason']='insufficient_initial_overlap';return finish(np.zeros(3),raw_depth)
    # Fit-only Z seed, bounded; local correspondences otherwise cannot capture depth bias.
    seed=np.zeros(3);seed[2]=np.clip(np.median(observed[overlap]-raw_depth[overlap]),-c.max_translation_m,c.max_translation_m)
    yy,xx=np.nonzero(fit);target=xyz[yy,xx].astype(float)
    rng=np.random.default_rng(0)
    if len(target)>10000:target=target[rng.choice(len(target),10000,replace=False)]
    tree=cKDTree(target);distance,index=tree.query(target,k=12)
    neighbors=target[index];centered=neighbors-neighbors.mean(axis=1,keepdims=True)
    eigen,vec=np.linalg.eigh(np.einsum('nki,nkj->nij',centered,centered)/12)
    good=(distance[:,-1]<.05)&(eigen[:,0]/np.maximum(eigen.sum(1),1e-12)<.08)
    target,normals=target[good],vec[good,:,0]
    report['normal_points']=len(target)
    if len(target)<c.min_points:report['reason']='insufficient_reliable_normals';return finish(np.zeros(3),raw_depth)
    # Only the raw visible surface participates; no rear vertices or fabricated depth.
    source=unproject(raw_depth,k)[overlap]
    if len(source)>c.max_points:source=source[rng.choice(len(source),c.max_points,replace=False)]
    start=time.perf_counter();step,history=solve_translation(source+seed,target,normals,c);candidate=seed+step
    if np.linalg.norm(candidate)>c.max_translation_m:candidate*=c.max_translation_m/np.linalg.norm(candidate)
    report.update(solver_seconds=time.perf_counter()-start,iterations=history,candidate_translation_m=candidate.tolist())
    candidate_depth=render(candidate)
    fixed=valid&guard
    def guard_score(depth):
        error=np.minimum(np.abs(depth[fixed]-observed[fixed]),.1)
        error[depth[fixed]<=.05]=.1
        model=depth>.05
        return dict(capped_mae_mm=float(error.mean()*1000),coverage=float((mask&model).sum()/mask.sum()),iou=float((mask&model).sum()/max(1,(mask|model).sum())))
    before,after=guard_score(raw_depth),guard_score(candidate_depth)
    report.update(guard_before=before,guard_candidate=after)
    if after['capped_mae_mm']>before['capped_mae_mm']-.5:reason='no_guard_improvement'
    elif after['iou']<before['iou']-c.max_silhouette_drop:reason='silhouette_degraded'
    elif after['coverage']<before['coverage']-c.max_coverage_drop:reason='coverage_degraded'
    elif np.linalg.norm(candidate)>=c.max_translation_m*.999:reason='translation_bound_reached'
    else:reason='accepted'
    report.update(reason=reason,accepted=reason=='accepted')
    if report['accepted']:
        report['delta_translation_m']=candidate.tolist();return finish(candidate,candidate_depth)
    return finish(np.zeros(3),raw_depth)
