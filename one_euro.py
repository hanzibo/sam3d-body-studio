"""Causal One Euro filter for arrays, using actual irregular sample timestamps."""
import math
import numpy as np


class OneEuroFilter:
    def __init__(self,min_cutoff=1.5,beta=5.,d_cutoff=1.):
        values=(min_cutoff,beta,d_cutoff)
        if not all(math.isfinite(x) for x in values) or min_cutoff<=0 or beta<0 or d_cutoff<=0:
            raise ValueError('One Euro 截止频率必须为正数，beta 必须非负')
        self.min_cutoff=min_cutoff;self.beta=beta;self.d_cutoff=d_cutoff
        self.reset()

    def reset(self):
        self.stamp=None;self.raw=None;self.filtered=None;self.derivative=None;self.samples=0

    @staticmethod
    def alpha(cutoff,dt):
        return 1./(1.+1./(2.*np.pi*cutoff*dt))

    def __call__(self,value,stamp):
        value=np.asarray(value,dtype=np.float64)
        if not np.isfinite(value).all() or not math.isfinite(stamp):raise ValueError('One Euro 输入或时间戳无效')
        if self.stamp is None or stamp<=self.stamp or value.shape!=self.raw.shape:
            self.filtered=value.copy();self.derivative=np.zeros_like(value);self.samples=1
        else:
            dt=stamp-self.stamp
            derivative=(value-self.raw)/dt
            a=self.alpha(self.d_cutoff,dt)
            self.derivative=a*derivative+(1-a)*self.derivative
            cutoff=self.min_cutoff+self.beta*np.abs(self.derivative)
            a=self.alpha(cutoff,dt)
            self.filtered=a*value+(1-a)*self.filtered
            self.samples+=1
        self.raw=value.copy();self.stamp=stamp
        return self.filtered.astype(np.float32).copy()
