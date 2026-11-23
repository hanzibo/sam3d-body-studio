# SPDX-License-Identifier: GPL-3.0-only
"""Actual EGL shared-context rendering and cleanup regression."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from studio.rgbd.rgbd_comparison_report import DepthRenderer,SceneRenderer
from studio.rgbd.rgbd_comparison_data import load_npz
import json
root=Path(sys.argv[1]);a=load_npz(root/'frames/000000.npz');faces=load_npz(root/'topology.npz')['faces'];k=json.loads((root/'inputs.json').read_text())['k']
r=DepthRenderer(faces,k);v=a['vertices'][2];center=(v.min(0)+v.max(0))*.5*[1,-1,-1];s=SceneRenderer(faces,center,r.renderer)
try:
    for _ in range(2):
        r.renderer.viewport_width=k['width'];r.renderer.viewport_height=k['height'];z=r.render(v);assert z.shape==(k['height'],k['width']) and (z>.05).sum()>1000
        image=s.render(v,v[::5],np.full((len(v[::5]),3),200,np.uint8));assert image.shape==(360,640,3) and image.std()>5
finally:s.close();r.close()
print('REAL_SHARED_EGL_RENDER_AND_CLEANUP_OK')
