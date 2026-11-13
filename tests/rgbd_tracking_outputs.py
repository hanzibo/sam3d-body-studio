# SPDX-License-Identifier: GPL-3.0-only
"""Opt-in integrity audit of real tracking artifacts, no GPU/models."""
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rgbd_tracking_points import project


def audit(directory):
    directory=Path(directory);inputs=json.loads((directory/'inputs.json').read_text());total=0
    for mode in ('candidate','oracle'):
        folder=directory/f'{mode}-15';complete=json.loads((folder/'complete.json').read_text())
        frames=[json.loads(x) for x in (folder/'frames.jsonl').read_text().splitlines()]
        assert len(frames)==len(inputs['frames'])==complete['frames']
        for i,row in enumerate(frames):
            assert row['frame']==i and row['source_index']==inputs['frames'][i]['source_index']
            with np.load(folder/'points'/f'{i:06d}.npz',allow_pickle=False) as f:
                uv=f['pixel_uv'];ids=f['vertex_ids'];b=f['batch']
                assert len(uv)==len(ids)==len(b)
                assert np.isfinite(uv).all() and ((ids>=0)&(ids<18439)).all()
                if len(uv):
                    target=f['target_xyz'];normal=f['normal_xyz']
                    np.testing.assert_allclose(project(target,inputs['k']),uv,atol=.001)
                    np.testing.assert_allclose(np.linalg.norm(normal,axis=1),1.,atol=.0001)
                    assert sum(f['batch_used_steps'])==row['fit']['steps']
                    assert np.all(f['batch_used_steps']>0)
                    for batch in np.unique(b):assert (b==batch).sum()<=1200
                if row['state']=='LOST':assert len(uv)==0 and not row['valid']
                total+=len(uv)
            if row['state']!='LOST':
                with np.load(folder/'states'/f'{i:06d}.npz',allow_pickle=False) as f:
                    assert np.isfinite(f['params']).all() and np.isfinite(f['translation']).all()
                    assert bool(f['valid'])==row['valid']
    print(directory.name,'TRACKING_OUTPUTS_OK',len(frames)*2,'frames',total,'correspondences')

if __name__=='__main__':
    for arg in sys.argv[1:]:audit(arg)
