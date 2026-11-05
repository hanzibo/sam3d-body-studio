# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""JSON-lines process boundary for the Studio parameter laboratory."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback
import numpy as np
from mhr_lab_engine import LabEngine


def emit(kind, **kw):
    print('@@MHRLAB '+json.dumps(dict(type=kind, **kw), ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output', required=True); ap.add_argument('--device', default='auto')
    args = ap.parse_args(); out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    from gpu_lease import acquire_gpu_lease
    global _gpu_lease
    _gpu_lease = acquire_gpu_lease('mhr') if args.device != 'cpu' else None
    engine = LabEngine(args.device)
    emit('ready', metadata=engine.metadata())
    for line in sys.stdin:
        request = {}
        try:
            request = json.loads(line)
            kind = request['type']
            if kind == 'shutdown': break
            raw_parameters = None
            if kind == 'seed':
                source = request.get('source', '默认中性人体')
                data = request.get('data')
                if request.get('path'):
                    with np.load(request['path'], allow_pickle=False) as f:
                        data = {k:f[k].copy() for k in f.files}
                if data is not None:
                    keys=('pred_pose_raw','global_rot','body_pose_params','hand_pose_params','scale_params','shape_params','expr_params','pred_cam_t','mhr_model_params','body_pose','hand','scale','shape','face')
                    raw_parameters={k:np.asarray(data[k]).tolist() for k in keys if k in data}
                    if 'lab_raw_json' in data:raw_parameters=json.loads(str(data['lab_raw_json']))
                state, result = engine.seed(data, source)
            elif kind == 'render':
                state = request['state']; result = engine.render(state)
            else:
                raise ValueError('未知实验室命令')
            target = out/'frame.npz'; temp = out/'frame.tmp.npz'
            np.savez(temp, **result, faces=engine.faces, baseline_vertices=engine.baseline)
            os.replace(temp, target)
            emit('frame', request_id=request['request_id'], path=str(target), state=state,
                 base=engine.base if kind=='seed' else None, source=engine.source,raw_parameters=raw_parameters)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            emit('error', request_id=request.get('request_id'), message=str(exc))


if __name__ == '__main__':
    try: main()
    except Exception as exc:
        traceback.print_exc(); emit('fatal', message=str(exc)); sys.exit(1)
