#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Specialize the publisher's static 512 graph to 384 without changing Q/DQ.

Run after prepare_embedl_onnx.py --fp16. External weight data is referenced,
not duplicated. Position angles are rebuilt from the original FP32 graph's
BF16-rounded periods, with a 512 reconstruction check before conversion.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper


def prepare(folder):
    source = folder/'embedl_sam3dbody_int8_folded_fp16.onnx'
    destination = folder/'embedl_sam3dbody_int8_384_folded_fp16.onnx'
    if destination.exists():
        raise FileExistsError(destination)
    model = onnx.load(source, load_external_data=False)
    original = onnx.load(folder/'embedl_sam3dbody_int8.onnx', load_external_data=False)
    originals = {t.name: t for t in original.graph.initializer}
    consumers = {}
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            consumers.setdefault(name, []).append((node.op_type, index))
    changes = []
    positions = []
    for tensor in model.graph.initializer:
        if tensor.name.startswith('div_') and list(tensor.dims) == [1024, 2, 16]:
            old = numpy_helper.to_array(originals[tensor.name], base_dir=str(folder))
            # Exported periods came from BF16 model buffers. Round back onto
            # that exact grid, rather than interpolate already-rounded angles.
            recovered = np.float32(-2*np.pi*31/32)/old[0, 0]
            bits = recovered.astype(np.float32).view(np.uint32)
            periods = ((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000).view(np.float32)
            def angles(grid):
                axis = (np.arange(grid, dtype=np.float32)+.5)/grid*2-1
                coords = np.stack(np.meshgrid(axis, axis, indexing='ij'), -1).reshape(-1, 2)
                return np.float32(2*np.pi)*coords[:, :, None]/periods
            error = float(np.max(np.abs(angles(32)-old)))
            if error > 1e-6:
                raise ValueError(f'Unknown position encoding: {tensor.name}, reconstruction error {error}')
            new = angles(24).astype(np.float16)
            tensor.CopyFrom(numpy_helper.from_array(new, tensor.name))
            positions.append(dict(name=tensor.name, reconstruction_max_abs=error))
        elif tensor.data_type == onnx.TensorProto.INT64 and np.prod(tensor.dims) <= 5:
            old = numpy_helper.to_array(tensor, base_dir=str(folder))
            # Only Reshape shape inputs; never touch weights/scales or slice indices.
            if consumers.get(tensor.name) and all(op == 'Reshape' and index == 1 for op, index in consumers[tensor.name]):
                new = old.copy()
                new[old == 1024] = 576
                new[old == 1029] = 581
                if tensor.name == 'val_5019':
                    if old.tolist() != [1, 32, 32, -1]:
                        raise ValueError('Unexpected final feature reshape')
                    new = np.array([1, 24, 24, -1], dtype=np.int64)
                if not np.array_equal(old, new):
                    changes.append(dict(name=tensor.name, before=old.tolist(), after=new.tolist()))
            else:
                new = old
            tensor.CopyFrom(numpy_helper.from_array(new, tensor.name))
    if len(positions) != 32 or len(changes) != 7:
        raise ValueError(f'Unexpected graph layout: {len(positions)} position tables, {len(changes)} shapes')
    for value, shape in [(model.graph.input[0], [1, 3, 384, 384]),
                         (model.graph.output[0], [1, 1280, 24, 24])]:
        for dim, number in zip(value.type.tensor_type.shape.dim, shape):
            dim.dim_value = number
    # All intermediate annotations are stale; recompute without loading weights.
    del model.graph.value_info[:]
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.save_model(model, destination)
    onnx.checker.check_model(str(destination))
    external = sorted({entry.value for t in model.graph.initializer for entry in t.external_data if entry.key == 'location'})
    report = dict(source=str(source), destination=str(destination), position_tables=positions,
                  reshapes=changes, external_data_reused=external,
                  warning='Experimental 384 specialization; original 512 quantization scales unchanged, no 384 recalibration')
    destination.with_suffix('.prepare.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, default=Path('/run/media/hzb/Data/models/embedl-sam-3d-body'))
    prepare(parser.parse_args().models)
