#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Fold constant ONNX QuantizeLinear weights with bounded host RAM.

Preserves the publisher's calibration scales and all activation Q/DQ nodes.
Writes a separate graph and external data file; source downloads stay intact.
Every folded tensor is checked against the ONNX reference implementation.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
from onnx.external_data_helper import set_external_data
from onnx.reference import ReferenceEvaluator


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--fp16',action='store_true',help='Cast remaining FLOAT tensors and IO to FP16 after exact constant quantization')
    a=p.parse_args()
    assert a.source.resolve()!=a.output.resolve()
    data_path=a.output.with_suffix('.onnx.data')
    if a.output.exists() or data_path.exists():
        raise FileExistsError('Refusing to overwrite prepared output')
    model=onnx.load(str(a.source),load_external_data=False)
    initial={t.name:t for t in model.graph.initializer}
    def array(t):
        if t.external_data:
            e={v.key:v.value for v in t.external_data}
            return np.memmap(a.source.parent/e['location'],mode='r',offset=int(e.get('offset',0)),
                             dtype=helper.tensor_dtype_to_np_dtype(t.data_type),shape=tuple(t.dims))
        return numpy_helper.to_array(t)
    folds={n.output[0]:n for n in model.graph.node if n.op_type=='QuantizeLinear' and all(k in initial for k in n.input)}
    nodes=[n for n in model.graph.node if not(n.op_type=='QuantizeLinear' and n.output[0] in folds)]
    used={k for n in nodes for k in n.input}
    keep=[t for t in model.graph.initializer if t.name in used]
    report=dict(source=str(a.source),output=str(a.output),folded_weights=len(folds),checks=[])
    with data_path.open('wb') as f:
        def write(name,arr):
            arr=np.ascontiguousarray(arr)
            offset=f.tell();f.write(memoryview(arr).cast('B'))
            t=TensorProto(name=name,data_type=helper.np_dtype_to_tensor_dtype(arr.dtype),dims=arr.shape)
            # set_external_data requires a raw_data field; clear it afterwards.
            t.raw_data=b'x'
            set_external_data(t,location=data_path.name,offset=offset,length=arr.nbytes)
            t.ClearField('raw_data')
            return t
        new=[]
        for name,node in folds.items():
            w,s,z=[array(initial[k]) for k in node.input]
            attrs={x.name:helper.get_attribute_value(x) for x in node.attribute}
            axis=attrs.get('axis',1)
            shape=[1]*w.ndim
            if s.ndim:shape[axis]=s.size
            bounds=np.iinfo(z.dtype)
            # Opset20: saturate(round-to-even(x/scale)+zero_point).
            q=np.clip(np.rint(w/s.reshape(shape))+z.reshape(shape),bounds.min,bounds.max).astype(z.dtype)
            mini=helper.make_model(helper.make_graph([node],'constant_weight_reference',
                [helper.make_tensor_value_info(k,t.data_type,t.dims) for k,t in zip(node.input,[initial[k] for k in node.input])],
                [helper.make_tensor_value_info(name,initial[node.input[2]].data_type,w.shape)]),
                opset_imports=list(model.opset_import),ir_version=model.ir_version)
            ref=ReferenceEvaluator(mini).run(None,dict(zip(node.input,[w,s,z])))[0]
            if not np.array_equal(q,ref):raise AssertionError(node.name)
            new.append(write(name,q))
            report['checks'].append(dict(node=node.name,shape=list(w.shape),equal_to_onnx_reference=True))
            del w,s,z,q,ref,mini
            gc.collect()
        for tensor in keep:
            values=array(tensor)
            if a.fp16 and tensor.data_type==TensorProto.FLOAT:
                values=np.clip(values,-65504,65504).astype(np.float16)
            new.append(write(tensor.name,values))
    del model.graph.node[:];model.graph.node.extend(nodes)
    del model.graph.initializer[:];model.graph.initializer.extend(new)
    if a.fp16:
        for vi in list(model.graph.input)+list(model.graph.output)+list(model.graph.value_info):
            if vi.type.tensor_type.elem_type==TensorProto.FLOAT:
                vi.type.tensor_type.elem_type=TensorProto.FLOAT16
    # Remove intermediate type entries for deleted constant inputs, if any.
    onnx.save_model(model,str(a.output))
    onnx.checker.check_model(str(a.output))
    report.update(source_data_bytes=a.source.with_suffix('.onnx.data').stat().st_size,
                  folded_data_bytes=data_path.stat().st_size,remaining_quantize=sum(n.op_type=='QuantizeLinear' for n in nodes),remaining_floats_cast_to_fp16=a.fp16)
    with data_path.open('rb') as f:report['data_sha256']=hashlib.file_digest(f,'sha256').hexdigest()
    a.output.with_suffix('.prepare.json').write_text(json.dumps(report,indent=2))
    print('PREPARE_OK',json.dumps({k:v for k,v in report.items() if k!='checks'}),flush=True)


if __name__=='__main__':main()
