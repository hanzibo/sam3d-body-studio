# SPDX-License-Identifier: LicenseRef-SAM-3D-Body
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modifications Copyright (c) 2026 hanzibo and contributors.
# Runtime transforms operate on pinned Fast inherited Meta methods; see
# docs/FAST_EXECUTION.md and LICENSES/SAM-3D-Body.txt.
"""Owned final-core CUDA replay and body-only synchronization/transfer adapter."""
import ast
import hashlib
import inspect
import textwrap
from types import MethodType

import torch


class FinalBodyReplay:
    def __init__(self, model, original):
        self.model = model
        self.original = original
        self.enabled = True
        self.cache = None

    def __call__(self, *args, **kwargs):
        # The upstream caller passes nine tensors and one scalar positional flag.
        eligible = (self.enabled and not self.model.training and not torch.is_grad_enabled()
                    and not kwargs and len(args) == 10 and args[0].is_cuda
                    and args[0].shape[0] == 1
                    and all(isinstance(x, torch.Tensor) and x.is_cuda for x in args[:9])
                    and isinstance(args[9], bool))
        if not eligible:
            return self.original(*args, **kwargs)
        signature = tuple((tuple(x.shape), x.dtype, x.device) for x in args[:9]) + (args[9],)
        if self.cache is None or self.cache[0] != signature:
            inputs = tuple(x.clone() for x in args[:9]) + (args[9],)
            stream = torch.cuda.Stream(device=args[0].device)
            stream.wait_stream(torch.cuda.current_stream(args[0].device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.original(*inputs)
            torch.cuda.current_stream(args[0].device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                outputs = self.original(*inputs)
            self.cache = (signature, inputs, graph, outputs)
        _, inputs, graph, outputs = self.cache
        for target, source in zip(inputs[:9], args[:9]):
            target.copy_(source)
        graph.replay()
        # Includes views of inputs (focal length), so clone every returned tensor.
        return tuple(x.clone() for x in outputs)

    def status(self):
        return dict(enabled=self.enabled, graph_captured=self.cache is not None,
                    version='final_body_core_replay_v1')


def install_final_body_replay(model):
    """Install after apply_compile, before warmup; full hand mode is supported
    only through its original separate paths. Uncompiled core is left unchanged.
    """
    if hasattr(model, '_studio_final_replay'):
        return model._studio_final_replay
    original = getattr(model, '_compiled_interm_pred_body', None)
    if original is None:
        return None
    replay = FinalBodyReplay(model, original)
    model._compiled_interm_pred_body = replay
    model._studio_final_replay = replay
    return replay


def packed_to_cpu(value, original, needed):
    """One asynchronous D2H copy per dtype/device, one completion event.

    CPU outputs own fresh pinned storage. Do not reuse that storage next frame.
    Only pack the estimator fields it actually reads; all public output fields
    remain present. Other recursive_to targets retain the original implementation.
    """
    if not isinstance(value, dict):
        return original(value, 'cpu')
    result = {k: v for k, v in value.items() if k in needed}
    groups = {}
    for k, v in result.items():
        if isinstance(v, torch.Tensor) and v.is_cuda:
            groups.setdefault((v.device, v.dtype), []).append((k, v))
    if not groups:
        return original(result, 'cpu')
    devices = {device for device, _ in groups}
    if len(devices) != 1:
        return original(result, 'cpu')
    for (_, dtype), rows in groups.items():
        flat = torch.cat([v.reshape(-1) for _, v in rows])
        cpu = torch.empty(flat.numel(), dtype=dtype, device='cpu', pin_memory=True)
        cpu.copy_(flat, non_blocking=True)
        offset = 0
        for key, tensor in rows:
            result[key] = cpu[offset:offset+tensor.numel()].view(tensor.shape)
            offset += tensor.numel()
    complete = torch.cuda.Event()
    complete.record(torch.cuda.current_stream(next(iter(devices))))
    complete.synchronize()  # CPU/NumPy consumers must see completed transfers.
    return result


class _QuietBody(ast.NodeTransformer):
    def visit_Expr(self, node):
        call = node.value
        if isinstance(call, ast.Call):
            name = ast.unparse(call.func)
            if name in ('print', 'torch.cuda.synchronize', 'torch.cuda.empty_cache'):
                return ast.copy_location(ast.Pass(), node)
        return self.generic_visit(node)


def _quiet_method(bound, pack=False):
    raw = inspect.unwrap(bound.__func__)
    source = textwrap.dedent(inspect.getsource(raw))
    tree = ast.parse(source)
    fn = tree.body[0]
    fn.decorator_list = []
    needed = {node.slice.value for node in ast.walk(fn)
              if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
              and node.value.id == 'out' and isinstance(node.slice, ast.Constant)
              and isinstance(node.slice.value, str)}
    tree = _QuietBody().visit(tree)
    ast.fix_missing_locations(tree)
    namespace = dict(raw.__globals__)
    if pack:
        recursive = namespace['recursive_to']
        def transfer(value, target):
            return packed_to_cpu(value, recursive, needed) if target == 'cpu' else recursive(value, target)
        namespace['recursive_to'] = transfer
    exec(compile(tree, f'<studio-quiet:{raw.__name__}>', 'exec'), namespace)
    return MethodType(namespace[raw.__name__], bound.__self__), hashlib.sha256(source.encode()).hexdigest()


def install_body_io(estimator):
    """Patch only this estimator/model instance, never torch global functions.

    Body-only, no external FOV model. Full/hand, training, gradients and detail
    timing fall back. The original CPU output boundary remains blocking.
    """
    if hasattr(estimator, '_studio_body_io'):
        return estimator._studio_body_io
    model = estimator.model
    controller = dict(enabled=True, active=False, version='body_quiet_packed_v1', sources={})
    for name in ('run_inference', 'forward_pose_branch'):
        original = getattr(model, name)
        quiet, digest = _quiet_method(original)
        controller['sources'][name] = digest
        def wrapped(*args, _original=original, _quiet=quiet, **kwargs):
            return (_quiet if controller['active'] else _original)(*args, **kwargs)
        setattr(model, name, wrapped)
    original = estimator.process_one_image
    quiet, digest = _quiet_method(original, pack=True)
    controller['sources']['process_one_image'] = digest
    signature = inspect.signature(original)
    def process(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        eligible = (controller['enabled'] and bound.arguments.get('inference_type', 'full') == 'body'
                    and estimator.fov_estimator is None and not model.training
                    and not torch.is_grad_enabled()
                    and not getattr(inspect.getmodule(type(model)), '_INTERM_DETAIL_TIMING', False))
        if not eligible:
            return original(*args, **kwargs)
        controller['active'] = True
        try:
            return quiet(*args, **kwargs)
        finally:
            controller['active'] = False
    estimator.process_one_image = process
    estimator._studio_body_io = controller
    return controller
