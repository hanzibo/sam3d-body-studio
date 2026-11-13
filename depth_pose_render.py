# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Depth-only renderer with persistent scene, topology and vertex buffer.

Uses pyrender's buffer layout only for a position-only primitive. Fail closed
when the installed layout is unexpected; original evaluation remains separate.
"""
from rgbd_evaluate import DepthRenderer  # Sets EGL before importing OpenGL.
import numpy as np
import pyrender
from OpenGL.GL import GL_ARRAY_BUFFER, glBindBuffer, glBufferSubData


class FittingDepthRenderer(DepthRenderer):
    def __init__(self, faces, k, shared_renderer=None):
        self.owns_renderer = shared_renderer is None
        if self.owns_renderer:
            super().__init__(faces, k)
        else:
            self.faces, self.k, self.renderer = faces, k, shared_renderer
        self.scene = pyrender.Scene()
        self.scene.add(pyrender.IntrinsicsCamera(k['fx'], k['fy'], k['cx'], k['cy'], znear=.05, zfar=20))
        self.primitive = None
        self.calls = 0
        self.seconds = 0.

    def render(self, vertices):
        import time
        began = time.perf_counter()
        vertices = np.asarray(vertices, np.float32).copy(); vertices[:, 1:] *= -1
        if self.primitive is None:
            self.primitive = pyrender.Primitive(positions=vertices, indices=self.faces)
            self.mesh = pyrender.Mesh([self.primitive])
            self.scene.add(self.mesh)
        else:
            primitive = self.primitive
            if vertices.shape != primitive.positions.shape or len(primitive._buffers) not in (0, 3):
                raise RuntimeError('unsupported pyrender position-only buffer layout')
            if primitive._buffers:
                self.renderer._platform.make_current()
                glBindBuffer(GL_ARRAY_BUFFER, primitive._buffers[0])
                glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.nbytes, vertices)
                glBindBuffer(GL_ARRAY_BUFFER, 0)
            primitive.positions = vertices
            self.mesh._bounds = None
        depth = self.renderer.render(self.scene, flags=pyrender.RenderFlags.DEPTH_ONLY | pyrender.RenderFlags.SKIP_CULL_FACES)
        self.calls += 1; self.seconds += time.perf_counter()-began
        return depth

    def close(self):
        if self.owns_renderer:
            super().close()
