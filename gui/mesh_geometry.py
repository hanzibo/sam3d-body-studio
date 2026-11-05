# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors, Studio modifications.
# Copyright (c) 2012 University of North Carolina at Chapel Hill.
# Luke Campagnola and pyqtgraph contributors; see LICENSES/pyqtgraph-MIT.txt.
# Adapted MeshData.vertexNormals cache/index interface from pyqtgraph 0.13.7;
# modified normal accumulation to vectorized NumPy operations.
"""Vectorized equivalent of pyqtgraph's smooth vertex normals."""
import numpy as np
from pyqtgraph.opengl import MeshData


def vertex_normals(vertices, faces):
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    result = np.zeros_like(vertices, dtype=np.float32)
    np.add.at(result, faces.reshape(-1), np.repeat(normals, 3, axis=0))
    result /= np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-20)
    return np.ascontiguousarray(result)


class FastMeshData(MeshData):
    def vertexNormals(self, indexed=None):
        if self._vertexNormals is None:
            self._vertexNormals = vertex_normals(self.vertexes(), self.faces())
        if indexed is None:
            return self._vertexNormals
        if indexed == 'faces':
            return self._vertexNormals[self.faces()]
        raise ValueError('Invalid normal indexing')
