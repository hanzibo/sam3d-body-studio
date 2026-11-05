# Attribution and licensing boundaries

Copyright © 2026 hanzibo and contributors, for original Studio contributions. The default grant is GPL-3.0-only, except for files explicitly marked otherwise. In particular, `calibrated_visualization.py` is an adaptation under the SAM License, **not** the root GPL grant. Upstream copyright and permission notices are retained for adaptations.

SAM 3D Body Studio builds on the following projects. It does not claim authorship of their neural networks, learned parameters, acceleration methods, rig definitions, or licensed assets. Full upstream repositories are obtained separately at the revisions in `dependencies.json`; some Studio files adapt upstream loader, rendering or conversion workflows, as identified in their headers and [the source audit](docs/LICENSE_AUDIT.md). Bibliographic entries in `CITATIONS.bib` are reproduced from the SAM 3D Body and Fast SAM 3D Body project READMEs (checked 2026-11-05).

The audit found **unresolved combined-work licensing questions**, especially GPL components used with SAM/DINO custom-license components and optional AGPL code. Process isolation and separate downloads do not by themselves establish license compatibility. The repository does not represent a cleared, uniformly GPL-licensed inference stack. Attribution corrections and copies of license texts do not resolve those questions or establish model access authorization.

| Project | Contribution used / referenced | Upstream terms / source |
| --- | --- | --- |
| Meta SAM 3D Body | Original human mesh recovery model, metadata and visualization utilities | [SAM License](https://github.com/facebookresearch/sam-3d-body/blob/main/LICENSE); [project](https://github.com/facebookresearch/sam-3d-body) |
| Fast SAM 3D Body | Accelerated inference path; optional learned MHR→SMPL and smoothing networks | [MIT license in Fast repository](https://github.com/yangtiming/Fast-SAM-3D-Body/blob/main/LICENSE); inherited upstream components and assets keep their terms; [project](https://github.com/yangtiming/Fast-SAM-3D-Body) |
| Momentum Human Rig (MHR) | Human rig interfaces, TorchScript asset and conversion mapping | [Repository code: Apache-2.0](https://github.com/facebookresearch/MHR/blob/main/LICENSE); verify the exact asset's release terms and provenance separately; a SAM checkpoint's embedded material is not automatically relicensed Apache-2.0 |
| ComfyUI | SAM3DBody inference implementation, model management and patching | [GPL-3.0](https://github.com/Comfy-Org/ComfyUI/blob/master/LICENSE) |
| SAM3 | Optional point-prompt image segmentation | [SAM License at the pinned revision](https://github.com/facebookresearch/sam3/blob/0570b3a5be9c4e694f23d85232fb55f4a6f1f7fc/LICENSE), copied in `LICENSES/SAM3.txt` |
| DINOv3 | Fast backbone source | [Custom DINOv3 License at the pinned revision](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/LICENSE.md), copied in `LICENSES/DINOv3.txt`; not MIT or Apache-2.0 |
| SMPL | Optional neutral body model, used for joint generation | [Model license](https://smpl.is.tue.mpg.de/modellicense.html); registration and separate authorization required; no original or converted SMPL data is distributed here |
| Ultralytics YOLO | Fast person detection and pose prompts | [Upstream license](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) (AGPL-3.0 or separate commercial terms); applies independently to code and models |
| PyQt5 | Qt desktop interface | [GPLv3 / commercial dual license](https://www.riverbankcomputing.com/software/pyqt/) |
| pyqtgraph | OpenGL widget and mesh data infrastructure; `gui/mesh_geometry.py` implements vectorized smooth vertex normals compatible with MeshData | [MIT license](https://github.com/pyqtgraph/pyqtgraph/blob/master/LICENSE.txt) |
| Orbbec Python SDK | Gemini 336L RGB and depth acquisition API | [Python wrapper repository: Apache-2.0](https://github.com/orbbec/pyorbbecsdk/blob/main/LICENSE); separately installed SDK binaries and device firmware may have additional terms |
| Gemini 336L examples by hanzibo | Camera workflow reference; Studio's frame adapter is self-contained | [Examples](https://github.com/hanzibo/orbbec-gemini-336l-python-examples) |

Additional runtime dependencies (PyTorch, NumPy, SciPy, OpenCV, trimesh, TensorRT and others) retain their own licenses. Installing them is separate from licensing the Studio source. The dependency fetch script downloads code only, not checkpoints or datasets, and does not accept model licenses on the user's behalf.

License snapshots and their immutable sources/hashes are listed in [LICENSES/README.md](LICENSES/README.md). The file-by-file inventory in [docs/provenance.json](docs/provenance.json) describes the reviewed source scope, not a complete transitive dependency SBOM. Fast's MIT notice covers its authors' contributions; files retaining Meta copyright and inherited SAM/DINO components must be reviewed under their original terms. Optional learned conversion/smoothing weights are not assumed to be MIT merely because their repository contains an MIT file.

The GPL license for this project's original integration code grants no additional rights to third-party weights, gated models, photographs, datasets, SMPL assets, or generated TensorRT engine binaries. Public availability of a mirror does not establish permission to redistribute its contents. Source and model acquisition must comply with the upstream terms; this repository provides no mechanism to bypass an access decision.
