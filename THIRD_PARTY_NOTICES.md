# Attribution and licensing boundaries

Copyright © 2026 hanzibo and contributors, for the Studio integration code in this repository. SPDX: GPL-3.0-only, except where stated otherwise.

SAM 3D Body Studio builds on the following projects. It does not claim authorship of their neural networks, learned parameters, acceleration methods, rig definitions, or licensed assets. Upstream source is obtained separately at the revisions in `dependencies.json`, not copied into this repository. Bibliographic entries in `CITATIONS.bib` are reproduced from the SAM 3D Body and Fast SAM 3D Body project READMEs (checked 2026-11-05).

| Project | Contribution used / referenced | Upstream terms / source |
| --- | --- | --- |
| Meta SAM 3D Body | Original human mesh recovery model, metadata and visualization utilities | [SAM License](https://github.com/facebookresearch/sam-3d-body/blob/main/LICENSE); [project](https://github.com/facebookresearch/sam-3d-body) |
| Fast SAM 3D Body | Accelerated inference path; optional learned MHR→SMPL and smoothing networks | [MIT license in Fast repository](https://github.com/yangtiming/Fast-SAM-3D-Body/blob/main/LICENSE); inherited upstream components and assets keep their terms; [project](https://github.com/yangtiming/Fast-SAM-3D-Body) |
| Momentum Human Rig (MHR) | Human rig, TorchScript asset and conversion mapping | [upstream repository and license](https://github.com/facebookresearch/MHR); obtain required assets separately |
| ComfyUI | SAM3DBody inference implementation, model management and patching | [GPL-3.0](https://github.com/Comfy-Org/ComfyUI/blob/master/LICENSE) |
| SAM3 / DINOv3 | Optional image segmentation / Fast backbone source | [SAM3](https://github.com/facebookresearch/sam3), [DINOv3](https://github.com/facebookresearch/dinov3); review their respective licenses and model access terms |
| SMPL | Optional neutral body model, used for joint generation | [Model license](https://smpl.is.tue.mpg.de/modellicense.html); registration and separate authorization required; no original or converted SMPL data is distributed here |
| Ultralytics YOLO | Fast person detection and pose prompts | [Upstream license](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) (AGPL-3.0 or separate commercial terms); applies independently to code and models |
| PyQt5 | Qt desktop interface | [GPLv3 / commercial dual license](https://www.riverbankcomputing.com/software/pyqt/) |
| pyqtgraph | OpenGL widget and mesh data infrastructure; `gui/mesh_geometry.py` implements vectorized smooth vertex normals compatible with MeshData | [MIT license](https://github.com/pyqtgraph/pyqtgraph/blob/master/LICENSE.txt) |
| Orbbec Python SDK | Gemini 336L RGB and depth acquisition API | [SDK repository](https://github.com/orbbec/pyorbbecsdk); SDK binaries and device configuration are installed separately |
| Gemini 336L examples by hanzibo | Camera workflow reference; Studio's frame adapter is self-contained | [Examples](https://github.com/hanzibo/orbbec-gemini-336l-python-examples) |

Additional runtime dependencies (PyTorch, NumPy, SciPy, OpenCV, trimesh, TensorRT and others) retain their own licenses. Installing them is separate from licensing the Studio source. The dependency fetch script downloads code only, not checkpoints or datasets, and does not accept model licenses on the user's behalf.

The GPL license for this project's original integration code grants no additional rights to third-party weights, gated models, photographs, datasets, SMPL assets, or generated TensorRT engine binaries. Public availability of a mirror does not establish permission to redistribute its contents. Source and model acquisition must comply with the upstream terms; this repository provides no mechanism to bypass an access decision.
