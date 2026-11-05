# Publication validation — 2026-11-05

This release was checked on the existing Ubuntu / RTX 4060 8GB deployment. No environment or model is included in the repository.

| Check | Result / scope |
| --- | --- |
| Machine configuration, derived paths, profile validation, engine registration | 6 standard-library tests passed |
| Fast person selection | 3 unit tests passed |
| MHR One Euro | Synthetic tests passed: raw preservation, position projection, reset, legacy profile handling and export |
| GPU model exclusion | Separate-process tests passed: body/MHR vs SAM3, duplicate SAM3 exclusion and crash recovery |
| RGB-D geometry | Local EGL rendering test passed: deprojection, nearest-Z, absolute translation error, triangle distance and missing coverage |
| GUI startup | Three tabs created without loading model or camera; offscreen Qt, isolated settings |
| Standard GPU inference | Existing authorized ComfyUI-format BF16 checkpoint; body / 512; two warmups then one upstream dancing image; finite 18439×3 vertices |
| Fast GPU inference | Existing original-format checkpoint; body / 384; backbone + YOLO TensorRT, decoder/MHR/intermediate compile enabled; two warmups then same image; finite 18439×3 vertices |
| Clean source export | Exported only Git index files to a separate directory; unconfigured GUI startup passed; configured existing local resources then Fast GPU inference passed |
| Release scope | Tracked file audit excludes models, licensed SMPL assets, nested repositories, photos, history, logs, local absolute paths and obvious credential patterns |

The clean-export test reused existing environments, weights and accelerator caches. It is not proof of a fresh machine's dependency installation, a cold TensorRT build, camera capture, every startup combination, or an accuracy dataset evaluation. No benchmark claim is made from these single-image checks. Private logs and generated meshes stay under local `outputs/release-validation/` and are not distributed.

Run the checks described in [INSTALL.md](INSTALL.md) for your own machine. GitHub Actions runs source compilation, model-free tests and offscreen GUI construction; GPU inference is intentionally not part of CPU CI.
