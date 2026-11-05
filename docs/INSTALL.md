# 安装、资源与可复现运行

## 1. 准备方式

Studio 使用独立子进程隔离 GUI、标准推理、Fast、相机 SDK 和 SAM3。无需启动 ComfyUI 服务；只引用其固定版本 Python 实现。标准路径使用 ComfyUI 的 BF16 safetensors，Fast 使用原始 ckpt+MHR 格式，不能互换。

推荐 Ubuntu + NVIDIA CUDA、Python 3.11，Qt5 使用 X11／XWayland。已使用的本机版本：PyTorch 2.10.0+cu128、torchvision 0.25.0+cu128、NumPy 1.26.4；Fast TensorRT 10.16.1.11，Ultralytics 8.3.228。这是验证过的版本组合，不代表所有 CUDA／驱动组合兼容。384 引擎限身体模式；完整手部精修使用 512。

本仓库没有模型、SDK wheel、Conda 环境、真实拍摄数据或编译缓存。源码检查可无 GPU 运行；完整人体推理需要获授权的模型及 CUDA。

## 2. 源码与环境

先复制配置，按已有环境修改路径。相对路径基于仓库根目录，支持 `~`；`SAM3D_STUDIO_PATHS` 可以指向另一份配置。路径配置与界面启动方案 `BODYSTUDIO_CONFIG` 是两件事。

```bash
cp config/paths.example.json config/local.json
python3 scripts/fetch_dependencies.py
# 需要分割时，另加 --sam3
```

脚本只克隆 `dependencies.json` 中的固定提交；遇到已有不同提交会停止，不覆盖源码。已有上游仓库可直接填写绝对路径，无需再次克隆。所有依赖保持单独许可证，不采用 git submodule，也不随 Studio 发布。

创建环境示例（已有可用环境可以直接复用其解释器路径，不要覆盖）：

```bash
conda create -p "$PWD/envs/standard" python=3.11 pip -y
conda activate "$PWD/envs/standard"
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements/gui.txt -r requirements/body.txt
python -m pip install ./third_party/detectron2 --no-build-isolation --no-deps
```

Detectron2 编译需匹配的 CUDA Toolkit 与 GCC。不要直接使用不受 CUDA 支持的新系统 GCC；按 PyTorch／CUDA 要求配置编译器。也可参考 [官方安装文档](https://github.com/facebookresearch/sam-3d-body/blob/main/INSTALL.md)。`requirements/body.txt` 是直接依赖列表，部分版本随上游变化，并非承诺全平台锁定；干净机器的 GPU 安装尚需验证。

Fast 推荐从上述环境克隆，再安装额外依赖，避免改变正常环境：

```bash
conda create -p "$PWD/envs/fast" --clone "$PWD/envs/standard" --offline -y
conda activate "$PWD/envs/fast"
python -m pip install -r requirements/fast-extra.txt
```

两个环境完成后执行各自 `python -m pip check`。根据实际路径填写 `standard_python` 和 `fast_python`。GUI 在标准环境运行；基础 GUI 预览只需 `requirements/gui.txt`，不加载 torch。相机 SDK 与 SAM3 使用额外独立环境，见下文。

## 3. 必要模型

请从有授权的来源按上游模型说明获取。Studio 不自动申请权限或下载权重，也不重新分发模型。文件路径可通过 `config/local.json` 修改。

| 功能 | 所需资源 |
| --- | --- |
| 标准版 | `standard_checkpoint` 指向 ComfyUI 格式 `sam_3d_body_dinov3_bf16.safetensors`；[格式说明／模型卡](https://huggingface.co/Comfy-Org/sam-3d-body) |
| Fast | `fast_checkpoint_dir/model.ckpt`、`model_config.yaml`、`assets/mhr_model.pt`；默认位于 Fast 源码的 `checkpoints/sam-3d-body-dinov3/`，见 [原模型卡](https://huggingface.co/facebook/sam-3d-body-dinov3) |
| Fast 检测 | `yolo_checkpoint` 指向 `yolo11m-pose.pt`，默认位于 Fast 源码 `checkpoints/yolo/`，按 [Ultralytics](https://github.com/ultralytics/ultralytics) 条款获取 |
| 离线骨干源码 | `dinov3_repo`，默认 `models/torch-hub/hub/facebookresearch_dinov3_main`；通过固定提交依赖脚本获取 |
| 可选 TensorRT | 对应 512／384 骨干 engine、YOLO engine；本机生成后登记，见下一节 |

Fast 当前验证的原始模型文件 SHA256 固定在 `fast_worker.py`，标准 safetensors 也有固定校验；不兼容或被改写的文件会拒绝加载。更换其他模型版本需要审查并适配，不能仅重命名。TensorRT 引擎登记只用于防止文件意外变化，不代表安全扫描。

```bash
python3 studio_paths.py --doctor
./launch_gui.sh
```

Doctor 只报告路径和存在性，不证明依赖可导入、CUDA 可用或模型推理成功。GUI 预热失败时检查 `outputs/sessions/`、`outputs/gui/desktop.log`。没有相机也可以预热：相机尺寸布局使用示例图缩放，不会打开设备。

## 4. TensorRT 与编译

引擎必须匹配当前 GPU、TensorRT 和输入尺寸。不要把其他 GPU 上的 engine 当作可移植权重。

```bash
# Fast 环境内，通过其解释器使用 Studio 的隔离启动器
./build-backbone.sh --export --build --image-size 512 --output-dir models/engines/512
./build-backbone.sh --export --build --image-size 384 --body-only --output-dir models/engines/384
# YOLO 在 Fast 环境中按上游工具生成，传入绝对模型路径
# cd third_party/Fast-SAM-3D-Body
# /absolute/path/to/envs/fast/bin/python convert_yolo_pose_trt.py --model /absolute/path/yolo11m-pose.pt --imgsz 640 --half
```

确认 `backbone_engine_512`、`backbone_engine_384`、`yolo_engine` 路径指向实际文件，再登记你可信的本地构建：

```bash
python3 engine_registry.py --register models/engines/512/backbone_dinov3_fp16.engine
python3 engine_registry.py --register models/engines/384/backbone_dinov3_fp16.engine
python3 engine_registry.py --register third_party/Fast-SAM-3D-Body/checkpoints/yolo/yolo11m-pose.engine
```

界面选好分辨率、执行器与编译开关之后再“应用并预热”。更换参数会创建不同启动方案；编译缓存放在 `inductor_cache`，首次编译明显慢于已预热推理。Fast 默认不用 MoGe，内参来自照片元数据／固定标定／默认估计。

## 5. Gemini 336L 与 SAM3

相机代码针对 Gemini 336L（PID 0x0807），RGB 使用 1280×800 MJPG 30 FPS；不是通用任意型号相机选择器。先按 [Orbbec Python SDK](https://github.com/orbbec/pyorbbecsdk) 与 [336L 示例](https://github.com/hanzibo/orbbec-gemini-336l-python-examples) 确认设备及 USB 权限，在独立相机环境安装兼容 SDK、NumPy、OpenCV、SciPy，再填写 `camera_python`。不要将与当前 SDK 不兼容的其他版本覆盖进该环境。

相机采集时保存逐张真实内参。已有照片的 `camera.json` 优先；固定相机内参是可选的 `config/gemini336l_rgb.json`，格式为 `{"intrinsics":{"width":1280,"height":800,"fx":...,"fy":...,"cx":...,"cy":...}}`，数值必须来自自己的标定。没有该文件时上传照片使用默认估计，程序不会把作者的相机标定当作你的设备标定。

SAM3 按 [上游安装说明](https://github.com/facebookresearch/sam3) 在独立环境部署，填写 `sam3_python`、`sam3_repo`、`sam3_checkpoint`。Studio 的前／背景点分割会先释放人体／MHR 模型，再启动 SAM3；分割进程退出后才能预热人体模型。锁只协调同一 Studio 目录的进程，无法替其他程序管理显存。

## 6. 可选 SMPL

不使用 SMPL 时可以完全跳过；MHR 骨架和 One Euro 不依赖 SMPL。使用时单独从 [SMPL 官网](https://smpl.is.tue.mpg.de/) 注册、同意对应条款并获得 neutral 模型，放入 `assets/smpl/SMPL_NEUTRAL.pkl` 或数值 NPZ。只加载可信 PKL；不在本仓库分享原始或转换后的 SMPL 文件。旧版含 Chumpy 对象的 PKL 需要在兼容的独立环境转换为数值 NPZ；本仓库不安装旧版 Chumpy。数值 NPZ 必须包含 `v_template`、`shapedirs`（至少前 10 个系数）、稠密 `J_regressor` 和 `kintree_table`，不要分发转换结果。

此外需 Fast 源码里的 `mhr2smpl/experiments/multiview_n30000_e500/` 网络与采样索引，以及 `assets/smpl/mhr2smpl_mapping.npz`；映射单独从 [MHR conversion assets](https://github.com/facebookresearch/MHR/tree/main/tools/mhr_smpl_conversion/assets) 获取并遵守其条款。MLP 平滑还需 Fast 的 `experiments/smoother_w5/`。若依赖 Git LFS，需按上游方式获取实际文件，不能使用 LFS 指针代替模型。缺少资源时界面会列出。

## 7. 验证与本地数据

```bash
python3 -m unittest discover -s tests -v
# 使用已配置的标准／Fast 环境与模型；只做一次预热后单图推理
python3 scripts/check_inference.py --backend standard
python3 scripts/check_inference.py --backend fast
# Fast PyTorch 512 初始验证
python3 scripts/check_inference.py --backend fast --eager
```

检查会启动独立 worker，校验配置、就绪信号、输出顶点形状和有限性，最后退出并释放模型。默认使用上游 dancing 示例，不随本仓库复制。结果仅保存在 `outputs/release-validation/`。不涉及相机实拍、机器人控制或精度数据集评测。

无 GPU GUI 启动检查：激活 GUI 环境后运行 `QT_QPA_PLATFORM=offscreen python tests/gui_startup.py`。离屏检查不验证 OpenGL 场景渲染；正常桌面运行需要真实 OpenGL。安装桌面入口用 `python3 scripts/install_desktop.py --desktop`，GNOME 可能需要右键“允许启动”。
