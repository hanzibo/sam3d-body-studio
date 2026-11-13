# Embedl INT8 编码器验证（2026-11-09）

首次测量为独立实验；现已作为可选方案接入 Studio 正式 worker，原有 FP16 引擎和默认方案保留。

与 Fast384 的 12 姿态对比见 [以 Standard512 为参照的测试](FAST384_VS_INT8_512.md)，Fast 均包含逐次 YOLO 检测。

后续已完成本地 INT8 384 改图实验，见 [INT8 384 测试](EMBEDL_INT8_384.md)。本页保留最初的 512 测试口径。

## Studio 启动与正式 worker 验收

重启 Studio，在启动方案中选择 **Fast INT8 512 身体（Embedl）**，然后点击“应用并预热”。
也可手动将“骨干精度”切换到 INT8；界面会切换为身体、TensorRT 骨干，并锁定这两个选项。分辨率可选 512 或本地实验版 384，各自使用独立引擎。
恢复 FP16 后解除限制。检测器、编译、反馈次数、SMPL 和滤波等选项仍可按已有规则设置。
保存／导入／导出方案和历史详细配置都会保留 `backbone_precision`。旧方案未记录此字段时按 FP16 读取。
本次新增字段参与配置哈希，因此重新保存的旧方案可能获得新 ID；不改写历史记录。

2026-11-09 正式 worker 验收，各预热 10 次＋测速预热 10 次，正式测量各 200 次；没有提供 ROI，实际执行 YOLO 检测。

| 配置（512 身体，全部现有优化） | 平均 | 中位数 | P95 |
|---|---:|---:|---:|
| FP16 | 77.285 ms | 77.082 ms | 79.662 ms |
| Embedl INT8/FP16 | **61.099 ms** | **61.074 ms** | **62.933 ms** |

节省 16.186 ms（20.94%）。与下方首次实验结果吻合；其约 61.2 ms 已包含已有优化，不能再重复扣减收益。
正式 worker 的 INT8 网格与首次独立实验同图对应顶点最大单坐标差 `4.77e-7 m`，接口接入未引入显著额外差异。
另外通过三帧已录制 Gemini 图像的正式流式协议测试，输出均为有限的 18439 顶点网格。
这是记录图像的流式协议验收，不是实际相机采集／显示帧率测试。

已从运行报告确认以下优化实际启用：

- YOLO TensorRT、INT8/FP16 DINO、FP32 解码器／MHR／中间预测编译。
- 每次解码内反馈缓存（3 次计算、2 次命中）。
- MHR70 专用路径（468 个支持顶点）及其 CUDA Graph。
- 最终完整身体预测 CUDA Graph 重放。
- 身体路径诊断输出／冗余同步清理和所需 CPU 字段打包回传。

正式运行记录：`/run/media/hzb/Data/experiments/embedl-studio-20261109/results.json`。
各子目录 `prediction/inference.json` 保存实际引擎哈希、优化状态及输入／输出尺寸，`ready.json` 保存预热记录。
验收脚本：`scripts/check_embedl_profile.py`，仅使用正式 `launch_fast_model.sh`，顺序启动与释放两个 worker。

```bash
systemd-run --user --scope -p MemoryMax=7G -p MemorySwapMax=2G \
  /run/media/hzb/Data/conda-envs/fast-sam3d-body/bin/python \
  /home/hzb/Work/sam3d-body-deploy/scripts/check_embedl_profile.py \
  --runs 200 --output /run/media/hzb/Data/experiments/embedl-studio-repeat
```

其他机器需在 `config/local.json` 设置 `backbone_engine_int8_512`，并用 `engine_registry.py --register` 注册可信本机构建的引擎。
启动时只加载已存在的引擎，不自动转换或重建。默认位置见 `config/paths.example.json`。
适配器 `embedl_backbone.py` 使用 Torch CUDA 缓冲区和当前流，不依赖 PyCUDA；严格校验所选 384/512 静态引擎／单人／FP16 IO。
未修改上游源码，也未将模型或引擎加入仓库。

## 实测结论

RTX 4060 Laptop 8GB、TensorRT 10.16.1.11；固定 512×512、身体模式、单人。
双方使用相同 YOLO、FP32 编译解码器/MHR、三次中间预测和现有推理优化。
预热各 10 次，ABBA/BAAB 交替计时各 240 次。

| 阶段 | 原 FP16 | INT8/FP16 混合精度 | 降幅 |
|---|---:|---:|---:|
| 编码器均值 | 52.875 ms | 36.158 ms | 31.62% |
| 完整推理均值 | 77.613 ms | 61.204 ms | 21.14% |
| 完整推理 P95 | 79.841 ms | 63.214 ms | — |

完整推理包含检测、裁剪、网络和 CPU 结果回传，不含加载、编译、文件读写和渲染。
这不是实时相机显示帧率；本节最初实验仅测试 INT8 512，后续 384 结果见上方链接。

13 张图的对应网格顶点平均距离为 12.767 mm，最远顶点 49.702 mm；
相机平移三维差平均 24.067 mm、最大 79.449 mm；双肘几何夹角差平均 1.445°、最大 3.862°。
这些是相对 FP16 的输出差异，不是真实人体误差，尚未用深度或连续流验证准确性和抖动。

## 文件位置

- 发布源：<https://huggingface.co/embedl/sam-3d-body>。下载与使用须遵守发布者和上游许可证。
- 下载模型：`/run/media/hzb/Data/models/embedl-sam-3d-body/`
- 成功引擎：该目录下 `embedl_sam3dbody_int8_trt10_fp16.engine`，1,008,350,148 字节（约 962 MiB）。
- 引擎 SHA256：`85ffe61d501517df9ae29d0b36dbae10574707ebb265fe616b9ab2740b5edae8`
- 实验根目录：`/run/media/hzb/Data/experiments/embedl-sam3d-20261109/`
- 总报告：`REPORT.md`、`summary.json`；逐次计时：`encoder/encoder.json`、`full/full.json`。
- 各输入的成对结果：`full/case-*.npz`；对照图：`visuals/comparison.jpg` 和各 `case-*.jpg`。
- 原始构建和测速脚本快照、日志、来源哈希保存在各实验子目录，适合复核本次结果。

## 低内存构建

首次直接构建触发系统 OOM。后续使用 systemd cgroup 限制主机内存；TensorRT workspace 仅限制 GPU 工作区，不能替代这个限制。
6 GiB 受限构建仍超限中止，未再次发生全局 OOM。最终强类型构建成功，耗时约 62 秒。

`scripts/prepare_embedl_onnx.py` 先折叠 128 个固定权重量化节点，每个张量都与 ONNX ReferenceEvaluator 逐值检查；
再将剩余浮点权重和 IO 转为 FP16，保留激活 Q/DQ。原始下载文件保留。
固定权重按原 FP32 量化后才转换剩余数据，因此不能宣称整个准备后的图与原 FP32 图完全等价。
最终引擎是显式 INT8 量化与 FP16 混合执行，不是整个模型全部 INT8。

使用的解释器：`/run/media/hzb/Data/conda-envs/fast-sam3d-body/bin/python`。
后续复测示例（已有引擎，无需重建；输出另建目录）：

```bash
systemd-run --user --scope -p MemoryMax=7G -p MemorySwapMax=2G \
  /run/media/hzb/Data/conda-envs/fast-sam3d-body/bin/python \
  /home/hzb/Work/sam3d-body-deploy/scripts/benchmark_embedl_encoder.py \
  --stage encoder --warmup 10 --runs 60 \
  --output /run/media/hzb/Data/experiments/embedl-repeat-encoder
```

脚本拒绝无主机内存上限的运行，并使用项目 GPU 互斥锁。启动前仍应关闭其他 GPU 模型并检查可用主机内存。
构建时允许使用已有交换空间，未修改系统全局 swap 配置。
大文件及实验结果全部在 Data 盘，不应提交到 Git。
