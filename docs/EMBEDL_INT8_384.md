# 本地 INT8 384 实验

从 Embedl 512 静态 ONNX 改图：修改 7 个形状常量，重新生成 32 份 24×24 位置角度表，保留全部权重、Q/DQ 和量化尺度。
输入 [1,3,384,384]，输出 [1,1280,24,24]。位置表恢复源文件中的 BF16 周期值，先在 512 上逐表重建验证，再计算 384 网格；没有插值旧角度。
这不是发布者提供或重新标定的 384 量化模型，仍需独立评估。

## 完整推理时间

RTX 4060 Laptop 8GB，正式 worker，YOLO 每次运行；3 次反馈，解码/MHR/中间预测编译，反馈缓存、关键点专用路径、最终图重放及 CPU IO 优化开启。
各 worker 启动预热 10 次，各姿态再预热 3 次、计时 20 次，共 240 次测量。CUDA 同步，包含检测/裁剪/网络/CPU 输出，不含磁盘读写/导出/显示。
两个 384 方案为本轮顺序复测；512 列引用前轮同数据同框结果，仅供参照，未做四方案本轮交替测速。

| 方案 | 测量批次 | 平均 ms | P95 ms |
|---|---|---:|---:|
| Fast FP16 384 | 本轮 | 53.99 | 56.46 |
| Fast INT8 384 | 本轮 | 41.71 | 43.53 |
| Fast INT8 512 | 前轮参照 | 60.90 | 62.83 |
| Standard512 | 前轮参照 | 150.00 | 151.48 |

## 相对 Standard512 的输出差异

12 张相同 RGB 与相机 K；YOLO 实际框逐张检查与前轮一致。Standard512 是冻结的网络预测参照，不是真值。
骨盆对齐仅移除整体平移，不拟合旋转或尺度；相机系指标不做对齐。

| 平均差异 | FP16 384 | INT8 384 | INT8 512（前轮） |
|---|---:|---:|---:|
| 对齐网格 mm | 17.452 | 19.327 | 13.817 |
| 对齐127关节 mm | 17.790 | 19.252 | 13.401 |
| 对齐双臂关键点 mm | 16.107 | 17.588 | 12.260 |
| 相机系网格 mm | 38.611 | 39.926 | 16.246 |
| 相机系骨盆 mm | 32.148 | 35.970 | 9.778 |
| 骨盆深度有符号差 mm | 31.125 | 34.828 | -0.051 |
| 双肘角度 ° | 2.070 | 2.411 | 1.280 |

## 与 FP16 384 直接对比

平均骨盆对齐网格距离 12.931 mm；双肘角度差 1.526°；相机系骨盆距离 12.422 mm。

逐姿态指标及所有计时见 `summary.json`，原始输出见 `fast384/pose-*` 与 `int8384/pose-*`。
未验证实时相机端到端帧率、深度真值误差或连续动作抖动；INT8 384 保留降分辨率与量化两种误差来源。

## Studio 使用

重启 Studio，在启动方案中选择“实验：Fast INT8 384 身体（Embedl）”，然后“应用并预热”。也可选择 Fast、INT8、384；切换分辨率会选择对应的独立引擎。512 INT8 和所有 FP16 方案保留。

新引擎配置键为 `backbone_engine_int8_384`，本机映射到 Data 盘的 `models/embedl-sam-3d-body/embedl_sam3dbody_int8_384_trt10_fp16.engine`。

## 可复核文件

- 完整报告及原始数据：`/run/media/hzb/Data/experiments/embedl384-20261109/full/`
- 对照图：该目录 `visuals/comparison.jpg`，12 张单姿态对照亦保留。
- 编码器 ABBA/BAAB 各240次：`/run/media/hzb/Data/experiments/embedl384-20261109/encoder/encoder.json`，FP16 31.740 ms → INT8 19.829 ms。
- 构建报告：`/run/media/hzb/Data/experiments/embedl384-20261109/build/build.json`，耗时46.44秒，引擎1,013,538,356字节。
- SHA256：`910f80c9a0798fed55dac43f1e9b9b2cc378b943bece79d1fed2113f9255b090`。
- ONNX改图审计：模型目录 `graph384-audit.json`，全部计算节点逐字节相同，仅32个位置表和7个形状常量数值改变。

`scripts/prepare_embedl_384.py` 读取原始FP32 ONNX及已准备的FP16 Q/DQ ONNX，保留共享外部权重数据；384 ONNX不可脱离原FP16 `.onnx.data`单独移动。
构建示例（已有文件时拒绝覆盖）：

```bash
systemd-run --user --scope -p MemoryMax=7G -p MemorySwapMax=2G \
  /run/media/hzb/Data/conda-envs/fast-sam3d-body/bin/python \
  /home/hzb/Work/sam3d-body-deploy/scripts/benchmark_embedl_encoder.py \
  --stage build --strong-fp16 --image-size 384 \
  --onnx-name embedl_sam3dbody_int8_384_folded_fp16.onnx \
  --engine-name embedl_sam3dbody_int8_384_trt10_fp16.engine \
  --output /run/media/hzb/Data/experiments/embedl384-rebuild
```

复测脚本：`scripts/check_embedl_384.py`；报告和投影图：`scripts/report_embedl_384.py`。
未进行桌面操作或真实相机测试；配置界面用offscreen Qt检查，正式worker完成12姿态推理。
