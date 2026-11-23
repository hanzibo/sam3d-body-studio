# Fast384 与 INT8 512，以 Studio Standard512 为参照

2026-11-09，RTX 4060 Laptop 8GB，已录制 Gemini 336L 的 12 个全身姿态。
Standard 指 Studio 的 Comfy BF16 512 身体路径，并非人体真值。

## 最终结果

| 指标 | Fast FP16 384 | Fast INT8/FP16 512 |
|---|---:|---:|
| 完整推理平均耗时，含每次 YOLO | 53.52 ms | 60.90 ms |
| 完整推理 P95 | 56.37 ms | 62.83 ms |
| 相对 Standard：骨盆平移对齐后的网格平均距离 | 17.452 mm | 13.817 mm |
| 相对 Standard：骨盆平移对齐后的 127 关节平均距离 | 17.790 mm | 13.401 mm |
| 相对 Standard：骨盆平移对齐后的双臂 6 关键点平均距离 | 16.107 mm | 12.260 mm |
| 相对 Standard：双肘几何夹角平均差 | 2.070° | 1.280° |
| 相对 Standard：相机系网格平均距离，未对齐 | 38.611 mm | 16.246 mm |
| 相对 Standard：相机系骨盆位置平均距离，未对齐 | 32.148 mm | 9.778 mm |

Standard 在已给定相同检测框、不含检测的条件下平均 150.00 ms，P95 151.48 ms。
INT8 512 比 Fast384 多约 7.38 ms；网格距离在 10/12 个姿态上更接近 Standard。
姿态 3、4 的对齐网格距离是 Fast384 更小；个别肘角也不总是 INT8 更近。

## 可复核实验设置

- 三方案使用相同原始 RGB 和各张照片的标定内参，均为身体模式。
- 两种 Fast 每次实际执行 YOLO11m-Pose TensorRT，未提供 ROI、预先框或视频缓存键。
- 两种 Fast 的实际检测框逐张一致（容差 1e-4 px）；Standard 使用这些框，消除目标裁剪差异。
- Fast 保留三次中间反馈、全部编译选项、反馈缓存、MHR70 专用 CUDA Graph、最终身体图重放和 IO 优化。
- 不使用网络掩膜提示、SMPL、One Euro、MLP 平滑或深度拟合。
- 各工作进程启动预热 10 次；每姿态再预热 3 次、正式计时 20 次，共每方案 240 个计时样本。
- Fast 每个工作进程共 286 次真实 YOLO 调用，逐姿态核对累积计数。`studio/body/fast_selection.py` 增加只读诊断计数，不更改检测和人物选择逻辑。
- 使用 CUDA 同步后的 worker 推理时间，含 Fast 检测、裁剪、网络和 CPU 输出，不含模型加载、文件读写、导出或渲染；不是实时相机显示帧率。
- 三个工作进程顺序加载与释放，不是 ABBA 交替测试；没有同时加载多个模型。
- 对齐指标仅减去骨盆位置差，不做旋转、尺度拟合。双臂点为 MHR70 的左右肩、肘、腕。

实验根目录：`/run/media/hzb/Data/experiments/standard-reference-yolo-20261109/`。
其中 `REPORT.md`、`summary.json`、`comparison.csv` 为汇总，`checks.json` 为共同框／图像哈希／YOLO 计数验收；
`visuals/comparison.jpg` 为四姿态概览，另有全部 12 姿态对照图。各方案目录保存原始网格、配置、日志和逐次计时。

```bash
systemd-run --user --scope -p MemoryMax=7G -p MemorySwapMax=2G \
  /run/media/hzb/Data/conda-envs/fast-sam3d-body/bin/python \
  /home/hzb/Work/sam3d-body-deploy/scripts/compare_backbones_standard.py \
  --runs 20 --output /run/media/hzb/Data/experiments/standard-reference-yolo-repeat
```

汇总及离线渲染脚本为 `scripts/report_standard_reference.py`，建议在 3 GiB 内存上限的独立进程中运行。
初次 `standard-reference-20261109` 的 Standard／Fast 裁剪不一致，已用 `INVALIDATED.json` 和报告首行标为作废。
其中 Fast 实际也执行了 YOLO；“共同框使 Fast 跳过 YOLO”的旧说明有误。
以上最终数字全部来自修正后的 `standard-reference-yolo-20261109`。

这些数值只说明与 Standard 的一致程度。未用独立人体真值验证绝对精度，也未验证连续动作抖动。
