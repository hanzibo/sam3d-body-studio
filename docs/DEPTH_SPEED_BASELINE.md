# 单帧 MHR 深度拟合加速基线

本阶段只使用已有静态 RGB-D 和冻结的 SAM 输出，不训练新模型，不修改 GUI 默认算法。原 CPU 60 步保留为参考。

## 加速内容

工程加速保留原目标函数、60 步、角度/位移边界和接受阈值：

- MHR 前向/反向移至 GPU，用 CUDA Graph 复用 Adam 计算。
- 缓存同一帧固定的身份形状和表情计算。
- 深度专用渲染不生成法线、颜色，不重复创建拓扑，复用顶点缓冲区。

另测 30/15/10 步预算。减少步数会改变结果，不能称为无损加速。

Standard 保留姿态形变修正。原 TorchScript 修正分支会重建稀疏矩阵并创建 CPU 临时张量，不能直接捕获。加速版初始化时缓存第一层稀疏权重的稠密矩阵，以等价的 XYZ 旋转列计算替代 CPU 临时张量。Fast 保持与原预测一致的修正模式。没有删除修正功能或使用 FP16。

CUDA 与 CPU 浮点规约顺序不同，可见性和最近点筛选也有离散边界，因此迭代轨迹可能不同，必须检查最终质量，不能宣称逐位一致。

## 计时口径

试验获取 Studio GPU 独占锁。每个变体先预热，再重复执行；每次重置参数、Adam 动量和步数。质量保留第一次正式执行的结果，不选择最好看的一次。

- **求解**：MHR、损失、反向、Adam、边界投影、对应更新的模型计算/渲染/匹配、参数输出。
- **后处理路径**：内存中的 RGB-D 与已恢复的 MHR 参数，至候选接受后的网格；包括每次重新计算法线/点云树、初始 guard、求解与接受回退。
- **冷启动**：加载、CUDA Graph 构建、EGL 初始化单列。图构建需要秒级时间。
- **排除**：文件读取、原生参数恢复/导入检查、SAM 推理、GUI、导出、精确三角面评估。
- **网络＋后处理估算**：已有网络同步测速与本次后处理相加，不是新的端到端 FPS，尚未包含 v1 位置校正、采集和显示。

主试验所有变体从同一个已保存的 v2 rigid 结果开始，是 **pose-only 对照**；不能把它当成完整后处理。`--mode pipeline` 另从 v1 平移结果执行 rigid 和 pose 两阶段。

均值先取每张图重复耗时的中位数，再对姿态等权平均；P95 使用所有热帧重复值合并计算。包含全部姿态及回退。guard 用于接受判定，不是独立真值。

## 运行

`PYTHON` 替换成部署配置中对应 Conda Python 的完整路径。无需新权重或 TensorRT 转换。

```bash
PYTHON depth_pose_benchmark.py \
  --output outputs/depth-fusion/speed-RUN \
  --profiles fast384 fast512 standard512 \
  --variants cpu60 cuda60 cuda30 cuda15 cuda10 --warmups 1 --repeats 3

PYTHON depth_pose_benchmark.py \
  --output outputs/depth-fusion/speed-pipeline-RUN \
  --profiles fast384 --variants cpu60 cuda60 cuda30 \
  --mode pipeline --warmups 1 --repeats 3

PYTHON depth_speed_report.py outputs/depth-fusion/speed-RUN \
  --pipeline outputs/depth-fusion/speed-pipeline-RUN
```

默认输入是本机 `outputs/depth-fusion/experiment-v1`、`experiment-v2` 与 `masks/masks.json`。验证 RGB、深度、内参、掩膜、源预测 SHA-256；要求全新输出目录，不覆盖历史。报告包含逐图结果、接受/回退、冷启动、渲染一致性与 cProfile。

## 离线实验可选后端

原默认行为不变，可显式选择 CUDA 后端：

```bash
PYTHON depth_pose_experiment.py \
  --source outputs/depth-fusion/experiment-v1 \
  --masks outputs/depth-fusion/masks/masks.json \
  --output outputs/depth-fusion/experiment-cuda-RUN \
  --profiles fast384 --fit-backend cuda-graph --pose-steps 60
```

首次使用每种 correctives/rigid/pose 模式会建图，随后复用。首张总时间包含建图，报告也记录是否本次建图。CPU 是 `--fit-backend cpu`。不同后端和步数须分目录，禁止混用续跑。

`depth_pose_accelerated.AcceleratedMhrSurfaceModel` 提供原有 `fit_pose` 接口。调用方必须持有 GPU 锁。目前尚未更改 GUI/实时流，也未测量与常驻完整 SAM 共存时的显存和延迟。

## 验证

```bash
PYTHON -m unittest discover -s tests -p 'test_depth*py'
PYTHON tests/depth_speed_integration.py outputs/depth-fusion/experiment-v2
```

单元测试核对旋转特征和导数。集成检查使用本地资产，验证人体变形时的网格、梯度、深度、轮廓、顶点缓冲更新与优化器重置。测试和性能采集独立运行。

单帧表面误差改善不能证明遮挡骨架、关节中心或轴向旋转正确。连续 RGB-D 采集与跟踪留待下一阶段讨论。

## 2026-11-06：36 组冻结输入结果

RTX 4060 Laptop 8 GB；12 姿态 × Fast384/Fast512/Standard512；每个变体预热 1 次、正式测量 3 次。以下是同一个 rigid 初值上的 pose-only 对照：

| 配置 | CPU60 求解 ms | CUDA60 求解 ms | CPU60 / CUDA60 后处理 ms | CPU60 / CUDA60 深度 MAE mm |
| --- | ---: | ---: | ---: | ---: |
| Fast384 | 1359.2 | 126.0 | 1511.1 / 206.6 | 19.06 / 19.27 |
| Fast512 | 1465.0 | 126.1 | 1634.6 / 208.9 | 20.29 / 20.31 |
| Standard512 | 4312.6 | 483.0 | 4497.0 / 591.3 | 17.89 / 17.69 |

CUDA60 后处理 P95 分别为 237.4、236.9、650.9 ms。Fast384 的平均 MAE 差为 +0.20 mm，最差单张为 +2.35 mm（姿态 2），接受数从 12/12 变为 11/12；Fast512 最差差值 +0.26 mm，Standard512 +0.10 mm。因此称为工程近似等价，不能称为完全无损。

Fast384 减至 30/15/10 步，求解分别为 63.4/37.4/24.6 ms，但平均 MAE 变为 22.14/26.91/27.71 mm。保留 CUDA60 为质量参考；较少步数只作实验档位，不自动设为默认。

主结果在本机忽略目录 `outputs/depth-fusion/speed-baseline-v1/`。源代码、掩膜清单有内容指纹与 `source-snapshot/`。历史输入不覆盖。

### 完整 rigid＋pose 两阶段复测

从 v1 平移校正结果重新开始，Fast384 的全部 12 张图，每种变体预热 1 次、正式测量 2 次；每阶段分别执行所列步数：

| 变体 | 两阶段求解 ms | 后处理路径 ms | 后处理 P95 ms | 深度 MAE mm |
| --- | ---: | ---: | ---: | ---: |
| CPU60 | 2833.2 | 3089.3 | 3323.9 | 19.06 |
| CUDA60 | 243.1 | 355.5 | 379.2 | 19.26 |
| CUDA30 | 127.0 | 240.0 | 262.5 | 23.83 |

CUDA60 后处理约提速 8.7 倍，但仅此路径也只有约 2.8 Hz；再加历史网络耗时约为 422 ms，且仍未计入 v1 平移校正、采集和 GUI。因此本阶段没有达到端到端 10 Hz。不能用 pose-only 的 126 ms 或 10 步的 25 ms 宣称整条链路实时。

建议保留 CUDA60 为下一阶段质量基线。CUDA30 虽然更快，但两阶段平均误差上升约 4.77 mm；15/10 步在部分姿态有明显退化。后续需重点处理法线/对应与接受检查的 CPU、渲染、数据往返开销，以及减少每帧完整优化的需求。

两阶段原始记录在 `outputs/depth-fusion/speed-pipeline-v1/`。本地 HTML 报告（含最差样本对照）在 `outputs/depth-fusion/speed-baseline-v1/report.html`。
