# Fast 推理加速交接说明（2026-11-08）

## 给接手 agent 的任务背景

用户最终要接 Orbbec Gemini 336L 做人体重建实时流，当前重点是每帧推理阶段。最近工作集中于 Fast 后端执行优化；不是用上一帧姿态替代当前帧网络预测，也不是继续深度优化跟踪实验。每帧仍独立执行检测、DINOv3、六层解码器和最终完整人体预测。

主要修改仓库：`/home/hzb/Work/sam3d-body-deploy`。当前分支为 `codex/rgbd-depth-refinement`，HEAD 为 `31af863`。工作区有大量未提交改动，包含更早的 RGB-D 深度优化、录制、跟踪与对比功能。**最近的三个 Fast 适配器及其测试/文档仍是未跟踪文件，尚未提交推送。仅克隆远程仓库拿不到本轮实现；需要访问这个本地工作区。不要 reset/clean 或混入其他未提交功能。**

上游只读参考：

- `/home/hzb/Work/work_files/sam-3d-body`
- `/home/hzb/Work/work_files/Fast-SAM-3D-Body`，本轮依据版本 `d72aa36913a1673ace029d345f437561a85ec9e2`
- 本机既有诊断加载器 `/home/hzb/Work/fast-sam3d-body-deploy/run.py`，只有 benchmark 脚本复用；正式 Studio 不依赖它。

遵守 `/home/hzb/Work/AGENTS.md`：第三方 checkout 不直接改。三项适配器均在 Studio 内运行时安装。实际资源路径以 `studio_paths.py` 和未公开的 `config/local.json` 为准，不要重装环境或重新下载权重。

Fast Python：`/run/media/hzb/Data/conda-envs/fast-sam3d-body/bin/python`。本机 GPU 为 RTX 4060 Laptop 8GB。GPU 基准使用 `gpu_lease.acquire_gpu_lease('exclusive')`，避免与 Studio/SAM3 同时加载。

## 建议阅读顺序

1. `/home/hzb/Work/sam3d-body-deploy/docs/FAST_EXECUTION.md`：最新阶段、计时边界、消融、连续帧验收。
2. `/home/hzb/Work/sam3d-body-deploy/docs/FAST_KEYPOINT_PATH.md`：中间 MHR70 专用几何计算及 CUDA Graph。
3. `/home/hzb/Work/sam3d-body-deploy/docs/FAST_FEEDBACK_CACHE.md`：单次解码内反馈缓存和减少复制。
4. `/home/hzb/Work/sam3d-body-deploy/fast_worker.py`：安装时机、开关、共用 forward、ready/inference.json 报告。
5. `/home/hzb/Work/sam3d-body-deploy/fast_feedback.py`、`fast_keypoints.py`、`fast_execution.py`：具体实现。
6. `launch_config.py`、`studio_paths.py`、`stream_protocol.py`、`gui/live_stream.py`：启动配置、资源、实时流协议与调度。文件均位于上述 Studio 根目录。

## 最近已完成的四类优化

### 1. 单次解码内反馈缓存和减少复制

实现：`fast_feedback.py`，入口 `install_fast_feedback_cache(model)`。

- 默认三个中间预测对应五次反馈；同一预测结果会连续用于后续层，原来重复做图像采样与二维/三维位置编码。
- 在一次 decoder 调用内缓存同一 pose-output 对象对应的不可变反馈量，将默认五次计算变为三次计算、两次命中。
- 每层仍对该层的新 token 累加反馈，没有跳过六层中的任何一层，也没有少做中间预测。
- Token 与附加项各只复制一次，减少只读坐标/深度的冗余复制。
- 缓存不跨图像、人物、提示轮次或 decoder 调用；训练、有梯度、不支持的布局等走原回调。

### 2. 中间关键点专用 MHR70 路径与图重放

实现：`fast_keypoints.py`，`SelectedMHR70`、`install_fast_keypoint_path(model)`。

- 从原 `[308,18566]` 回归矩阵前70行全部非零系数提取依赖，不按阈值近似。
- 中间阶段只需要468个表面顶点和50个骨架节点参与回归；原路径先计算18439顶点和308地标。
- 体型/表情基和蒙皮影响项相应裁剪：51337个影响项减少到1347个。
- **全部127个骨架节点的变换仍计算**，以保留祖先链；MHR70不是前70个骨架节点。
- 单人 CUDA 中间几何计算用 CUDA Graph 重放。每次更新当前参数，输出独立存储，防止下一层/下一帧覆盖旧结果。
- 仅裁剪顶点的首版未提速，甚至略慢；加入图重放后才有明确收益。不要将失败首版当作当前实现。
- 需要编译中间预测和 slim 路径；最终完整输出、correctives、手部专用 head 等保留原数学路径。

### 3. 最终完整人体预测核心 CUDA Graph

实现：`fast_execution.py`，`FinalBodyReplay`、`install_final_body_replay(model)`。

- 包装已编译的完整身体预测核心，覆盖姿态 head、完整 MHR、相机及投影计算。
- 预热时捕获图，每次复制九个张量输入；相机主点布尔选项也在图签名内。
- 全部19个输出张量复制到独立存储，包含可能引用输入的焦距视图，避免跨帧覆盖。
- Fast 单人、无梯度、已编译身体模式默认启用；不支持的调用走原核心。完整手部模式不默认启用该最终图。
- 最终网格仍为18439顶点、36874三角面，保留127骨架和其余公开输出。

### 4. 身体模式同步、分配与 GPU→CPU 传输优化

实现：`fast_execution.py`，`install_body_io()`、`_quiet_method()`、`packed_to_cpu()`。

- 对本实例的 `process_one_image`、`run_inference`、`forward_pose_branch` 生成运行时 AST 适配版本。
- 只删除独立诊断语句：`print()`、分段计时 `torch.cuda.synchronize()`、每帧 `empty_cache()`。
- 保留原有结果缓存重置、数学赋值、检测/裁剪、异常和流依赖；没有替换全局 torch 函数。
- 只下载原公开结果组装实际读取的字段，按 device/dtype 合并后复制到本次独占的 pinned CPU 存储。
- CPU/NumPy消费前仍等待传输完成事件；外层 worker 的完整推理边界也保留 CUDA 完成等待，所以不是异步提前返回制造虚假低耗时。
- 身体模式、无额外FOV模型、无梯度、未开启中间详细计时时适用；完整手部模式保留原同步路径。

## 当前共用数据路径

`stream JPEG/单图 → 解码/BGR转RGB/内参 → fast_worker.forward → YOLO11m-Pose → 裁剪与batch → DINOv3 FP16 TensorRT → 六层身体解码器（中间MHR70图＋单次解码反馈缓存）→ 最终完整预测图 → 合并CPU输出 → 检查 → stream网格/姿态输出或单图导出`。

四项优化没有更换权重、重建TensorRT引擎、减少分辨率、删减反馈次数或训练网络。标准版后端未接入这些适配器。384/512仍是用户选择的既有启动配置，本轮不是靠降低输入尺寸取得收益。

## 性能证据：同轮对照才可以作差

以下均为RTX4060 Laptop、身体模式、骨干/YOLO TensorRT、解码器/MHR/中间预测编译、三次中间反馈，预热后测量。

| 阶段 | 本轮基线→启用新优化，完整推理均值 | 样本数/变体 |
| --- | --- | --- |
| 反馈缓存，384 | 65.407→63.876 ms，节省1.531 ms / 2.34% | 240 |
| MHR70专用图，384；双方已有反馈缓存 | 64.966→60.376 ms，节省4.590 ms / 7.07% | 240 |
| 最终图＋执行优化，384；双方已有前两项 | 57.443→53.820 ms，节省3.623 ms / 6.31% | 240 |
| 最终图＋执行优化，512；双方已有前两项 | 81.531→77.112 ms，节省4.419 ms / 5.42% | 80 |

不同日期/运行时的温度、频率和调度有波动，**不能把表中不同轮次首尾直接作差，不能累加百分比声称总加速率**。尚未做全部四项关闭与全部开启的最新同轮总消融。

最新384四路消融，每变体120帧：基线57.740ms、仅最终图55.851ms、仅执行优化55.841ms、两项启用54.073ms。单独插桩：最终完整核心3.029→0.827ms、解码器15.460→12.796ms。

“完整推理”包括检测、裁剪、网络及CPU输出；排除加载、编译/图捕获、文件读写、GUI和渲染。分项CUDA事件另行插桩，包含调度/流等待，不是纯GPU核函数活跃时间。单图耗时的倒数不等于相机实际显示FPS。

## 数值与连续帧验收

- 原测速图＋已有12姿态，共13张图；另覆盖3/2/1/0次中间反馈、掩膜和内参切换。
- 最新384：最大顶点单坐标差0.001878mm、相机平移分量差0.003338mm、二维投影差0.013916px；这是相对旧路径的数值差异，不是相对真实人体的精度误差，也不是逐位相同。
- 最终图已检查新输入生效、旧GPU输出不被覆盖；打包传输已检查混合dtype逐值一致和旧CPU输出保持不变。
- 512真实13图二维最大差0.001350px；压力用例最大1.972656px来自截断矩形掩膜＋0.8倍焦距，原预测某点相机Z约0.00515m，投影已在图像外约(-32378,26930)px。不要把该极值当成有效视野内普遍偏移，报告有配对NPZ和范围说明。
- 10项Fast相关单元测试、语法检查、`git diff --check`已通过。
- 正式worker完成加载、预热、单图导出；报告确认最终图实际捕获、执行优化启用。
- 正式stream协议连续处理3张不同已录制Gemini图像，推理54.478/54.233/54.427ms，worker完成61.552/59.797/60.074ms，帧号、任务号和完整网格检查通过。
- 上述stream检查不是实时相机采集测试，也没有测GUI显示FPS。本轮没有改变相机、GUI调度或渲染代码。

## 原始结果位置

均位于 `/home/hzb/Work/sam3d-body-deploy/` 下：

- `outputs/profiling/feedback-cache-384-poses-20261108/report.json`
- `outputs/profiling/keypoint-path-384-final-20261108/report.json`
- `outputs/profiling/execution-384-final-20261108/report.json`
- `outputs/profiling/execution-512-20261108/report.json`
- `outputs/release-validation/20261108-194307-628158-fast/prediction/inference.json`
- `outputs/release-validation/20261108-194307-628158-fast/stream-checks.json`

目录中有被测适配器快照/hash、配对NPZ与日志。原始真人数据、权重及这些outputs受.gitignore保护，不要纳入开源提交。许可标注见适配器文件头与 `docs/provenance.json`；三项适配器标为 `LicenseRef-SAM-3D-Body`，原创测试/基准脚本使用GPL-3.0-only，不要统一抹成上游Fast MIT。

## 运行状态和独立回退

正常用户需要“释放模型→应用并预热”，无需重新转换TensorRT引擎。启动worker前可分别设置：

| 环境变量置0 | 回退内容 |
| --- | --- |
| `BODYSTUDIO_FEEDBACK_CACHE=0` | 原始逐层反馈回调 |
| `BODYSTUDIO_KEYPOINT_PATH=0` | 原始完整中间MHR几何 |
| `BODYSTUDIO_FINAL_REPLAY=0` | 原始已编译最终完整核心，不做手工图重放 |
| `BODYSTUDIO_BODY_IO=0` | 原始同步、empty_cache、逐项CPU搬运 |

四项满足适用条件时默认开启；“开关开启”不等于任意配置都实际走该路径。ready/inference.json检查：

- `feedback_cache`：默认三次反馈方案正常为computations=3、hits=2。
- `keypoint_path`：enabled、eligible_config、graph_captured；eligible_config不是逐调用计数。
- `execution.final_replay`：enabled、graph_captured；未编译时可能为null。
- `execution.body_io`：enabled、active、version、sources哈希；active=false表示调用已结束，不代表优化关闭。

四项是实现层默认优化，不是新GUI启动参数；同一个launch_config/config_id在代码升级后可能有不同实现。对比记录需要同时保留代码diff/版本、适配器hash和上述状态，不能只看config_id。运行时切换编译guard后要重新预热，不将重编译计入稳定时延。

## 复查命令

```bash
cd /home/hzb/Work/sam3d-body-deploy
/run/media/hzb/Data/conda-envs/fast-sam3d-body/bin/python -m unittest discover -s tests -p 'test_fast_*.py' -v

# GPU功能验收；需要先释放其他Studio模型
BODYSTUDIO_GPU_KIND=exclusive /run/media/hzb/Data/conda-envs/fast-sam3d-body/bin/python scripts/check_inference.py --backend fast --stream-smoke

# 输出目录必须换成全新目录；不覆盖已有报告
/run/media/hzb/Data/conda-envs/fast-sam3d-body/bin/python scripts/benchmark_fast_execution.py --image outputs/gui/20261103-174926-349375/input.png --poses outputs/rgbd-sets/recording-20261106-135615-851252 --output outputs/profiling/NEW_UNIQUE_DIR --runs 60
```

其他基准/测试入口：`scripts/benchmark_fast_feedback.py`、`scripts/benchmark_fast_keypoints.py`、`scripts/profile_fast_decoder.py`，以及 `tests/test_fast_feedback.py`、`tests/test_fast_keypoints.py`、`tests/test_fast_execution.py`。

接手后若继续优化，应先在当前全部默认开启的基线上重新剖析剩余推理耗时。DINOv3、解码器进一步图捕获、检测器降频等仅是候选，未在本轮实现或验证。用户当前优先级是推理阶段，未经新的需求不要把工作重心改为UI/渲染或深度迭代跟踪。
