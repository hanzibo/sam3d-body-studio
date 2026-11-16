# SAM 3D Body 手部实验 → AmazingHand 接入交接

更新：2026-11-13。代码基线：`main` / `d31dbc0792f8529a9ea8985cf8c1d86a64562884`。本文为同一台 Ubuntu 主机的下游 agent 编写，路径是当前实机快照，跨机器需重新配置。

## 1. 任务边界与建议

上游已经完成：Gemini 336L RGB 采集、手框检测、单/双手 SAM 手部分支推理、手部网格/骨架/旋转输出、左右手限制、离线预热、TRT/编译加速、One Euro 姿态滤波与实时可视化。

下游需要完成：在 AmazingHand 界面接入这份姿态来源，提取与机构自由度相符的特征，沿用下游已有标定、方向、舵机范围和发送链路，将其映射为舵机刻度。**当前没有 SAM→AmazingHand 的映射实现或经过硬件验证的舵机参数。** 本次只交付交接文档，没有改动下游程序。

推荐复用现有独立 worker，不把 PyTorch、TensorRT 或相机 SDK 导入灵巧手 GUI 主进程。先接完整骨架输出；不需要先搬运整个 Studio 界面，也不需要加载全身推理页。

最重要的四个事实：

1. “手部实验”使用官方 SAM 3D Body 手部分支和共享 checkpoint；WiLoR 只提供检测框/左右标签，**没有使用 WiLoR/MANO 重建**。
2. 输出为 23 个 MHR 手部骨架节点，不是 MediaPipe/MANO 的标准 21 点顺序。
3. 现有实时 GUI 设置 `compact_result=true`，该包为显示精简过；下游首次接入应设 **`compact_result=false`**，才能取得完整每侧名称、旋转、相机平移及原始参数。
4. 现有接口是 stdin/stdout JSON 控制＋临时 NPZ 数据文件；**没有已经实现的骨架 WebSocket/ROS/UDP 服务**。需要此类传输时由下游在适配层增加。

## 2. 目录、环境与本地资源

### 2.1 源码归属

| 目录 | 用途 |
| --- | --- |
| `/home/hzb/Work/sam3d-body-deploy/` | 维护的 Studio 仓库；下游应从这里阅读和复用 |
| `/home/hzb/Work/work_files/sam-3d-body/` | 官方第三方 checkout；视为只读，不在这里写二次开发 |
| `/home/hzb/Work/work_files/Fast-SAM-3D-Body/` | 社区 Fast checkout/共享权重所在；视为只读 |
| `/home/hzb/Work/ORBBEC-Gemini-336L/` | 相机 SDK 使用示例；Studio 已有独立采集 worker |
| `/run/media/hzb/Data/experiments/wilor/datasets/` | 已录制的单手/双手 RGB 数据及内参，适合先离线调接口 |

不要因为数据目录名含 `wilor` 就认为这里使用了 WiLoR 三维网络。

### 2.2 当前解析出的环境与资源

以 [studio_paths.py](/home/hzb/Work/sam3d-body-deploy/studio_paths.py) 和本机忽略的 `/home/hzb/Work/sam3d-body-deploy/config/local.json` 为准。`SAM3D_STUDIO_PATHS` 可指定另一份路径配置；可提交模板是 `/home/hzb/Work/sam3d-body-deploy/config/paths.example.json`。

| 配置键 | 当前路径 |
| --- | --- |
| `fast_python` | `/run/media/hzb/Data/conda-envs/fast-sam3d-body/bin/python3.12` |
| `standard_python`（Studio GUI） | `/run/media/hzb/Data/conda-envs/sam3d-body/bin/python3.12` |
| `camera_python` | `/home/hzb/miniconda3/envs/orbbec336l/bin/python3.11` |
| `fast_checkpoint_dir` | `/home/hzb/Work/work_files/Fast-SAM-3D-Body/checkpoints/sam-3d-body-dinov3/` |
| `hand_detector_checkpoint` | `/run/media/hzb/Data/models/wilor/pretrained_models/detector.pt` |
| `hand_detector_engine` | `/run/media/hzb/Data/models/sam3d-hands/detector/detector_fp16.engine` |
| `backbone_engine_512` | `/run/media/hzb/Data/models/fast-sam3d-body/backbone_trt/backbone_dinov3_fp16.engine` |
| `backbone_engine_int8_512` | `/run/media/hzb/Data/models/embedl-sam-3d-body/embedl_sam3dbody_int8_trt10_fp16.engine` |
| `inductor_cache` | `/run/media/hzb/Data/models/torch-cache/fast-sam3d-body/` |
| `engine_registry` | `/home/hzb/Work/sam3d-body-deploy/config/local.engines.json` |

共享 checkpoint 需要 `model.ckpt`、`model_config.yaml`、`assets/mhr_model.pt`；适配器会校验关键权重的固定 SHA256。权重/引擎不随仓库发布。TensorRT 引擎还需在本机 registry 中登记并通过哈希校验，不能随意替换模型。

检查路径，不启动模型：

```bash
cd /home/hzb/Work/sam3d-body-deploy
/usr/bin/python3 studio_paths.py --doctor
```

`launch_hand.py` 会切换到已配置 Fast Conda Python，并设置官方 checkout + Studio 的导入路径、离线 Torch Hub、Inductor 缓存和线程环境。不要直接照搬 body worker 的 PYTHONPATH：手部主模型来自官方 checkout。

## 3. 总体数据链路

```text
Gemini 彩色 MJPG（SDK 独立进程）
  → 本地 socket → 接收线程解码/最新帧槽
  → 取最新 JPEG + 对应内参 + 帧号 + 主机单调时间
  → hand_worker 常驻独立进程
      → WiLoR YOLOv8m-Pose 手检测（原生/TRT）
      → 置信度筛选 + 左右手策略 + 框连续性
      → 每手裁剪为 512×512（左手先镜像为右手）
      → DINOv3 编码器（原生FP16 / TRT FP16 / TRT混合INT8）
      → 手部解码器 + 可选次数的中间反馈
      → MHR 骨架/手部表面
      → 可选 One Euro 手指系数 + SO(3)手掌朝向 / 位置
      → 恢复左手几何 → 相机系投影 → packed scene
  → 下游读取每侧骨架/旋转
  → 下游计算相对关节/弯曲/对掌特征
  → 下游机构约束与标定 → 舵机刻度 → 已有控制后端
```

当前手部实时链路只用 RGB 和相机内参，**没有融合 Gemini 深度，也不是身体页的 RGB-D 拟合/连续细化**。每个采用帧都会重新检测和网络预测；框连续性用于身份关联，不是省略编码器的姿态跟踪。

## 4. 建议阅读顺序与文件职责

所有以下相对名均相对于 `/home/hzb/Work/sam3d-body-deploy/`；点击链接可直接打开。

| 顺序 | 文件 | 重点 |
| --- | --- | --- |
| 1 | [hand_config.py](/home/hzb/Work/sam3d-body-deploy/hand_config.py) | 加载配置及严格校验 |
| 2 | [launch_hand.py](/home/hzb/Work/sam3d-body-deploy/launch_hand.py)、[hand_worker.py](/home/hzb/Work/sam3d-body-deploy/hand_worker.py) | 环境隔离、启动、JSON 请求/事件、结果保存 |
| 3 | [hand_stream.py](/home/hzb/Work/sam3d-body-deploy/hand_stream.py) | 每帧协议、滤波上下文、NPZ 完整/精简包、原子替换 |
| 4 | [hand_automatic.py](/home/hzb/Work/sam3d-body-deploy/hand_automatic.py) | 检测、单手强制侧、框关联、`pack_scene` |
| 5 | [hand_engine.py](/home/hzb/Work/sam3d-body-deploy/hand_engine.py)、[hand_geometry.py](/home/hzb/Work/sam3d-body-deploy/hand_geometry.py) | `predict_many`、实际字段、骨架子集、坐标/镜像 |
| 6 | [hand_temporal.py](/home/hzb/Work/sam3d-body-deploy/hand_temporal.py)、[one_euro.py](/home/hzb/Work/sam3d-body-deploy/one_euro.py) | 姿态/位置滤波和重置 |
| 7 | [gui/hand_live.py](/home/hzb/Work/sam3d-body-deploy/gui/hand_live.py) | 最新帧调度、一请求在途、显示独立调度、计时 |
| 8 | [gui/hand_lab.py](/home/hzb/Work/sam3d-body-deploy/gui/hand_lab.py) | 启动/释放、快捷方案、GUI 控件及互斥 |

其他相关文件：

- `/home/hzb/Work/sam3d-body-deploy/hand_acceleration.py`：TRT 编码器、MHR 设备化/Graph、编译安装。
- `/home/hzb/Work/sam3d-body-deploy/hand_decoder.py`：中间预测调度与回调分段整体编译。
- `/home/hzb/Work/sam3d-body-deploy/hand_feedback.py`：反馈复制缓存、关键点依赖裁剪、最终手部顶点路径。
- `/home/hzb/Work/sam3d-body-deploy/hand_detector_preprocess.py`：检测器 RGB 固定缓冲预处理。
- `/home/hzb/Work/sam3d-body-deploy/hand_warmup.py`：已有照片离线预热。
- `/home/hzb/Work/sam3d-body-deploy/hand_sequence.py`、`hand_sequence_metrics.py`、`gui/hand_sequence_panel.py`：离线 RGB 序列推理、导出与计时，不是相机采集程序。
- `/home/hzb/Work/sam3d-body-deploy/camera_worker.py`、`launch_camera.sh`、`gui/camera_client.py`、`gui/camera_receiver.py`、`orbbec_color.py`：相机完整链路。
- `/home/hzb/Work/sam3d-body-deploy/gpu_lease.py`、`engine_registry.py`、`studio_paths.py`：互斥、引擎信任、环境/路径。
- `/home/hzb/Work/sam3d-body-deploy/gui/overlay_view.py`：彩色网格叠加；控制数据消费不应依赖它的渲染完成。

较早的 `docs/HAND_LAB.md` 有阶段性的原生耗时及“尚未接实时流”描述；当前实现与 `docs/HAND_LIVE.md`、本文优先，不能把早期状态当现状。

## 5. 启动配置、精度与预热

### 5.1 推荐首轮配置

保守保精度对照：FP16 512；需要复用现有低延迟方案时可选 INT8 512。手部目前固定 512，**不要把身体用的 INT8/FP16 384 引擎填入手部**。

```json
{
  "backbone": "trt_int8",
  "detector": "trt_fp16",
  "decoder_compile": true,
  "mhr_compile": true,
  "mhr_graph": true,
  "tf32": false,
  "compile_mode": "default",
  "feedback_cache": true,
  "intermediate_keypoints": true,
  "final_hand_vertices": true,
  "intermediate_compile": true,
  "intermediate_predictions": 3
}
```

把 `backbone` 改为 `trt_fp16` 即 FP16 TRT 对照。`native` 为原生编码器；解码器和 MHR 当前均 FP32。`HandExecution()` 原生默认不启用加速开关，中间预测默认 5 次。

中间预测范围 0–5，默认5；减少会改变预测，不保证精度不变。六层解码和最终预测始终执行。编译/精度/次数是**加载参数**，修改后需要重启 worker、重新预热。`hand_selection`、confidence、One Euro 参数是**每帧参数**，不需要重载。

DINO 输入每手 `[1,3,512,512]`、特征 `[1,1280,32,32]`。双手若引擎只支持 B1，则顺序编码两次，之后合批解码；一个 `backbone` Python 调用不代表只有一次 TRT enqueue。看 `backbone_runtime.executions` 和 `backbone_engine_executions`。

### 5.2 预热及生命周期

worker 默认读 `hand_warmup_image`，本机是：
`/run/media/hzb/Data/experiments/wilor/datasets/20261112_173116_two_hands_acdf29/frames/000150.jpg`。

支持 `--startup-warmup-image /absolute/file.jpg`；`--no-startup-warmup` 禁用。离线预热覆盖 B1/B2、原图和 1280×800 布局、可选滤波后 MHR 输入签名，完成后清空时间滤波历史。无需用户在线摆手。

接收到 `ready` 后仍必须检查 `startup_warmup.status`：可能是 `ready/disabled/skipped/failed`，**ready 事件只代表模型可接请求，不保证预热成功**。首次编译可能较久，不宜把 GUI 首次加载超时固定为几秒。

Studio 用 systemd user scope 约束工作进程（`MemoryMax=7G`、`MemorySwapMax=2G`）。直接用 `launch_hand.py` 不会自动获得该限制，下游应复用相同 scope 生命周期及整组退出方案。

GPU 锁位于 `/home/hzb/Work/sam3d-body-deploy/outputs/studio-model-gpu.lock`；worker 持有 exclusive flock 至退出，与 Studio 的其他模型互斥。不要删除锁文件来“解锁”，不要拷贝仓库后使用另一把锁启动第二份 GPU 模型。若下游已采集同一台相机，只保留一个相机拥有者。

## 6. 推荐接入方式：常驻 worker 协议

### 6.1 启动与事件

用 QProcess 或 subprocess 启动 `/usr/bin/python3 /home/hzb/Work/sam3d-body-deploy/launch_hand.py --execution '<JSON>'`，cwd 设为维护仓库。实际运行的是配置的 Fast Conda Python。避免 shell 字符串拼接，参数用 argv 列表。

stdin 每行一份 JSON（末尾 `\n`）；stdout/stderr 有模型普通日志，只把 **以 `@@HAND ` 开头的行**当事件 JSON。异步持续消费日志，避免管道塞满。常用事件：

| 事件 type | 含义 |
| --- | --- |
| `stage` | 进度，可能没有 job |
| `ready` | 模型加载完成，带 execution/startup_warmup |
| `hand_stream_result` | 实时帧处理完成，带 job/stream_id/frame_index/path/report |
| `result` / `auto_result` | 手动单图 / 自动单图完成 |
| `warmed` | 手动手框预热完成 |
| `error` | 单次请求失败，检查 job，不能当成有新姿态 |
| `fatal` | 启动/模型级失败，重启前释放进程及资源 |

`{"type":"stop"}` 后换行可退出；控制器结束时先停止提交，再发送 stop 并等待退出，超时才终止整个作用域。

### 6.2 实时请求（完整骨架版）

```json
{
  "type": "hand_stream",
  "job": "唯一请求ID",
  "stream_id": "本次会话唯一ID",
  "frame_index": 123,
  "received_monotonic": 123456.789,
  "jpeg": "原始JPEG字节的base64",
  "camera_matrix": [[608.0,0.0,640.0],[0.0,608.0,400.0],[0.0,0.0,1.0]],
  "output": "/tmp/amazinghand-sam-本次会话",
  "confidence": 0.4,
  "hand_selection": "right",
  "padding": 1.1,
  "warmup": false,
  "compact_result": false,
  "one_euro": {"pose":true,"position":false,"min_cutoff":1.5,"beta":0.5,"d_cutoff":1.0}
}
```

K 和时间是格式示意，**不能把示例数值当相机标定**。K 必须对应未镜像、同分辨率、同裁切的 JPEG。镜像仅供界面预览时，不要把镜像后的 QImage 和原始 K 混用。`received_monotonic` 采用该帧到主机 SDK 的 `time.monotonic()`，不是发送时刻、设备微秒时间戳或 wall clock。跨主机时需另外建立时钟对应。

另一输入形式：省略 jpeg，写 `image=/tmp/.../input.jpg`。文件必须和 `output` 目录相同，名称必须为 `input.jpg`；不是任意磁盘路径。base64 JPEG 分支没有此文件限制，便于复用下游相机帧。当前实际 GUI 用文件形式，后台 OpenCV 解码 BGR 后转 RGB。

返回 `path=/tmp/.../scene.npz`，通过 `scene.pending.npz` 原子替换。**每个 worker 最多一个请求在途，读完并复制 NPZ 所需数据后才提交下一帧**，否则下一帧会覆盖同一文件。完成会话后才删除临时目录。

### 6.3 单图与直接调用

手动框单图：`type=run`，传 `job,image,hand_bbox=[x1,y1,x2,y2],side,padding,camera_matrix,output`；输出 `hand.npz`、`raw_hand.npz`、报告和 PLY。
自动单图：`type=auto_run`，传 `job,image,confidence,hand_selection,padding,camera_matrix,output`；输出 `hands.npz` 和每侧原始张量。预热为 `warmup/auto_warmup`。离线数据集为 `sequence`，详见 worker 和 sequence 文件。

若下游自行写无界面服务，可在**独立 GPU 进程**中持有 `HandEngine` 和 `AutomaticHands`，调用 `AutomaticHands.process(rgb,...)`，得到 `(packed, outputs, report)`；每个 output 为 `(data, raw, crop, report)`。`rgb` 必须 RGB uint8 HWC；OpenCV 默认 BGR。

注意：直接 `AutomaticHands.process` 并不自动建立 One Euro 实时上下文。现有 `hand_stream.process_hand_frame` 负责该生命周期。若绕开它，需显式管理 `engine.temporal_request=(timestamp,options)`、流/缺失/异常重置，并用 try/finally 清掉 request。首次建议直接复用 worker 协议，避免漏掉这些行为。

## 7. 输出结构与坐标契约

### 7.1 单手 data / 每侧 packed 字段

当前资产实际抽查：2321 手部顶点、4613 三角形、23 骨架节点、22 条边。以实际输出为准，不把这些尺寸硬编码为协议永久常量。

| `predict_many` 单手 data 键 | 尺寸 | 语义 |
| --- | --- | --- |
| `vertices` | `[2321,3]` | 米，CV 轴下局部网格 |
| `faces` | `[4613,3]` | 三角形局部顶点索引 |
| `joints` | `[23,3]` | 米，CV 轴下局部骨架坐标 |
| `global_rotations` | `[23,3,3]` | MHR 全局旋转经 CV 基变换；不是机器人角度 |
| `edges` | `[22,2]` | 当前23点数组内的父→子索引 |
| `camera_translation` | `[3]` | 米，局部数据到相机位置的平移 |
| `camera_matrix` | `[3,3]` | 原始彩色图内参 |
| `vertices_2d` | `[2321,2]` | 原始彩色图像像素坐标 |
| `joint_names/joint_ids` | `[23]` | 实际左右侧名称及完整 MHR 骨架编号 |
| `canonical_joint_ids/canonical_vertex_ids` | `[23]/[2321]` | 规范右手载体编号 |
| `wrist` | `[3]` | 单手 data 中的局部腕点；packed 不单独复制此键 |

自动完整包的每侧字段加 `right_` 或 `left_` 前缀，如 `right_joints/right_global_rotations/right_joint_names/right_edges/right_camera_translation`；`sides` 给出本帧存在的手。`wrist`、`joint_ids`、canonical IDs 不由当前 pack_scene 自动打包；需要时由名称/单手 data 取，或扩展协议。

整场景的 `vertices/joints` 则已经加过各自的 camera_translation，双手按实际 sides 顺序拼接。其 camera_translation 为零；faces/edges 也加过偏移。**不要再给整场景 joints 加一次平移。**

下游建议读每侧字段，按名称查找 wrist。相机点 `p_cam=p_local+t_cam`；投影 `q=K @ p_cam`，`uv=q[:2]/q[2]`。CV：x向右、y向下、z向前，距离米，旋转矩阵无量纲。摆脱相机平移可使用 `joints-joints[wrist_index]`。

### 7.2 23 节点次序（当前右手资产实测）

```text
 0 r_wrist
 1 r_pinky0   2 r_pinky1   3 r_pinky2   4 r_pinky3   5 r_pinky_null
 6 r_ring1    7 r_ring2    8 r_ring3    9 r_ring_null
10 r_middle1 11 r_middle2 12 r_middle3 13 r_middle_null
14 r_index1  15 r_index2  16 r_index3  17 r_index_null
18 r_thumb0  19 r_thumb1  20 r_thumb2  21 r_thumb3  22 r_thumb_null
```

当前右手完整 MHR joint_ids 为 42–64；左手导出名称变为 `l_...`，joint_ids 按实际左侧名称查表，不照抄右手编号。不要简单截取前21点。`*_null` 是骨架末端/辅助节点，不代表又多一个可独立驱动舵机的解剖关节；拇指及小指存在额外节点，应以父子链和机构自由度为依据。

父→子链：wrist→pinky0→pinky1→pinky2→pinky3→pinky_null；其他手指依 `edges` 读取，不根据数组相邻关系连接不同手指。

### 7.3 旋转、左手与 raw

内部先把 MHR 关节旋转变成 `R_cv=C @ R_mhr @ C`，`C=diag(1,-1,-1)`。左手走镜像右手推理路径，几何和平移用 `S=diag(-1,1,1)` 反射，旋转 `R_left=S @ R_right @ S`，并翻转三角形绕序。

因此导出的每侧 data 已是实际侧的 CV 表达；`raw_hand.npz` 和 packed 的 `left_raw_hand/left_raw_global_rot/...` **仍是规范右手网络参数**，不要再把它们当实际左手角度。

从全局旋转提取父相对旋转可用 `R_rel=R_parent.T @ R_child`（按正确父节点与矩阵约定），但结果还包含该骨架自身的绑定/静止关系。要得到机器人可用的轴角或弯曲量，仍需参考姿态、轴向、方向和机构标定，不能直接取一个矩阵元素当刻度。

原始 `hand/shape/scale/mhr_model_params/global_rot` 是模型潜在/原生参数；不是 23 个舵机值，也不是23个独立三自由度机器人关节。下游优先使用名称化骨架/相对旋转，而非直接映射潜在系数。

### 7.4 过滤后的结果与 compact 陷阱

开启滤波后，每侧 `*_vertices/joints/global_rotations/camera_translation` 是用于显示的选用结果；另有 `*_raw_vertices/raw_joints/raw_global_rotations/raw_camera_translation` 和 `*_filtered_...` 以供对照。`*_raw_hand/shape/...` 保留网络原始参数。不要把 filtered_joints 与 raw_global_rotations 混成同一姿态。

当前 `compact_result=true` 的固定主字段只有 `vertices,faces,joints,edges,camera_matrix`；滤波开启时还可能保留 `*_raw_*`。它没有选用结果的每侧 `global_rotations/joint_names/sides`。即使看到部分 raw 字段，也不能认为完整姿态协议已经存在。

建议先 `compact_result=false`；后续若压缩传输，在下游服务侧生成只含 `side,names,edges,joints,rotations,frame_index,timestamp,valid,filter_options` 的骨架小包。该小包目前是建议，尚未实现。

## 8. 左右手、置信度与身份连续性

- `hand_selection=both`：按检测标签+框连续性分配左右槽，每侧最多一只；不是多人多手追踪。
- `right/left`：用户硬指定解剖学右/左手；忽略 YOLO 原始左右分类，只用框和置信度竞争一个槽，最多一只手。避免把右手误标左手导致消失/重复。
- confidence 默认0.4，允许0.05–0.95；提高会增加漏检风险。每个框的 score 是**检测器置信度**，不是3D姿态质量。
- report.hands 保留 `side,raw_side,forced_side（单手）,bbox,score,track_id,identity_source`。track_id 实际是 left/right 槽名称，不是全局稳定人员ID。
- 单手模式有多个真实候选时，仍可能选错人；仅把用户的类别假设套到选中框，不能保证是同一只物理手。
- 没有目标时输出空场景，清掉旧网格。下游也必须主动发出“本帧无有效手”状态，不要持续沿用旧姿态而不标记过期。

## 9. One Euro 的实际范围

`one_euro={pose,position,min_cutoff,beta,d_cutoff}`，默认两个开关都是false；建议首先只打开 pose，初值 `1.5 / 0.5 / 1.0`。

- 手指：滤波网络最终手部潜在系数，再走原有映射生成骨架/表面，不逐顶点拉平网格。
- 手掌：SO(3) 最短旋转差 + 自适应截止频率；不直接平均欧拉角或矩阵元素。
- position 可选滤波规范相机平移；关闭时保留原始相机平移。
- pose 开启会多运行一次最终 MHR（可复用紧凑表面、编译与Graph），不会重新跑 DINO、检测器或解码器。
- 左右手各维护历史；新stream、换侧/内参/尺寸/参数、禁用、缺失、非递增时间、间隔>0.5秒、框大跳变或异常会重置。
- 调低 min_cutoff 更稳但更滞后；增大 β 更跟手；d_cutoff 初期保持1。存在因果响应滞后，没有“真正零延迟”。形状/尺度仍取原始预测，不能保证消除所有外形抖动。

`report.one_euro` 含 `processing_ms/history_frames`，以及平铺的 `pose/position/min_cutoff/beta/d_cutoff` 参数字段；不在原有 reconstruct_ms 中冒充网络耗时。下游也有滤波器时，先明确哪些量在哪一级平滑，避免无意叠加两次造成明显拖尾。

## 10. AmazingHand 下游映射建议（尚未实现）

第一阶段只接一个指定侧。优先保留已有下游“姿态来源→特征→机构映射→标定刻度→控制”的分层，只替换姿态来源。

1. 建立 `SamHandSource`：工作进程状态、最新帧、取完整结果、每侧名称/边、时间戳、valid/error。
2. 建立独立 `SamHandFeatures`：根据名称找腕和各指链，计算掌坐标系、指节夹角或父相对旋转。固定同一静止参考姿态消除绑定方向差异。
3. 建立机构相关映射：人体多关节可能需要拟合到更少驱动自由度；弯曲和外展/对掌各取哪些特征应由下游机构确认。当前文档不假定该台 AmazingHand 有几路驱动。
4. 使用下游已有每舵机的开/合端点、正反向、非线性标定、限幅及发送接口，不把人体角度直接当 PWM/编码器刻度。
5. 先显示原始/滤波特征与映射后的目标刻度，暂不发送；再由下游按原控制流程接硬件。

三点弯曲特征示意：`flexion=π-acos(dot(a-b,c-b)/(|a-b||c-b|))`，在接近直线时约0；需检查退化长度、clamp到[-1,1]，再转度数。此公式不识别弯曲符号，也不能单独测轴向旋转/拇指对掌；不要把每个相邻节点都视为独立机构转轴。

目标传输建议字段（**新建接口建议，不是当前已实现协议**）：

```json
{"schema":"sam_hand_source_v1","frame_index":123,"source_monotonic":123456.789,
 "inference_finished_monotonic":123456.880,"side":"right","valid":true,
 "joint_names":["r_wrist","..."],"joints_wrist_m":[[0,0,0],[0.01,0,0]],
 "global_rotations_cv":[[[1,0,0],[0,1,0],[0,0,1]]],
 "detector_score":0.92,"pose_filtered":true}
```

示例数组仅示意结构，实际长度必须与23节点一致。丢手/错误应发valid=false或明确清空状态。下游保留其既有超时、停止、速率/限幅和连接管理逻辑；本文没有改变已有设备边界。

## 11. 调度、相机与计时

- `camera_worker` 使用单独 Orbbec 环境；socket 消息是4字节网络序长度+JSON。`CameraClient/CameraReceiver` 已处理拆包、后台解码与最新帧。
- 实时开始时请求 `preview_rate=30`，结束恢复15；这是预览发送目标，不等于重建或屏幕FPS。
- 只允许一个推理请求在途；繁忙期间覆盖最新相机帧，不建立长队。丢弃重复帧和接收龄>0.5秒的旧帧。
- GUI 实现会在隐藏/最小化时不提交；下游独立控制源不应直接复用这一 UI 可见性判断。应由显式“启动/停止跟随”管理。
- 数据读取和发送控制目标应在推理完成后进行，不等待OpenGL frameSwapped，也不必导出PLY、视频、截图。渲染仍可低频、独立刷新。
- 当前单手/双手实时模板以“一个人在画面中”为假设。多人、同一侧多手、严重遮挡没有可靠3D置信度或全局重识别。

计时定义：`detect_ms` 包含检测预处理/网络/后处理；`reconstruct_ms` 是共享批量的裁剪+H2D+网络+MHR+同步，不含D2H、后处理、导出、显示；`one_euro.processing_ms` 单独包含可选滤波和额外MHR；`worker_ms` 才包含工作进程本次完整处理。GPU子阶段是嵌套的，不能和reconstruct_ms重复相加。

此前 RTX4060、INT8 512、3次中间预测、其他手部加速开启，录制帧固定框测试：滤波+额外MHR约单手4.64ms、双手4.89ms，完整 `predict_many` wall均值约72.21/119.15ms（不含检测/显示）。这些不是下游舵机跟随延迟。GUI已有“帧到显示”约定从主机SDK接收到OpenGL交换，不包含曝光/USB前段，更不是机器人动作完成时间。

## 12. 数据落盘与验证入口

| 位置 | 内容 |
| --- | --- |
| `/home/hzb/Work/sam3d-body-deploy/outputs/hands/` | 单图输入、裁剪、NPZ、PLY和请求/报告 |
| `/home/hzb/Work/sam3d-body-deploy/outputs/hand-sequences/`（可配置） | 连续序列帧结果、计时、叠加视频 |
| `/home/hzb/Work/sam3d-body-deploy/outputs/hand-streams/` | 实时会话摘要；不是完整RGB或逐帧网格归档 |
| `/home/hzb/Work/sam3d-body-deploy/outputs/sessions/hand-*.log` | GUI拉起的手部worker日志 |
| `/tmp/sam3d-hand-live-*/` | 当前GUI临时输入/scene，结束删除 |
| `/run/media/hzb/Data/experiments/hand-one-euro-20261113/` | 单/双手真实GPU滤波检查NPZ与validation.json |
| `/run/media/hzb/Data/experiments/hand-euro-gui-20261113/` | 录制帧GUI实时调参验证 |

关键测试均位于 `/home/hzb/Work/sam3d-body-deploy/tests/`：

- `test_hand_geometry.py`：拓扑/镜像/投影；`test_hand_automatic.py`：强制侧/重复框/置信度。
- `test_hand_config.py`、`test_hand_decoder.py`、`test_hand_feedback.py`：启动配置和调度边界。
- `test_hand_stream.py`：实时协议、临时文件限制及精简包。
- `test_hand_temporal.py`：旋转±180度短路径、噪声抑制、独立历史与重置。
- `hand_temporal_integration.py`：真实GPU、MHR、原始结果保留；需要本地资产。
- `hand_startup_warmup_gui.py`：不接物理相机，已有录制帧模拟实时帧源，可测试预热和开关切换。
- `camera_receiver_smoke.py`：socket/接收线程测试，物理相机测试是可选分支。

CPU/协议回归命令：

```bash
cd /home/hzb/Work/sam3d-body-deploy
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /run/media/hzb/Data/conda-envs/fast-sam3d-body/bin/python3.12 -m pytest \
  tests/test_hand_config.py tests/test_hand_geometry.py tests/test_hand_automatic.py \
  tests/test_hand_stream.py tests/test_hand_temporal.py -q
```

推荐验收顺序：已有JPEG→完整NPZ解析→录制帧持续输入→真实相机仅显示控制特征→接入下游标定映射→按已有下游流程验证实机。检查强制左右侧、丢手/重入、帧过期、worker崩溃、镜像预览以及滤波拖尾。除已有上游测试外，这些下游验收仍待完成。

## 13. 可直接发给下游 agent 的任务提示词

> 请接入 SAM 3D Body 作为 AmazingHand 的新姿态来源。先阅读 `/home/hzb/Work/sam3d-body-deploy/docs/HANDOFF_AMAZINGHAND.md`，再按文档顺序阅读对应源码。上游当前为main/d31dbc0，第三方checkout只读。先分析下游已有控制架构和标定接口，复用现有舵机范围、方向、发送与停止机制；不重新猜测硬件参数。优先使用独立常驻 `launch_hand.py/hand_worker.py`，先以 `hand_stream + compact_result=false` 获取按侧分组的23个MHR节点、名称、旋转与时间戳。先接录制数据，再接Gemini实时RGB；只维护最新帧和一个在途请求。先做仅显示的特征/目标刻度对照，再按下游原有流程打通硬件。请区分已有上游功能与需要新增的骨架传输、机构映射，不把MHR参数/检测置信度直接当舵机刻度/三维精度。保持上游环境隔离与GPU互斥，避免同时启动另一份Studio模型。给出可回退的改动和分阶段验证证据。

## 14. 来源、许可与本次交付范围

实现建立在 SAM 3D Body、Fast 社区加速经验、可选 Embedl 量化编码器、WiLoR检测器与Ultralytics之上。源码自身标注、上游派生部分及模型授权应分别保留；复制文件到下游不等于获得权重授权。参见 `/home/hzb/Work/sam3d-body-deploy/THIRD_PARTY_NOTICES.md`、`docs/provenance.json`、`docs/LICENSE_AUDIT.md`。

本次按当前源码、配置解析和既有NPZ进行静态交接核对；没有在本次重跑模型、打开相机、连接或驱动AmazingHand。现有GPU/GUI录制流测试证明的是上游链路，不代表灵巧手映射精度或真实控制稳定性。
