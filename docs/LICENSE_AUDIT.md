# 许可证与代码来源审查（2026-11-05）

## 结论与范围

本次完成的是**发布源码的工程来源审查及声明修正**，不是整个运行栈的法律兼容性批准。原创 Studio 代码默认 GPL-3.0-only；明确的上游改写另有来源/许可标记。不能将所有文件、模型和依赖统称为“统一 GPL”或“Fast MIT”。

基线是本仓库 `e4efd99aaee9d7248fd5e91382aae96a225ab60f` 的 78 个已跟踪文件：53 个 Python、7 个 Shell，以及文档、配置、图标、许可证等。本次新增的审查文件也纳入 [逐文件清单](provenance.json)。清单的 `basis` 给出每个文件的初步来源判断；不是仅根据 import 来判定衍生关系。

对全部发布文件建立清单并检查文件头、导入/动态加载、资源来源；对加载器、渲染、MHR 实验室、SMPL 转换、网格法线等重点路径逐段比较。另用去掉注释/字符串的 50-token 连续窗口筛查已固定上游中的 Python：SAM 66 个、Fast 109 个、ComfyUI 1028 个、SAM3 172 个、DINOv3 173 个、pyqtgraph 351 个。筛查发现的较长匹配主要是父节点数字表和通用字典表达式，不能据此判定复制；人工比较仍识别出经过改写的加载/渲染流程。未匹配不能证明独立创作，也不能代替完整法律审查。未检查每个传递依赖的全部源文件、训练数据权利或所有历史提交。

## 逐项发现与处理

| 优先级 / 状态 | 发现和依据 | 已处理 / 尚需处理 |
| --- | --- | --- |
| 高 / 未决 | GPLv3 的整体作品要求及不得追加限制，与 SAM/DINO 自定义条款对衍生材料、用途和逆向等的要求存在兼容性疑问。GUI 中也有同进程读取 SAM 元数据的路径；进程边界不是法律结论。 | 不宣称整个组合已获兼容性确认。公开声明审查限制；在打包完整运行栈、二进制或商业交付前，需要明确组合边界、获得所需授权或由专业人士核验。 |
| 高 / 未决 | Fast 根目录 MIT 不能覆盖其继承的全部材料。例如 `sam_3d_body/models/heads/mhr_head.py` 仍有 Meta 版权；DINOv3 根目录是自定义 DINOv3 License。 | 分别列出上游条款；不因镜像开放下载或根目录 MIT 就认定权重、转换/平滑权重、TensorRT 引擎具有再分发权。资产需逐项确认来源、版本及授权。 |
| 中 / 已修正 | `resident_worker.py::Engine.__init__` 与 `run_comfy.py` 中键名改写、去掉 hand 分类头、模型装载、CoreModelPatcher 参数链，改写自 ComfyUI 的 `SAM3DBody_Loader.execute`。 | 文件头明确 ComfyUI 来源、固定版本、修改内容，GPL 默认保留。此前“没有复制上游源码”的笼统声明已纠正。 |
| 中 / 已修正；组合问题仍未决 | `calibrated_visualization.py` 的材质、坐标旋转、相机位移符号、灯光、RGBA 合成及四联图布局，改写自官方 `renderer.py::Renderer.__call__` 与 `tools/vis_utils.py::visualize_sample_together`。 | 保留 Meta 版权，标记 `LicenseRef-SAM-3D-Body`，附完整 SAM License；不再用默认 GPL 覆盖该文件。修改点是精确内参、资源清理、单人布局。这不解决它与 GPL 路径组合的问题。 |
| 中 / 已修正 | `smpl_stage.py` 采用 Fast 的采样→重心居中→单视角转换及输出约定，并动态加载其网络。`gui/mesh_geometry.py` 沿用 pyqtgraph 的 MeshData 法线缓存/索引接口。 | 分别补 Fast MIT、pyqtgraph MIT 完整版权和许可文本及文件头说明；原创改写仍按 GPL 发布。采样映射、SMPL 数据、学习权重没有纳入发布。 |
| 中 / 未决 | 可选 YOLO 采用 AGPL-3.0 或商业许可；PyQt5 为 GPLv3/商业双许可。只公开 Studio 源码不等于满足组合发行、修改源码或网络服务的全部义务。 | 声明实际运行依赖；未制作闭源分发包、统一模型包或在线推理服务。若提供这些形态需重新核查对应源码、许可及服务义务。 |
| 中 / 未决 | SMPL 模型有独立用途/再分发限制；MHR 仓库代码 Apache-2.0 不能直接证明本机 SAM 资产包内 TorchScript/映射的具体授权。 | 所有原始及转换模型、映射、引擎和历史真人数据继续排除；资产下载者需留存具体发布来源及授权依据。科研 demo 不等于已取得商业许可。 |
| 低 / 已修正 | 发布文件缺少逐文件许可标识，也缺少完整来源清单。 | Python/Shell 文件加 SPDX 与版权说明，清单覆盖所有发布文件，CI 检查清单和许可快照一致性。 |

## MHR 实验室与 SMPL：没有把“调用接口”误判为“复制网络”

- `mhr_lab_engine.py` 调用加载资产的 `character_torch` 变换/蒙皮方法，读取名称、父节点、限制并覆写局部变换。没有在 Studio 分发 MHR 蒙皮/校正网络实现，也没有提取并发布权重、TorchScript 反编译代码或计算图。官方 MHR 源码本身公开 `character_torch` 路径，不能仅因访问这一属性就断定违反逆向条款。但本机实际资产的许可和具体使用方式仍需核验；本次不作最终法律判断。
- `recover_fast_parameters` 读取 checkpoint 的均值/分量和手部索引，并调用 Fast 包中的手部映射函数。参数维数、索引及数值变换属于兼容性接口依据，本次未据此将整个适配器改标为 SAM。实际导入的上游函数及 checkpoint 保留原条款。
- `smpl_stage.py` 不包含转换网络或 SmootherMLP 架构实现；这些由 Fast 代码动态加载。其 FK 是 24 关节位置计算，未发布 SMPL 模型数据；通用父子树、Gram–Schmidt 或数学公式不单独作为复制证据。
- `pose_data.py` 运行时从单独获取的官方源码读取关键点名称，未内置完整上游 metadata 模块。这个同进程数据读取仍应计入组合边界审查。

## 依赖与发布边界

| 路径 / 组件 | 发布内容 | 运行时来源与边界 |
| --- | --- | --- |
| Qt GUI、实时调度、历史对照、RGB-D 评分、录制、配置 | Studio 原创集成与通用计算，逐文件清单标记初步判断 | PyQt5 GPL/商业、pyqtgraph MIT、NumPy/SciPy/OpenCV/trimesh 等各自条款；不是完整传递依赖审计 |
| 标准推理 | 改写的 ComfyUI 加载器和调用协议 | 单独获取的 ComfyUI；其中 MHR rig 文件另有 Meta/Apache 来源声明，不能只看根 GPL |
| Fast 推理 / 引擎构建 | 参数配置、资源校验、调度、导出/编译入口适配 | Fast、SAM、DINO、YOLO、TensorRT 各自代码/模型条款；未把转换器实现复制入 Studio |
| SAM3 分割 | 点提示、进程协议与显存互斥适配 | 单独获取的 SAM3 源码与授权模型 |
| Gemini 相机采集 | Studio 适配器；参考同一作者的相机示例，无完整 SDK/示例包拷贝 | Python wrapper 上游 Apache-2.0；安装的 SDK 二进制/固件条款另查 |
| MHR / SMPL | 参数编辑/转换/关节计算接口 | 外部人体资产、学习权重、映射；未随源码发布 |
| 图片、点云、姿态历史、标定、日志、模型 | 不发布 | Git ignore 与 release guard 排除，导出/隐私权利由操作者另外确认 |

完整上游仓库不是内嵌分发；**局部改写已经在文件头承认并保留原条款**。没有更改、删除上游 clone，没有删除本地模型或用户记录，没有用切换许可证来掩盖组合问题。此次修正不重写历史；历史版本的声明不应当作完整授权证明。

## 建议的后续决策

1. 继续发布源码时，保留本报告、文件级例外与第三方通知，避免宣称“全栈已通过开源许可审查”。引用论文和致谢不能替代授权。
2. 在打包/商业化前，确认 GPL 与 SAM/DINO 组合和可选 AGPL 路径。需要授权时，获得相应权利人的许可；或在确认边界后替换/拆除相关实现。单独下载、分进程不自动消除衍生作品问题。
3. 单独建立私有资产来源台账：模型/映射/学习转换权重的发布版本、校验值、许可、取得授权记录；不要公开账号资料或受限文件。当前工程审查不等于权重授权审查通过。

## 核验与复查

```bash
python3 scripts/audit_release.py
python3 scripts/audit_provenance.py
python3 -m unittest discover -s tests -v
```

来源清单必须与 `git ls-files` 一致，因此新文件先加入 Git 暂存区再运行一致性检查。许可快照字节由 SHA-256 固定，避免今后升级依赖时悄悄沿用旧条款。CI 只核对完整性/一致性；不会自动判断著作权、许可兼容性或模型授权。源码变更以声明、注释、文档及该检查为主，没有改变推理算法或运行参数；本次无需加载 GPU 做新的速度或精度结论。

本次本地验证：87 个发布文件的范围/来源检查通过；54 个 Python 文件编译通过；6 项无模型移植检查、3 项人物选择检查、MHR 滤波/原始数据保留/导出检查通过；隔离设置的无模型 GUI 检查成功构建三个标签页，未启动相机或模型。新增来源检查器通过正常样本及缺失清单、额外清单、许可快照被改、源码头不一致四种失败样本。对基线的 53 个 Python 文件比较 AST 完全一致，7 个 Shell 文件去掉声明注释后命令完全一致。本次没有重新测试相机、模型推理、精度或 GPU 性能。

## 主要原始依据

- [SAM 3D Body LICENSE](https://github.com/facebookresearch/sam-3d-body/blob/b5c765a0d89d789985e186d396315e7590887b94/LICENSE)
- [SAM renderer.py](https://github.com/facebookresearch/sam-3d-body/blob/b5c765a0d89d789985e186d396315e7590887b94/sam_3d_body/visualization/renderer.py)
- [SAM vis_utils.py](https://github.com/facebookresearch/sam-3d-body/blob/b5c765a0d89d789985e186d396315e7590887b94/tools/vis_utils.py)
- [Fast LICENSE](https://github.com/yangtiming/Fast-SAM-3D-Body/blob/d72aa36913a1673ace029d345f437561a85ec9e2/LICENSE)
- [Fast inherited MHR head](https://github.com/yangtiming/Fast-SAM-3D-Body/blob/d72aa36913a1673ace029d345f437561a85ec9e2/sam_3d_body/models/heads/mhr_head.py)
- [Fast conversion workflow](https://github.com/yangtiming/Fast-SAM-3D-Body/blob/d72aa36913a1673ace029d345f437561a85ec9e2/mhr2smpl/multi_view/infer_multiview.py)
- [ComfyUI loader](https://github.com/Comfy-Org/ComfyUI/blob/858fd58dcd47b1407a6778e0920f02a9b40fd538/comfy_extras/nodes_sam3d_body.py)
- [ComfyUI inherited MHR attribution](https://github.com/Comfy-Org/ComfyUI/blob/858fd58dcd47b1407a6778e0920f02a9b40fd538/comfy/ldm/sam3d_body/mhr/mhr_rig.py)
- [GPLv3 (sections 5 and 10)](https://github.com/Comfy-Org/ComfyUI/blob/858fd58dcd47b1407a6778e0920f02a9b40fd538/LICENSE)
- [DINOv3 LICENSE](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/LICENSE.md)
- [SAM3 LICENSE](https://github.com/facebookresearch/sam3/blob/0570b3a5be9c4e694f23d85232fb55f4a6f1f7fc/LICENSE)
- [MHR source / Apache-2.0](https://github.com/facebookresearch/MHR/blob/main/mhr/mhr.py)
- [MHR LICENSE](https://github.com/facebookresearch/MHR/blob/main/LICENSE)
- [SMPL model license](https://smpl.is.tue.mpg.de/modellicense.html)
- [YOLO LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)
- [PyQt licensing](https://www.riverbankcomputing.com/software/pyqt/)
- [Orbbec Python wrapper LICENSE](https://github.com/orbbec/pyorbbecsdk/blob/main/LICENSE)

SAM/Fast/ComfyUI/SAM3/DINO 来源固定在 `dependencies.json`，许可证全文快照见 [LICENSES](../LICENSES/README.md)。MHR、SDK 等未列入该固定源码清单的链接是审查日参考，不是对任意安装版本的授权认定。
