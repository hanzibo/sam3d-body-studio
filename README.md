# SAM 3D Body Studio

基于 **[Meta SAM 3D Body](https://github.com/facebookresearch/sam-3d-body)** 与 **[Fast SAM 3D Body](https://github.com/yangtiming/Fast-SAM-3D-Body)** 的 Ubuntu 桌面人体重建与 RGB-D 实验工具。

本项目的工作是桌面界面、相机接入、启动方案管理、可视化与实验对照；人体重建模型和 Fast 加速方法来自上游作者。这是独立社区项目，非 Meta 或 Fast 作者的官方产品。请同时保留并引用原项目，见 [引用](CITATIONS.bib) 和 [第三方说明](THIRD_PARTY_NOTICES.md)。

## 功能

- 图片上传、Gemini 336L 相机预览／倒计时拍摄／实时流推理；网格先显示，图像对照后台生成。
- 标准路径（ComfyUI 的 SAM 3D Body 适配）与 Fast 路径切换；预加载、预热、显存释放。
- Fast 身体／手部精修、384／512、骨干及检测器 TensorRT、解码器／MHR／中间预测编译开关，保存启动方案与实际配置。
- 可旋转的 3D 预览、网格投影叠加、MHR 骨架、几何关节角、动作录制导出；原始与滤波数据分别保存。
- 可选 MHR→SMPL 24 关节转换、MLP 平滑、MHR／SMPL One Euro 位置滤波。
- 独立 MHR 参数实验室：导入最近预测，或创建默认人体，修改原生参数和关节观察局部变化。
- RGB-D 几何验证：相机内参与深度生成点云，SAM3 前／背景点分割，网格与可见表面对照。
- 多姿态采集、同一组输入跨方案运行、历史详细配置／删除／两项对照及联合统计。
- SAM3 与人体模型进程互斥，避免两套模型同时占用显存（不管理其他应用）。

## 开始使用

支持环境：Ubuntu、NVIDIA CUDA GPU；本地已在 RTX 4060 8GB 上使用。其他平台／显卡需要自行验证。模型不包含在仓库里。

```bash
git clone https://github.com/hanzibo/sam3d-body-studio.git
cd sam3d-body-studio
cp config/paths.example.json config/local.json
# 按 docs/INSTALL.md 创建环境、获取源码／获授权的模型，填写本机路径
python3 studio_paths.py --doctor
./launch_gui.sh
```

完整步骤见 **[安装与模型准备](docs/INSTALL.md)**、[界面使用说明](gui/使用说明.md)、[RGB-D 实验说明](gui/RGB-D几何验证说明.md)。安装桌面图标：

```bash
python3 scripts/install_desktop.py --desktop
```

默认“演示最快”需要先构建并登记 384 TensorRT 引擎。没有引擎时，在预热前选择 Fast 512、骨干／检测器 PyTorch，并关闭三个编译开关进行初次验证，或者使用标准版；不静默下载或切换执行路径。

## 目录与数据

| 路径 | 用途 | 是否发布 |
| --- | --- | --- |
| `gui/`、根目录 Python／启动脚本 | 界面、隔离推理、采集和评估 | 是 |
| `docs/`、`requirements/`、`dependencies.json`、`scripts/`、`tests/` | 文档、依赖版本、部署及验证 | 是 |
| `third_party/` | 按固定提交获取上游源码 | 否 |
| `models/`、`assets/smpl/` | 权重、TensorRT 引擎、授权 SMPL 模型 | 否 |
| `config/local*.json` | 本机解释器／源码／资产路径、引擎校验记录 | 否 |
| `config/gemini336l_rgb.json` | 可选固定相机标定；照片元数据优先 | 否 |
| `outputs/`、`profiles/`、缓存、日志 | 照片、RGB-D、历史、录制、用户方案 | 否 |

应用不会上传图片。运行产物默认保留在本地 `outputs/`；手动导出到仓库外的数据需自行管理。不要用 `git add -f` 添加模型或真人数据。

## 验证与范围

```bash
python3 -m unittest discover -s tests -v
# 以下两个测试需要 requirements/gui.txt 中的数值依赖
python test_fast_selection.py
python mhr_filter_test.py
# 可选 RGB-D 渲染检查：还需 pyrender 和可用的 EGL/OpenGL
python rgbd_geometry_test.py
```

本次检查范围见 [验证记录](docs/VALIDATION.md)。持续集成检查源码编译与无模型测试；真实推理使用 `python scripts/check_inference.py --backend standard` 或 `--backend fast`，需要本地模型和 CUDA。GUI 与相机交互说明见安装文档。

384、减少中间预测、身体模式可能影响精度；RGB-D 评分仅是可见表面一致性，不是真值姿态或完整人体精度。计时须区分加载、预热、模型推理、传输与渲染，本仓库不承诺通用帧率。关节角为几何角度，滤波仅处理位置，不能代替有效旋转或机器人关节指令；本应用不控制机器人。

## 致谢与引用

- [SAM 3D Body 项目](https://github.com/facebookresearch/sam-3d-body)，[论文](https://arxiv.org/abs/2602.15989)：人体重建模型与基本推理流程。
- [Fast SAM 3D Body 项目](https://github.com/yangtiming/Fast-SAM-3D-Body)，[主页](https://yangtiming.github.io/Fast-SAM-3D-Body-Page/)，[论文](https://arxiv.org/abs/2603.15603)：加速推理及 MHR→SMPL／平滑模块。
- [Momentum Human Rig (MHR)](https://github.com/facebookresearch/MHR)：参数化人体表示；[ComfyUI](https://github.com/Comfy-Org/ComfyUI)：标准推理适配；[SAM3](https://github.com/facebookresearch/sam3)：可选人体分割。
- [Orbbec SDK](https://github.com/orbbec/pyorbbecsdk) 与 [Gemini 336L Python 示例](https://github.com/hanzibo/orbbec-gemini-336l-python-examples)：相机采集接口与实践参考。

原论文 BibTeX 已按两个上游 README 整理至 [CITATIONS.bib](CITATIONS.bib)。研究中使用本工具时请引用原方法，而非将 Studio 的集成工作视为原模型或加速算法贡献。

## 许可证

本仓库原创的 Studio 集成代码默认采用 **GPL-3.0-only**，见 [LICENSE](LICENSE)，文件头另有声明的除外。`calibrated_visualization.py` 改写自官方渲染代码，按 **SAM License** 标注；ComfyUI 加载器及 Fast 转换流程的改写来源也已单独声明。第三方权重、数据和人体资产不随本仓库发布，本许可证不重新授权这些内容。

**许可审查尚有未决项：** GPL、SAM／DINOv3 自定义许可及可选 AGPL 组件的组合分发尚未确认兼容；分进程、单独下载和补齐署名不能自动解决这些问题。不要将整个运行栈理解为已确认可统一 GPL 发布或商业交付。逐项依据、已完成的修正和后续需要确认的授权，见 [许可证与来源审查](docs/LICENSE_AUDIT.md)、[逐文件清单](docs/provenance.json) 和 [第三方说明](THIRD_PARTY_NOTICES.md)。
