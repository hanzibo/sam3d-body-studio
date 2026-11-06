# 单图 SAM3 掩膜提示实验

第一页新增“**SAM3 点提示 / 掩膜提示实验**”。手动人物框仍可使用。该功能使用网络的独立掩膜提示接口，**保留原始 RGB 背景**，不是把背景抹黑。

1. 在人体模型尚未启动时，点击“RGB-D 倒计时拍照”，或者打开已有图片 / RGB-D 样本中的 `rgb.png`。RGB-D 按第一页延时设置采集，显示倒计时和彩色预览，保存配准深度、点云和内参；此入口不会拍完自动重建。
2. 勾选“点提示模式”：左键添加人体前景点，右键添加背景点。可撤销点；最多 128 点，至少一个前景点。
3. 点击“SAM3 分割”，等待候选掩膜显示。候选仅在 SAM3 **进程完全退出**后应用。
4. 选择候选；拖动滑杆。负值腐蚀、正值膨胀，范围 −40～+40 **原图像素半径**，采用椭圆结构元。0 为原始掩膜。绿色半透明区域是实际提示区域。腐蚀为空时拒绝重建。
5. 应用并预热人体模型，然后重建。关闭“向人体网络传入掩膜提示”可做无提示对照；原始图像和人物框仍保留。CPU 上的掩膜形态调整和候选选择不需重新加载人体模型。

SAM3 需要单独获取的权重和独立环境，见 [安装说明](INSTALL.md)。人体模型加载、预热、已就绪或正在退出时，“SAM3 分割”禁用；不会自动释放人体模型。MHR 实验室的 GPU 模型也必须先释放。分割期间禁止加载人体模型、启动实时流或切换输入。取消后也等到进程真正退出才能加载模型。原有跨进程 GPU 文件锁提供第二道检查，包括其他 Studio 窗口。外部程序不受此锁管理。

## 对照方法与保存内容

- 首次选候选时取其原始掩膜外接矩形；膨胀 / 腐蚀不会重新计算人物框。可以切回手动框选，随后保持同一个框做对照。
- 标准版使用这一人物框裁剪，并传入独立掩膜；Fast 保留 YOLO 检测与手腕关键点流程，人物框用作单人选择 ROI，预计算掩膜通过估计器的 segmentor 接口注入。Fast 最终裁剪框仍以 YOLO 实际检测为准，完整报告会记录。
- 掩膜和 RGB 经过后端同一裁剪 / 缩放变换，输入网络前二值为 0/1；RGB 不变。TensorRT 骨干输入尺寸与引擎不变。
- `outputs/segmentation/<时间>/`：规范化 RGB、提示点请求、SAM3 候选及质量分数、日志。质量分数不是实测 IoU。
- `outputs/rgbd/<时间>/`：第一页面新增入口拍摄的完整 RGB-D 样本，可在几何验证页打开。
- `outputs/gui/<时间>/`：该次重建的 `input.png`、`mask_original.png`、`mask_prompt.png`、`request.json` 和 `inference.json`。报告保存开关、形态参数、候选、提示点、图像 / 掩膜 SHA256 与人物框。后台拒绝图像或掩膜已被修改、尺寸不一致、空掩膜及静态掩膜误用于实时流的请求。
- “运行历史 / 对照”新增掩膜半径与哈希摘要；完整报告可双击查看。“数据管理”按完整分割任务管理，正在使用的数据受保护，GPU 互斥锁只读。

比较顺序建议：同图、同框、同内参、同启动方案，分别关闭提示、原始掩膜、腐蚀、膨胀。使用真实人体表面评价时，评估用的人体掩膜应固定，不能跟随网络提示的膨胀 / 腐蚀而改变。第一页调整不会改写几何验证页的评估掩膜。静态图片对照不采用跨帧平滑。

## 验证

不加载模型的检查：

```bash
python -m unittest discover -s tests -p test_mask_prompt.py -v
QT_QPA_PLATFORM=offscreen python tests/image_segmentation_gui.py
```

真实掩膜通路对照，先完成一次 SAM3 分割并等待其进程退出，再**顺序**运行两个后端：

```bash
python scripts/check_mask_prompt.py --backend standard --segmentation outputs/segmentation/<时间>/sam3
python scripts/check_mask_prompt.py --backend fast --segmentation outputs/segmentation/<时间>/sam3
```

需使用带 NumPy / OpenCV 的 Studio 环境。脚本逐个测试关闭、原始、腐蚀 8 px、膨胀 8 px；检查有限网格、固定实际人物框，以及输出对掩膜的敏感性。它验证接口实际工作，不能据此宣称精度改善，也不是正式推理测速。掩膜提示改善或恶化必须结合固定 RGB-D 样本实测。

接口参考：[官方估计器](https://github.com/facebookresearch/sam-3d-body/blob/main/sam_3d_body/sam_3d_body_estimator.py)、[官方模型](https://github.com/facebookresearch/sam-3d-body/blob/main/sam_3d_body/models/meta_arch/sam3d_body.py)。没有修改上游克隆和权重。
