# Kinect v2 相机备选

2026-11-15：Studio 新增 Kinect v2，保留 Gemini 336L。第一标签页、手部实验实时区、RGB-D 几何验证的相机来源下拉框同步；先关闭相机/停止推理再切换。

- 彩色预览/倒计时照片/身体与手部实时源：1920×1080 BGR，工厂彩色内参随 ready 和 camera.json 传递。SDK 在独立 Kinect Conda 环境运行，不导入 GUI 或模型环境。
- RGB-D 验证/多姿态采集：SDK 将彩色注册到去畸变深度平面，输出 512×424、深度相机内参、米制点云及 Studio 样本格式。不是原始1080p彩色坐标系。该模式的 rgb_raw.png 也是注册彩色，camera.json 明确记录。
- Kinect 的配准多项式不当作刚体外参，也不复用 Gemini 标定。RGB 实时和 RGB-D 样本的内参不可混用。
- 当前实机 RGB/深度配对时间戳差约56ms，默认20ms门限会拒绝保存。RGB-D页可显式放宽最大时间差；80ms门限仅用于本次静止场景连通测试，运动准确性未验证，不表示严格同步。建议静止后拍摄。超限仍报错。
- 独立 RGB-D 视频录制工具暂仍是 Gemini 链路，本次没有扩展它；身体/手部推理本身不因此增加深度融合。

## 配置与实现

本机 config/local.json 新增 kinect_python=/home/hzb/miniconda3/envs/kinect2/bin/python；可移植模板见 config/paths.example.json。需要该环境已安装 /home/hzb/Work/kinect_ws 的 kinect2 SDK及OpenCV。无需复制/改动SDK源码。

- studio_paths.py：camera --backend kinect 选择独立解释器/worker。
- kinect_camera_worker.py：既有 Unix socket 协议的 Kinect 实现，支持 ready/frame/capture/countdown/photo/cancel/stop/error。
- kinect_rgbd_capture.py：RGB-D采样、预览、时间差门限及配准导出，复用 @@RGBD 事件格式。
- gui/camera_client.py、gui/app.py：相机选择、通用元数据和进程生命周期。
- gui/hand_live.py、gui/rgbd_validation.py：同步相机入口和RGB-D路由。

## 本次验证

- 真实 Kinect 由 Studio launcher 启动：收到 ready、彩色帧和倒计时 photo，保存1920×1080照片及工厂内参。
- 实际 MainWindow / CameraClient / CameraReceiver 在 Qt offscreen 下连接物理相机：收到1920×1080 QImage，三个选择框同步，正常关闭释放设备。未做人工桌面布局观感验收。
- 默认RGB-D20ms门限：实测时间差超限后拒绝保存，符合预期。
- 放宽到80ms进行静止场景格式验证：实际时间差55.875ms，导出424×512深度/彩色和197937个有效深度点；导出点云投影回像素的最大数值误差0.000031px。此检查验证坐标计算自洽，不是物理精度测量。
- 本次未加载GPU人体模型；确认的是输入接口兼容与真实相机采集，不是Kinect场景人体/手部重建准确性或FPS。

测试临时文件在 /tmp/studio-kinect-*，不提交原始图像、日志或内参快照。
