# 已标定的电脑前置摄像头

2026-11-16：在身体页及手部实验实时区选择“电脑前置摄像头（已标定）”，使用相同相机协议进行预览、倒计时照片和连续 RGB 推理。相机来源在三个页面同步。普通摄像头无深度，RGB-D采集/预览按钮禁用；仍可打开已有RGB-D样本。

本机 config/local.json 中的 webcam_calibration 指向用户指定的 AprilTag calibration.json；模板路径为 config/local.webcam.json（Git忽略）。该JSON须包含 schema=apriltag-webcam-intrinsics-v1、V4L2 device 路径、color_intrinsic 和 BROWN_CONRADY color_distortion。标定图像、内参数值不纳入版本库。

worker 固定请求标定文件的分辨率，实机当前为1280×720；使用稳定的 /dev/v4l/by-id 设备路径，禁用设备支持的自动对焦，与标定采集方式一致。若图像实际尺寸不同则报错，不缩放套用内参。请保持标定时的镜头/焦距设置。

图像先通过预计算 OpenCV remap 去畸变，保留完整画幅，使用同一 K 作为输出内参，不裁切、不镜像。边缘可能有黑色无效区域。预览、PNG照片和推理输入使用同样的校正；不应再次去畸变。camera.json 中保存原始畸变、校正标志、有效零畸变、标定文件SHA256和验证指标。身体与手部读取 intrinsics；原始K保持有效。选择前置摄像头时自动开启身体页“使用相机真实内参”，用户之后仍可手动关闭进行对照。

启动器 --launch camera --backend webcam 使用已有 camera_python 环境中的 OpenCV，不引入相机SDK到GUI/模型进程。实现见 webcam_camera_worker.py、studio_paths.py、gui/app.py、gui/rgbd_validation.py；手部复用现有共享 CameraClient，无需单独打开设备。

V4L2默认没有暴露传感器时间戳，received_monotonic 记录 cap.read 完成时的主机时间，device_timestamp_us=null。界面供帧目标15/30FPS不等于实际采集/推理FPS。

本次已验证真实设备经 Qt MainWindow/CameraClient/CameraReceiver 返回1280×720图像与标定内参、正常关闭释放，以及RGB-D入口禁用。另验证倒计时照片和读取内参。测试采用Qt offscreen，没有进行桌面布局人工验收；未加载GPU模型，尚不包含该摄像头下的人体/手部重建准确性测试。
