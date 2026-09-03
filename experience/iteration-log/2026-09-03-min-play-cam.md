# 2026-09-03 最低可玩门槛：笔记本摄像头 vs USB cam

## 假设

用户侧只要 **IrisGreen（蓝牙）+ 笔记本自带摄像头** 就能玩闭环观察，不必再买 USB cam。

## 本机探测（开发工作站）

- 机型：MSI MS-7B89，`PCSystemType=1` **台式机**
- 系统成像设备：仅 `Microsoft LifeCam HD-3000`（USB）
- OpenCV `cam#0` 1280×720 MJPG 出帧成功（mean≈131）
- 无笔记本前置/后置摄像头可测

## 软件改动

`irisloop/camera.py`：打开失败时从强制 MJPG 回退到摄像头默认格式（内置摄像头常见 YUY2/NV12）。

探测脚本：`python tools/min_play_test.py`；真机投射加 `--ble --group 1`。

## 下一步

换一台带摄像头的笔记本，把镜头对准投影面再跑同一脚本。普通笔记本没有后置镜头：拍桌面可把屏幕合到 20–40°，拍墙面用二合一帐篷模式。
