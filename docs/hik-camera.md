# 海康工业相机（MV-CA013-A0UM）接入记录

标定实验用相机。2026-10-10 完成 MVS SDK 安装与 Python 链路 bring-up，全部操作验证通过。

## 硬件

| 项目 | 值 |
|---|---|
| 型号 | MV-CA013-A0UM（M 后缀 = 黑白） |
| 序列号 | K65564865 |
| 接口 | USB3（USB3 Vision 协议，**非 UVC**，OpenCV 打不开，必须走 MVS SDK） |
| 分辨率 | 1280 × 1024 |
| 像素格式 | Mono8（0x01080001） |
| 实测帧率 | 曝光 5ms 时 88.5 fps；曝光 30ms 时 27.9 fps（受曝光时长限制） |
| 曝光 | 手动设微秒，已验证 5000–30000us，回读一致 |
| 增益 | 手动设 dB（相机会取整到最近档位，如设 10 回读 10.02） |

同一台电脑还枚举到 3 台 GigE 设备（ainstec 3D 相机 PRO-WR / PRO-S-Enhanced / PRO-L，网络不可达，与本实验无关）。代码默认自动选第一台 USB3 相机。

## 软件环境

- MVS SDK：`MVS_Win_STD_5.1.0_260909`（安装包在 `~/Downloads`，静默安装命令 `MVS_STD_5.1.0_260909.exe /S`，需经 cmd 调用）
- 安装目录：`C:\Program Files (x86)\MVS`
- Python 绑定：`C:\Program Files (x86)\MVS\Development\Samples\Python\MvImport\`（非 pip 包，注入 sys.path 使用）
- C 运行时 DLL：`C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64\`（**不在** MVS 安装目录下，导入绑定前需加入 PATH，hik_camera.py 已自动处理）

## 代码

- 封装：`irisloop/hik_camera.py`（`HikCamera`，接口对齐 `camera.py` 的 `UsbCamera`：open/read/release/info，另有 set_exposure_us / set_gain / grab_average / measure_fps）
- 冒烟测试：`tools/hik_smoke.py`

```bash
python tools/hik_smoke.py                          # 枚举+抓帧存 tmp/hik_smoke.png
python tools/hik_smoke.py --exposure 30000 --gain 10 --fps
python tools/hik_smoke.py --serial K65564865       # 按序列号指定相机
```

## 踩坑记录

1. 枚举序号 0 可能是 GigE 设备（本机 3 台 3D 相机），对其 OpenDevice 报 `0x80000201 MV_E_INVALID_ADDRESS`——操作 USB3 相机要用 `--serial` 或让工具自动选。
2. MVS 5.1.0 的 Python 绑定结构与旧版不同：没有 `MvCameraControl_header.py`，改为 `CameraParams_const.py / CameraParams_header.py / PixelType_header.py`。
3. 设备信息结构体字段是 `SpecialInfo`（大写 S）。
4. 绑定里无 `MV_TRIGGER_MODE_OFF` 常量，关触发直接传 0。
5. 解压出的安装 exe 在 Git Bash 里缺执行权限，需 `cmd //c` 方式启动。

## 下一步

逆相机模型联合标定（IrisGreen 激光投影 ↔ 本相机）：打印棋盘格 → 相机内参 → 投影点阵采集 → 投影仪内外参。
