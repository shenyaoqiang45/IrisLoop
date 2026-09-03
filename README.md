# IrisLoop 🌈 虹环

**MEMS 激光投影自适应闭环系统** —— *生成 → 投射 → 观察 → 优化*

通过 BLE 控制 **IrisGreen（AINSTEC）** MEMS 激光投影仪（绿色激光，640×480 1bpp），
同时用本机摄像头（笔记本自带或 USB）采集物理表面上的实际投影效果，交由 AI 视觉分析迭代优化投影内容。

当前版本 v0.1.0：摄像头采集、BLE 协议栈、素材播放、投射+采集联动已打通；AI 闭环规划中。

## 系统组成

```
irisloop/            核心包
  camera.py          摄像头采集（MSMF 优先；MJPG 失败则回退默认格式，兼容笔记本内置）
  recorder.py        视频录制（按扩展名编码回退 mp4v/MJPG/XVID）
  capture.py/cli.py  采集主流程与命令行（python -m irisloop）
  projector.py       IrisGreen 设备档案（GATT UUID、MTU、设备识别）
  protocol.py        蓝牙协议 V1.9 帧构造/解析（type+len+data，0x80 响应）
  ble_client.py      BLE 客户端（命令通道 adb40003 write+indicate）
  image_pack.py      640×480 1bpp 图片打包（62B BMP 头 + MSB 位打包，真机已验证）
tools/               实机联调脚本（见下文「实机联调」）
doc/                 《Iris Green蓝牙通信协议_20260324_V1.9.xlsx》
test-data/           从设备拉取的真机 BMP 素材（组 1、组 20）
captures/            采集输出
```

## 安装

```bash
pip install -r requirements.txt   # opencv-python, numpy, bleak
```

## 使用

```bash
# 最低可玩门槛：探测本机摄像头（笔记本自带能否代替 USB cam）
python tools/min_play_test.py
# 真机：BLE 投射 + 本机摄像头
python tools/min_play_test.py --ble --group 1 --seconds 5

# 列出可用摄像头
python -m irisloop --list

# 默认采集（1280x720，手动按 q 停止，保存到 captures/）
python -m irisloop

# 录制 10 秒
python -m irisloop -t 10

# 后台录制，不显示预览窗
python -m irisloop --no-preview -t 10

# 指定输出路径与分辨率
python -m irisloop -o test.mp4 --width 640 --height 480
```

交互按键：`q` 或 `ESC` 停止，`p` 暂停/继续。

## 本机环境实测结论

- 摄像头 `Microsoft LifeCam HD-3000`，最高支持 **1280x720 @30fps**（请求 1080p 会回落）
- 可用编码：`mp4v`(.mp4)、`MJPG`/`XVID`(.avi)；`avc1`(H.264) 不可用（缺少 OpenH264）
- DSHOW 后端上报 `fps=0` 不可信，因此启动时会实测帧率并用于写入

## IrisGreen 投影仪（BLE）

设备档案（2026-09-01 实机确认）：Model `IrisGreen` / Manufacturer `AINSTEC`，
广播名前缀 `Iris-G`，协商 MTU 512（固件为批量图片传输做过优化）。

### GATT 通道（主服务 `adb401c0-...`）

| 特征 | 属性 | 用途 |
|---|---|---|
| `adb40001` | read | 设备支持特性 |
| `adb40002` | read+notify | 设备状态上报（8 字节自发状态帧，非命令响应） |
| `adb40003` | write+indicate | **命令通道**（响应经同通道 indicate 回执） |
| `adb40004` | write_no_rsp | 数据流通道 |
| `adb40005` | write | WiFi 配置（写 `0x02` 开 WiFi AP） |
| 服务 `adb40006-...-0001` | — | 文件传输服务：0002 开始 / 0003 数据 / 0004 结束 |

### 协议要点（V1.9，doc/ 下有 xlsx）

- 写帧：`type(1) + len(1) + data(n)`；响应帧：`80 + type + len + data`
- 播放素材组 `0xA0`：`loop + total(u16le) + interval(u16le) + group`（时间单位 100ms）
- 停止 `0xA1`；亮度 `0x11`；FOV `0x1F`（10×5 / 30×10 / 20×10）；畸变校正 `0x1C/0x1D`；镜像 `0x1E`；翻转 `0x12`
- 素材组 ID：1–24（开机/充电/动画/提示词等），背景组从 `0x33` 起；总张数 ≤ 200
- ⚠️ 文档勘误（代码已修正）：响应前缀文档写「80」示例实为 0x80 首字节；BMP 头文档写 124 字节实为 **62 字节**
- 设备无 BLE 读图命令；取图需开 WiFi AP 走 HTTP：`http://192.168.4.1/upload/group_<gid>_<n>.bmp`，完毕 `/switch/ble` 切回

### 实机联调（tools/）

```bash
# 扫描 / 探测 / 读信息
python tools/bt_scan_ble.py
python tools/bt_probe_device.py <addr>
python tools/bt_read_info.py <addr>

# 协议冒烟（只读，不改设备状态）
python tools/ble_smoke.py

# 监听状态帧节奏
python tools/ble_listen.py <addr> 12

# 播放 / 停止素材组
python tools/play_group.py --group 1
python tools/play_group.py --stop

# 投射 + 摄像头同步采集（亮度/亮区占比/LapVar 自动判读）
python tools/project_and_capture.py --group 1 --seconds 6

# 从设备拉取素材（WiFi HTTP 通道）
python tools/pull_images.py

# 生成并推送 640x480 1bpp 图片到设备（文件传输服务，调试用）
python tools/push_image.py --dry-run        # 只生成本地图
python tools/push_image.py --kind checker
```

## 当前进展

- ✅ 摄像头采集 / 录像 / 帧率校准
- ✅ BLE 连接、GATT 枚举、只读命令、素材组播放 / 停止
- ✅ 640×480 1bpp 图片打包（与 MATLAB `im2Bytes.m` 语义一致，真机 BMP 已验证）
- ✅ 投射 + 摄像头同步采集联动（闭环第一步）
- ✅ 设备素材拉取（WiFi HTTP）
- ✅ BLE 推图（逆向 iris-g-sdk 源码，修正通道映射+帧格式，实测落盘成功）
- ⚠️ AI 视觉分析闭环（loop.py + analyzer.py 已搭好，规则模式可跑，AI 模式待调通）

## 核心认知：闭环调内容，不调设备

设备出厂已校准好畸变/FOV/亮度等硬件参数。IrisLoop 闭环的迭代对象是**投影内容本身**：

```
AI 生成内容（编剧）→ BLE 投射 → 摄像头观察 → AI 看懂效果（观众）
  → AI 改进内容（导演）→ 再投射 → ... 收敛
```

## 路线图（两个核心 AI 多模态验证点）

1. **内容生成端** —— 提示词 → 大模型生成 3~5s 视频 → 拆帧 1bpp → BLE 投射
   - 挑战：生成内容复杂度 vs MEMS 1bpp 单色表达力
   - 判据：投射内容可辨认 + 动画流畅
2. **视觉理解端** —— 大模型看懂实拍投影视频 → 评估内容可辨识度 → 直接生成更易辨认的新内容
   - 挑战：模型理解 1bpp 投影可辨识度规律（线条粗细/细节密度/对比度）
   - 判据：3 轮内内容质量明显提升，不反复横跳
