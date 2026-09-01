# IrisLoop

USB 摄像头视频流采集工具。

## 安装

```bash
pip install -r requirements.txt
```

## 使用

```bash
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
