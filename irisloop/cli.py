"""命令行入口：irisloop 摄像头视频采集。"""

from __future__ import annotations

import argparse
import sys

from .camera import probe_cameras
from .capture import capture


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="irisloop",
        description="USB 摄像头视频流采集并保存到本地",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-o", "--output", default=None, help="输出文件路径，默认 captures/capture_<时间戳>.mp4")
    p.add_argument("-d", "--output-dir", default="captures", help="未指定 --output 时的输出目录")
    p.add_argument("-i", "--camera-index", type=int, default=0, help="摄像头索引")
    p.add_argument("--width", type=int, default=1280, help="请求宽度")
    p.add_argument("--height", type=int, default=720, help="请求高度")
    p.add_argument("--fps", type=int, default=30, help="请求帧率")
    p.add_argument("-t", "--duration", type=float, default=None, help="录制时长(秒)，不设则手动停止")
    p.add_argument("--no-preview", action="store_true", help="不显示预览窗口（后台录制）")
    p.add_argument("--no-timestamp", action="store_true", help="画面上不叠加时间戳")
    p.add_argument("--no-measure", action="store_true", help="跳过启动时的实测帧率")
    p.add_argument("--codec", default=None, help="强制指定编码，如 mp4v / MJPG / XVID")
    p.add_argument("--list", action="store_true", help="列出可用摄像头后退出")
    return p


def cmd_list() -> int:
    cams = probe_cameras()
    if not cams:
        print("未发现可用摄像头")
        return 1
    print("可用摄像头：")
    for c in cams:
        print(f"  [{c['index']}] {c['width']}x{c['height']} @{c['fps']:.0f}fps backend={c['backend']}")
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.list:
        return cmd_list()

    try:
        capture(
            camera_index=args.camera_index,
            width=args.width,
            height=args.height,
            fps=args.fps,
            output=args.output,
            output_dir=args.output_dir,
            duration=args.duration,
            preview=not args.no_preview,
            timestamp=not args.no_timestamp,
            measure=not args.no_measure,
            codec=args.codec,
        )
    except KeyboardInterrupt:
        print("\n[info] 已中断（视频已保存）")
    except Exception as e:
        print(f"[error] {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
