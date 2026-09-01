"""采集主流程：读取摄像头 -> 写盘 -> 可选预览。"""

from __future__ import annotations

import time
from typing import Optional

import cv2

from .camera import UsbCamera
from .recorder import VideoRecorder, default_output_path


def _draw_hud(frame, fps: float, frames: int, elapsed: float, paused: bool, recording: bool):
    out = frame.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, h - 74), (w, h), (0, 0, 0), -1)
    state = "PAUSED" if paused else ("REC" if recording else "STOP")
    color = (0, 200, 255) if paused else ((0, 0, 255) if recording else (128, 128, 128))
    cv2.circle(out, (22, h - 52), 7, color, -1)
    cv2.putText(out, state, (38, h - 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    cv2.putText(
        out,
        f"FPS {fps:5.1f}   frames {frames}   {elapsed:.1f}s   q=quit p=pause",
        (12, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    return out


def capture(
    camera_index: int = 0,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    output: Optional[str] = None,
    output_dir: str = "captures",
    duration: Optional[float] = None,
    preview: bool = True,
    timestamp: bool = True,
    measure: bool = True,
    codec: Optional[str] = None,
) -> dict:
    cam = UsbCamera(camera_index, width, height, fps)
    cam.open()
    print(f"[camera] {cam.info()}")

    if measure or cam.actual_fps <= 1.0:
        measured = cam.measure_fps(seconds=1.5)
        print(f"[camera] 实测帧率 {measured:.1f} fps")

    # 后端上报值不可信时回退到请求值，避免写入 fps=0 的视频
    write_fps = cam.actual_fps if cam.actual_fps > 1.0 else float(fps)
    print(f"[camera] 写入帧率 {write_fps:.2f} fps")

    path = output or default_output_path(output_dir)
    rec = VideoRecorder(path, write_fps, cam.actual_size, codec)
    codec = rec.open()
    print(f"[record] {path}  codec={codec}  {cam.actual_size[0]}x{cam.actual_size[1]}")

    win = "capture"
    if preview:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, cam.actual_size[0], cam.actual_size[1])

    paused = False
    frame_count = 0
    start = time.perf_counter()
    last = start
    smooth_fps = 0.0
    print("[info] 采集中：q 退出，p 暂停/继续")

    try:
        while True:
            ok, frame = cam.read()
            if not ok or frame is None:
                print("[warn] 读取帧失败，跳过")
                continue

            now = time.perf_counter()
            dt = now - last
            last = now
            if dt > 0:
                smooth_fps = 0.9 * smooth_fps + 0.1 / dt if smooth_fps > 0 else 1.0 / dt

            if not paused:
                if timestamp:
                    cv2.putText(
                        frame,
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                        (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
                rec.write(frame)
                frame_count += 1

            if preview:
                elapsed = now - start
                cv2.imshow(win, _draw_hud(frame, smooth_fps, frame_count, elapsed, paused, not paused))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q / ESC
                break
            if key == ord("p"):
                paused = not paused
                print(f"[info] {'已暂停' if paused else '继续采集'}")

            if duration is not None and (time.perf_counter() - start) >= duration:
                print(f"[info] 达到设定时长 {duration}s，停止")
                break
    finally:
        elapsed = time.perf_counter() - start
        stats = rec.release()
        if preview:
            cv2.destroyAllWindows()
        cam.release()

    stats["elapsed_s"] = round(elapsed, 2)
    stats["avg_fps"] = round(frame_count / elapsed, 2) if elapsed > 0 else 0.0
    print(f"[done] {stats['frames']} 帧 / {elapsed:.1f}s -> {stats['path']} "
          f"({stats['size_mb']} MB, codec={stats['codec']})")
    return stats
