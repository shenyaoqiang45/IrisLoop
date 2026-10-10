"""Hikrobot (海康) industrial camera wrapper via the MVS SDK Python binding.

The MVS SDK ships ctypes bindings at
    <MVS_DIR>/Development/Samples/Python/MvImport/MvCameraControl_class.py
They are NOT a pip package, so we inject that directory into sys.path.
Resolution order for MVS_DIR: constructor arg > $MVS_DIR > default install path.

Usage mirrors irisloop.camera.UsbCamera (open/read/release/info) so calibration
code can treat both camera kinds polymorphically.
"""

from __future__ import annotations

import os
import sys
import time
from ctypes import byref, cast, memset, POINTER, sizeof
from typing import List, Optional, Tuple

import numpy as np

DEFAULT_MVS_DIR = r"C:\Program Files (x86)\MVS"
MVIMPORT_REL = os.path.join("Development", "Samples", "Python", "MvImport")
# MVS 5.x puts the C runtime here (64-bit / 32-bit)
RUNTIME_DIRS = (
    r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64",
    r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win32_i86",
)


def _find_mvimport(mvs_dir: Optional[str] = None) -> str:
    base = mvs_dir or os.environ.get("MVS_DIR") or DEFAULT_MVS_DIR
    path = os.path.join(base, MVIMPORT_REL)
    if not os.path.isfile(os.path.join(path, "MvCameraControl_class.py")):
        raise RuntimeError(
            f"找不到 MVS Python 绑定: {path}\n"
            "请先安装 MVS 客户端（安装包内含 USB3 驱动），"
            "或用 --mvs-dir / 环境变量 MVS_DIR 指定安装目录。"
        )
    return path


def _add_dll_search_path() -> None:
    """The binding does WinDLL("MvCameraControl.dll") by bare name — the runtime
    dir must be on the DLL search path (PATH is consulted with winmode=0)."""
    import sys as _sys

    idx = 0 if _sys.maxsize > 2**32 else 1
    candidates = [RUNTIME_DIRS[idx]] + [RUNTIME_DIRS[1 - idx]]
    for d in candidates:
        if os.path.isfile(os.path.join(d, "MvCameraControl.dll")):
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
            try:
                os.add_dll_directory(d)
            except (AttributeError, OSError):
                pass
            return


def _load_binding(mvs_dir: Optional[str] = None):
    path = _find_mvimport(mvs_dir)
    _add_dll_search_path()
    if path not in sys.path:
        sys.path.insert(0, path)
    import MvCameraControl_class  # type: ignore
    return MvCameraControl_class


class HikCamera:
    """MVS SDK wrapper for a single Hikrobot USB3/GigE camera."""

    def __init__(self, mvs_dir: Optional[str] = None):
        self.mv = _load_binding(mvs_dir)
        self.cam = self.mv.MvCamera()
        self._open = False
        self._grabbing = False
        self.model = ""
        self.serial = ""
        self.actual_size: Tuple[int, int] = (0, 0)
        self.pixel_type = 0

    # ---------------- enumeration ----------------

    def enumerate(self) -> List[dict]:
        mv = self.mv
        dev_list = mv.MV_CC_DEVICE_INFO_LIST()
        tlayer = mv.MV_GIGE_DEVICE | mv.MV_USB_DEVICE
        ret = mv.MvCamera.MV_CC_EnumDevices(tlayer, dev_list)
        if ret != 0:
            raise RuntimeError(f"MV_CC_EnumDevices failed: 0x{ret:08x}")
        out = []
        for i in range(dev_list.nDeviceNum):
            info = cast(dev_list.pDeviceInfo[i], POINTER(mv.MV_CC_DEVICE_INFO)).contents
            if info.nTLayerType == mv.MV_USB_DEVICE:
                usb = info.SpecialInfo.stUsb3VInfo
                decode = lambda b: bytes(b).split(b"\x00")[0].decode("utf-8", "replace")
                out.append({
                    "index": i,
                    "interface": "USB3",
                    "model": decode(usb.chModelName),
                    "serial": decode(usb.chSerialNumber),
                })
            elif info.nTLayerType == mv.MV_GIGE_DEVICE:
                gige = info.SpecialInfo.stGigEInfo
                decode = lambda b: bytes(b).split(b"\x00")[0].decode("utf-8", "replace")
                out.append({
                    "index": i,
                    "interface": "GigE",
                    "model": decode(gige.chModelName),
                    "serial": decode(gige.chSerialNumber),
                })
        return out

    # ---------------- open / close ----------------

    def open(self, index: int = 0, serial: Optional[str] = None) -> None:
        mv = self.mv
        dev_list = mv.MV_CC_DEVICE_INFO_LIST()
        tlayer = mv.MV_GIGE_DEVICE | mv.MV_USB_DEVICE
        ret = mv.MvCamera.MV_CC_EnumDevices(tlayer, dev_list)
        if ret != 0:
            raise RuntimeError(f"MV_CC_EnumDevices failed: 0x{ret:08x}")
        if dev_list.nDeviceNum == 0:
            raise RuntimeError("未枚举到海康相机（检查 USB3 线缆/驱动）")

        # NOTE: device indices are only valid within THIS enumeration — never
        # mix an index from a previous MV_CC_EnumDevices call (order changes).
        def _decode(b) -> str:
            return bytes(b).split(b"\x00")[0].decode("utf-8", "replace")

        candidates = []
        for i in range(dev_list.nDeviceNum):
            info = cast(dev_list.pDeviceInfo[i], POINTER(mv.MV_CC_DEVICE_INFO)).contents
            if info.nTLayerType == mv.MV_USB_DEVICE:
                d = info.SpecialInfo.stUsb3VInfo
                candidates.append((i, "USB3", _decode(d.chModelName), _decode(d.chSerialNumber), info))
            elif info.nTLayerType == mv.MV_GIGE_DEVICE:
                d = info.SpecialInfo.stGigEInfo
                candidates.append((i, "GigE", _decode(d.chModelName), _decode(d.chSerialNumber), info))

        target = None
        for i, iface, model, sn, info in candidates:
            if serial is not None and sn == serial:
                target = (i, iface, model, sn, info)
                break
            if serial is None and i == index:
                target = (i, iface, model, sn, info)
                break
        if target is None:
            raise RuntimeError(
                f"未找到相机 index={index} serial={serial}; 已枚举: "
                f"{[(c[0], c[1], c[3]) for c in candidates]}")
        i, iface, model, sn, info = target

        ret = self.cam.MV_CC_CreateHandle(info)
        if ret != 0:
            raise RuntimeError(f"MV_CC_CreateHandle failed: 0x{ret:08x}")
        ret = self.cam.MV_CC_OpenDevice(mv.MV_ACCESS_Exclusive, 0)
        if ret != 0:
            self.cam.MV_CC_DestroyHandle()
            raise RuntimeError(f"MV_CC_OpenDevice failed: 0x{ret:08x}")
        self.model, self.serial = model, sn

        # continuous free-run acquisition; disable trigger (0 = off)
        self.cam.MV_CC_SetEnumValue("TriggerMode", 0)

        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            raise RuntimeError(f"MV_CC_StartGrabbing failed: 0x{ret:08x}")
        self._open = True
        self._grabbing = True

        w = self._get_int("Width")
        h = self._get_int("Height")
        self.actual_size = (w or 0, h or 0)
        self.pixel_type = self._get_enum("PixelFormat") or 0

    def close(self) -> None:
        if self._grabbing:
            self.cam.MV_CC_StopGrabbing()
            self._grabbing = False
        if self._open:
            self.cam.MV_CC_CloseDevice()
            self.cam.MV_CC_DestroyHandle()
            self._open = False

    release = close

    # ---------------- parameters ----------------

    def _get_int(self, key: str) -> Optional[int]:
        mv = self.mv
        val = mv.MVCC_INTVALUE()
        memset(byref(val), 0, sizeof(val))
        if self.cam.MV_CC_GetIntValue(key, val) != 0:
            return None
        return int(val.nCurValue)

    def _get_enum(self, key: str) -> Optional[int]:
        mv = self.mv
        val = mv.MVCC_ENUMVALUE()
        memset(byref(val), 0, sizeof(val))
        if self.cam.MV_CC_GetEnumValue(key, val) != 0:
            return None
        return int(val.nCurValue)

    def _get_float(self, key: str) -> Optional[float]:
        mv = self.mv
        val = mv.MVCC_FLOATVALUE()
        memset(byref(val), 0, sizeof(val))
        if self.cam.MV_CC_GetFloatValue(key, val) != 0:
            return None
        return float(val.fCurValue)

    def set_exposure_us(self, us: float) -> None:
        """Manual exposure in microseconds (ExposureAuto=Off first)."""
        self.cam.MV_CC_SetEnumValue("ExposureAuto", 0)
        ret = self.cam.MV_CC_SetFloatValue("ExposureTime", float(us))
        if ret != 0:
            raise RuntimeError(f"设置曝光失败: 0x{ret:08x}")

    def get_exposure_us(self) -> Optional[float]:
        return self._get_float("ExposureTime")

    def set_gain(self, db: float) -> None:
        self.cam.MV_CC_SetEnumValue("GainAuto", 0)
        ret = self.cam.MV_CC_SetFloatValue("Gain", float(db))
        if ret != 0:
            raise RuntimeError(f"设置增益失败: 0x{ret:08x}")

    def get_gain(self) -> Optional[float]:
        return self._get_float("Gain")

    # ---------------- grabbing ----------------

    def read(self, timeout_ms: int = 1000) -> Tuple[bool, Optional[np.ndarray]]:
        """Grab one frame, returned as BGR uint8 (mono sources are converted)."""
        if not self._grabbing:
            return False, None
        from ctypes import c_ubyte

        mv = self.mv
        w, h = self.actual_size
        buf_size = w * h * 4 + 2048
        buf = (c_ubyte * buf_size)()
        frame_info = mv.MV_FRAME_OUT_INFO_EX()
        memset(byref(frame_info), 0, sizeof(frame_info))
        ret = self.cam.MV_CC_GetOneFrameTimeout(byref(buf), buf_size, frame_info, timeout_ms)
        if ret != 0:
            return False, None

        fi = frame_info
        src = np.frombuffer(buf, dtype=np.uint8, count=fi.nFrameLen)
        return True, self._to_bgr(src, fi)

    def _to_bgr(self, src: np.ndarray, fi) -> np.ndarray:
        mv = self.mv
        w, h, ptype = fi.nWidth, fi.nHeight, fi.enPixelType
        mono8 = getattr(mv, "PixelType_Gvsp_Mono8", 0x01080001)
        bgr8 = getattr(mv, "PixelType_Gvsp_BGR8_Packed", 0x02180014)

        if ptype == mono8:
            gray = src[: w * h].reshape(h, w)
            return np.stack([gray, gray, gray], axis=-1)

        if ptype == bgr8:
            return src[: w * h * 3].reshape(h, w, 3)

        # Bayer/YUV/etc. -> BGR8 through the SDK converter
        from ctypes import c_ubyte

        dst_size = w * h * 3
        dst = (c_ubyte * dst_size)()
        src_buf = src.ctypes.data_as(POINTER(c_ubyte))
        param = mv.MV_CC_PIXEL_CONVERT_PARAM()
        memset(byref(param), 0, sizeof(param))
        param.nWidth = w
        param.nHeight = h
        param.enSrcPixelType = ptype
        param.pSrcData = src_buf
        param.nSrcDataLen = len(src)
        param.enDstPixelType = bgr8
        param.pDstBuffer = cast(dst, POINTER(c_ubyte))
        param.nDstBufferSize = dst_size
        ret = self.cam.MV_CC_ConvertPixelType(param)
        if ret != 0:
            raise RuntimeError(f"像素格式转换失败 pixelType=0x{ptype:08x}: 0x{ret:08x}")
        return np.frombuffer(dst, dtype=np.uint8).reshape(h, w, 3)

    def flush(self, n: int = 3) -> None:
        """Discard n frames — SDK buffers frames captured with old parameters,
        so flush after changing exposure/gain before the real capture."""
        for _ in range(n):
            self.read(500)

    def grab_average(self, n: int = 5, timeout_ms: int = 1000) -> Optional[np.ndarray]:
        """Average n frames (float accumulation) — suppresses laser scan stripes."""
        acc = None
        got = 0
        for _ in range(n):
            ok, frame = self.read(timeout_ms)
            if not ok or frame is None:
                continue
            f = frame.astype(np.float64)
            acc = f if acc is None else acc + f
            got += 1
        if got == 0:
            return None
        return np.clip(acc / got, 0, 255).astype(np.uint8)

    def measure_fps(self, seconds: float = 2.0) -> float:
        count = 0
        start = time.perf_counter()
        while time.perf_counter() - start < seconds:
            ok, _ = self.read(500)
            count += int(ok)
        elapsed = time.perf_counter() - start
        return count / elapsed if elapsed > 0 else 0.0

    def info(self) -> str:
        w, h = self.actual_size
        return f"{self.model} sn={self.serial} {w}x{h} pixelType=0x{self.pixel_type:08x}"

    def __enter__(self) -> "HikCamera":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
