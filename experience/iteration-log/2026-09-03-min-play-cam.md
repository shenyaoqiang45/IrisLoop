# 2026-09-03 Minimum play barrier: laptop camera vs USB cam

## Hypothesis

Users should be able to run the observe step with **IrisGreen (BLE) + a laptop built-in camera**,
without buying a USB webcam.

## Probe on the developer workstation

- Host: MSI MS-7B89, `PCSystemType=1` **desktop**
- Imaging devices: only `Microsoft LifeCam HD-3000` (USB)
- OpenCV `cam#0` 1280×720 MJPG still OK (mean ≈ 131)
- No laptop front/rear camera available on this machine

## Software changes

`irisloop/camera.py`: if MJPG fails, fall back to the camera’s native format
(built-in webcams often use YUY2/NV12).

Probe script: `python tools/min_play_test.py`; add `--ble --group 1` for on-device projection.

## Next

Re-run the same script on a laptop with a built-in camera aimed at the projection surface.
Most notebooks have no rear camera: for a desk projection, fold the lid to ~20–40° so the
top webcam looks down; for a wall, use a 2-in-1 tent mode.
