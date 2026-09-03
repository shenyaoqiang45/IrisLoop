# IrisLoop

**Adaptive MEMS laser projection loop** — *generate → project → observe → improve*

Drive an **IrisGreen (AINSTEC)** MEMS laser projector over BLE (green laser, 640×480 1bpp),
capture the physical projection with a local camera (laptop built-in or USB), and let AI vision
iterate on the **projected content**.

Current status **v0.1.0**: camera capture, BLE stack, asset playback, and project+capture are wired;
the AI closed loop is in progress.

## Layout

irisloop/            Core package
  camera.py          Camera capture (MSMF first; MJPG then native fourcc for laptop webcams)
  recorder.py        Video recording (codec fallback by extension: mp4v / MJPG / XVID)
  capture.py/cli.py  Capture flow and CLI (`python -m irisloop`)
  projector.py       IrisGreen profile (GATT UUIDs, MTU, device identity)
  protocol.py        BLE protocol V1.9 framing (type+len+data, 0x80 responses)
  ble_client.py      BLE client (command channel adb40003 write+indicate)
  image_pack.py      640×480 1bpp packing (62B BMP header + MSB bit packing; device-verified)
tools/               On-device bring-up scripts (see “Device bring-up” below)
doc/                 Iris Green BLE protocol workbook (V1.9 xlsx)
test-data/           BMP assets pulled from device (groups 1 and 20)
captures/            Capture outputs
docs/                Product / growth notes
experience/          Brand experience officer notes and laser logs
hub/                 Community submissions and featured demos
benchmark/           Groundtruth stills and VU eval artifacts
```

## Install

```bash
pip install -r requirements.txt   # opencv-python, numpy, bleak
```

## Usage

```bash
# Lowest play barrier: probe whether a built-in webcam can replace a USB cam
python tools/min_play_test.py
# On device: BLE project + local camera
python tools/min_play_test.py --ble --group 1 --seconds 5

# List cameras
python -m irisloop --list

# Default capture (1280x720, press q to stop, save under captures/)
python -m irisloop

# Record 10 seconds
python -m irisloop -t 10

# Headless record (no preview window)
python -m irisloop --no-preview -t 10

# Custom output path and resolution
python -m irisloop -o test.mp4 --width 640 --height 480
```

Keys: `q` or `ESC` to stop, `p` to pause / resume.

## Local environment notes

- Camera `Microsoft LifeCam HD-3000` tops out at **1280x720 @30fps** (1080p requests fall back)
- Built-in laptop webcams are supported too: they usually don't accept MJPG or the requested resolution, so the opener automatically falls back to the native fourcc/resolution (info line shows `fourcc=native`). Use `-i 0` for the built-in cam, `--list` to enumerate devices
- Working codecs: `mp4v` (.mp4), `MJPG` / `XVID` (.avi); `avc1` (H.264) unavailable without OpenH264
- DSHOW often reports `fps=0`; startup measures real FPS for writing

## IrisGreen projector (BLE)

Device profile (verified 2026-09-01): Model `IrisGreen` / Manufacturer `AINSTEC`,
advertisement name prefix `Iris-G`, negotiated MTU 512 (firmware tuned for bulk image transfer).

### GATT channels (primary service `adb401c0-...`)

| Characteristic | Properties | Role |
|---|---|---|
| `adb40001` | read | Device capability flags |
| `adb40002` | read+notify | Status stream (8-byte unsolicited frames, not command ACKs) |
| `adb40003` | write+indicate | **Command channel** (ACK via indicate on the same characteristic) |
| `adb40004` | write_no_rsp | Data stream |
| `adb40005` | write | Wi-Fi config (write `0x02` to enable Wi-Fi AP) |
| Service `adb40006-...-0001` | — | File transfer: 0002 start / 0003 data / 0004 end |

### Protocol notes (V1.9 workbook under `doc/`)

- Write frame: `type(1) + len(1) + data(n)`; response: `0x80 + type + len + data`
- Play group `0xA0`: `loop + total(u16le) + interval(u16le) + group` (time unit 100 ms)
- Stop `0xA1`; brightness `0x11`; FOV `0x1F` (10×5 / 30×10 / 20×10); distortion `0x1C`/`0x1D`; mirror `0x1E`; flip `0x12`
- Asset group IDs: 1–24 (boot / charge / animations / prompts, etc.); background groups start at `0x33`; total frames ≤ 200
- Spec errata (code already corrected): response prefix samples omit the leading `0x80` byte; BMP header is **62 bytes**, not 124
- Device has no BLE read-image command; pull assets over Wi-Fi AP HTTP: `http://192.168.4.1/upload/group_<gid>_<n>.bmp`, then `/switch/ble` to return to BLE

### Device bring-up (`tools/`)

```bash
# Scan / probe / read info
python tools/bt_scan_ble.py
python tools/bt_probe_device.py <addr>
python tools/bt_read_info.py <addr>

# Protocol smoke test (read-only; does not change device state)
python tools/ble_smoke.py

# Listen to status-frame cadence
python tools/ble_listen.py <addr> 12

# Play / stop an asset group
python tools/play_group.py --group 1
python tools/play_group.py --stop

# Project + synchronized camera capture (brightness / bright-area / LapVar heuristics)
python tools/project_and_capture.py --group 1 --seconds 6

# Pull assets from device (Wi-Fi HTTP)
python tools/pull_images.py

# Build and push a 640x480 1bpp image (file-transfer service; for debugging)
python tools/push_image.py --dry-run        # generate locally only
python tools/push_image.py --kind checker
```

## Status

- Camera capture / recording / FPS calibration
- BLE connect, GATT enum, read-only commands, asset play / stop
- 640×480 1bpp packing (matches MATLAB `im2Bytes.m`; verified against device BMPs)
- Project + camera capture (first closed-loop step)
- Asset pull over Wi-Fi HTTP
- BLE image push (reversed from iris-g-sdk; channel map + framing fixed; on-device write confirmed)
- AI vision loop (`loop.py` + `analyzer.py` scaffolded; rule mode runs; AI mode still being tuned)

## Core idea: tune content, not the device

Factory calibration already covers distortion / FOV / brightness hardware knobs.
IrisLoop iterates on the **projected content itself**:

```
AI generates content → BLE project → camera observe → AI reads the result
  → AI revises content → project again → ... until it converges
```

## Roadmap (two multimodal AI checkpoints)

1. **Content generation** — prompt → model generates 3–5s video → 1bpp frames → BLE project
   - Challenge: generative complexity vs MEMS 1bpp expressive power
   - Success: projected content is recognizable and motion is readable
2. **Visual understanding** — model watches real projection footage → scores recognizability → proposes clearer content
   - Challenge: learning 1bpp recognizability (stroke weight / detail density / contrast)
   - Success: clear quality gains within 3 rounds, without oscillating

## Docs

- [Developer growth plan (AI + projection)](docs/developer-growth-plan.md)
- [Experience / brand field notes](experience/README.md)
- [Hub submissions](hub/README.md)
