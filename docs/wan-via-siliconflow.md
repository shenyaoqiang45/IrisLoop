# Wan writer via SiliconFlow (preferred)

SiliconFlow Wan path (alternate). **Preferred writer for cost+speed is now Bailian**
`wan2.2-t2v-plus` @ `832*480` — see `tools/wan_bailian_smoke.py` and `tools/loop_round1.py`.

This doc keeps the SiliconFlow API notes.

## Why switch from Kling

| | Bailian Kling v3 | SiliconFlow Wan2.2-T2V-A14B |
| --- | --- | --- |
| Length | 3–15 s (min 3 s, per-second bill) | ~**5 s** fixed clip |
| Billing | prepaid Bailian units + workspace | SiliconFlow API balance / per video |
| Auth | `DASHSCOPE_API_KEY` + workspace | **`SILICONFLOW_API_KEY` only** |
| Role | alternate writer | **preferred writer** |

Director seat is unchanged: **Kimi K3** owns “good enough” on real laser stills.

## Setup

1. Open [SiliconFlow API keys](https://cloud.siliconflow.cn/account/ak)
2. Create a key → export locally (never commit):

```powershell
$env:SILICONFLOW_API_KEY = "sk-..."
# optional:
# $env:SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
# $env:SILICONFLOW_WAN_MODEL = "Wan-AI/Wan2.2-T2V-A14B"
# $env:SILICONFLOW_WAN_SIZE = "1280x720"   # official enum only
```

**`image_size` must match the official enum**
([video-submit-post](https://api-docs.siliconflow.cn/docs/api/video-submit-post)):

`1280x720` | `720x1280` | `960x960`

Default is **`1280x720`** (16:9). Marketing copy mentions 480P, but this submit API
does **not** document a 480p size — do not send `832x480`.

## Smoke test

```bash
python tools/wan_siliconflow_smoke.py "bold green whale silhouette swimming, high contrast, simple shapes"
```

Outputs under `captures/wan/<timestamp>/`:

- `wan.mp4` — ~5 s clip @ **480p**  
- `frames/frame_00.jpg` … — evenly sampled (default 3)

For group-1 projection (max 10 BMPs):

```bash
python tools/wan_siliconflow_smoke.py "…" --frames 10 -o captures/wan/run1
python tools/push_group.py --dir captures/wan/run1/frames --group 1 --count 10
python tools/play_group.py --group 1 --loop --interval 3 --total 300
```

## Cost posture

- One Wan call ≈ one ~5 s video (platform list price; check console).
- Downstream cost is free: sequential extract → 1bpp → BLE.
- Prefer **fewer writer calls**, more K3 review on projected stills.

## API shape

- Submit: `POST /v1/video/submit` → `requestId`
- Poll: `POST /v1/video/status` with `{ "requestId" }` until `Succeed`
- Download `results.videos[0].url` promptly (link expires)
