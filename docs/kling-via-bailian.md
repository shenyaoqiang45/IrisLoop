# Kling writer via Alibaba Cloud Bailian (fallback)

**Preferred writer is now SiliconFlow Wan2.2** — see [`wan-via-siliconflow.md`](wan-via-siliconflow.md).

This doc keeps the Bailian Kling path as a fallback: call Kling through
[Bailian / Model Studio](https://help.aliyun.com/zh/model-studio/kling-video-generation-api-reference/),
not the consumer Kling web plan.

## Why Bailian

- Official listing: `kling/kling-v3-video-generation` (also turbo / omni variants)
- New-user free video quota is enough for a handful of short closed-loop trials
- One key: `DASHSCOPE_API_KEY` (never commit it)

## Cost posture

| Want | Reality |
| --- | --- |
| “Generate 1 second” | **Not offered** — Bailian Kling v3 duration is **3–15 s** |
| Cheap probe | Use **`duration=3`** (API floor), then **extract 3 frames** for K3 |

Billing is per generated second. IrisLoop defaults to 3 s + 3 stills so the director
(Kimi K3) can judge without paying for long cinema clips.

## Open the model

1. Bailian console → search **kling** → **立即开通**
2. Create a **Beijing-region** API Key → `DASHSCOPE_API_KEY`
3. Copy workspace / 业务空间 ID → `DASHSCOPE_WORKSPACE_ID`
4. (Optional) `DASHSCOPE_KLING_MODEL=kling/kling-v3-video-generation`

PowerShell (session):

```powershell
$env:DASHSCOPE_API_KEY = "sk-..."
$env:DASHSCOPE_WORKSPACE_ID = "your-workspace-id"
```

## Smoke test

```bash
python tools/kling_bailian_smoke.py "bold green whale silhouette swimming, high contrast, simple shapes"
```

Outputs under `captures/kling/<timestamp>/`:

- `kling.mp4` — 3 s clip
- `frames/frame_00.jpg` … `frame_02.jpg` — evenly sampled stills

Flags: `--duration 3` (minimum), `--frames 3`, `--mode std`.

## Role split

- **Kimi K3** — director: watches real laser / stills, owns “good enough”
- **Kling (Bailian)** — writer: shortest clip → frames → later 1bpp / project

Keys stay in the environment only (`DASHSCOPE_*`, `MOONSHOT_API_KEY` / `KIMI_API_KEY`).
