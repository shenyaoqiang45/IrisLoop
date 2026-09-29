# One-round closed loop (`loop_round1`)

```text
wish → director (K3: silhouette + creative author) → writer (Wan)
    → push group 1 → project + camera capture → director review
```

Director `pass` means the author **loves the piece in its chosen 1bpp idiom**, not merely that a human can name the subject. `recognizable` stays a separate fact. If `pass` is false, `next_wan_prompt` must be a new English T2V string (`next_intent` is the round-2 creative goal).

## Env

```powershell
$env:MOONSHOT_API_KEY = "sk-..."          # or KIMI_API_KEY
$env:DASHSCOPE_API_KEY = "sk-..."         # Bailian
$env:DASHSCOPE_WORKSPACE_ID = "llm-..."   # Beijing workspace
```

Never commit keys.

## Run

```bash
python tools/loop_round1.py "dramatic volcanic eruption"
python tools/loop_round1.py "dramatic volcanic eruption" --rounds 2
```

To place the generated animation inside a region selected in the camera view,
first calibrate the fixed camera/projector pair, then pass the region as camera
pixel coordinates:

```bash
python -m irisloop.calibrate run
python tools/loop_round1.py "a butterfly opening its wings" --target-rect 500 240 760 430
```

The content is aspect-fitted into the rectangle and mapped through
`captures/calibration/homography.npz`. The camera and projector must remain
fixed after calibration.

Useful flags:

| Flag | Meaning |
| --- | --- |
| `--frames 10` | extract/push count (group 1 max 10) |
| `--size 832*480` | Wan 480P 16:9 (default) |
| `--exposure -7` | camera exposure for laser |
| `--rounds 2` | max cycles (capped at 2); stop if director loves the cut |
| `--target-rect X0 Y0 X1 Y1` | camera-pixel rectangle for spatial placement |
| `--calibration PATH` | alternate homography `.npz` file |
| `--spatial-padding 0.05` | inset inside the target rectangle |

## Outputs

Outputs:

- one round → `captures/loop_round1/<timestamp>/`
- `--rounds N` → `captures/loop_rounds/<timestamp>/r01` … `rN` + `session_report.json`

Per round:

- `01_director_brief.json` — authored silhouette concept + Wan prompt  
- `writer/wan.mp4` + `writer/frames/`  
- `projected_frames/` — final 640x480 spatially mapped frames (spatial mode only)
- `capture/group1.mp4` + stills  
- `04_director_review.json` — loved / pass / next_intent / next_wan_prompt  
- `round_report.json` — full round artifact  

Tracked sample: `captures/loop_rounds/20260903_180639/` (dancing kitten, 3 rounds unsigned).
