# omni-video-live-demo

A **live, presenter-driven demo** of the persistent streaming-video session for
Qwen3-Omni (vLLM-Omni): a video streams in, you ask questions in real time, answers
stream back, and a live dashboard shows **GPU memory / KV staying flat while the stream
grows** — across position-refreshes, with the **opening always recalled**.

This is the interactive companion to the reproducible reference in the vLLM-Omni repo
(`examples/online_serving/qwen3_omni/persistent_video_session/`) and its captured
recording. The model server is **unchanged**; this repo is just the demo front-end.

```
[ browser UI ] ⇄ (1 ws) ⇄ [ demo backend (this repo) ] ⇄ (omni ws) ⇄ [ omni server ]
 video + dashboard +        FastAPI: stream clip @ sampling_fps,        /v1/video/chat/stream
 chat                       relay questions, demux answers,             (persistent mode,
                            sample KV-alive + GPU mem ────────────────── unchanged)
```

The backend exists for one reason: the money shot — *"memory stays flat"* — is a
server-side number the browser can't see. It tails the omni server log (live KV blocks)
and `nvidia-smi` (GPU memory) and pushes them to the page alongside the answers.

## fps vocabulary

| term | meaning |
|------|---------|
| `source_fps` | native rate of the clip (what the viewer watches) |
| `sampling_fps` | frames/sec fed to the model (fixed at **2** in this MVP) |
| `effective_fps` | rate after the optional EVS similarity filter (≤ `sampling_fps`) |

Feeding at `sampling_fps` in real time naturally keeps pace with the clip playing at
`source_fps` (a 5-min clip → 600 frames @2fps → fed at 2/s → 5 min), so the video panel
and the model's "now" stay aligned, and answers land ~1.5 s after a question.

## Prerequisites

- A **running Qwen3-Omni omni server in persistent mode**, started with the streaming-KV
  eviction build and logging its stdout to a file. Use the vLLM-Omni example's
  `run_server.sh`:
  ```bash
  ./run_server.sh > server.log 2>&1 &   # serves on :8901, logs KV/eviction to server.log
  ```
- This backend **co-located with the omni server** (so it can read `server.log` +
  `nvidia-smi`).
- Python 3.10+; `pip install -r backend/requirements.txt`.
- One or more demo clips (a striking, unambiguous **opening** makes the recall payoff land).

## Run

```bash
pip install -r backend/requirements.txt
mkdir -p clips && cp /path/to/your_clip.mp4 clips/         # clips are git-ignored

OMNI_PORT=8901 OMNI_LOG=server.log CLIP_DIR=clips GPU_INDEX=0 PORT=8800 ./run.sh
# then, from your laptop:
#   ssh -L 8800:localhost:8800 <demo-box>     # port-forward
#   open http://localhost:8800
```

In the UI: pick a clip → **Start session** → ask via the box or the preset buttons.

## The narrative (what to show)

1. Start the clip — note the memory-gauge baseline.
2. ~30 s in: **"What is happening now?"** → it nails the current scene.
3. Let it run; the scene changes; point at **tokens climbing while memory stays flat**;
   a **↻ refresh** flashes → "it just reset its rotary position."
4. Minutes in, after the scene moved on and survived refreshes: **"What was shown at the
   very beginning?"** → it recalls the opening. The payoff.

## Configuration (env)

| var | default | meaning |
|-----|---------|---------|
| `OMNI_HOST` / `OMNI_PORT` | `localhost` / `8901` | the omni server WS |
| `OMNI_LOG` | `server.log` | omni server stdout log (for live KV / epoch metrics) |
| `CLIP_DIR` | `clips` | folder of demo clips |
| `GPU_INDEX` | `0` | GPU to read memory from (the thinker GPU) |
| `PORT` | `8800` | the demo backend port |

Per-session knobs are sent from the UI: `sampling_fps` (2), `sink_frames` (6),
`num_frames` (10), `refresh_at` (2000), `evs` (off).

## MVP limitations / notes

- **Backend must be co-located with the omni server** for the KV/GPU metrics (it reads
  the server log + local `nvidia-smi`). A remote metrics channel is a later option.
- **GPU memory is pre-allocated** by vLLM, so it's flat partly by construction — the
  load-bearing signal is **live KV blocks staying flat while tokens climb**. The UI
  foregrounds KV; GPU memory is secondary.
- `sampling_fps` is **fixed at 2** here. A live slider needs `second_per_grid_ts` to move
  in lockstep (else the model mis-judges elapsed time) — a Phase-2 item.
- Video↔model sync is approximate (both run ~clip-duration); good enough for a demo.

See `../vllm-omni/examples/online_serving/qwen3_omni/persistent_video_session/` for the
serve flags, the reproducible scripts, and the design background.
