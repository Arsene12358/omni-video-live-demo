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

- A **running Qwen3-Omni omni server in persistent mode** on the **v0.26.0 pair**:
  - **vLLM v0.26.0 + the streaming-KV overlay**: `pip install vllm==0.26.0`, then overlay
    the branch's changed files from
    [`Arsene12358/vllm@feat/streaming-kv-v026`](https://github.com/Arsene12358/vllm/tree/feat/streaming-kv-v026).
  - **vLLM-Omni from source**:
    [`Arsene12358/vllm-omni@feat/persistent-video-session-v026`](https://github.com/Arsene12358/vllm-omni/tree/feat/persistent-video-session-v026)
    (`pip install -e .`).
  - The exact install recipe (overlay snippet included) lives in that branch's example
    README: `examples/online_serving/qwen3_omni/persistent_video_session/README.md`.

  Then serve with the example's `run_server.sh`, logging stdout to a file:
  ```bash
  ./run_server.sh > server.log 2>&1 &   # serves on :8901, logs KV/eviction to server.log
  ```
  `run_server.sh` now defaults to **CUDA graphs** (the validated production config) —
  see **Performance — CUDA graphs** below, including the constraint if you must run eager.
- This backend **co-located with the omni server** (so it can read `server.log` +
  `nvidia-smi`).
- Python 3.10+; `pip install -r backend/requirements.txt`.
- One or more demo clips (a striking, unambiguous **opening** makes the recall payoff land).

## Run

```bash
pip install -r backend/requirements.txt
mkdir -p clips && cp /path/to/your_clip.mp4 clips/         # clips are git-ignored

OMNI_PORT=8901 OMNI_LOG=/abs/path/to/persistent_video_session/server.log \
  CLIP_DIR=clips GPU_INDEX=0 PORT=8800 ./run.sh
# then, from your laptop:
#   ssh -L 8800:localhost:8800 <demo-box>     # port-forward
#   open http://localhost:8800
```

`OMNI_LOG` is the absolute path to the omni server's log, wherever you started it — the
backend tails that file for the live KV numbers, so pointing it at this repo's directory
leaves the dashboard dark.

In the UI: pick a clip → **Start session** → ask via the box or the preset buttons.

### Run it live on 2×H200 (computelab-aus)

The exact path for a live run on the validated hardware, from the `computelab-aus-01-frontend-01` login node:

```bash
# 1) allocate 2 H200s (avoid vkg-prod-673/674-au — they die under sustained load)
srun -p 'h200@cr+mp/None@cr+mp/8gpu-224cpu-2048gb' --gres=gpu:h200:2 \
     -x vkg-prod-673-au,vkg-prod-674-au -t 04:00:00 --pty bash
hostname   # note the compute node, e.g. vkg-prod-670-au — needed for the ssh -L below

# 2) get the code + install the v0.26.0 pair (once) — see Prerequisites above
WORK=$PWD
git clone -b feat/persistent-video-session-v026 \
    https://github.com/Arsene12358/vllm-omni.git
git clone https://github.com/Arsene12358/omni-video-live-demo.git
pip install vllm==0.26.0        # then overlay Arsene12358/vllm@feat/streaming-kv-v026
pip install -e "$WORK/vllm-omni"

# 3) serve (CUDA graphs default; ready in ~4 min: weights + torch.compile + capture)
SERVE_DIR="$WORK/vllm-omni/examples/online_serving/qwen3_omni/persistent_video_session"
cd "$SERVE_DIR"
MODEL=Qwen/Qwen3-Omni-30B-A3B-Instruct PORT=8901 ./run_server.sh > server.log 2>&1 &
server_pid=$!
# Ready == /v1/models answers 200 AND its `data` array is non-empty: a bare 200 can
# land before the model is registered. Bounded at 10 min, loud if the server dies.
ready=""
for _ in $(seq 1 120); do
  kill -0 "$server_pid" 2>/dev/null || { echo "omni server exited during startup:"; tail -40 server.log; break; }
  if curl -sf localhost:8901/v1/models 2>/dev/null \
     | python3 -c 'import json,sys; sys.exit(0 if json.load(sys.stdin).get("data") else 1)' 2>/dev/null; then
    ready=1; break
  fi
  sleep 5
done
[ -n "$ready" ] || { echo "omni server not ready — check $SERVE_DIR/server.log"; }

# 4) demo backend (this repo), co-located with the server
cd "$WORK/omni-video-live-demo"
pip install -r backend/requirements.txt
mkdir -p clips && cp /path/to/your_clip.mp4 clips/
OMNI_PORT=8901 OMNI_LOG="$SERVE_DIR/server.log" \
  CLIP_DIR=clips GPU_INDEX=0 PORT=8800 ./run.sh &

# 5) from your laptop: forward through the login node to the compute node
ssh -L 8800:<compute-node>:8800 computelab-aus-01-frontend-01
# then open http://localhost:8800 → pick the clip → Start session
```

## Performance — CUDA graphs (the default)

The omni example's `run_server.sh` now serves **with CUDA graphs** (no `--enforce-eager`) — the validated production config on the v0.26.0 stack. Eager decode at batch size 1, as in this single-stream demo, leaves the GPU **launch-bound**: every generated token issues thousands of tiny kernels, one `cudaLaunchKernel` at a time, with idle gaps in between. CUDA graphs replay the same kernels back-to-back via `cudaGraphLaunch` with no per-kernel launch overhead.

Measured on 2×H200 (Qwen3-Omni-30B-A3B, v0.26.0 + streaming-KV overlay; 800-frame session, queries every 50 frames, flood-fed):

| metric | eager | CUDA graphs | change |
|--------|-------|-------------|--------|
| decode throughput (per-answer median) | 26.5 tok/s | **213 tok/s** | **~8× faster** |
| session wall (800 frames + 16 answers) | ~62 s | **31.4 s** | ~2× faster |
| ingest e2e (first frame → session done) | ~13 f/s | **25.5 f/s** | ~2× faster |
| query cycle (50-frame ingest + answer) | 2.8–4.3 s | **~1.4 s** | ~2–3× faster |
| server ready (cold) | 182 s | 222 s | +40 s (torch.compile ~39 s + graph capture ≤2 s/stage) |
| answer quality / opening recall | — | — | unchanged (recall verified across refreshes) |

(The numbers previously cited here — ~35 → ~199 tok/s decode, ~5.7× — were measured on the pre-port v0.20-era branch; same mechanism, different stack.)

Notes:

- On v0.26 the graph path goes through torch.compile, so cold start pays ~40 s extra once per server start; decode then runs ~8× faster.
- The streaming-KV eviction attention backend is CUDA-graph-safe, validated end-to-end **including across position-refreshes** (the opening is still recalled after a reset).
- **If you must run eager**: eager + the default async scheduling wedges persistent streaming sessions. Set `async_scheduling: false` on stages 0/1 **through a deploy config** (`--deploy-config your.yaml`) — the `--no-async-scheduling` CLI flag is *not* sufficient. See the example README's Troubleshooting section.

An nsys timeline makes it concrete: eager shows ~360k individual kernel launches separated by gaps; CUDA graphs collapse the hot decode loop into ~15k graph replays at the **same per-kernel GPU time** — i.e. the speedup comes from eliminating the inter-kernel gaps, not from faster kernels.

**Engine-rebase mode (unbounded sessions) costs nothing at steady state.** The demo also works against the engine-rebase variant of the persistent session (`engine_rebase: true` from the client + `--streaming-kv-rebase-at` on the server; see the example README's "Two boundedness modes"): the engine rebases M-RoPE positions in place, one request runs for the whole session, and the dashboard flashes **↻ position rebase/refresh — opening retained** on each event just like a refresh. Measured on the same 2×H200 stack: per-answer median decode is **221.5 tok/s** in rebase mode vs the 213 tok/s refresh baseline (unchanged within noise), and the rebase event itself costs **~7.7–7.8 ms** once per ~38,400 positions (~28 min of 2 fps video). Four concurrent rebase sessions on one server (`--max-num-seqs 4`) each held a flat KV band and recalled their own opening across 16 rebases apiece.

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
| `MAX_FRAMES` | `1200` | cap on frames read from a clip upfront (~10 min @2fps) |
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
