# SPDX-License-Identifier: Apache-2.0
"""Live-demo backend: bridge a browser UI to a running Qwen3-Omni omni server.

On "start" it streams a clip's frames at `sampling_fps` into the persistent
`/v1/video/chat/stream` session, relays the presenter's questions, and samples GPU
memory + the live KV working set (from the omni server log + nvidia-smi) — pushing
answers and metrics to the browser over one websocket.

Run it co-located with the omni server (so it can read the server log + nvidia-smi):

  OMNI_LOG=server.log CLIP_DIR=./clips OMNI_PORT=8901 GPU_INDEX=0 \
    uvicorn demo_backend:app --host 0.0.0.0 --port 8800
"""
import asyncio
import base64
import json
import os
import re
import subprocess
from pathlib import Path

import cv2
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse

OMNI_HOST = os.environ.get("OMNI_HOST", "localhost")
OMNI_PORT = int(os.environ.get("OMNI_PORT", "8901"))
OMNI_LOG = os.environ.get("OMNI_LOG", "")  # omni server stdout log, for KV/epoch metrics
CLIP_DIR = Path(os.environ.get("CLIP_DIR", "clips"))
GPU_INDEX = os.environ.get("GPU_INDEX", "0")
MAX_FRAMES = int(os.environ.get("MAX_FRAMES", "1200"))  # cap the upfront read (~10 min @2fps)
WEB = Path(__file__).resolve().parent.parent / "web"

BRIEF_SYS = (
    "You are a video understanding assistant. Answer concisely in one or two short "
    "sentences. Do not add extra commentary."
)
_EV = re.compile(r"computed=(\d+) total_blocks=(\d+) alive=(\d+)")
_SESS = re.compile(r"vsess-[a-f0-9]+-(\d+)")
# FROZEN: "[streaming-kv] rebase req=%s delta=%d new_base=%d recent_tokens=%d"
_REBASE = re.compile(r"\[streaming-kv\] rebase req=(\S+) delta=(\d+) new_base=(\d+) recent_tokens=(\d+)")

app = FastAPI()


@app.get("/")
async def index():
    return HTMLResponse((WEB / "index.html").read_text())


@app.get("/clips")
async def clips():
    if not CLIP_DIR.exists():
        return {"clips": []}
    exts = {".mp4", ".mov", ".webm", ".mkv"}
    return {"clips": sorted(p.name for p in CLIP_DIR.iterdir() if p.suffix.lower() in exts)}


@app.get("/clip/{name}")
async def clip(name: str):
    # guard traversal by name (not by resolve(), which would follow symlinked clips
    # out of CLIP_DIR and 404 them); is_file() still follows the symlink to its target.
    if "/" in name or "\\" in name or ".." in name:
        return HTMLResponse("not found", status_code=404)
    p = CLIP_DIR / name
    if not p.is_file():
        return HTMLResponse("not found", status_code=404)
    return FileResponse(p)


def build_session_config(msg):
    """Pure: the browser's "start" message -> the omni server's session.config.

    `engine_rebase` (the UI checkbox) hands boundedness to the engine's
    --streaming-kv-rebase-at: one request for the whole session, no driver-side
    refresh. `refresh_at_position` is then meaningless, and the server logs a
    warning if it is set anyway, so it is omitted in that mode.
    """
    engine_rebase = bool(msg.get("engine_rebase", False))
    cfg = {
        "type": "session.config", "modalities": ["text"], "persistent": True,
        "sink_frames": int(msg.get("sink_frames", 6)), "num_frames": int(msg.get("num_frames", 10)),
        "enable_frame_filter": bool(msg.get("evs", False)),
        "engine_rebase": engine_rebase,
        "system_prompt": BRIEF_SYS,
    }
    if not engine_rebase:
        cfg["refresh_at_position"] = int(msg.get("refresh_at", 2000))
    return cfg


def read_sampled(path, sampling_fps):
    """Uniform-stride sample a clip to `sampling_fps`; return (jpeg-b64 frames, source_fps)."""
    cap = cv2.VideoCapture(str(path))
    src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, round(src / max(0.1, sampling_fps)))
    frames, i = [], 0
    while len(frames) < MAX_FRAMES:
        ok, f = cap.read()
        if not ok:
            break
        if i % stride == 0:
            ok2, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok2:
                frames.append(base64.b64encode(buf.tobytes()).decode())
        i += 1
    cap.release()
    return frames, src


def gpu_mem_mb():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", GPU_INDEX],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        return int(out.splitlines()[0])
    except Exception:
        return None


def parse_kv_tail(tail):
    """Pure parse of a server-log tail -> (tokens_computed, kv_alive, epoch, rebases).

    epoch steps with new vsess-<hex>-<epoch>[-<stagehex>] ids (refresh mode); rebases
    counts "[streaming-kv] rebase ..." lines (engine-rebase mode, id pinned at epoch 0).
    """
    comp = al = ep = None
    for m in _EV.finditer(tail):
        comp, al = int(m.group(1)), int(m.group(3))
    eps = _SESS.findall(tail)
    if eps:
        ep = max(int(e) for e in eps)
    return comp, al, ep, len(_REBASE.findall(tail))


def read_kv():
    """Return (tokens_computed, kv_alive, epoch, rebases) from the tail of the omni log."""
    if not OMNI_LOG or not os.path.exists(OMNI_LOG):
        return None, None, None, None
    try:
        with open(OMNI_LOG, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - 65536))
            tail = fh.read().decode("utf-8", "ignore")
    except Exception:
        return None, None, None, None
    return parse_kv_tail(tail)


async def streamer(browser, omni, frames, sampling_fps):
    dt = 1.0 / max(0.1, sampling_fps)
    for i, fb in enumerate(frames):
        await omni.send(json.dumps({"type": "video.frame", "data": fb}))
        if (i + 1) % 4 == 0 or i + 1 == len(frames):
            await browser.send_json({"type": "frames", "sent": i + 1, "total": len(frames)})
        await asyncio.sleep(dt)
    await omni.send(json.dumps({"type": "video.done"}))


async def reader(browser, omni):
    cur = ""
    async for raw in omni:
        d = json.loads(raw)
        ty = d.get("type")
        if ty == "response.start":
            cur = ""
            await browser.send_json({"type": "answer.start"})
        elif ty == "response.text.delta":
            cur += d.get("delta", "")
            await browser.send_json({"type": "answer.delta", "text": d.get("delta", "")})
        elif ty == "response.text.done":
            await browser.send_json({"type": "answer.done", "text": d.get("text", "") or cur})
        elif ty == "session.done":
            await browser.send_json({"type": "session_done"})
            return
        elif ty == "error":
            await browser.send_json({"type": "error", "msg": d.get("message", "")})


async def metrics(browser):
    seen_epoch, prev_comp, offset, prev_rebases = -1, 0, 0, None
    while True:
        comp, al, ep, rb = await asyncio.to_thread(read_kv)
        mem = await asyncio.to_thread(gpu_mem_mb)
        tokens = None
        if comp is not None:
            if comp + 1 < prev_comp:  # computed resets each refresh -> accumulate
                offset += prev_comp
            prev_comp = comp
            tokens = offset + comp
        await browser.send_json({"type": "metric", "gpu_mem_mb": mem, "kv_alive": al, "tokens": tokens, "epoch": ep})
        if ep is not None and ep > seen_epoch:
            if seen_epoch >= 0:
                await browser.send_json({"type": "refresh", "epoch": ep})
            seen_epoch = ep
        if rb is not None:
            # engine-rebase mode: no epoch bumps; in-tail rebase-line count rising is the
            # signal. Compare to the previous poll (not a high-water mark) so lines that
            # scroll out of the 64 KiB tail re-arm detection; first poll only baselines.
            if prev_rebases is not None and rb > prev_rebases:
                await browser.send_json({"type": "refresh", "epoch": ep})
            prev_rebases = rb
        await asyncio.sleep(1.0)


@app.websocket("/ws")
async def ws(browser: WebSocket):
    await browser.accept()
    omni = None
    tasks = []

    async def teardown():
        for t in tasks:
            t.cancel()
        tasks.clear()
        nonlocal omni
        if omni is not None:
            try:
                await omni.close()
            except Exception:
                pass
            omni = None

    try:
        while True:
            msg = json.loads(await browser.receive_text())
            t = msg.get("type")
            if t == "start":
                await teardown()
                clip_name = msg.get("clip", "")
                sampling_fps = float(msg.get("sampling_fps", 2))
                frames, src_fps = await asyncio.to_thread(read_sampled, CLIP_DIR / clip_name, sampling_fps)
                if not frames:
                    await browser.send_json({"type": "error", "msg": f"no frames read from {clip_name}"})
                    continue
                await browser.send_json({
                    "type": "started", "clip": clip_name, "source_fps": round(src_fps, 1),
                    "sampling_fps": sampling_fps, "n_frames": len(frames),
                })
                uri = f"ws://{OMNI_HOST}:{OMNI_PORT}/v1/video/chat/stream"
                omni = await websockets.connect(uri, max_size=64 * 1024 * 1024)
                await omni.send(json.dumps(build_session_config(msg)))
                tasks = [
                    asyncio.create_task(streamer(browser, omni, frames, sampling_fps)),
                    asyncio.create_task(reader(browser, omni)),
                    asyncio.create_task(metrics(browser)),
                ]
            elif t == "query" and omni is not None:
                await omni.send(json.dumps({"type": "video.query", "text": msg.get("text", "")}))
            elif t == "stop":
                await teardown()
                await browser.send_json({"type": "session_done"})
    except WebSocketDisconnect:
        pass
    finally:
        await teardown()
