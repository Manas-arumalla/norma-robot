"""Simple Robot Oracle — control from iPhone browser via text/dictation.

Run:
    uv run python oracle_simple.py \
        --checkpoint ../../ai/smolvla_py/checkpoints/yes-no-laugh/final \
        --bus-serial 5B61034836
"""

from __future__ import annotations

import argparse
import asyncio
import boto3
import io
import json
import logging
import sys
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse
import threading

import numpy as np
import torch
from PIL import Image

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[4]
sys.path.insert(0, str(_REPO / "software" / "station" / "shared"))
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "software" / "ai" / "smolvla_py"))

from station_py import new_station_client, send_commands  # noqa: E402
from target.gen_python.protobuf.drivers.inferences import normvla  # noqa: E402
from target.gen_python.protobuf.drivers.st3215 import st3215  # noqa: E402
from target.gen_python.protobuf.station import commands, drivers  # noqa: E402

from smolvla import SmolVLAPolicy  # noqa: E402
from smolvla.normalize import normalize_state, unnormalize_action  # noqa: E402
from smolvla.stats import load_stats  # noqa: E402

QUEUE_ID = "inference/normvla"
ST3215_TARGET_POS_REGISTER = 0x2A
IMAGE_KEYS = ("observation.images.cam0", "observation.images.cam1")

GESTURE_MAP  = {"yes": "nod yes", "no": "shake no", "laugh": "laugh"}
GESTURE_EMOJI = {"yes": "✅ YES", "no": "❌ NO", "laugh": "😂 LAUGH"}
GESTURE_HOME  = [0.497, 0.036, 0.439, 0.970, 0.481, 0.613, 0.483, 0.494]

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("oracle_simple")

HTML = """<!DOCTYPE html>
<html>
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Robot Oracle</title>
  <style>
    body { font-family: -apple-system, sans-serif; background:#0f0f1a; color:#fff;
           display:flex; flex-direction:column; align-items:center;
           padding:40px 20px; gap:20px; }
    h1   { font-size:26px; }
    p    { color:#888; font-size:14px; text-align:center; }
    input { width:100%; max-width:420px; padding:16px; font-size:18px;
            border-radius:12px; border:none; background:#1e1e2e; color:#fff; }
    button { width:100%; max-width:420px; padding:16px; font-size:20px;
             font-weight:bold; border:none; border-radius:12px;
             background:#3a3aff; color:#fff; cursor:pointer; }
    #result { font-size:22px; font-weight:bold; min-height:40px; }
    #reply  { font-size:16px; color:#aaa; max-width:420px; text-align:center; }
    #gesture { font-size:72px; }
    #status { font-size:14px; color:#888; }
  </style>
</head>
<body>
  <h1>🤖 Robot Oracle</h1>
  <p>Type or use the 🎤 mic on your keyboard to ask anything</p>
  <input id="q" type="text" placeholder="Ask me anything..." autocomplete="off">
  <button onclick="ask()">Ask the Robot</button>
  <div id="status"></div>
  <div id="gesture"></div>
  <div id="result"></div>
  <div id="reply"></div>

<script>
async function ask() {
  const q = document.getElementById('q').value.trim();
  if (!q) return;
  document.getElementById('status').textContent = '⏳ Thinking...';
  document.getElementById('gesture').textContent = '';
  document.getElementById('result').textContent = '';
  document.getElementById('reply').textContent = '';
  try {
    const r = await fetch('/ask?q=' + encodeURIComponent(q));
    const d = await r.json();
    document.getElementById('gesture').textContent = d.emoji.split(' ')[0];
    document.getElementById('result').textContent  = d.emoji;
    document.getElementById('reply').textContent   = d.reply;
    document.getElementById('status').textContent  = '';
    document.getElementById('q').value = '';
  } catch(e) {
    document.getElementById('status').textContent = 'Error — is the robot running?';
  }
}
document.getElementById('q').addEventListener('keydown', e => { if(e.key==='Enter') ask(); });
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Shared state

class State:
    policy = None
    stats  = None
    client = None
    motor_ids = None
    device = None
    args   = None
    busy   = False
    loop   = None

state = State()

# ---------------------------------------------------------------------------
# AWS Bedrock

_bedrock_client = None

def get_bedrock_client():
    global _bedrock_client
    if _bedrock_client is None:
        _bedrock_client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
    return _bedrock_client


def ask_claude(question: str, model: str) -> tuple[str, str]:
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 80,
        "system": (
            "You control a robot with three gestures: yes, no, laugh. "
            "- Direct commands like 'laugh', 'say yes', 'say no' → follow them directly. "
            "- Math questions → compute correctly → yes if true, no if false. "
            "- Funny/silly/joke → laugh. "
            "- Yes/no questions → answer factually. "
            "Reply ONLY as JSON: {\"gesture\":\"yes\"|\"no\"|\"laugh\",\"reply\":\"one sentence\"}"
        ),
        "messages": [{"role": "user", "content": question}],
    })
    resp = get_bedrock_client().invoke_model(body=body, modelId=model)
    text = json.loads(resp["body"].read())["content"][0]["text"].strip()
    try:
        d = json.loads(text[text.find("{"):text.rfind("}")+1])
        g = d.get("gesture","yes").lower()
        return (g if g in GESTURE_MAP else "yes"), d.get("reply","")
    except Exception:
        return "yes", text


# ---------------------------------------------------------------------------
# Robot motion

async def fetch_fresh_frame(client, last_id, timeout):
    while True:
        qr    = client.read_from_tail(QUEUE_ID, offset=b"\x00", limit=1, step=1, buf_size=1)
        entry = await asyncio.wait_for(qr.data.get(), timeout=timeout)
        if entry is None:
            raise RuntimeError("queue closed")
        frame = normvla.FrameReader(memoryview(bytes(entry.Data)))
        fid   = bytes(frame.get_global_frame_id())
        if fid != last_id:
            return frame, fid
        await asyncio.sleep(0.02)


def frame_to_batch(frame, stats, device):
    joints = frame.get_joints() or []
    images = frame.get_images() or []
    s  = torch.tensor([j.get_position_norm() for j in joints], dtype=torch.float32, device=device).unsqueeze(0)
    n  = s.shape[-1]
    ss = {"state_mean": stats["state_mean"][:n], "state_std": stats["state_std"][:n]}
    batch = {"observation.state": normalize_state(s, ss)}
    for i, key in enumerate(IMAGE_KEYS):
        if i >= len(images): break
        with Image.open(io.BytesIO(bytes(images[i].get_jpeg()))) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        batch[key] = torch.from_numpy(arr.copy()).permute(2,0,1).float().unsqueeze(0).to(device)/255.
    ranges = [(int(j.get_range_min()), int(j.get_range_max())) for j in joints]
    return batch, ranges


def build_sync_write(bus_serial, motor_ids, raws):
    motors = [st3215.ST3215SyncWriteCommand_MotorWrite(motor_id=m, value=r.to_bytes(2,"little"))
              for m, r in zip(motor_ids, raws)]
    sync = st3215.ST3215SyncWriteCommand(address=ST3215_TARGET_POS_REGISTER, motors=motors)
    cmd  = st3215.Command(target_bus_serial=bus_serial, sync_write=sync)
    return commands.DriverCommand(type=drivers.StationCommandType.STC_ST3215_COMMAND, body=cmd.encode())


@torch.no_grad()
def predict_chunk(frame, policy, stats, task, device):
    batch, ranges = frame_to_batch(frame, stats, device)
    tok, mask = policy.tokenize_task(task, device=device)
    batch["observation.language.tokens"]        = tok
    batch["observation.language.attention_mask"] = mask
    pred = policy.predict_action_chunk(batch)[0]
    n    = pred.shape[-1]
    chunk = unnormalize_action(pred, {"action_mean": stats["action_mean"][:n],
                                      "action_std":  stats["action_std"][:n]}).cpu().clamp(0,1).numpy()
    return chunk, ranges


async def go_home(steps=30):
    frame, _ = await fetch_fresh_frame(state.client, b"", state.args.timeout)
    joints = frame.get_joints() or []
    cur  = [int(j.get_position()) for j in joints[:len(state.motor_ids)]]
    home = []
    for i, j in enumerate(joints[:len(state.motor_ids)]):
        rmin, rmax = int(j.get_range_min()), int(j.get_range_max())
        norm = GESTURE_HOME[i] if i < len(GESTURE_HOME) else 0.5
        home.append(max(rmin, min(rmax, int(round(rmin + norm*(rmax-rmin))))))
    for step in range(1, steps+1):
        t = step/steps
        interp = [int(round(c + t*(h-c))) for c,h in zip(cur, home)]
        await send_commands(state.client, [build_sync_write(state.args.bus_serial, state.motor_ids, interp)])
        await asyncio.sleep(0.05)


async def perform_once(task):
    last_id, chunk, ranges, pos = b"", None, [], 0
    for _ in range(state.args.exec_ticks):
        frame, last_id = await fetch_fresh_frame(state.client, last_id, state.args.timeout)
        if chunk is None or pos >= state.args.replan_every or pos >= len(chunk):
            chunk, ranges = predict_chunk(frame, state.policy, state.stats, task, state.device)
            pos = 0
        goals = chunk[pos]; n = len(goals)
        joints = frame.get_joints() or []
        raws = [max(rmin, min(rmax, int(round(rmin + float(g)*(rmax-rmin)))))
                for g,(rmin,rmax) in zip(goals, ranges[:n])]
        deltas = [abs(int(r)-int(j.get_position())) for r,j in zip(raws,joints)]
        pos += 1
        if not (state.args.max_delta_ticks and max(deltas or [0]) > state.args.max_delta_ticks):
            await send_commands(state.client, [build_sync_write(state.args.bus_serial, state.motor_ids[:n], raws)])


async def run_gesture(gesture):
    task = GESTURE_MAP[gesture]
    await go_home()
    await asyncio.sleep(0.3)
    await perform_once(task)
    await perform_once(task)
    await go_home()


# ---------------------------------------------------------------------------
# HTTP server (runs in thread, schedules robot work on asyncio loop)

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_): pass

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML.encode())

        elif path == "/ask":
            if state.busy:
                self._json({"error": "busy"}, 429); return
            q = parse_qs(urlparse(self.path).query).get("q", [""])[0].strip()
            if not q:
                self._json({"error": "empty"}); return

            print(f"  Question: {q!r}")
            state.busy = True
            try:
                gesture, reply = ask_claude(q, state.args.claude_model)
                print(f"  → {gesture}: {reply}")
                future = asyncio.run_coroutine_threadsafe(run_gesture(gesture), state.loop)
                future.result(timeout=60)
                self._json({"gesture": gesture, "reply": reply,
                            "emoji": GESTURE_EMOJI[gesture]})
            except Exception as e:
                self._json({"error": str(e)}, 500)
            finally:
                state.busy = False
        else:
            self.send_response(404); self.end_headers()

    def _json(self, data, code=200):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)


# ---------------------------------------------------------------------------
# Main

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--bus-serial", required=True)
    p.add_argument("--server", default="localhost")
    p.add_argument("--motor-ids", default="1,2,3,4,5,6,7,8")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--exec-ticks",      type=int,   default=60)
    p.add_argument("--replan-every",    type=int,   default=25)
    p.add_argument("--max-delta-ticks", type=int,   default=1000)
    p.add_argument("--timeout",         type=float, default=5.0)
    p.add_argument("--claude-model", default="us.anthropic.claude-haiku-4-5-20251001-v1:0")
    return p.parse_args()


async def main():
    args = parse_args()
    state.args      = args
    state.motor_ids = [int(x) for x in args.motor_ids.split(",")]
    state.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state.loop      = asyncio.get_event_loop()

    stats_path = args.checkpoint / "stats.safetensors"
    state.stats = {k: v.to(state.device) for k, v in load_stats(stats_path).items()}

    print("Loading checkpoint ...")
    state.policy = SmolVLAPolicy.from_pretrained(
        args.checkpoint, config_overrides={"load_vlm_weights": False}, strict=False
    ).to(state.device)
    state.policy.eval()

    state.client = await new_station_client(args.server, logger)

    # Start HTTP server in background thread
    httpd = HTTPServer(("0.0.0.0", args.port), Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()

    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80)); ip = s.getsockname()[0]
    finally:
        s.close()

    print(f"\n✅ Open this on your iPhone Safari:")
    print(f"   http://{ip}:{args.port}")
    print(f"\nType or dictate a question → tap Ask → robot gestures.")
    print(f"Press Ctrl-C to stop.\n")

    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        print("\nShutting down ...")
        httpd.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
