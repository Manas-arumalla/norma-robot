"""Robot Oracle with web interface — control from iPhone on same WiFi.

Opens a web page at http://<laptop-ip>:8080 that the iPhone can open in Safari.
Press the button, speak, and the robot gestures the answer.

Run:
    uv run python oracle_web.py \
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
import tempfile
from pathlib import Path

import numpy as np
import torch
import whisper
from aiohttp import web
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

GESTURE_MAP = {
    "yes":   "nod yes",
    "no":    "shake no",
    "laugh": "laugh",
}

GESTURE_EMOJI = {"yes": "✅", "no": "❌", "laugh": "😂"}

GESTURE_HOME = [0.497, 0.036, 0.439, 0.970, 0.481, 0.613, 0.483, 0.494]

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("oracle_web")


# ---------------------------------------------------------------------------
# HTML page served to iPhone

HTML = """<!DOCTYPE html>
<html>
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Robot Oracle</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, sans-serif; background: #0f0f1a; color: #fff;
           display: flex; flex-direction: column; align-items: center;
           min-height: 100vh; padding: 30px 20px; }
    h1  { font-size: 28px; margin-bottom: 6px; }
    .sub { color: #888; font-size: 14px; margin-bottom: 40px; }
    #btn { width: 180px; height: 180px; border-radius: 50%; border: none;
           background: #3a3aff; color: #fff; font-size: 18px; font-weight: bold;
           cursor: pointer; transition: all 0.15s; box-shadow: 0 0 30px #3a3aff88; }
    #btn.recording { background: #ff3a3a; box-shadow: 0 0 40px #ff3a3a88; transform: scale(1.08); }
    #btn.waiting   { background: #555; box-shadow: none; }
    #status { margin-top: 28px; font-size: 15px; color: #aaa; min-height: 22px; }
    #transcript { margin-top: 18px; font-size: 17px; color: #fff;
                  background: #1e1e2e; border-radius: 12px; padding: 14px 18px;
                  width: 100%; max-width: 420px; min-height: 48px; }
    #response   { margin-top: 14px; font-size: 20px; font-weight: bold;
                  background: #1a2a1a; border-radius: 12px; padding: 14px 18px;
                  width: 100%; max-width: 420px; min-height: 48px; }
    #gesture    { font-size: 64px; margin-top: 20px; }
  </style>
</head>
<body>
  <h1>🤖 Robot Oracle</h1>
  <p class="sub">Ask me anything — I answer with YES, NO, or LAUGH</p>

  <button id="btn" onclick="toggleRecord()">Tap to<br>Ask</button>
  <div id="status">Ready</div>
  <div id="transcript"></div>
  <div id="response"></div>
  <div id="gesture"></div>

<script>
let mediaRecorder, chunks = [], recording = false;

async function toggleRecord() {
  if (recording) {
    stopRecord();
  } else {
    await startRecord();
  }
}

async function startRecord() {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
  chunks = [];
  mediaRecorder.ondataavailable = e => chunks.push(e.data);
  mediaRecorder.onstop = sendAudio;
  mediaRecorder.start();
  recording = true;
  document.getElementById('btn').className = 'recording';
  document.getElementById('btn').innerHTML = 'Tap to<br>Stop';
  document.getElementById('status').textContent = '🎤 Listening ...';
  document.getElementById('transcript').textContent = '';
  document.getElementById('response').textContent = '';
  document.getElementById('gesture').textContent = '';
}

function stopRecord() {
  mediaRecorder.stop();
  mediaRecorder.stream.getTracks().forEach(t => t.stop());
  recording = false;
  document.getElementById('btn').className = 'waiting';
  document.getElementById('btn').innerHTML = 'Processing...';
  document.getElementById('status').textContent = '⏳ Thinking ...';
}

async function sendAudio() {
  const blob = new Blob(chunks, { type: 'audio/webm' });
  const form = new FormData();
  form.append('audio', blob, 'recording.webm');
  try {
    const resp = await fetch('/ask', { method: 'POST', body: form });
    const data = await resp.json();
    document.getElementById('transcript').textContent = '🗣 ' + data.question;
    document.getElementById('response').textContent = '🤖 ' + data.reply;
    document.getElementById('gesture').textContent = data.emoji;
    document.getElementById('status').textContent = 'Done! Ask another?';
  } catch(e) {
    document.getElementById('status').textContent = 'Error: ' + e.message;
  }
  document.getElementById('btn').className = '';
  document.getElementById('btn').innerHTML = 'Tap to<br>Ask';
}
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Shared state (set during startup)

class RobotState:
    policy = None
    stats = None
    client = None
    motor_ids = None
    device = None
    args = None
    whisper_model = None
    busy = False

state = RobotState()


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
            "You control a physical robot that can perform three gestures: yes, no, or laugh. "
            "Rules:\n"
            "- If user says 'laugh', 'haha', 'funny', 'make me laugh' → gesture must be 'laugh'.\n"
            "- For yes/no questions → answer correctly with 'yes' or 'no'.\n"
            "- For math: compute and answer → 'yes' if true, 'no' if false.\n"
            "- For jokes or silly questions → 'laugh'.\n"
            "Reply ONLY as JSON: {\"gesture\": \"yes\"|\"no\"|\"laugh\", \"reply\": \"one sentence\"}"
        ),
        "messages": [{"role": "user", "content": question}],
    })
    resp = get_bedrock_client().invoke_model(body=body, modelId=model)
    text = json.loads(resp["body"].read())["content"][0]["text"].strip()
    try:
        data = json.loads(text[text.find("{"):text.rfind("}") + 1])
        gesture = data.get("gesture", "yes").lower()
        if gesture not in GESTURE_MAP:
            gesture = "yes"
        return gesture, data.get("reply", "")
    except Exception:
        return "yes", text


# ---------------------------------------------------------------------------
# Whisper

def transcribe(audio_bytes: bytes, model_name: str) -> str:
    if state.whisper_model is None:
        print("Loading Whisper ...")
        state.whisper_model = whisper.load_model(model_name)
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    result = state.whisper_model.transcribe(
        tmp_path, fp16=torch.cuda.is_available(), language="en"
    )
    Path(tmp_path).unlink(missing_ok=True)
    return result["text"].strip()


# ---------------------------------------------------------------------------
# Robot motion

async def fetch_fresh_frame(client, last_id: bytes, timeout: float):
    while True:
        qr = client.read_from_tail(QUEUE_ID, offset=b"\x00", limit=1, step=1, buf_size=1)
        entry = await asyncio.wait_for(qr.data.get(), timeout=timeout)
        if entry is None:
            raise RuntimeError(f"{QUEUE_ID} closed")
        frame = normvla.FrameReader(memoryview(bytes(entry.Data)))
        fid = bytes(frame.get_global_frame_id())
        if fid != last_id:
            return frame, fid
        await asyncio.sleep(0.02)


def frame_to_batch(frame, stats, device):
    joints = frame.get_joints() or []
    images = frame.get_images() or []
    s = torch.tensor([j.get_position_norm() for j in joints],
                     dtype=torch.float32, device=device).unsqueeze(0)
    n = s.shape[-1]
    ss = {"state_mean": stats["state_mean"][:n], "state_std": stats["state_std"][:n]}
    batch = {"observation.state": normalize_state(s, ss)}
    for i, key in enumerate(IMAGE_KEYS):
        if i >= len(images): break
        with Image.open(io.BytesIO(bytes(images[i].get_jpeg()))) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        batch[key] = torch.from_numpy(arr.copy()).permute(2,0,1).float().unsqueeze(0).to(device)/255.
    ranges = [(int(j.get_range_min()), int(j.get_range_max())) for j in joints]
    return batch, ranges


def build_sync_write(bus_serial, motor_ids, raw_goals):
    motors = [st3215.ST3215SyncWriteCommand_MotorWrite(
        motor_id=mid, value=raw.to_bytes(2, byteorder="little"))
        for mid, raw in zip(motor_ids, raw_goals)]
    sync = st3215.ST3215SyncWriteCommand(address=ST3215_TARGET_POS_REGISTER, motors=motors)
    cmd = st3215.Command(target_bus_serial=bus_serial, sync_write=sync)
    return commands.DriverCommand(type=drivers.StationCommandType.STC_ST3215_COMMAND, body=cmd.encode())


@torch.no_grad()
def predict_chunk(frame, policy, stats, task, device):
    batch, ranges = frame_to_batch(frame, stats, device)
    tokens, mask = policy.tokenize_task(task, device=device)
    batch["observation.language.tokens"] = tokens
    batch["observation.language.attention_mask"] = mask
    pred_norm = policy.predict_action_chunk(batch)[0]
    n = pred_norm.shape[-1]
    chunk = unnormalize_action(pred_norm, {
        "action_mean": stats["action_mean"][:n],
        "action_std":  stats["action_std"][:n],
    }).cpu().clamp(0.0, 1.0).numpy()
    return chunk, ranges


async def go_home(steps=30):
    frame, _ = await fetch_fresh_frame(state.client, b"", state.args.timeout)
    joints = frame.get_joints() or []
    cur  = [int(j.get_position()) for j in joints[:len(state.motor_ids)]]
    home = []
    for i, j in enumerate(joints[:len(state.motor_ids)]):
        rmin, rmax = int(j.get_range_min()), int(j.get_range_max())
        norm = GESTURE_HOME[i] if i < len(GESTURE_HOME) else 0.5
        home.append(max(rmin, min(rmax, int(round(rmin + norm * (rmax - rmin))))))
    for step in range(1, steps + 1):
        t = step / steps
        interp = [int(round(c + t * (h - c))) for c, h in zip(cur, home)]
        await send_commands(state.client, [build_sync_write(state.args.bus_serial, state.motor_ids, interp)])
        await asyncio.sleep(0.05)


async def perform_once(task: str):
    last_id = b""
    chunk = None
    ranges = []
    chunk_pos = 0
    replan = state.args.replan_every
    for _ in range(state.args.exec_ticks):
        frame, last_id = await fetch_fresh_frame(state.client, last_id, state.args.timeout)
        if chunk is None or chunk_pos >= replan or chunk_pos >= len(chunk):
            chunk, ranges = predict_chunk(frame, state.policy, state.stats, task, state.device)
            chunk_pos = 0
        goals = chunk[chunk_pos]; n = len(goals)
        joints = frame.get_joints() or []
        raws = [max(rmin, min(rmax, int(round(rmin + float(g) * (rmax - rmin)))))
                for g, (rmin, rmax) in zip(goals, ranges[:n])]
        deltas = [abs(int(r) - int(j.get_position())) for r, j in zip(raws, joints)]
        chunk_pos += 1
        if not (state.args.max_delta_ticks and max(deltas or [0]) > state.args.max_delta_ticks):
            await send_commands(state.client, [build_sync_write(state.args.bus_serial, state.motor_ids[:n], raws)])


async def run_gesture(gesture: str):
    task = GESTURE_MAP[gesture]
    await go_home()
    await asyncio.sleep(0.3)
    print(f"  [rep 1] {gesture}")
    await perform_once(task)
    print(f"  [rep 2] {gesture}")
    await perform_once(task)
    await go_home()


# ---------------------------------------------------------------------------
# Web handlers

async def handle_index(request):
    return web.Response(text=HTML, content_type="text/html")


async def handle_ask(request):
    if state.busy:
        return web.json_response({"error": "Robot is busy"}, status=429)
    state.busy = True
    try:
        data = await request.post()
        audio_file = data["audio"]
        audio_bytes = audio_file.file.read()

        loop = asyncio.get_event_loop()

        # Transcribe in thread (CPU/GPU bound)
        question = await loop.run_in_executor(
            None, transcribe, audio_bytes, state.args.whisper_model
        )
        print(f"  Question: {question!r}")

        # Ask Claude in thread
        gesture, reply = await loop.run_in_executor(
            None, ask_claude, question, state.args.claude_model
        )
        print(f"  Gesture: {gesture}  Reply: {reply}")

        # Run robot gesture (async, on event loop)
        await run_gesture(gesture)

        return web.json_response({
            "question": question,
            "gesture": gesture,
            "reply": reply,
            "emoji": GESTURE_EMOJI[gesture],
        })
    finally:
        state.busy = False


# ---------------------------------------------------------------------------
# Startup

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--bus-serial", required=True)
    p.add_argument("--server", default="localhost")
    p.add_argument("--motor-ids", default="1,2,3,4,5,6,7,8")
    p.add_argument("--port", type=int, default=8443)
    p.add_argument("--cert", default="/tmp/oracle_cert.pem")
    p.add_argument("--key",  default="/tmp/oracle_key.pem")
    p.add_argument("--exec-ticks", type=int, default=60)
    p.add_argument("--replan-every", type=int, default=25)
    p.add_argument("--max-delta-ticks", type=int, default=1000)
    p.add_argument("--timeout", type=float, default=5.0)
    p.add_argument("--whisper-model", default="base",
                   choices=["tiny", "base", "small", "medium"])
    p.add_argument("--claude-model", default="us.anthropic.claude-haiku-4-5-20251001-v1:0")
    return p.parse_args()


async def main():
    args = parse_args()
    state.args = args
    state.motor_ids = [int(x) for x in args.motor_ids.split(",")]
    state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stats_path = args.checkpoint / "stats.safetensors"
    if not stats_path.exists():
        raise SystemExit(f"No stats.safetensors in {args.checkpoint}")
    state.stats = {k: v.to(state.device) for k, v in load_stats(stats_path).items()}

    print("Loading SmolVLA checkpoint ...")
    state.policy = SmolVLAPolicy.from_pretrained(
        args.checkpoint, config_overrides={"load_vlm_weights": False}, strict=False
    ).to(state.device)
    state.policy.eval()

    print("Loading Whisper ...")
    state.whisper_model = whisper.load_model(args.whisper_model)

    state.client = await new_station_client(args.server, logger)

    # Start web server
    app = web.Application(client_max_size=50 * 1024 * 1024)
    app.router.add_get("/", handle_index)
    app.router.add_post("/ask", handle_ask)

    import ssl, socket
    ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_ctx.load_cert_chain(args.cert, args.key)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", args.port, ssl_context=ssl_ctx)
    await site.start()

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    print(f"\n✅ Ready! Open on your iPhone Safari:")
    print(f"   https://{ip}:{args.port}")
    print(f"\n⚠️  Safari will warn 'not secure' — tap Advanced → visit website to proceed.")
    print(f"\nPress Ctrl-C to stop.\n")

    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        print("\nShutting down ...")
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
