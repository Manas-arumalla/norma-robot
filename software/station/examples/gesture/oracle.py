"""Robot Q&A Oracle — Claude answers any question with a physical gesture.

Claude decides yes / no / laugh based on the question.
The robot performs the gesture TWICE without returning to home between repeats.

Run:
    uv run python oracle.py \
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

import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd
import tempfile
import torch
import whisper
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

# Mean of first frames across all gesture training episodes
GESTURE_HOME = [0.497, 0.036, 0.439, 0.970, 0.481, 0.613, 0.483, 0.494]

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("oracle")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Robot Q&A Oracle with Claude + SmolVLA")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--bus-serial", required=True)
    p.add_argument("--server", default="localhost")
    p.add_argument("--motor-ids", default="1,2,3,4,5,6,7,8")
    p.add_argument("--exec-ticks", type=int, default=60,
                   help="Ticks per gesture repetition")
    p.add_argument("--replan-every", type=int, default=25)
    p.add_argument("--max-delta-ticks", type=int, default=1000)
    p.add_argument("--timeout", type=float, default=5.0)
    p.add_argument("--claude-model", default="us.anthropic.claude-haiku-4-5-20251001-v1:0")
    p.add_argument("--voice", action="store_true",
                   help="Use microphone instead of keyboard input")
    p.add_argument("--whisper-model", default="base",
                   choices=["tiny", "base", "small", "medium"],
                   help="Whisper model size (base is fast and accurate enough)")
    p.add_argument("--record-seconds", type=float, default=4.0,
                   help="Seconds to record after pressing Enter")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Claude via AWS Bedrock

_bedrock_client = None

def get_bedrock_client():
    global _bedrock_client
    if _bedrock_client is None:
        _bedrock_client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
    return _bedrock_client


def ask_claude(question: str, model: str) -> tuple[str, str]:
    """Ask Claude the question. Returns (gesture, explanation).

    gesture is one of: yes | no | laugh
    """
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 80,
        "system": (
            "You control a physical robot that can perform three gestures: yes, no, or laugh. "
            "Rules for choosing the gesture:\n"
            "- If the user says 'laugh', 'haha', 'lol', 'funny', 'make me laugh', or anything asking you to laugh → gesture must be 'laugh'.\n"
            "- If the user asks a yes/no question, answer it correctly → 'yes' or 'no'.\n"
            "- For math: compute and answer correctly → 'yes' if true, 'no' if false.\n"
            "- For jokes, silly or absurd statements → 'laugh'.\n"
            "- For general knowledge questions → 'yes' or 'no' based on facts.\n"
            "Reply in exactly this JSON format with no extra text: "
            "{\"gesture\": \"yes\"|\"no\"|\"laugh\", \"reply\": \"one sentence\"}"
        ),
        "messages": [{"role": "user", "content": question}],
    })
    resp = get_bedrock_client().invoke_model(body=body, modelId=model)
    text = json.loads(resp["body"].read())["content"][0]["text"].strip()
    try:
        start, end = text.find("{"), text.rfind("}") + 1
        data = json.loads(text[start:end])
        gesture = data.get("gesture", "yes").lower()
        reply   = data.get("reply", "")
        if gesture not in GESTURE_MAP:
            gesture = "yes"
        return gesture, reply
    except Exception:
        return "yes", text


# ---------------------------------------------------------------------------
# Voice input

_whisper_model = None

def load_whisper(model_name: str):
    global _whisper_model
    if _whisper_model is None:
        print(f"  Loading Whisper '{model_name}' model ...")
        _whisper_model = whisper.load_model(model_name)
        print("  Whisper ready.")
    return _whisper_model


def record_and_transcribe(seconds: float, whisper_model_name: str, samplerate: int = 16000) -> str:
    """Record from microphone for `seconds` then transcribe with Whisper."""
    print(f"  🎤 Recording for {seconds:.0f}s ... speak now!")
    audio = sd.rec(int(seconds * samplerate), samplerate=samplerate, channels=1, dtype="float32")
    sd.wait()
    print("  Transcribing ...")
    model = load_whisper(whisper_model_name)
    audio_np = audio.squeeze()
    result = model.transcribe(audio_np, fp16=torch.cuda.is_available(), language="en")
    text = result["text"].strip()
    print(f"  You said: {text!r}")
    return text


# ---------------------------------------------------------------------------
# Station helpers

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
    state = torch.tensor(
        [j.get_position_norm() for j in joints], dtype=torch.float32, device=device
    ).unsqueeze(0)
    n = state.shape[-1]
    state_stats = {"state_mean": stats["state_mean"][:n], "state_std": stats["state_std"][:n]}
    batch = {"observation.state": normalize_state(state, state_stats)}
    for i, key in enumerate(IMAGE_KEYS):
        if i >= len(images):
            break
        jpeg = bytes(images[i].get_jpeg())
        with Image.open(io.BytesIO(jpeg)) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        batch[key] = (
            torch.from_numpy(arr.copy()).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
        )
    ranges = [(int(j.get_range_min()), int(j.get_range_max())) for j in joints]
    return batch, ranges


def build_sync_write(bus_serial, motor_ids, raw_goals):
    motors = [
        st3215.ST3215SyncWriteCommand_MotorWrite(
            motor_id=mid, value=raw.to_bytes(2, byteorder="little")
        )
        for mid, raw in zip(motor_ids, raw_goals)
    ]
    sync = st3215.ST3215SyncWriteCommand(address=ST3215_TARGET_POS_REGISTER, motors=motors)
    cmd = st3215.Command(target_bus_serial=bus_serial, sync_write=sync)
    return commands.DriverCommand(
        type=drivers.StationCommandType.STC_ST3215_COMMAND,
        body=cmd.encode(),
    )


@torch.no_grad()
def predict_chunk(frame, policy, stats, task, device):
    batch, ranges = frame_to_batch(frame, stats, device)
    tokens, mask = policy.tokenize_task(task, device=device)
    batch["observation.language.tokens"] = tokens
    batch["observation.language.attention_mask"] = mask
    pred_norm = policy.predict_action_chunk(batch)[0]
    n_pred = pred_norm.shape[-1]
    action_stats = {
        "action_mean": stats["action_mean"][:n_pred],
        "action_std":  stats["action_std"][:n_pred],
    }
    chunk = unnormalize_action(pred_norm, action_stats).cpu().clamp(0.0, 1.0).numpy()
    return chunk, ranges


async def go_home(client, frame, bus_serial, motor_ids, steps=30):
    """Smoothly interpolate from current position to home over N steps."""
    joints = frame.get_joints() or []
    current_raws = [int(j.get_position()) for j in joints[:len(motor_ids)]]
    home_raws = []
    for i, j in enumerate(joints[:len(motor_ids)]):
        rmin, rmax = int(j.get_range_min()), int(j.get_range_max())
        norm = GESTURE_HOME[i] if i < len(GESTURE_HOME) else 0.5
        home_raws.append(max(rmin, min(rmax, int(round(rmin + norm * (rmax - rmin))))))

    for step in range(1, steps + 1):
        t = step / steps
        interp = [int(round(cur + t * (tgt - cur))) for cur, tgt in zip(current_raws, home_raws)]
        cmd = build_sync_write(bus_serial, motor_ids, interp)
        await send_commands(client, [cmd])
        await asyncio.sleep(0.05)


async def perform_gesture_once(client, policy, stats, task, args, motor_ids, device):
    """Run one repetition of a gesture."""
    last_id = b""
    sent = 0
    chunk = None
    ranges = []
    chunk_pos = 0

    for tick in range(args.exec_ticks):
        frame, last_id = await fetch_fresh_frame(client, last_id, args.timeout)

        if chunk is None or chunk_pos >= args.replan_every or chunk_pos >= len(chunk):
            chunk, ranges = predict_chunk(frame, policy, stats, task, device)
            chunk_pos = 0

        goals = chunk[chunk_pos]
        n_pred = len(goals)
        joints = frame.get_joints() or []
        raws = []
        for g_norm, (rmin, rmax) in zip(goals, ranges[:n_pred]):
            raw = int(round(rmin + float(g_norm) * (rmax - rmin)))
            raws.append(max(rmin, min(rmax, raw)))

        deltas = [abs(int(r) - int(j.get_position())) for r, j in zip(raws, joints)]
        actual_max = max(deltas) if deltas else 0
        chunk_pos += 1

        if not (args.max_delta_ticks and actual_max > args.max_delta_ticks):
            cmd = build_sync_write(args.bus_serial, motor_ids[:n_pred], raws)
            await send_commands(client, [cmd])
            sent += 1

    return sent


# ---------------------------------------------------------------------------
# Main

async def main_async():
    args = parse_args()
    motor_ids = [int(x) for x in args.motor_ids.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stats_path = args.checkpoint / "stats.safetensors"
    if not stats_path.exists():
        raise SystemExit(f"No stats.safetensors in {args.checkpoint}")
    stats = {k: v.to(device) for k, v in load_stats(stats_path).items()}

    print(f"Loading checkpoint ...")
    policy = SmolVLAPolicy.from_pretrained(
        args.checkpoint,
        config_overrides={"load_vlm_weights": False},
        strict=False,
    ).to(device)
    policy.eval()

    client = await new_station_client(args.server, logger)

    # Pre-load whisper once at startup
    await asyncio.get_event_loop().run_in_executor(None, load_whisper, args.whisper_model)

    print("\n=== Robot Oracle ===")
    print("  • Press Enter (empty) → speak your question")
    print("  • Type a question    → press Enter to send")
    print("  • Type 'quit'        → exit\n")

    try:
        while True:
            try:
                typed = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: input("You (type or Enter to speak): ").strip()
                )
            except (EOFError, KeyboardInterrupt):
                break

            if typed.lower() in ("quit", "exit", "q"):
                break

            if typed == "":
                # Empty → voice mode
                question = await asyncio.get_event_loop().run_in_executor(
                    None, record_and_transcribe, args.record_seconds, args.whisper_model
                )
                if not question:
                    continue
            else:
                question = typed

            # Ask Claude
            print("  Thinking ...")
            gesture, reply = await asyncio.get_event_loop().run_in_executor(
                None, ask_claude, question, args.claude_model
            )
            task = GESTURE_MAP[gesture]

            print(f"\n  Robot: {reply}")
            print(f"  Gesture: {gesture.upper()} × 2\n")

            # Go to home before starting
            frame, _ = await fetch_fresh_frame(client, b"", args.timeout)
            await go_home(client, frame, args.bus_serial, motor_ids)
            await asyncio.sleep(0.3)

            # Perform gesture TWICE without going home between repetitions
            print(f"  [rep 1] ...")
            await perform_gesture_once(client, policy, stats, task, args, motor_ids, device)
            print(f"  [rep 2] ...")
            await perform_gesture_once(client, policy, stats, task, args, motor_ids, device)

            # Return home after both reps
            frame, _ = await fetch_fresh_frame(client, b"", args.timeout)
            await go_home(client, frame, args.bus_serial, motor_ids)
            print()

    except KeyboardInterrupt:
        pass
    finally:
        print("Returning to home ...")
        try:
            frame, _ = await fetch_fresh_frame(client, b"", args.timeout)
            await go_home(client, frame, args.bus_serial, motor_ids)
        except Exception:
            pass
        print("Done.")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
