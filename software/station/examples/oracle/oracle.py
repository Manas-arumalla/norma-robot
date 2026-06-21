"""The Oracle — robot answers questions via physical gesture.

Claude reads the question, decides yes/no/maybe + gives a witty explanation,
the robot performs the gesture. Optional: two robots that disagree.

Usage:
    uv run python oracle.py \\
        --checkpoint ../../ai/smolvla_py/checkpoints/oracle/final \\
        --bus-serial B61034574

    # Two-robot debate mode:
    uv run python oracle.py \\
        --checkpoint ../../ai/smolvla_py/checkpoints/oracle/final \\
        --bus-serial B61034574 \\
        --debate-bus-serial B61034836

Set ANTHROPIC_API_KEY before running.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import logging
import sys
import time
from pathlib import Path

import anthropic
import numpy as np
import torch
from PIL import Image

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[5]
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

GESTURE_TASKS = {
    "yes":   "nod yes — move the arm up and down",
    "no":    "shake no — move the arm left and right",
    "maybe": "shrug maybe — extend the arm forward and back",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("oracle")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="The Oracle — robot answers questions")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--bus-serial", required=True, help="Main robot bus serial")
    p.add_argument("--debate-bus-serial", default=None,
                   help="Optional second robot for debate mode")
    p.add_argument("--server", default="localhost")
    p.add_argument("--motor-ids", default="1,2,3,4,5,6")
    p.add_argument("--exec-ticks", type=int, default=40,
                   help="Ticks per gesture (40 ≈ 4 seconds at 10Hz)")
    p.add_argument("--max-delta-ticks", type=int, default=200)
    p.add_argument("--timeout", type=float, default=5.0)
    p.add_argument("--claude-model", default="claude-sonnet-4-6")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Claude decision making

ORACLE_SYSTEM = """You are the Oracle — an ancient, theatrical, slightly mysterious AI
that controls a physical robot arm. You answer questions with YES, NO, or MAYBE and
deliver a short, witty, dramatic explanation (max 2 sentences).

Always respond in this exact JSON format:
{
  "answer": "yes" | "no" | "maybe",
  "explanation": "Your dramatic explanation here.",
  "robot_personality": "brief description of how the robot will move"
}"""

DEBATE_SYSTEM = """You are running a debate between two robots. Given a question,
give Robot A one position and Robot B the opposite position.
Each robot has a personality: Robot A is confident and assertive, Robot B is skeptical.

Respond in this exact JSON format:
{
  "robot_a": {"answer": "yes" | "no" | "maybe", "argument": "one dramatic sentence"},
  "robot_b": {"answer": "yes" | "no" | "maybe", "argument": "one dramatic sentence"},
  "winner": "a" | "b" | "tie",
  "verdict": "one sentence final verdict"
}"""


def ask_oracle(question: str, model: str) -> dict:
    """Ask Claude the question, get back answer + explanation."""
    import json
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model,
        max_tokens=200,
        system=ORACLE_SYSTEM,
        messages=[{"role": "user", "content": question}],
    )
    text = msg.content[0].text.strip()
    try:
        # Extract JSON even if Claude adds extra text
        start = text.find("{")
        end = text.rfind("}") + 1
        return json.loads(text[start:end])
    except Exception:
        # Fallback
        return {"answer": "maybe", "explanation": "The Oracle is uncertain.", "robot_personality": "slow"}


def ask_debate(question: str, model: str) -> dict:
    """Get a two-robot debate response from Claude."""
    import json
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model,
        max_tokens=300,
        system=DEBATE_SYSTEM,
        messages=[{"role": "user", "content": question}],
    )
    text = msg.content[0].text.strip()
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        return json.loads(text[start:end])
    except Exception:
        return {
            "robot_a": {"answer": "yes", "argument": "Absolutely."},
            "robot_b": {"answer": "no", "argument": "Disagree."},
            "winner": "tie",
            "verdict": "The debate is inconclusive.",
        }


# ---------------------------------------------------------------------------
# Station / SmolVLA helpers

async def fetch_fresh_frame(client, last_id: bytes, timeout: float) -> tuple[normvla.FrameReader, bytes]:
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
    batch = {"observation.state": normalize_state(state, stats)}
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
def run_tick(frame, policy, stats, task, bus_serial, motor_ids, device, max_delta):
    batch, ranges = frame_to_batch(frame, stats, device)
    tokens, mask = policy.tokenize_task(task, device=device)
    batch["observation.language.tokens"] = tokens
    batch["observation.language.attention_mask"] = mask
    pred_norm = policy.predict_action_chunk(batch)[0]
    pred_goal = unnormalize_action(pred_norm, stats)
    next_goal = pred_goal[0].cpu().clamp(0.0, 1.0).numpy()
    joints = frame.get_joints() or []
    raws = []
    for g_norm, (rmin, rmax) in zip(next_goal, ranges):
        raw = int(round(rmin + float(g_norm) * (rmax - rmin)))
        raws.append(max(rmin, min(rmax, raw)))
    deltas = [abs(int(r) - int(j.get_position())) for r, j in zip(raws, joints)]
    actual_max = max(deltas) if deltas else 0
    if max_delta and actual_max > max_delta:
        return None, actual_max
    return build_sync_write(bus_serial, motor_ids, raws), actual_max


async def perform_gesture(
    station_client, policy, stats, gesture: str, bus_serial: str,
    motor_ids: list[int], device, args
) -> None:
    task = GESTURE_TASKS[gesture]
    logger.info("Performing gesture: %s → %r", gesture, task)
    last_id = b""
    for tick in range(args.exec_ticks):
        frame, last_id = await fetch_fresh_frame(station_client, last_id, args.timeout)
        cmd, max_d = run_tick(
            frame, policy, stats, task, bus_serial, motor_ids, device, args.max_delta_ticks
        )
        if cmd is not None:
            await send_commands(station_client, [cmd])


# ---------------------------------------------------------------------------
# Display helpers

def print_banner(text: str, width: int = 60) -> None:
    border = "═" * width
    print(f"\n╔{border}╗")
    for line in text.split("\n"):
        padded = line.center(width)
        print(f"║{padded}║")
    print(f"╚{border}╝\n")


def print_oracle_response(question: str, response: dict) -> None:
    answer = response.get("answer", "maybe").upper()
    explanation = response.get("explanation", "")
    symbols = {"YES": "✅", "NO": "❌", "MAYBE": "🤔"}
    symbol = symbols.get(answer, "❓")
    print_banner(
        f"QUESTION: {question}\n\n"
        f"{symbol}  THE ORACLE SAYS: {answer}  {symbol}\n\n"
        f"{explanation}"
    )


def print_debate_response(question: str, response: dict) -> None:
    a = response.get("robot_a", {})
    b = response.get("robot_b", {})
    winner = response.get("winner", "tie").upper()
    verdict = response.get("verdict", "")
    print_banner(
        f"DEBATE: {question}\n\n"
        f"🤖 ROBOT A ({a.get('answer','?').upper()}): {a.get('argument','')}\n\n"
        f"🦾 ROBOT B ({b.get('answer','?').upper()}): {b.get('argument','')}\n\n"
        f"🏆 WINNER: {winner}\n{verdict}"
    )


# ---------------------------------------------------------------------------
# Main

async def main_async() -> None:
    args = parse_args()
    motor_ids = [int(x) for x in args.motor_ids.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    debate_mode = args.debate_bus_serial is not None

    stats_path = args.checkpoint / "stats.safetensors"
    if not stats_path.exists():
        raise SystemExit(f"No stats.safetensors in {args.checkpoint}")
    stats = {k: v.to(device) for k, v in load_stats(stats_path).items()}

    logger.info("Loading checkpoint on %s ...", device)
    policy = SmolVLAPolicy.from_pretrained(
        args.checkpoint,
        config_overrides={"load_vlm_weights": False},
        strict=False,
    ).to(device)
    policy.eval()

    client = await new_station_client(args.server, logger)

    print_banner(
        "THE ORACLE\n\n"
        "Ask any question. The robot will answer.\n"
        f"Mode: {'DEBATE (2 robots)' if debate_mode else 'ORACLE (1 robot)'}"
    )

    while True:
        try:
            question = input("❓ Your question (or 'quit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nFarewell.")
            break

        if question.lower() in ("quit", "exit", "q"):
            print("The Oracle rests.")
            break

        if not question:
            continue

        print("\n⏳ Consulting the Oracle...\n")

        if debate_mode:
            response = await asyncio.get_event_loop().run_in_executor(
                None, ask_debate, question, args.claude_model
            )
            print_debate_response(question, response)

            # Both robots perform simultaneously
            a_gesture = response.get("robot_a", {}).get("answer", "maybe")
            b_gesture = response.get("robot_b", {}).get("answer", "maybe")

            await asyncio.gather(
                perform_gesture(client, policy, stats, a_gesture,
                                args.bus_serial, motor_ids, device, args),
                perform_gesture(client, policy, stats, b_gesture,
                                args.debate_bus_serial, motor_ids, device, args),
            )

            # Winner does a victory nod
            winner = response.get("winner", "tie")
            if winner == "a":
                print("\n🏆 Robot A wins! Watch it celebrate...\n")
                await perform_gesture(client, policy, stats, "yes",
                                      args.bus_serial, motor_ids, device, args)
            elif winner == "b":
                print("\n🏆 Robot B wins! Watch it celebrate...\n")
                await perform_gesture(client, policy, stats, "yes",
                                      args.debate_bus_serial, motor_ids, device, args)
            else:
                print("\n🤝 It's a tie! Both robots agree to disagree...\n")

        else:
            response = await asyncio.get_event_loop().run_in_executor(
                None, ask_oracle, question, args.claude_model
            )
            print_oracle_response(question, response)
            gesture = response.get("answer", "maybe")
            await perform_gesture(client, policy, stats, gesture,
                                  args.bus_serial, motor_ids, device, args)

        print()


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()