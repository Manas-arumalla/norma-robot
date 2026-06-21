"""Test yes / no / laugh gestures interactively.

Type 'yes', 'no', or 'laugh' at the prompt and the robot performs the gesture.

Run:
    uv run python gesture_test.py \
        --checkpoint ../../ai/smolvla_py/checkpoints/yes-no-laugh/final \
        --bus-serial 5B61034836
"""

from __future__ import annotations

import argparse
import asyncio
import io
import logging
import sys
from pathlib import Path

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

GESTURE_MAP = {
    "yes":   "nod yes",
    "no":    "shake no",
    "laugh": "laugh",
}

# Mean of first frames across all gesture training episodes
GESTURE_HOME = [0.497, 0.036, 0.439, 0.970, 0.481, 0.613, 0.483, 0.494]

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("gesture_test")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test yes/no/laugh gestures")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--bus-serial", required=True)
    p.add_argument("--server", default="localhost")
    p.add_argument("--motor-ids", default="1,2,3,4,5,6,7,8")
    p.add_argument("--exec-ticks", type=int, default=70,
                   help="Ticks per gesture (70 = ~7 sec at 10Hz)")
    p.add_argument("--replan-every", type=int, default=25,
                   help="Higher = smoother (predict once, follow chunk longer)")
    p.add_argument("--max-delta-ticks", type=int, default=1000)
    p.add_argument("--timeout", type=float, default=5.0)
    return p.parse_args()


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

    # Current raw positions
    current_raws = [int(j.get_position()) for j in joints[:len(motor_ids)]]

    # Target home raw positions
    home_raws = []
    for i, j in enumerate(joints[:len(motor_ids)]):
        rmin, rmax = int(j.get_range_min()), int(j.get_range_max())
        norm = GESTURE_HOME[i] if i < len(GESTURE_HOME) else 0.5
        home_raws.append(max(rmin, min(rmax, int(round(rmin + norm * (rmax - rmin))))))

    print(f"  [home] smoothly returning over {steps} steps ...")
    for step in range(1, steps + 1):
        t = step / steps  # 0.0 → 1.0
        interp = [
            int(round(cur + t * (tgt - cur)))
            for cur, tgt in zip(current_raws, home_raws)
        ]
        cmd = build_sync_write(bus_serial, motor_ids, interp)
        await send_commands(client, [cmd])
        await asyncio.sleep(0.05)  # 20Hz interpolation


async def perform_gesture(client, policy, stats, task, args, motor_ids, device):
    print(f"  → performing: '{task}' for {args.exec_ticks} ticks ...")
    last_id = b""
    sent = 0
    aborted = 0
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

        if args.max_delta_ticks and actual_max > args.max_delta_ticks:
            aborted += 1
            if aborted == 1:
                print(f"  [safety] max|Δ|={actual_max} > {args.max_delta_ticks} — increase --max-delta-ticks")
        else:
            cmd = build_sync_write(args.bus_serial, motor_ids[:n_pred], raws)
            await send_commands(client, [cmd])
            sent += 1

    print(f"  done — sent={sent} aborted={aborted}")


async def main_async():
    args = parse_args()
    motor_ids = [int(x) for x in args.motor_ids.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stats_path = args.checkpoint / "stats.safetensors"
    if not stats_path.exists():
        raise SystemExit(f"No stats.safetensors in {args.checkpoint}")
    stats = {k: v.to(device) for k, v in load_stats(stats_path).items()}

    print(f"Loading checkpoint from {args.checkpoint} ...")
    policy = SmolVLAPolicy.from_pretrained(
        args.checkpoint,
        config_overrides={"load_vlm_weights": False},
        strict=False,
    ).to(device)
    policy.eval()
    print("Ready.\n")

    client = await new_station_client(args.server, logger)

    print("=== Gesture Test ===")
    print("Commands: yes | no | laugh | quit\n")

    last_id = b""

    while True:
        try:
            gesture = await asyncio.get_event_loop().run_in_executor(
                None, lambda: input("Gesture: ").strip().lower()
            )
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if gesture in ("quit", "q", "exit"):
            print("Goodbye.")
            break

        if gesture not in GESTURE_MAP:
            print(f"  Unknown: '{gesture}'. Try: yes, no, laugh, quit")
            continue

        task = GESTURE_MAP[gesture]
        # Go to home position first so every gesture starts from same pose
        frame, _ = await fetch_fresh_frame(client, b"", args.timeout)
        await go_home(client, frame, args.bus_serial, motor_ids)
        await asyncio.sleep(0.3)  # brief pause at home before gesture starts
        await perform_gesture(client, policy, stats, task, args, motor_ids, device)
        print()


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
