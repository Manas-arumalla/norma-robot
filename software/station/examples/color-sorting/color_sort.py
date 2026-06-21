"""Color-sorting robot orchestrator.

Pipeline per cycle:
  1. Fetch a camera frame from inference/normvla.
  2. Send the image to Claude API — it identifies the color of the object
     inside the center box (red / blue / other / empty).
  3. Map color → task string:
       red   → "pick up the cap from the center box and place it to the right"
       blue  → "pick up the cap from the center box and place it to the left"
       other → "pick up the cap from the center box and move it forward"
       empty → wait and retry
  4. Run SmolVLA for --exec-ticks steps using that task string.
  5. Return to home position (optional --home-ticks), then repeat.

Run:
    uv run python color_sort.py \\
        --checkpoint ../../ai/smolvla_py/checkpoints/color-sort/final \\
        --bus-serial  B61034574 \\
        --camera-index 0

Requirements:
    ANTHROPIC_API_KEY env var must be set.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import logging
import sys
import time
from pathlib import Path

import base64
import boto3
import json
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

TASK_MAP_PUSH = {
    "red":   "push the cap to the right",
    "blue":  "push the cap to the left",
    "other": "push the cap forward",
}

TASK_MAP_PICKUP = {
    "red":   "pick up the cap from the center box and place it to the right",
    "blue":  "pick up the cap from the center box and place it to the left",
    "other": "pick up the cap from the center box and move it forward",
}

# Mean starting position across all training episodes (normalized 0-1 per joint)
# Derived from first frame of each training episode in dataset_red.parquet
HOME_POSITION_NORM = [0.531, 0.032, 0.464, 0.964, 0.496, 0.470, 0.494, 0.044]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("color_sort")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Color-sorting robot with Claude + SmolVLA")
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="SmolVLA checkpoint directory (contains model.safetensors, stats.safetensors, config.json)")
    p.add_argument("--bus-serial", required=True,
                   help="ST3215 bus serial from station web UI")
    p.add_argument("--server", default="localhost",
                   help="Station hostname (default: localhost)")
    p.add_argument("--motor-ids", default="1,2,3,4,5,6,7,8",
                   help="Comma-separated motor IDs")
    p.add_argument("--task-style", default="push", choices=["push", "pickup"],
                   help="push=color-sort-v2 checkpoint, pickup=original color-sort checkpoint")
    p.add_argument("--camera-index", type=int, default=0,
                   help="Which camera to use for color detection (0=cam0, 1=cam1)")
    p.add_argument("--exec-ticks", type=int, default=150,
                   help="Number of SmolVLA inference ticks to execute per object")
    p.add_argument("--replan-every", type=int, default=10,
                   help="Re-predict action chunk every N ticks (higher = smoother but less reactive)")
    p.add_argument("--max-delta-ticks", type=int, default=400,
                   help="Safety limit: abort if any joint would move more than this many ticks")
    p.add_argument("--poll-interval", type=float, default=1.5,
                   help="Seconds to wait between checks when box is empty")
    p.add_argument("--timeout", type=float, default=5.0)
    p.add_argument("--claude-model", default="us.anthropic.claude-haiku-4-5-20251001-v1:0",
                   help="Bedrock model ID for color detection")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Claude color detection

_bedrock_client = None

def get_bedrock_client():
    global _bedrock_client
    if _bedrock_client is None:
        _bedrock_client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
    return _bedrock_client


def detect_color(jpeg_bytes: bytes, camera_index: int, model: str) -> str:
    """Detect cap color using Claude via AWS Bedrock.

    Returns: 'red' | 'blue' | 'other' | 'empty'
    """
    # Crop to center third of image so background doesn't confuse Claude
    with Image.open(io.BytesIO(jpeg_bytes)) as im:
        w, h = im.size
        box = im.crop((w // 3, h // 3, 2 * w // 3, 2 * h // 3))
        buf = io.BytesIO()
        box.save(buf, format="JPEG")
        cropped_bytes = buf.getvalue()

    b64 = base64.standard_b64encode(cropped_bytes).decode()

    bedrock = get_bedrock_client()
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 10,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": b64,
                    },
                },
                {
                    "type": "text",
                    "text": (
                        "This image is taken under warm yellow lighting. "
                        "What is the dominant color of the cap or bottle cap in the center of this image? "
                        "Answer with exactly one word: red, blue, other, or empty. "
                        "- 'red' means the cap is red or orange-red even under warm light. "
                        "- 'blue' means the cap is clearly blue or dark blue. "
                        "- 'empty' means no cap is visible. "
                        "- 'other' means a cap is present but neither red nor blue."
                    ),
                },
            ],
        }],
    })
    response = bedrock.invoke_model(body=body, modelId=model)
    result = json.loads(response["body"].read())
    answer = result["content"][0]["text"].strip().lower()

    if   "red"   in answer: color = "red"
    elif "blue"  in answer: color = "blue"
    elif "empty" in answer: color = "empty"
    else:                   color = "other"

    print(f"  [Claude cam{camera_index}] raw={answer!r:10s}  → detected: {color.upper()}")
    return color


# ---------------------------------------------------------------------------
# Station / normvla helpers

async def fetch_frame(client, timeout: float) -> normvla.FrameReader:
    qr = client.read_from_tail(QUEUE_ID, offset=b"\x00", limit=1, step=1, buf_size=1)
    entry = await asyncio.wait_for(qr.data.get(), timeout=timeout)
    if entry is None:
        raise RuntimeError(f"{QUEUE_ID} closed")
    return normvla.FrameReader(memoryview(bytes(entry.Data)))


async def fetch_fresh_frame(
    client, last_id: bytes, timeout: float
) -> tuple[normvla.FrameReader, bytes]:
    while True:
        frame = await fetch_frame(client, timeout)
        fid = bytes(frame.get_global_frame_id())
        if fid != last_id:
            return frame, fid
        await asyncio.sleep(0.02)


def get_jpeg(frame: normvla.FrameReader, camera_index: int) -> bytes:
    images = frame.get_images() or []
    if len(images) <= camera_index:
        raise RuntimeError(f"Frame only has {len(images)} image(s), camera_index={camera_index} invalid")
    return bytes(images[camera_index].get_jpeg())


# ---------------------------------------------------------------------------
# SmolVLA inference helpers (mirrored from run_policy.py)

def frame_to_batch(
    frame: normvla.FrameReader,
    stats: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[dict, list[tuple[int, int]]]:
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


def build_sync_write(bus_serial: str, motor_ids: list[int], raw_goals: list[int]):
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
def predict_chunk(
    frame: normvla.FrameReader,
    policy: SmolVLAPolicy,
    stats: dict[str, torch.Tensor],
    task: str,
    device: torch.device,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Run one forward pass; return (chunk [T, n_joints], ranges)."""
    batch, ranges = frame_to_batch(frame, stats, device)
    tokens, mask = policy.tokenize_task(task, device=device)
    batch["observation.language.tokens"] = tokens
    batch["observation.language.attention_mask"] = mask

    pred_norm = policy.predict_action_chunk(batch)[0]   # [T, n_joints]
    n_pred = pred_norm.shape[-1]
    action_stats = {
        "action_mean": stats["action_mean"][:n_pred],
        "action_std":  stats["action_std"][:n_pred],
    }
    chunk = unnormalize_action(pred_norm, action_stats).cpu().clamp(0.0, 1.0).numpy()
    return chunk, ranges


def chunk_step_to_cmd(
    chunk: np.ndarray,
    step: int,
    ranges: list[tuple[int, int]],
    frame: normvla.FrameReader,
    bus_serial: str,
    motor_ids: list[int],
    max_delta: int,
) -> tuple[object, int]:
    """Convert one step of a pre-computed chunk to a motor command."""
    goals = chunk[step]                  # shape [n_joints]
    n_pred = len(goals)
    joints = frame.get_joints() or []
    raws: list[int] = []
    for g_norm, (rmin, rmax) in zip(goals, ranges[:n_pred]):
        raw = int(round(rmin + float(g_norm) * (rmax - rmin)))
        raws.append(max(rmin, min(rmax, raw)))

    deltas = [abs(int(r) - int(j.get_position())) for r, j in zip(raws, joints)]
    actual_max = max(deltas) if deltas else 0

    if max_delta and actual_max > max_delta:
        return None, actual_max

    cmd = build_sync_write(bus_serial, motor_ids[:n_pred], raws)
    return cmd, actual_max


# ---------------------------------------------------------------------------
# Home position

async def go_home(
    client,
    frame: normvla.FrameReader,
    bus_serial: str,
    motor_ids: list[int],
    stats: dict[str, torch.Tensor],
    steps: int = 30,
) -> None:
    """Smoothly interpolate from current position to training home over N steps."""
    joints = frame.get_joints() or []
    current_raws = [int(j.get_position()) for j in joints[:len(motor_ids)]]
    home_raws = []
    for i, j in enumerate(joints[:len(motor_ids)]):
        rmin, rmax = int(j.get_range_min()), int(j.get_range_max())
        norm = HOME_POSITION_NORM[i] if i < len(HOME_POSITION_NORM) else 0.5
        home_raws.append(max(rmin, min(rmax, int(round(rmin + norm * (rmax - rmin))))))

    print(f"  [home] smoothly returning over {steps} steps ...")
    for step in range(1, steps + 1):
        t = step / steps
        interp = [int(round(cur + t * (tgt - cur))) for cur, tgt in zip(current_raws, home_raws)]
        cmd = build_sync_write(bus_serial, motor_ids, interp)
        await send_commands(client, [cmd])
        await asyncio.sleep(0.05)


# ---------------------------------------------------------------------------
# Main loop

async def execute_task(
    client,
    policy: SmolVLAPolicy,
    stats: dict[str, torch.Tensor],
    task: str,
    args: argparse.Namespace,
    motor_ids: list[int],
    device: torch.device,
) -> None:
    """Run SmolVLA with temporal action chunking for smooth motion.

    Predicts a full action chunk every replan_every ticks and executes
    consecutive steps from it — avoids zig-zag from re-predicting each tick.
    """
    logger.info("Executing: %r  (%d ticks, replan every %d)",
                task, args.exec_ticks, args.replan_every)
    last_id = b""
    sent = 0
    aborted = 0
    chunk: np.ndarray | None = None
    ranges: list[tuple[int, int]] = []
    chunk_pos = 0

    for tick in range(args.exec_ticks):
        frame, last_id = await fetch_fresh_frame(client, last_id, args.timeout)

        # Re-predict when the chunk is exhausted or at replan interval
        if chunk is None or chunk_pos >= args.replan_every or chunk_pos >= len(chunk):
            chunk, ranges = predict_chunk(frame, policy, stats, task, device)
            chunk_pos = 0
            joints_now = frame.get_joints() or []
            cur = [f"{j.get_position_norm():.2f}" for j in joints_now]
            pred_first = [f"{chunk[0][k]:.2f}" for k in range(chunk.shape[1])]
            pred_last  = [f"{chunk[-1][k]:.2f}" for k in range(chunk.shape[1])]
            print(f"  [predict] current : {' '.join(cur)}")
            print(f"  [predict] chunk[0]: {' '.join(pred_first)}")
            print(f"  [predict] chunk[-1]: {' '.join(pred_last)}")

        cmd, max_d = chunk_step_to_cmd(
            chunk, chunk_pos, ranges, frame,
            args.bus_serial, motor_ids, args.max_delta_ticks,
        )
        chunk_pos += 1

        if cmd is None:
            aborted += 1
            logger.warning("tick %d/%d aborted — max|Δ|=%d > %d",
                           tick + 1, args.exec_ticks, max_d, args.max_delta_ticks)
        else:
            await send_commands(client, [cmd])
            sent += 1
            logger.info("tick %d/%d sent  max|Δ|=%d  chunk_pos=%d",
                        tick + 1, args.exec_ticks, max_d, chunk_pos)

    print(f"  [execute] sent={sent}  aborted={aborted}  (aborted=safety limit triggered)")
    logger.info("Done: %d sent, %d aborted", sent, aborted)


async def main_async() -> None:
    args = parse_args()
    motor_ids = [int(x) for x in args.motor_ids.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load policy once
    stats_path = args.checkpoint / "stats.safetensors"
    if not stats_path.exists():
        raise SystemExit(f"No stats.safetensors in {args.checkpoint}")
    stats = {k: v.to(device) for k, v in load_stats(stats_path).items()}

    logger.info("Loading checkpoint from %s on %s ...", args.checkpoint, device)
    policy = SmolVLAPolicy.from_pretrained(
        args.checkpoint,
        config_overrides={"load_vlm_weights": False},
        strict=False,
    ).to(device)
    policy.eval()
    logger.info("Policy loaded")

    # Connect to station once
    client = await new_station_client(args.server, logger)
    logger.info("Connected to station at %s", args.server)

    print("\n=== Color Sorting Robot ===")
    print(f"  Bus:      {args.bus_serial}")
    print(f"  Camera:   cam{args.camera_index}")
    print(f"  Model:    {args.claude_model}")
    print(f"  Safety:   max_delta_ticks={args.max_delta_ticks}")
    print(f"  Exec:     {args.exec_ticks} ticks per object")
    print("\nPlace an object in the center box. Ctrl-C to stop.\n")

    task_map = TASK_MAP_PUSH if args.task_style == "push" else TASK_MAP_PICKUP
    print(f"  Tasks:    {args.task_style} → {list(task_map.values())}")

    cycle = 0
    last_id = b""

    try:
        while True:
            # 1. Fetch a fresh camera frame
            frame, last_id = await fetch_fresh_frame(client, last_id, args.timeout)
            jpeg = get_jpeg(frame, args.camera_index)

            # 1b. Print current joint positions for debugging
            joints = frame.get_joints() or []
            pos_norm = [f"{j.get_position_norm():.2f}" for j in joints]
            print(f"  [joints] current: {' '.join(pos_norm)}")

            # 2. Ask Claude via Bedrock what color the object is
            color = await asyncio.get_event_loop().run_in_executor(
                None, detect_color, jpeg, args.camera_index, args.claude_model
            )

            # 3. Empty box → wait and retry
            if color == "empty":
                logger.info("Box is empty, waiting %.1fs ...", args.poll_interval)
                await asyncio.sleep(args.poll_interval)
                continue

            # 4. Map color → task string
            task = task_map[color]
            cycle += 1
            print(f"\n[cycle {cycle}] Detected: {color.upper()} → {task}")

            # 5. Execute with SmolVLA
            await execute_task(client, policy, stats, task, args, motor_ids, device)

            # 6. Return to home position
            home_frame, last_id = await fetch_fresh_frame(client, last_id, args.timeout)
            await go_home(client, home_frame, args.bus_serial, motor_ids, stats)

            print(f"[cycle {cycle}] Done. Place next object in the box.\n")
            await asyncio.sleep(args.poll_interval)

    except KeyboardInterrupt:
        print("\nStopped — returning to home position ...")
        try:
            home_frame, _ = await fetch_fresh_frame(client, last_id, args.timeout)
            await go_home(client, home_frame, args.bus_serial, motor_ids, stats)
            print("Home position reached.")
        except Exception as e:
            print(f"Could not go home: {e}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()