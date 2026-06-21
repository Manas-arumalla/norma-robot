"""Read all inference tags from the running station and print frame numbers.

Run from norma-core root:
    python3 software/station/examples/read_tags.py
"""

import asyncio
import struct
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO / "software" / "station" / "shared"))
sys.path.insert(0, str(_REPO))

from station_py import new_station_client  # noqa: E402

TAGS_QUEUE = "inference-tags/rx"
MAX_TAGS = 1000


def decode_varint(data: bytes, pos: int):
    result = 0
    shift = 0
    while pos < len(data):
        b = data[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
    return result, pos


def decode_rx_envelope(data: bytes) -> dict:
    """Minimal proto3 decoder for inference_tags.RxEnvelope."""
    pos = 0
    fields = {}
    while pos < len(data):
        tag_val, pos = decode_varint(data, pos)
        field_num = tag_val >> 3
        wire_type = tag_val & 0x07
        if wire_type == 0:  # varint
            val, pos = decode_varint(data, pos)
            fields[field_num] = val
        elif wire_type == 2:  # length-delimited
            length, pos = decode_varint(data, pos)
            fields[field_num] = data[pos:pos + length]
            pos += length
        elif wire_type == 1:  # 64-bit
            fields[field_num] = data[pos:pos + 8]
            pos += 8
        elif wire_type == 5:  # 32-bit
            fields[field_num] = data[pos:pos + 4]
            pos += 4
        else:
            break  # unknown wire type, stop
    return fields


def ptr_to_decimal(ptr_bytes: bytes) -> int:
    """Convert little-endian bytes to decimal integer (inference-states frame ptr)."""
    padded = ptr_bytes + b'\x00' * (8 - len(ptr_bytes))
    return struct.unpack_from('<Q', padded[:8])[0]


async def main():
    import logging
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger("read_tags")

    client = await new_station_client("localhost", logger)

    # Read last MAX_TAGS entries from inference-tags/rx
    offset_bytes = struct.pack('<Q', MAX_TAGS)
    qr = client.read_from_tail(TAGS_QUEUE, offset=offset_bytes, limit=MAX_TAGS, step=1, buf_size=MAX_TAGS)

    tags = []
    while True:
        entry = await asyncio.wait_for(qr.data.get(), timeout=5.0)
        if entry is None:
            break
        raw = bytes(entry.Data)
        fields = decode_rx_envelope(raw)

        # field 10 = type (0=add, 1=remove)
        # field 11 = inference_queue_ptr (bytes)
        # field 12 = tag (string)
        cmd_type = fields.get(10, 0)
        ptr_bytes = fields.get(11, b"")
        tag_name = fields.get(12, b"")
        if isinstance(tag_name, (bytes, bytearray)):
            tag_name = tag_name.decode("utf-8", errors="replace")

        frame_num = ptr_to_decimal(ptr_bytes) if ptr_bytes else 0

        tags.append({
            "tag": tag_name,
            "frame": frame_num,
            "type": "remove" if cmd_type == 1 else "add",
        })

    # Sort by frame number
    tags.sort(key=lambda x: x["frame"])

    print(f"\n{'Frame':>18}  {'Type':6}  Tag")
    print("-" * 50)
    for t in tags:
        marker = "×" if t["type"] == "remove" else "+"
        print(f"{t['frame']:>18}  [{marker}]    {t['tag']}")

    print(f"\nTotal: {len(tags)} tag entries")

    # Show dataset-generator commands for the main tags
    tag_map = {}
    for t in tags:
        tag_map[t["tag"]] = t["frame"]

    print("\n\n=== Dataset pairs ===")
    # (label, start_tag, end_tag, task_desc, episode_duration_sec)
    pairs = [
        ("red",        "r_s",   "r_e",   "push the cap to the right", 45),
        ("blue",       "b_s",   "b_e",   "push the cap to the left",  45),
        ("other",      "0_s",   "o_e",   "push the cap forward",      45),
        ("other_caps", "a_a_s", "a_a_e", "push the cap forward",      45),
        ("yes",        "yes_start", "yes_stop", "nod yes",    10),
        ("no",         "no_start",  "no_stop",  "shake no",   10),
        ("laugh",      "laugh_start", "laugh_stop", "laugh",  10),
    ]
    for label, start_tag, end_tag, task_desc, ep_dur in pairs:
        s = tag_map.get(start_tag)
        e = tag_map.get(end_tag)
        if s and e:
            print(f"\n[{label}] {start_tag}={s} → {end_tag}={e}")
            print(f"  ./dataset-generator \\")
            print(f"    --from {s} \\")
            print(f"    --to   {e} \\")
            print(f"    --task \"{task_desc}\" \\")
            print(f"    -episode.duration {ep_dur} \\")
            print(f"    --output datasets/dataset_{label}")
        else:
            print(f"\n[{label}] MISSING: {start_tag}={s}, {end_tag}={e}")


if __name__ == "__main__":
    asyncio.run(main())
