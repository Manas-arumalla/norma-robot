# Claude Supervisor — an LLM that controls (and rescues) a robot arm

**Berlin AI × Robotics hackathon · NormaCore "AI-Powered Robot Control" track**

> **One line:** A large language model ([Claude](https://claude.com/claude-code)) is wired directly into NormaCore's 8-DOF **ElRobot** arm — it sees through the robot's cameras, reasons about the scene, drives the arm to do a real **pick-and-place**, and acts as a live **supervisor** over a fast SmolVLA policy: when the policy stalls or misses, Claude **pauses it and takes over with corrected joint commands**.

---

## 1. The problem we set out to solve

Learned robot policies (like SmolVLA / VLA models) are fast and reactive, but **brittle** — they hover near the object, misjudge a grasp, or freeze when a motor faults, and they have no way to *notice* they failed and recover. Classic LLM-on-robot work (SayCan, RT-H) uses the LLM at *plan time* — to pick the next high-level step — and then hands control to a black-box policy with no oversight while it executes.

**Our question:** what if the LLM stays in the loop *during* execution — watching, judging, and **physically correcting** the policy in real time?

That's the system we built. We call it **The Supervisor**.

---

## 2. The architecture — two brains, two speeds

```
   ┌────────────────────────────┐          ┌────────────────────────────────┐
   │  SLOW brain   (~0.3 Hz)     │          │  FAST brain   (~10 Hz)         │
   │  Claude (this LLM session)  │          │  SmolVLA policy on the GPU     │
   │  perceive → judge → act     │  preempt │  observation → motor action   │
   │  → verify, and PREEMPT       │ ───────► │  (reactive, but brittle)      │
   └─────────────┬──────────────┘          └───────────────┬────────────────┘
                 │  MCP tools                                │  inference/normvla
                 │  (station_mcp.py)                         │  (shared-mem bridge)
                 └───────────────────┬───────────────────────┘
                                     ▼
                  NormaCore  station  —  normfs API (Protobuf / TCP :8888)
                       ST3215 servo bus   +   2× USB cameras
                                     ▼
                            ElRobot — 8-DOF arm + parallel-jaw gripper
```

- **Fast brain (SmolVLA).** Runs on the `inference/normvla` bridge at ~10 Hz: reads the two camera images + joint state, predicts the next joint target, writes it to the motors. Great at *gross* reaching; inconsistent at the *fine* grasp.
- **Slow brain (Claude).** Connected through a **custom MCP server** we wrote (`station_mcp.py`) at ~0.3 Hz. It calls `observe()` (image + joints in one shot), reasons about whether the policy is succeeding, and when it isn't, calls `pause_vla()` → issues **direct, bounded joint moves** to finish the grasp / recover from a fault → `resume_vla()`.
- **The novelty** is the **preempt-and-correct loop at runtime** — the LLM isn't just choosing a plan, it's a closed-loop safety-and-recovery layer wrapped around a learned controller.

And crucially: **Claude can also run the entire pick-and-place by itself, with no policy at all.** That's our guaranteed, reproducible demo — and it's the part that works flawlessly.

---

## 3. What we actually built (and the hard parts)

Everything below is in this folder, built on top of NormaCore's station in this repo.

### a) Claude's hands and eyes — the MCP server (`station_mcp.py`)
A [Model Context Protocol](https://modelcontextprotocol.io/) server that exposes the robot to the LLM as a handful of safe tools: `observe`, `get_joint_state`, `move_joints`, `set_gripper`, `pause_vla` / `resume_vla` / `set_vla_goal`, and a skill library (`save_skill` / `replay_skill`). **All safety limits live in the server, never in the model** — every target is clamped to the joint's calibrated range and to a max step per call, read tools are marked read-only and motion tools destructive. The model literally cannot command an out-of-range or oversized move.

### b) The control layer (`robot_lib.py`)
A thin, well-grounded wrapper over NormaCore's `normfs` Protobuf API: connect, `observe()` (camera JPEGs + normalized joint state together), atomic multi-joint **sync-writes**, and bus discovery. Control is normalized `[0,1]` per joint, mapped through each joint's calibrated range.

### c) Learned inverse kinematics by demonstration (`grasp_model.py` + `grid_grasp.json`)
We never wrote analytic kinematics. Instead we **hand-taught a 3×3 grid** of grasp poses over the work sheet (corners + edges + center) by guiding the arm, then **bilinearly interpolate** those poses to get a joint target for *any* `(u,v)` point on the sheet. Same idea for the drop motion (`teach_drop.py` → `drop_path.json`).

### d) The pick-and-place executor (`smooth_pick.py`)
Runs the full task as one autonomous, **overload-safe** sequence: `approach → grip → lift → carry to box → drop → come up → home`.

### e) A preemptible policy runner
We patched NormaCore's SmolVLA runner so it **re-reads its language goal and a pause flag from disk every loop** (`vla_goal.txt`, `vla_pause`) — that's what lets the supervisor change the goal or freeze the policy mid-run. And we **fine-tuned SmolVLA on our own teleoperated demonstrations** (final checkpoint, loss ≈ 0.045).

### Two engineering problems that actually decided whether this worked

1. **Camera-independent motion.** NormaCore's camera→inference bridge can *freeze* (we traced this to two root causes in the station: a bus-selection ambiguity with the leader+follower both connected, and the arm being de-energized after a restart). If motion read joint state from the (frozen) camera frame, every move would stall mid-way. So we made all motion drive off the **live ST3215 motor bus** (`current_st3215`, with retries) — **the arm cannot be blocked by a frozen camera.** This single decision is why the manual demo is rock-solid.

2. **Overload-safe motion.** The shoulder joint (`j2`) faults (`Overload`) if commanded straight to a reaching pose — gravity torque trips the servo and the whole arm goes dead. We solved it two ways: every move **creeps in small steps with pauses** (no torque spikes, with automatic stall-relief), and every lift **tucks the elbow in first, then raises the shoulder** so the load stays near the base. No faults, every time.

---

## 4. Results

| Capability | Status |
|---|---|
| **LLM-driven manual pick-and-place** (carrot off a sheet → into a box → home) | ✅ **Works reliably and repeatably** — overload-safe, camera-freeze-proof. This is the hero demo. |
| **SmolVLA fine-tuned + deployed on the real arm** | ✅ Loads and runs clean (no faults/skips); **autonomously reaches the carrot and attempts the grasp** |
| **Consistent autonomous grip + lift by the policy alone** | ⚠️ Inconsistent — the policy hovers/explores the grasp |
| **Supervisor closing that gap** (Claude preempts → completes the grasp) | 🎯 The exact role the slow brain is built for |

The headline: **an LLM, given safe low-level tools and a learned IK, can perform a full real-world manipulation task end-to-end — and can supervise a neural policy doing the same.**

---

## 5. Run it

```bash
# 1. Start the NormaCore station (arm + cameras attached). The -t flag is REQUIRED for the API.
sudo ./claude-supervisor/fix_camera_perms.sh
./station --config claude-supervisor/station.yaml -t --web        # normfs :8888, web UI :8889

# 2. Drive the full pick-and-place yourself (camera-independent, overload-safe)
NORMA_CORE_REPO=$PWD python claude-supervisor/smooth_pick.py pickdrop <u> <v>
#    approach -> grip -> lift -> carry to box -> drop -> come up -> home

# 3. Run the fine-tuned SmolVLA policy on the arm (watch it!)
./claude-supervisor/run_hero_demo.sh

# 4. Launch Claude as the live supervisor (APPEND the prompt — never replace it,
#    or you strip Claude Code's tooling)
claude --dangerously-skip-permissions --append-system-prompt "$(cat claude-supervisor/SUPERVISOR.md)"
```

> `robot_lib.py` imports the station's generated Python client from this repo — set `NORMA_CORE_REPO` to the repo root (where `software/` and `target/gen_python/` live).

---

## 6. What's in this folder

| File | What it is |
|---|---|
| `station_mcp.py` | The MCP server — Claude's safe tool interface to the robot |
| `robot_lib.py` | Control layer over NormaCore's normfs API (the freeze-proof `current_st3215` lives here) |
| `smooth_pick.py` | The full overload-safe pick-and-place executor |
| `grasp_model.py` + `grid_grasp.json` | Learned IK — bilinear interpolation over a hand-taught grasp grid |
| `drop_path.json`, `pick_recipe.json`, `grasp_offsets.md` | Taught drop path + gripper/lift constants + grasp corrections learned on hardware |
| `teach_points.py`, `teach_drop.py` | Teach-by-demonstration recorders (guide the arm by hand, it logs the poses) |
| `station_probe.py`, `slow_servos.py`, `fix_camera_perms.sh`, `run_hero_demo.sh` | Diagnostics + setup helpers |
| `.mcp.json`, `station.yaml` | MCP + station config (station.yaml pins the follower bus — one of our freeze fixes) |
| `SUPERVISOR.md` | Claude's control-loop system prompt (perceive → judge → act → verify; preempt-and-correct) |
| `PLAN.md` | Full plan, demo script, fallback tiers |
| `CLAUDE.md` | Deep engineering notes: normfs protocol, hardware gotchas (gripper = `j8`, `j7` = rotation, `j2` overload), the manual-pick pipeline |
