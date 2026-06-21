<!-- ════════════════════════════════════════════════════════════════════════
     HACKATHON SUBMISSION — Berlin AI × Robotics · "AI-Powered Robot Control"
     This section is the MAIN project. The original NormaCore platform README
     (the base we built on) follows further down.
     ════════════════════════════════════════════════════════════════════════ -->

# 🤖 The Supervisor — an LLM that controls (and rescues) a robot arm

### Berlin AI × Robotics hackathon · NormaCore "AI-Powered Robot Control" track

**We wired a large language model ([Claude](https://claude.com/claude-code)) directly into NormaCore's 8-DOF ElRobot arm.** It sees through the robot's cameras, reasons about the scene, drives the arm to do a real **pick-and-place** (pick a carrot off a sheet → drop it in a box → return home), *and* acts as a live **supervisor** over a fast SmolVLA neural policy: when the policy hovers, misses, or freezes, **Claude pauses it and takes over with corrected joint commands.**

> 📂 **All of our code lives in [`claude-supervisor/`](claude-supervisor/)** — see [its README](claude-supervisor/README.md) for the file-by-file guide.

---

## 🧠 The big picture (mind map)

```
                              ┌──────────────────────────────────┐
                              │   GOAL (in plain English):       │
                              │   "pick the carrot, put it in    │
                              │    the box, come home"           │
                              └────────────────┬─────────────────┘
                                               │
                        ╔══════════════════════▼══════════════════════╗
                        ║        CLAUDE   ⟷   ElRobot  (8-DOF arm)     ║
                        ║      an LLM as the robot's reasoning layer    ║
                        ╚══════════════════════╤══════════════════════╝
          ┌──────────────┬─────────────────────┼─────────────────────┬──────────────┐
          │              │                     │                     │              │
     ┌────▼────┐    ┌────▼─────┐         ┌─────▼─────┐         ┌─────▼─────┐   ┌────▼─────┐
     │  EYES   │    │  BRAIN   │         │   HANDS   │         │  SKILLS   │   │ TWO-BRAIN│
     │ 2 USB   │    │  Claude  │         │  ST3215   │         │ taught on │   │ supervise│
     │ cameras │    │  (LLM)   │         │  servos   │         │ hardware  │   │ the VLA  │
     └────┬────┘    └────┬─────┘         └─────┬─────┘         └─────┬─────┘   └────┬─────┘
          │              │                     │                     │              │
   observe() =     reason over the      bounded, absolute     • learned IK by   pause → correct
   image + joint   scene; decide;       joint targets via      demonstration     → resume:
   in ONE call     PREEMPT the policy   atomic sync-writes     (3×3 grasp grid)  Claude finishes
   each cycle      when it fails        (safety-clamped)      • overload-safe    the grasp the
                                                               creep motion       policy can't
                                                              • camera-freeze-
                                                                proof control
          │              │                     │                     │              │
          └──────────────┴──────────┬──────────┴─────────────────────┴──────────────┘
                                     ▼
              NormaCore  station  —  normfs API  (Protobuf over TCP :8888)
                              the real-time robotics platform
```

---

## ⚙️ The idea: two brains, two speeds

Learned robot policies (VLA models) are fast and reactive but **brittle** — they hover near the object, fumble a grasp, or freeze on a motor fault, with no way to notice and recover. We put an LLM *in the loop during execution* to watch and fix that.

```
   ┌────────────────────────────┐          ┌────────────────────────────────┐
   │  SLOW brain   ~0.3 Hz       │          │  FAST brain   ~10 Hz           │
   │  Claude (LLM session)       │ PREEMPT  │  SmolVLA policy on the GPU     │
   │  perceive → judge → act     │ ───────► │  observation → motor action   │
   │  → verify, and take over     │          │  (great at reaching,          │
   │  when the policy fails       │ ◄─────── │   shaky at the fine grasp)    │
   └────────────────────────────┘  resume   └────────────────────────────────┘
```

- **Fast brain — SmolVLA** (fine-tuned by us on our own teleop demos): reads both cameras + joint state at ~10 Hz and predicts the next motor target. Good at *gross* reaching.
- **Slow brain — Claude**: connected through a **custom MCP server we wrote**, at ~0.3 Hz. It reasons about whether the policy is succeeding and, when it isn't, **pauses the policy and issues direct corrected joint moves** to finish the grasp or recover from a fault, then resumes it.
- **Why it's novel:** prior LLM-on-robot work (SayCan, RT-H) uses the LLM at *plan time* and then trusts a black-box policy. Ours keeps the LLM as a **runtime preempt-and-correct safety layer** wrapped around the policy.
- **And** Claude can run the whole pick-and-place *by itself, no policy needed* — that's the reproducible hero demo, and it works flawlessly.

---

## 🛠️ How Claude actually controls the arm (the stack)

```
   Claude (LLM)
        │  calls safe tools:  observe · move_joints · set_gripper · pause_vla …
        ▼
   station_mcp.py        ← MCP server.  ALL safety limits live here, not in the model
        │                  (every target clamped to calibrated range + max step/call)
        ▼
   robot_lib.py          ← control layer over normfs:  observe(), atomic sync-write moves,
        │                  and current_st3215() — reads joints off the MOTOR BUS, so motion
        ▼                  never stalls when the camera feed freezes
   NormaCore station  →  ST3215 servo bus  →  ElRobot 8-DOF arm
```

| Layer | File | Role |
|---|---|---|
| **Tools for the LLM** | `claude-supervisor/station_mcp.py` | MCP server: `observe`, `move_joints`, `set_gripper`, `pause_vla`… **safety clamps enforced in the server, not the model** |
| **Control layer** | `claude-supervisor/robot_lib.py` | Wraps NormaCore's `normfs` API; the freeze-proof `current_st3215` motor-bus reader lives here |
| **Learned IK** | `claude-supervisor/grasp_model.py` + `grid_grasp.json` | Bilinear interpolation over a **hand-taught 3×3 grasp grid** → joint target for any `(u,v)` on the sheet |
| **Pick-and-place** | `claude-supervisor/smooth_pick.py` | The full overload-safe sequence: approach → grip → lift → carry → drop → home |
| **Teaching** | `claude-supervisor/teach_points.py`, `teach_drop.py` | Teach-by-demonstration: guide the arm by hand, it records the poses |

### The two engineering problems that decided whether this worked
1. **Camera-independent motion.** NormaCore's camera→inference bridge can *freeze*. If motion read joint state from the frozen camera frame, every move would stall mid-way. So all motion drives off the **live ST3215 motor bus** (`current_st3215`, with retries) — **the arm can't be blocked by a frozen camera.** This is why the manual demo is rock-solid.
2. **Overload-safe motion.** The shoulder (`j2`) faults under gravity if commanded straight to a reach pose. Fix: every move **creeps in small steps with pauses** (with auto stall-relief), and every lift **tucks the elbow in first, then raises the shoulder** — load stays near the base, no fault, every time.

*(We also diagnosed and fixed two deep station-side freeze bugs — a leader/follower bus-selection ambiguity, and the arm being left de-energized after a restart. Details in [`claude-supervisor/CLAUDE.md`](claude-supervisor/CLAUDE.md).)*

---

## ✅ Results

| Capability | Status |
|---|---|
| **LLM-driven manual pick-and-place** (carrot → box → home) | ✅ **Reliable & repeatable** — overload-safe, camera-freeze-proof. *The hero demo.* |
| **Fine-tuned SmolVLA deployed on the real arm** (loss ≈ 0.045) | ✅ Loads & runs clean — **autonomously reaches the carrot and attempts the grasp** |
| **Policy completing grip + lift on its own** | ⚠️ Inconsistent — it hovers/explores the grasp |
| **Supervisor closing the gap** (Claude preempts → finishes the grasp) | 🎯 Exactly the role the slow brain is built for |

**Headline:** *an LLM, given safe low-level tools and a learned IK, can perform a full real-world manipulation task end-to-end — and supervise a neural policy doing the same.*

---

## ▶️ Run it

```bash
# 1. Start the NormaCore station (arm + cameras attached). The -t flag is REQUIRED.
sudo ./claude-supervisor/fix_camera_perms.sh
./station --config claude-supervisor/station.yaml -t --web        # normfs :8888, web UI :8889

# 2. Drive the full pick-and-place yourself (camera-independent, overload-safe)
NORMA_CORE_REPO=$PWD python claude-supervisor/smooth_pick.py pickdrop <u> <v>

# 3. Run the fine-tuned SmolVLA policy on the arm (watch it!)
./claude-supervisor/run_hero_demo.sh

# 4. Launch Claude as the live supervisor (APPEND the prompt, never replace it)
claude --dangerously-skip-permissions --append-system-prompt "$(cat claude-supervisor/SUPERVISOR.md)"
```

**→ Full details, file-by-file, in [`claude-supervisor/README.md`](claude-supervisor/README.md).**

<br>

---
---

<!-- ════════════════════════════════════════════════════════════════════════
     Below: the original NormaCore platform README — the base we built on.
     ════════════════════════════════════════════════════════════════════════ -->

# NormaCore

### The Unified Toolkit for Physical System Development & Operations

> *The platform our hackathon project (above) is built on.*

**NormaCore** is a unified toolkit designed to facilitate the development and deployment of physical systems. From complex robotics to distributed sensor networks and hobby projects, the system provides a solid foundation to manage them all. To achieve this goal, the platform combines a unified API, high-performance data pipelines, and visual tooling to help you build and manage your entire ecosystem as one.

**Developer experience sits at the heart of our design philosophy.**

To fully realize the potential of this approach, we had to build a lot from scratch, rethinking traditional solutions from a practical perspective. This includes not just software, but complete hardware systems like our **7+1 DoF robotic arm** with a **parallel jaw gripper** - tools designed to open up a whole new dimension of applications for home and research robotics without significant cost or investment.

## What's inside

| Project | Path | Description |
|---|---|---|
| **🏆 Claude Supervisor** | [`claude-supervisor/`](claude-supervisor/) | **Our hackathon project (see top of this README):** an LLM controlling the ElRobot arm — pick-and-place + supervising/preempting the SmolVLA policy |
| **ElRobot** | [`hardware/elrobot/`](hardware/elrobot/) | Fully 3D-printed 7+1 DoF robotic arm for imitation learning |
| **Parallel Jaw Gripper** | [`hardware/pgripper/`](hardware/pgripper/) | Modular gripper for the SO-101 arm |
| **Station** | [`software/station/bin/station/`](software/station/bin/station/) | Real-time robotics platform — data collection, inference, control. Single binary, web UI |
| **SmolVLA fine-tune** | [`software/ai/smolvla_py/`](software/ai/smolvla_py/) | Train + deploy a [SmolVLA](https://huggingface.co/docs/lerobot/smolvla) policy on the SO-101 arm |
| **Gremlin** | [`shared/gremlin_go/`](shared/gremlin_go/) · [`shared/gremlin_py/`](shared/gremlin_py/) | High-performance Protobuf SDK for Go and Python — used across the station + drivers stack |

**Website:** [normacore.dev](https://normacore.dev)

**Follow us:**
- 🐦 [X/Twitter](https://x.com/norma_core_dev)
- 🎥 [YouTube](https://www.youtube.com/@normacoredev)
- 💼 [LinkedIn](https://www.linkedin.com/company/normacore/)
- 📢 [Reddit](https://www.reddit.com/r/NormaCore/)

**Join & Contribute:**
- 💬 [Discord](https://discord.gg/Z4Ytw3QfHP) - Chat with the community
- 🐙 [GitHub](https://github.com/norma-core/norma-core) - Source code & issues
