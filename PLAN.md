# SESAME-3D lightweight study: plan (Step 1 decision)

> Sections 1-6 are the 2026-10-21 record (SmolVLA was later replaced by REKI-2B, see memory).
> Current decisions: Section 7.

Date: 2026-10-21. Laptop: RTX 3070 Ti (8 GB), 253 GB free, Claude Code logged in
(claude.ai account, no API key). Goal: a second simulation study for the paper
(Sec. IV-B of `paper_outline.txt`) that runs on the laptop, exercises the full
pipeline (3DSG -> LangGraph plan -> two-stage pruning -> per-row CBF-QP in front
of a VLA), and later carries the robust (uncertainty-aware) CBF.

## 1. Benchmark decision

### Primary: RoboCasa365 (MuJoCo) with the PandaOmron mobile manipulator

| item | value |
|---|---|
| simulator | robosuite 1.5.2 + RoboCasa365 (MuJoCo 3.8), `~/Documents/Chewy/code/third_party/{robocasa,robosuite}` |
| robot | PandaOmron: Franka arm on a holonomic base, 12-D action = base motion (4) + control-mode switch (1) + EE delta pose (6) + gripper (1), 20 Hz |
| scenes | procedurally generated kitchens (layouts x styles), tens of fixtures (counters, cabinets, stove, fridge, sink, ...) plus free objects from the lightwheel pack |
| policy | `lerobot/smolvla_robocasa` (SmolVLA, 0.9 GB, chunk 50, 3 cameras); fits the 8 GB GPU. `lerobot/pi05_robocasa` exists but is too large |
| env wrapper | `lerobot.envs.robocasa` (lerobot 0.6.2, editable clone in `~/GRA/AnyCam/third_party/lerobot`) |
| conda env | `sesame-robocasa` (clone of `anycam-libero` + robosuite master + robocasa `--no-deps` + qpsolvers/daqp + anthropic) |
| tasks (candidates) | navigation family: `NavigateKitchen` (450 steps). manipulation family: `PickPlaceCounterToStove` (600), `PickPlaceCounterToCabinet` (750), `CoffeeSetupMug` (600), `OpenDrawer` (750), `CloseFridge` (900). composite (both families, if the policy copes): `PrepareCoffee`, `KettleBoiling` |

Why: it is the only laptop-feasible benchmark with a *mobile* manipulator, so
both barrier families of the paper (base rectangle and end-effector spheres)
and both segment kinds (navigate / pick / place) are exercised, and the 3DSG has
real room-scale structure (fixtures as the "furniture" nodes, free objects on
counters). RoboCasa365 is also where the current VLA community reports
(pi0.5, GR00T, SmolVLA baselines), so a safety-filter result there is visible.

Risks (to be measured in the smoke test before committing):
1. Success rate of `smolvla_robocasa` is not published on the model card; if it
   is far below ~30% on the chosen tasks, task-success contrasts between arms are
   weak (safety contrasts still stand).
2. Kitchens have no "passable" objects (no curtains/mats). The semantic stage's
   job becomes the other half of the contract: exempting *task-relevant* contact
   objects (the target object, its receptacle, the fixture handle the task must
   touch) and dropping irrelevant ones, which a naive all-object filter cannot do
   without stalling. If a passable analog is wanted, a thin floor mat object can
   be injected from the SafeLIBERO/lightwheel assets.
3. Asset download is several GB (lightweight packs only; the objaverse pack,
   ~30 GB, is skipped and the object registry is restricted to lightwheel).

### Secondary (comparison to prior work, optional): SafeLIBERO on the fixed-base Franka

VLSA/AEGIS (Hu et al. 2025) built SafeLIBERO: 16 LIBERO tasks x 2 obstacle
levels x 50 init states, with CAR / TSR / ETS metrics; follow-ups (constrained
flow matching, VLPSA) report on it. Everything needed is already on the laptop
(`anycam-libero` env, `HuggingFaceVLA/smolvla_libero` cached, 75% baseline on
LIBERO-Object). The SafeLIBERO layer is a thin graft: 17 bddl files, 32
init-state files, one `obstacle_objects.py`, an asset folder, and a suite
registration (cloned to the scratchpad `downloads/vlsa/vlsa-aegis`). It gives a
same-benchmark comparison against AEGIS's one-obstacle ellipsoid CBF, but the
robot is fixed-base, so only the end-effector family is exercised. Keep it as a
half-day add-on once the RoboCasa pipeline runs.

## 2. Pipeline port (what is reused from `signal_3d`, what is new)

Source package: `~/Documents/Chewy/code/signal_3d/src/semantic_safe_vla` (158k
lines, OmniGibson + pink/pinocchio + vLLM). Only the contract-level pieces are
ported; the new package is `sesame_lite/` (pure numpy + qpsolvers/daqp).

| component | reuse | notes |
|---|---|---|
| oracle 3DSG | mimic `scene_graph/types.py` (`SceneNode` with pose + semi-axes, half_extents = semi_axes/sqrt(3)) | built from MuJoCo: fixtures from `env.fixtures` (pos, quat, size), free objects from `env.obj_body_id` + geom AABBs, robot node; rooms = one kitchen; categories from RoboCasa names |
| barriers | copy `safety/box_geometry.py` (point-vs-OBB, point-vs-cylinder, base-rectangle-vs-OBB SAT with log-sum-exp) and the needed parts of `safety/numerical.py` (fragments, active set, acceleration bounds) | pure numpy already |
| per-row QP | re-implement `_per_row_constraints` without pink: rows `J_h v >= -alpha(h) h + d`, alpha = 1 (10 when h < 0), d = 0.015 m/s, margin m = 0.005 m, EE sphere r = 0.05 m; variables = base twist (3) + EE linear velocity (3); DAQP via qpsolvers; fail-closed hold on infeasibility | the VLA's EE delta pose / dt is the nominal EE velocity, the base motion command / dt is the nominal base twist; the filtered velocities are converted back to the 12-D action |
| swept-volume admission | copy `safety/swept_volume.py` (beta 0.5 m, rho_v 1.0 m, ds 0.05 m) | for navigation the planned path is the straight/A* path to the fixture's base pose from RoboCasa's `compute_robot_base_placement_pose`; for manipulation the reach disc |
| semantic classifier | port `tool_calling/pruning_classifier.py` prompt, vocabulary, decode rules, `ConservativeFallbackClassifier`, `StickyPassabilityClassifier`, allow-list override | LLM backend replaced by Claude Code headless (below) |
| planner | port the LangGraph node structure and the five tools (`find_entities`, `get_planning_context`, `get_room_route`, `query_safety_context`, `commit_plan`) over the oracle graph | plan steps navigate / pick / place / pick_and_place; the RoboCasa task language string is the goal. An offline (scripted) plan per task is kept as a fallback, which the user allowed |
| evaluation | arms A (no filter, barriers passive), B (naive per-row CBF over every node), C (pruned); matched init states | metrics: min h, ticks with h < -m, CAR-style contact (non-task object displaced > 1 mm, as in SafeLIBERO), task success, steps to success, filter latency, row counts, classifier fallbacks |

### Layout

```
src/sesame_lite/            framework + simulation (pip install -e .)
  geometry/                 fragments, box_geometry, swept_volume: verbatim ports from signal_3d
  scene_graph.py            oracle 3DSG captured from the MuJoCo kitchen
  planner.py                segments, A* base path, offline plan, two-stage pruning bundle
  filter.py                 per-row CBF-QP (DAQP), action <-> realized-velocity mapping
  env.py                    SafetyFilterEnv(gym.Wrapper) over lerobot's RoboCasaEnv, NavFollower, SmolVLAPolicy
  agents/pruning_classifier.py   the LLM agent side: prompt, schema, decode, fail-closed/sticky wrappers, Claude Code backend
scripts/run_episode.py      one episode = one process (CLI)
scripts/run_campaign.sh     matched-seed campaign driver
outputs/                    receipts (JSON per episode); delete what is not evidence
```

### How the oracle 3DSG is built (`scene_graph.capture_scene_graph`)

No perception: like the OmniGibson oracle adapter in the paper, the graph is
read from simulator ground truth once per episode after `reset`.
- One node per RoboCasa *fixture* (counters, cabinets, drawers, appliances,
  sink, walls) and per free *object*, plus a robot node. Floors and the
  "backing" walls are dropped (as ceilings/floors were in OmniGibson).
- Geometry: for a fixture, the world AABB of its MuJoCo collision geoms
  (group 0, contype/conaffinity non-zero), after discarding RoboCasa's 1 cm
  marker geoms and geoms more than 3 m from the fixture pose; kitchen fixtures
  are axis-aligned up to 90 degrees so the AABB is tight. For a free object,
  an oriented box: the body quaternion plus the AABB of its collision geoms in
  the body frame. Half-extents are at least 1 mm.
- Semantics: category = RoboCasa class name in snake case (`hinge_cabinet`,
  `stove`, ...) or the object category; the name offered to the classifier is
  RoboCasa's natural-language fixture name or the category, with the measured
  extent appended, `"counter (approx 1.01 x 0.65 x 0.92 m box)"`, exactly as
  signal_3d formatted candidates.
- Node IDs are `fixture:<name>` / `object:<name>`; the snapshot id is the
  SHA-256 of the canonical node list; every receipt binds to it.
- Relations (on_top/inside) are not built because the plan is deterministic
  here; RoboCasa's placement configs can provide them if the LLM planner is
  wired in.

### Base actuator model

RoboCasa's base joints are MuJoCo velocity actuators (ctrlrange 1 m/s, 1 m/s,
1.5 rad/s; kv 1000/1000/1500) working against `frictionloss` 250 N, so the
steady speed is command - 0.25 m/s (0.17 rad/s for yaw) and zero below that
deadband (measured: 0.3 -> 0.05, 0.5 -> 0.24, 0.75 -> 0.47 m/s). The QP's
decision variable is the realized velocity; `filter.velocity_to_action`
inverts the deadband. Without this, a CBF's approach speed near an obstacle
(about alpha*h) falls under the deadband and the base stalls 25 cm away.

### LLM backend: Claude Code headless

No `ANTHROPIC_API_KEY` and no `ant` profile on the laptop, but Claude Code is
logged in. A one-shot call

```
claude -p --no-session-persistence --output-format json --model claude-opus-5-5 \
  --tools "" --effort low --system-prompt "<classifier prompt>" \
  --json-schema '<labels schema>' '<{"task_goal":..., "objects":[...]}>'
```

returned a schema-valid `structured_output` in 5.4 s wall (2.1 s API) on the
mat/counter/light probe (passable / obstacle / irrelevant). `--bare` must not be
used (it drops the OAuth login). One call per segment, so a few calls per
episode. The same backend serves the planner (one call per planner turn with
the tool schemas, or the offline plan).

## 3. Protocol for Step 2 (regenerate safe successful trials)

1. Smoke test: 5 episodes each of `NavigateKitchen`, `PickPlaceCounterToStove`,
   `CloseFridge` with the raw policy (arm A) to get success rates and timing.
2. Pick 2 navigation-heavy and 2 manipulation tasks; 10 matched init states
   each; run arms A/B/C (120 episodes). Record per-tick receipts like the
   OmniGibson campaign.
3. Fill Sec. IV-B of the outline: one table (safety, success, latency, rows)
   and one figure (a kitchen top view with the corridor, admitted set, and the
   trajectories of the three arms).

## 4. Step 3 sketch (robust CBF under 3DSG uncertainty)

Nothing in `signal_3d` implements this (docs/phase-6-uncertainty-tracking-* are
planning only; DEC-019 rejected plain geometry inflation as too conservative).
Plan:
- Noise model on the oracle graph: per-node pose error (translation bounded by
  eps_p, yaw by eps_th) and extent error bounded by eps_e, calibrated to the
  hardware anchor error reported in the paper; replay-based sweep first (no
  policy needed), then closed loop.
- Robust barrier: for bounded set-valued geometry the tightened barrier
  h_rob(x) = min over the uncertainty set of h(x; G) has a closed form for
  OBBs (eps_p + sqrt(3) eps_e plus a yaw term proportional to the box
  diagonal), which is less conservative than inflating every box by the
  worst case; alternatively an ISSf-style term on the row bound
  (d -> d + L_h * eps) that scales with the local gradient. Compare naive
  inflation vs the tightened barrier on admission recall, minimum clearance
  and task success.

## 5. Findings so far (2026-10-21)

- Memory, not the GPU, is the binding resource: one RoboCasa kitchen process
  peaks at ~6 GB RSS and the desktop uses ~7 GB of the 15 GB, so the laptop
  OOM-killed lerobot-eval on its second episode. Every episode now runs in its
  own process (`python -m sesame_lite.runner ...`); close Edge/VMs for campaigns.
- `lerobot/smolvla_robocasa` did not navigate (0/3 on `NavigateKitchen`, the
  base barely moves). Navigation segments therefore use a path follower on the
  planned polyline as the nominal policy, which mirrors the hardware deployment
  (Nav2 drove the base, the VLA the arm); the VLA is kept for manipulation
  segments on fixture tasks (`CloseFridge`, `OpenDrawer`, ...). Pick-and-place
  tasks need the 30 GB objaverse pack (object categories missing from the
  lightwheel registry), so they are excluded unless that pack is downloaded.
- Filter tick with 62 rows: ~16 ms (DAQP solve < 0.01 ms; the time is barrier
  evaluation in numpy), i.e. 20 Hz is comfortable.
- Classifier backend verified: `claude -p` returns schema-valid labels in ~5 s.

## 6. Status

- [x] Repos recovered: `code/{signal_3d,signal3d_hardware,dexcontrol-ros}`.
- [x] Env `sesame-robocasa`, lightweight assets (4.6 GB), RoboCasa pins.
- [x] `sesame_lite`: graph, barriers, QP, pruning, Claude backend,
      `SafetyFilterEnv(gym.Wrapper)` runner with nav / SmolVLA nominals.
- [x] Seed 0 of `NavigateKitchen` (nav nominal): A succeeds in 233 steps with
      60 contact ticks (h_min -9 cm); C succeeds in 200 steps with no contact
      (h_min -0.9 mm, recovery band only), 60 rows, 14.6 ms/tick.
- [ ] 10-seed campaign A/B/C on `NavigateKitchen` (`outputs/nav_prior`, running);
      summarize with `python scripts/summarize.py outputs/nav_prior`.
- [ ] Manipulation task with the VLA nominal; composite mission.
- [ ] Claude classifier on the real candidate lists; Step 3 robust CBF.
- [ ] SafeLIBERO graft (secondary).

## 7. 3DSG review decisions (2026-11-02, with Kasra)

1. **Edges.** Keep the minimal edge set (`on` from RoboCasa placement configs, no hierarchy).
   A single kitchen has no room/place layer to speak of. The paper states this plainly.
2. **"Irrelevant" allow-list is RoboCasa-specific.** Replace the OmniGibson list (ceiling,
   downlight, ...) with the RoboCasa categories that no segment can ever touch: `wall_accessory`,
   `window`, `window_proc`, plus whatever else turns up. Overhead fixtures (hood, upper cabinets)
   are left to the height test in item 3, so the list holds no categories that could actually be
   touched. Get the final names by listing the categories of the target-split kitchens (scene
   builds only, one process). Make it a config constant.
3. **Stage 1 = graph query, no LLM.** A purely geometric query on the 3DSG. Keep a node if any
   of its *parts* (not the node's union box) does both of the following:
   - intersects the corridor (navigation) or the reach region (VLA);
   - overlaps the robot's fixed vertical envelope [0, H]. H is computed once from the robot's
     collision geoms: body top at the navigation pose for nav segments, top of the arm's reach
     for VLA segments.
   Only the survivors go to the LLM. Report N -> stage 1 -> stage 2 counts.
4. **No real-time 3DSG feedback is claimed.** The graph is a snapshot taken at episode start.
   - Re-capturing at segment boundaries is allowed only if we cite a perception pipeline that
     updates the graph between tasks.
   - Tracking object and part poses inside a segment (SAM-3-style masks plus depth) is listed
     as a limitation / future work, not simulated with oracle updates.
   - Target parts that move are covered by contact exemptions.
5. **Frames.** Everything lives in the 3DSG/world frame. A part is stored as a world centre,
   a world quaternion and half-extents along its own axes. Fix `perturb` to apply yaw noise
   about world z (left-multiply). The fix is cosmetic for upright RoboCasa fixtures.
6. **Episode visualization and report (design below; build after sign-off).** Receipts stay
   JSON. `scripts/report.py` turns one episode into one self-contained HTML page: inline SVG
   plus vanilla JS, data embedded, no CDN, under 1 MB, per-tick data decimated to 5 Hz.
   Video only with `--video` (top view, 256 px).
   - **Header:** outcome, success, contacts, true h_min, rows B -> C, tick ms p50/p95,
     classifier calls.
   - **Map:** top view drawn from the 3DSG part footprints, each part coloured by status
     (outside stage 1 / stage-1 candidate / LLM passable / LLM irrelevant / admitted /
     exempt part). It also shows the planned path, the corridor or reach disc, the trajectory
     coloured by time, and segment boundaries.
   - **Scrubber:** a slider over ticks that moves the robot footprint and draws lines to the
     active rows. Strips show h_min, commanded vs filtered speed, and rows/tick.
   - **Admission audit (admitting too little or too much):** one row per part with status, the
     true minimum distance to the robot and contact ticks. Each row gets one of three flags:
     - MISSED (too little): not admitted, but came within delta = 0.10 m of the robot or was
       touched;
     - IDLE (too much): admitted, but never came closer than 0.5 m;
     - OK otherwise.
   - **Exemption audit:** the exact body names exempted per role, with min distance and contact
     ticks. This is how over-broad substring matches show up (e.g. both fridge doors).
   - **Classifier transcript:** the strings sent, the labels returned, and the overrides.
   - **Campaign index:** an `index.html` linking every episode, with sortable metrics and the
     MISSED/IDLE totals.
7. **Geometry source for planning vs filter.** Open: brainstorm on cuRobo / nvblox first.
