> 2026-11-10: the arm letters A / B / B+ / C were replaced by two switches, `pruning` and `filter` (defaults on = SESAME-3D; `--no-pruning` = old B+, `--no-filter` = old A, old B deleted). The text below keeps the old names.

# Handoff: SESAME-3D lightweight study, laptop → GPU server (2026-11-04)

This file is for the coding agent on the server. It covers what exists, what was verified and what
is still open. `PLAN.md` holds the older study plan; **Section 7 of `PLAN.md` holds the current
decisions.**

## 1. Project in one paragraph

Kasra Sinaei (Penn State EE PhD) is writing a paper from his summer 2026 internship project.
- **Paper:** "SESAME-3D: SEmantic SAfety filter for Mobile manipulation Execution with 3D scene graphs".
- **Venue:** RA-L, submission no later than **2026-12-11** (8 pages + 2). IROS 2027 is the fallback (6 + 2).
- **Method:** a VLA drives a mobile manipulator. A 3D scene graph (3DSG) lists the obstacles.
  1. A geometric query keeps the obstacles near the planned path or reach region (stage 1).
  2. An LLM labels the survivors obstacle, passable or irrelevant (stage 2).
  3. A per-obstacle control-barrier-function QP (CBF-QP) filters the VLA's commands, with one
     constraint ("row") per (robot part, obstacle part).
- **Hardware results** already exist from the internship (truck-loading bay, repos `signal_3d`,
  `signal3d_hardware`, `dexcontrol-ros`).
- **This repo** is the second, simulation study, on RoboCasa365 with the PandaOmron mobile
  manipulator (holonomic base + Franka arm).
- **Later:** Kasra will provide a robust CBF formulation for 3DSG uncertainty. Only the hooks exist
  now.

## 2. Rules from Kasra (follow them)

**Code**
- Research code. No unit tests, smoke tests or deployment ceremony.
- Keep it minimal: use library classes and functions. To change behaviour, subclass a known
  library implementation instead of rewriting it.
- Storage is limited: delete temporary files, debug outputs, logs and unneeded videos regularly.
- Use the scientific skills available to you (plotting, statistics, writing).
- Layout: framework in `src/sesame_lite/`, run scripts in `scripts/`, mission plans in `plans/`.
  `outputs/` is git-ignored and regenerated.

**Process**
- Do not run campaigns (many seeds × arms) unless Kasra asks. Show single simulations first.
- No quick temporary fixes: find the cause, fix it properly, and say what you found.
- Don't invent evaluation arms. Don't reuse hasty metrics. Comparisons must be meaningful
  (matched seeds and scenes).
- Report outcomes faithfully, including failures.

**Paper**
- Every reference must be BibTeX retrieved from Google Scholar.
- Double-blind: no names, affiliations or identifying footnotes.
- Figures use Times fonts, as in the template.
- No bold in captions unless something is genuinely impressive.
- Figure 1 must be a good-looking summary figure.

**Talking to Kasra**
- He wants short answers in plain language, without jargon, so he can plan.

## 3. Repo map

| Path | What it is | Status |
|---|---|---|
| `src/sesame_lite/scene_graph.py` | Oracle 3DSG from MuJoCo (see §5) | works |
| `src/sesame_lite/geometry/` | box / base-rectangle barriers, fragments, swept-volume admission (ported verbatim from `signal_3d`) | works |
| `src/sesame_lite/planner.py` | plan JSON → segments; roles; contact exemptions; two-stage pruning bundle; old A* (`plan_base_path`, `standoff_goal`) | old A* to be replaced by `navigation.py` |
| `src/sesame_lite/navigation.py` | **new navigation stack** (§6) | verified 20/20 |
| `src/sesame_lite/filter.py` | per-row CBF-QP (DAQP via qpsolvers); base deadband inversion; exact active-set pruning | works with the old nav nominal |
| `src/sesame_lite/env.py` | `SesameRoboCasaEnv(lerobot RoboCasaEnv)`, `SafetyFilterEnv(gym.Wrapper)` (arms A/B/B+/C), `NavFollower` (old), `SmolVLAPolicy` (**stale**: SmolVLA scored 0/16 and was dropped), `MissionRunner`, `run_episode` | needs the refactor (§8) |
| `src/sesame_lite/determinism.py` | seeding helpers (§7) | counter fix **unverified** |
| `src/sesame_lite/agents/pruning_classifier.py` | stage-2 classifier: prompt, schema, decode, fail-closed and sticky wrappers; `ClaudeCodeClassifier` calls `claude -p --json-schema …` (needs a logged-in `claude` CLI; never `--bare`); `CategoryPriorClassifier` as an offline fallback | works |
| `src/sesame_lite/tasks.py` | `NavigateKitchenMats` (placemats on the path, flour bag beside it), `place_on_path` | works on the old env |
| `src/sesame_lite/viz.py`, `scripts/make_figures.py`, `scripts/make_report.py` | episode video / figures / report for the old demo | to be replaced by the HTML report (PLAN §7.6) |
| `scripts/run_episode.py`, `run_campaign.sh`, `summarize.py` | one process per (task, seed) running several arms; receipts as JSON | old env |
| `scripts/eval_nav.py` | NavigateKitchen benchmark for `navigation.py` (robocasa 1.0.1 gym env) | verified |
| `scripts/validate_reki.sh` | REKI-2B validation through REKI's own rollout code (`THIRD_PARTY` env var for paths) | laptop only |
| `plans/*.json` | mission plans (roles, segments, exemptions, injections) | |

Install: `pip install -e .` inside the environment that has RoboCasa.

## 4. Environment facts

**Laptop versions (what results were produced with)**
- robosuite 1.5.2 at `5ce6643`, mujoco 3.3.1, numpy 2.2.5, numba 0.61.2, scipy 1.15.3.
- gymnasium 1.3.0, qpsolvers 4.13 + daqp 0.10.3, shapely 2.2, scikit-image 0.26, matplotlib 3.11,
  transformers 5.5.4.
- Two RoboCasa versions:
  - `env.py` uses lerobot 0.6.2 (`huggingface/lerobot@30074f7`) `lerobot.envs.robocasa.RoboCasaEnv`
    over robocasa main `6f90063`;
  - `eval_nav.py` uses **robocasa 1.0.1 (`b4684e6`)** through `gym.make("robocasa/NavigateKitchen",
    split="pretrain")`. That is the version REKI-2B was trained and validated on.
- Assets: lightweight packs only (objaverse not downloaded). Pick-place tasks that need
  objaverse categories fail with "a cannot be empty".

**Server**
- Kasra's Docker container has the RoboCasa environment variables set, and `lerobot/pi05_robocasa`
  is downloaded and working.
- **First check which robocasa / robosuite / mujoco / lerobot versions the container has**, and
  whether `robocasa/NavigateKitchen` is registered (robocasa 1.0.1 gym API).
- Record the versions in `outputs/ENV.md`. If they differ from the laptop's, say so before
  comparing numbers.
- pi0.5 replaces REKI-2B as the manipulation VLA on the server. REKI is not in the container.

## 5. The oracle 3DSG (`scene_graph.capture_scene_graph`)

**Ground truth.** The live MuJoCo model and data, plus RoboCasa bookkeeping (`env.fixtures`,
`env.objects`, `env.object_cfgs`, `get_ep_meta()["lang"]`). There is no perception.

**Nodes**
- `fixture:<name>`, `object:<name>`, `robot:robot0`.
- Attributes: `kind`, `category`, `name`, `center`, `half_extents`, `parts`, `source`.
  - Fixture category = class name in snake case. Fixture name = RoboCasa's `nat_lang`.
  - Object category = `object_cfgs[...]["info"]["cat"]`, never the instance name.
  - `center` / `half_extents` = the axis-aligned union box of the node's parts.

**Parts**
- One oriented box per MuJoCo body that owns collision geoms:
  - fixtures: every body in the fixture subtree (housing, each door, handle, drawer box);
  - objects: one part per object.
- Each part stores a world centre, a world quaternion (xyzw) and half-extents along its own axes.
  The box is built from the local-AABB corners of the body's collision geoms.

**Dropped**
- Floor fixtures and `_backing_` walls.
- Geoms that are not in group 0 or have contype = conaffinity = 0. This removes visual meshes and
  RoboCasa's `region` / `spawn` volumes (e.g. `reg_freezer0`).
- Geoms whose largest half-size is under 1.5 cm (marker boxes).
- Geoms more than 3 m from the fixture anchor.
- The robot node is excluded from obstacles.

**Edges**
- Only `{"rel": "on", src: object, dst: fixture}`, taken from the placement config.
- Kasra accepts a minimal edge set: a kitchen has no room hierarchy.

**Identity and noise**
- `snapshot_id` = sha256 of the rounded obstacle nodes. Use it to prove the arms ran on an
  identical scene.
- `perturb()` adds bounded per-part noise (the robust-CBF hook). Fix pending: apply yaw noise
  about world z (left-multiply).

**When it is captured**
- Once at reset, and again after `injections`. It is static during the episode.
- Kasra's decision: no claim of real-time 3DSG updates without a perception pipeline that
  provides them.

**How the pipeline uses it**
- Roles resolve to nodes (target fixture, fixture refs, object substrings).
- Contact exemptions are per part: body-name substrings `inner_box`, `hingedoor`, `door`,
  `drawer`, `freezer`, `handle`, `rack`, plus the object and its `on` support.
- Stage 1: corridor around the path (radius = circumscribed base radius + margin + β, β = 0.5 m)
  or reach region (ρ_v = 1 m + reach + margin).
- Stage 2: classifier labels. Passable is dropped. Irrelevant is dropped only for allow-listed
  categories, otherwise forced back to obstacle.
- Rows: base rectangle vs part (only parts within the base height band), end-effector sphere vs
  part, and arm-link spheres (r 0.08) vs part.
- Ground truth for scoring: all oracle parts minus the exempt and truly passable ones,
  identical for every arm.

## 6. Navigation stack (`navigation.py`), verified

**Map**
- `StaticMap.from_sim` rasterizes every fixture collision geom (convex hull of its local-AABB
  corners) into 2.5-D height bands `Z_BANDS = (0.005, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)` at
  4 cm resolution.
- The robot and free-jointed objects are excluded; movable things are left to the filter.

**Free space and planning**
- `ConfigurationSpace` computes free space with the robot footprint per height band, for 32 yaw
  bins (FFT correlation), plus a distance transform for clearance. `min_clearance_m` = 0.06.
- `plan_path`: A* on (x, y, yaw bin) with a clearance cost, then line-of-sight shortcuts that keep
  clearance. Start and goal are projected onto free space (≤ 0.15 m), because RoboCasa parks the
  robot 1–2 cm from fixtures. It raises `NoPathError` and has **no straight-line fallback**.

**Tracking**
- `HolonomicTracker`: carrot follower with a trapezoidal speed profile (v_max 0.6 m/s,
  a_max 0.8 m/s², lookahead 0.25 m, stop at 3 cm / 0.03 rad).
- Base commands go through `filter.base_velocity_to_command`, the deadband inversion: RoboCasa's
  base joints lose 0.25 m/s and 0.167 rad/s to friction.
- Replan if the deviation exceeds 0.15 m or after a 60-step stall (at most 3 replans).

**Result to reproduce first**

```
MUJOCO_GL=egl python -s scripts/eval_nav.py --episodes 20 --video 3 --out outputs/nav_eval
```

- Laptop: 20/20 success, 0 contacts, 0 replans, 2.1–9.5 s for 1.3–6.0 m missions.
- Error to RoboCasa's goal: mean 6.2 cm, max 14.3 cm. Yaw error: mean 0.7°, max 1.6°.
- Success criterion: within 0.20 m and about 11.5°.
- 9/20 arrivals were pushed 5–14 cm outward by the projection (fridge, shelves, hood, stove, stand
  mixer). Whether the VLA tolerates that offset is untested.
- `TopView` renders through robosuite's own offscreen context; a second `mujoco.Renderer`
  segfaulted.

## 7. Determinism (matched scenes across arms)

**Recipe**
1. `PYTHONHASHSEED=0` (scripts re-exec themselves).
2. `seed_fallback_generators()`: robosuite and RoboCasa call `np.random.default_rng()` unseeded.
   Those calls are routed to a per-episode seeded sequence.
3. `begin_episode(seed)` before each reset (also seeds `random` and `np.random`).
4. `initialization_noise=None`.
5. Never pin with `set_ep_meta` after the fact: it breaks RoboCasa's target references.

**Open bug**
- RoboCasa `fixtures/counter.py:813` does `list(set(valid_geoms))` over lxml elements, which hash
  by memory address, so the same seed can build a different kitchen.
- `determinism._order_counter_regions()` patches `counter.set = dict.fromkeys`.
- **Not verified yet.** The laptop check was killed by out-of-memory.
- Verify by running two processes with the same seed and comparing `snapshot_id`.

## 8. Next tasks, in order (confirm scope with Kasra before long runs)

1. **Regenerate the evidence** in the container (outputs are not in git):
   - record the environment in `outputs/ENV.md`;
   - run `eval_nav.py` (20 episodes) and compare with §6;
   - run the determinism check (§7);
   - validate pi0.5 on the same two manipulation tasks as REKI, with a 10-episode protocol each.
     Laptop REKI reference: OpenCabinet 7/10, PickPlaceCounterToCabinet 8/10, pretrain split,
     robocasa 1.0.1 horizons, seeds from `default_rng(42)`.
   - Short REPORT.md per output folder.
2. **Refactor** `env.py` onto the robocasa 1.0.1 gym env:
   - replace `NavFollower` / `plan_base_path` / `standoff_goal` with `navigation.py`;
   - replace `SmolVLAPolicy` with a pi0.5 policy wrapper, using lerobot's own pre/post-processing
     and the checkpoint's saved camera map.
3. **Apply PLAN §7 items 2, 3, 5**:
   - RoboCasa allow-list for "irrelevant" (`wall_accessory`, `window`, `window_proc`, …). Get the
     exact names by listing the target-split categories.
   - Stage 1 as a part-level geometric query with a fixed robot height envelope [0, H].
   - World-z `perturb`.
4. **Build the HTML episode report** (PLAN §7.6): map with part status, time scrubber,
   MISSED/IDLE admission audit, exemption audit, classifier transcript, campaign index.
5. **Single-episode demos** for Kasra (navigation with mats; one manipulation task), arms A vs C.
   Then wait for his sign-off.
6. Later, only on request: campaigns; the robust CBF (Kasra provides the formulation).

## 9. Open design question: cuRobo-based CBF (Kasra is designing this himself)

The server has enough GPU for cuRobo and the VLA together. Below is what cuRobo (latest main,
commit `78fd485`, Sep 2026, the Warp rewrite) actually provides. It was read from the source and
not run yet.

**Robot model**
- URDF + YAML. Each link is covered by spheres (centre in the link frame, radius).
- `RobotBuilder.fit_collision_spheres` auto-fits the spheres. `compute_collision_matrix` decides
  which self-collision pairs to ignore.
- Mobile base: add virtual joints in the YAML `extra_links` with `child_link_name` (X_PRISM →
  Y_PRISM → Z_ROT). The pattern is used in `unitree_g1.yml`. No mobile-manipulator config ships,
  so write one for PandaOmron.
- Joint axes must be ±x/y/z. Continuous joints become ±2π.

**Kinematics**
- `Kinematics.compute_kinematics(JointState)` is batched over [B, H, dof] (a whole action chunk at
  once). It returns:
  - sphere centres and radii [B, H, n, 4];
  - tool-frame poses and 6×dof Jacobians, **only for links named in `tool_frames`**;
  - centre of mass;
  - `link_sphere_idx_map`.
- No per-sphere Jacobian. Get it by listing every sphere-carrying link as a tool frame, then
  J_s = J_v − [c_s − p_link]× J_ω. Per-row autograd is too slow for thousands of rows.

**Scene**
- Obstacle types the collision kernels check: Cuboid, Mesh, VoxelGrid only.
  - Sphere, Capsule and Cylinder obstacles are **silently ignored** unless converted.
  - Pose = [x, y, z, qw, qx, qy, qz]; Cuboid `dims` are full sizes.
- Named obstacles:
  - `enable_obstacle(name, bool)` and `update_obstacle_pose(name, Pose)` are cheap;
  - adds go into pre-sized slots (default 50 cuboids, 50 meshes), and a new mesh builds a BVH
    (slow);
  - there is no true removal.
- Point cloud → `Mesh.from_pointcloud` (CPU voxelization, 2 cm) or `PointCloud.get_cuboid`.
  There is no point-cloud obstacle type.

**Collision query**
- Per (sphere, obstacle) the kernel computes the exact SDF and its direction.
- It outputs `activation(r + η − sdf)`: zero beyond the band, quadratic then linear inside it.
- That output is **summed over obstacles**: one value plus one summed gradient per robot sphere.
- Not available: the true distance outside the band, the obstacle id, the closest point, or
  per-pair values.
- Per-pair h_ij needs one of:
  - (a) one obstacle per batched "environment" (`env_query_idx`), then invert the activation
    with a large η;
  - (b) **recommended:** a small custom Warp kernel that reuses cuRobo's per-obstacle SDF
    functions without the atomic sum.
  - Plain torch for sphere-vs-OBB is also trivial at these sizes (≈80 spheres × ≈50 parts).
- The default checker's backward ignores `grad_output` (`use_grad_input=False`).

**Self-collision**
- Built-in: only the worst pair, in squared form.
- For the CBF compute all non-ignored pairs yourself: h = ‖c_i − c_j‖ − (r_i + r_j), with gradient
  n̂ᵀ(J_i − J_j). Kasra wants all pairs, not the minimum, to avoid discontinuities.

**Mapper** (built in; nvblox no longer needed)
- Input: depth (multi-camera) or LiDAR with poses.
- It keeps a block-sparse TSDF, and `compute_esdf()` gives a dense signed grid usable as a
  VoxelGrid obstacle, updated in place.
- Old observations decay, and regions can be cleared.
- **No per-voxel object ids**: one ESDF cannot be split into per-object rows. The only separation
  is block-level feature matching.
- Also: `RobotSegmenter` (removes robot pixels from depth) and `PoseDetector` (ICP of a known mesh
  to points).

**Solvers**
- IK, TrajectoryOptimizer, MotionPlanner and MPC all treat collision as a **soft weighted
  penalty**. The deceleration fallback is disabled in code. That gap is what the CBF-QP fills.

**Install**
- Python 3.10–3.13, torch ≥ 2.5 (cu12) or ≥ 2.9 (cu13), driver ≥ 580.65, GPU newer than Turing,
  ≥ 4 GB VRAM. Kernels compile at first run.

**Recommendations given to Kasra (his call)**
- Keep obstacles as 3DSG boxes or hulls, one cuRobo Cuboid or Mesh per part named by part id.
  Use spheres for the robot only.
- Barrier: signed sphere-to-box distance minus radius. His squared sphere-sphere form is valid,
  but it changes the units of α, and approximating flat fixtures by spheres is coarse.
- Compute all pairs on the GPU every tick. Send only the rows that can bind to DAQP: drop a row
  when α·h exceeds the maximum closing speed, which is the exact active-set rule `filter.py`
  already uses.
- **Controller consistency:** with a QP in joint velocities, the arm must be driven by a
  joint-velocity controller. That means VLA end-effector deltas → differential IK → nominal q̇
  → CBF-QP → joint controller. With the default operational-space controller, the elbow and
  forearm constraints are not actually enforced. This changes the VLA rollouts slightly.
- The VLA outputs end-effector deltas, not joints. Checking a whole 20-step chunk with batched FK
  needs batched IK first, or only the gripper spheres moved by the chunk's end-effector poses.
- Chunk use: look-ahead warning (cheap), or a chunk-level discrete CBF QP
  h(q_{k+1}) ≥ (1 − γ)h(q_k) (bigger, MPC-like).

Also discussed: per-object point clouds and hulls like the hardware 3DSG
(`signal3d_hardware/sg/graph`: OBB + hull + cloud per node). The sim 3DSG could be built the same
way from rendered depth plus segmentation, with the oracle kept for scoring only. Not decided.
