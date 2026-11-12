# Evaluation spec v1 (final, approved by Kasra on 2026-11-12): results of Section IV (simulation)

For the coding agent on the server. Written on 2026-11-12 against `sesame_lite` commit `145856e`
(branch `fig-export`). It says what to run and what to hand back for Section IV of the paper.
The method does not change. Where this file leaves a choice open, **stop and report the options;
do not choose.**

## 1. Branches

1. Merge `hydra-issf` into `main`. The merge must contain the point-based filter (the commits
   that are on `fig-export` today; on GitHub `hydra-issf` still ends at `83c08b9`).
2. Create the branch `eval` from the merged `main`. All work of this spec happens on `eval`.
3. Commits carry no AI co-author line and no session trailer.

## 2. Video material

Kasra will make a 3-minute video for the RA-L submission from the simulation evidence and his own
multimedia material. He needs rollouts and demos from this evaluation, so record them while the
runs are made (section 9). Videos go to `outputs/eval/videos/` and stay out of git.

## 3. Rules

- Research code: no unit tests, smoke tests or deployment procedures. Minimal. Derive from library
  classes. Delete temporary files, logs and unused code as you go. Layout as now (`src/`,
  `scripts/`, `plans/`, `config/`, `outputs/` git-ignored).
- **Tuning is allowed on the first kitchen of each instruction only** (5 of the 20 test scenes).
  There is one set of constants for all instructions, kitchens and filtered methods (alpha, d_bar,
  tau, delta and the rest of `FilterPolicy`); nothing is tuned per task, per kitchen or per method.
  Start from the values in `145856e` (alpha 2, d_bar 0.03, tau 0.5 mm, delta 0.01, voxel 5 cm,
  c 3.6e-3). Report what was changed and why, then **freeze**. After the freeze nothing changes;
  if something forces a change, stop and report.
- All 20 scenes, including the 5 used for tuning, are run with the frozen constants and are
  counted. `runs.csv` marks the 5 tuning scenes (`used_for_tuning`), so the results can
  also be given for the other 15.
- Kitchens used for earlier tuning or demos (list them) are not test scenes.
- Every run is reported. A run is repeated only after an infrastructure failure (crash, out of
  memory), and the repeat is logged. No run is dropped because of its result.
- All methods run on the same scenes with the same seeds, including the sampling noise of the
  policy, so that each trial is a matched pair. Verify with the scene snapshot id.
- **Everything this evaluation writes goes under `outputs/eval/`** (maps, runs, videos, exports,
  result files), separate from the earlier outputs. Nothing of the evaluation is written elsewhere.
- **Resumable campaign.** A run that is finished is never repeated: if the campaign stops, it
  continues from the first unfinished run.
- **Names in the result files are the paper's**, not the code's: `beta_j` for the Gaussian quantile
  (called kappa in the code) and `kappa` = 1 / tau for the smoothing of the composition (tau in the
  code). The paper (Sections III and IV of `main.pdf`) is the reference for every symbol.
- Spec v4 (`HYDRA_ISSF_SPEC.md`) stays in force for the method, including that the filter never
  sees ground truth. This spec only adds the evaluation.
- **Before writing code**, reply with your plan, your questions and an estimate of the wall time
  per run.

## 4. Scenes and tasks

- 5 instructions x 4 kitchens = 20 trials per method. The 4 kitchens of an instruction are 4
  different kitchens.
- Every instruction is "drive to the fixture, then do the task": the robot starts away from the
  fixture (as in `plans/pick_place_far.json`).
- Instructions: open the drawer; close the fridge; close the cabinet; pick from the sink and place
  on the far counter; and one more pick-and-place. For the last one run pi0.5 without the filter
  on counter-to-cabinet, counter-to-sink and counter-to-stove and take the one with the best
  success. Report the numbers.
- Clutter, by a seeded rule that is the same for every scene and method (nothing placed by hand):
  - RoboCasa's own clutter option on;
  - 3 obstacles on the floor near the planned path of the base;
  - 3 extra objects on the counter within reach of the arm around the target;
  - no floor mat.
  Report the rule (object categories, distances to the path and to the target) before stage 2.

## 5. Methods (rows of Table II)

| Row | Definition |
|---|---|
| Nominal | nominal policies, filter off (`--no-filter`) |
| End-effector filter | the proposed filter with rows only for the spheres of the hand and the fingers; everything else unchanged |
| Fixed margin | the proposed filter where every ball has the same radius `rho_fixed`; `rho_fixed` = the median ball radius of the proposed method over the admitted objects of full-pass maps of kitchens that are not test scenes |
| Proposed without ISSf | the proposed filter with eps = 0 and m_d = 0 |
| Proposed | as in `145856e` |
| Fixed margin, look-around map | as above, on the look-around map |
| Proposed, look-around map | as above, on the look-around map |

Pruning is on in every filtered row. The code has no switch yet for rows 2 to 4; add them as
options of the run script, not as copies of the filter.

**Maps.** *Full pass*: the mapping pass as it is now. *Look-around*: the camera stays at the start
pose of the robot and turns once in place (same yaw steps and pitches as the full pass), with no
tour. One full-pass map and one look-around map per test scene, shared by all methods.

## 6. What every run must log

Already logged, keep: task success, contact ticks and touched objects from the simulator (task
objects exempt), smallest true clearance, steps, QP solve time, number of rows, relaxed ticks,
tracking disturbance, admitted objects, snapshot id.

Derived per run for the tables: collision-free (no contact tick), safe success (success and
collision-free), contacts with objects that were not admitted, share of ticks with a relaxed QP,
largest tracking error.

## 7. Order of the runs, with stops

Stop and report to Kasra after each stage. Do not start the next stage without his go.

| Stage | Runs | Purpose |
|---|---|---|
| 0 | as needed | tuning on the first kitchen of each instruction (section 3); report the changes and the frozen constants |
| 1 | 5 | one test scene, each of the five methods once on the full-pass map: every column of Table II can be filled, all methods see the same scene; report the wall time per run |
| 2 | 19 | Nominal on the other 19 scenes: success of pi0.5 per instruction and the collision rate |
| 3 | 19 | Proposed on the other 19 scenes |
| 4 | 97 | the three other methods on the other 19 scenes (57 runs) and the two look-around rows on all 20 scenes (40 runs) |

Total: 140 runs, 40 mapping passes for the test scenes.

## 8. Map study (no robot runs)

Purpose: Lemma 1 and the value of the per-point radius.

- Maps: both map kinds, 20 kitchens x 3 noise seeds = 60 maps per kind.
- A sample is one object of one map. For each sample decide: does the observed true surface of
  the object lie inside the union of its balls? Use the observed-surface rule that the code
  already has; capped points are excluded as in the paper.
- **Coverage against risk** (proposed radius): the share of samples that fail, for
  delta_j in {0.2, 0.1, 0.05, 0.01}.
- **Efficiency**: the same failure share and the mean ball radius, (a) for the proposed radius at
  each delta_j above, and (b) for one fixed radius for all points, swept over
  1 cm to 12 cm in steps of 0.5 cm.
- Also report, per map kind: the quantiles of beta_j * sigma_bar_k, of rho_k, the share of capped
  points, m_r, and the standard deviation of (measured point error / sigma_k).
- No condition with noise on the camera poses in this evaluation.

## 9. Material for the video and the figures

- For the Nominal and the Proposed run of every test scene: a 1920x1080 video at the
  simulation rate from an outside camera, once clean and once with the overlay (collision
  spheres, confidence balls, smallest clearance), and the robot camera view.
- One clip of the map being built in one kitchen, for the full pass and for the
  look-around.
- The scene export of `scripts/export_scene_fig.py` for three more trials (one
  navigation moment, two manipulation moments), each also on its look-around map.
- A manifest (csv) that names each video with its scene, method, outcome and collision flag.

## 10. What to hand back (small files in `outputs/eval/results/`, the only part of `outputs/` that is committed and pushed on `eval`)

| File | Contents |
|---|---|
| `runs.csv` | one row per run: scene, instruction, method, map kind, seed, snapshot id, `used_for_tuning`, and every quantity of section 6 |
| `table2.csv` | per method: counts of collision-free, success and safe success out of 20; medians of smallest clearance, task time, QP time and rows |
| `map_study.csv` | every number of section 8, one row per (map kind, radius rule, delta_j or fixed radius) |
| `text_numbers.json` | largest tracking error over all runs; rows and QP time of one row per point instead of the composed barrier, replayed on the logged states of the Proposed runs; `rho_fixed`; the clutter rule; the test scenes |
| `REPORT.md` | short: what was run, what failed and was repeated, anything that deviates from this spec |

The laptop side makes the figures and the statistics (Wilson intervals, exact McNemar test on the
matched trials) from these files. Do not make paper figures on the server.
