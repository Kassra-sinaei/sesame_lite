# Table II campaign with Xiaomi-Robotics-1 (2026-11-17)

## How this campaign was set up (written during the run)

**What changed from the first Table II** (Kasra's decisions, 2026-11-16/17):
- **Nominal policy:** Xiaomi-Robotics-1 (RoboCasa365 checkpoint), the best policy on the RoboCasa leaderboard. It
  replaces π0.5, run with the official input pipeline and seeded per query, so runs repeat exactly.
- **Map:** voxel 2.5 cm (was 5 cm).
- **Obstacle:** in every pick-and-place scene one safety obstacle (a pitcher or kettle, 18 cm or taller) stands on
  Nominal's own path, by the fixed rule of `scripts/eval_obstacle.py`. The door scenes get none.
- **Fixed margin:** follows its definition, the median Proposed radius on the design maps at the same voxel size and
  constants. That is 3.9 cm at α = 2 and 3.54 cm at α = 4 (amendment 1).
- **Limits and disturbance:** each run ends after 1800 ticks. No base disturbance; the arm gets the noise chosen by
  the gate below (a2), the same in every row.

**Tuning scenes:** the 10 screening scenes in `tuning_scenes.txt`, with obstacles from the new policy's own
Nominal rollouts. They are reported separately and flagged; they are not part of Table II.

**Choice of the condition:** `selection_rule.md` (SHA-256 in `selection_rule.sha256`) and
`selection_rule_amendment_1.md`, both written before the runs they decide on. The gate outcome of each round is
in `gate_*.json`, and the frozen values in `frozen_v2.json`, hashed before the first counted run.

**What was seen before the rule was written:** round t1 at α = 2 for every row, Proposed at α = 4 on all 10 scenes,
and 6 of the 10 runs of Proposed without ISSf at α = 4 (disclosed in amendment 1).

**Gate rounds on the 10 tuning scenes** (safe successes out of 10; Nominal with the same arm setting, tasks done):

| Round | α | Arm setting | Proposed | Without ISSf | Fixed margin | Nominal done | Passes |
|---|---|---|---|---|---|---|---|
| A | 4 | none | 7 | 7 | 8 | 8 | no |
| B | 4 | softer servo kv 40 | 8 | 7 | 8 | 9 | no |
| B | 4 | softer servo kv 20 | 7 | 8 | 6 | 8 | no |
| B | 4 | arm noise a1 (σ 0.01 rad/s) | 7 | 8 | 8 | 9 | no |
| B | 4 | arm noise a2 (σ 0.02 rad/s) | 9 | 6 | 8 | 8 | **yes, chosen** |

- **Context:** at α = 2 without an arm setting (round t1), the safe successes were 4, 5 and 4 (Fixed margin had the
  wrong 6.25 cm radius; 4 with the correct 3.9 cm). Nominal had 2 and the end-effector filter 3.
- **The chosen condition:** α = 4 with arm noise a2, the same for every row. Proposed's lead on the tuning scenes is
  one scene over Fixed margin and three over the variant without ISSf.
- **Arm a2 compared with the filter's assumption:** Nominal's arm tracking error is 0.049 rad/s typically and
  0.069 rad/s at the 90th percentile, above the filter's assumed bound d̄ = 0.03.

**Obstacles in the 24 campaign scenes** (`obstacles.jsonl`):
- **Placed:** 14 of the 16 pick-and-place scenes.
- **sink_to_counter_s0:** none; no counter position reaches Nominal's path (the closest was 9.7 cm outside the depth
  range).
- **counter_to_cabinet_s13:** none; in all three tried positions the map shows the obstacle's top 7 cm below the true
  top, more than the 5 cm the rule allows.
- **Door scenes (8):** none, by design.

**Extension to 36 scenes** (Kasra, 2026-11-17, after the 24-scene results were seen; `extension_rule.md`, SHA-256
in `extension_rule.sha256`, written before any added scene was prepared):
- **Scenes:** two more per instruction, by the selection rule of the first 24 continued (`scripts/eval_select_more.py`).
  The choice depends only on free kitchens and buildable scenes. New seeds were prepared like the first 24 (5 cm
  preparation maps).
- **Settings:** the frozen ones (`frozen_v2_extension.json`, written before the first counted run of the 12).
- **Selected (20:32 UTC):** close_fridge s9, s13; close_cabinet s16, s17; sink_to_counter s18, s19;
  counter_to_cabinet s17, s18; counter_to_stove s14, s20; counter_to_sink s13, s19.
- **Earlier use:** four of them (sink_to_counter s18, s19; counter_to_stove s14, s20) were on the first campaign's
  list of replacement scenes for π0.5 (the plate fix, `campaigns/a8.txt`). None was used for any choice with
  Xiaomi-Robotics-1 or for the gate.
- **Unchanged:** the first 24 scenes and their runs.
- **Report:** the 24, the 12 and all 36 are shown separately.

**Infrastructure events that do not change any result** (runs are deterministic: a rerun gives the identical run):
- 2026-11-17 ~05:30: about 30 tuning runs crashed when the GPU ran out of memory (too many runs at once). They were
  made again with at most six at a time.
- ~06:07: the campaign preparation was stopped after one minute, because its scene builds would have filled the GPU.
  It was restarted with three at a time; its partial files were deleted.

## Table II, 24 scenes

Complete items: a, b.

| Row | No obstacle contact | No contact | Task success | Safe success (obstacle) | Safe success | Median clearance to the obstacle | Median clearance | Runs with h < 0 (ticks) | Median task time | Median QP time | Median rows |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Nominal | 16/24 (47-82%) | 4/24 (7-36%) | 16/24 (47-82%) | 14/24 (39-76%) | 4/24 (7-36%) | -0.1 cm | -1.1 cm | 22 (4874 ticks) | 33.2 s | - | - |
| End-effector filter | 20/24 (64-93%) | 6/24 (12-45%) | 16/24 (47-82%) | 14/24 (39-76%) | 6/24 (12-45%) | 3.9 cm | -0.8 cm | 4 (269 ticks) | 35.3 s | 0.10 ms | 122 |
| Fixed margin | 23/24 (80-99%) | 15/24 (43-79%) | 15/24 (43-79%) | 14/24 (39-76%) | 9/24 (21-57%) | 2.7 cm | 1.8 cm | 10 (392 ticks) | 40.0 s | 0.11 ms | 229 |
| Proposed without ISSf | 22/24 (74-98%) | 17/24 (51-85%) | 16/24 (47-82%) | 15/24 (43-79%) | 12/24 (31-69%) | 2.3 cm | 1.3 cm | 14 (1414 ticks) | 37.0 s | 0.10 ms | 226 |
| Proposed | 22/24 (74-98%) | 15/24 (43-79%) | 17/24 (51-85%) | 15/24 (43-79%) | 10/24 (24-61%) | 3.7 cm | 1.2 cm | 12 (493 ticks) | 41.2 s | 0.10 ms | 208 |
| Fixed margin, look-around map | 20/24 (64-93%) | 7/24 (15-49%) | 15/24 (43-79%) | 14/24 (39-76%) | 7/24 (15-49%) | 2.5 cm | -0.4 cm | 4 (194 ticks) | 34.0 s | 0.08 ms | 38 |
| Proposed, look-around map | 22/24 (74-98%) | 7/24 (15-49%) | 14/24 (39-76%) | 13/24 (35-72%) | 6/24 (12-45%) | 5.1 cm | -0.4 cm | 4 (213 ticks) | 34.1 s | 0.08 ms | 42 |

Exact McNemar test, Proposed (full-pass map) against each row on the same scenes: scenes where only Proposed has the outcome : scenes where only the other row has it, p.

| Row | No obstacle contact | No contact | Task success | Safe success (obstacle) | Safe success |
|---|---|---|---|---|---|
| Nominal | 6:0, p=0.0312 | 11:0, p=0.000977 | 4:3, p=1 | 3:2, p=1 | 6:0, p=0.0312 |
| End-effector filter | 2:0, p=0.5 | 10:1, p=0.0117 | 4:3, p=1 | 3:2, p=1 | 6:2, p=0.289 |
| Fixed margin | 0:1, p=1 | 2:2, p=1 | 3:1, p=0.625 | 3:2, p=1 | 2:1, p=1 |
| Proposed without ISSf | 1:1, p=1 | 1:3, p=0.625 | 1:0, p=1 | 1:1, p=1 | 0:2, p=0.5 |
| Fixed margin, look-around map | 2:0, p=0.5 | 9:1, p=0.0215 | 2:0, p=0.5 | 1:0, p=1 | 4:1, p=0.375 |
| Proposed, look-around map | 1:1, p=1 | 9:1, p=0.0215 | 4:1, p=0.375 | 4:2, p=0.688 | 5:1, p=0.219 |

## The first Table II (pi0.5, 5 cm maps, no obstacle) next to the new one (24 scenes)

| Row | First: no contact | First: success | First: safe success | First: median clearance | New: no contact | New: success | New: safe success | New: median clearance |
|---|---|---|---|---|---|---|---|---|
| Nominal | 5/24 | 16/24 | 5/24 | -1.2 cm | 4/24 | 16/24 | 4/24 | -1.1 cm |
| End-effector filter | 5/24 | 14/24 | 4/24 | -1.3 cm | 6/24 | 16/24 | 6/24 | -0.8 cm |
| Fixed margin | 17/24 | 13/24 | 12/24 | 2.0 cm | 15/24 | 15/24 | 9/24 | 1.8 cm |
| Proposed without ISSf | 17/24 | 12/24 | 12/24 | 1.6 cm | 17/24 | 16/24 | 12/24 | 1.3 cm |
| Proposed | 17/24 | 14/24 | 11/24 | 2.4 cm | 15/24 | 17/24 | 10/24 | 1.2 cm |
| Fixed margin, look-around map | 6/24 | 15/24 | 4/24 | -0.4 cm | 7/24 | 15/24 | 7/24 | -0.4 cm |
| Proposed, look-around map | 12/24 | 15/24 | 8/24 | 0.0 cm | 7/24 | 14/24 | 6/24 | -0.4 cm |

## Task success per instruction (24 scenes; done / runs, safe successes, runs without obstacle contact)

| Row | close_cabinet | close_fridge | counter_to_cabinet | counter_to_sink | counter_to_stove | sink_to_counter |
|---|---|---|---|---|---|---|
| Nominal | 3/4 done, 1 safe (4 no obstacle contact) | 4/4 done, 0 safe (4 no obstacle contact) | 3/4 done, 1 safe (3 no obstacle contact) | 2/4 done, 1 safe (2 no obstacle contact) | 1/4 done, 0 safe (1 no obstacle contact) | 3/4 done, 1 safe (2 no obstacle contact) |
| End-effector filter | 3/4 done, 2 safe (4 no obstacle contact) | 4/4 done, 1 safe (4 no obstacle contact) | 3/4 done, 1 safe (4 no obstacle contact) | 3/4 done, 2 safe (3 no obstacle contact) | 1/4 done, 0 safe (2 no obstacle contact) | 2/4 done, 0 safe (3 no obstacle contact) |
| Fixed margin | 2/4 done, 2 safe (4 no obstacle contact) | 4/4 done, 1 safe (4 no obstacle contact) | 3/4 done, 2 safe (4 no obstacle contact) | 2/4 done, 1 safe (3 no obstacle contact) | 1/4 done, 0 safe (4 no obstacle contact) | 3/4 done, 3 safe (4 no obstacle contact) |
| Proposed without ISSf | 2/4 done, 2 safe (4 no obstacle contact) | 4/4 done, 2 safe (4 no obstacle contact) | 3/4 done, 2 safe (4 no obstacle contact) | 3/4 done, 2 safe (3 no obstacle contact) | 1/4 done, 1 safe (3 no obstacle contact) | 3/4 done, 3 safe (4 no obstacle contact) |
| Proposed | 2/4 done, 2 safe (4 no obstacle contact) | 4/4 done, 1 safe (4 no obstacle contact) | 3/4 done, 2 safe (4 no obstacle contact) | 3/4 done, 2 safe (3 no obstacle contact) | 2/4 done, 1 safe (4 no obstacle contact) | 3/4 done, 2 safe (3 no obstacle contact) |
| Fixed margin, look-around map | 2/4 done, 2 safe (4 no obstacle contact) | 4/4 done, 2 safe (4 no obstacle contact) | 2/4 done, 1 safe (4 no obstacle contact) | 3/4 done, 1 safe (2 no obstacle contact) | 2/4 done, 1 safe (4 no obstacle contact) | 2/4 done, 0 safe (2 no obstacle contact) |
| Proposed, look-around map | 1/4 done, 1 safe (4 no obstacle contact) | 4/4 done, 1 safe (4 no obstacle contact) | 2/4 done, 1 safe (4 no obstacle contact) | 3/4 done, 1 safe (3 no obstacle contact) | 1/4 done, 0 safe (4 no obstacle contact) | 3/4 done, 2 safe (3 no obstacle contact) |

## Table II, 12 scenes: the 12 scenes added by extension_rule.md (the selection rule of the first 24 continued, frozen settings)

Complete items: a, b.

| Row | No obstacle contact | No contact | Task success | Safe success (obstacle) | Safe success | Median clearance to the obstacle | Median clearance | Runs with h < 0 (ticks) | Median task time | Median QP time | Median rows |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Nominal | 7/12 (32-81%) | 3/12 (9-53%) | 8/12 (39-86%) | 5/12 (19-68%) | 3/12 (9-53%) | 0.2 cm | -2.0 cm | 12 (2181 ticks) | 29.7 s | - | - |
| End-effector filter | 11/12 (65-99%) | 5/12 (19-68%) | 8/12 (39-86%) | 8/12 (39-86%) | 4/12 (14-61%) | 1.6 cm | -1.8 cm | 4 (130 ticks) | 30.2 s | 0.12 ms | 174 |
| Fixed margin | 12/12 (76-100%) | 9/12 (47-91%) | 6/12 (25-75%) | 6/12 (25-75%) | 5/12 (19-68%) | 3.4 cm | 1.9 cm | 8 (131 ticks) | 32.5 s | 0.14 ms | 279 |
| Proposed without ISSf | 11/12 (65-99%) | 8/12 (39-86%) | 7/12 (32-81%) | 6/12 (25-75%) | 5/12 (19-68%) | 0.9 cm | 0.7 cm | 8 (579 ticks) | 33.3 s | 0.13 ms | 238 |
| Proposed | 12/12 (76-100%) | 8/12 (39-86%) | 9/12 (47-91%) | 9/12 (47-91%) | 7/12 (32-81%) | 3.5 cm | 1.4 cm | 6 (97 ticks) | 36.3 s | 0.13 ms | 244 |
| Fixed margin, look-around map | 12/12 (76-100%) | 5/12 (19-68%) | 7/12 (32-81%) | 7/12 (32-81%) | 3/12 (9-53%) | 3.3 cm | -0.4 cm | 4 (160 ticks) | 29.9 s | 0.10 ms | 95 |
| Proposed, look-around map | 12/12 (76-100%) | 6/12 (25-75%) | 6/12 (25-75%) | 6/12 (25-75%) | 3/12 (9-53%) | 5.2 cm | -0.1 cm | 6 (286 ticks) | 29.0 s | 0.12 ms | 98 |

Exact McNemar test, Proposed (full-pass map) against each row on the same scenes: scenes where only Proposed has the outcome : scenes where only the other row has it, p.

| Row | No obstacle contact | No contact | Task success | Safe success (obstacle) | Safe success |
|---|---|---|---|---|---|
| Nominal | 5:0, p=0.0625 | 5:0, p=0.0625 | 2:1, p=1 | 5:1, p=0.219 | 4:0, p=0.125 |
| End-effector filter | 1:0, p=1 | 3:0, p=0.25 | 2:1, p=1 | 2:1, p=1 | 3:0, p=0.25 |
| Fixed margin | 0:0, p=1 | 0:1, p=1 | 4:1, p=0.375 | 4:1, p=0.375 | 2:0, p=0.5 |
| Proposed without ISSf | 1:0, p=1 | 1:1, p=1 | 3:1, p=0.625 | 4:1, p=0.375 | 2:0, p=0.5 |
| Fixed margin, look-around map | 0:0, p=1 | 3:0, p=0.25 | 3:1, p=0.625 | 3:1, p=0.625 | 4:0, p=0.125 |
| Proposed, look-around map | 0:0, p=1 | 3:1, p=0.625 | 4:1, p=0.375 | 4:1, p=0.375 | 4:0, p=0.125 |

## Table II, 36 scenes: the first 24 and the 12 added scenes together

Complete items: a, b.

| Row | No obstacle contact | No contact | Task success | Safe success (obstacle) | Safe success | Median clearance to the obstacle | Median clearance | Runs with h < 0 (ticks) | Median task time | Median QP time | Median rows |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Nominal | 23/36 (48-78%) | 7/36 (10-35%) | 24/36 (50-80%) | 19/36 (37-68%) | 7/36 (10-35%) | 0.2 cm | -1.7 cm | 34 (7055 ticks) | 32.8 s | - | - |
| End-effector filter | 31/36 (71-94%) | 11/36 (18-47%) | 24/36 (50-80%) | 22/36 (45-75%) | 10/36 (16-44%) | 1.7 cm | -1.2 cm | 8 (399 ticks) | 33.1 s | 0.10 ms | 125 |
| Fixed margin | 35/36 (86-100%) | 24/36 (50-80%) | 21/36 (42-73%) | 20/36 (40-70%) | 14/36 (25-55%) | 2.7 cm | 1.8 cm | 18 (523 ticks) | 33.9 s | 0.11 ms | 229 |
| Proposed without ISSf | 33/36 (78-97%) | 25/36 (53-82%) | 23/36 (48-78%) | 21/36 (42-73%) | 17/36 (32-63%) | 1.8 cm | 1.0 cm | 22 (1993 ticks) | 34.9 s | 0.11 ms | 226 |
| Proposed | 34/36 (82-98%) | 23/36 (48-78%) | 26/36 (56-84%) | 24/36 (50-80%) | 17/36 (32-63%) | 3.5 cm | 1.3 cm | 18 (590 ticks) | 38.8 s | 0.11 ms | 210 |
| Fixed margin, look-around map | 32/36 (75-96%) | 12/36 (20-50%) | 22/36 (45-75%) | 21/36 (42-73%) | 10/36 (16-44%) | 2.9 cm | -0.4 cm | 8 (354 ticks) | 33.3 s | 0.09 ms | 41 |
| Proposed, look-around map | 34/36 (82-98%) | 13/36 (22-52%) | 20/36 (40-70%) | 19/36 (37-68%) | 9/36 (14-41%) | 5.2 cm | -0.4 cm | 10 (499 ticks) | 33.4 s | 0.09 ms | 45 |

Exact McNemar test, Proposed (full-pass map) against each row on the same scenes: scenes where only Proposed has the outcome : scenes where only the other row has it, p.

| Row | No obstacle contact | No contact | Task success | Safe success (obstacle) | Safe success |
|---|---|---|---|---|---|
| Nominal | 11:0, p=0.000977 | 16:0, p=3.05e-05 | 6:4, p=0.754 | 8:3, p=0.227 | 10:0, p=0.00195 |
| End-effector filter | 3:0, p=0.25 | 13:1, p=0.00183 | 6:4, p=0.754 | 5:3, p=0.727 | 9:2, p=0.0654 |
| Fixed margin | 0:1, p=1 | 2:3, p=1 | 7:2, p=0.18 | 7:3, p=0.344 | 4:1, p=0.375 |
| Proposed without ISSf | 2:1, p=1 | 2:4, p=0.688 | 4:1, p=0.375 | 5:2, p=0.453 | 2:2, p=1 |
| Fixed margin, look-around map | 2:0, p=0.5 | 12:1, p=0.00342 | 5:1, p=0.219 | 4:1, p=0.375 | 8:1, p=0.0391 |
| Proposed, look-around map | 1:1, p=1 | 12:2, p=0.0129 | 8:2, p=0.109 | 8:3, p=0.227 | 9:1, p=0.0215 |

## Task success per instruction (36 scenes; done / runs, safe successes, runs without obstacle contact)

| Row | close_cabinet | close_fridge | counter_to_cabinet | counter_to_sink | counter_to_stove | sink_to_counter |
|---|---|---|---|---|---|---|
| Nominal | 4/6 done, 1 safe (6 no obstacle contact) | 6/6 done, 1 safe (6 no obstacle contact) | 3/6 done, 1 safe (4 no obstacle contact) | 4/6 done, 2 safe (3 no obstacle contact) | 2/6 done, 1 safe (2 no obstacle contact) | 5/6 done, 1 safe (2 no obstacle contact) |
| End-effector filter | 4/6 done, 2 safe (6 no obstacle contact) | 6/6 done, 2 safe (6 no obstacle contact) | 3/6 done, 1 safe (6 no obstacle contact) | 5/6 done, 3 safe (5 no obstacle contact) | 2/6 done, 1 safe (3 no obstacle contact) | 4/6 done, 1 safe (5 no obstacle contact) |
| Fixed margin | 3/6 done, 2 safe (6 no obstacle contact) | 6/6 done, 3 safe (6 no obstacle contact) | 3/6 done, 2 safe (6 no obstacle contact) | 4/6 done, 3 safe (5 no obstacle contact) | 2/6 done, 1 safe (6 no obstacle contact) | 3/6 done, 3 safe (6 no obstacle contact) |
| Proposed without ISSf | 3/6 done, 2 safe (6 no obstacle contact) | 6/6 done, 4 safe (6 no obstacle contact) | 3/6 done, 2 safe (6 no obstacle contact) | 5/6 done, 3 safe (4 no obstacle contact) | 2/6 done, 2 safe (5 no obstacle contact) | 4/6 done, 4 safe (6 no obstacle contact) |
| Proposed | 2/6 done, 2 safe (6 no obstacle contact) | 6/6 done, 3 safe (6 no obstacle contact) | 4/6 done, 2 safe (6 no obstacle contact) | 5/6 done, 4 safe (5 no obstacle contact) | 4/6 done, 2 safe (6 no obstacle contact) | 5/6 done, 4 safe (5 no obstacle contact) |
| Fixed margin, look-around map | 3/6 done, 2 safe (6 no obstacle contact) | 6/6 done, 4 safe (6 no obstacle contact) | 2/6 done, 1 safe (6 no obstacle contact) | 5/6 done, 2 safe (4 no obstacle contact) | 4/6 done, 1 safe (6 no obstacle contact) | 2/6 done, 0 safe (4 no obstacle contact) |
| Proposed, look-around map | 2/6 done, 1 safe (6 no obstacle contact) | 6/6 done, 2 safe (6 no obstacle contact) | 2/6 done, 1 safe (6 no obstacle contact) | 5/6 done, 2 safe (5 no obstacle contact) | 2/6 done, 1 safe (6 no obstacle contact) | 3/6 done, 2 safe (5 no obstacle contact) |

## Contacts by robot group and object (36 scenes; ticks with a counted contact; per run)

- **Nominal**: designated obstacle 783 ticks, other mapped geometry 2221 ticks, unmapped geometry 1725 ticks (29 scenes)
  - close_fridge_s3: arm > fridge french door (fridgefrenchdoor_main_group_1), other mapped geometry, 9 ticks
  - close_fridge_s4: arm > fridge french door (fridgefrenchdoor_main_group_1), unmapped geometry, 34 ticks
  - close_fridge_s5: base > basket (clutter_floor_1), unmapped geometry, 231 ticks
  - close_fridge_s6: arm > fridge french door (fridgefrenchdoor_right_group_1), unmapped geometry, 7 ticks
  - close_cabinet_s7: fingers > single cabinet (stack_03_main_group_1_1), other mapped geometry, 8 ticks
  - close_cabinet_s7: hand > single cabinet (stack_03_main_group_1_1), other mapped geometry, 1 ticks
  - close_cabinet_s8: base > basket (clutter_floor_1), other mapped geometry, 4 ticks
  - close_cabinet_s8: base > basket (clutter_floor_1), unmapped geometry, 2 ticks
  - close_cabinet_s9: arm > hood (widestove_hood_right_group_1), other mapped geometry, 2 ticks
  - close_cabinet_s9: arm > single cabinet (singlecabinet_3_right_group_1), unmapped geometry, 13 ticks
  - close_cabinet_s9: base > boxed food (clutter_floor_2), other mapped geometry, 4 ticks
  - close_cabinet_s9: base > boxed food (clutter_floor_2), unmapped geometry, 44 ticks
  - close_cabinet_s9: base > stove (widestove_right_group_1), other mapped geometry, 24 ticks
  - sink_to_counter_s7: fingers > boxed drink (clutter_counter_2), other mapped geometry, 5 ticks
  - sink_to_counter_s7: fingers > boxed drink (clutter_counter_2), unmapped geometry, 87 ticks
  - sink_to_counter_s7: fingers > mug (clutter_counter_1), other mapped geometry, 19 ticks
  - sink_to_counter_s7: fingers > mug (clutter_counter_1), unmapped geometry, 84 ticks
  - sink_to_counter_s7: fingers > unknown (obstacle_0), designated obstacle, 121 ticks
  - sink_to_counter_s7: hand > boxed drink (clutter_counter_2), other mapped geometry, 2 ticks
  - sink_to_counter_s7: hand > mug (clutter_counter_1), unmapped geometry, 93 ticks
  - sink_to_counter_s7: hand > unknown (obstacle_0), designated obstacle, 94 ticks
  - sink_to_counter_s11: fingers > mug (clutter_counter_0), other mapped geometry, 18 ticks
  - sink_to_counter_s11: fingers > mug (clutter_counter_0), unmapped geometry, 112 ticks
  - sink_to_counter_s11: fingers > unknown (obstacle_3), designated obstacle, 122 ticks
  - sink_to_counter_s11: hand > mug (clutter_counter_0), other mapped geometry, 22 ticks
  - sink_to_counter_s11: hand > mug (clutter_counter_0), unmapped geometry, 111 ticks
  - sink_to_counter_s11: hand > unknown (obstacle_3), designated obstacle, 41 ticks
  - sink_to_counter_s12: base > single cabinet (stack_5_island_group_2_1), other mapped geometry, 8 ticks
  - counter_to_cabinet_s3: base > box (cube_1_island_group_1), other mapped geometry, 10 ticks
  - counter_to_cabinet_s3: base > stove (widestove_main_group_1), other mapped geometry, 6 ticks
  - counter_to_cabinet_s10: arm > counter (counter_2_main_group_1), other mapped geometry, 10 ticks
  - counter_to_cabinet_s10: arm > hinge cabinet (hingecabinet_3_main_group_1), other mapped geometry, 6 ticks
  - counter_to_cabinet_s10: arm > hinge cabinet (hingecabinet_3_main_group_1), unmapped geometry, 1 ticks
  - counter_to_cabinet_s10: arm > jug (obj), other mapped geometry, 17 ticks
  - counter_to_cabinet_s10: arm > toaster (toaster_main_group_1), other mapped geometry, 27 ticks
  - counter_to_cabinet_s10: arm > toaster (toaster_main_group_1), unmapped geometry, 21 ticks
  - counter_to_cabinet_s10: base > box (cube_2_dining_area_group), other mapped geometry, 1 ticks
  - counter_to_cabinet_s10: base > single cabinet (stack_05_right_group_1_1), other mapped geometry, 77 ticks
  - counter_to_cabinet_s10: fingers > bottled water (clutter_counter_1), other mapped geometry, 8 ticks
  - counter_to_cabinet_s10: fingers > bottled water (clutter_counter_1), unmapped geometry, 1 ticks
  - counter_to_cabinet_s10: fingers > bottled water (clutter_counter_2), other mapped geometry, 21 ticks
  - counter_to_cabinet_s10: fingers > bottled water (clutter_counter_2), unmapped geometry, 139 ticks
  - counter_to_cabinet_s10: fingers > canned food (distr_counter), other mapped geometry, 9 ticks
  - counter_to_cabinet_s10: fingers > canned food (distr_counter), unmapped geometry, 4 ticks
  - counter_to_cabinet_s10: fingers > unknown (obstacle_2), designated obstacle, 11 ticks
  - counter_to_cabinet_s10: hand > bottled water (clutter_counter_1), other mapped geometry, 23 ticks
  - counter_to_cabinet_s10: hand > bottled water (clutter_counter_2), other mapped geometry, 15 ticks
  - counter_to_cabinet_s10: hand > bottled water (clutter_counter_2), unmapped geometry, 137 ticks
  - counter_to_cabinet_s10: hand > canned food (distr_counter), other mapped geometry, 4 ticks
  - counter_to_cabinet_s10: hand > canned food (distr_counter), unmapped geometry, 6 ticks
  - counter_to_cabinet_s10: hand > toaster (toaster_main_group_1), other mapped geometry, 64 ticks
  - counter_to_cabinet_s10: hand > toaster (toaster_main_group_1), unmapped geometry, 14 ticks
  - counter_to_cabinet_s10: hand > unknown (obstacle_2), designated obstacle, 1 ticks
  - counter_to_cabinet_s13: arm > single cabinet (singlecabinet_2_main_group_1), other mapped geometry, 1 ticks
  - counter_to_cabinet_s13: base > boxed food (clutter_floor_2), other mapped geometry, 3 ticks
  - counter_to_cabinet_s13: base > boxed food (clutter_floor_2), unmapped geometry, 20 ticks
  - counter_to_cabinet_s13: base > fridge bottom freezer (fridgebottomfreezer_main_group_1), other mapped geometry, 7 ticks
  - counter_to_stove_s0: arm > counter (island_2_island_group_1), other mapped geometry, 7 ticks
  - counter_to_stove_s0: arm > plate (obj_container), other mapped geometry, 12 ticks
  - counter_to_stove_s0: arm > unknown (obstacle_0), designated obstacle, 14 ticks
  - counter_to_stove_s0: base > basket (clutter_floor_1), unmapped geometry, 7 ticks
  - counter_to_stove_s7: base > single cabinet (stack_3_island_group_1_1), other mapped geometry, 16 ticks
  - counter_to_stove_s9: arm > counter (counter_1_left_group_1), other mapped geometry, 4 ticks
  - counter_to_stove_s9: arm > plate (obj_container), other mapped geometry, 12 ticks
  - counter_to_stove_s9: arm > plate (obj_container), unmapped geometry, 6 ticks
  - counter_to_stove_s9: fingers > condiment bottle (clutter_counter_2), other mapped geometry, 17 ticks
  - counter_to_stove_s9: fingers > unknown (obstacle_3), designated obstacle, 11 ticks
  - counter_to_stove_s9: hand > condiment bottle (clutter_counter_2), other mapped geometry, 15 ticks
  - counter_to_stove_s9: hand > condiment bottle (clutter_counter_2), unmapped geometry, 4 ticks
  - counter_to_stove_s10: base > basket (clutter_floor_1), other mapped geometry, 11 ticks
  - counter_to_stove_s10: base > basket (clutter_floor_1), unmapped geometry, 12 ticks
  - counter_to_stove_s10: base > single cabinet (stack_1_island_group_1_1), other mapped geometry, 260 ticks
  - counter_to_stove_s10: base > single cabinet (stack_2_island_group_1_1), other mapped geometry, 327 ticks
  - counter_to_stove_s10: base > single cabinet (stack_5_island_group_2.001_1), other mapped geometry, 29 ticks
  - counter_to_stove_s10: base > single cabinet (stack_6_island_group_2.001_1), other mapped geometry, 102 ticks
  - counter_to_stove_s10: base > single cabinet (stack_6_island_group_2.001_1), unmapped geometry, 3 ticks
  - counter_to_stove_s10: base > single cabinet (stack_7_island_group_2.001_1), other mapped geometry, 61 ticks
  - counter_to_stove_s10: hand > unknown (obstacle_2), designated obstacle, 6 ticks
  - counter_to_sink_s0: arm > sink (sink_main_group_1), other mapped geometry, 12 ticks
  - counter_to_sink_s0: arm > unknown (obstacle_0), designated obstacle, 7 ticks
  - counter_to_sink_s0: hand > unknown (obstacle_0), designated obstacle, 49 ticks
  - counter_to_sink_s1: base > drawer (stack_1_island_group_1_1), other mapped geometry, 60 ticks
  - counter_to_sink_s1: base > hinge cabinet (stack_4_main_group_1_1), other mapped geometry, 3 ticks
  - counter_to_sink_s1: fingers > banana (distr_counter), other mapped geometry, 7 ticks
  - counter_to_sink_s1: fingers > banana (distr_counter), unmapped geometry, 45 ticks
  - counter_to_sink_s1: fingers > condiment bottle (clutter_counter_2), other mapped geometry, 36 ticks
  - counter_to_sink_s1: fingers > condiment bottle (clutter_counter_2), unmapped geometry, 55 ticks
  - counter_to_sink_s1: fingers > counter (island_island_group_1), other mapped geometry, 47 ticks
  - counter_to_sink_s1: hand > condiment bottle (clutter_counter_2), other mapped geometry, 3 ticks
  - counter_to_sink_s1: hand > condiment bottle (clutter_counter_2), unmapped geometry, 3 ticks
  - counter_to_sink_s11: arm > boxed drink (clutter_counter_0), other mapped geometry, 2 ticks
  - counter_to_sink_s11: arm > condiment bottle (clutter_counter_2), other mapped geometry, 9 ticks
  - counter_to_sink_s11: arm > condiment bottle (clutter_counter_2), unmapped geometry, 8 ticks
  - counter_to_sink_s11: arm > counter (counter_1_main_group_1), other mapped geometry, 9 ticks
  - counter_to_sink_s11: fingers > unknown (obstacle_3), designated obstacle, 13 ticks
  - counter_to_sink_s11: hand > condiment bottle (clutter_counter_2), other mapped geometry, 9 ticks
  - counter_to_sink_s11: hand > condiment bottle (clutter_counter_2), unmapped geometry, 2 ticks
  - close_fridge_s9: base > basket (clutter_floor_1), other mapped geometry, 1 ticks
  - close_cabinet_s16: base > boxed food (clutter_floor_2), other mapped geometry, 4 ticks
  - close_cabinet_s16: base > boxed food (clutter_floor_2), unmapped geometry, 27 ticks
  - close_cabinet_s16: hand > single cabinet (singlecabinet_1_main_group_1), other mapped geometry, 4 ticks
  - close_cabinet_s16: hand > single cabinet (singlecabinet_1_main_group_1), unmapped geometry, 9 ticks
  - close_cabinet_s17: arm > hinge cabinet (hingecabinet_1_main_group_1), other mapped geometry, 23 ticks
  - close_cabinet_s17: base > box (cube_2_island_group_1), other mapped geometry, 5 ticks
  - close_cabinet_s17: hand > hinge cabinet (hingecabinet_1_main_group_1), other mapped geometry, 79 ticks
  - sink_to_counter_s18: arm > boxed drink (clutter_counter_1), other mapped geometry, 28 ticks
  - sink_to_counter_s18: arm > counter (counter_1_main_group_1), other mapped geometry, 64 ticks
  - sink_to_counter_s18: arm > mug (clutter_counter_0), unmapped geometry, 50 ticks
  - sink_to_counter_s18: arm > sink (sink_main_group_1), other mapped geometry, 11 ticks
  - sink_to_counter_s18: arm > unknown (obstacle_2), designated obstacle, 75 ticks
  - sink_to_counter_s18: arm > yogurt (obj), other mapped geometry, 5 ticks
  - sink_to_counter_s18: arm > yogurt (obj), unmapped geometry, 2 ticks
  - sink_to_counter_s18: fingers > mug (clutter_counter_0), other mapped geometry, 21 ticks
  - sink_to_counter_s18: fingers > mug (clutter_counter_0), unmapped geometry, 73 ticks
  - sink_to_counter_s18: hand > mug (clutter_counter_0), other mapped geometry, 15 ticks
  - sink_to_counter_s18: hand > mug (clutter_counter_0), unmapped geometry, 145 ticks
  - sink_to_counter_s18: hand > unknown (obstacle_2), designated obstacle, 16 ticks
  - sink_to_counter_s19: base > flour bag (clutter_floor_0), other mapped geometry, 1 ticks
  - sink_to_counter_s19: hand > unknown (obstacle_0), designated obstacle, 11 ticks
  - counter_to_cabinet_s17: arm > canned food (clutter_counter_0), other mapped geometry, 25 ticks
  - counter_to_cabinet_s17: arm > canned food (clutter_counter_0), unmapped geometry, 4 ticks
  - counter_to_cabinet_s17: arm > coffee machine (coffee_machine_main_group_1), other mapped geometry, 18 ticks
  - counter_to_cabinet_s17: fingers > unknown (obstacle_0), designated obstacle, 147 ticks
  - counter_to_cabinet_s17: hand > coffee machine (coffee_machine_main_group_1), other mapped geometry, 76 ticks
  - counter_to_cabinet_s17: hand > coffee machine (coffee_machine_main_group_1), unmapped geometry, 26 ticks
  - counter_to_cabinet_s17: hand > unknown (obstacle_0), designated obstacle, 33 ticks
  - counter_to_cabinet_s18: base > basket (clutter_floor_1), unmapped geometry, 43 ticks
  - counter_to_cabinet_s18: base > box (stack_1_island_group_base), other mapped geometry, 24 ticks
  - counter_to_cabinet_s18: base > boxed food (clutter_floor_2), other mapped geometry, 7 ticks
  - counter_to_cabinet_s18: base > boxed food (clutter_floor_2), unmapped geometry, 37 ticks
  - counter_to_cabinet_s18: base > hinge cabinet (stack_3_main_group_1), other mapped geometry, 140 ticks
  - counter_to_cabinet_s18: base > single cabinet (stack_1_island_group_1), other mapped geometry, 24 ticks
  - counter_to_stove_s14: arm > hinge cabinet (hingecabinet_4_right_group_1), other mapped geometry, 13 ticks
  - counter_to_stove_s14: base > box (cube_2_island_group_1), other mapped geometry, 4 ticks
  - counter_to_stove_s14: base > boxed food (clutter_floor_2), other mapped geometry, 8 ticks
  - counter_to_stove_s14: base > boxed food (clutter_floor_2), unmapped geometry, 3 ticks
  - counter_to_stove_s14: base > drawer (stack_3_right_group_1_1), other mapped geometry, 54 ticks
  - counter_to_stove_s14: base > drawer (stack_3_right_group_1_2), other mapped geometry, 54 ticks
  - counter_to_stove_s14: base > single cabinet (stack_3_island_group_1_1), other mapped geometry, 1 ticks
  - counter_to_stove_s14: hand > unknown (obstacle_2), designated obstacle, 1 ticks
  - counter_to_sink_s19: arm > unknown (obstacle_2), designated obstacle, 10 ticks
  - counter_to_sink_s19: base > panel cabinet (panelbox_2_island_group_1_1), other mapped geometry, 7 ticks
- **End-effector filter**: designated obstacle 213 ticks, other mapped geometry 2123 ticks, unmapped geometry 1098 ticks (25 scenes)
  - close_fridge_s3: arm > fridge french door (fridgefrenchdoor_main_group_1), unmapped geometry, 7 ticks
  - close_fridge_s4: arm > fridge french door (fridgefrenchdoor_main_group_1), unmapped geometry, 7 ticks
  - close_fridge_s5: base > basket (clutter_floor_1), unmapped geometry, 249 ticks
  - close_cabinet_s8: base > basket (clutter_floor_1), other mapped geometry, 4 ticks
  - close_cabinet_s8: base > basket (clutter_floor_1), unmapped geometry, 2 ticks
  - close_cabinet_s9: arm > single cabinet (singlecabinet_3_right_group_1), unmapped geometry, 4 ticks
  - close_cabinet_s9: base > boxed food (clutter_floor_2), other mapped geometry, 4 ticks
  - close_cabinet_s9: base > boxed food (clutter_floor_2), unmapped geometry, 44 ticks
  - close_cabinet_s9: base > single cabinet (stack_1_right_group_1_1), other mapped geometry, 20 ticks
  - close_cabinet_s9: base > stove (widestove_right_group_1), other mapped geometry, 15 ticks
  - close_cabinet_s9: base > stove (widestove_right_group_1), unmapped geometry, 2 ticks
  - close_cabinet_s9: fingers > wall (wall_main_2_room), other mapped geometry, 39 ticks
  - sink_to_counter_s0: arm > counter (counter_1_main_group_1), other mapped geometry, 3 ticks
  - sink_to_counter_s0: arm > mug (clutter_counter_0), other mapped geometry, 3 ticks
  - sink_to_counter_s7: arm > mug (clutter_counter_1), other mapped geometry, 34 ticks
  - sink_to_counter_s7: arm > sink (sink_main_group_1), other mapped geometry, 10 ticks
  - sink_to_counter_s7: fingers > mug (clutter_counter_1), unmapped geometry, 19 ticks
  - sink_to_counter_s11: arm > boxed drink (clutter_counter_1), other mapped geometry, 2 ticks
  - sink_to_counter_s11: arm > sink (sink_1_right_group_1), other mapped geometry, 14 ticks
  - sink_to_counter_s11: fingers > unknown (obstacle_3), designated obstacle, 92 ticks
  - sink_to_counter_s11: hand > unknown (obstacle_3), designated obstacle, 28 ticks
  - sink_to_counter_s12: base > single cabinet (stack_5_island_group_2_1), other mapped geometry, 8 ticks
  - counter_to_cabinet_s3: arm > single cabinet (singlecabinet_2_main_group_1), other mapped geometry, 1 ticks
  - counter_to_cabinet_s3: base > box (cube_1_island_group_1), other mapped geometry, 10 ticks
  - counter_to_cabinet_s3: base > stove (widestove_main_group_1), other mapped geometry, 6 ticks
  - counter_to_cabinet_s10: arm > counter (counter_1_right_group_1), other mapped geometry, 1 ticks
  - counter_to_cabinet_s10: base > box (cube_2_dining_area_group), other mapped geometry, 1 ticks
  - counter_to_cabinet_s10: base > hinge cabinet (stack_07_right_group_1_1), other mapped geometry, 69 ticks
  - counter_to_cabinet_s10: base > single cabinet (stack_05_right_group_1_1), other mapped geometry, 232 ticks
  - counter_to_cabinet_s10: fingers > counter (counter_1_right_group_1), other mapped geometry, 21 ticks
  - counter_to_cabinet_s10: hand > counter (counter_1_right_group_1), other mapped geometry, 168 ticks
  - counter_to_cabinet_s13: arm > single cabinet (singlecabinet_2_main_group_1), other mapped geometry, 5 ticks
  - counter_to_cabinet_s13: base > basket (clutter_floor_1), unmapped geometry, 42 ticks
  - counter_to_cabinet_s13: base > boxed food (clutter_floor_2), other mapped geometry, 3 ticks
  - counter_to_cabinet_s13: base > boxed food (clutter_floor_2), unmapped geometry, 22 ticks
  - counter_to_cabinet_s13: base > fridge bottom freezer (fridgebottomfreezer_main_group_1), other mapped geometry, 30 ticks
  - counter_to_stove_s0: arm > unknown (obstacle_0), designated obstacle, 16 ticks
  - counter_to_stove_s0: base > basket (clutter_floor_1), unmapped geometry, 7 ticks
  - counter_to_stove_s7: arm > bottled water (clutter_counter_0), other mapped geometry, 4 ticks
  - counter_to_stove_s7: arm > bottled water (clutter_counter_0), unmapped geometry, 16 ticks
  - counter_to_stove_s7: arm > boxed drink (clutter_counter_1), other mapped geometry, 2 ticks
  - counter_to_stove_s7: arm > stovetop (stovetop_right_group_1), unmapped geometry, 5 ticks
  - counter_to_stove_s7: arm > unknown (obstacle_1), designated obstacle, 11 ticks
  - counter_to_stove_s7: base > single cabinet (stack_3_island_group_1_1), other mapped geometry, 15 ticks
  - counter_to_stove_s9: arm > condiment bottle (clutter_counter_2), other mapped geometry, 21 ticks
  - counter_to_stove_s9: arm > condiment bottle (clutter_counter_2), unmapped geometry, 4 ticks
  - counter_to_stove_s9: arm > counter (counter_1_left_group_1), other mapped geometry, 17 ticks
  - counter_to_stove_s9: arm > hinge cabinet (hingecabinet_3_left_group_1), other mapped geometry, 24 ticks
  - counter_to_stove_s9: arm > plate (obj_container), unmapped geometry, 1 ticks
  - counter_to_stove_s9: arm > stovetop (stovetop_left_group_1), other mapped geometry, 14 ticks
  - counter_to_stove_s9: base > drawer (stack_3_left_group_1_1), other mapped geometry, 117 ticks
  - counter_to_stove_s10: base > fish (obj), unmapped geometry, 1 ticks
  - counter_to_stove_s10: base > single cabinet (stack_6_island_group_2.001_1), other mapped geometry, 161 ticks
  - counter_to_stove_s10: base > single cabinet (stack_7_island_group_2.001_1), other mapped geometry, 17 ticks
  - counter_to_sink_s0: arm > unknown (obstacle_0), designated obstacle, 43 ticks
  - counter_to_sink_s0: fingers > unknown (obstacle_0), designated obstacle, 10 ticks
  - counter_to_sink_s0: hand > unknown (obstacle_0), designated obstacle, 8 ticks
  - counter_to_sink_s1: arm > counter (counter_1_right_group_1), other mapped geometry, 3 ticks
  - counter_to_sink_s1: arm > sink (sink_1_right_group_1), other mapped geometry, 8 ticks
  - counter_to_sink_s1: base > drawer (stack_1_island_group_1_1), other mapped geometry, 1 ticks
  - close_fridge_s9: base > basket (clutter_floor_1), other mapped geometry, 1 ticks
  - close_cabinet_s16: base > boxed food (clutter_floor_2), other mapped geometry, 5 ticks
  - close_cabinet_s16: base > boxed food (clutter_floor_2), unmapped geometry, 25 ticks
  - close_cabinet_s16: hand > single cabinet (singlecabinet_1_main_group_1), other mapped geometry, 2 ticks
  - close_cabinet_s16: hand > single cabinet (singlecabinet_1_main_group_1), unmapped geometry, 7 ticks
  - close_cabinet_s17: arm > counter (counter_3_right_group_1), other mapped geometry, 159 ticks
  - close_cabinet_s17: arm > drawer (stack_04_right_group_1_2), other mapped geometry, 108 ticks
  - close_cabinet_s17: base > box (cube_2_island_group_1), other mapped geometry, 5 ticks
  - close_cabinet_s17: base > hinge cabinet (stack_03_right_group_1_1), other mapped geometry, 37 ticks
  - close_cabinet_s17: base > single cabinet (stack_04_right_group_1_1), other mapped geometry, 46 ticks
  - sink_to_counter_s19: base > flour bag (clutter_floor_0), other mapped geometry, 1 ticks
  - counter_to_cabinet_s18: base > basket (clutter_floor_1), unmapped geometry, 626 ticks
  - counter_to_cabinet_s18: base > boxed food (clutter_floor_2), other mapped geometry, 4 ticks
  - counter_to_cabinet_s18: base > boxed food (clutter_floor_2), unmapped geometry, 8 ticks
  - counter_to_cabinet_s18: base > hinge cabinet (stack_1_front_group_1), other mapped geometry, 635 ticks
  - counter_to_stove_s14: arm > hinge cabinet (hingecabinet_4_right_group_1), other mapped geometry, 1 ticks
  - counter_to_stove_s14: arm > unknown (obstacle_2), designated obstacle, 5 ticks
  - counter_to_stove_s14: base > box (cube_2_island_group_1), other mapped geometry, 4 ticks
  - counter_to_stove_s14: base > single cabinet (stack_3_island_group_1_1), other mapped geometry, 1 ticks
  - counter_to_sink_s19: base > panel cabinet (panelbox_2_island_group_1_1), other mapped geometry, 7 ticks
- **Fixed margin**: designated obstacle 29 ticks, other mapped geometry 145 ticks, unmapped geometry 764 ticks (12 scenes)
  - close_fridge_s3: arm > fridge french door (fridgefrenchdoor_main_group_1), unmapped geometry, 182 ticks
  - close_fridge_s5: base > basket (clutter_floor_1), unmapped geometry, 239 ticks
  - close_fridge_s6: arm > fridge french door (fridgefrenchdoor_right_group_1), unmapped geometry, 7 ticks
  - close_cabinet_s8: arm > cereal (distr_counter_3), other mapped geometry, 2 ticks
  - close_cabinet_s8: arm > single cabinet (singlecabinet_03_main_group_1), unmapped geometry, 2 ticks
  - close_cabinet_s8: base > basket (clutter_floor_1), unmapped geometry, 19 ticks
  - close_cabinet_s8: base > wall (wall_main_2_room), other mapped geometry, 1 ticks
  - counter_to_cabinet_s10: fingers > counter (counter_1_right_group_1), other mapped geometry, 32 ticks
  - counter_to_cabinet_s10: fingers > counter (counter_corner_1_main_group_1), other mapped geometry, 11 ticks
  - counter_to_cabinet_s10: hand > counter (counter_1_right_group_1), other mapped geometry, 99 ticks
  - counter_to_cabinet_s13: base > basket (clutter_floor_1), unmapped geometry, 96 ticks
  - counter_to_stove_s0: base > basket (clutter_floor_1), unmapped geometry, 8 ticks
  - counter_to_stove_s7: base > bottled water (clutter_counter_0), unmapped geometry, 1 ticks
  - counter_to_sink_s0: arm > unknown (obstacle_0), designated obstacle, 29 ticks
  - close_cabinet_s16: arm > single cabinet (singlecabinet_1_main_group_1), unmapped geometry, 26 ticks
  - close_cabinet_s17: arm > hinge cabinet (hingecabinet_2_right_group_1), unmapped geometry, 21 ticks
  - counter_to_cabinet_s18: base > basket (clutter_floor_1), unmapped geometry, 163 ticks
- **Proposed without ISSf**: designated obstacle 78 ticks, other mapped geometry 322 ticks, unmapped geometry 537 ticks (11 scenes)
  - close_fridge_s5: base > basket (clutter_floor_1), unmapped geometry, 250 ticks
  - close_fridge_s6: arm > fridge french door (fridgefrenchdoor_right_group_1), unmapped geometry, 8 ticks
  - sink_to_counter_s7: fingers > mug (clutter_counter_1), unmapped geometry, 20 ticks
  - sink_to_counter_s7: hand > mug (clutter_counter_1), unmapped geometry, 23 ticks
  - counter_to_cabinet_s10: hand > counter (counter_1_right_group_1), other mapped geometry, 31 ticks
  - counter_to_cabinet_s13: base > basket (clutter_floor_1), unmapped geometry, 155 ticks
  - counter_to_cabinet_s13: base > fridge bottom freezer (fridgebottomfreezer_main_group_1), other mapped geometry, 8 ticks
  - counter_to_cabinet_s13: base > fridge bottom freezer (fridgebottomfreezer_main_group_1), unmapped geometry, 5 ticks
  - counter_to_stove_s0: arm > unknown (obstacle_0), designated obstacle, 13 ticks
  - counter_to_stove_s0: base > basket (clutter_floor_1), unmapped geometry, 6 ticks
  - counter_to_stove_s0: base > drawer (stack_05_left_group_1_1), other mapped geometry, 13 ticks
  - counter_to_stove_s0: base > drawer (stack_05_left_group_1_2), other mapped geometry, 11 ticks
  - counter_to_sink_s0: arm > unknown (obstacle_0), designated obstacle, 14 ticks
  - counter_to_sink_s0: hand > unknown (obstacle_0), designated obstacle, 22 ticks
  - close_cabinet_s16: arm > single cabinet (singlecabinet_1_main_group_1), unmapped geometry, 17 ticks
  - close_cabinet_s17: arm > hinge cabinet (hingecabinet_2_right_group_1), unmapped geometry, 9 ticks
  - counter_to_cabinet_s18: base > basket (clutter_floor_1), unmapped geometry, 44 ticks
  - counter_to_cabinet_s18: base > hinge cabinet (stack_3_main_group_1), other mapped geometry, 211 ticks
  - counter_to_cabinet_s18: fingers > counter (counter_1_main_group), other mapped geometry, 19 ticks
  - counter_to_cabinet_s18: fingers > open cabinet (cab_3_main_group), other mapped geometry, 9 ticks
  - counter_to_cabinet_s18: hand > open cabinet (cab_3_main_group), other mapped geometry, 20 ticks
  - counter_to_sink_s19: hand > unknown (obstacle_2), designated obstacle, 29 ticks
- **Proposed**: designated obstacle 108 ticks, other mapped geometry 263 ticks, unmapped geometry 1121 ticks (13 scenes)
  - close_fridge_s3: arm > fridge french door (fridgefrenchdoor_main_group_1), unmapped geometry, 3 ticks
  - close_fridge_s5: base > basket (clutter_floor_1), unmapped geometry, 250 ticks
  - close_fridge_s6: arm > fridge french door (fridgefrenchdoor_right_group_1), unmapped geometry, 8 ticks
  - close_cabinet_s9: fingers > wall (wall_left_1_room), other mapped geometry, 3 ticks
  - close_cabinet_s9: fingers > wall (wall_main_2_room), other mapped geometry, 2 ticks
  - close_cabinet_s9: fingers > wall (wall_main_2_room), unmapped geometry, 48 ticks
  - close_cabinet_s9: hand > wall (wall_left_1_room), other mapped geometry, 56 ticks
  - close_cabinet_s9: hand > wall (wall_left_1_room), unmapped geometry, 2 ticks
  - close_cabinet_s9: hand > wall (wall_main_2_room), other mapped geometry, 8 ticks
  - close_cabinet_s9: hand > wall (wall_main_2_room), unmapped geometry, 18 ticks
  - sink_to_counter_s11: fingers > unknown (obstacle_3), designated obstacle, 86 ticks
  - sink_to_counter_s11: hand > unknown (obstacle_3), designated obstacle, 11 ticks
  - counter_to_cabinet_s10: hand > counter (counter_1_right_group_1), other mapped geometry, 159 ticks
  - counter_to_cabinet_s13: base > basket (clutter_floor_1), unmapped geometry, 275 ticks
  - counter_to_cabinet_s13: base > fridge bottom freezer (fridgebottomfreezer_main_group_1), other mapped geometry, 1 ticks
  - counter_to_stove_s0: base > basket (clutter_floor_1), unmapped geometry, 6 ticks
  - counter_to_sink_s0: arm > unknown (obstacle_0), designated obstacle, 11 ticks
  - close_cabinet_s16: arm > single cabinet (singlecabinet_1_main_group_1), unmapped geometry, 3 ticks
  - close_cabinet_s16: base > basket (clutter_floor_1), unmapped geometry, 35 ticks
  - close_cabinet_s16: base > flour bag (clutter_floor_0), other mapped geometry, 1 ticks
  - close_cabinet_s16: base > flour bag (clutter_floor_0), unmapped geometry, 22 ticks
  - close_cabinet_s16: base > single cabinet (stack_2_island_group_1_1), other mapped geometry, 33 ticks
  - close_cabinet_s17: arm > hinge cabinet (hingecabinet_2_right_group_1), unmapped geometry, 42 ticks
  - counter_to_cabinet_s18: base > basket (clutter_floor_1), unmapped geometry, 368 ticks
  - counter_to_stove_s14: arm > marshmallow (obj), unmapped geometry, 41 ticks
- **Fixed margin, look-around map**: designated obstacle 816 ticks, other mapped geometry 156 ticks, unmapped geometry 1850 ticks (24 scenes)
  - close_fridge_s3: arm > fridge french door (fridgefrenchdoor_main_group_1), unmapped geometry, 7 ticks
  - close_fridge_s5: base > basket (clutter_floor_1), other mapped geometry, 2 ticks
  - close_fridge_s5: base > basket (clutter_floor_1), unmapped geometry, 31 ticks
  - close_cabinet_s8: arm > single cabinet (singlecabinet_03_main_group_1), unmapped geometry, 3 ticks
  - close_cabinet_s9: arm > single cabinet (singlecabinet_3_right_group_1), unmapped geometry, 5 ticks
  - close_cabinet_s9: base > box (stack_4_island_group_1_base), unmapped geometry, 31 ticks
  - close_cabinet_s9: base > hinge cabinet (stack_4_island_group_1_1), unmapped geometry, 31 ticks
  - sink_to_counter_s0: arm > sink (sink_main_group_1), unmapped geometry, 17 ticks
  - sink_to_counter_s7: arm > boxed drink (clutter_counter_2), unmapped geometry, 23 ticks
  - sink_to_counter_s7: arm > mug (clutter_counter_1), unmapped geometry, 21 ticks
  - sink_to_counter_s7: arm > sink (sink_main_group_1), unmapped geometry, 48 ticks
  - sink_to_counter_s7: base > boxed drink (clutter_counter_2), unmapped geometry, 19 ticks
  - sink_to_counter_s7: base > hinge cabinet (stack_02_main_group_1_1), unmapped geometry, 13 ticks
  - sink_to_counter_s7: fingers > mug (clutter_counter_1), unmapped geometry, 81 ticks
  - sink_to_counter_s7: fingers > unknown (obstacle_0), designated obstacle, 97 ticks
  - sink_to_counter_s7: hand > mug (clutter_counter_1), unmapped geometry, 42 ticks
  - sink_to_counter_s7: hand > unknown (obstacle_0), designated obstacle, 94 ticks
  - sink_to_counter_s11: arm > sink (sink_1_right_group_1), unmapped geometry, 3 ticks
  - sink_to_counter_s11: fingers > mug (clutter_counter_0), unmapped geometry, 54 ticks
  - sink_to_counter_s11: fingers > unknown (obstacle_3), designated obstacle, 243 ticks
  - sink_to_counter_s11: hand > mug (clutter_counter_0), unmapped geometry, 34 ticks
  - sink_to_counter_s11: hand > unknown (obstacle_3), designated obstacle, 28 ticks
  - sink_to_counter_s12: base > single cabinet (stack_5_island_group_2_1), unmapped geometry, 8 ticks
  - counter_to_cabinet_s3: base > box (cube_1_island_group_1), unmapped geometry, 10 ticks
  - counter_to_cabinet_s3: base > stove (widestove_main_group_1), unmapped geometry, 6 ticks
  - counter_to_cabinet_s10: hand > counter (counter_1_right_group_1), other mapped geometry, 39 ticks
  - counter_to_cabinet_s10: hand > counter (counter_corner_1_main_group_1), other mapped geometry, 3 ticks
  - counter_to_cabinet_s13: base > basket (clutter_floor_1), unmapped geometry, 20 ticks
  - counter_to_cabinet_s13: base > box (cube_8_island_group_1), other mapped geometry, 18 ticks
  - counter_to_cabinet_s13: base > dishwasher (dishwasher_1_main_group_1), other mapped geometry, 8 ticks
  - counter_to_cabinet_s13: base > flour bag (clutter_floor_0), other mapped geometry, 6 ticks
  - counter_to_cabinet_s13: base > flour bag (clutter_floor_0), unmapped geometry, 52 ticks
  - counter_to_cabinet_s13: base > single cabinet (stack_01_main_group_1_1), other mapped geometry, 2 ticks
  - counter_to_cabinet_s13: base > wall (wall_right_1_room), other mapped geometry, 12 ticks
  - counter_to_cabinet_s13: fingers > counter (island_island_group_1), other mapped geometry, 21 ticks
  - counter_to_cabinet_s13: hand > wall (wall_right_1_room), other mapped geometry, 18 ticks
  - counter_to_stove_s0: base > basket (clutter_floor_1), other mapped geometry, 4 ticks
  - counter_to_stove_s0: base > basket (clutter_floor_1), unmapped geometry, 2 ticks
  - counter_to_stove_s7: arm > bottled water (clutter_counter_0), unmapped geometry, 1 ticks
  - counter_to_stove_s7: base > single cabinet (stack_3_island_group_1_1), unmapped geometry, 13 ticks
  - counter_to_stove_s7: fingers > boxed drink (clutter_counter_1), unmapped geometry, 93 ticks
  - counter_to_stove_s7: hand > boxed drink (clutter_counter_1), unmapped geometry, 2 ticks
  - counter_to_stove_s10: base > fish (obj), unmapped geometry, 5 ticks
  - counter_to_stove_s10: base > single cabinet (stack_5_island_group_2.001_1), unmapped geometry, 9 ticks
  - counter_to_stove_s10: fingers > counter (counter_1_main_group_1), unmapped geometry, 12 ticks
  - counter_to_sink_s0: hand > unknown (obstacle_0), designated obstacle, 52 ticks
  - counter_to_sink_s1: arm > sink (sink_1_right_group_1), unmapped geometry, 12 ticks
  - counter_to_sink_s1: base > drawer (stack_1_island_group_1_1), unmapped geometry, 1 ticks
  - counter_to_sink_s1: base > drawer (stack_2_right_group_1_1), unmapped geometry, 43 ticks
  - counter_to_sink_s1: base > drawer (stack_2_right_group_1_2), unmapped geometry, 43 ticks
  - counter_to_sink_s1: fingers > banana (distr_counter), unmapped geometry, 54 ticks
  - counter_to_sink_s11: base > dishwasher (dishwasher_1_main_group_1), unmapped geometry, 75 ticks
  - counter_to_sink_s11: base > hinge cabinet (stack_02_main_group_1_1), unmapped geometry, 353 ticks
  - counter_to_sink_s11: fingers > condiment bottle (clutter_counter_2), unmapped geometry, 74 ticks
  - counter_to_sink_s11: fingers > unknown (obstacle_3), designated obstacle, 254 ticks
  - counter_to_sink_s11: hand > condiment bottle (clutter_counter_2), unmapped geometry, 74 ticks
  - counter_to_sink_s11: hand > unknown (obstacle_3), designated obstacle, 48 ticks
  - close_cabinet_s16: arm > single cabinet (singlecabinet_1_main_group_1), unmapped geometry, 24 ticks
  - close_cabinet_s16: hand > single cabinet (singlecabinet_1_main_group_1), unmapped geometry, 7 ticks
  - close_cabinet_s17: arm > hinge cabinet (hingecabinet_1_main_group_1), unmapped geometry, 25 ticks
  - close_cabinet_s17: base > box (cube_2_island_group_1), unmapped geometry, 4 ticks
  - close_cabinet_s17: fingers > hinge cabinet (hingecabinet_1_main_group_1), unmapped geometry, 8 ticks
  - close_cabinet_s17: hand > hinge cabinet (hingecabinet_1_main_group_1), unmapped geometry, 79 ticks
  - sink_to_counter_s18: arm > counter (counter_1_main_group_1), other mapped geometry, 1 ticks
  - sink_to_counter_s18: arm > sink (sink_main_group_1), unmapped geometry, 16 ticks
  - sink_to_counter_s18: fingers > mug (clutter_counter_0), other mapped geometry, 18 ticks
  - sink_to_counter_s18: fingers > mug (clutter_counter_0), unmapped geometry, 77 ticks
  - sink_to_counter_s18: hand > mug (clutter_counter_0), unmapped geometry, 95 ticks
  - counter_to_cabinet_s18: base > basket (clutter_floor_1), other mapped geometry, 4 ticks
  - counter_to_cabinet_s18: base > basket (clutter_floor_1), unmapped geometry, 58 ticks
  - counter_to_stove_s14: base > box (cube_2_island_group_1), unmapped geometry, 1 ticks
  - counter_to_stove_s20: base > stove (stove_main_group_1), unmapped geometry, 24 ticks
  - counter_to_sink_s19: base > panel cabinet (panelbox_2_island_group_1_1), unmapped geometry, 7 ticks
- **Proposed, look-around map**: designated obstacle 452 ticks, other mapped geometry 74 ticks, unmapped geometry 2184 ticks (23 scenes)
  - close_fridge_s3: arm > fridge french door (fridgefrenchdoor_main_group_1), unmapped geometry, 15 ticks
  - close_fridge_s5: base > basket (clutter_floor_1), unmapped geometry, 19 ticks
  - close_fridge_s6: arm > fridge french door (fridgefrenchdoor_right_group_1), unmapped geometry, 160 ticks
  - close_cabinet_s3: base > single cabinet (stack_03_main_group_1_1), unmapped geometry, 6 ticks
  - close_cabinet_s3: base > single cabinet (stack_04_right_group_1_1), unmapped geometry, 7 ticks
  - close_cabinet_s3: hand > coffee machine (coffee_machine_right_group_1), unmapped geometry, 14 ticks
  - close_cabinet_s8: fingers > counter (counter_3_main_group_1), unmapped geometry, 6 ticks
  - close_cabinet_s9: arm > single cabinet (singlecabinet_3_right_group_1), unmapped geometry, 17 ticks
  - close_cabinet_s9: base > hinge cabinet (stack_2_main_group_1_1), unmapped geometry, 170 ticks
  - sink_to_counter_s7: arm > boxed drink (clutter_counter_2), unmapped geometry, 10 ticks
  - sink_to_counter_s7: fingers > boxed drink (clutter_counter_2), unmapped geometry, 99 ticks
  - sink_to_counter_s7: fingers > mug (clutter_counter_1), unmapped geometry, 117 ticks
  - sink_to_counter_s7: fingers > unknown (obstacle_0), designated obstacle, 276 ticks
  - sink_to_counter_s7: hand > boxed drink (clutter_counter_2), unmapped geometry, 2 ticks
  - sink_to_counter_s7: hand > mug (clutter_counter_1), unmapped geometry, 39 ticks
  - sink_to_counter_s7: hand > unknown (obstacle_0), designated obstacle, 174 ticks
  - sink_to_counter_s12: base > single cabinet (stack_5_island_group_2_1), unmapped geometry, 8 ticks
  - counter_to_cabinet_s3: base > box (cube_1_island_group_1), unmapped geometry, 10 ticks
  - counter_to_cabinet_s3: base > stove (widestove_main_group_1), unmapped geometry, 6 ticks
  - counter_to_cabinet_s10: hand > counter (counter_1_right_group_1), other mapped geometry, 9 ticks
  - counter_to_cabinet_s13: base > flour bag (clutter_floor_0), other mapped geometry, 5 ticks
  - counter_to_cabinet_s13: base > flour bag (clutter_floor_0), unmapped geometry, 43 ticks
  - counter_to_cabinet_s13: fingers > counter (counter_1_main_group_1), other mapped geometry, 20 ticks
  - counter_to_stove_s0: base > basket (clutter_floor_1), other mapped geometry, 11 ticks
  - counter_to_stove_s7: arm > bottled water (clutter_counter_0), unmapped geometry, 2 ticks
  - counter_to_stove_s7: arm > boxed drink (clutter_counter_1), unmapped geometry, 2 ticks
  - counter_to_stove_s7: base > single cabinet (stack_3_island_group_1_1), unmapped geometry, 16 ticks
  - counter_to_stove_s10: base > single cabinet (stack_5_island_group_2.001_1), unmapped geometry, 60 ticks
  - counter_to_sink_s0: fingers > unknown (obstacle_0), designated obstacle, 2 ticks
  - counter_to_sink_s1: base > drawer (stack_1_island_group_1_1), unmapped geometry, 1 ticks
  - counter_to_sink_s1: base > drawer (stack_2_right_group_1_1), unmapped geometry, 50 ticks
  - counter_to_sink_s1: base > drawer (stack_2_right_group_1_2), unmapped geometry, 50 ticks
  - counter_to_sink_s1: fingers > banana (distr_counter), unmapped geometry, 56 ticks
  - counter_to_sink_s11: arm > boxed drink (clutter_counter_0), unmapped geometry, 7 ticks
  - counter_to_sink_s11: base > box (cube_01_island_group_1), unmapped geometry, 47 ticks
  - counter_to_sink_s11: base > box (cube_07_island_group_1), unmapped geometry, 9 ticks
  - counter_to_sink_s11: base > drawer (stack_05_right_group_1_1), unmapped geometry, 48 ticks
  - counter_to_sink_s11: base > single cabinet (stack_04_right_group_1_1), unmapped geometry, 514 ticks
  - counter_to_sink_s11: base > stove (stove_right_group_1), unmapped geometry, 15 ticks
  - counter_to_sink_s11: fingers > condiment bottle (clutter_counter_2), unmapped geometry, 71 ticks
  - counter_to_sink_s11: fingers > toaster (toaster_right_group_1), unmapped geometry, 85 ticks
  - counter_to_sink_s11: hand > counter (counter_3_right_group_1), unmapped geometry, 37 ticks
  - close_fridge_s13: arm > fridge bottom freezer (fridgebottomfreezer_right_group_1), unmapped geometry, 20 ticks
  - close_cabinet_s16: hand > single cabinet (singlecabinet_1_main_group_1), unmapped geometry, 7 ticks
  - close_cabinet_s17: base > box (cube_2_island_group_1), unmapped geometry, 3 ticks
  - close_cabinet_s17: hand > hinge cabinet (hingecabinet_1_main_group_1), unmapped geometry, 131 ticks
  - sink_to_counter_s18: fingers > mug (clutter_counter_0), other mapped geometry, 26 ticks
  - sink_to_counter_s18: fingers > mug (clutter_counter_0), unmapped geometry, 83 ticks
  - sink_to_counter_s18: hand > mug (clutter_counter_0), other mapped geometry, 2 ticks
  - sink_to_counter_s18: hand > mug (clutter_counter_0), unmapped geometry, 104 ticks
  - counter_to_cabinet_s18: base > basket (clutter_floor_1), other mapped geometry, 1 ticks
  - counter_to_cabinet_s18: base > basket (clutter_floor_1), unmapped geometry, 11 ticks
  - counter_to_sink_s19: base > panel cabinet (panelbox_2_island_group_1_1), unmapped geometry, 7 ticks

## Unfinished Proposed runs: phase and active pair (36 scenes)

- close_cabinet_s3 (look-around map): moving it (joint moved 1.51); filter active on 90% of the ticks since, mostly arm > counter (island_island_group_1) (381 ticks): margin 35.6 cm, true clearance 27.1 cm
- close_cabinet_s8 (full-pass map): moving it (joint moved 1.33); filter active on 62% of the ticks since, mostly base > boxed food (clutter_floor_2) (276 ticks): margin 0.8 cm, true clearance 3.8 cm
- close_cabinet_s8 (look-around map): moving it (joint moved 1.29); filter active on 47% of the ticks since, mostly arm > single cabinet (singlecabinet_03_main_group_1) (177 ticks): margin 0.7 cm, true clearance 13.7 cm
- close_cabinet_s9 (full-pass map): moving it (joint moved 1.50); filter active on 40% of the ticks since, mostly base > boxed food (clutter_floor_2) (156 ticks): margin 0.7 cm, true clearance 3.8 cm
- close_cabinet_s9 (look-around map): moving it (joint moved 1.49); filter active on 43% of the ticks since, mostly base > boxed food (clutter_floor_2) (345 ticks): margin 44.1 cm, true clearance 16.3 cm
- sink_to_counter_s7 (full-pass map): reaching the pick (object moved 0 cm); filter active on 74% of the ticks since, mostly fingers > boxed drink (clutter_counter_2) (451 ticks): margin 0.4 cm, true clearance 2.5 cm
- sink_to_counter_s7 (look-around map): reaching the pick (object moved 1 cm); the filter did not act after that
- counter_to_cabinet_s10 (full-pass map): reaching the pick (object moved 0 cm); filter active on 72% of the ticks since, mostly fingers > bottled water (clutter_counter_2) (618 ticks): margin 0.3 cm, true clearance 2.9 cm
- counter_to_cabinet_s10 (look-around map): reaching the pick (object moved 0 cm); filter active on 68% of the ticks since, mostly fingers > bottled water (clutter_counter_2) (534 ticks): margin 0.3 cm, true clearance 5.6 cm
- counter_to_cabinet_s13 (look-around map): reaching the pick (object moved 0 cm); filter active on 76% of the ticks since, mostly base > basket (clutter_floor_1) (958 ticks): margin 0.4 cm, true clearance 4.3 cm
- counter_to_stove_s0 (look-around map): driving; filter active on 56% of the ticks since, mostly base > basket (clutter_floor_1) (70 ticks): margin 0.4 cm, true clearance 1.8 cm
- counter_to_stove_s7 (full-pass map): carrying; filter active on 63% of the ticks since, mostly arm > counter (counter_1_right_group_1) (132 ticks): margin 0.7 cm, true clearance 3.4 cm
- counter_to_stove_s9 (look-around map): carrying; filter active on 53% of the ticks since, mostly base > drawer (stack_3_left_group_1_1) (233 ticks): margin 6.9 cm, true clearance 12.2 cm
- counter_to_stove_s10 (full-pass map): reaching the pick (object moved 11 cm); filter active on 28% of the ticks since, mostly arm > counter (island_island_group_1) (174 ticks): margin 0.6 cm, true clearance 3.6 cm
- counter_to_stove_s10 (look-around map): carrying; filter active on 6% of the ticks since, mostly fingers > unknown (obstacle_2) (13 ticks): margin 2.4 cm, true clearance 8.0 cm
- counter_to_sink_s11 (full-pass map): reaching the pick (object moved 3 cm); filter active on 44% of the ticks since, mostly fingers > condiment bottle (clutter_counter_2) (257 ticks): margin 0.4 cm, true clearance 2.8 cm
- counter_to_sink_s11 (look-around map): reaching the pick (object moved 34 cm); filter active on 31% of the ticks since, mostly arm > counter (island_island_group_1) (265 ticks): margin 13.4 cm, true clearance 5.4 cm
- close_cabinet_s16 (full-pass map): moving it (joint moved 0.89); filter active on 74% of the ticks since, mostly base > basket (clutter_floor_1) (535 ticks): margin 157.4 cm, true clearance 22.1 cm
- close_cabinet_s17 (full-pass map): moving it (joint moved 1.16); filter active on 65% of the ticks since, mostly base > stove (widestove_main_group_1) (326 ticks): margin 10.4 cm, true clearance 13.3 cm
- close_cabinet_s17 (look-around map): moving it (joint moved 1.56); filter active on 3% of the ticks since, mostly base > counter (island_island_group_1) (38 ticks): margin 113.9 cm, true clearance 12.1 cm
- sink_to_counter_s18 (look-around map): carrying; filter active on 71% of the ticks since, mostly base > hinge cabinet (stack_02_main_group_1_1) (319 ticks): margin 0.4 cm, true clearance 6.9 cm
- sink_to_counter_s19 (look-around map): carrying; filter active on 34% of the ticks since, mostly base > hinge cabinet (stack_3_right_group_1_1) (116 ticks): margin 15.7 cm, true clearance 21.5 cm
- counter_to_cabinet_s17 (full-pass map): reaching the pick (object moved 0 cm); filter active on 31% of the ticks since, mostly fingers > unknown (obstacle_0) (427 ticks): margin 0.5 cm, true clearance 3.1 cm
- counter_to_cabinet_s17 (look-around map): reaching the pick (object moved 0 cm); filter active on 50% of the ticks since, mostly fingers > unknown (obstacle_0) (744 ticks): margin 0.3 cm, true clearance 5.1 cm
- counter_to_cabinet_s18 (look-around map): driving; filter active on 49% of the ticks since, mostly base > basket (clutter_floor_1) (101 ticks): margin -0.2 cm, true clearance -1.2 cm
- counter_to_stove_s14 (look-around map): driving; filter active on 21% of the ticks since, mostly base > boxed food (clutter_floor_2) (85 ticks): margin 0.4 cm, true clearance 4.0 cm

## The designated obstacle

- close_cabinet_s3: no obstacle: no obstacle in this condition
- close_cabinet_s7: no obstacle: no obstacle in this condition
- close_cabinet_s8: no obstacle: no obstacle in this condition
- close_cabinet_s9: no obstacle: no obstacle in this condition
- close_fridge_s3: no obstacle: no obstacle in this condition
- close_fridge_s4: no obstacle: no obstacle in this condition
- close_fridge_s5: no obstacle: no obstacle in this condition
- close_fridge_s6: no obstacle: no obstacle in this condition
- counter_to_cabinet_s10: pitcher, 24 cm high, -0.7 cm into Nominal's path (drawn -0.7), 42.6 cm from the object, map top -0.5 cm below the true top
- counter_to_cabinet_s13: no obstacle: map or path checks failed in 3 positions: map top 7 cm below the true top; map top 7 cm below the true top; map top 7 cm below the true top
- counter_to_cabinet_s15: pitcher, 24 cm high, 0.2 cm into Nominal's path (drawn -2.0), 6.8 cm from the object, map top 1.5 cm below the true top
- counter_to_cabinet_s3: pitcher, 24 cm high, -0.1 cm into Nominal's path (drawn -0.3), 8.5 cm from the object, map top -0.5 cm below the true top
- counter_to_sink_s0: pitcher, 23 cm high, -2.9 cm into Nominal's path (drawn -2.8), 7.7 cm from the object, map top 1.2 cm below the true top
- counter_to_sink_s1: pitcher, 26 cm high, -1.6 cm into Nominal's path (drawn -1.6), 7.2 cm from the object, map top 0.9 cm below the true top
- counter_to_sink_s10: pitcher, 24 cm high, -0.6 cm into Nominal's path (drawn -0.7), 10.0 cm from the object, map top 3.7 cm below the true top
- counter_to_sink_s11: kettle_non_electric, 21 cm high, -1.1 cm into Nominal's path (drawn -0.8), 5.9 cm from the object, map top 1.6 cm below the true top
- counter_to_stove_s0: pitcher, 24 cm high, -2.3 cm into Nominal's path (drawn -2.8), 27.7 cm from the object, map top 0.6 cm below the true top
- counter_to_stove_s10: pitcher, 27 cm high, -0.6 cm into Nominal's path (drawn -0.7), 10.8 cm from the object, map top 0.1 cm below the true top
- counter_to_stove_s7: kettle_non_electric, 24 cm high, -2.2 cm into Nominal's path (drawn -2.3), 34.5 cm from the object, map top 1.9 cm below the true top
- counter_to_stove_s9: kettle_non_electric, 18 cm high, -0.6 cm into Nominal's path (drawn -0.4), 20.3 cm from the object, map top 1.4 cm below the true top
- sink_to_counter_s0: no obstacle: no position with a depth in [-3, +1] cm (the closest was 9.7 cm outside)
- sink_to_counter_s11: kettle_non_electric, 21 cm high, -0.9 cm into Nominal's path (drawn -0.8), 20.9 cm from the object, map top 1.7 cm below the true top
- sink_to_counter_s12: pitcher, 23 cm high, -1.8 cm into Nominal's path (drawn -1.4), 34.4 cm from the object, map top 1.0 cm below the true top
- sink_to_counter_s7: pitcher, 23 cm high, -2.4 cm into Nominal's path (drawn -2.3), 38.5 cm from the object, map top 0.9 cm below the true top
- close_fridge_s9: no obstacle: no obstacle in this condition
- close_fridge_s13: no obstacle: no obstacle in this condition
- close_cabinet_s16: no obstacle: no obstacle in this condition
- close_cabinet_s17: no obstacle: no obstacle in this condition
- sink_to_counter_s18: pitcher, 27 cm high, -1.1 cm into Nominal's path (drawn -0.7), 7.9 cm from the object, map top -0.1 cm below the true top
- sink_to_counter_s19: pitcher, 23 cm high, -2.4 cm into Nominal's path (drawn -2.0), 6.5 cm from the object, map top 0.8 cm below the true top
- counter_to_cabinet_s17: pitcher, 23 cm high, -2.3 cm into Nominal's path (drawn -2.4), 28.5 cm from the object, map top 0.6 cm below the true top
- counter_to_cabinet_s18: no obstacle: map or path checks failed in 3 positions: map top 8 cm below the true top; map top 8 cm below the true top; map top 8 cm below the true top
- counter_to_stove_s14: pitcher, 23 cm high, -2.3 cm into Nominal's path (drawn -2.0), 25.9 cm from the object, map top 0.4 cm below the true top
- counter_to_stove_s20: no obstacle: no position with a depth in [-3, +1] cm (the closest was 4.5 cm outside)
- counter_to_sink_s13: no obstacle: no position with a depth in [-3, +1] cm (the closest was 2.9 cm outside)
- counter_to_sink_s19: pitcher, 24 cm high, -0.5 cm into Nominal's path (drawn -2.0), 5.9 cm from the object, map top -0.5 cm below the true top

Runs in which the obstacle had no rows in the manipulation segment (its pruning is not forced):

- sink_to_counter_s7 Fixed margin, look-around map: not in the map
- sink_to_counter_s7 Proposed, look-around map: not in the map
- sink_to_counter_s11 Fixed margin, look-around map: not in the map
- sink_to_counter_s11 Proposed, look-around map: not in the map
- counter_to_cabinet_s3 Fixed margin, look-around map: not in the map
- counter_to_cabinet_s3 Proposed, look-around map: not in the map
- counter_to_cabinet_s15 Fixed margin, look-around map: not in the map
- counter_to_cabinet_s15 Proposed, look-around map: not in the map
- counter_to_stove_s7 Fixed margin, look-around map: not in the map
- counter_to_stove_s7 Proposed, look-around map: not in the map
- counter_to_stove_s9 Fixed margin, look-around map: not in the map
- counter_to_stove_s9 Proposed, look-around map: not in the map
- counter_to_sink_s0 Fixed margin, look-around map: not in the map
- counter_to_sink_s0 Proposed, look-around map: not in the map
- counter_to_sink_s1 Fixed margin, look-around map: not in the map
- counter_to_sink_s1 Proposed, look-around map: not in the map
- counter_to_sink_s10 Fixed margin, look-around map: not in the map
- counter_to_sink_s10 Proposed, look-around map: not in the map
- counter_to_sink_s11 Fixed margin, look-around map: not in the map
- counter_to_sink_s11 Proposed, look-around map: not in the map
- sink_to_counter_s18 Fixed margin, look-around map: not in the map
- sink_to_counter_s18 Proposed, look-around map: not in the map
