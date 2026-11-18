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
