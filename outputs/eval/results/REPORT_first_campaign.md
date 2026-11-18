# Evaluation report (Section IV), state of 2026-11-15

Nothing of this state is committed. Frozen state of the counted runs: `frozen.json` (second freeze, sha256 in
`frozen.sha256`). Amendments and their reasons: `tuning_amendments.md`.

## What was run

- **Instructions.** Chosen by a rule written before any counted result: an instruction is in Table II if pi0.5 without
  a filter succeeds in at least 3 of 6 screening scenes (kitchens that are never test scenes). Passed: close fridge 6/6,
  sink to counter 6/6, close cabinet 4/6, counter to sink, counter to cabinet, counter to stove 3/6 each. Not in the
  table: open drawer 2/6, open cabinet 2/6; slide dishwasher rack and stove to counter (3/6) lost the tie rule
  (`instructions.json`, `screening.csv`).
- **Scenes.** 6 instructions x 4 kitchens = 24 test scenes (`test_scenes.json`), plus the 4 open-drawer scenes of the
  earlier set, reported outside the table.
- **Runs.** 168 Table II runs (7 rows x 24 scenes) and 11 runs on the open-drawer scenes; every one is in `runs.csv`.
- **Map study.** 28 scenes x 2 map kinds x 3 noise seeds = 168 maps (`map_study.csv`, `map_study_stats.json`).
- **Videos.** Nominal and Proposed on the full-pass map for every test scene: outside camera clean and with overlay,
  robot camera, hand camera (`videos.csv`).

## Table II (`table2.csv`, per instruction in `table2_by_instruction.csv`)

| Row | Collision-free | Success | Safe success | Median smallest clearance |
|---|---|---|---|---|
| Nominal | 5 / 24 | 16 / 24 | 5 / 24 | -1.2 cm |
| End-effector filter | 5 / 24 | 14 / 24 | 4 / 24 | -1.3 cm |
| Fixed margin | 17 / 24 | 13 / 24 | 12 / 24 | 2.0 cm |
| Proposed without ISSf | 17 / 24 | 12 / 24 | 12 / 24 | 1.6 cm |
| Proposed | 17 / 24 | 14 / 24 | 11 / 24 | 2.4 cm |
| Fixed margin, look-around map | 6 / 24 | 15 / 24 | 4 / 24 | -0.4 cm |
| Proposed, look-around map | 12 / 24 | 15 / 24 | 8 / 24 | 0.0 cm |

- Proposed against Nominal, matched scenes: 12 scenes are collision-free only with Proposed, none only with Nominal.
- Proposed against Fixed margin and against Proposed without ISSf on the full-pass map: no difference in collision-free
  scenes (one scene each way). The hypothesis that the per-point radius and the ISSf term reduce collisions on the
  full-pass map is not supported by these runs.
- Look-around map: 7 scenes are collision-free only with Proposed, 1 only with Fixed margin.
- QP time (median over runs): 0.10 ms with about 740 rows. One row per map point instead of the composed barrier,
  replayed on the Proposed runs: 186 000 rows and 4.0 ms in the QP, 281 ms per tick (`text_numbers.json`).

## Why the filtered methods still collided (`collisions.csv`, from the logs; `scripts/eval_collisions.py`)

Proposed, full-pass map, 7 scenes with a collision (a scene can have two causes):

| Cause | Scenes | What happened |
|---|---|---|
| Object not in the map | 3 | One basket model (Basket044, a floor obstacle) is not reconstructed by the 5 cm map in 3 of the 4 scenes it appears in; the base drives into it. Nominal and every filter collide with it. |
| Robot sphere touches, barrier still positive | 2 | The upper half of a condiment bottle is not in the map (points end at 1.00 m, the bottle at 1.08 m); the hand touches the cap. |
| Robot body outside its spheres | 3 | Arm or hand touches while no sphere does (bottle, boxed drink, the faucet of the sink, which is not in the map either). |
| Object outside the segment's reach | 1 | The policy drove the base 2.9 m from the stop point during manipulation; the objects there are outside the 1.5 m region for which the segment's objects were selected. |

The same causes hold for Fixed margin and Proposed without ISSf. On the look-around maps, 10 scenes of the Fixed margin
row and 11 of the Proposed row collide with an object the map does not contain.

## Map study

Share of objects with reconstructed surface outside their balls, full-pass maps: Proposed 7.4 % at risk level 0.01
(mean radius 6.3 cm); fixed radius 6 cm 12.2 %, 8 cm 4.3 %, 10 cm 1.3 %. Look-around maps: Proposed 2.1 % (mean radius
8.3 cm); fixed 6 cm 9.6 %, 8 cm 3.1 %. The bound of 1 % is not met on either map kind.

Two surface definitions are reported. "Reconstructed" (above): the true surface within reach of the object's map points.
"Seen in a frame": every true surface point a camera frame saw; with it nearly every object fails on the full-pass map,
because thin parts seen by the camera have no mesh at 5 cm voxels. The first definition was added after the first
numbers were seen (post hoc).

## Check of smaller voxels on the objects the 5 cm maps missed (post hoc, mapping only, 4 collision scenes)

Highest reconstructed point of the object, by voxel size (the maps themselves were deleted in the housekeeping of
2026-11-15; they can be rebuilt from the scenes):

| Object (true top) | 5 cm | 2.5 cm | 2 cm |
|---|---|---|---|
| Wire basket Basket044 (0.19 m) | not reconstructed | not reconstructed | not reconstructed |
| Condiment bottle (1.08 m), two scenes | 0.99 m | 1.07 m | 1.07 m |
| Boxed drink (1.02 m) | 0.98 m | 1.02 m | 1.02 m |
| Sink with faucet (1.22 m) | 0.94 m | 1.12 m | 1.21 m |

Mapping one scene: 2.5 cm 7 to 10 min and about 14 GB of memory; 2 cm 25 to 39 min and 17 to 19 GB (5 cm: about 100 s).

## Task success of the filtered methods

- Counter to stove: every whole-body filter 0 / 4 (Nominal 2 / 4). The picked object lies on a plate; the counter under
  the plate kept its rows against the gripper (see A8 below).
- Open drawer (outside the table, 4 scenes): Nominal 1 success, Proposed 1 success; the gripper is held off the
  neighbouring drawer fronts.

## Language model

Qwen3-14B labels the objects of each segment once per scene and map (`model_decisions.jsonl`). Its choice of the
objects the gripper may touch is not used in Table II; against the plan on the 24 scenes it gives, full-pass maps:
33 correct, 20 wrongly allowed, 11 missed, 429 correctly kept; look-around maps: 16, 16, 4, 142
(`qwen_object_choice.jsonl`).

## Failures and repeats (`../repeats.csv`)

- 4 look-around runs (close_fridge_s6 and counter_to_cabinet_s15, both rows) crashed twice: no object had rows in the
  manipulation segment and the row builder could not handle an empty set. Code error, fixed after the campaign (an
  empty set gives no rows); the 4 runs were then made once.
- Screening: one out-of-memory failure (run repeated alone) and one scene whose snapshot did not reproduce (clutter
  positions were stored rounded; fixed before any counted run).
- First freeze (same day): withdrawn after its first scene; its run files are in `outputs/eval_debug/first_freeze/` (videos deleted).

## Deviations from EVAL_SPEC.md

- 6 instructions and 24 scenes instead of 5 and 20; instruction set by the screening rule above.
- Horizon 1800 ticks instead of 1200.
- Allowed contacts are pairs (gripper: fingers, hand, wrist flange; object), taken from the plan, instead of whole
  exempt objects. Contacts on these pairs are logged and not counted, for every method.
- The base path is planned once on the full-pass map with the clutter, with two layers (base; column and shoulder).
- QP cost in joint velocities as fractions of their limits; desired base input is the base velocity under the
  policy's commands alone. The paper needs one sentence for the cost (text in `tuning_amendments.md`, A6).
- Mapping camera without multisampling; grouped fixtures restored in the ground-truth graph.
- Section 9: the map-building clips are plain renders of the mesh vertices in the order they were first seen
  (`videos/map_building/`, with the mesh data); the three scene exports are not made yet.

## A8: correction of the allowed contacts, held-out check (post hoc)

See `tuning_amendments.md`, A8. The gripper may now also touch what each plan object stands on (the counter under the
plate). It changes the pairs only in the 8 scenes of counter to stove and sink to counter. It is checked on 8 new
scenes of these two instructions (`test_scenes_a8.json`, `frozen_a8.json`); `table2_a8.csv` combines them with the 16
untouched scenes.

Result on the 8 new scenes (full-pass map; collision-free / success of 8): Nominal 1 / 4, End-effector 3 / 2, Fixed
margin 7 / 2, Proposed without ISSf 6 / 1, Proposed 5 / 1; look-around map: Fixed margin 4 / 2, Proposed 5 / 1. The
correction does not restore task success on scenes it was not derived from: Proposed finishes 1 of 8 (the 8 replaced
scenes under the old rule: 2 of 8; the not-counted rerun of those scenes with the correction: 4 of 8). The explanation
"the counter under the plate blocks the gripper" is therefore not sufficient for the low success of the filtered
methods on these two instructions.

`table2_a8.csv` (16 untouched + 8 new scenes; collision-free / success / safe success of 24): Nominal 5 / 16 / 5,
End-effector 8 / 14 / 4, Fixed margin 19 / 13 / 12, Proposed without ISSf 18 / 10 / 10, Proposed 17 / 13 / 10; look-around
map: Fixed margin 8 / 12, Proposed 14 / 13 (collision-free / success). `collisions.csv` covers the second-freeze scenes only.
