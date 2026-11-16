# Amendments to the tuning procedure (made after a tuning round; each approved by Kasra)

(Rounds 1-3 and their amendments A1-A3 were removed on Kasra's instruction, 2026-11-13; they remain in the git history of
the eval branch. Still in force from them: the floor-obstacle clip of 3 cm with the 0.15 m way-past check, and logs and
raw tuning runs in outputs/eval_debug/.)

## A4 (2026-11-13, after round 3; Kasra's decisions for round 4)

- The shared path is planned on the full-pass map that contains the clutter (planner unchanged: base footprint band,
  inflation = base half-width 0.25 m + 0.10 m = 0.35 m around every ball centre plus its radius); clutter placed by
  the same seeded rule (relative to the path planned without clutter). One plan per scene and voxel size, never
  replanned; look-ahead 0.2 m. The planner found a path in all five tuning scenes.
- Base spheres: a complete cover of the base box (64 spheres, radius 0.106 m, 4.6 cm past the faces; largest gap
  2.9 mm on 5 mm samples; 136 robot spheres in all).
- Voxel check on counter_to_sink_s0 at 2.5 and 2 cm (voxel_check.json): the slab edge stays mostly outside the
  balls. Round 4 at 5 cm and at 2 cm (tuning_r4_v0.05.csv, tuning_r4_v0.02.csv).
- The prior classifier now accepts more than 100 candidates (checked in chunks; labels unchanged): the 10 m path of
  open_drawer_s1 had more candidates in its corridor (first attempt failed twice, logged in repeats.csv).

## A5 (2026-11-13, after round 4; Kasra's decisions for round 5)

- Pair-level pruning (method "pairs"): robot groups base, arm (links 0-7), hand, fingers. During manipulation the pairs
  (hand, object) and (fingers, object) are dropped from the QP for the whole object, for the task's objects: the plan's
  roles (target, container, the sink of a sink task) and the support the picked object stands on. Every other pair,
  and every pair while driving, stays. The pairs come from a fixed rule over the plan's roles (no language model);
  every run logs them (dropped_pairs).
- Robot spheres back to the 100-sphere set (the 64-sphere base cover of round 4 is dropped).
- Round 5: Nominal, Proposed with the round-4 finger patch (reference) and Proposed with pair-level pruning, on the
  five tuning scenes, on the existing 5 cm and 2 cm maps (round-4 plans), horizon 1800 ticks. Every contact is
  logged with its robot group, contacts on exempt pairs separately.

Correction to A5 (Kasra, same day): the language model chooses the pairs, in the loop; that is the Proposed method.
- "pairs_model" (Proposed): Qwen3-14B on the local vLLM server (QwenPairClassifier, same endpoint, sampling and seed as
  the existing QwenClassifier), once per manipulation step, gets the instruction, that step's object list (stage-1
  candidates: stable_id, category with box size, kind) and the four robot groups, and returns the pairs to drop. The
  object pruning uses the existing Qwen classifier instead of the category rule. A reply that cannot be used drops no
  pair (all stay active) and is logged with its reply time.
- "pairs_rule": the fixed rule above, as the reference (category rule for the object pruning).
- Runs per scene and map: Nominal, Proposed with the finger patch (round-4 reference), pairs_model, pairs_rule.
- Pair prompt frozen before any model run: PAIR_SYSTEM_PROMPT sha256 5a4ac623762234f268850e2baed75bbf90bd4197405d879c26423af4a773fe71, reply schema sha256 35b8cdc9ac8c6af4271a42d7976441fe41f807f214672fbd64cb3e5d42d41c25
  (object-pruning prompt unchanged, sha256 23977911def81c2c5823a57886c22f043b428d85b8a6545cf27d1612e2e6877f).

## A6 (2026-11-15, after round 5; Kasra's decisions for round 6)

Diagnosis of round 5: in every failed filtered run the filter moved the base 0.2-1.9 m during manipulation while the
policy commanded only the arm; the path planner looked only at the base's height band and drove the base under a
counter overhang (the shoulder hit the slab).
- Path planner: two layers, the base (0-0.38 m, inflation 0.35 m as before) and the rigid column and shoulder
  (0.30-1.09 m, inflation = its largest reach from the base centre + the same 0.10 m margin = 0.27 m). Plans of the
  tuning scenes replanned (outputs/eval_debug/scenes_v0.05_p2, scenes_v0.02_p2).
- QP (PDF eq. 9 / paper eq. 19): (a) the desired base input u_d is the base velocity under the policy's commands
  alone (before: the coasting of the measured velocity, which carried the filter's own corrections over); (b) the cost
  is taken in normalized units, each joint velocity as a fraction of its velocity limit (base 0.45 m/s, 0.45 m/s,
  1.33 rad/s; arm 2.17 rad/s for joints 1-4 and 2.61 rad/s for joints 5-7). The rows are unchanged. One-step replay
  on the 227 logged close-fridge states: base speed commanded 0.111 m/s (round 5), 0.091 with (a), 0.019 with (a)+(b).
- Exempt robot part of the pair pruning: fixed, the whole gripper = fingers, hand and wrist flange (link 7, now in
  the group "hand"); the model chooses only the objects (an object it names in a pair with the hand or the fingers);
  pairs it returns with the base or the arm are not applied and are logged. Pair prompt unchanged.
- No change to the robot spheres (100).
- Round 6: Proposed with the finger patch, with model pairs and with rule pairs on close_fridge_s4,
  sink_to_counter_s0 and counter_to_sink_s0 (the tuning scenes that Nominal completes), 5 cm maps (the paper's voxel
  size), horizon 1800 ticks.
- Scoring (Kasra, 2026-11-15): a contact on an exempt pair is not a collision. With pair pruning this includes a
  gripper contact with a task object of the plan that the map has no node for (the filter has no constraint against
  it and the model cannot name it). Re-scored from the logged contacts, the model-pairs run of sink_to_counter_s0 in
  round 6 has 0 collision ticks (148 before, all fingers on the unmapped garlic).

## A7 (2026-11-16; preparation of the campaign, after round 6)

- Pair prompt version 2 (PAIR_SYSTEM_PROMPT sha256 b882316d286771e93fc9c3acec7c546bf276863f71674a2c10b60bcfb4c72d2c),
  written once before any run with it: the object list also gives each object's distance from the robot, its
  direction as seen from the robot and the height of its centre, computed from the map relative to the plan's standoff
  pose, nearest first. Reason: with version 1 the model could not tell which of several drawers or cabinets the
  instruction meant (round 5). No other wording changed.
- The model's decisions (object labels per segment, dropped pairs) are asked once per scene and map kind and stored in
  the scene folder (model_<map>.json); every method of the scene uses them, and Nominal is scored with the same pairs.
- Table II methods: in every filtered row the model prunes the objects and chooses the pairs (the proposed pruning);
  the rows differ only in what Section 5 of EVAL_SPEC.md names (rows for the hand only, one fixed radius, no ISSf).

## Freeze (2026-11-16, Kasra's decisions)

Round 6 (5 cm maps, 1800 ticks): all nine filtered runs on close_fridge_s4, sink_to_counter_s0 and counter_to_sink_s0
finished their task (round 5: one of six). Constants unchanged since 145856e (alpha 2, d_max 0.03, kappa 2000 1/m,
delta 0.01, voxel 5 cm, c 3.6e-3); what changed is recorded in A4-A7. Frozen state: frozen.json (with the hashes of the
prompts, the test scenes and the model's decisions). Instructions: open drawer, close fridge, close cabinet, sink to
counter, counter to cabinet, counter to stove (second screening: 3/6 each for counter to cabinet, counter to stove and
stove to counter; Kasra chose the first two and added sink to counter). 24 test scenes rebuilt and selected with the
final rules (test_scenes.json); a scene in a kitchen that a tuning round ran on is marked used_for_tuning (4 scenes).
After this point nothing changes; if something forces a change, the campaign stops.

## Second freeze (2026-11-16)

Stage 1 of the first freeze (open_drawer_s3, now marked used_for_tuning; its runs are kept in outputs/eval_debug/first_freeze
and are not counted): no method opened the drawer; Proposed had 162 collision ticks, all with two objects outside the
1.0 m manipulation region, after the policy drove the base 0.5 m; the model had freed the gripper against another drawer
of the stack. The model's choice of objects, checked against the plan on the 24 scenes: full maps TP 33, FP 20, FN 11,
TN 429; look-around maps TP 16, FP 16, FN 4, TN 142 (qwen_object_choice.jsonl).
Changes (Kasra, 2026-11-16: no deviation from main.pdf):
- Allowed contacts come from the plan, as main.pdf states them ("an object that the plan does not allow it to touch"):
  the gripper may touch the plan's task objects and the support of the picked object. The language model labels the
  objects of each segment (main.pdf eq. 20). Asking the model for the pairs is not used in Table II.
- Manipulation region 1.0 -> 1.5 m (code constant): successful unfiltered runs move the base up to 0.39 m from the stop
  point, from where the arm reaches about 1.45 m.
- Stored labels: a method whose balls differ slightly (no ISSf margin) takes its objects' labels from the scene's stored
  requests.
Check before the freeze (outputs/eval_debug/tuning/r7, not counted, the five used_for_tuning scenes): close_fridge_s4
done (also without ISSf), sink_to_counter_s0 done, open_drawer_s0 done, all without collision; open_drawer_s3 not done,
0 collision ticks; open_drawer_s1 not done, 5 collision ticks (hand on the counter).
Frozen state: frozen.json, sha256 in frozen.sha256. No commit until Kasra asks for it.

## Instruction set by the policy's own ability (2026-11-16, written before any counted result of the second freeze was read)

Kasra: the VLA is not the contribution; instructions that pi0.5 cannot do without the filter say nothing about the
filter and are replaced by instructions it does better. Rule, fixed here:
- Every candidate instruction is screened with pi0.5 WITHOUT the filter on the first 6 buildable seeds from 1000
  (kitchens that are never test scenes), 1800 ticks, full evaluation scenes (drive, clutter).
- Candidates: open drawer, close fridge, close cabinet, sink to counter, counter to sink, open cabinet, slide the
  dishwasher rack (to be screened now); counter to cabinet, counter to stove, stove to counter (screened: 3/6 each).
- An instruction is in Table II if the unfiltered policy succeeds in at least 3 of its 6 screening scenes. If more than
  six pass, the six with the most screening successes are taken; ties go first to the instructions of the frozen set
  (open drawer, close fridge, close cabinet, sink to counter, counter to cabinet, counter to stove), then in the order
  of the candidate list above.
- The choice uses only the unfiltered policy on screening kitchens: no result of a filtered method and no test scene.
  Counted runs already made for an instruction that does not pass are kept and reported in REPORT.md, outside Table II.

## A8 (2026-11-15, after the second-freeze campaign; POST HOC, written by Claude while Kasra was away, pending his decision)

Second-freeze result on the 24 scenes (full-pass map): collision-free / success / safe success: Nominal 5 / 16 / 5,
End-effector 5 / 14 / 4, Fixed margin 17 / 13 / 12, Proposed without ISSf 17 / 12 / 12, Proposed 17 / 14 / 11. These
numbers stand as the pre-registered result (table2.csv) whatever follows.

Finding from these test results: every whole-body filter is 0 of 4 on counter to stove. The allowed contacts are "the
plan's task objects and the support the picked object stands on". The code took the support from RoboCasa's placement
fixture of the picked object only. In counter to stove the picked object lies on a plate (a task object) and has no
placement fixture, so the counter under the plate kept its rows against the gripper; in sink to counter the same holds
for the counter under the destination plate. With 5 cm voxels the gripper is then held about 7 cm above that counter.

Change (code only, same sentence of the rule): the gripper may touch the plan's task objects and what each of them
stands on (the placement fixture of every plan object, not only of the picked one). By a dry check on all 24 scenes
this adds the counter under the plate in the 8 scenes of counter to stove and sink to counter and changes no pair in
the other 16 scenes. Scoring follows the same pairs for every method, Nominal included.

Diagnostic on the same 8 scenes (outputs/eval_debug/tuning/r8, Proposed only, not counted, not held out): success 2 -> 4
of 8 (counter_to_stove_s0 and sink_to_counter_s11 now done), collision-free runs 5 -> 5 of 8; counter_to_stove_s9
collided on more ticks (13 -> 45).

Because the change follows from test results, it is judged on scenes that no run has used:
- Replacement scenes: for sink to counter and counter to stove (in this order), the seeds after the last one considered
  in test_scenes.json, by the same selection rule (kitchen not excluded, not a screening kitchen, not the kitchen of any
  test scene or earlier replacement, scene could be built); the first 4 taken per instruction (test_scenes_a8.json).
- All seven rows are run on these 8 scenes with the changed rule and every other value of the second freeze.
- Reported side by side, whatever the outcome: table2.csv (second freeze, 24 scenes, unchanged) and table2_a8.csv (the
  16 scenes the change does not touch, with their second-freeze runs, and the 8 replacement scenes). The 8 replaced
  scenes stay in runs.csv. Which table the paper uses is Kasra's decision.

Two crash fixes made after the campaign, neither changes a finished run: (1) filter.py could not build rows for a segment
with no object at all (4 look-around runs crashed; fixed and run once, see repeats.csv); (2) scene preparation crashed on
a task object without collision geometry (now recorded as a seed that cannot be built).

## Housekeeping (2026-11-15, Kasra's request)

Deleted to save storage: the raw runs of tuning rounds 4 to 6 (their summaries are tuning_r4_*.csv, tuning_r5_*.csv and
the text above), all tuning videos, the voxel-variant scenes (scenes_v0.02, scenes_v0.025, scenes_v0.05 and their _p2
copies), the videos of the first freeze, and the maps of prepared scenes that no run used (their scene.json or
failed.json stays, so the scene selection can be reproduced). Kept in outputs/eval_debug/: tuning/r7, tuning/r8, the run
files of the first freeze, the logs of the campaign.

## Overnight plan, step 1 (2026-11-16): back to the first Table II version

The code is back to the version that produced table2.csv (second freeze). Removed by hand: the A8 rule (re-added in
step 3) and the stage-1 code (2.5 cm voxels, frame subsampling, command delay and base lag, pooled fixed radius, map
checks); env.py, filter.py, eval_run.py and eval_scene.py match the hashes recorded in frozen_a8.json for the parts that
were not changed after it. Kept: the row builder's fix for a segment without any object (filter.py; the 4 look-around
runs of table2.csv were made with it), the scene preparation's error for a task object without collision geometry, and
the analysis scripts eval_collisions.py and eval_map_clip.py. Check (revert_check.json): Nominal and Proposed on three
of the 24 scenes with the stored maps; two scenes reproduce tick for tick, close_fridge_s3 Proposed does not (GPU
nondeterminism of pi0.5 and of the sphere model; its outcome changed in all three reruns).
