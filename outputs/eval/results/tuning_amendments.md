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
