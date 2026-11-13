# Amendments to the tuning procedure (made after a tuning round; each approved by Kasra)

## A1 (2026-11-13, after round 1): floor-obstacle placement

Round 1: every Proposed run deadlocked at a floor obstacle while driving (no contact): the filter keeps about 5 cm
true clearance, so with a 6 cm clip the base needs about 11 cm sideways, and the path follower pulls it back.
Constants cannot remove this. Kasra: "Change the floor obstacle position such that the robot can pass around it with
filter and collides without filter."

Change (scene rule, eval_scene.py RULE): clip 6 cm -> 3 cm (above the median path-tracking error of the unfiltered
runs, 0.5-2.5 cm); way-past check: robot shifted by clip + 0.05 m -> clip + 0.12 m must touch nothing. The test
scenes are rebuilt and reselected with the same selection rule. Round 2 runs Nominal and Proposed on the (new)
tuning scenes with the round-1 constants.

## A2 (2026-11-13): location of the raw tuning runs only

On Kasra's instruction, `outputs/eval/` holds only output the paper can reference. The raw runs of the tuning rounds
(not counted) and all logs live in `outputs/eval_debug/` (`tuning/<round>/`, `logs/`), not in `outputs/eval/tuning/`
as tuning_procedure.md says. Nothing else changes; the round summaries stay in `outputs/eval/results/tuning_<round>.csv`.

## A3 (2026-11-13, after round 2): the nominal base follower's look-ahead (Kasra's decision)

Only the nominal path follower changes (pure pursuit, look-ahead 0.2 m before); filter, planner, clutter rule, maps
(5 cm voxels) and the shared, never replanned path stay. One look-ahead for every scene and method; no obstacle
knowledge in the follower. Check before the change: in the 8 floor-obstacle stalls of rounds 1-2 the filtered base
command averaged 1-2 mm/s (about 1 % of the follower's 0.13 m/s) and the base moved 0.2-0.8 mm/s in that direction:
the QP, not the base deadband, stopped it.

Round 3, for L = 0.4, 0.6, 0.8, 1.0 m in this order, stopping at the first L that removes the deadlocks:
1. Guard without clutter (all six clutter objects stored under the floor; filter off; driving only), L = 0.2 m and L:
   the drive must reach the standoff with the same final pose as with 0.2 m (done check unchanged) and must not touch
   a fixture; otherwise L is not used.
2. Nominal and Proposed on the 5 tuning scenes (full-pass map, round-1 constants, patch 0.15 m).
3. Deadlock removed: no Proposed run ends "stalled" while driving without a contact.

Result of A3 (2026-11-13): the guard (follower_guard.jsonl) allowed only L = 0.4 (0.6 touches two fixtures in
close_cabinet_s0 without clutter; 0.8 and 1.0 get stuck on fixtures in two scenes). Round 3 with L = 0.4
(tuning_r3_L0.4.csv): Proposed still stalls while driving in 4 of 5 scenes and ends on the island slab in the fifth.
For information only, Proposed with L = 0.6, 0.8, 1.0 (tuning_r3_L0.6/0.8/1.0.csv): stalls in all 5 scenes at every
value. No value removes the deadlocks; the look-ahead stays 0.2 m and stage 0b stops for Kasra.

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
