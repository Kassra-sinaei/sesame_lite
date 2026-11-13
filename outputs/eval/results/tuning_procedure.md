# Stage 0b tuning procedure (written 2026-11-13, before any tuning run)

Scenes: the tuning scene of each instruction (the first test scene, `test_scenes.json`, `used_for_tuning`).

Round 1 (`outputs/eval/tuning/r1/`): Nominal and Proposed on the full-pass map of the 5 tuning scenes, with the
constants of `145856e` (alpha 2, d_bar 0.03 m, tau 0.5 mm (paper kappa = 2000 1/m), delta 0.01, voxel 5 cm,
c 3.6e-3) and the finger patch radius 0.15 m (proposed to Kasra on 2026-11-13).

Per run, reported: task success, contact ticks and touched objects, smallest true clearance, relaxed and hold
ticks, stalls, the pairs that bound the QP when progress stopped, the ticks used.

What may change: only constants shared by every filtered method (FilterPolicy: alpha, d_bar, tau, delta, relaxation
penalty, goal standoff) and the finger patch radius. One set for all instructions, kitchens and methods. Scene rules,
maps, plans and the evaluation code are not changed for a score.

When a change is made: only to remove a failure mechanism seen in a tuning run and named in the report (for example
the filter stops a motion that Nominal completes without contact, or a Proposed run touches an object). Each later
round runs Proposed on all 5 tuning scenes again (`tuning/r2`, `tuning/r3`). At most 3 rounds; then the constants
are frozen and reported with the reason for every change. If no change is needed after round 1, round 1's constants
are frozen.

Horizon: the ticks every Proposed run needs are reported; if a run is cut off at 1200 ticks while still progressing,
a longer common horizon is proposed to Kasra before the freeze.

Tuning runs are not counted. After the freeze, all 20 test scenes, the tuning scenes included, are run again
(stages 1 to 4).
