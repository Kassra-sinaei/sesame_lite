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
