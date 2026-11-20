# OmniGuide row: tuning rule for the guidance strength

Written 2026-11-19, before the first tuning run. Its SHA-256 is in tuning_rule.sha256.

## Rule (Kasra, 2026-11-19, verbatim)

- Scenes: the tuning scenes of the three tasks (sink_to_counter s1000 s1001, counter_to_cabinet s1003 s1004,
  counter_to_stove s1001 s1002). Never the 14 campaign scenes.
- Values: guidance_factor in {0.25, 0.5, 1, 2} with collision_weight 0.1 (0.5 is their default), plus 0 as the
  unguided reference.
- Rule: keep the largest value whose task success on the tuning scenes is not below the unguided one. Report the
  whole sweep (success and contacts per value), not only the winner.
- If no value passes, or all pass, stop and report. Do not extend the grid on your own.

## How it is applied (fixed now, before any tuning run)

- Scene files: /workspace/sesame_lite/outputs/eval_debug/scenes_obs_xr1/<scene> (the tuning scenes with their
  designated obstacles).
- One run per scene and value, 30 runs, condition of Table II (frozen_v2):
  `scripts/eval_run.py --scene S --method omniguide --guidance-factor G --map full --vla xr1 --disturbance a2
  --policy '{"alpha": 4.0, "d_bar": 0.03}' --horizon 1800 --scenes-root <scene files> --tuning omniguide_gf<G>`,
  code of branch baseline-omniguide at commit eefcfaf, server src/omniguide/server.py on port 10087.
  Value 0 runs the same pipeline with the guidance off (its chunks equal the standard server's).
- Task success of a value: the number of the six scenes whose run completes the task (result "success"; the same as
  "success" of score_segment.py on the manipulation segment).
- "Not below the unguided one": success count of the value >= success count of value 0.
- Contacts per value: Agent A's scripts/score_segment.py (branch baseline-aegis, commit df53606), used unchanged, with
  `--segment manipulation --scenes <the six scenes>`; reported: runs with no obstacle contact, runs with no counted
  contact, safe success, and contacts by robot part. Contacts do not enter the rule.
- Settings of the row that stay fixed for the sweep and the campaign (in the code at eefcfaf): the robot points'
  scale factors gamma_pos (0.01057, 0.01155, 0.00954) and gamma_rot 0.0806 (gamma_fit.json, fitted before any run of
  the row), the distance clamped to 1 mm before the log, the distance field (1 cm voxels, every map point except the
  plan's role objects, box of +-2.5 m around the plan's standoff and -0.1 to 2.3 m in height), the noise pick
  (4 new noises) and the guided pass as in src/omniguide/sampler.py.
- Order: one scene first (counter_to_stove_s1001, guidance_factor 0.5, with video and the cost over time), checked
  before the other 29 runs. A code change after that check is reported, and every run made before it is repeated.
- Anything changed after the sweep's results are seen is reported as post hoc.
