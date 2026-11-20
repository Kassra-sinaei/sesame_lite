# OmniGuide row (Agent B), comparison campaign v3

Tags: **[m]** measured, **[c]** read in the code, **[e]** my estimate. Branch `baseline-omniguide`, commit `eefcfaf`
(off `4c0a7e4`, not pushed). Every run of the sweep and the campaign used this commit [m]. All files named here are in
this folder: `/workspace/sesame_lite/outputs/eval/results/v3/omniguide/`.

## Result on the 14 scenes (guidance_factor 2)

Scored with Agent A's `scripts/score_segment.py` (commit `df53606`, unchanged), manipulation segment only [m]. Proposed
is the stored v2 run of each scene; its navigation is identical to ours, tick by tick, in all 14 scenes [m].

| Row | No obstacle contact | No contact | Task success | Safe success (obstacle) | Safe success | Median clearance |
|---|---|---|---|---|---|---|
| OmniGuide | 11/14 | 4/14 | 7/14 | 6/14 | 4/14 | -0.4 cm |
| Proposed | 13/14 | 11/14 | 9/14 | 8/14 | 7/14 | 1.8 cm |

Only "no contact" differs clearly from Proposed (8 scenes where only Proposed has no contact, 1 the other way; exact
McNemar p = 0.04) [m]. Full tables: `campaign.md`, `campaign_table.csv`, `campaign_runs.csv`. Row template for the
four-row table: `runs/{scene}/omniguide_full.json`. One caveat: see question 1 (sink_to_counter_s7 was recorded with video).

## Tuning (rule written before the first tuning run)

`tuning_rule.md` (SHA-256 in `tuning_rule.sha256`), 30 runs, scored with the same script [m]:

| guidance_factor | Task success | No contact | No obstacle contact |
|---|---|---|---|
| 0 (unguided) | 5/6 | 3/6 | 4/6 |
| 0.25 | 4/6 | 2/6 | 4/6 |
| 0.5 | 4/6 | 3/6 | 4/6 |
| 1 | 5/6 | 4/6 | 5/6 |
| 2 | 5/6 | 3/6 | 5/6 |

Values 1 and 2 are not below the unguided success, 0.25 and 0.5 are; the rule keeps 2 [m]. Neither stop condition
applies. `tuning_result.json` (with SHA-256) fixed the value before the first campaign run. Full tables: `tuning_sweep.md`.

## What was built [c]

- `src/omniguide/sampler.py`, `server.py`: XR1's 5-step loop rewritten outside the no-gradient forward, language model
  pass without gradients, port 10087. Both of their stages, with their update form `x + dt v + gf * grad`. Collision
  term only, collision_weight 0.1.
- Robot points: the gripper and the point 0.25 m along its -z axis, integrated from the chunk's deltas, put into the
  world with the base pose at the query (the base's motion during the chunk is ignored, as in theirs).
- Cost: theirs, with the distance clamped to 1 mm before the log. Their code takes the log with no guard
  (gr00t_n1d6.py:571-574, 709-710): a point on an occupied voxel gives minus infinity.
- `field.py`: every map point except the plan's role objects, 1 cm grid, Euclidean distance transform, built once per
  scene on the simulation side and sent once. Lookup: trilinear value, finite-difference gradient.
- `client.py`: navigation with the Proposed filter, manipulation unfiltered (base included) with the guided policy.
- `fit_gamma.py`, `check_query.py`, `plot_cost.py`: the scale-factor fit, the one-query checks, the cost plot.
- The whole omniguide-gr00t repository is under the NVIDIA License (not only esdf.py), so every ported function is
  rewritten, with its source file and line in a comment. Clones in `third_party/` at `d649d69` / `ef38d07`.

## Checks

- Guidance 0 gives the standard server's actions bit for bit on one real query, same seed [m] (`query_check/`).
- Navigation identical to the stored Proposed runs: 30 of 30 tuning runs, 14 of 14 campaign runs [m].
- Field gradient against finite differences: median cosine 0.99 [m]. With the gripper lowered near the counter, the
  guidance raises the cost value as expected, more at higher strength [m].
- Single scenes (tuning counter_to_stove_s1001, campaign sink_to_counter_s7): video, log and cost over time looked right
  (`tuning_videos/`, `videos/`, `*.omniguide.png`) [m].

## Scale factors of our drive (no new runs)

Fitted from the stored Nominal logs of the 22 non-campaign scenes (`gamma_fit.json`): position (0.0106, 0.0116,
0.0095), rotation 0.081; theirs (0.011, 0.011, 0.02) and 0.15 [m]. Error at the end of a 16-step chunk, on the six
tuning scenes' Nominal logs: gripper 5.6 cm RMS (their factors 8.0 cm, no motion 12.9 cm), the -z point 7.4 cm,
rotation 10 degrees [m]. Over all chunks of the fit logs, including those where the base moves, the gripper error is
about 16 cm, since the model has no base [m].

## Time and reruns

- Extra time per query: about 0.7 s (median round trip 1.16 s guided, 0.47 s unguided, GPU shared with Agent A) [m].
- Reruns with the same seed give the same actions: three reruns of one query, guidance acting, identical [m]; two
  reruns of a whole tuning episode without video, identical tick by tick [m]. This needs strict deterministic mode
  during the backward passes; in the server's warn-only mode the attention backward is non-deterministic and reruns
  differed by up to 0.008 in an action [m].
- Recording video changes the policy's camera inputs: the same episode with video departs from the runs without video
  at the second query, with the robot state still identical [m]. Cause not looked into [e: the extra renderer].

## Differences from their code worth knowing

- Their noise pick compares 4 new noises against the first pass's noise and keeps a new one only if it scores strictly
  higher (gr00t_n1d6.py:598-660); ported as is [c]. It replaced the first noise in 44% of campaign queries [m].
- Their 0.1 m removal of points around the gripper belongs to their camera pipeline and is not applied to our map [c].
- XR1 has 5 steps (theirs 4): the pick scores the estimate after the second step, as theirs [c].
- The field box (2.5 m around the standoff) was fixed before any run. In counter_to_stove_s10 the policy drove the
  base up to 2.6 m away (84 of 752 campaign queries over 1.5 m); there, points beyond the box get no gradient [m, c].
- Guidance was active (a point within 0.15 m) in 56% of campaign queries [m].

## Lines added to existing files [c]

Only `scripts/eval_run.py`; nothing in `src/sesame_lite/`:
- line 36 and 38: "omniguide" in METHODS and TITLES;
- line 60: `if method in ("proposed", "omniguide"):` (the Proposed filter settings, used for navigation);
- line 80: the `--guidance-factor` option;
- lines 90-93: the environment class (`OmniGuideEnv`, the segment switch) for method omniguide;
- lines 96-100: the guided policy for method omniguide (`elif` on the XR1 line);
- lines 115-116: write the query record `<run>.omniguide.json`.

## Requests for Agent A

1. Row template: `/workspace/sesame_lite/outputs/eval/results/v3/omniguide/runs/{scene}/omniguide_full.json`.
2. Video on changes the policy's inputs (above). Please check the video setting of your Nominal and AEGIS runs.
3. "h < 0" in my row counts the Proposed rows (with the stored Qwen labels), evaluated without acting. Please check
   that the Nominal row of the table uses the same rows.

## Questions for Kasra

1. sink_to_counter_s7 was the single check scene, recorded with video, and its run was kept as the campaign run; the
   other 13 ran without video. Count it as it is, or run it again without video and count that run? (Decide before the
   second run exists. It failed with contacts; the other rows' runs were probably made without video, since frozen_v2
   lists eval_replay_video.py, but I did not verify that.)

## Housekeeping

Scratch scripts, logs and frames deleted [m]. The guided server on port 10087 is stopped [m]. Two rerun episodes are
kept as evidence in `rerun_check/`. In the worktree, `outputs/eval/runs`, `outputs/eval/videos`,
`outputs/eval_debug/tuning` and `tuning_videos` are symlinks into this folder (how eval_run.py wrote here).
