## Read first

- **The campaign was stopped at 15:07 UTC at Kasra's request**, after 10 counted runs of item a (Nominal and Proposed
  on the 5 close-fridge/close-cabinet scenes listed first). No item is complete, so there is no new Table II. The 10
  runs are reported below and in runs.csv. In none of them did the robot touch the designated obstacle; its smallest
  clearance to it was 6-46 cm (34-46 cm in four of the five scenes). The obstacle, as the rule places it, barely
  engages the filter.
- **The report was not ready at 07:30 EST, and the campaign started late.** Steps 1-5 were done by 05:41 UTC. Between
  05:41 and 14:28 UTC this session did not run at all (no command was issued, the GPU was idle), so step 6 started
  only at 14:29 UTC (09:29 EST). Expected: item a (24 scenes, five rows, 120 runs) about 4 hours after the start, item
  b (48 runs) about 1.5 hours after that. Until then the tables below are empty or partial; this file is regenerated
  when an item completes. Item c (12 further scenes) will not fit in the day.
- **Step 1 check, one scene differs.** Nominal and Proposed were rerun on three of the 24 scenes with the stored maps
  and labels (`results/revert_check.json`). close_cabinet_s3 and sink_to_counter_s0 reproduce tick for tick.
  close_fridge_s3 does not: the configuration is identical, but the runs are not bit-reproducible on the GPU. Nominal's
  joint state is identical for 600 ticks, then pi0.5 returns a slightly different action (by at most 0.0035) for the
  same input; Proposed differs by 1e-14 from tick 349 and the difference grows. Its outcome changed in every rerun
  (stored: not done, no contact; reruns: not done with 14 contact ticks; done; done). Nothing in the configuration
  could be fixed; the plan's fallback (continue, state it here) was taken. Single runs of a scene are therefore one
  sample of a policy whose outcome can flip between identical runs.
- **Step 5 used other scenes than the plan names.** The plan names counter_to_sink_s0, sink_to_counter_s0 and
  close_fridge_s4, but these are three of the 24 campaign scenes, while the step and the general rules say:
  non-campaign scenes only, nothing tuned on campaign scenes. They were replaced, by a rule written before any run
  (`results/step5_rule.json`, SHA-256 in `step5_rule.sha256`), by the first screening scene of the same instruction:
  counter_to_sink_s1000, sink_to_counter_s1000, close_fridge_s1001 (close_fridge_s1000 could not be built).
- **The new 24-scene table is a post hoc condition on scenes that were seen before.** The obstacle was added after the
  first table's results were known. The constants, the filter and the pruning are those of the first table.

## What was done

1. Code reverted by hand to the version that produced the first table (commit "Revert to the first Table II
   version"); see the check above.
2. Storage: 35.0 GB freed (outputs/ 36 GB -> 1.2 GB). Deleted: outputs/eval_debug (all tuning, stage-1 and voxel
   files), every video, every map and label file of the prepared scenes, the runs of the A8 check. Kept: the first
   table's run files (outputs/eval/runs_first_campaign/), result files (suffix _first_campaign) and reports, the
   scenes' scene files, the map-study files, the design maps of rho_fixed. Also kept, not part of the evaluation:
   outputs/interview/ (85 MB, Kasra's interview videos of 2026-11-16), and the older milestone folders
   outputs/archive_2026-11-10, maps, viz, demos (257 MB together).
3. Plate fix (A8) and positions logged every tick (commit 2): free objects' positions, the whole joint state, and the
   position of every counted contact.
4. Designated safety obstacle (commit 3), rule in `src/sesame_lite/eval_scene.py` (OBSTACLE_RULE, OBSTACLE_TARGET).
   - Implementation note: the three candidates (a bottle, a can, a box) are created from a copy of RoboCasa's random
     stream and left out of its placement. A first version that added them as ordinary task objects changed the
     kitchen RoboCasa builds for a seed (RoboCasa draws placements from the same stream and rebuilds the kitchen when
     a placement fails); this was caught by the check below and corrected before any scene was built.
   - Every rebuilt scene is checked against its earlier runs: every fixture and object at the same position (within
     1 mm; observed at most 0.05 mm, rounding). For the four A8 kitchens of step 5 only the layout and style could be
     checked: their earlier runs had been deleted in step 2.
   - In practice the obstacle is a bottle: the boxes and cans drawn so far were 8-14 cm high, below the 15 cm minimum.
5. D = 0.18 m (commit 4), by the frozen rule (`results/step5_rule.json`, result in `step5_result.json`): Proposed
   finished 6 of the 7 scenes without the obstacle and 2 of 7 with it at 0.12 m, so 0.12 m failed the rule.
   - In no run at 0.12 m did the robot touch the obstacle; its smallest clearance to it was 15-44 cm. The lost tasks
     are not the filter holding the gripper off the obstacle. Whether the obstacle changes pi0.5's own behaviour (it
     is in the camera images) or the outcome flipped by run-to-run variation (see the step 1 check) cannot be told
     from these runs: Nominal was not run in step 5.
   - Not run in step 5: Nominal in every condition, and Proposed at 0.18 m. They do not enter the rule and were left
     out so as not to delay the campaign. The maps and runs of step 5 were deleted as the plan says; their numbers are
     in step5_result.json.
6. Campaign: runs without videos (there was no time to render them); every run logs its states.
