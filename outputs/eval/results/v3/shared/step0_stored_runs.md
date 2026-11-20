Segment: manipulation; 14 scenes; one run per scene and row.

| Row | No obstacle contact | No contact | Task success | Safe success (obstacle) | Safe success | Median clearance to the obstacle | Median clearance | Runs with h < 0 (ticks) | Median task time | Median QP time | Median rows |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Nominal (stored) | 4/14 (12-55%) | 4/14 (12-55%) | 7/14 (27-73%) | 4/14 (12-55%) | 4/14 (12-55%) | 0.0 cm | -0.4 cm | 12 (3650 ticks) | 13.2 s | - | - |
| End-effector filter (stored) | 10/14 (45-88%) | 5/14 (16-61%) | 6/14 (21-67%) | 5/14 (16-61%) | 4/14 (12-55%) | 1.7 cm | -0.1 cm | 5 (269 ticks) | 12.1 s | 0.11 ms | 309 |
| Proposed | 13/14 (69-99%) | 11/14 (52-92%) | 9/14 (39-84%) | 8/14 (33-79%) | 7/14 (27-73%) | 3.6 cm | 1.8 cm | 8 (319 ticks) | 14.1 s | 0.13 ms | 517 |

Exact McNemar test, Proposed against each row on the same scenes: scenes where only Proposed has the outcome : scenes where only the other row has it, p.

| Row | No obstacle contact | No contact | Task success | Safe success (obstacle) | Safe success |
|---|---|---|---|---|---|
| Nominal (stored) | 9:0, p=0.00391 | 7:0, p=0.0156 | 4:2, p=0.688 | 5:1, p=0.219 | 4:1, p=0.375 |
| End-effector filter (stored) | 3:0, p=0.25 | 6:0, p=0.0312 | 4:1, p=0.375 | 3:0, p=0.25 | 3:0, p=0.25 |

Counted contacts by robot part: runs with a contact of that part (ticks).

| Row | base | arm | hand | fingers |
|---|---|---|---|---|
| Nominal (stored) | 3 (688) | 6 (383) | 9 (970) | 6 (1019) |
| End-effector filter (stored) | 3 (597) | 8 (184) | 2 (196) | 3 (132) |
| Proposed | 0 (0) | 1 (41) | 2 (170) | 1 (86) |
