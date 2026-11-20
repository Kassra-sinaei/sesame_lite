Segment: manipulation; 14 scenes; one run per scene and row.

| Row | No obstacle contact | No contact | Task success | Safe success (obstacle) | Safe success | Median clearance to the obstacle | Median clearance | Runs with h < 0 (ticks) | Median task time | Median QP time | Median rows |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Nominal | 8/14 (33-79%) | 4/14 (12-55%) | 7/14 (27-73%) | 6/14 (21-67%) | 4/14 (12-55%) | 0.9 cm | -0.4 cm | 13 (3431 ticks) | 15.7 s | - | - |
| AEGIS | 11/14 (52-92%) | 6/14 (21-67%) | 7/14 (27-73%) | 6/14 (21-67%) | 5/14 (16-61%) | 1.1 cm | -0.2 cm | 13 (3828 ticks) | 15.7 s | 0.04 ms | 1 |
| Proposed | 13/14 (69-99%) | 11/14 (52-92%) | 9/14 (39-84%) | 8/14 (33-79%) | 7/14 (27-73%) | 3.6 cm | 1.8 cm | 8 (319 ticks) | 14.1 s | 0.13 ms | 517 |

Exact McNemar test, Proposed against each row on the same scenes: scenes where only Proposed has the outcome : scenes where only the other row has it, p.

| Row | No obstacle contact | No contact | Task success | Safe success (obstacle) | Safe success |
|---|---|---|---|---|---|
| Nominal | 6:1, p=0.125 | 8:1, p=0.0391 | 3:1, p=0.625 | 4:2, p=0.688 | 4:1, p=0.375 |
| AEGIS | 3:1, p=0.625 | 6:1, p=0.125 | 3:1, p=0.625 | 3:1, p=0.625 | 3:1, p=0.625 |

Counted contacts by robot part: runs with a contact of that part (ticks).

| Row | base | arm | hand | fingers |
|---|---|---|---|---|
| Nominal | 3 (405) | 6 (166) | 6 (795) | 5 (1621) |
| AEGIS | 3 (302) | 5 (208) | 5 (335) | 4 (463) |
| Proposed | 0 (0) | 1 (41) | 2 (170) | 1 (86) |
