Segment: manipulation; 24 scenes; one run per scene and row.

| Row | No obstacle contact | No contact | Task success | Safe success (obstacle) | Safe success | Median clearance to the obstacle | Median clearance | Runs with h < 0 (ticks) | Median task time | Median QP time | Median rows |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Nominal | 15/24 (43-79%) | 8/24 (18-53%) | 14/24 (39-76%) | 12/24 (31-69%) | 8/24 (18-53%) | 1.5 cm | -0.5 cm | 22 (5374 ticks) | 13.9 s | - | - |
| AEGIS | 19/24 (60-91%) | 11/24 (28-65%) | 15/24 (43-79%) | 13/24 (35-72%) | 10/24 (24-61%) | 1.5 cm | -0.2 cm | 21 (5007 ticks) | 15.7 s | 0.04 ms | 1 |
| OmniGuide | 19/24 (60-91%) | 9/24 (21-57%) | 15/24 (43-79%) | 13/24 (35-72%) | 8/24 (18-53%) | 2.5 cm | -0.4 cm | 23 (4442 ticks) | 13.2 s | - | - |
| Proposed | 23/24 (80-99%) | 20/24 (64-93%) | 15/24 (43-79%) | 14/24 (39-76%) | 13/24 (35-72%) | 3.5 cm | 1.9 cm | 12 (480 ticks) | 13.2 s | 0.12 ms | 521 |

Exact McNemar test, Proposed against each row on the same scenes: scenes where only Proposed has the outcome : scenes where only the other row has it, p.

| Row | No obstacle contact | No contact | Task success | Safe success (obstacle) | Safe success |
|---|---|---|---|---|---|
| Nominal | 9:1, p=0.0215 | 13:1, p=0.00183 | 3:2, p=1 | 5:3, p=0.727 | 6:1, p=0.125 |
| AEGIS | 5:1, p=0.219 | 10:1, p=0.0117 | 3:3, p=1 | 3:2, p=1 | 4:1, p=0.375 |
| OmniGuide | 5:1, p=0.219 | 12:1, p=0.00342 | 3:3, p=1 | 4:3, p=1 | 6:1, p=0.125 |

Counted contacts by robot part: runs with a contact of that part (ticks).

| Row | base | arm | hand | fingers |
|---|---|---|---|---|
| Nominal | 3 (405) | 10 (295) | 12 (1154) | 10 (2340) |
| AEGIS | 3 (302) | 8 (275) | 8 (646) | 9 (1237) |
| OmniGuide | 3 (222) | 11 (346) | 7 (218) | 4 (611) |
| Proposed | 0 (0) | 2 (42) | 2 (170) | 1 (86) |
