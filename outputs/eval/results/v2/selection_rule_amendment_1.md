# Amendment 1 to selection_rule.md (2026-11-17 ~06:10 UTC, before any Fixed margin run at alpha = 4)

Rule 1 named rho_fixed = 3.9 cm. That number is the definition (EVAL_SPEC.md section 5: the median Proposed ball
radius on the seed-100 design maps) evaluated with the default constants (alpha = 2). Every Proposed radius contains
m_d = d_bar^2 / (4 eps alpha), which is 0.75 cm at alpha = 2 and 0.375 cm at alpha = 4, so at alpha = 4 the definition
gives 3.915 - 0.375 = 3.54 cm. The Fixed margin row now follows the definition for every constant set
(scripts/eval_run.py: rho_fixed.json minus the default m_d plus the run's m_d; the value used is in each result file,
"rho_fixed_m"). At alpha = 2 nothing changes. No Fixed run at alpha = 4 had been made when this was written.

Disclosure about the header of selection_rule.md: when it was written, Proposed's alpha = 4 runs (10 of 10) and 6 of
the 10 runs of Proposed without ISSf at alpha = 4 had already been seen (an incomplete comparison printed at 05:50);
the Fixed margin and end-effector runs at alpha = 4 and all arm-setting checks had not.
