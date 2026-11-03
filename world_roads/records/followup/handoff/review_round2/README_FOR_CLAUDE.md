# Claude: review feedback and correction package

Please read `INDEPENDENT_REVIEW_FOR_CLAUDE.md` first.

The uploaded follow-up research makes real progress; I independently ran its 15 reproduction checks, all passing. This package documents two material faults in the analysis output, supplies a repaired model-comparison join and runnable source, and identifies where statistical claims should be narrowed.

Contents:
- `INDEPENDENT_REVIEW_FOR_CLAUDE.md`: prioritized review, evidence and fix list.
- `repair_comparison.py`: standalone repair script (expects the existing `followup/` folder adjacent to this script or update the `ROOT` path).
- `comparison_join_fixed.csv`: four core variants correctly aligned by stable athlete IDs.
- `comparison_join_collisions.csv`: two aliases in original outputs collapsing to one corrected identity.
- `comparison_join_unresolved.csv`: original-to-v2 ID matches that could not be resolved by source crosswalk (currently zero; historical intentionally removed records still appear as original-only rows in the comparison).

Critical work: implement the identity crosswalk in `run_all.py` so all variants, not just these four, are correctly aligned; decide whether the category-tail calibration is exponential or GPD, then align implementation, equations and labels.

Return the changed code, refreshed outputs and newly added tests. Do not claim significance from an expected count below one without a properly calibrated multiple-testing framework.
