# Greatest Seasons: Claude handoff

This is a self-contained handoff of the original analysis and an independent follow-up audit, prepared November 2, 2026.

Read in this order:

1. `CLAUDE_TASK.md` for what to do and expected deliverables.
2. `proposal/RESEARCH_PLAN.md` for the proposed statistical redesign.
3. `proposal/FINDINGS_AND_WARNINGS.md` for verified preliminary findings and limitations.
4. `original_project/AUDIT.md` for the original author's detailed method and limitations.
5. `original_project/audit_recompute.py` and `exploration/era_field_adjustment.py` to examine implementations.

The original project is preserved intact under `original_project/` (taken from `greatest_seasons_audit.zip`). Our experimental scripts and computed tables are under `exploration/`.

Quick reproducibility commands (requires Python 3 and numpy):

```
python original_project/audit_recompute.py
python exploration/validate_critical.py
python exploration/era_field_adjustment.py
python exploration/era_field_robustness.py
python exploration/package_scenarios.py
python exploration/scan_data_quality.py
```

Use these as research baselines, not as settled corrections. The original model can be reproduced accurately, but the proposed league-size and scoring normalization is model-dependent, only applied to NBA/NHL, and is not ready for publication as a definitive ranking.

No external raw HTML caches are supplied. The original author's own data-fetching pipeline may require those missing caches or new source requests; the two standalone calculators need only the included CSV inputs.
