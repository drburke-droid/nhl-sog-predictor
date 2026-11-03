#!/bin/sh
# Regenerate every follow-up output from data/ (no network, no download cache needed). Run from followup/.
set -e
cd "$(dirname "$0")"
python3 tests/test_reproduction.py
python3 analysis/cv_window.py v2 > outputs/cv_window_trend.txt
python3 analysis/cv_comparator.py > outputs/cv_comparator.txt
python3 analysis/env_field_tests.py v2 > /dev/null
python3 analysis/beta_diagnostics.py > /dev/null
python3 analysis/simulate.py > /dev/null
python3 analysis/run_all.py ${QUICK:+--quick}
python3 analysis/corrections_log.py
python3 analysis/plots.py
python3 analysis/report.py
python3 tests/test_round2.py
echo done
