#!/bin/bash
# Render the scale cards and the line-up of rocks: film/render_scale.sh <quality> [shots...]
set -u
Q=${1:-animatic}
shift || true
SHOTS=${@:-P_lineup S1_1m S2_60m S3_1km S4_10km S5_100km N3_1km N4_10km N5_100km}
cd "$(dirname "$0")/.."
B=/opt/blender_dl/blender-4.5.16-linux-x64/blender
for S in $SHOTS; do
  $B -b -P film/render.py -- $S 0 9999 $Q 2>&1 | grep --line-buffered -E "frame [0-9]+/|Error|Traceback"
done
echo ALL_DONE
