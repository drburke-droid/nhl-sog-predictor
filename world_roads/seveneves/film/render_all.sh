#!/bin/bash
# Render every shot. Usage: film/render_all.sh <quality> [step] [out_root]
# Resumable: frames already on disk are skipped.
Q=${1:-animatic}; STEP=${2:-1}; ROOT=${3:-/opt/seveneves_render}
B=${BLENDER:-/opt/blender_dl/blender-4.5.16-linux-x64/blender}
cd "$(dirname "$0")/.."
for S in A_moon B_breakup C_merge D_wide E_rain F_steam H_ladder; do
  "$B" -b -P film/render.py -- $S 0 9999 $Q $ROOT $STEP 2>&1 | grep --line-buffered -E "frame [0-9]+/|Error|Traceback"
done
echo "ALL DONE $Q"
