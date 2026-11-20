#!/bin/bash
# Render the impact-ladder film's shots. Usage: film/render_ladder.sh <quality> [step] [out_root]
Q=${1:-animatic}; STEP=${2:-1}; ROOT=${3:-/opt/seveneves_render}
B=${BLENDER:-/opt/blender_dl/blender-4.5.16-linux-x64/blender}
cd "$(dirname "$0")/.."
for S in H_ladder I1_1m I2_60m I3_1km E3_1km I4_10km E4_10km I5_100km E5_100km; do
  "$B" -b -P film/render.py -- $S 0 9999 $Q $ROOT $STEP 2>&1 | grep --line-buffered -E "frame [0-9]+/|Error|Traceback"
done
echo "ALL DONE $Q"
