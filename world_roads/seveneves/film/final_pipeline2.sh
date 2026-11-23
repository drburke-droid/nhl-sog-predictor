#!/bin/bash
# Second pass of the final render: the explainer shots added later (runs after final_pipeline.sh).
set -u
B=${BLENDER:-/opt/blender_dl/blender-4.5.16-linux-x64/blender}
ROOT=/opt/seveneves_render
cd "$(dirname "$0")/.."
while pgrep -f final_pipeline.sh >/dev/null; do sleep 60; done
for S in J_ejecta K_entry P_lineup S1_1m S2_60m S3_1km S4_10km S5_100km N1_1m N3_1km N4_10km N5_100km; do
  OUT=media/shots/${S}_final.mp4
  [ -f "$OUT" ] && { echo "skip $S"; continue; }
  "$B" -b -P film/render.py -- $S 0 9999 final $ROOT 1 2>&1 | grep --line-buffered -E "frame [0-9]+/|Error|Traceback"
  ffmpeg -y -loglevel error -framerate 24 -i $ROOT/final/$S/%04d.png -c:v libx264 -pix_fmt yuv420p -crf 14 -preset slow "$OUT" \
    && cp $ROOT/final/$S/times.json media/shots/${S}_final_times.json \
    && { [ -f $ROOT/final/$S/labels.json ] && cp $ROOT/final/$S/labels.json media/shots/${S}_final_labels.json; true; } \
    && git add "$OUT" media/shots/${S}_final_*.json \
    && git commit -q -m "Impact-ladder film: final render of shot $S" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01S7EUN3W6exPjy1TpJCVruD" \
    && for i in 1 2 3 4; do git push -q -u origin claude/brave-goldberg-819x9p && break || sleep $((2**i)); done
  echo "SHOT DONE $S"
done
echo "ALL DONE final2"
