#!/bin/bash
# Final render of the impact-ladder film, shot by shot. After each shot: encode it to
# media/shots/<shot>_final.mp4 (lossless-ish CRF 14), commit and push, so finished work survives a
# recycled container. Resumable (frames on disk are skipped; committed shots are skipped).
set -u
B=${BLENDER:-/opt/blender_dl/blender-4.5.16-linux-x64/blender}
ROOT=/opt/seveneves_render
cd "$(dirname "$0")/.."
mkdir -p media/shots
for S in I3_1km I4_10km I5_100km E3_1km E4_10km E5_100km I1_1m I2_60m H_ladder; do
  OUT=media/shots/${S}_final.mp4
  [ -f "$OUT" ] && { echo "skip $S"; continue; }
  "$B" -b -P film/render.py -- $S 0 9999 final $ROOT 1 2>&1 | grep --line-buffered -E "frame [0-9]+/|Error|Traceback"
  ffmpeg -y -loglevel error -framerate 24 -i $ROOT/final/$S/%04d.png -c:v libx264 -pix_fmt yuv420p -crf 14 -preset slow "$OUT" \
    && cp $ROOT/final/$S/times.json media/shots/${S}_final_times.json \
    && git add "$OUT" media/shots/${S}_final_times.json \
    && git commit -q -m "Impact-ladder film: final render of shot $S" \
    && for i in 1 2 3 4; do git push -q -u origin claude/brave-goldberg-819x9p && break || sleep $((2**i)); done
  echo "SHOT DONE $S"
done
echo "ALL DONE final"
