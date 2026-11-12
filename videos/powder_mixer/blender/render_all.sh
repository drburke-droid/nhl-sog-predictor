#!/bin/sh
# Renders the nine shots on the GPU, then assembles the film.
# Safe to stop and start again: frames that already exist are skipped.
# Usage: ./render_all.sh [path to blender]
BLENDER="${1:-blender}"
cd "$(dirname "$0")" || exit 1
for S in Q1 Q2 Q3 Q4 S1 S2 F1 X6 X5; do
    "$BLENDER" -b "shot_$S.blend" -P use_gpu.py -a || exit 1
done
"$BLENDER" -b the_mix_off_MASTER.blend -a && echo "Finished: the_mix_off_4K0001-5613.mp4"
