#!/bin/bash
# Render every shot at 1920 wide on the CPU: four single-thread workers share each shot.
# Resumable: finished frames are kept, half-written ones are cleared and redone.
#   ./render_hd.sh [samples]
cd "$(dirname "$0")" || exit 1
PY=${PY:-/opt/mixenv/bin/python}
SPP=${1:-16}
mkdir -p renders/hd
for K in Q1 Q2 Q3 Q4 S1 S2 F1 X6 X5; do
    mkdir -p renders/hd/$K/frames
    find renders/hd/$K/frames -name '*.png' -size 0 -delete
    echo "$(date '+%m-%d %H:%M') rendering $K" >> renders/hd/progress.log
    for i in 1 2 3 4; do
        $PY -W ignore render_shot.py blender/shot_$K.blend --percent 50 --samples $SPP --out renders/hd --threads 1 \
            > renders/hd/$K.worker$i.log 2>&1 &
    done
    wait
done
echo "$(date '+%m-%d %H:%M') render complete" >> renders/hd/progress.log
