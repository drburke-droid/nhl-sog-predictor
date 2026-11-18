#!/bin/bash
# Render the single-mixer "chase for 100%" shot on the CPU: every frame for the first 24 s
# of stirring, then every sixth frame (shown as 6x fast-forward).  Resumable.
cd "$(dirname "$0")" || exit 1
PY=${PY:-/opt/mixenv/bin/python}
OUT=renders/chase
mkdir -p $OUT/CH/frames
find $OUT/CH/frames -name '*.png' -size 0 -delete
echo "$(date '+%m-%d %H:%M') rendering" >> $OUT/progress.log
for PASS in "--start 1 --end 576 --step 1" "--start 582 --end 1455 --step 6"; do
    for i in 1 2 3 4; do
        $PY -W ignore render_shot.py cache/chase/shot_CH.blend --percent 50 --samples 16 --out $OUT --threads 1 $PASS \
            > $OUT/worker$i.log 2>&1 &
    done
    wait
done
echo "$(date '+%m-%d %H:%M') render complete" >> $OUT/progress.log
