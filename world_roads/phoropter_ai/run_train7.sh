#!/bin/bash
# A long practice run on the version-7 exam that keeps going: if training stops for any reason (a crash, out of
# memory), it picks up from the last full checkpoint (latest.pt, written every 20 updates) after 10 seconds.
# It stops when the run finishes (final.pt) or when a file named STOP appears in the run folder.
#
#   OUT=runs/p ./run_train7.sh >> runs_p.log 2>&1 &      (start, or restart after a machine restart)
#   touch runs/p/STOP                                     (stop after the current attempt; delete the file to allow restarts)
cd "$(dirname "$0")"
OUT=${OUT:-runs/p}
STEPS=${STEPS:-2e9}
ARGS="--clinic --env7 --remark 0.5 --gamma 1.0 --steps $STEPS --ent 0.01 --lr 3e-4 --eval_every 40 --out $OUT"
mkdir -p "$OUT"
while [ ! -f "$OUT/ckpt/final.pt" ] && [ ! -f "$OUT/STOP" ]; do
  if [ -f "$OUT/ckpt/latest.pt" ]; then
    echo "$(date -u '+%F %T') resuming from $OUT/ckpt/latest.pt"
    python train.py $ARGS --resume "$OUT/ckpt/latest.pt"
  else
    echo "$(date -u '+%F %T') starting $OUT from random weights"
    python train.py $ARGS
  fi
  code=$?
  [ -f "$OUT/ckpt/final.pt" ] && break
  echo "$(date -u '+%F %T') training stopped (exit $code); resuming in 10 s"
  sleep 10
done
echo "$(date -u '+%F %T') supervisor done"
