#!/bin/sh
# render every section (each uses four worker processes), then the score and the final cut
cd "$(dirname "$0")"
SCALE=${1:-1}
for s in open how sprints hurdles distance jumps throws compare close; do
  python3 film.py "$s" "$SCALE" > "out/log_$s.txt" 2>&1 || { echo "failed: $s"; exit 1; }
  echo "done $s"
done
python3 assemble.py
