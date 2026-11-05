#!/bin/bash
# Render the sections of one film (a, b or c) that aren't already in out_<film>/ (or only those named).
#   ./render.sh b                 # every missing section of film B
#   ./render.sh a c6 c2 curve     # just these
cd "$(dirname "$0")"
f="$1"; shift
mod="${f}film"
secs="${@:-$(python3 -c "import sys; sys.argv=['x','x']; import $mod; print(' '.join($mod.ORDER))" 2>/dev/null)}"
mkdir -p "out_$f"
for s in $secs; do
  if [ ! -s "out_$f/$s.mp4" ]; then
    echo "render $f $s $(date -u +%H:%M:%S)"
    python3 "$mod.py" "$s" > "out_$f/render_$s.log" 2>&1 || { echo "FAILED $f $s"; tail -5 "out_$f/render_$s.log"; exit 1; }
  fi
done
echo "ALL_DONE $f"
