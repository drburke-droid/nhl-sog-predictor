#!/bin/bash
# Render every section of film C that isn't already in out_c/ (or only those named).
cd "$(dirname "$0")"
secs="${@:-$(python3 -c "import sys; sys.argv=['x','x']; import cfilm; print(' '.join(cfilm.ORDER))")}"
for s in $secs; do
  if [ ! -s "out_c/$s.mp4" ]; then
    echo "render $s $(date -u +%H:%M:%S)"
    python3 cfilm.py "$s" > "out_c/render_$s.log" 2>&1 || { echo "FAILED $s"; exit 1; }
  fi
done
echo ALL_DONE
