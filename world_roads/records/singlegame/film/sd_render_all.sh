#!/bin/bash
# Render every section of the single-day film that isn't already in out/.
cd "$(dirname "$0")"
for s in $(python3 -c "import sdfilm; print(' '.join(sdfilm.ORDER))"); do
  if [ ! -s "out/$s.mp4" ]; then
    python3 sdfilm.py "$s" || exit 1
  fi
done
echo ALL_DONE
