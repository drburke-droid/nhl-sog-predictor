#!/bin/bash
# Render every section of the film to out/<section>.mp4, in order.
cd "$(dirname "$0")"
for s in $(python3 -c "import film; print(' '.join(film.ORDER))"); do
  if [ ! -s "out/$s.mp4" ]; then
    python3 film.py "$s" || exit 1
  fi
done
echo ALL_DONE
