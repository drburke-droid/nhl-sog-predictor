#!/bin/bash
# most-countries chain: order + layered path, zoom-10 check, repeat until clean
cd "$(dirname "$0")"
[ "$1" = "resume" ] || python tour.py || exit 1          # regions + distance matrix (border-margin summits)
for r in 1 2 3 4 5 6 7 8 9 10; do
  echo "=== round $r"
  python tour.py order || exit 1
  if python verify_tour.py; then echo VERIFIED; exit 0; fi
done
echo "not clean after 8 rounds"; exit 1
