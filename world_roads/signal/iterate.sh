#!/bin/sh
# search -> close the loop -> verify at zoom 10; repeat until every fire and sightline checks out
cd "$(dirname "$0")"
for i in 1 2 3 4 5 6 7 8; do
  echo "=== round $i"
  python chain.py | tail -2 && python closing.py | head -12 && python verify.py && { echo "VERIFIED"; exit 0; }
done
exit 1
