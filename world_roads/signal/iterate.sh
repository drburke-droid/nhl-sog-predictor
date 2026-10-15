#!/bin/sh
# search -> close the loop -> verify at zoom 10; repeat until every fire and sightline checks out
cd "$(dirname "$0")"
for i in 1 2 3 4 5 6 7 8; do
  echo "=== round $i"
  python chain.py > data/chain.log && tail -2 data/chain.log && python closing.py > data/closing.log && head -14 data/closing.log && python verify.py && { echo "VERIFIED"; exit 0; }
done
exit 1
