#!/bin/sh
# render every section, three at a time (each section is resumable)
cd "$(dirname "$0")"
mkdir -p out
[ -f out/paper_tex.png ] || python3 make_paper.py
run() { for s in "$@"; do python3 film.py "$s" > "out/log_$s.txt" 2>&1; done; }
run blur angle eye close &
run retina chart demo &
run dark recipe open &
wait
echo all done
