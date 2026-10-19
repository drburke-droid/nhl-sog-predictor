#!/bin/sh
# render every section, three at a time (each section is resumable)
cd "$(dirname "$0")"
run() { for s in "$@"; do python3 film.py "$s" > "out/log_$s.txt" 2>&1; done; }
run retina smaller open &
run how back eye close &
run uses screen numbers &
wait
echo all done
