#!/bin/sh
# render every section at 4K, four workers (each section is resumable: finished 5 s chunks are kept)
cd "$(dirname "$0")"
mkdir -p out
run() { for s in "$@"; do python3 film.py "$s" > "out/log_$s.txt" 2>&1; done; }
run clay why60 close &
run egypt sky tens &
run decimal ladder open &
run greece others clocks &
wait
echo all done
