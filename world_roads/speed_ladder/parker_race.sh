#!/bin/sh
# the race with the Parker Solar Probe (RACE=1 PARKER=1, race.py): the machines' renders and dust for the frames that
# differ from the 48x race (frame 413 on; the earlier ones are hard links to the race's), then the cut. Resumable:
# each step skips what is done; markers in render/cuts.log.
cd /home/user/lot/src
B=/home/user/bpyenv/bin/python
L=/home/user/lot/render
M=$L/cuts.log
export RACE=1 PARKER=1 OUTSUF=_pk
done_() { grep -q "^$1\$" $M 2>/dev/null; }
mark() { echo "$1" >> $M; }
for n in x15 f35 spitfire wright; do
  done_ PK_RENDER_$n || { SAMPLES=16 nice -n 5 $B render_flyby.py $n race >> $L/fly_${n}_race_pk.log 2>&1 &&
                          grep -q "ALLDONE $n" $L/fly_${n}_race_pk.log && mark PK_RENDER_$n; }
done
for n in x15 f35; do
  done_ PK_DUST_$n || { python3 flyby_dust.py $n >> $L/fly_${n}_race_dust_pk.log 2>&1 && mark PK_DUST_$n; }
done
mark PK_INPUTS_DONE
