#!/bin/sh
# the replays again at the one rate of the 48x cut (a camera at 1,200 frames a second), then the final cut's race
cd /home/user/lot/src
B=/home/user/bpyenv/bin/python
L=/home/user/lot/render
for n in ${R48_LIST:-x1 x15 spitfire}; do
  REPLAY48=1 OUTSUF=_r48 SAMPLES=16 $B render_flyby.py $n replay >> $L/fly_${n}_replay_r48.log 2>&1
done
echo R48_DONE >> $L/r48.log
for n in ${RACE_LIST:-x15 f35 spitfire wright}; do
  RACE=1 SAMPLES=16 $B render_flyby.py $n race >> $L/fly_${n}_race.log 2>&1
done
echo RACE_DONE >> $L/r48.log
