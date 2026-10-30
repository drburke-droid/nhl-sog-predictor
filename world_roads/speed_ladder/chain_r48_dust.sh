#!/bin/sh
# once the 48x replays are rendered: their dust at the same moments
cd /home/user/lot/src
L=/home/user/lot/render
while ! grep -q R48_DONE $L/r48.log 2>/dev/null; do sleep 30; done
for n in x1 x15; do
  REPLAY48=1 OUTSUF=_r48 python3 flyby_dust.py $n > $L/fly_${n}_dust_r48.log 2>&1
done
echo R48_DUST_DONE >> $L/r48.log
