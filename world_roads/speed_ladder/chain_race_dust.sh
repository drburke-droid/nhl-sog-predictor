#!/bin/sh
# once the race is rendered (its depth hides the dust behind the machines): the race's dust
cd /home/user/lot/src
L=/home/user/lot/render
while ! grep -q RACE_DONE $L/r48.log 2>/dev/null; do sleep 30; done
for n in f35 x15; do
  RACE=1 python3 flyby_dust.py $n > $L/fly_${n}_race_dust.log 2>&1
done
echo RACE_DUST_DONE >> $L/r48.log
