#!/bin/sh
# the 48x and real-speed cuts composited again (the shock refraction and the ramp readout), while the race renders run
cd /home/user/lot/src
L=/home/user/lot/render
grep -q "^SLOW48B_DONE$" $L/cuts.log || { REPLAY48=1 nice -n 10 python3 make_cuts.py slow48 > $L/cut_slow48b.log 2>&1 && echo SLOW48B_DONE >> $L/cuts.log; }
grep -q "^RT_DONE$" $L/cuts.log || { nice -n 10 python3 make_cuts.py realtime > $L/cut_rt2.log 2>&1 && echo RT_DONE >> $L/cuts.log; }
