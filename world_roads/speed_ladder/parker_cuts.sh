#!/bin/sh
# after parker_race.sh's renders and the jets' dust: the race with Parker, then the same with light. Markers in
# render/cuts.log; each step skipped once done.
cd /home/user/lot/src
L=/home/user/lot/render
M=$L/cuts.log
export RACE=1 PARKER=1 OUTSUF=_pk
done_() { grep -q "^$1\$" $M 2>/dev/null; }
mark() { echo "$1" >> $M; }
until done_ PK_RENDER_spitfire && done_ PK_RENDER_wright && done_ PK_DUSTRUN_x15 && done_ PK_DUSTRUN_f35; do sleep 30; done
done_ PK_CUT_DONE || { nice -n 5 python3 make_race_parker.py > $L/cut_race_parker.log 2>&1 && mark PK_CUT_DONE; }
done_ LT_CUT_DONE || { nice -n 5 python3 make_race_light.py > $L/cut_race_light.log 2>&1 && mark LT_CUT_DONE; }
mark PK_ALL_DONE
