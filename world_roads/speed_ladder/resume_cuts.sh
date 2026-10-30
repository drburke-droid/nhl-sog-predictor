#!/bin/sh
# everything still to do for the 48x cut and the race, resumable after a container restart (each step skips what is
# done; markers in render/cuts.log). Two lanes: Blender renders, and the numpy dust and compositing.
cd /home/user/lot/src
B=/home/user/bpyenv/bin/python
L=/home/user/lot/render
M=$L/cuts.log
done_() { grep -q "^$1\$" $M 2>/dev/null; }
mark() { echo "$1" >> $M; }
lane_render() {
  for n in spitfire wright; do
    RACE=1 SAMPLES=16 $B render_flyby.py $n race >> $L/fly_${n}_race.log 2>&1
  done
  grep -q "ALLDONE wright" $L/fly_wright_race.log && grep -q "ALLDONE spitfire" $L/fly_spitfire_race.log && mark RACE_DONE
}
lane_comp() {
  done_ R48_DUST_DONE || { REPLAY48=1 OUTSUF=_r48 python3 flyby_dust.py x15 >> $L/fly_x15_dust_r48.log 2>&1 && mark R48_DUST_DONE; }
  done_ SLOW48_DONE || { REPLAY48=1 python3 make_cuts.py slow48 > $L/cut_slow48.log 2>&1 && mark SLOW48_DONE; }
  done_ RT_DONE || { python3 make_cuts.py realtime > $L/cut_rt2.log 2>&1 && mark RT_DONE; }
  while ! done_ RACE_DONE; do sleep 30; done
  for n in f35 x15; do
    done_ RACE_DUST_$n || { RACE=1 python3 flyby_dust.py $n >> $L/fly_${n}_race_dust.log 2>&1 && mark RACE_DUST_$n; }
  done
  done_ RACE_CUT_DONE || { RACE=1 python3 make_race.py > $L/cut_race.log 2>&1 && mark RACE_CUT_DONE; }
}
done_ RACE_DONE || lane_render &
lane_comp
wait
mark ALL_DONE
