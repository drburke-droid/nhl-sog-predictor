#!/bin/sh
until [ -f /home/user/lot/render/s50.log ]; do sleep 30; done
cd /home/user/lot/src
L=/home/user/lot/render
python3 make_cuts.py realtime > $L/cut_rt.log 2>&1
python3 make_cuts.py pegged > $L/cut_peg.log 2>&1
for n in x1 f35 x15; do python3 fly_sound.py $n > /dev/null 2>&1; done
python3 make_film.py > $L/film.log 2>&1
NO_PARKER=1 python3 make_film.py > $L/film_np.log 2>&1
echo CUTS_DONE > $L/cuts.log
