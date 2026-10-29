#!/bin/sh
# after a container restart: finish the X-15's dust and composite it, then the F-35, then both films
cd /home/user/lot/src
B=/home/user/bpyenv/bin/python
L=/home/user/lot/render
rm -f /home/user/lot/render/fly_f35_replay/tmp_*.exr
python3 flyby_dust.py x15 > $L/fly_x15_dust.log 2>&1
rm -rf /home/user/lot/out/frames_x15; python3 make_fly.py x15 > $L/comp_x15.log 2>&1 && echo X15_DONE >> $L/resume.log
SAMPLES=48 $B render_flyby.py f35 replay > $L/fly_f35_replay.log 2>&1
python3 flyby_dust.py f35 > $L/fly_f35_dust.log 2>&1
rm -rf /home/user/lot/out/frames_f35; python3 make_fly.py f35 > $L/comp_f35.log 2>&1 && echo F35_DONE >> $L/resume.log
python3 make_film.py > $L/film.log 2>&1
NO_PARKER=1 python3 make_film.py > $L/film_np.log 2>&1
echo ALL_DONE >> $L/resume.log
