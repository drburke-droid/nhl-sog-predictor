#!/bin/sh
# after the first chain and its composites: the F-35, then both films
until grep -q POST_DONE /home/user/lot/render/post.log 2>/dev/null; do sleep 60; done
cd /home/user/lot/src
B=/home/user/bpyenv/bin/python
L=/home/user/lot/render
SAMPLES=48 $B render_flyby.py f35 > $L/fly_f35.log 2>&1
SAMPLES=48 $B render_flyby.py f35 replay > $L/fly_f35_replay.log 2>&1
python3 flyby_dust.py f35 > $L/fly_f35_dust.log 2>&1
rm -rf /home/user/lot/out/frames_f35
python3 make_fly.py f35 > $L/comp_f35.log 2>&1
python3 make_film.py > $L/film.log 2>&1
NO_PARKER=1 python3 make_film.py > $L/film_np.log 2>&1
echo F35_DONE > $L/f35.log
