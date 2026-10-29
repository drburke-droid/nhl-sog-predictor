#!/bin/sh
# render every flyby (real time and replay) and the dust of the supersonic ones, one after another
cd /home/user/lot/src
B=/home/user/bpyenv/bin/python
L=/home/user/lot/render
SAMPLES=40 $B parker_render.py > $L/parker_render.log 2>&1
SAMPLES=48 $B render_flyby.py wright > $L/fly_wright.log 2>&1
SAMPLES=48 $B render_flyby.py spitfire > $L/fly_spitfire.log 2>&1
SAMPLES=48 $B render_flyby.py spitfire replay > $L/fly_spitfire_replay.log 2>&1
SAMPLES=48 $B render_flyby.py x1 > $L/fly_x1.log 2>&1
SAMPLES=48 $B render_flyby.py x1 replay > $L/fly_x1_replay.log 2>&1
python3 flyby_dust.py x1 > $L/fly_x1_dust.log 2>&1
SAMPLES=40 $B render_flyby.py x15 > $L/fly_x15.log 2>&1
SAMPLES=40 $B render_flyby.py x15 replay > $L/fly_x15_replay.log 2>&1
python3 flyby_dust.py x15 > $L/fly_x15_dust.log 2>&1
echo CHAIN_DONE > $L/chain.log
