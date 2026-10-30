#!/bin/sh
# the real-time passes again with a 180-degree shutter (1/50 s): only the frames with a machine in them
cd /home/user/lot/src
B=/home/user/bpyenv/bin/python
for n in x15 f35 x1 spitfire; do
  EXPO=0.02 OUTSUF=_s50 SAMPLES=48 $B render_flyby.py $n > /home/user/lot/render/fly_${n}_s50.log 2>&1
done
echo S50_DONE > /home/user/lot/render/s50.log
