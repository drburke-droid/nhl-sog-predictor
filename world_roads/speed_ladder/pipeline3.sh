#!/bin/bash
# after the props renders: dust with the props' depth, then composite and assemble test 3
cd /home/user/lot/src
until grep -q PROPS_ALLDONE /home/user/lot/render/props.log; do sleep 20; done
python3 roll_dust.py > /home/user/lot/render/roll_dust.log 2>&1
DUST_SCALE=1.0 python3 blast_dust.py 60 > /home/user/lot/render/dust_full.log 2>&1
python3 replay_dust.py > /home/user/lot/render/replay_dust.log 2>&1
rm -rf /home/user/lot/out/frames3 && python3 make_test3.py > /home/user/lot/render/make3.log 2>&1
rm -f /home/user/lot/out/replay/*.png && FIRST=test3_a NAME=test3 python3 make_replay.py 3 >> /home/user/lot/render/make3.log 2>&1
echo PIPE3_DONE >> /home/user/lot/render/make3.log
