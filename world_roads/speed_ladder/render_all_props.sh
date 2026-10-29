#!/bin/bash
# all three sequences, resumable; logs to /home/user/lot/render/props.log
cd /home/user/lot/src
for seq in roll blast replay; do
  SAMPLES=32 /home/user/bpyenv/bin/python -u render_props.py $seq 2>&1 | grep --line-buffered -E "^(roll|blast|replay) |ALLDONE|Traceback|Error:" 
done
echo PROPS_ALLDONE
