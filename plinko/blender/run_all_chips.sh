#!/bin/bash
# Render every shot's chip pass, one Blender at a time (each uses all cores).
cd "$(dirname "$0")/.."
for s in "$@"; do
  blender -b --factory-startup -P blender/render_chips.py -- "$s" 2>&1 | grep -E --line-buffered "CHIPS_DONE|Error|Traceback"
done
