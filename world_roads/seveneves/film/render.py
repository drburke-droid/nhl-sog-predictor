"""Render a shot's frames (skips frames already on disk).

blender -b -P film/render.py -- <shot> <first> <last> <quality> [out_root]
quality: preview (half size, 6 samples) | draft (half size, 12) | final (1920x1080, 16)
Writes <out_root>/<quality>/<shot>/0001.png ... and <shot>_times.json (simulated time per frame).
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import shots
import impact  # registers the impact-ladder shots
import scale   # registers the scale cards
import night   # registers the night-sky views from the ground

argv = sys.argv[sys.argv.index('--') + 1:]
name, first, last, quality = argv[0], int(argv[1]), int(argv[2]), argv[3]
root = argv[4] if len(argv) > 4 else '/opt/seveneves_render'
step = int(argv[5]) if len(argv) > 5 else 1
only = None                                       # optional JSON {shot: [frames]}: render just the frames a cut uses
if len(argv) > 6:
    only = set(json.load(open(argv[6])).get(name, []))
out = os.path.join(root, quality, name)
os.makedirs(out, exist_ok=True)
sc = shots.setup(quality)
s = shots.build(name, sc)
last = min(last, s.frames - 1)
tfile = os.path.join(out, 'times.json')
if hasattr(s, 'labels'):                         # screen positions of the on-screen labels, every frame
    lab = {}
    for f in range(s.frames):
        s.frame(f)
        lab[str(f)] = dict(s.labels)
    json.dump(lab, open(os.path.join(out, 'labels.json'), 'w'))
times = json.load(open(tfile)) if os.path.exists(tfile) else {}
for f in range(first, last + 1, step):
    path = os.path.join(out, f'{f + 1:04d}.png')
    if os.path.exists(path) or (only is not None and f not in only):
        continue
    t0 = time.time()
    tsim = s.frame(f)
    times[str(f)] = tsim
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)
    json.dump(times, open(tfile, 'w'))
    print(f'{name} frame {f + 1}/{s.frames}  t={tsim:.0f}s  {time.time() - t0:.1f}s', flush=True)
