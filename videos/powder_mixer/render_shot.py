"""
Render one of the blender/shot_*.blend files on this machine's CPU at a reduced
size.  Several copies can run at once on the same shot: each skips frames that
exist or that another copy has claimed, so the job is resumable and shareable.

    python render_shot.py blender/shot_Q1.blend --percent 50 --samples 16 --out renders/hd --threads 1
"""
import argparse
import os
import sys

import bpy

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
ap = argparse.ArgumentParser()
ap.add_argument("blend")
ap.add_argument("--percent", type=int, default=50)
ap.add_argument("--samples", type=int, default=16)
ap.add_argument("--out", required=True, help="frames go to OUT/<shot>/frames/f_####.png")
ap.add_argument("--threads", type=int, default=0, help="0 = all cores")
ap.add_argument("--start", type=int)
ap.add_argument("--end", type=int)
a = ap.parse_args(argv)

bpy.ops.wm.open_mainfile(filepath=os.path.abspath(a.blend))
sc = bpy.context.scene
r = sc.render
key = os.path.splitext(os.path.basename(a.blend))[0].replace("shot_", "")
sc.cycles.device = "CPU"
sc.cycles.samples = a.samples
sc.cycles.adaptive_min_samples = min(a.samples, 12)
r.resolution_percentage = a.percent
if a.threads:
    r.threads_mode = "FIXED"
    r.threads = a.threads
if a.start:
    sc.frame_start = a.start
if a.end:
    sc.frame_end = a.end
r.filepath = os.path.join(os.path.abspath(a.out), key, "frames", "f_")
r.use_overwrite = False
r.use_placeholder = True
bpy.ops.render.render(animation=True)
print("done", key, flush=True)
