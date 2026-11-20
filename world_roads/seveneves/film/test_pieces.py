"""blender -b -P film/test_pieces.py -- <out_dir> <spread> [samples] [pct]"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
from mathutils import Vector, Matrix
import look
import moon_pieces

argv = sys.argv[sys.argv.index('--') + 1:]
out_dir, spread = argv[0], float(argv[1])
samples = int(argv[2]) if len(argv) > 2 else 16
pct = int(argv[3]) if len(argv) > 3 else 50
sc = look.reset_scene()
sc.cycles.samples = samples
sc.render.resolution_percentage = pct
sun_dir = Vector((0.2, -1.0, 0.25)).normalized()
look.sun(sun_dir, 5.0)
look.add_stars(0.003)
t = time.time()
obs = moon_pieces.build_pieces(subdiv=6)
print(f'built pieces in {time.time() - t:.1f}s')
for ob in obs:
    off = Vector(ob['offset'])
    ob.location = off * spread
    ob.data.materials[0].node_tree.nodes['glow'].outputs[0].default_value = 1.0
look.camera((0.6, -9.0, 1.2), (0, 0, 0), 55)
sc.render.filepath = os.path.join(out_dir, f'pieces_{spread}.png')
t = time.time()
bpy.ops.render.render(write_still=True)
print(f'RENDER {time.time() - t:.1f}s')
