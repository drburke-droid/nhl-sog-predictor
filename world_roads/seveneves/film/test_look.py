"""Look test: blender -b -P film/test_look.py -- <out_dir> <which> [samples]
which: earth | moon | both"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
from mathutils import Vector
import look

argv = sys.argv[sys.argv.index('--') + 1:]
out_dir, which = argv[0], argv[1]
samples = int(argv[2]) if len(argv) > 2 else 48
parts = argv[3].split(',') if len(argv) > 3 else ['surface', 'clouds', 'atmo']
pct = int(argv[4]) if len(argv) > 4 else 100
sc = look.reset_scene()
sc.cycles.samples = samples
sc.render.resolution_percentage = pct
sun_dir = Vector((1.0, -0.35, 0.12)).normalized()
look.sun(sun_dir, 5.0)
look.add_stars(0.003)
if which in ('earth', 'both'):
    if 'surface' in parts: look.earth(tuple(sun_dir))
    if 'clouds' in parts: look.clouds()
    if 'atmo' in parts: look.atmosphere()
    if 'cl' in parts:
        look.add_clouds_to_surface(bpy.data.objects['Earth'], tuple(sun_dir))
    if 'fast' in parts or 'shell' in parts:
        look.atmosphere_fast(tuple(sun_dir))
    if 'fast' in parts or 'haze' in parts:
        look.add_aerial_haze(bpy.data.objects['Earth'], tuple(sun_dir))
if which == 'earth':
    look.camera((-6, -24, 5), (0, 0, 0), 50)
elif which == 'moon':
    look.moon((0, 0, 0))
    look.camera((-2.0, -7.5, 0.6), (0, 0, 0), 60)
else:
    look.moon((30, 10, 4))
    look.camera((40, 30, 8), (0, 0, 0), 35)
sc.render.filepath = os.path.join(out_dir, f'look_{which}_{"-".join(parts)}.png')
nfr = int(os.environ.get('NFRAMES', '1'))
sc.render.use_persistent_data = True
for i in range(nfr):
    sc.frame_set(i + 1)
    bpy.context.scene.camera.location.x += 0.05 * i
    sc.render.filepath = os.path.join(out_dir, f'look_{which}_{"-".join(parts)}_{i}.png')
    t = time.time()
    bpy.ops.render.render(write_still=True)
    print(f'RENDER {which} frame {i}: {time.time() - t:.1f}s')
