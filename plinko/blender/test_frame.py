"""Benchmark / look-dev: build the scene, place a chip, render one frame.

    blender -b --factory-startup -P blender/test_frame.py -- out/blender_test.png 48 wide
"""
import math
import sys
import time

import bpy

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import scene as S  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
out = argv[0] if argv else "out/blender_test.png"
samples = int(argv[1]) if len(argv) > 1 else 48
shot = argv[2] if len(argv) > 2 else "wide"

engine = argv[3] if len(argv) > 3 else "CYCLES"
M, rig, signs = S.build()
S.render_settings(samples=samples, motion_blur=False, engine=engine)
chip = S.new_chip("chip_test", M)
loc, rot = S.chip_world(5.0 * S.W + 0.03, 6.4 * S.H, 0.6)
chip.location, chip.rotation_euler = loc, rot
mid = S.local_to_world(S.board_to_local(5 * S.W, 1.0, 0))
if shot == "wide":
    S.camera(lens=32, loc=(-2.2, -6.4, 2.1), target=(mid.x + 0.3, mid.y, mid.z - 0.1), fstop=5.6)
elif shot == "close":
    S.camera(lens=60, loc=(loc.x - 0.35, loc.y - 1.25, loc.z + 0.25), target=tuple(loc), fstop=2.8)
for k in range(2):
    loc, rot = S.chip_world(5.0 * S.W + 0.03, (6.4 + 0.2 * k) * S.H, 0.6 + 0.3 * k)
    chip.location, chip.rotation_euler = loc, rot
    bpy.context.scene.render.filepath = out.replace(".png", f"_{k}.png")
    t0 = time.time()
    bpy.ops.render.render(write_still=True)
    print(f"RENDER_TIME frame{k} {time.time() - t0:.1f}s samples={samples} shot={shot} engine={engine}")
