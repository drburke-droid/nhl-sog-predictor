"""Quick preview frames of a Sandglass film's Cycles scene, on the CPU.

    blender -b out/blender_film/every_grain_of_us.blend -P quick_render.py -- OUT_DIR PCT SAMPLES FRAME [FRAME ...]
    e.g. ... -- previews 25 16 4800 5130 5559 5896 6640      (25% of 4K = 960x540, ~15 s a frame)

Writes OUT_DIR/f_NNNN.png. Uses the scene's own camera, lights and grade, so what you see is the film.
"""
import bpy, sys, time

argv = sys.argv[sys.argv.index("--") + 1:]
out, pct, spp = argv[0], int(argv[1]), int(argv[2])
frames = [int(v) for v in argv[3:]]
sc = bpy.data.scenes["Film"]
sc.render.resolution_percentage = pct
sc.cycles.samples = spp
sc.cycles.device = "CPU"
for f in frames:
    sc.frame_set(f)
    sc.render.filepath = f"{out}/f_{f:04d}.png"
    t = time.time()
    bpy.ops.render.render(write_still=True, scene="Film")
    print("FRAME", f, f"{time.time() - t:.1f}s", flush=True)
