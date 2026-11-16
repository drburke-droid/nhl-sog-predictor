import bpy, sys, time
argv = sys.argv[sys.argv.index("--") + 1:]
out, frames, spp = argv[0], [int(v) for v in argv[1].split(",")], int(argv[2])
sc = bpy.data.scenes["Film"]
sc.cycles.device = "CPU"; sc.cycles.samples = spp
sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = int(argv[3]), int(argv[4]), 100
cam = sc.camera.data
cam.sensor_fit = "AUTO"     # the film's horizontal field of view becomes the vertical one: a wider, taller view
for f in frames:
    sc.frame_set(f); sc.render.filepath = f"{out}/v_{f}.png"
    t = time.time(); bpy.ops.render.render(write_still=True, scene="Film"); print("SHOT", f, f"{time.time()-t:.0f}s", flush=True)
