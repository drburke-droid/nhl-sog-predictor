"""Render thumbnail stills from a Sandglass film .blend with custom cameras (any frame, any angle).

    blender -b FILM.blend -P thumb_shot.py -- shots.json OUTDIR PCT SAMPLES CODE_DIR [name,name]

shots.json: [{"name": "exit", "frame": 1105, "pose": [tx, ty, tilt, dist, heading], "fstop": 1.1}, ...]
  pose is the film's camera pose (camera.py): look at map px (tx, ty) from `dist` Blender units, tilted
  `tilt` degrees from straight down, turned `heading` degrees (0 = looking north, 90 = east, -90 = west).
  Leave out "pose" to use the film's own camera at that frame. "focus" (default dist) and "lens" optional.
CODE_DIR: the film's code folder (for camera.py), e.g. ancestry_sand.
PCT 25 + SAMPLES 24 is a ~20 s CPU draft at 960x540; PCT 50 + SAMPLES 96 is the final 1920x1080 still.
Low cameras (tilt 60-76) with f/0.9-2 depth of field read as macro photographs of a real model.
"""
import bpy, sys, json, time
from mathutils import Matrix
argv = sys.argv[sys.argv.index("--") + 1:]
shots = json.load(open(argv[0])); out = argv[1]; pct = int(argv[2]); spp = int(argv[3])
sys.path.insert(0, argv[4])
import camera as CM
only = set(argv[5].split(",")) if len(argv) > 5 else None
sc = bpy.data.scenes["Film"] if "Film" in bpy.data.scenes else bpy.context.scene
sc.cycles.device = "CPU"; sc.render.resolution_percentage = pct; sc.cycles.samples = spp
cam = sc.camera
cam.animation_data_clear(); cam.data.animation_data_clear()
for s in shots:
    if only and s["name"] not in only:
        continue
    sc.frame_set(s["frame"])
    pose = s.get("pose") or CM.pose_at(s["frame"] / CM.FPS)
    loc, r, u, f = CM.basis(pose)
    m = Matrix.Identity(4)
    for i in range(3):
        m[i][0], m[i][1], m[i][2], m[i][3] = r[i], u[i], -f[i], loc[i]
    cam.matrix_world = m
    cam.data.lens = s.get("lens", CM.LENS)
    cam.data.dof.use_dof = True
    cam.data.dof.focus_distance = s.get("focus", pose[3])
    cam.data.dof.aperture_fstop = s.get("fstop", 0.9)
    sc.render.filepath = f"{out}/{s['name']}.png"
    t = time.time(); bpy.ops.render.render(write_still=True, scene=sc.name)
    print("SHOT", s["name"], f"{time.time() - t:.0f}s", flush=True)
