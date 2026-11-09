"""Render a close-up from the film's Cycles scene for a thumbnail (run from cycles/, beside the .blend).

The thumbnails' two renders (frame 6302, mid-1944, the densest war column: Ukraine and Belarus):
  T_GLOW=1.6 T_STREAK=1.5 T_STREAK_GAIN_MAX=4 T_MIN_PX=0.5 \
    blender -b two_thousand_years_of_catastrophe.blend -y -P ../thumbnails/thumb_render.py -- \
    6302 962 262 6 78 5.5 20 50 8 1920 1080 128 render_embers_1944.png
  T_GLOW=1.4 T_STREAK=1.5 T_STREAK_GAIN_MAX=4 T_MIN_PX=0.5 T_SHIFT_X=0.15 T_SHIFT_Y=-0.04 \
    blender -b two_thousand_years_of_catastrophe.blend -y -P ../thumbnails/thumb_render.py -- \
    6302 1010 258 14.9 55 0.8 30 100 8 1920 1080 128 render_macro_1944.png
T_* environment variables: the loader's glow constants for a close camera (the film's are set for the wide
shot), and the lens shift that puts the focused grain off centre.

blender -b two_thousand_years_of_catastrophe.blend -y -P thumb_render.py -- \
    FRAME TX TY TZ TILT DIST HEADING LENS FSTOP WIDTH HEIGHT SAMPLES OUT
TX, TY: map pixels the camera looks at; TZ: height of that point (map px); TILT: degrees from straight down;
DIST: Blender units from the target; HEADING: degrees the camera turns about the vertical (0 = looking north).
"""
import math, sys
import bpy
from mathutils import Matrix, Vector

a = sys.argv[sys.argv.index("--") + 1:]
frame = int(a[0]); tx, ty, tz, tilt, dist, head, lens, fstop = map(float, a[1:9])
W, H, samples, out = int(a[9]), int(a[10]), int(a[11]), a[12]

sc = bpy.data.scenes["Film"]
if bpy.context.window_manager.windows:
    bpy.context.window_manager.windows[0].scene = sc
tgt = Vector((tx / 100.0 - 9.6, 5.4 - ty / 100.0, tz / 100.0))
t, h = math.radians(tilt), math.radians(head)
back = Vector((-math.sin(h) * math.sin(t), -math.cos(h) * math.sin(t), math.cos(t)))
loc = tgt + back * dist
f = -back
r = f.cross(Vector((0, 0, 1))).normalized()
u = r.cross(f)
cam = sc.camera
m = Matrix.Identity(4)
for i in range(3):
    m[i][0], m[i][1], m[i][2], m[i][3] = r[i], u[i], -f[i], loc[i]
cam.matrix_world = m
cam.data.lens = lens
import os as _os
cam.data.shift_x = float(_os.environ.get('T_SHIFT_X', 0))
cam.data.shift_y = float(_os.environ.get('T_SHIFT_Y', 0))
cam.data.dof.use_dof = fstop > 0
cam.data.dof.focus_distance = dist
cam.data.dof.aperture_fstop = max(fstop, 0.1)
sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = W, H, 100
sc.cycles.samples = samples
sc.render.use_persistent_data = False
import os
src = bpy.data.texts["sand_loader.py"].as_string()          # the film's loader, with a calmer glow up close
for k in ("GLOW", "STREAK", "STREAK_GAIN_MAX", "MIN_PX"):
    if os.environ.get("T_" + k):
        import re
        src = re.sub(rf"^{k} = [0-9.]+", f"{k} = {os.environ['T_' + k]}", src, flags=re.M)
exec(compile(src, "sand_loader_thumb", "exec"), {"__name__": "sand_loader_thumb"})
sc.frame_set(frame)
print("grains", len(bpy.data.objects["Sand"].data.vertices), "rising", len(bpy.data.objects["Lifted"].data.vertices), flush=True)
sc.render.filepath = out
sc.render.image_settings.file_format = "PNG"
bpy.ops.render.render(write_still=True, scene=sc.name)
print("THUMB", out, flush=True)
