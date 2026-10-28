"""Test 1: a 20 cm steel ball rolls across the road 12 m out at 50 mph (22.35 m/s), left to right."""
import bpy, sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene as S
FPS, V, R, Y = 25, 22.352, 0.10, 12.0
s = S.reset(samples=48)
s.render.use_motion_blur = True
s.render.motion_blur_shutter = 0.5
b = S.ball(s, R)
x0, n = -13.0, 36
for f in range(0, n + 1):
    x = x0 + V * f / FPS
    b.location = (x, Y, R)
    b.rotation_euler = (0.0, (x - x0) / R, 0.0)
    b.keyframe_insert("location", frame=f)
    b.keyframe_insert("rotation_euler", frame=f)
for fc in b.animation_data.action.fcurves if hasattr(b.animation_data.action, "fcurves") else []:
    for k in fc.keyframe_points: k.interpolation = "LINEAR"
s.frame_start, s.frame_end = 0, n
# only the band the ball and its shadow pass through
s.render.use_border, s.render.use_crop_to_border = True, False
s.render.border_min_x, s.render.border_max_x = 0.0, 1.0
s.render.border_min_y, s.render.border_max_y = 1 - 580 / 804, 1 - 440 / 804
s.render.filepath = "/home/user/lot/render/roll/r"
bpy.ops.render.render(animation=True)
