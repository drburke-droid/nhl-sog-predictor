"""Render the frames of a saved .blend that aren't on disk yet:  python resume.py file.blend first last"""
import bpy, sys
blend, a, b = sys.argv[-3], int(sys.argv[-2]), int(sys.argv[-1])
bpy.ops.wm.open_mainfile(filepath=blend)
s = bpy.context.scene
s.render.use_overwrite, s.render.use_placeholder = False, True
s.frame_start, s.frame_end = a, b
bpy.ops.render.render(animation=True)
print("ALLDONE")
