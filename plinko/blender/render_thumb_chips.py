"""Render single Plinko chips with a transparent background, for compositing onto a photo of a board.

    blender -b --factory-startup -P blender/render_thumb_chips.py
Writes blender/assets/thumb/chip_red.png and chip_blue.png (600 x 600 RGBA, the chip facing the camera, lit by the
studio's lights).
"""
import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene as S  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "thumb")
M, rig, signs = S.build()
S.render_settings(samples=96, res=(600, 600), motion_blur=False)
sc = bpy.context.scene
sc.render.film_transparent = True
sc.render.image_settings.color_mode = "RGBA"
loc, rot = S.chip_world(5.0 * S.W, 6.0 * S.H, 0.35)
cam, tgt = S.camera(lens=200, loc=(loc.x, loc.y - 1.6, loc.z + 0.12), target=tuple(loc), fstop=16)
for face, name in (("chip_face", "chip_red"), ("chip_blue_face", "chip_blue")):
    chip = S.new_chip(name, M, face)
    chip.location, chip.rotation_euler = loc, rot
    for ob in sc.objects:
        if ob.type in ("MESH", "CURVE", "FONT"):
            ob.hide_render = ob is not chip
    sc.render.filepath = os.path.join(OUT, f"{name}.png")
    bpy.ops.render.render(write_still=True)
    chip.hide_render = True
