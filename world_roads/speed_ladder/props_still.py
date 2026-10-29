"""A still of every prop in place over the plate, to check the models, the layout and the light."""
import bpy, sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene as S, props as P, layout as L
s = S.reset(samples=int(sys.argv[1]) if len(sys.argv) > 1 else 32)
objs = L.build()
s.render.filepath = "/home/user/lot/render/props_still.png"
import time; t = time.time()
bpy.ops.render.render(write_still=True)
print("render seconds", round(time.time() - t, 1))
