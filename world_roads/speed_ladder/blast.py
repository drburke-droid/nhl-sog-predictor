"""Test 2: the same ball at 17,500 mph (orbital speed, 7.82 km/s) 0.6 m above the road, 12 m out.

It crosses the 24 m of road in view in 3 ms, inside a single frame, so in real time you see one frame of glowing
streak, then the aftermath: the air along its path takes about 1 MJ per metre (drag, rough estimate; a quarter of a
kilo of TNT per metre), a cylindrical blast that throws a wall of dust and gravel off the road and the sand.
This renders what Blender does well: the streak, and the gravel and sand clods (Newtonian particles thrown from a strip
along the path, bouncing on the ground). The dust is blast_dust.py; comp.py puts them together.
  python blast.py [scale%] [samples] [first last]"""
import bpy, sys, os, math, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene as S
argv = sys.argv[1:]
SCALE = int(argv[0]) if argv else 50
SAMPLES = int(argv[1]) if len(argv) > 1 else 32
FPS, K, END = 25, 8, 68                       # the pass happens during frame K
Y, Z = 12.0, 0.6
OUT = f"/home/user/lot/render/blast_{SCALE}/b"

s = S.reset(samples=SAMPLES, scale=SCALE)
s.frame_start, s.frame_end = K, END              # nothing to render before the pass
s.render.use_border, s.render.use_crop_to_border = True, False
s.render.border_min_x, s.render.border_max_x = 0.0, 1.0
s.render.border_min_y, s.render.border_max_y = 1 - 720 / 804, 1 - 150 / 804
s.render.use_motion_blur, s.render.motion_blur_shutter = True, 0.5
s.cycles.volume_step_rate = 2.0
ground = bpy.data.objects["ground"]
col = ground.modifiers.new("collision", "COLLISION")
ground.collision.damping_factor, ground.collision.friction_factor = 0.55, 0.7


def link(o):
    if o.name not in s.collection.objects: s.collection.objects.link(o)


def keyvis(o, frames):
    """visible to the renderer only on these frames"""
    for f in range(0, END + 1):
        o.hide_render = f not in frames
        o.keyframe_insert("hide_render", frame=f)


# --- the streak: what a camera with a 1/50 s shutter records of a glowing ball crossing in 3 ms
bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.07, depth=40, location=(0, Y, Z), rotation=(0, math.pi / 2, 0))
streak = bpy.context.active_object; streak.name = "streak"
m = bpy.data.materials.new("plasma"); m.use_nodes = True
nt = m.node_tree; nt.nodes.clear()
em = nt.nodes.new("ShaderNodeEmission"); em.inputs["Color"].default_value = (0.82, 0.88, 1.0, 1); em.inputs["Strength"].default_value = 60
out = nt.nodes.new("ShaderNodeOutputMaterial"); nt.links.new(em.outputs[0], out.inputs["Surface"])
streak.data.materials.append(m)
streak.visible_shadow = False
keyvis(streak, {K})
# --- gravel and sand clods
bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=1.0, location=(0, 0, -50))
rock = bpy.context.active_object; rock.name = "rock"
random.seed(4)
for v in rock.data.vertices:
    v.co *= 0.7 + 0.6 * random.random()
rm = bpy.data.materials.new("gravel"); rm.use_nodes = True
rm.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.32, 0.24, 0.17, 1)
rm.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.9
rock.data.materials.append(rm)
me = bpy.data.meshes.new("strip")
me.from_pydata([(-15, Y - 0.8, 0.02), (15, Y - 0.8, 0.02), (15, Y + 0.8, 0.02), (-15, Y + 0.8, 0.02)], [], [(0, 1, 2, 3)])
em_o = bpy.data.objects.new("debris", me); link(em_o)
ps = em_o.modifiers.new("debris", "PARTICLE_SYSTEM").particle_system.settings
ps.count, ps.frame_start, ps.frame_end, ps.lifetime = 2500, K, K + 1, 400
ps.emit_from = "FACE"
ps.normal_factor, ps.factor_random = 9.0, 6.0
ps.object_align_factor = (0.0, 0.0, 3.0)
ps.physics_type = "NEWTON"
ps.render_type, ps.instance_object = "OBJECT", rock
ps.particle_size, ps.size_random = 0.03, 0.85
ps.use_rotations, ps.rotation_factor_random, ps.angular_velocity_mode, ps.angular_velocity_factor = True, 1.0, "RAND", 20.0
ps.use_die_on_collision = False
em_o.show_instancer_for_render = False

s.render.filepath = OUT
if __name__ == "__main__":
    import time
    if len(argv) > 3:
        s.frame_start, s.frame_end = int(argv[2]), int(argv[3])
    bpy.ops.wm.save_as_mainfile(filepath="/home/user/lot/blast.blend")
    t = time.time()
    bpy.ops.render.render(animation=True)
    print("render seconds", round(time.time() - t))
