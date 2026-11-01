"""Side flyby: the B-2 crosses the Bonneville Salt Flats, Utah, left to right at 300 mph, low (125 ft), 500 m out,
in front of the Silver Island Mountains. The plate is the Bureau of Land Management's photograph of the flats
(Flickr blmutah 28776016261, public domain; on Commons as "Bonneville Salt Flats (28776016261).jpg").

Matching the camera: the salt is flat to the horizon, so the far edge of the salt is eye level; it runs level across
the photograph, so the camera is level. The lens isn't recorded (a film scan); ~30 degrees across fits both the
mountains (Graham Peak rising ~900 m above the flats ~15 km off) and the salt crust (5-10 cm clumps at the foot of
the picture, 14 m from a camera 1.6 m up). The sun is low (~15 degrees) and to the right of and a little behind the
camera: the crust's ridges cast shadows to the left, and the faces of the mountains turned to the camera are lit.

The B-2's shadow falls ~140 m beyond it and to the left, a dark sliver racing across the salt near the horizon. The
white salt lights its underside (the ground is a shadow catcher that still bounces light).

    python side.py plate | prep | render | comp | mux
"""
import json, math, os, sys
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

S = dict(
    name="side_bonneville",
    plate_src=os.environ.get("SIDE_PLATE", "bonneville_blm_28776016261.jpg"),   # in $B2_WORK/plates
    horizon_frac=0.3465,                          # far edge of the salt, as a fraction of the photograph's height
    crop_top_frac=0.0,                            # 2.39:1 crop of the full width, from this far down
    hfov=30.0,                                    # degrees across the picture
    cam_z=1.6,
    # flat out (630 mph, its published top speed) at 65 ft, 350 m out: low enough that the air it shoves down and
    # aside, and its jets' wash, tear the salt crust up behind it.  Held after it has gone while the roar arrives
    # and the dust rolls up.
    dist=350.0, alt=20.0, mph=630.0, t_pass=2.0, dur=7.5,
    bank=20.0, aoa=2.0,                           # banked 20 degrees toward the camera so its planform shows (the low
                                                  # wingtip clears the salt by 11 m)

    sun_rel_az=120.0, sun_el=15.0,                # sun: degrees right of the view direction, and up
    sun_w=9.0, sun_rgb=(1.0, 0.82, 0.62), sky=(0.48, 0.56, 0.74), sky_w=0.55,
    salt_albedo=0.72,
    shutter=0.15,                                 # 1/200 s: a 1.4 m smear at 630 mph
    haze=0.018,                                   # blend toward the sky over the 350 m to the aircraft
    debris=True,                                  # the salt it tears up (debris.py)
    paint_gain=None,   # the model's paint is ~2.1x darker than the real aircraft's (topdown's calibration against
                       # the B-2 on Whiteman's ramp); None reads it from that calibration
)
OUT = os.path.join(C.WORK, S["name"])
PW, PH = C.W, C.PIC_H


def v_ms():
    return S["mph"] * C.MPH


def f_px():
    return (PW / 2) / math.tan(math.radians(S["hfov"] / 2))


def horizon_row():
    im_h, im_w = _src_size()[1], _src_size()[0]
    crop_h = im_w / 2.39
    top = S["crop_top_frac"] * im_h
    return (S["horizon_frac"] * im_h - top) / crop_h * PH


def _src_size():
    p = os.path.join(C.WORK, "plates", S["plate_src"])
    return Image.open(p).size


def sun_vec():
    a, e = math.radians(S["sun_rel_az"]), math.radians(S["sun_el"])
    return np.array([math.cos(e) * math.sin(a), math.cos(e) * math.cos(a), math.sin(e)])   # x right, y ahead, z up


def omega():
    """turn rate of a level turn at this bank: g tan(bank) / v (rad/s)"""
    return 9.81 * math.tan(math.radians(S["bank"])) / v_ms()


def pose(t):
    """-> position (x right, y ahead, z up), heading (rad, anticlockwise from +x).  A level turn to the right
    (toward the camera), crossing x = 0 at t_pass heading straight across the picture."""
    t = np.asarray(t, dtype=float)
    w, v, tau = omega(), v_ms(), t - S["t_pass"]
    x = v / w * np.sin(w * tau)
    y = S["dist"] + v / w * (np.cos(w * tau) - 1)
    return np.stack([x, y, np.full_like(x, S["alt"])], -1), -w * tau


def plane_x(t):
    return pose(t)[0][..., 0]


def project(P):
    """world (x right, y ahead, z up) -> picture pixels"""
    P = np.atleast_2d(P)
    u = PW / 2 + f_px() * P[:, 0] / P[:, 1]
    v = horizon_row() - f_px() * (P[:, 2] - S["cam_z"]) / P[:, 1]
    return u, v


def plate():
    im = Image.open(os.path.join(C.WORK, "plates", S["plate_src"])).convert("RGB")
    w, h = im.size
    ch = w / 2.39; top = S["crop_top_frac"] * h
    im = im.crop((0, int(round(top)), w, int(round(top + ch)))).resize((PW, PH), Image.LANCZOS)
    os.makedirs(OUT, exist_ok=True)
    im.save(os.path.join(OUT, "plate.png"))
    print("plate", im.size, "horizon row", round(horizon_row(), 1), "f px", round(f_px(), 1),
          "aircraft row", round(float(project(np.array([0, S["dist"], S["alt"]]))[1][0]), 1),
          "span px", round(52.4 * f_px() / S["dist"], 1))


def scene():
    import bpy
    from mathutils import Vector
    import b2_model
    bpy.ops.wm.read_factory_settings(use_empty=True)
    root = b2_model.build()
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"; sc.cycles.device = "CPU"; sc.cycles.samples = 64; sc.cycles.use_denoising = True
    sc.cycles.use_adaptive_sampling = True
    sc.render.film_transparent = True
    sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = PW, PH, 100
    sc.view_settings.view_transform = "Standard"; sc.view_settings.look = "None"
    sc.render.image_settings.file_format = "PNG"; sc.render.image_settings.color_mode = "RGBA"
    sc.render.image_settings.color_depth = "16"
    sc.render.fps = C.FPS
    sc.render.use_motion_blur = True; sc.render.motion_blur_shutter = S["shutter"]; sc.render.motion_blur_position = "CENTER"
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); sc.collection.objects.link(cam); sc.camera = cam
    cam.data.sensor_fit = "HORIZONTAL"; cam.data.sensor_width = 36.0
    cam.data.lens = f_px() * 36.0 / PW
    cam.data.shift_y = (horizon_row() - PH / 2) / PW
    cam.data.clip_end = 20000
    cam.location = (0, 0, S["cam_z"]); cam.rotation_euler = (math.pi / 2, 0, 0)
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN")); sc.collection.objects.link(sun)
    sun.data.energy = S["sun_w"]; sun.data.angle = math.radians(0.53); sun.data.color = S["sun_rgb"]
    sun.rotation_euler = Vector(sun_vec()).to_track_quat("Z", "Y").to_euler()
    w = bpy.data.worlds.new("sky"); sc.world = w; w.use_nodes = True
    bg = w.node_tree.nodes["Background"]; bg.inputs[0].default_value = (*S["sky"], 1); bg.inputs[1].default_value = S["sky_w"]
    bpy.ops.mesh.primitive_plane_add(size=6000, location=(0, 2500, 0))
    g = bpy.context.active_object; g.name = "salt"
    m = bpy.data.materials.new("salt"); m.use_nodes = True
    p = m.node_tree.nodes["Principled BSDF"]; a = S["salt_albedo"]
    p.inputs["Base Color"].default_value = (a, a, a * 1.02, 1); p.inputs["Roughness"].default_value = 0.95
    g.data.materials.append(m)
    g.is_shadow_catcher = True
    root.location = (0, S["dist"], S["alt"])
    return bpy, sc, root, cam, g


def _linear(obj):
    act = obj.animation_data.action
    curves = list(getattr(act, "fcurves", []) or [])
    if not curves and hasattr(act, "layers"):
        for layer in act.layers:
            for strip in layer.strips:
                for cb in strip.channelbags:
                    curves += list(cb.fcurves)
    for fc in curves:
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"


def frames():
    """frames that need rendering and the box (aircraft and its shadow) each covers"""
    n = int(round(S["dur"] * C.FPS))
    s = sun_vec()
    out = []
    blur = v_ms() * S["shutter"] / C.FPS
    for i in range(n):
        c = pose(i / C.FPS)[0]
        P = np.array([[c[0] + dx, c[1] + dy, c[2] + dz] for dx in (-16 - blur, 16 + blur) for dy in (-27, 27)
                      for dz in (-18, 18)])
        Gs = P - (P[:, 2:3] / s[2]) * s                    # shadow on the salt
        u, v = project(np.vstack([P, Gs]))
        u0, u1, v0, v1 = u.min() - 24, u.max() + 24, v.min() - 24, v.max() + 24
        if u1 < 0 or u0 > PW:
            continue
        out.append((i, max(0, u0), min(PW, u1), max(0, v0), min(PH, v1)))
    return out


def prep():
    """exposure: the salt rendered (not as a shadow catcher) against the plate's salt near the horizon"""
    bpy, sc, root, cam, g = scene()
    g.is_shadow_catcher = False
    root.hide_render = True
    for o in root.children_recursive:
        o.hide_render = True
    sc.render.use_motion_blur = False
    sc.render.filepath = os.path.join(OUT, "calib.png"); bpy.ops.render.render(write_still=True)
    rgb, al = C.load_rgba16(os.path.join(OUT, "calib.png"))
    plate = C.to_lin(np.asarray(Image.open(os.path.join(OUT, "plate.png"))).astype(np.float32) / 255)
    hr = int(horizon_row())
    band = slice(hr + 12, hr + 90)
    r = rgb[band].reshape(-1, 3).mean(0); p = plate[band].reshape(-1, 3).mean(0)
    gain = (p / r).tolist()                             # per channel: the CG takes the film's colour cast
    json.dump(dict(render_lin=r.tolist(), plate_lin=p.tolist(), gain=gain), open(os.path.join(OUT, "calib.json"), "w"),
              indent=1)
    print("salt render", (C.to_srgb(r) * 255).round(1), "plate", (C.to_srgb(p) * 255).round(1), "gain", np.round(gain, 3))


def render(start=0, stop=None, step=1):
    bpy, sc, root, cam, g = scene()
    n = int(round(S["dur"] * C.FPS))
    root.rotation_mode = "XYZ"
    for f in range(-1, n + 2):                          # a key every frame (and one either side, for the blur)
        P, psi = pose(f / C.FPS)
        root.location = tuple(P)
        root.rotation_euler = (math.radians(S["bank"]), -math.radians(S["aoa"]), float(psi))
        root.keyframe_insert("location", frame=f); root.keyframe_insert("rotation_euler", frame=f)
    _linear(root)
    sc.frame_start, sc.frame_end = 0, n
    os.makedirs(os.path.join(OUT, "cg"), exist_ok=True)
    for i, u0, u1, v0, v1 in frames()[start:stop:step]:
        p = os.path.join(OUT, "cg", f"f{i:04d}.png")
        if os.path.exists(p):
            continue
        sc.render.use_border = True; sc.render.use_crop_to_border = False
        sc.render.border_min_x, sc.render.border_max_x = u0 / PW, u1 / PW
        sc.render.border_min_y, sc.render.border_max_y = 1 - v1 / PH, 1 - v0 / PH
        sc.frame_set(i)
        sc.render.filepath = p
        bpy.ops.render.render(write_still=True)
        print("rendered", i, flush=True)


def paint_gain():
    if S["paint_gain"]:
        return S["paint_gain"]
    p = os.path.join(C.WORK, "topdown_whiteman", "calib.json")
    return float(np.mean(json.load(open(p))["gain"])) if os.path.exists(p) else 2.1


def comp():
    plate_lin = C.to_lin(np.asarray(Image.open(os.path.join(OUT, "plate.png"))).astype(np.float32) / 255)
    gain = np.array(json.load(open(os.path.join(OUT, "calib.json")))["gain"], dtype=np.float32) * paint_gain()
    sky = plate_lin[int(horizon_row()) - 120:int(horizon_row()) - 60].reshape(-1, 3).mean(0)
    n = int(round(S["dur"] * C.FPS))
    D = None
    if S["debris"]:
        import debris
        D = debris.Debris(); D.calibrate()
        hr = int(horizon_row())
        salt = plate_lin[hr + 12:hr + 90].reshape(-1, 3).mean(0)
    vid = C.Video(os.path.join(OUT, S["name"] + "_silent.mp4"))
    for i in range(n):
        t = i / C.FPS
        pic = plate_lin.copy()
        if D is not None:                                # dust and chips beyond the track, behind the aircraft
            tn, tf = D.tau_layers(t)
            (cn, an), (cf, af) = D.chips_layer(t, salt)
            T = np.exp(-tf)[..., None]; pic = pic * T + D.light(tf, salt) * (1 - T)
            pic = cf + pic * (1 - af[..., None])
        cg = os.path.join(OUT, "cg", f"f{i:04d}.png")
        if os.path.exists(cg):
            rgb, al = C.load_rgba16(cg)
            rgb = rgb * gain
            rgb = rgb * (1 - S["haze"]) + sky * al * S["haze"]
            # the shadow catcher's shadow arrives as black with alpha; the aircraft as colour with alpha
            pic = rgb + pic * (1 - al)
        if D is not None:                                # and in front of it
            T = np.exp(-tn)[..., None]; pic = pic * T + D.light(tn, salt) * (1 - T)
            pic = cn + pic * (1 - an[..., None])
        out = (C.to_srgb(pic) * 255 + 0.5).astype(np.uint8)
        vid.write(C.frame_out(out, f"{S['mph']:.0f} mph", "B-2 Spirit, 1989"))
    vid.close()
    print("wrote", S["name"] + "_silent.mp4")


def mux():
    v = os.path.join(OUT, S["name"] + "_silent.mp4"); a = os.path.join(OUT, S["name"] + ".wav")
    o = os.path.join(OUT, S["name"] + ".mp4")
    os.system(f"ffmpeg -y -loglevel error -i {v} -i {a} -map 0:v -map 1:a -c:v copy -c:a aac -b:a 192k -shortest "
              f"-movflags +faststart {o}")
    print("wrote", o)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "plate": plate()
    elif cmd == "prep": prep()
    elif cmd == "render":
        a = [int(x) for x in sys.argv[2:]]
        render(*a)
    elif cmd == "frames": f = frames(); print(len(f), f[:2], f[-1:])
    elif cmd == "comp": comp()
    elif cmd == "mux": mux()
