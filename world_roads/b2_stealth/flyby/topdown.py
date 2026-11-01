"""Top-down flyby: the B-2 crosses Whiteman AFB, Missouri (its home base), seen from a camera hanging straight
down over the flight line, the ground the USDA's NAIP aerial photograph of the base (28 June 2022, 0.6 m, public
domain; `fetch_naip.py whiteman -93.5479 38.7303 1500`).

The camera is 900 m up with a 22.5 mm lens (77 degrees across), so the ground fills the picture at 0.75 m a pixel,
the photograph's own resolution. The B-2 passes 166 m below it (734 m up, 2,400 ft) at 300 mph, heading WNW (295),
straight away from the sun, so its shadow runs 390 m ahead of it along the same line: the shadow glides across the
base first, then the aircraft itself crosses in a little over two seconds, eight times faster across the picture
than its shadow because it is that much nearer the camera.

The sun is the photograph's: about 117 degrees (ESE) and 62 degrees up (the hangars' short shadows fall WNW), so the
render is lit as the ground was. The B-2's colour is matched to the real one parked on the ramp in the same photograph
(mean sRGB 141, 146, 163 over its planform): a calibration render at the same nadir view gives the gain.

    python topdown.py plate      # the ground, rotated so the heading runs left to right -> plate.png
    python topdown.py prep       # Blender: the shadow (the mesh projected along the sun onto the ground) and the
                                 # colour calibration render
    python topdown.py render     # Blender: the frames the aircraft is in (only the box it covers)
    python topdown.py comp       # shadow and aircraft over the plate, label -> topdown_whiteman_silent.mp4
    python flyby_sound.py topdown && python topdown.py mux
"""
import json, math, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

S = dict(
    name="topdown_whiteman", naip="whiteman",
    center_px=(1900, 1980),        # picture centre in the NAIP square (5000 px at 0.6 m, north up)
    heading=295.0,                 # compass degrees
    sun_az=117.0, sun_el=62.0,
    mpp=0.75, cam_h=900.0, alt=734.0, y_off=0.0,
    mph=300.0, t_pass=3.2, dur=6.2,
    shutter=0.3,                   # of a frame: 1/100 s at 30 fps
    shadow_k=0.30,                 # light left in the shadow (sky only)
    target_srgb=(141.0, 146.0, 163.0),
    sun_w=4.0, sky=(0.42, 0.56, 0.80), sky_w=0.9,
    cg_sat=0.8,                    # the render is a touch more lavender than the real one on the ramp
)
OUT = os.path.join(C.WORK, S["name"])
PW, PH = C.W, C.PIC_H


def v_ms():
    return S["mph"] * C.MPH


def f_px():
    return S["cam_h"] / S["mpp"]                    # the ground (z = 0) at mpp metres a pixel


def axes():
    """world +x = the heading, +y = 90 degrees to its left, z up; as (east, north) unit vectors"""
    th = math.radians(S["heading"])
    return (math.sin(th), math.cos(th)), (-math.cos(th), math.sin(th))


def sun_vec():
    X, Y = axes()
    az, el = math.radians(S["sun_az"]), math.radians(S["sun_el"])
    e, n = math.cos(el) * math.sin(az), math.cos(el) * math.cos(az)
    return np.array([e * X[0] + n * X[1], e * Y[0] + n * Y[1], math.sin(el)])


def plane_x(t):
    return v_ms() * (t - S["t_pass"])


def ground_to_px(X, Y):
    return PW / 2 + X / S["mpp"] - 0.5, PH / 2 - Y / S["mpp"] - 0.5


def plate():
    Image.MAX_IMAGE_PIXELS = None
    src = Image.open(os.path.join(C.WORK, "naip", S["naip"] + ".png")).convert("RGB")
    meta = json.load(open(os.path.join(C.WORK, "naip", S["naip"] + ".json")))
    mpp_src = meta["m_per_px"]
    (XE, XN), (YE, YN) = axes()
    ss = 2
    ow, oh, m = PW * ss, PH * ss, S["mpp"] / ss
    u0, v0 = ow / 2 - 0.5, oh / 2 - 0.5
    cx, cy = S["center_px"]
    a, b = m * XE / mpp_src, -m * YE / mpp_src
    c = cx + (-u0 * m * XE + v0 * m * YE) / mpp_src
    d, e = -m * XN / mpp_src, m * YN / mpp_src
    f = cy - (-u0 * m * XN + v0 * m * YN) / mpp_src
    im = src.transform((ow, oh), Image.AFFINE, (a, b, c, d, e, f), resample=Image.BICUBIC)
    im = im.resize((PW, PH), Image.LANCZOS)
    os.makedirs(OUT, exist_ok=True)
    im.save(os.path.join(OUT, "plate.png"))
    print("plate", im.size, "sun (world)", sun_vec().round(3), "shadow lead m",
          round(-S["alt"] * sun_vec()[0] / sun_vec()[2], 1))


# ---------------------------------------------------------------- Blender
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
    sc.render.use_motion_blur = True; sc.render.motion_blur_shutter = S["shutter"]
    sc.render.motion_blur_position = "CENTER"
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); sc.collection.objects.link(cam); sc.camera = cam
    cam.data.sensor_fit = "HORIZONTAL"; cam.data.sensor_width = 36.0
    cam.data.lens = f_px() * 36.0 / PW
    cam.data.clip_end = 5000
    cam.location = (0, 0, S["cam_h"]); cam.rotation_euler = (0, 0, 0)
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN")); sc.collection.objects.link(sun)
    sun.data.energy = S["sun_w"]; sun.data.angle = math.radians(0.53)
    sun.rotation_euler = Vector(sun_vec()).to_track_quat("Z", "Y").to_euler()
    w = bpy.data.worlds.new("sky"); sc.world = w; w.use_nodes = True
    bg = w.node_tree.nodes["Background"]; bg.inputs[0].default_value = (*S["sky"], 1); bg.inputs[1].default_value = S["sky_w"]
    root.location = (0, S["y_off"], S["alt"])
    return bpy, sc, root, cam


def _triangles(bpy, root):
    deps = bpy.context.evaluated_depsgraph_get()
    tris = []
    for o in bpy.data.objects:
        if o.type != "MESH" or o.hide_render:
            continue
        eo = o.evaluated_get(deps); me = eo.to_mesh(); me.calc_loop_triangles()
        M = np.array(o.matrix_world)
        V = np.array([v.co for v in me.vertices]); V = V @ M[:3, :3].T + M[:3, 3]
        for t in me.loop_triangles:
            tris.append(V[list(t.vertices)])
        eo.to_mesh_clear()
    return np.array(tris)


def prep():
    bpy, sc, root, cam = scene()
    bpy.context.view_layer.update()
    tris = _triangles(bpy, root)                        # world, aircraft at x = 0
    s = sun_vec()
    g = tris - (tris[..., 2:3] / s[2]) * s              # along the sun onto z = 0
    res = 0.25                                          # metres a pixel while rasterising
    lo = g[..., :2].reshape(-1, 2).min(0) - 20; hi = g[..., :2].reshape(-1, 2).max(0) + 20
    n = np.ceil((hi - lo) / res).astype(int)
    can = Image.new("L", (int(n[0]), int(n[1])), 0); d = ImageDraw.Draw(can)
    for t in g:
        d.polygon([((p[0] - lo[0]) / res, (hi[1] - p[1]) / res) for p in t], fill=255)
    k = int(round(S["mpp"] / res))
    small = can.resize((can.width // k, can.height // k), Image.BOX)
    pen = S["alt"] / s[2] * 0.00465                     # the sun's half-width: the shadow's soft edge, metres
    small = small.filter(ImageFilter.GaussianBlur(float(pen / S["mpp"] * 0.7)))
    mask = np.asarray(small).astype(np.float32) / 255
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, "shadow.npz"), mask=mask, x0=lo[0], y1=hi[1], mpp=k * res)
    print("shadow", mask.shape, "penumbra m", round(pen, 2), "ground box", lo.round(1), hi.round(1))
    # colour calibration: aircraft straight below, no blur
    sc.render.use_motion_blur = False
    root.location = (0, 0, S["alt"])
    sc.render.filepath = os.path.join(OUT, "calib.png"); bpy.ops.render.render(write_still=True)
    rgb, al = C.load_rgba16(os.path.join(OUT, "calib.png"))
    m = al[..., 0] > 0.98
    mean = (rgb[m] / al[m]).mean(0)
    tgt = C.to_lin(np.array(S["target_srgb"]) / 255)
    gain = tgt / mean
    json.dump(dict(render_lin=mean.tolist(), target_lin=tgt.tolist(), gain=gain.tolist()),
              open(os.path.join(OUT, "calib.json"), "w"), indent=1)
    print("calibration gain", gain.round(3), "render sRGB", (C.to_srgb(mean) * 255).round(1))


def frames():
    """frames the aircraft is in, and its box in the picture (pixels) for each"""
    n = int(round(S["dur"] * C.FPS))
    depth = S["cam_h"] - S["alt"]
    sc_ = f_px() / depth                                 # pixels a metre at the aircraft
    out = []
    for i in range(n):
        x = plane_x(i / C.FPS)
        cu = PW / 2 + x * sc_
        cv = PH / 2 - S["y_off"] * sc_
        hw = (14 + v_ms() * S["shutter"] / C.FPS) * sc_ + 30; hh = 30 * sc_ + 30
        if cu + hw < 0 or cu - hw > PW:
            continue
        out.append((i, max(0, cu - hw), min(PW, cu + hw), max(0, cv - hh), min(PH, cv + hh)))
    return out


def render(start=0, stop=None, step=1):
    bpy, sc, root, cam = scene()
    v = v_ms()
    n = int(round(S["dur"] * C.FPS))
    root.location = (plane_x(0), S["y_off"], S["alt"]); root.keyframe_insert("location", frame=0)
    root.location = (plane_x(n / C.FPS), S["y_off"], S["alt"]); root.keyframe_insert("location", frame=n)
    _linear(root)
    sc.frame_start, sc.frame_end = 0, n
    os.makedirs(os.path.join(OUT, "cg"), exist_ok=True)
    todo = frames()[start:stop:step]
    for i, u0, u1, v0, v1 in todo:
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


def _linear(obj):
    """make every key on the object's animation linear (Blender 4.4+ keeps them in layered actions)"""
    act = obj.animation_data.action
    curves = []
    if hasattr(act, "fcurves"):
        curves = list(act.fcurves)
    if not curves and hasattr(act, "layers"):
        for layer in act.layers:
            for strip in layer.strips:
                for cb in strip.channelbags:
                    curves += list(cb.fcurves)
    for fc in curves:
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"


def comp():
    plate_lin = C.to_lin(np.asarray(Image.open(os.path.join(OUT, "plate.png"))).astype(np.float32) / 255)
    sh = np.load(os.path.join(OUT, "shadow.npz"))
    mask, sx0, sy1 = sh["mask"], float(sh["x0"]), float(sh["y1"])
    gain = np.array(json.load(open(os.path.join(OUT, "calib.json")))["gain"], dtype=np.float32)
    n = int(round(S["dur"] * C.FPS))
    vid = C.Video(os.path.join(OUT, S["name"] + "_silent.mp4"))
    k = S["shadow_k"]
    for i in range(n):
        t = i / C.FPS
        x = plane_x(t)
        pic = plate_lin.copy()
        # shadow sprite: its top-left ground corner (sx0 + x, sy1) -> picture pixels, sub-pixel by resampling
        u, v = ground_to_px(sx0 + x, sy1)
        iu, iv = int(math.floor(u)), int(math.floor(v)); fu = u - iu
        m = mask if fu < 1e-3 else (1 - fu) * mask + fu * np.pad(mask, ((0, 0), (1, 0)))[:, :-1]
        h_, w_ = m.shape
        a0, a1 = max(0, iu), min(PW, iu + w_); b0, b1 = max(0, iv), min(PH, iv + h_)
        if a1 > a0 and b1 > b0:
            mm = m[b0 - iv:b1 - iv, a0 - iu:a1 - iu][..., None]
            pic[b0:b1, a0:a1] *= 1 - (1 - k) * mm
        cg = os.path.join(OUT, "cg", f"f{i:04d}.png")
        if os.path.exists(cg):
            rgb, al = C.load_rgba16(cg)
            rgb = rgb * gain
            lum = (rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32))[..., None]
            rgb = lum + (rgb - lum) * S["cg_sat"]
            pic = rgb + pic * (1 - al)
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
        render(*(a + [0, None, 1][len(a):]) if a else ())
    elif cmd == "frames": print(len(frames()), frames()[:3], frames()[-1:])
    elif cmd == "comp": comp()
    elif cmd == "mux": mux()
