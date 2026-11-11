"""
Kitchen mixing tools for the round-bowl simulation: their shapes, the collider
points the physics uses, and how each one is moved.  Pure numpy, so both
bowl_sim.py (physics) and bowl_scene.py (Blender) build from the same data.

Everything hangs from one rig: a rotor turning about the bowl axis carries the
tool at some radius.  That covers every case in the kitchen:

  rigid on the rotor        a hand tool stirred in circles, face leading
  rigid, radius breathing   stirring in a spiral so the whole bowl is covered
  spinning on the rotor     a stand mixer's planetary attachment
  two spinning parts        hand-mixer beaters carried round the bowl
  radius 0, spinning        a tool twirled on the spot in the middle

The bowl stays still.  Spinning the bowl under a fixed tool is the same motion
seen from the bowl, except that it also flings the powder outward.

Units: millimetres.  The bowl is a half-scale model of a 22 cm kitchen bowl.
"""
import math

import numpy as np

BOWL = dict(
    R=55.0,        # inner radius
    rf=10.0,       # fillet radius between floor and wall
    wall_h=36.0,   # inner wall height
    bed=14.0,      # target powder depth
    glass=2.2,     # glass thickness (render only)
    lift=42.0,     # how far the tool is raised clear of the bowl
    clear=0.7,     # tool bottom above the floor when down
)


# ---------------------------------------------------------------------------
# shapes: flat outlines (u across, w up from the tool's bottom edge) and wires
# ---------------------------------------------------------------------------
def _arc(cx, cz, r, a0, a1, n=8):
    return [(cx + r * math.cos(math.radians(a0 + (a1 - a0) * k / n)),
             cz + r * math.sin(math.radians(a0 + (a1 - a0) * k / n))) for k in range(n + 1)]


def gate_outline(Rp, H, bar=4.0, rib=3.0, spine=2.5, rc=7.0):
    """Flat beater: a windowed frame, two wings."""
    rci = max(rc - bar, 1.0)
    outer = [(-Rp, H)] + _arc(-Rp + rc, rc, rc, 180, 270) + _arc(Rp - rc, rc, rc, 270, 360) + [(Rp, H)]
    holes = []
    rib_c = 0.5 * (spine + (Rp - bar))
    for sgn in (1, -1):
        a0, a1 = spine, rib_c - rib / 2
        holes.append([(sgn * a0, bar), (sgn * a1, bar), (sgn * a1, H - bar), (sgn * a0, H - bar)])
        b0, b1 = rib_c + rib / 2, Rp - bar
        h = [(b0, bar)] + _arc(b1 - rci, bar + rci, rci, 270, 360, 5) + [(b1, H - bar), (b0, H - bar)]
        holes.append([(sgn * x, zz) for x, zz in h])
    return outer, holes


def fork_outline(n=4, pitch=6.0, tine_w=2.0, tine_len=26.0, bridge=7.0):
    half = (n - 1) * pitch / 2 + tine_w / 2
    H = tine_len + bridge
    outer = [(-half, H)]
    for k in range(n):
        c = -half + tine_w / 2 + k * pitch
        outer += [(c - tine_w / 2, tine_len), (c - tine_w / 2, 0.0), (c + tine_w / 2, 0.0), (c + tine_w / 2, tine_len)]
    outer[1] = (-half, tine_len)
    return outer + [(half, H)], []


def fork_visual(top, n=4, pitch=6.0, tine_w=2.0, tine_len=26.0, bridge=7.0):
    """What the fork looks like (render only): the same tines and bridge the
    physics uses, with rounded tips, shoulders, a neck and a flat handle up to `top`."""
    half = (n - 1) * pitch / 2 + tine_w / 2
    H = tine_len + bridge
    neck, hw_top, sh = 2.4, 4.6, 14.0
    right = [(neck + (half - neck) * (1 + math.cos(math.pi * t / 8)) / 2, H + sh * t / 8) for t in range(9)]
    right += [(neck, H + sh + 8.0), (hw_top, top)]
    pts = [(-x, z) for x, z in reversed(right)] + [(-half, tine_len)]
    c = 0.7
    for k in range(n):
        x0 = -half + k * pitch
        x1 = x0 + tine_w
        pts += [(x0, tine_len), (x0, c), (x0 + c, 0.0), (x1 - c, 0.0), (x1, c), (x1, tine_len)]
    return pts + right


def spoon_outline(a=11.0, b=15.0, n=48):
    return [(a * math.cos(2 * math.pi * k / n), b + b * math.sin(2 * math.pi * k / n)) for k in range(n)], []


def spoon_dish(u, w, a=11.0, b=15.0, depth=3.0):
    """Out-of-plane offset of the spoon's bowl."""
    q = 1.0 - (u / a) ** 2 - ((w - b) / b) ** 2
    return -depth * np.clip(q, 0.0, None)


def cage_wires(Rb=13.0, H=30.0, rc=5.0, wings=4):
    """Hand-mixer beater: wire loops round a centre shaft."""
    prof = [(0.8, 0.0), (Rb - rc, 0.0)] + _arc(Rb - rc, rc, rc, 270, 360, 6) + \
           _arc(Rb - rc, H - rc, rc, 0, 90, 6) + [(1.5, H)]
    out = []
    for k in range(wings):
        al = 2 * math.pi * k / wings
        out.append([(r * math.cos(al), r * math.sin(al), z) for r, z in prof])
    return out


def whisk_wires(Rw=14.0, H=48.0, loops=5, n=40):
    """Balloon whisk: loops through the tip, widest a third of the way up."""
    out = []
    for j in range(loops):
        al = math.pi * j / loops
        pts = []
        for i in range(-n, n + 1):
            t = abs(i) / n
            r = math.copysign(Rw * math.sin(math.pi * t ** 0.55), i)
            pts.append((r * math.cos(al), r * math.sin(al), H * t))
        out.append(pts)
    return out


def inside_outline(u, w, outer, holes):
    """Even-odd point-in-polygon for arrays u, w."""
    def inside(poly):
        px = np.array([p[0] for p in poly]); pz = np.array([p[1] for p in poly])
        res = np.zeros(u.shape, bool)
        j = len(poly) - 1
        for i in range(len(poly)):
            cond = (pz[i] > w) != (pz[j] > w)
            xint = (px[j] - px[i]) * (w - pz[i]) / (pz[j] - pz[i] + 1e-12) + px[i]
            res ^= cond & (u < xint)
            j = i
        return res
    m = inside(outer)
    for h in holes:
        m &= ~inside(h)
    return m


def shape_geometry(shape, p):
    """Render description of a shape: flat outline or 3-D wires."""
    if shape == "gate":
        outer, holes = gate_outline(p["Rp"], p["H"], rc=p.get("rc", 7.0))
        return dict(kind="outline", outer=outer, holes=holes, H=p["H"])
    if shape == "fork":
        outer, holes = fork_outline()
        return dict(kind="outline", outer=outer, holes=holes, H=max(z for _, z in outer))
    if shape == "spoon":
        outer, holes = spoon_outline(p["a"], p["b"])
        return dict(kind="spoon", outer=outer, holes=holes, H=2 * p["b"], a=p["a"], b=p["b"], depth=p.get("depth", 3.0))
    if shape == "cage":
        return dict(kind="wires", wires=cage_wires(p["Rb"], p["H"]), H=p["H"])
    if shape == "whisk":
        return dict(kind="wires", wires=whisk_wires(p["Rw"], p["H"], p["loops"]), H=p["H"])
    raise ValueError(shape)


def shape_points(shape, p, d):
    """Collider sphere centres (u, v, w) for a shape; spheres have radius d/2."""
    g = shape_geometry(shape, p)
    s = 0.6 * d
    if g["kind"] in ("outline", "spoon"):
        us = [q[0] for q in g["outer"]]
        ws = [q[1] for q in g["outer"]]
        uu = np.arange(min(us) + d / 2, max(us) - d / 2 + 1e-6, s)
        uu = uu - uu.mean() + 0.5 * (min(us) + max(us))
        ww = np.arange(min(ws) + d / 2, max(ws) - d / 2 + 1e-6, s)
        U, W = np.meshgrid(uu, ww, indexing="ij")
        U, W = U.ravel(), W.ravel()
        m = inside_outline(U, W, g["outer"], g["holes"])
        U, W = U[m], W[m]
        V = spoon_dish(U, W, g["a"], g["b"], g["depth"]) if g["kind"] == "spoon" else np.zeros_like(U)
        return np.stack([U, V, W], 1)
    pts = []
    for wire in g["wires"]:
        w = np.array(wire)
        seg = np.sqrt(((w[1:] - w[:-1]) ** 2).sum(1))
        L = np.concatenate([[0], np.cumsum(seg)])
        t = np.arange(0, L[-1], s)
        pts.append(np.stack([np.interp(t, L, w[:, k]) for k in range(3)], 1))
    P = np.concatenate(pts)
    P[:, 2] = np.maximum(P[:, 2], d / 2)          # wire sits on its radius
    return P


# ---------------------------------------------------------------------------
# the contestants
# ---------------------------------------------------------------------------
# orbit / spin are in rev/s at speed 1.  spin is relative to the rotor; a
# negative spin turns against the orbit, as a stand mixer's planetary does.
# spiral: (innermost radius, orbits per in-and-out cycle) for stirring that
# works its way in to the middle and back out.
TOOLS = {
    "twin_beaters": dict(
        label="Hand mixer (twin beaters)", unit="laps",
        parts=[dict(shape="cage", p=dict(Rb=13.0, H=30.0), du="twin-", spin=5.0, phase=0.0),
               dict(shape="cage", p=dict(Rb=13.0, H=30.0), du="twin+", spin=-5.0, phase=math.pi / 4)],
        orbit=0.4, e="fit", spiral=None),
    "paddle_large": dict(
        label="Stand mixer, large paddle", unit="orbits",
        parts=[dict(shape="gate", p=dict(Rp=34.0, H=30.0, rc=10.0), du=0.0, spin=-2.4, phase=0.0)],
        orbit=0.8, e="fit", spiral=None),
    "paddle_small": dict(
        label="Stand mixer, small paddle", unit="orbits",
        parts=[dict(shape="gate", p=dict(Rp=15.0, H=30.0, rc=8.0), du=0.0, spin=-2.4, phase=0.0)],
        orbit=0.8, e="fit", spiral=None),
    "wire_whip": dict(
        label="Stand mixer, wire whip", unit="orbits",
        parts=[dict(shape="whisk", p=dict(Rw=24.0, H=52.0, loops=6), du=0.0, spin=-2.4, phase=0.0)],
        orbit=0.8, e="fit", spiral=None),
    "whisk": dict(
        label="Whisk, by hand", unit="strokes",
        parts=[dict(shape="whisk", p=dict(Rw=14.0, H=48.0, loops=5), du=0.0, spin=0.0, phase=0.0)],
        orbit=1.5, e="fit", spiral=(8.0, 3.3)),
    "fork": dict(
        label="Fork", unit="strokes",
        parts=[dict(shape="fork", p={}, du=0.0, spin=0.0, phase=0.0)],
        orbit=1.5, e="fit", spiral=(8.0, 3.3)),
    "spoon": dict(
        label="Spoon", unit="strokes",
        parts=[dict(shape="spoon", p=dict(a=11.0, b=15.0), du=0.0, spin=0.0, phase=0.0)],
        orbit=1.5, e="fit", spiral=(8.0, 3.3)),
    "spoon_centre": dict(
        label="Spoon twirled in the middle", unit="turns",
        parts=[dict(shape="spoon", p=dict(a=11.0, b=15.0), du=0.0, spin=1.5, phase=0.0)],
        orbit=0.0, e=0.0, spiral=None),
    # variations used by the science runs, not contestants
    "spoon_circle": dict(
        label="Spoon, same circle every time", unit="strokes",
        parts=[dict(shape="spoon", p=dict(a=11.0, b=15.0), du=0.0, spin=0.0, phase=0.0)],
        orbit=1.5, e="fit", spiral=None),
}
# One identity colour per contestant, used for its mixer housing, name, bar and
# chart line.  Every pair that meets in the knockout passes the colour-vision
# checks (dataviz validator, all-pairs mode).
TOOL_COLOR = {
    "spoon": "#eb6834", "spoon_circle": "#eb6834", "spoon_centre": "#1baf7a", "whisk": "#2a78d6",
    "fork": "#008300", "twin_beaters": "#e34948", "paddle_large": "#6250d6", "paddle_small": "#eda100",
    "wire_whip": "#e87ba4",
}
# The mixers themselves are painted in colours KitchenAid actually sells (names from
# KitchenAid and retailer listings; the RGB values are matched by eye, there is no
# published code), one per contestant, in the same hue family as its identity colour.
KITCHENAID_COLOR = {
    "spoon": ("Scorched Orange", "#D2562B"), "spoon_circle": ("Scorched Orange", "#D2562B"),
    "spoon_centre": ("Aqua Sky", "#74C7CB"), "whisk": ("Blue Velvet", "#27508F"),
    "fork": ("Pistachio", "#B4CC9A"), "twin_beaters": ("Empire Red", "#B3121C"),
    "paddle_large": ("Lavender Cream", "#B5A6D8"), "paddle_small": ("Butter", "#F0DC8C"),
    "wire_whip": ("Dried Rose", "#C88C8C"),
}
CONTESTANTS = ["twin_beaters", "paddle_large", "paddle_small", "wire_whip", "whisk", "fork", "spoon", "spoon_centre"]


def bowl_radius_at(z, d, bowl=BOWL):
    """Largest radius a collider centre may have at height z."""
    m = d / 2 + 0.6
    R, rf = bowl["R"], bowl["rf"]
    z = np.asarray(z, float)
    inner = R - rf + np.sqrt(np.clip((rf - m) ** 2 - (rf - z) ** 2, 0, None))
    return np.where(z >= rf, R - m, inner)


def twin_separation(pts, d):
    """Closest the two counter-rotating beaters can sit without touching."""
    Rb = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2).max()
    A0 = pts
    best = 2 * Rb + d
    for sep in np.arange(2 * Rb + d, 1.5 * Rb, -0.5):     # real beaters overlap by up to half a radius
        ok = True
        for a in np.linspace(0, math.pi / 2, 60, endpoint=False):
            ca, sa = math.cos(a), math.sin(a)
            A = np.stack([A0[:, 0] * ca - A0[:, 1] * sa, A0[:, 0] * sa + A0[:, 1] * ca, A0[:, 2]], 1)
            b = -a + math.pi / 4
            cb, sb = math.cos(b), math.sin(b)
            B = np.stack([sep + A0[:, 0] * cb - A0[:, 1] * sb, A0[:, 0] * sb + A0[:, 1] * cb, A0[:, 2]], 1)
            dd = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1).min()
            if dd < (1.35 * d) ** 2:
                ok = False
                break
        if not ok:
            break
        best = sep
    return float(best)


def build_tool(name, d, bowl=BOWL):
    """Resolve a tool: collider points per part, mount offsets, fitted orbit radius."""
    spec = TOOLS[name]
    parts = []
    for p in spec["parts"]:
        pts = shape_points(p["shape"], p["p"], d)
        parts.append(dict(shape=p["shape"], p=p["p"], pts=pts, spin=p["spin"], phase=p["phase"], du=p["du"],
                          geom=shape_geometry(p["shape"], p["p"])))
    if isinstance(parts[0]["du"], str):                       # twin beaters: fit the gap
        sep = twin_separation(parts[0]["pts"], d)
        parts[0]["du"], parts[1]["du"] = -sep / 2, sep / 2
    # largest rotor radius at which every collider stays inside the bowl
    e_max = 1e9
    for p in parts:
        z = bowl["clear"] + p["pts"][:, 2]
        rin = bowl_radius_at(z, d, bowl)
        if p["spin"] != 0.0:
            reach = np.sqrt(p["pts"][:, 0] ** 2 + p["pts"][:, 1] ** 2)
            e_max = min(e_max, float((rin - reach - p["du"]).min()))
        else:
            e_max = min(e_max, float((np.sqrt(np.clip(rin ** 2 - p["pts"][:, 1] ** 2, 0, None))
                                      - p["pts"][:, 0] - p["du"]).min()))
    e = e_max if spec["e"] == "fit" else float(spec["e"])
    return dict(name=name, label=spec["label"], unit=spec["unit"], parts=parts, orbit=spec["orbit"],
                e=e, spiral=spec["spiral"])


class Motion:
    """Rotor angle, mount radius, part angles and plunge as functions of time."""

    def __init__(self, tool, seconds, speed=1.0, t_ramp=0.5, dt=1e-3):
        self.tool, self.seconds = tool, seconds
        self.t = np.arange(0.0, seconds + 2.0, dt)
        up = np.clip(self.t / t_ramp, 0, 1)
        dn = np.clip((self.t - (seconds - t_ramp)) / t_ramp, 0, 1)
        sm = lambda u: u * u * (3 - 2 * u)
        env = sm(up) * (1 - sm(dn))
        self.plunge = env                                         # 1 = tool down in the powder
        self.turns = np.cumsum(env) * dt * speed                  # seconds of full-speed running x speed
        self.speed = speed

    def at(self, t):
        tool = self.tool
        E = np.interp(t, self.t, self.turns)
        pl = np.interp(t, self.t, self.plunge)
        phi = 2 * math.pi * tool["orbit"] * E
        e = tool["e"]
        if tool["spiral"]:
            e_min, n_orb = tool["spiral"]
            e = e_min + (e - e_min) * (0.5 + 0.5 * math.cos(phi / n_orb))
        angles = [phi + 2 * math.pi * p["spin"] * E + p["phase"] for p in tool["parts"]]
        return phi, e, angles, pl

    def count(self, t):
        """How many strokes / orbits / turns the tool has made by time t."""
        E = np.interp(t, self.t, self.turns)
        rate = self.tool["orbit"] if self.tool["orbit"] else abs(self.tool["parts"][0]["spin"])
        return rate * E


def mix_curve(z):
    """% mixed curve of a run, on a scale where the bowl as loaded is 0.

    The raw index compares against perfectly separated powders, but thin rings
    or wedge tips already share sample cells before any stirring (rings start
    at 26 %).  Runs saved before this was built in are rescaled here."""
    m = np.asarray(z["mix"], float)
    if "mix_raw" in getattr(z, "files", z):
        return m
    return (m - m[0]) / (1.0 - m[0])
