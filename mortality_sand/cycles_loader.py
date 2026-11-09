# ---- sand loader: runs inside Blender (numpy only). blender_cycles.py pastes grains.py above this and
# stores the whole as the text block "sand_loader.py" in the .blend, registered to run when the file opens.
# Every frame it fills two point clouds from the cache folder next to the .blend: "Sand" (the resting
# grains a camera can see) and "Lifted" (the grains rising out of the frame, glowing in their cause's colour).
import os
import bpy
import numpy as np

STREAK = 4.0          # velocity scale for the rising grains' streaks: with the half-frame shutter, 2 frames of travel
GLOW = 3.0            # emission strength of a lifted grain at full glow (with the compositor's glare)
NEAR = 1.5            # grains nearer the camera than this (Blender units) are dropped, as the painter does


def _srgb_to_linear(c):
    c = np.asarray(c, np.float32)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


class SandCache:
    def __init__(self, root):
        self.root = root
        z = np.load(os.path.join(root, "static.npz"))
        self.st = {k: z[k] for k in z.files}
        st = self.st
        self.n = int(st["nframes"])
        self.fps = float(st["fps"])
        self.chunk = int(st["chunk"])
        self.nb = st["nb"].astype(np.int64)
        self.cell_xy = st["cell_xy"].astype(np.float64)
        self.sand_lin = _srgb_to_linear(st["sand_srgb"])
        self.cause_lin = _srgb_to_linear(st["cause_srgb"])
        z = np.load(os.path.join(root, "lifted.npz"))
        self.L = {k: z[k] for k in z.files}
        self.L["ph"] = self.L["ph"].reshape(-1, 3)
        self.L_t0 = self.L["t0"].astype(np.float64)
        self._chunk_id, self._chunk = -1, None
        self._h = {}                     # film frame -> pile heights (the last few)

    def _load_chunk(self, cid):
        if cid != self._chunk_id:
            z = np.load(os.path.join(self.root, "sand", f"{cid:03d}.npz"))
            self._chunk = {k: z[k] for k in z.files}
            self._chunk_id = cid
        return self._chunk

    def heights(self, f):
        """Pile height in every cell at film frame f (0-based)."""
        f = min(max(f, 0), self.n - 1)
        if f in self._h:
            return self._h[f]
        cid, j = divmod(f, self.chunk)
        c = self._load_chunk(cid)
        start = f - 1 if (f - 1) in self._h and (f - 1) // self.chunk == cid else None
        if start is not None:
            h = self._h[start].copy()
            k0 = j
        else:
            h = c["base"].astype(np.int64)
            k0 = 1
        off, idx, val = c["off"], c["idx"], c["val"]
        for k in range(k0, j + 1):                     # changes from frame k-1 to k of this chunk
            a, b = off[k - 1], off[k]
            h[idx[a:b]] = val[a:b]
        self._h[f] = h
        for old in [g for g in self._h if g < f - 4 or g > f]:
            del self._h[old]
        return h

    # ------------------------------------------------------------------ the resting sand
    def sand(self, f):
        st = self.st
        h = self.heights(f)
        cell, lev, hcol = visible_grains(h, self.nb)
        x, y, z, r = grain_place(cell, lev, self.cell_xy, float(st["gs"]), float(st["grain_dz"]))
        # new grains swell into place over a few frames, as in the painted film
        h1, h2, h3 = self.heights(f - 1), self.heights(f - 2), self.heights(f - 3)
        scale = np.ones(len(cell), np.float32)
        if f > 0:
            new0 = lev >= h1[cell]
            new1 = ~new0 & (lev >= h2[cell])
            new2 = ~new0 & ~new1 & (lev >= h3[cell])
            scale[new0], scale[new1], scale[new2] = 0.3, 0.52, 0.74
        shade = st["tint"][cell] * (0.88 + 0.24 * hash01(cell, lev, 5))
        col = np.clip(st["sand_srgb"][None, :] * shade[:, None], 0, 1)
        return x, y, z - r * (1 - scale), r * scale, _srgb_to_linear(col)

    # ------------------------------------------------------------------ the lifted grains
    def lifted(self, f):
        st = self.st
        s = f / self.fps
        life = float(st["life"])
        a = np.searchsorted(self.L_t0, s - life, "right")
        b = np.searchsorted(self.L_t0, s, "right")
        idx = np.arange(a, b)
        if not len(idx):
            return None
        u = s - self.L_t0[idx]
        x, y, z = rise_pos(self.L, idx, u)
        dt = 0.5 / self.fps
        xb, yb, zb = rise_pos(self.L, idx, np.maximum(u - dt, 0.0))
        xa, ya, za = rise_pos(self.L, idx, u + dt)
        P = np.stack([x / 100.0 - 9.6, 5.4 - y / 100.0, z / 100.0], 1)
        V = np.stack([(xa - xb) / 100.0, -(ya - yb) / 100.0, (za - zb) / 100.0], 1) / (2 * dt) * STREAK
        # drop what is behind the camera, too close to it, or far out of the frame
        rel = P - st["cam_loc"][None, :]
        d = rel @ st["cam_f"]
        fx = st["focal_px"] * (rel @ st["cam_r"]) / np.maximum(d, 1e-3)
        fy = st["focal_px"] * (rel @ st["cam_u"]) / np.maximum(d, 1e-3)
        ok = (d > NEAR) & (np.abs(fx) < st["width"] / 2 + 80) & (np.abs(fy) < st["height"] / 2 + 80)
        idx, u, P, V = idx[ok], u[ok], P[ok], V[ok]
        g = np.clip(u / float(st["tg"]), 0, 1)
        bright = (1.0 + 0.2 * g) * np.clip(1.1 - (u - float(st["tg"])) / 8.0, 0.7, 1.1)
        cause = self.L["cause"][idx].astype(np.int64)
        return dict(P=P, V=V, r=self.L["r"][idx] / 100.0, glow=g.astype(np.float32),
                    ecol=self.cause_lin[cause], estr=(GLOW * g * bright).astype(np.float32))


def _fill(ob, P, attrs):
    """Replace a point object's vertices and attributes."""
    me = ob.data
    me.clear_geometry()
    n = len(P)
    if n:
        me.vertices.add(n)
        me.vertices.foreach_set("co", np.ascontiguousarray(P, np.float32).ravel())
    for name, (kind, v) in attrs.items():
        a = me.attributes.get(name)
        if a is None:
            a = me.attributes.new(name, kind, "POINT")
        if not n:
            continue
        if kind == "FLOAT":
            a.data.foreach_set("value", np.ascontiguousarray(v, np.float32).ravel())
        elif kind == "FLOAT_VECTOR":
            a.data.foreach_set("vector", np.ascontiguousarray(v, np.float32).ravel())
        else:
            rgba = np.concatenate([np.asarray(v, np.float32), np.ones((n, 1), np.float32)], 1)
            a.data.foreach_set("color", rgba.ravel())
    me.update()


_CACHE = {}
_LAST = {}


def cache():
    root = bpy.path.abspath("//cache")
    if root not in _CACHE:
        _CACHE.clear()
        _CACHE[root] = SandCache(root)
    return _CACHE[root]


def update_sand(scene, *args):
    sand_ob, lift_ob = bpy.data.objects.get("Sand"), bpy.data.objects.get("Lifted")
    if sand_ob is None or lift_ob is None or not bpy.data.filepath:
        return
    if scene.name != sand_ob.users_scene[0].name:
        return
    # motion blur evaluates the scene a little before and after the frame: keep the frame's own geometry
    # (the lifted grains carry a velocity for their streaks)
    c = cache()
    f = int(round(scene.frame_current + scene.frame_subframe)) - scene.frame_start
    f = min(max(f, 0), c.n - 1)
    if _LAST.get("f") == f and _LAST.get("file") == bpy.data.filepath:
        return
    x, y, z, r, col = c.sand(f)
    _fill(sand_ob, np.stack([x / 100.0 - 9.6, 5.4 - y / 100.0, z / 100.0], 1),
          {"rad": ("FLOAT", r / 100.0), "col": ("FLOAT_COLOR", col)})
    L = c.lifted(f)
    if L is None:
        _fill(lift_ob, np.zeros((0, 3)), {"rad": ("FLOAT", None), "glow": ("FLOAT", None), "estr": ("FLOAT", None),
                                          "ecol": ("FLOAT_COLOR", None), "velocity": ("FLOAT_VECTOR", None)})
    else:
        _fill(lift_ob, L["P"], {"rad": ("FLOAT", L["r"]), "glow": ("FLOAT", L["glow"]), "estr": ("FLOAT", L["estr"]),
                                "ecol": ("FLOAT_COLOR", L["ecol"]), "velocity": ("FLOAT_VECTOR", L["V"])})
    _LAST.update(f=f, file=bpy.data.filepath)


def register():
    for hl in (bpy.app.handlers.frame_change_pre, bpy.app.handlers.load_post):
        for fn in [fn for fn in hl if getattr(fn, "__name__", "") in ("update_sand", "_on_load")]:
            hl.remove(fn)
    bpy.app.handlers.frame_change_pre.append(update_sand)
    _LAST.clear()
    if bpy.data.filepath:
        c = cache()
        print(f"sand_loader: {c.n} frames, {len(c.L_t0)} lifted grains, from {c.root}", flush=True)
    update_sand(bpy.context.scene)


register()
