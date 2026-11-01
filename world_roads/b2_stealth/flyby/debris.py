"""The salt the B-2 tears up flying flat out at 50 ft over the Bonneville Salt Flats (`side.py`).

Why there is any: at 630 mph and 15 m up (less than a third of its span) the aircraft rides in ground effect. The
~150 tonnes of lift it carries press on the air between it and the salt (around a kilopascal at the ground as it goes
over), which squirts out sideways and backward at tens of metres a second; its wingtip vortices hit the ground at
once and roll outward along it; and its four jets' wash comes down behind it. Loose salt and dust lift off the crust,
and chips of crust are thrown.

How it is drawn (not a fluid simulation, unlike the speed ladder's dust; a parcel model shaped to that physics):

* fine dust: parcels launched from under the track when the aircraft passes over each point (plus a few hundredths
  of a second), most from under the centre body and jets, the rest from under the wingtips; thrown outward, up and a
  little forward with the air it drags along, then relaxing to a light breeze within a second or so, still rising
  while it mixes, spreading as it goes (radius ~ sqrt(age) and age), heavier grains settling;
* crust chips (20-80 cm plates, and grit): ballistic, with air drag, thrown up at up to 18 m/s (at most ~15 m
  up), falling back and stopping;
* each frame the parcels are projected through the plate's camera and splatted as soft blobs into a map of optical
  depth (the blob's column density, ~ m / r^2, at its size on the screen), in two layers: in front of the aircraft's
  track and behind it, so the aircraft sits between them;
* lit like the plate's own salt in the sun (its colour measured near the horizon), darker and bluer where the dust
  is thick (the sun is low and behind the camera, so the sunward faces are the ones we see; the interior is lit by the
  sky only).
"""
import math, os, sys
import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import side as M

PW, PH = C.W, C.PIC_H
RNG_SEED = 11
N_PUFF, N_SUB, N_CHIP, N_GRIT = 2600, 24, 2600, 7000
X_RANGE = (-260.0, 260.0)          # along the track: where dust matters to the picture
WIND = np.array([-2.5, 1.2, 0.0])  # light breeze, m/s
TAU_TARGET = 2.4                   # optical depth of the thickest dust a second after the pass (99.5th percentile)


class Debris:
    def __init__(self):
        r = np.random.default_rng(RNG_SEED)
        v = M.v_ms()
        y0 = M.S["dist"]
        # ---- fine dust: puffs (each a clump of sub-parcels, so the cloud billows instead of fogging)
        n = N_PUFF
        xe = r.uniform(*X_RANGE, n)
        under = r.random(n) < 0.6
        d = np.where(under, r.normal(0, 9, n), np.sign(r.random(n) - 0.5) * (21 + r.normal(0, 4, n)))
        self.te = M.S["t_pass"] + xe / v + 0.012 + r.exponential(0.03, n)   # right behind the trailing edge
        self.p0 = np.stack([xe + r.normal(0, 2, n), y0 + d, np.full(n, 0.3)], 1)
        out = np.sign(d) * np.where(np.abs(d) > 6, 1.0, np.abs(d) / 6)
        tall = r.random(n) < 0.3                         # the rooster tail and the vortices' columns
        self.v0 = np.stack([r.normal(12, 14, n),
                            out * (7 + 22 * r.random(n)) + r.normal(0, 3, n),
                            np.where(tall, 10 + 10 * r.random(n), 2 + 8 * r.random(n))], 1)
        self.tau_air = 0.5 + 0.9 * r.random(n)
        self.rise = np.where(tall, 0.5 + 0.7 * r.random(n), 0.2 + 0.5 * r.random(n))
        self.settle = np.where(r.random(n) < 0.3, 0.4 + 0.8 * r.random(n), 0.03)
        self.R0 = 0.8 + 1.0 * r.random(n)
        self.m = r.lognormal(0, 0.85, n) * np.where(under, 1.3, 1.0)
        self.walk = r.normal(0, 1.3, (n, 3)); self.walk[:, 2] *= 0.5
        self.spin = r.normal(0, 0.8, n)
        # sub-parcels: fixed offsets in units of the puff's radius, each with its own share of the mass
        self.off = r.normal(0, 0.55, (n, N_SUB, 3)); self.off[..., 2] = np.abs(self.off[..., 2]) * 0.8
        self.subm = r.lognormal(0, 0.6, (n, N_SUB)); self.subm /= self.subm.sum(1, keepdims=True)
        # ---- crust chips (visible pieces) and grit (sparkle)
        def chips(k, size_lo, size_hi):
            xe_ = r.uniform(-200, 200, k)
            d_ = r.normal(0, 14, k)
            te_ = M.S["t_pass"] + xe_ / v + 0.02 + r.exponential(0.05, k)
            p_ = np.stack([xe_, y0 + d_, np.zeros(k)], 1)
            v_ = np.stack([r.normal(22, 14, k), np.sign(d_) * (3 + 16 * r.random(k)), 3 + 15 * r.random(k)], 1)
            size = r.uniform(size_lo, size_hi, k)
            return dict(te=te_, p0=p_, v0=v_, size=size, kdrag=0.06 / size, shade=r.uniform(0.35, 1.15, k))
        self.chip = chips(N_CHIP, 0.20, 0.80)
        self.grit = chips(N_GRIT, 0.02, 0.08)
        self.norm = 1.0
        self.salt = None

    # ------------------------------------------------------------------ motion
    def fine_at(self, t):
        """sub-parcel positions, radii, masses, alive (flattened puffs x sub-parcels)"""
        a = t - self.te
        alive = a > 0
        a = np.where(alive, a, 0.0)
        e = (1 - np.exp(-a / self.tau_air))[:, None]
        c = self.p0 + WIND * a[:, None] + (self.v0 - WIND) * self.tau_air[:, None] * e + self.walk * np.sqrt(a)[:, None]
        c[:, 2] += (self.rise - self.settle) * a * np.minimum(a / 0.6, 1)
        R = self.R0 + 1.6 * np.sqrt(a) + 0.45 * a                    # the puff's radius
        ang = self.spin * a                                           # billowing: the clump turns over
        ca, sa = np.cos(ang)[:, None], np.sin(ang)[:, None]
        ox, oz = self.off[..., 0], self.off[..., 2]
        off = np.stack([ox * ca - oz * sa, self.off[..., 1], ox * sa + oz * ca], -1)
        p = c[:, None, :] + off * R[:, None, None]
        p[..., 2] = np.maximum(p[..., 2], 0.3 + 0.2 * R[:, None])
        rad = np.broadcast_to((0.5 * R)[:, None], p.shape[:2])
        fade = (np.clip(a / 0.08, 0, 1) * np.exp(-a / 9.0))[:, None]
        m = self.m[:, None] * self.subm * fade
        al = np.broadcast_to(alive[:, None], p.shape[:2])
        return p.reshape(-1, 3), rad.reshape(-1), m.reshape(-1), al.reshape(-1)

    @staticmethod
    def ballistic(c, t):
        """chips: up and outward, slowed by drag (linear in speed, enough here), gravity; they stop on the salt"""
        a = np.maximum(t - c["te"], 0.0)
        k = c["kdrag"][:, None]
        g = np.array([0, 0, -9.81])
        # v' = -k v + g  ->  x = x0 + (v0 - g/k)(1 - e^-kt)/k + g t / k
        e = (1 - np.exp(-k * a[:, None])) / k
        p = c["p0"] + (c["v0"] - g / k) * e + g * a[:, None] / k
        landed = p[:, 2] <= 0
        p[:, 2] = np.maximum(p[:, 2], 0)
        return p, (t > c["te"]), landed

    # ------------------------------------------------------------------ drawing
    def tau_layers(self, t, scale=0.5):
        """optical-depth maps (full picture size) for dust in front of the track and behind it"""
        p, rad, m, alive = self.fine_at(t)
        f = M.f_px(); hr = M.horizon_row(); zc = M.S["cam_z"]
        y = np.maximum(p[:, 1], 1.0)
        u = PW / 2 + f * p[:, 0] / y
        v = hr - f * (p[:, 2] - zc) / y
        rs = f * rad / y
        w = m * (f / y) ** 2 * self.norm
        ok = alive & (u > -3 * rs) & (u < PW + 3 * rs) & (v > -3 * rs) & (v < PH + 3 * rs) & (w > 0)
        hw, hh = int(PW * scale), int(PH * scale)
        near = p[:, 1] < M.S["dist"]
        out = []
        for sel in (ok & near, ok & ~near):
            tau = np.zeros((hh, hw))
            if sel.any():
                us, vs, rss, ws = u[sel] * scale, v[sel] * scale, rs[sel] * scale, w[sel] * scale * scale
                edges = [0, 1.2, 2.5, 5, 10, 20, 40, 80, 1e9]
                for lo, hi in zip(edges[:-1], edges[1:]):
                    b = (rss >= lo) & (rss < hi)
                    if not b.any():
                        continue
                    H, _, _ = np.histogram2d(vs[b], us[b], bins=(hh, hw), range=((0, hh), (0, hw)), weights=ws[b])
                    sig = float(np.median(rss[b])) * 0.6
                    tau += gaussian_filter(H, max(sig, 0.6), mode="constant") if sig > 0.3 else H
            out.append(np.asarray(Image.fromarray(tau.astype(np.float32)).resize((PW, PH), Image.BILINEAR)))
        return out

    def calibrate(self):
        """scale the dust so its thickest part a second after the pass has optical depth TAU_TARGET"""
        self.norm = 1.0
        tn, tf = self.tau_layers(M.S["t_pass"] + 1.0)
        q = np.percentile(np.maximum(tn, tf), 99.5)
        self.norm = TAU_TARGET / max(q, 1e-9)
        return self.norm

    def light(self, tau, salt):
        """colour of the dust where its optical depth is tau: sunlit salt where thin, sky-lit and bluer inside"""
        dense = gaussian_filter(tau, 4)
        tsun = np.exp(-0.35 * dense)[..., None]
        sunlit = salt * 1.12
        shade = salt * np.array([0.58, 0.61, 0.70], dtype=np.float32)
        return sunlit * tsun + shade * (1 - tsun)

    def chips_layer(self, t, salt):
        """crust chips and grit, motion-blurred over the shutter: premultiplied colour and alpha, split in front of /
        behind the track"""
        f = M.f_px(); hr = M.horizon_row(); zc = M.S["cam_z"]
        sh = M.S["shutter"] / C.FPS
        layers = []
        for front in (True, False):
            col = np.zeros((PH, PW, 3), np.float32); al = np.zeros((PH, PW), np.float32)
            for c, alpha in ((self.chip, 0.95), (self.grit, 0.55)):
                for k in range(7):
                    tt = t - sh / 2 + sh * k / 6
                    p, alive, landed = self.ballistic(c, tt)
                    sel = alive & ((p[:, 1] < M.S["dist"]) == front) & ~landed
                    if not sel.any():
                        continue
                    y = p[sel, 1]
                    u = PW / 2 + f * p[sel, 0] / y; v = hr - f * (p[sel, 2] - zc) / y
                    size = np.maximum(f * c["size"][sel] / y, 0.6)
                    a_ = alpha * np.minimum(size, 1.0) / 7 * np.minimum(size, 3)
                    iu, iv = np.floor(u).astype(int), np.floor(v).astype(int)
                    fu, fv = u - iu, v - iv
                    shade = c["shade"][sel]
                    for du, dv, ww in ((0, 0, (1 - fu) * (1 - fv)), (1, 0, fu * (1 - fv)), (0, 1, (1 - fu) * fv), (1, 1, fu * fv)):
                        uu, vv = iu + du, iv + dv
                        ok = (uu >= 0) & (uu < PW) & (vv >= 0) & (vv < PH)
                        wa = (a_ * ww)[ok]
                        np.add.at(al, (vv[ok], uu[ok]), wa)
                        np.add.at(col, (vv[ok], uu[ok]), wa[:, None] * (salt[None, :] * shade[ok][:, None]))
            al = np.clip(al, 0, 1)
            layers.append((col, al))
        return layers


def preview(times):
    """dust and chips over the plate alone, for tuning: python debris.py preview 2.2,3,4,6"""
    plate = C.to_lin(np.asarray(Image.open(os.path.join(M.OUT, "plate.png"))).astype(np.float32) / 255)
    hr = int(M.horizon_row())
    salt = plate[hr + 12:hr + 90].reshape(-1, 3).mean(0)
    D = Debris(); print("norm", D.calibrate())
    for t in times:
        pic = plate.copy()
        tn, tf = D.tau_layers(t)
        (cn, an), (cf, af) = D.chips_layer(t, salt)
        for tau, (cc, aa) in ((tf, (cf, af)), (tn, (cn, an))):
            T = np.exp(-tau)[..., None]
            pic = pic * T + D.light(tau, salt) * (1 - T)
            pic = cc + pic * (1 - aa[..., None])
        Image.fromarray((C.to_srgb(pic) * 255 + 0.5).astype(np.uint8)).save(os.path.join(M.OUT, f"dust_{t:.2f}.png"))
        print("t", t, "tau max", round(float(max(tn.max(), tf.max())), 2))


if __name__ == "__main__":
    if sys.argv[1] == "preview":
        preview([float(x) for x in sys.argv[2].split(",")])
