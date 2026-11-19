"""Assemble "Every shore on Earth at light speed".

Timeline (video seconds; the fish's light-time t comes from a speed profile):
  title       world map, every shore still blue
  zoom in     to downtown Calgary, on the Bow River
  the rule    the fish sets off slowly: down the Bow, right at the first fork into the Elbow, up to the
              sources, back down the other bank; captions explain the rule
  speed up    log-linear from ~1 km/s to light speed while the camera pulls back to the whole world
  real time   the rest of the journey at the speed of light, clock running
  slow down   the last stretch down the Bow from the Rockies, zooming back in to where it started
  end card

    python film.py frames 0 9 12 40 ...    # PNG frames at these video times
    python film.py video [--workers 3]     # -> out/coast_light_speed.mp4 (no audio; music.py adds it)
    python film.py timeline                # print the event times
"""
import os, sys, math, json, time, argparse, subprocess
import numpy as np
import cv2
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cproj import W, H, RES, to_px  # noqa: E402
import render as R  # noqa: E402
import zoom as Zm  # noqa: E402

DATA, OUT = R.DATA, R.OUT
FPS = 30
C = R.C_KMS
START_LL = (-114.0719, 51.0569)           # the fish's start (tour.py), downtown Calgary
CONFLUENCE_LL = (-114.0381, 51.0452)       # the Elbow joins the Bow

# ---------------------------------------------------------------------------- speed profile (km/s)
TITLE_END = 4.0
ZOOM_IN_END = 8.0
FISH_START = 8.0
INTRO_V = [(8.0, 0.8), (10.4, 1.0), (12.0, 7.0), (14.0, 26.0), (17.0, 30.0), (20.0, 42.0), (23.5, 60.0),
           (25.0, 110.0), (27.0, 900.0), (29.0, 9000.0), (31.0, 80000.0), (32.5, C)]
OUTRO_V = [(0.0, C), (1.6, 60000.0), (3.2, 6000.0), (4.8, 700.0), (6.4, 80.0), (8.0, 9.0), (9.4, 1.6), (10.2, 0.9)]
END_HOLD = 5.4                             # after the fish arrives: a beat in Calgary, then back out to the world
PULL_OUT = (1.2, 4.6)                      # ... the pull-back, relative to arrival
CARD = 11.0                                # end card


def loglin(keys, x):
    xs = np.array([k[0] for k in keys]); vs = np.log(np.array([k[1] for k in keys]))
    return np.exp(np.interp(x, xs, vs))


class Timeline:
    def __init__(self, total_light_s):
        self.total = total_light_s
        total_km = total_light_s * C
        dt = 1e-4
        # intro: distance covered from FISH_START to the end of the ramp
        ti = np.arange(FISH_START, INTRO_V[-1][0] + dt, dt)
        vi = loglin(INTRO_V, ti)
        si = np.r_[0.0, np.cumsum(0.5 * (vi[1:] + vi[:-1]) * dt)]
        self.ti, self.si = ti, si
        self.ramp_end = INTRO_V[-1][0]
        # outro: distance covered over the slow-down, ending exactly at the finish
        to = np.arange(0.0, OUTRO_V[-1][0] + dt, dt)
        vo = loglin(OUTRO_V, to)
        so = np.r_[0.0, np.cumsum(0.5 * (vo[1:] + vo[:-1]) * dt)]
        self.to, self.so, self.vo = to, so, vo
        out_km = so[-1]
        main_km = total_km - si[-1] - out_km
        assert main_km > 0
        self.main_start = self.ramp_end
        self.outro_start = self.ramp_end + main_km / C
        self.arrive = self.outro_start + OUTRO_V[-1][0]
        self.card_start = self.arrive + END_HOLD
        self.end = self.card_start + CARD
        self.km_intro = si[-1]

    def km(self, tau):
        """distance swum (km) at video time tau"""
        if tau <= FISH_START:
            return 0.0
        if tau <= self.ramp_end:
            return float(np.interp(tau, self.ti, self.si))
        if tau <= self.outro_start:
            return self.km_intro + (tau - self.ramp_end) * C
        if tau <= self.arrive:
            u = tau - self.outro_start
            return self.km_intro + (self.outro_start - self.ramp_end) * C + float(np.interp(u, self.to, self.so))
        return self.total * C

    def speed(self, tau):
        if FISH_START <= tau <= self.ramp_end:
            return float(loglin(INTRO_V, tau))
        if self.ramp_end < tau <= self.outro_start:
            return C
        if self.outro_start < tau <= self.arrive:
            return float(loglin(OUTRO_V, tau - self.outro_start))
        return 0.0

    def tau_of_km(self, km):
        ts = np.linspace(0, self.end, 200001)
        ks = np.array([self.km(x) for x in ts[::50]])
        return float(np.interp(km, ks, ts[::50]))


# ---------------------------------------------------------------------------- camera
def px_per_km_at(lon, lat):
    x0, y0 = to_px(lon - 0.05, lat)
    x1, y1 = to_px(lon + 0.05, lat)
    return math.hypot(x1 - x0, y1 - y0) / (0.1 * 111.32 * math.cos(math.radians(lat)))


def smooth(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


class Camera:
    """camera = (centre in world pixels, zoom); zoom 1 = the world map"""

    def __init__(self, tl):
        self.tl = tl
        self.ppk = px_per_km_at(*START_LL)
        sx, sy = to_px(*START_LL)
        self.start_px = (float(sx), float(sy))
        mx, my = to_px(-114.55, 50.86)
        self.mid_px = (float(mx), float(my))
        # the close-up track: (tau, centre px, view width km)
        self.track = [(ZOOM_IN_END, self.start_px, 9.0), (10.6, self.start_px, 11.0), (12.6, None, 40.0),
                      (14.6, self.mid_px, 125.0), (23.0, self.mid_px, 125.0), (25.6, self.start_px, 160.0)]

    def zoom_for(self, w_km):
        return W / (w_km * self.ppk)

    def anchor_zoom(self, anchor, z_close, u_close):
        """u_close = 1: zoomed in on anchor at z_close; 0: the world map. Log zoom and the anchor's screen
        position move together, so the anchor glides from where it sits on the world map to the centre."""
        lz = u_close * math.log(z_close)
        z = math.exp(lz)
        f = lz / math.log(z_close)
        ax, ay = anchor
        sx = ax + (W / 2 - ax) * f
        sy = ay + (H / 2 - ay) * f
        return ax - (sx - W / 2) / z, ay - (sy - H / 2) / z, z

    def at(self, tau, fish_px=None):
        tl = self.tl
        if tau < TITLE_END:
            return W / 2, H / 2, 1.0
        if tau < ZOOM_IN_END:
            u = smooth((tau - TITLE_END) / (ZOOM_IN_END - TITLE_END))
            return self.anchor_zoom(self.start_px, self.zoom_for(9.0), u)
        last_t, last_c, last_w = self.track[-1]
        if tau <= last_t:
            for (t0, c0, w0), (t1, c1, w1) in zip(self.track[:-1], self.track[1:]):
                if t0 <= tau <= t1:
                    u = smooth((tau - t0) / (t1 - t0))
                    c0 = c0 or fish_px or self.start_px
                    c1 = c1 or fish_px or self.start_px
                    w = math.exp(math.log(w0) + u * (math.log(w1) - math.log(w0)))
                    return c0[0] + u * (c1[0] - c0[0]), c0[1] + u * (c1[1] - c0[1]), self.zoom_for(w)
        if tau < tl.ramp_end + 0.8:
            u = smooth((tau - last_t) / (tl.ramp_end + 0.8 - last_t))
            return self.anchor_zoom(last_c, self.zoom_for(last_w), 1 - u)
        if tau < tl.outro_start + 1.0:
            return W / 2, H / 2, 1.0
        if tau < tl.arrive:
            u = smooth((tau - tl.outro_start - 1.0) / (tl.arrive - tl.outro_start - 1.0))
            return self.anchor_zoom(self.start_px, self.zoom_for(26.0), u)
        u = smooth((tau - tl.arrive - PULL_OUT[0]) / (PULL_OUT[1] - PULL_OUT[0]))
        if u >= 1.0:
            return W / 2, H / 2, 1.0
        return self.anchor_zoom(self.start_px, self.zoom_for(26.0), 1.0 - u)


# ---------------------------------------------------------------------------- text
CAPTIONS = []           # (tau0, tau1, text), filled by make_captions


def make_captions(tl):
    ev = {k: tl.tau_of_km(v) for k, v in (("fork", 2.8), ("little_src", 135.1), ("elbow_src", 183.5), ("back", 386.9))}
    caps = [
        (FISH_START + 0.3, ev["fork"] - 0.2, "A fish of light sets off down the Bow River, in Calgary."),
        (ev["fork"] - 0.2, ev["little_src"] - 1.2, "Its one rule: keep the land on your right. So at every fork, it turns right."),
        (ev["little_src"] - 1.2, ev["back"] - 1.0, "At a river's source it turns round and swims back down the other bank."),
        (ev["back"] - 1.0, tl.ramp_end - 3.2, "Then on to the next fork, the next river, the next lake, the next coast..."),
        (tl.ramp_end - 3.2, tl.ramp_end + 1.6, "...until it has swum every shore on Earth. At the speed of light."),
        (tl.outro_start + 2.2, tl.arrive + 1.4, "Down the Bow from the Rockies, and back to where it started."),
        (tl.arrive + 1.4, tl.card_start + 0.4, "Every coastline, every lake shore, both banks of every river."),
    ]
    CAPTIONS[:] = caps
    return ev


def fade(tau, a, b, f=0.45):
    if tau < a or tau > b:
        return 0.0
    return min(1.0, (tau - a) / f, (b - tau) / f)


def fmt_clock(s):
    m = int(s // 60)
    return f"{m}:{s - 60 * m:05.2f}"


def fmt_speed(v):
    if v >= C * 0.999:
        return "light speed"
    if v >= 1000:
        return f"{v:,.0f} km/s"
    return f"{v:,.1f} km/s" if v < 10 else f"{v:,.0f} km/s"


class Overlay:
    def __init__(self, tl, stats):
        self.tl = tl
        self.stats = stats
        self.f_title = R.font("Inter-SemiBold.otf", 64)
        self.f_sub = R.font("Inter-Regular.otf", 28)
        self.f_head = R.font("Inter-SemiBold.otf", 30)
        self.f_head2 = R.font("Inter-Regular.otf", 18)
        self.f_cap = R.font("Inter-Medium.otf", 30)
        self.f_clock = R.font("InterDisplay-Medium.otf", 58)
        self.f_small = R.font("Inter-Medium.otf", 15)
        self.f_info = R.font("Inter-Regular.otf", 17)
        self.f_lab = R.font("Inter-Medium.otf", 19)
        self.f_note = R.font("Inter-Regular.otf", 13)
        self.f_card_big = R.font("InterDisplay-Medium.otf", 96)
        self.f_card = R.font("Inter-Regular.otf", 25)
        self.f_card_b = R.font("Inter-SemiBold.otf", 25)

    def draw(self, tau, t_light, cam, labels):
        tl = self.tl
        im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        r = RES
        # title card
        a = fade(tau, -1, ZOOM_IN_END - 1.4, 0.8)
        if a > 0:
            d.text((W / 2, H / 2 - 70 * r), "Every shore on Earth", font=self.f_title, fill=(240, 244, 250, int(255 * a)), anchor="mm")
            d.text((W / 2, H / 2 + 6 * r), "at the speed of light", font=self.f_sub, fill=(255, 196, 110, int(255 * a)), anchor="mm")
            d.text((W / 2, H / 2 + 58 * r), "Every coastline, every lake shore, both banks of every river: one fish, one rule.",
                   font=self.f_info, fill=(170, 190, 212, int(235 * a)), anchor="mm")
        # running header
        a = fade(tau, ZOOM_IN_END - 0.4, tl.card_start + 0.2, 0.6)
        if a > 0:
            d.text((56 * r, 32 * r), "Every shore on Earth at light speed", font=self.f_head, fill=(236, 242, 250, int(255 * a)))
            d.text((58 * r, 72 * r), "A fish of light · land always on its right", font=self.f_head2, fill=(150, 176, 205, int(255 * a)))
            right = W - 58 * r
            d.text((right, 30 * r), "ELAPSED", font=self.f_small, fill=(130, 150, 175, int(255 * a)), anchor="ra")
            d.text((right, 46 * r), fmt_clock(t_light), font=self.f_clock, fill=(246, 236, 214, int(255 * a)), anchor="ra")
            v = tl.speed(tau)
            if tau < tl.arrive:
                d.text((right, 118 * r), f"speed: {fmt_speed(v)}", font=self.f_info,
                       fill=(255, 196, 110, int(255 * a)) if v < C * 0.999 else (160, 178, 200, int(255 * a)), anchor="ra")
            d.text((right, 142 * r), f"{t_light * C / 1e6:,.2f} million km swum" if t_light * C >= 1e5 else f"{t_light * C:,.1f} km swum",
                   font=self.f_info, fill=(160, 178, 200, int(255 * a)), anchor="ra")
            if labels.get("now") and cam[2] < 1.01:
                na = labels.get("now_a", 1.0) * a
                d.text((right, 170 * r), labels["now"], font=self.f_lab, fill=(255, 214, 150, int(255 * na)), anchor="ra")
            # legend and sources along the bottom
            lx, ly = 58 * r, H - 50 * r
            for i, (col, txt) in enumerate((((60, 170, 255), "not reached yet"), ((150, 105, 45), "one riverbank done"),
                                            ((255, 176, 66), "coast, lake shore, or both banks"))):
                x = lx + [0, 160, 345][i] * r
                d.line([(x, ly + 9 * r), (x + 24 * r, ly + 9 * r)], fill=col + (int(255 * a),), width=3 * r)
                d.text((x + 32 * r, ly), txt, font=self.f_small, fill=(140, 165, 190, int(255 * a)))
            d.text((lx, ly + 24 * r), "Coastline, rivers, lakes © OpenStreetMap contributors (ODbL), via osmdata.openstreetmap.de and "
                   "Overture Maps. Rings: leaps where no water connects.", font=self.f_note, fill=(110, 130, 155, int(255 * a)))
        # captions
        for (a0, a1, txt) in CAPTIONS:
            ca = fade(tau, a0, a1)
            if ca > 0:
                tw = d.textlength(txt, font=self.f_cap)
                bx0, by0 = W / 2 - tw / 2 - 22 * r, H - 150 * r
                d.rounded_rectangle([bx0, by0, W / 2 + tw / 2 + 22 * r, by0 + 54 * r], radius=10 * r, fill=(4, 8, 16, int(170 * ca)))
                d.text((W / 2, by0 + 27 * r), txt, font=self.f_cap, fill=(240, 244, 250, int(255 * ca)), anchor="mm")
        # map labels in the close-ups
        for (x, y, txt, la, big) in labels.get("map", []):
            if la > 0:
                f = self.f_lab if not big else self.f_head
                d.text((x, y), txt, font=f, fill=(200, 215, 235, int(230 * la)), anchor="ls")
        # end card
        a = fade(tau, tl.card_start, tl.end + 5, 1.0)
        if a > 0:
            s = self.stats
            cy = H * 0.30
            d.text((W / 2, cy), fmt_clock(tl.total).lstrip("0:") if tl.total < 60 else f"{int(tl.total // 60)} min {tl.total % 60:.1f} s",
                   font=self.f_card_big, fill=(255, 214, 150, int(255 * a)), anchor="mm")
            d.text((W / 2, cy + 82 * r), "for light to swim every shore on Earth", font=self.f_card, fill=(230, 236, 245, int(255 * a)), anchor="mm")
            rows = [
                (f"{s['total_km'] / 1e6:.1f} million km in all", True),
                (f"coastline {s['coast_km'] / 1e6:.2f} M km · lake shores {s['lake_km'] / 1e6:.2f} M km · "
                 f"riverbanks 2 × {s['river_km'] / 1e6:.2f} M km", False),
                (f"short links across gaps in the river map {s['link_km'] / 1e6:.2f} M km · open-sea swims {s['swim_km'] / 1e6:.2f} M km", False),
                ("", False),
                (f"The CIA World Factbook counts 356,000 km of coastline: 1.2 light-seconds.", False),
                (f"Traced at OpenStreetMap's detail it is {s['coast_km'] / 1e6:.2f} million km. The closer you look, the longer it gets.", False),
            ]
            y = cy + 170 * r
            for txt, b in rows:
                if txt:
                    d.text((W / 2, y), txt, font=self.f_card_b if b else self.f_card, fill=(200, 212, 228, int(255 * a)), anchor="mm")
                y += 44 * r
        return im


# ---------------------------------------------------------------------------- frames
class Film:
    def __init__(self):
        self.world = R.World()
        self.bg = self.world.L["bg"]
        self.zoom = Zm.Zoom(self.bg)
        self.tl = Timeline(self.world.total)
        self.cam = Camera(self.tl)
        self.ev = make_captions(self.tl)
        self.stats = self._stats()
        self.ov = Overlay(self.tl, self.stats)
        self.now = self._now_labels()
        self.fish_cache = {}
        self.heading = 0.0

    def _stats(self):
        ns = json.load(open(os.path.join(DATA, "network_stats.json")))
        ts = json.load(open(os.path.join(DATA, "tour_stats.json")))
        z = np.load(os.path.join(DATA, "network.npz"))
        link_km = 2 * float(z["km"][z["kind"] == Zm.R.K_LINK].sum())
        return {"total_km": ts["total_km"], "coast_km": ns["coast_km"], "lake_km": ns["lake_shore_km"], "river_km": ns["river_km"],
                "link_km": link_km,
                "swim_km": 2 * ts["swim_km_each_way"], "leaps": ts["leaps"], "seconds": ts["seconds_at_c"]}

    def _now_labels(self):
        p = os.path.join(DATA, "now_labels.json")
        return json.load(open(p)) if os.path.exists(p) else []

    def now_label(self, t):
        best, ba = None, 0.0
        for (t0, t1, name) in self.now:
            if t0 - 0.25 <= t <= t1 + 0.25:
                a = min(1.0, (t - t0 + 0.25) / 0.25, (t1 + 0.25 - t) / 0.25)
                if a > ba:
                    best, ba = name, a
        return best, ba

    def frame(self, tau, clean=False):
        tl = self.tl
        km = tl.km(tau)
        t = km / C
        fish_world = None
        close = FISH_START - 0.01 <= tau <= tl.arrive + END_HOLD + 1 and (tau <= tl.ramp_end + 0.8 or tau >= tl.outro_start + 1.0)
        if close or tau < FISH_START:
            if t < 1.4 or t > tl.total - 1.4:
                fx, fy, _, _, _ = self.zoom.fish(min(max(t, 1e-9), tl.total))
                fish_world = (fx, fy)
        cx, cy, z = self.cam.at(tau, fish_world)
        v = tl.speed(tau)
        flash_t = 0.45 * max(v, 1.0) / C
        labels = {}
        if z > 1.0005:
            bank = 0.0
            w_km = W / (z * self.cam.ppk)
            if w_km < 60:
                bank = 5.0 * RES * min(1.0, (60 - w_km) / 30)
            E, bg = self.zoom.emission(t, cx, cy, z, flash_t, bank)
            self.zoom_fish(E, t, tau, cx, cy, z, bank, v)
            img = R.finish(E, bg)
            labels["map"] = self.map_labels(tau, cx, cy, z)
        else:
            if tau < FISH_START:
                self.world.reset_to(-1)
                E = self.world.emission(-1.0)
            else:
                E = self.world.emission(t, flash_s=min(R.FLASH_S, 0.45 * v / C) if v < C else R.FLASH_S)
                if tl.arrive > tau:
                    self.world.add_fish(E, t, trail_s=min(0.012, 0.3 * v / C))
                self.world.rings(E, t)
            if tau >= tl.card_start:
                E *= 1.0 - 0.72 * smooth((tau - tl.card_start) / 1.2)
            img = R.finish(E, self.bg)
            if tl.main_start <= tau <= tl.outro_start:
                labels["now"], labels["now_a"] = self.now_label(t)
        if not clean:
            ov = self.ov.draw(tau, min(t, tl.total), (cx, cy, z), labels)
            img = R.overlay(img, ov)
        return R.to8(img)

    def zoom_fish(self, E, t, tau, cx, cy, z, bank, v):
        if not (FISH_START <= tau <= self.tl.arrive + END_HOLD + 5):
            return
        def scr(x, y):
            return (np.asarray(x) - cx) * z + W / 2, (np.asarray(y) - cy) * z + H / 2
        in_fine = t < 1.4 or t > self.tl.total - 1.4
        if in_fine:
            fx, fy, dx, dy, river = self.zoom.fish(min(t, self.tl.total))
            back = 0.35 * max(v, 0.5) / C
            tx, ty, tt = self.zoom.trail(t, back)
        else:
            fx, fy, _ = self.world.fish_at(t)
            dx = dy = 0.0
            river = False
            tx = ty = np.zeros(0)
        sx, sy = scr(fx, fy)
        sx, sy = float(sx), float(sy)
        if math.hypot(dx, dy) > 0:
            ang = math.atan2(dy, dx)
            dd = (ang - self.heading + math.pi) % (2 * math.pi) - math.pi
            self.heading += 0.5 * dd
            nx, ny = -math.sin(self.heading), math.cos(self.heading)     # right of travel, screen y down
            if river and bank > 0:
                sx += nx * bank * 0.62
                sy += ny * bank * 0.62
        # trail along the fine path, on the same bank
        if len(tx) > 1:
            px, py = scr(tx, ty)
            pts = np.round(np.c_[px, py] * 16).astype(np.int32)
            tr = np.zeros((H, W), np.uint8)
            n = len(pts)
            for b in range(5):
                a_, b_ = b * (n - 1) // 5, (b + 1) * (n - 1) // 5 + 1
                if b_ - a_ >= 2:
                    cv2.polylines(tr, [pts[a_:b_]], False, int(40 + 160 * (b + 1) / 5), max(1, RES), cv2.LINE_AA, shift=4)
            ys, xs = np.nonzero(tr)
            E[ys, xs] += tr[ys, xs, None].astype(np.float32) / 255 * np.array([1.0, 0.82, 0.5], np.float32) * 0.9
        scale = 1.0 + 0.9 * min(1.0, max(0.0, math.log(z) / math.log(400)))
        key = (round(self.heading * 16) % 101, round(scale * 8))
        if key not in self.fish_cache:
            self.fish_cache[key] = R.fish_sprite(round(self.heading * 16) / 16, scale)
        R.add_sprite(E, sx, sy, self.fish_cache[key])

    def map_labels(self, tau, cx, cy, z):
        out = []
        w_km = W / (z * self.cam.ppk)
        items = [((-114.072, 51.047), "Calgary", 5, 400, True),
                 ((-114.115, 51.071), "Bow River", 4, 30, False),
                 ((-114.118, 51.012), "Elbow River", 4, 30, False),
                 ((-114.72, 50.88), "Elbow River", 40, 400, False),
                 ((-114.98, 50.73), "Little Elbow River", 40, 400, False),
                 ((-114.28, 51.16), "Bow River", 40, 400, False)]
        for (lon, lat), txt, wmin, wmax, big in items:
            if not (wmin <= w_km <= wmax):
                continue
            la = min(1.0, (w_km - wmin) / (0.3 * wmin + 1e-9), (wmax - w_km) / (0.3 * wmax))
            x, y = to_px(lon, lat)
            sx, sy = (float(x) - cx) * z + W / 2, (float(y) - cy) * z + H / 2
            if 0 < sx < W - 200 * RES and 60 * RES < sy < H - 160 * RES and not (sx > W - 520 * RES and sy < 200 * RES) \
                    and not (sx < 640 * RES and sy < 110 * RES):
                out.append((sx, sy, txt, la * fade(tau, ZOOM_IN_END - 1.0, self.tl.arrive + END_HOLD, 0.8), big))
        return out


def render_range(args):
    f0, f1, path = args
    F = Film()
    tl = F.tl
    proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                             "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-tune", "film",
                             "-pix_fmt", "yuv420p", "-profile:v", "high", "-movflags", "+faststart", path], stdin=subprocess.PIPE)
    t0 = time.time()
    first = f0 / FPS
    # warm the incremental world raster up to the first frame
    F.world.reset_to(max(0.0, tl.km(first) / C) - 1e-6)
    for f in range(f0, f1):
        proc.stdin.write(F.frame(f / FPS).tobytes())
        if (f - f0) % 150 == 0:
            print(f"  [{os.path.basename(path)}] {f - f0}/{f1 - f0} frames, {time.time() - t0:.0f}s", flush=True)
    proc.stdin.close()
    proc.wait()
    return path


def video(workers):
    from multiprocessing import Pool
    os.makedirs(OUT, exist_ok=True)
    F = Film()
    n = int(round(F.tl.end * FPS))
    print(f"film: {F.tl.end:.2f} s, {n} frames; main real time {F.tl.main_start:.2f}-{F.tl.outro_start:.2f}", flush=True)
    del F
    # cut where frames are cheap and balanced: close-ups cost more, so more cuts there
    cuts = np.linspace(0, n, workers * 3 + 1).astype(int)
    jobs = [(int(a), int(b), os.path.join(OUT, f"_part{i:02d}.mp4")) for i, (a, b) in enumerate(zip(cuts[:-1], cuts[1:]))]
    with Pool(workers) as pool:
        parts = pool.map(render_range, jobs, chunksize=1)
    lst = os.path.join(OUT, "_parts.txt")
    open(lst, "w").write("".join(f"file '{os.path.basename(p)}'\n" for p in parts))
    final = os.path.join(OUT, f"coast_light_speed_r{RES}.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
                    "-movflags", "+faststart", final], check=True)
    for p in parts:
        os.remove(p)
    os.remove(lst)
    print("wrote", final)


def frames(taus):
    os.makedirs(OUT, exist_ok=True)
    F = Film()
    for tau in taus:
        t = F.tl.km(tau) / C
        F.world.reset_to(max(0.0, t - 0.05))
        F.world.heading = 0
        for tt in np.linspace(tau - 0.1, tau, 4):
            img = F.frame(float(tt))
        p = os.path.join(OUT, f"frame_{tau:07.2f}.png")
        cv2.imwrite(p, img[..., ::-1])
        print(p, f"t={t:.4f}s")


def clean_frames(taus):
    """frames without any text, for thumbnails"""
    os.makedirs(OUT, exist_ok=True)
    F = Film()
    for tau in taus:
        t = F.tl.km(tau) / C
        F.world.reset_to(max(0.0, t - 0.05))
        for tt in np.linspace(tau - 0.1, tau, 4):
            img = F.frame(float(tt), clean=True)
        p = os.path.join(OUT, f"clean_{tau:07.2f}.png")
        cv2.imwrite(p, img[..., ::-1])
        print(p)


def timeline():
    F = Film()
    tl = F.tl
    print(json.dumps({"events_tau": F.ev, "ramp_end": tl.ramp_end, "km_intro": tl.km_intro, "outro_start": tl.outro_start,
                      "arrive": tl.arrive, "card_start": tl.card_start, "end": tl.end, "total_light_s": tl.total}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("taus", nargs="*", type=float)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    if a.cmd == "frames":
        frames(a.taus)
    elif a.cmd == "video":
        video(a.workers)
    elif a.cmd == "clean":
        clean_frames(a.taus)
    elif a.cmd == "timeline":
        timeline()
