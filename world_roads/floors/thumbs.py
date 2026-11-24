"""Three thumbnails for YouTube's Test & Compare (1280x720), each with a big robot vacuum and drawn from the film's
own renders, in the Roomba film's style (roomba/thumbs.py):

    A  the X-ray Empire State Building half cleaned, a robot vacuum streaking at it: YOUR ROBOT VACUUM / AT LIGHT SPEED
    B  Manhattan half gold:                                 EVERY FLOOR / IN NEW YORK / 7.4 s
    C  the burning planet from the ending:                  SPOTLESS. (the floors, anyway)
    D  the globe half gold, the answer up front:            VACUUM THE PLANET / 1 h 08 min
    E  the Empire State Building, all gold:                 THE EMPIRE STATE BUILDING / 3.5 ms
    F  the U-turn flash at the end of a row:                EVERY U-TURN = / 10 TSAR BOMBAS
    G  the opening close-up, robot at true size:            SLOWED 1,000,000,000× / ...IT'S AS FAST AS YOURS
    H  versus:                                              ROBOT VACUUM / VS. EARTH

    python thumbs.py [A ... H]    -> out/thumbs/every_floor_thumb_{A..H}.png
"""
import os, sys, math
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "signal"))
sys.path.insert(0, os.path.join(ROOT, "roomba"))
sys.path.insert(0, HERE)
import style as ST                                                    # noqa: E402
import video as V                                                     # noqa: E402  (this film's, before frames.py adds roomba/)
import xray as X                                                      # noqa: E402
import cityfilm as CF                                                 # noqa: E402
import frames as FR                                                   # noqa: E402

FR.SS = 1
TW, TH = 1280, 720
OUT = os.path.join(HERE, "out", "thumbs")
GOLD = (255, 186, 80)
WHITE = (255, 255, 255)
RED = (255, 52, 52)


def headline(d, xy, s, size, col, anchor="la"):
    d.text(xy, s, font=ST.font(size, "Bold"), fill=col, stroke_width=max(5, size // 16), stroke_fill=(0, 0, 0), anchor=anchor)


def marker(d, xy, s, size, col=RED, anchor="la"):
    f = ImageFont.truetype(os.path.join(ROOT, "thumb_fonts", "PermanentMarker-Regular.ttf"), size)
    d.text(xy, s, font=f, fill=col, anchor=anchor, stroke_width=3, stroke_fill=(0, 0, 0))


def hero_robot(im, x, y, width, heading=-30.0, streak_len=2.8):
    """a three-quarter-view robot vacuum at (x, y) driving toward `heading` (screen degrees, counter-clockwise from
    right), with a light-speed streak behind it and a warm glow under it (as in the Roomba film's thumbnails)"""
    import roomba3d
    arr = np.asarray(im.convert("RGB")).astype(np.float32)
    h, w = arr.shape[:2]
    hd = math.radians(heading)
    dx, dy = math.cos(hd), -math.sin(hd)
    px, py = -dy, dx
    gy = y + width * 0.10
    L, r0 = width * streak_len, width * 0.34
    tail = (x - dx * L, gy - dy * L)
    streak = np.zeros((h, w), np.float32)
    poly = np.array([(x + px * r0, gy + py * r0 * 0.55), (x - px * r0, gy - py * r0 * 0.55), (tail[0], tail[1])], np.int32)
    cv2.fillPoly(streak, [poly], 1.0, cv2.LINE_AA)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    along = ((xx - x) * -dx + (yy - gy) * -dy) / L
    streak *= np.clip(1 - along, 0, 1) ** 1.6
    streak = cv2.GaussianBlur(streak, (0, 0), width * 0.05)
    glow = np.zeros((h, w), np.float32)
    cv2.ellipse(glow, (int(x), int(gy + width * 0.05)), (int(width * 0.50), int(width * 0.24)), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
    glow = cv2.GaussianBlur(glow, (0, 0), width * 0.09)
    arr = arr + streak[..., None] * np.array([255, 236, 196]) * 0.9 + glow[..., None] * np.array(GOLD) * 0.6
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).convert("RGBA")
    sp = roomba3d.sprite(int(width), heading)
    out.alpha_composite(sp, (int(x - sp.width / 2), int(y - sp.height / 2)))
    return out.convert("RGB")


def save(im, k):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"every_floor_thumb_{k}.png")
    im.save(p)
    print("wrote", p, flush=True)


def thumb_A():
    """the Empire State Building in X-ray, cleaned up to its 60th floor, Midtown blue around it; the robot streaks in"""
    R = CF.CityRenderer(levels=("P1",))
    C = R.C
    tower = R.esb_k[np.argmax([(C.pz1[k] - C.pz0[k]) * (C.pnfl[k] > 0) for k in R.esb_k])]
    tw = [i for i, q in enumerate(R.esb_slabs) if q[1] == tower]
    L = R.esb_time_at_slab(tw[22], 0.55)
    cam = X.Cam(CF.PSI, CF.EL, 2.6, C.cx[R.esb], C.cy[R.esb], 190, ox=1420, oy=560, W=1920, H=1080, ss=1)
    img = R.frame(cam, L, nb=1.0)
    z, i, f = R.esb_state(L)
    _, k, j, _ = R.esb_slabs[i]
    x, y, _ = FR.active_floor(C, k, R.esb_slabs[i][0], f, cam.s, cam, img, dust_n=0, wake=False, rows_gain=1.0)
    CF.light_win(img, x, y, 1.1, gain=1.4)
    im = Image.fromarray(CF.finish(img)).resize((TW, TH), Image.LANCZOS)
    im = hero_robot(im, 470, 455, 330, heading=8, streak_len=2.4)
    d = ImageDraw.Draw(im)
    headline(d, (40, 36), "YOUR ROBOT VACUUM", 82, WHITE)
    headline(d, (40, 132), "AT LIGHT SPEED", 104, GOLD)
    save(im, "A")


def thumb_B():
    """Manhattan in X-ray, half gold, the robot streaking up the island"""
    R = CF.CityRenderer(levels=("P2", "P3"))
    C = R.C
    L = 1.6
    R.advance(L)
    ex, ey = C.cx[R.esb], C.cy[R.esb]
    cam = X.Cam(CF.PSI, CF.EL, 0.118, ex + 1500, ey + 2500, 0, ox=1180, oy=540, W=1920, H=1080, ss=1)
    img = R.frame(cam, L)
    xw, yw, zw = R.frontier_point(L)
    u, v = cam.project(np.array([xw]), np.array([yw]), np.array([zw]))
    CF.light_win(img, float(u[0]), float(v[0]), 0.8)
    im = Image.fromarray(CF.finish(img, bloom=(0.26, 0.17, 0.1))).resize((TW, TH), Image.LANCZOS)
    im = hero_robot(im, 930, 330, 380, -35)
    d = ImageDraw.Draw(im)
    headline(d, (40, 40), "EVERY FLOOR", 110, GOLD)
    headline(d, (44, 160), "IN NEW YORK", 72, WHITE)
    headline(d, (40, TH - 40), "7.4 s", 140, GOLD, "ls")
    save(im, "B")


def thumb_C():
    """the planet glowing orange-hot after 273 billion U-turns; the robot streaks past"""
    g = dict(cx=1220, cy=560, r=520, lon0=-35, lat0=22)
    fr = V.globe_img(g, V.WORLD_FLOOR_M2 * 1.0001, fire=1.6)
    im = Image.fromarray(fr).resize((TW, TH), Image.LANCZOS)
    im = hero_robot(im, 330, 470, 300, heading=-18, streak_len=2.2)
    d = ImageDraw.Draw(im)
    headline(d, (40, 40), "SPOTLESS.", 150, WHITE)
    marker(d, (52, 215), "(the floors, anyway)", 54)
    save(im, "C")


def thumb_D():
    """the globe half gold, the light at the frontier, the robot streaking across: the answer up front"""
    import globe as G
    S = V.Sweep()
    u = 30.0
    done = float(S.done(u))
    flon, flat, clon, clat = S.at(u)
    g = dict(cx=1250, cy=600, r=560, lon0=25, lat0=20)
    fr = V.globe_img(g, done, light=(flon, flat), light_size=1.3)
    im = Image.fromarray(fr).resize((TW, TH), Image.LANCZOS)
    im = hero_robot(im, 860, 470, 300, heading=-8, streak_len=2.6)
    d = ImageDraw.Draw(im)
    headline(d, (40, 36), "VACUUM", 108, WHITE)
    headline(d, (40, 150), "THE PLANET", 108, WHITE)
    headline(d, (40, TH - 40), "1 h 08 min", 120, GOLD, "ls")
    save(im, "D")


def thumb_E():
    """the Empire State Building done, all gold, Midtown still blue around it"""
    R = CF.CityRenderer(levels=("P1",))
    C = R.C
    L = R.esb_t1 + 1e-7
    cam = X.Cam(CF.PSI, CF.EL, 2.6, C.cx[R.esb], C.cy[R.esb], 190, ox=1380, oy=560, W=1920, H=1080, ss=1)
    img = R.frame(cam, L, nb=1.0)
    im = Image.fromarray(CF.finish(img, bloom=(0.3, 0.2, 0.12))).resize((TW, TH), Image.LANCZOS)
    im = hero_robot(im, 760, 560, 250, heading=60, streak_len=2.4)
    d = ImageDraw.Draw(im)
    headline(d, (40, 40), "THE EMPIRE STATE", 78, WHITE)
    headline(d, (40, 130), "BUILDING", 78, WHITE)
    headline(d, (36, TH - 50), "3.5 ms", 170, GOLD, "ls")
    save(im, "E")


def thumb_F():
    """the end of a row at the wall, the moment of the U-turn: a white-hot flash"""
    U = V.UTurn()
    C = U.C
    t = U.HIT - 0.05
    d_left = max(0.0, (U.HIT - t) * U.SPEED)
    f = U.f_end - d_left / U.cum[-1]
    img = np.zeros((1080, 1920, 3), np.float32) + X.BG
    x, y, hd = FR.active_floor(C, U.tower, U.zf, f, U.cam.s, U.cam, img, dust_n=250000, wake=True, rows_gain=0.30)
    CF.light_win(img, x, y, 7.0, gain=14.0)
    img += 0.35 * np.array([1.0, 0.85, 0.6], np.float32)
    im = Image.fromarray(CF.finish(img, bloom=(0.3, 0.25, 0.2))).resize((TW, TH), Image.LANCZOS)
    im = hero_robot(im, x * TW / 1920 + 150, y * TH / 1080 + 60, 280, heading=170, streak_len=1.8)
    d = ImageDraw.Draw(im)
    headline(d, (40, 36), "EVERY U-TURN =", 96, WHITE)
    headline(d, (40, TH - 40), "10 TSAR BOMBAS", 112, GOLD, "ls")
    save(im, "F")


def thumb_G():
    """the opening close-up: the robot on the 44th floor at true size, rows behind it, dust ahead"""
    Hk = V.Hook()
    t = 1.6
    fr, _ = Hk.city(t, Hk.L(t))
    # zoom on the robot: crop around it and scale up
    im = Image.fromarray(fr).crop((960 - 640, 690 - 430, 960 + 640, 690 + 290)).resize((TW, TH), Image.LANCZOS)
    im = hero_robot(im, 640, 470, 340, heading=72, streak_len=0.05)          # slowed down: no streak, just the robot
    d = ImageDraw.Draw(im)
    headline(d, (40, 36), "SLOWED 1,000,000,000×", 84, GOLD)
    headline(d, (40, TH - 40), "...IT'S AS FAST AS YOURS", 74, WHITE, "ls")
    save(im, "G")


def thumb_H():
    """versus: the robot against the whole planet"""
    S = V.Sweep()
    u = 22.0
    done = float(S.done(u))
    flon, flat, clon, clat = S.at(u)
    g = dict(cx=1340, cy=560, r=600, lon0=-20, lat0=25)
    fr = V.globe_img(g, done, light=(flon, flat), light_size=1.1)
    im = Image.fromarray(fr).resize((TW, TH), Image.LANCZOS)
    im = hero_robot(im, 300, 470, 340, heading=-20, streak_len=1.6)
    d = ImageDraw.Draw(im)
    headline(d, (40, 36), "ROBOT VACUUM", 100, WHITE)
    headline(d, (40, 150), "VS. EARTH", 130, GOLD)
    save(im, "H")


if __name__ == "__main__":
    for k in (sys.argv[1:] or list("ABCDEFGH")):
        globals()[f"thumb_{k}"]()
