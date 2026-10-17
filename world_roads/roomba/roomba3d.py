"""A robot vacuum in three-quarter view, ray-traced in numpy (for the thumbnails).

The body is a short cylinder (34 cm across, 9 cm tall): a glossy charcoal top with a silver faceplate and a
lit centre button, a silver bumper around the front half with a dark sensor window, a side brush poking
out at the front right, and a soft shadow on the ground.

    sprite(width_px, heading_deg) -> RGBA PIL image; heading is where the bumper points on screen
    (0 = right, 90 = up).
"""
import math
import numpy as np
from PIL import Image, ImageDraw

H = 0.53                     # height / radius (9 cm / 17 cm)
ZB = 0.05                    # ground clearance
ELEV = math.radians(34)      # camera elevation


def _norm(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def sprite(width, heading=20.0, led=(255, 196, 96), ss=3):
    """Robot vacuum seen from 34 degrees up, its front turned `heading` degrees (screen, counter-clockwise from right)."""
    # camera azimuth chosen so the bumper points along `heading` on screen and faces the viewer a little
    az = math.radians(-55.0)
    f = -_norm([math.cos(ELEV) * math.cos(az), math.cos(ELEV) * math.sin(az), math.sin(ELEV)])     # view direction
    rgt = _norm(np.cross(f, [0, 0, 1]))
    up = np.cross(rgt, f)
    # rotate the body so its front (+x) projects to `heading` on screen
    fx = np.array([rgt @ [1, 0, 0], up @ [1, 0, 0]])
    rot = math.radians(heading) - math.atan2(fx[1], fx[0])
    c, s = math.cos(rot), math.sin(rot)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])        # body -> world

    n = int(width * ss)
    scale = n / 3.0                                           # 3 radii across the sprite
    hgt = int(n * 0.86)
    cx, cy = n / 2, hgt * 0.5
    jj, ii = np.mgrid[0:hgt, 0:n]
    u, v = (ii - cx) / scale, (cy - jj) / scale + 0.18
    O = u[..., None] * rgt + v[..., None] * up - 8 * f        # ray origins (world)
    Ob = O @ R                                                # in body coordinates (R orthonormal)
    fb = f @ R
    zt = ZB + H
    # top cap
    t_top = (zt - Ob[..., 2]) / fb[2]
    P_top = Ob + t_top[..., None] * fb
    r_top = np.hypot(P_top[..., 0], P_top[..., 1])
    hit_top = r_top <= 1.0
    # side wall
    a = fb[0] ** 2 + fb[1] ** 2
    b = 2 * (Ob[..., 0] * fb[0] + Ob[..., 1] * fb[1])
    cc = Ob[..., 0] ** 2 + Ob[..., 1] ** 2 - 1
    disc = b * b - 4 * a * cc
    t_side = (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a)
    P_side = Ob + t_side[..., None] * fb
    hit_side = (disc >= 0) & (P_side[..., 2] >= ZB) & (P_side[..., 2] <= zt)
    use_top = hit_top & (~hit_side | (t_top <= t_side))
    use_side = hit_side & ~use_top
    body = use_top | use_side

    L = _norm(-0.5 * rgt + 0.25 * up + 0.85 * np.array([0, 0, 1]) - 0.2 * f)     # key light, from the upper left
    Lb = L @ R
    V = -fb
    Hh = _norm(Lb + V)
    col = np.zeros((hgt, n, 3))
    alpha = np.zeros((hgt, n))

    # ---- top: charcoal, silver faceplate, lit button, front sensor dome, bin seam, rim highlight
    th = np.arctan2(P_top[..., 1], P_top[..., 0])
    alb = np.zeros((hgt, n, 3)) + [40, 42, 47]
    spec_k = np.full((hgt, n), 0.55)
    face = (r_top > 0.22) & (r_top < 0.47)
    alb[face] = [168, 171, 177]
    spec_k[face] = 0.35
    alb[(r_top >= 0.47) & (r_top < 0.50)] = [70, 72, 78]
    btn = r_top <= 0.17
    alb[btn] = [214, 216, 220]
    ring = (r_top > 0.17) & (r_top <= 0.22)
    dome = np.hypot(P_top[..., 0] - 0.80, P_top[..., 1]) < 0.09
    alb[dome] = [18, 18, 22]
    spec_k[dome] = 1.0
    seam = (np.abs(P_top[..., 0] + 0.58) < 0.012) & (r_top > 0.5)
    alb[seam] = [18, 18, 20]
    rim = r_top > 0.955
    alb[rim] = [92, 95, 102]
    ndl = max(0.0, Lb[2])
    shade = 0.42 + 0.7 * ndl
    # a soft window reflection across the glossy lid, brightest toward the upper left
    refl = np.exp(-(((u + 0.35) * 0.9 + (v - 0.55) * 0.5) ** 2) / 0.10) * 0.0 + np.clip(0.55 - 0.45 * (u - v * 0.6), 0, 1)
    spec = spec_k * (max(0.0, Hh[2]) ** 40 * 255 + 70 * refl)
    top_col = alb * shade + spec[..., None]
    top_col[ring] = np.array(led, float) * 1.05                  # the lit button ring glows (no shading)
    col[use_top] = top_col[use_top]

    # ---- side: silver bumper over the front half (with a dark sensor window), charcoal at the back
    ths = np.arctan2(P_side[..., 1], P_side[..., 0])
    z = P_side[..., 2]
    Nx, Ny = np.cos(ths), np.sin(ths)
    ndl_s = np.maximum(0.0, Nx * Lb[0] + Ny * Lb[1] + 0 * Lb[2])
    ndh_s = np.maximum(0.0, Nx * Hh[0] + Ny * Hh[1])
    alb_s = np.zeros((hgt, n, 3)) + [52, 54, 60]
    ks = np.full((hgt, n), 0.35)
    bump = (np.abs(ths) < math.radians(98)) & (z > ZB + 0.07) & (z < zt - 0.035)
    alb_s[bump] = [176, 180, 188]
    ks[bump] = 0.45
    win = (np.abs(ths) < math.radians(38)) & (z > zt - 0.20) & (z < zt - 0.075)
    alb_s[win] = [16, 16, 20]
    ks[win] = 1.0
    seam_s = bump & (np.abs(np.abs(ths) - math.radians(98)) < 0.02)
    alb_s[seam_s] = [20, 20, 22]
    alb_s[(z > zt - 0.035)] = [70, 72, 78]                       # the top's edge band
    low = z < ZB + 0.07
    alb_s[low] = [30, 31, 34]
    ndv_s = np.maximum(0.0, Nx * V[0] + Ny * V[1])                        # fill light from the camera side
    side_col = alb_s * (0.42 + 0.55 * ndl_s + 0.45 * ndv_s)[..., None] + (ks * ndh_s ** 30 * 255)[..., None]
    col[use_side] = side_col[use_side]
    alpha[body] = 1.0

    # ---- ground shadow (soft, offset away from the light)
    t_g = (0 - Ob[..., 2]) / fb[2]
    G = Ob + t_g[..., None] * fb
    sh = np.array([-Lb[0], -Lb[1]]) * 0.25
    rg = np.hypot(G[..., 0] - sh[0], G[..., 1] - sh[1])
    sa = np.clip((1.30 - rg) / 0.45, 0, 1) * 0.6
    col[~body] = 0
    alpha[~body] = sa[~body]

    img = Image.fromarray(np.dstack([np.clip(col, 0, 255), alpha * 255]).astype(np.uint8), "RGBA")
    # ---- side brush under the front right (drawn behind the body: only the tips show)
    br = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(br)

    def proj(p):
        w = R @ np.asarray(p, float)
        return (cx + (w @ rgt) * scale, cy - ((w @ up) - 0.18) * scale)

    hub = np.array([0.66, -0.56, 0.035])
    for k in range(3):
        a0 = math.radians(25 + 120 * k)
        for dt in np.linspace(-0.13, 0.13, 7):
            tip = hub + 0.52 * np.array([math.cos(a0 + dt), math.sin(a0 + dt), -0.02 / 0.52])
            d.line([proj(hub), proj(tip)], fill=(205, 205, 200, 255), width=max(1, int(ss * width / 420)))
    out = Image.alpha_composite(br, img)
    # "CLEAN" on the button, laid flat on the lid (squashed like the button itself)
    try:
        from PIL import ImageFont
        fnt = ImageFont.truetype("/usr/share/fonts/opentype/inter/Inter-Bold.otf", int(0.085 * scale))
        lab = Image.new("RGBA", (int(0.5 * scale), int(0.16 * scale)), (0, 0, 0, 0))
        ImageDraw.Draw(lab).text((lab.width / 2, lab.height / 2), "CLEAN", font=fnt, fill=(70, 74, 82, 255), anchor="mm")
        sq = up @ np.array([0, 0, 1])                                       # how much the lid is foreshortened
        lab = lab.resize((lab.width, max(1, int(lab.height * abs(f[2])))), Image.LANCZOS)
        bx, by = proj(np.array([0, 0, zt]))
        out.alpha_composite(lab, (int(bx - lab.width / 2), int(by - lab.height / 2)))
    except OSError:
        pass
    # the light ring glows a little beyond the button
    return out.resize((width, int(hgt / ss)), Image.LANCZOS)


if __name__ == "__main__":
    bg = Image.new("RGBA", (1400, 560), (5, 8, 15, 255))
    for i, hd in enumerate((20, 160, 60)):
        sp = sprite(440, hd)
        bg.alpha_composite(sp, (20 + i * 460, 120))
    bg.convert("RGB").save("out/roomba3d_test.png")
