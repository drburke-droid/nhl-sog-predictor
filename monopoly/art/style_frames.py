"""Style frames for the Monopoly decisions video: the 1904 patent drawing, Atlantic City, 1930s banking paper.

No Hasbro artwork: the board is our own drawing in the manner of L. J. Magie's public-domain 1904 patent; money,
certificates, tickets and telegrams are original designs in a period style.

    python3 style_frames.py        # writes frames/*.png (downloads the open-licence fonts on first run)
"""
import math
import os
import urllib.request

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONTS = os.path.join(HERE, 'fonts')
OUT = os.path.join(HERE, 'frames')
W, H = 1920, 1080

FONT_FILES = {
    'deco': 'ofl/limelight/Limelight-Regular.ttf',
    'type': 'apache/specialelite/SpecialElite-Regular.ttf',
    'hand': 'ofl/architectsdaughter/ArchitectsDaughter-Regular.ttf',
    'serif': 'ofl/oldstandardtt/OldStandard-Regular.ttf',
    'serifb': 'ofl/oldstandardtt/OldStandard-Bold.ttf',
    'script': 'ofl/pinyonscript/PinyonScript-Regular.ttf',
    'fat': 'ofl/abrilfatface/AbrilFatface-Regular.ttf',
    'sc': 'ofl/playfairdisplaysc/PlayfairDisplaySC-Regular.ttf',
    'scb': 'ofl/playfairdisplaysc/PlayfairDisplaySC-Bold.ttf',
}

# palette: aged paper, printer's inks, the eight colour groups as watercolour washes
PAPER = (239, 228, 204)
INK = (31, 26, 20)
SEPIA = (110, 78, 46)
STAMP = (163, 39, 31)
NOTE_GREEN = (46, 91, 74)
BRASS = (168, 131, 58)
GROUP = {'brown': (138, 90, 60), 'lightblue': (130, 178, 204), 'pink': (196, 104, 146), 'orange': (217, 138, 61),
         'red': (184, 71, 58), 'yellow': (214, 181, 70), 'green': (78, 138, 91), 'darkblue': (60, 90, 138)}

rng = np.random.default_rng(1935)


def font(name, size):
    path = os.path.join(FONTS, os.path.basename(FONT_FILES[name]))
    if not os.path.exists(path):
        os.makedirs(FONTS, exist_ok=True)
        urllib.request.urlretrieve('https://raw.githubusercontent.com/google/fonts/main/' + FONT_FILES[name], path)
    return ImageFont.truetype(path, size)


# ---------------------------------------------------------------- textures
def paper(w=W, h=H, base=PAPER, seed=0):
    r = np.random.default_rng(seed)
    img = np.ones((h, w, 3)) * np.array(base, float)
    low = Image.fromarray((r.random((h // 40 + 2, w // 40 + 2)) * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
    low = np.asarray(low.filter(ImageFilter.GaussianBlur(30)), float) / 255.0 - 0.5
    grain = r.normal(0, 1, (h, w))
    img += low[..., None] * np.array([26, 24, 20]) + grain[..., None] * 4.0
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    img *= (1 - 0.16 * np.clip(d - 0.55, 0, 1) ** 1.6)[..., None]
    fib = r.random((h, w)) > 0.9993  # paper fibres and specks
    img[fib] -= 35
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).convert('RGBA')


def ink_layer(size):
    return Image.new('RGBA', (int(math.ceil(size[0])), int(math.ceil(size[1]))), (0, 0, 0, 0))


def roughen(layer, amount=0.25, seed=3):
    """Uneven ink: knock random specks out of a layer's alpha."""
    a = np.asarray(layer.split()[-1], float)
    r = np.random.default_rng(seed)
    n = Image.fromarray((r.random((a.shape[0] // 3 + 1, a.shape[1] // 3 + 1)) * 255).astype(np.uint8)).resize(
        (a.shape[1], a.shape[0]), Image.BICUBIC)
    n = np.asarray(n.filter(ImageFilter.GaussianBlur(1.2)), float) / 255
    a *= np.clip((n - amount) / (1 - amount) * 1.6, 0.15, 1)
    out = layer.copy()
    out.putalpha(Image.fromarray(a.astype(np.uint8)))
    return out


def wobble_line(d, pts, fill=INK, width=2, amp=0.8, seed=0):
    """A hand-inked line through pts."""
    r = np.random.default_rng(seed)
    out = []
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        n = max(2, int(math.hypot(x1 - x0, y1 - y0) / 12))
        walk = np.cumsum(r.normal(0, amp, n)); walk -= np.linspace(walk[0], walk[-1], n)
        nx, ny = -(y1 - y0), (x1 - x0); L = math.hypot(nx, ny) or 1
        for i in range(n):
            t = i / (n - 1)
            out.append((x0 + (x1 - x0) * t + nx / L * walk[i], y0 + (y1 - y0) * t + ny / L * walk[i]))
    d.line(out, fill=fill, width=width, joint='curve')


def hatch(img, box, spacing=7, angle=45, color=INK, width=1, alpha=200):
    x0, y0, x1, y1 = [int(v) for v in box]
    layer = ink_layer((x1 - x0, y1 - y0)); d = ImageDraw.Draw(layer)
    w, h = layer.size
    t = math.tan(math.radians(angle))
    for c in range(-h, w + h, spacing):
        d.line([(c, 0), (c + h / t if t else c, h)], fill=color + (alpha,), width=width)
    img.alpha_composite(layer, (x0, y0))


def wash(img, box, color, alpha=150, seed=0):
    """Watercolour wash: soft, granular, darker at the edges."""
    x0, y0, x1, y1 = [int(v) for v in box]
    w, h = x1 - x0, y1 - y0
    r = np.random.default_rng(seed)
    a = np.ones((h, w)) * alpha
    g = Image.fromarray((r.random((h // 4 + 2, w // 4 + 2)) * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
    a *= 0.75 + 0.5 * np.asarray(g, float) / 255
    yy, xx = np.mgrid[0:h, 0:w]
    edge = np.minimum.reduce([xx, yy, w - 1 - xx, h - 1 - yy])
    a *= 1 + 0.35 * np.exp(-edge / 2.5)
    layer = Image.new('RGBA', (w, h), color + (0,))
    layer.putalpha(Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)))
    img.alpha_composite(layer, (x0, y0))


def text(img, xy, s, fnt, fill=INK, anchor='la', angle=0, alpha=255, spacing=4, align='left'):
    if angle == 0:
        layer = ink_layer(img.size); d = ImageDraw.Draw(layer)
        d.multiline_text(xy, s, font=fnt, fill=fill + (alpha,), anchor=anchor, spacing=spacing, align=align)
        img.alpha_composite(layer)
        return
    probe = ImageDraw.Draw(ink_layer((10, 10)))
    bb = probe.multiline_textbbox((0, 0), s, font=fnt, spacing=spacing, align=align)
    tw, th = bb[2] - bb[0] + 8, bb[3] - bb[1] + 8
    t = ink_layer((tw, th)); ImageDraw.Draw(t).multiline_text((4 - bb[0], 4 - bb[1]), s, font=fnt, fill=fill + (alpha,), spacing=spacing, align=align)
    t = t.rotate(angle, expand=True, resample=Image.BICUBIC)
    img.alpha_composite(t, (int(xy[0] - t.size[0] / 2), int(xy[1] - t.size[1] / 2)))


def stamp(img, center, lines, angle=-8, size=64, color=STAMP, seed=5):
    f = font('scb', size)
    probe = ImageDraw.Draw(ink_layer((10, 10)))
    widths = [probe.textbbox((0, 0), l, font=f)[2] for l in lines]
    tw = int(max(widths)) + 70; th = len(lines) * int(size * 1.15) + 50
    s = ink_layer((tw, th)); d = ImageDraw.Draw(s)
    d.rounded_rectangle([6, 6, tw - 6, th - 6], radius=14, outline=color + (235,), width=7)
    d.rounded_rectangle([18, 18, tw - 18, th - 18], radius=8, outline=color + (235,), width=2)
    y = 25
    for l, wdt in zip(lines, widths):
        d.text((int((tw - wdt) / 2), y), l, font=f, fill=color + (235,)); y += int(size * 1.15)
    s = roughen(s, 0.3, seed).rotate(angle, expand=True, resample=Image.BICUBIC)
    img.alpha_composite(s, (int(center[0] - s.size[0] / 2), int(center[1] - s.size[1] / 2)))


def guilloche_frame(img, inset=26, band=46, color=NOTE_GREEN, alpha=150):
    """Banknote-style border: interlaced sine ropes and corner rosettes."""
    layer = ink_layer(img.size); d = ImageDraw.Draw(layer)
    w, h = img.size
    x0, y0, x1, y1 = inset, inset, w - inset, h - inset
    c = color + (alpha,)
    for k in range(9):
        ph = k * math.pi / 9
        for side in range(4):
            pts = []
            if side in (0, 2):
                yb = y0 + band / 2 if side == 0 else y1 - band / 2
                for x in range(int(x0 + band), int(x1 - band) + 1, 3):
                    pts.append((x, yb + (band / 2 - 6) * math.sin(x / 22.0 + ph) * math.cos(x / 97.0)))
            else:
                xb = x0 + band / 2 if side == 3 else x1 - band / 2
                for y in range(int(y0 + band), int(y1 - band) + 1, 3):
                    pts.append((xb + (band / 2 - 6) * math.sin(y / 22.0 + ph) * math.cos(y / 97.0), y))
            d.line(pts, fill=c, width=1)
    for r_, wdt in ((0, 3), (band, 2)):
        d.rectangle([x0 + r_, y0 + r_, x1 - r_, y1 - r_], outline=color + (220,), width=wdt)
    for cx, cy in ((x0 + band / 2, y0 + band / 2), (x1 - band / 2, y0 + band / 2), (x0 + band / 2, y1 - band / 2), (x1 - band / 2, y1 - band / 2)):
        for k in range(14):
            pts = [(cx + (band * 0.42 + 9 * math.cos(7 * t + k * 0.45)) * math.cos(t),
                    cy + (band * 0.42 + 9 * math.cos(7 * t + k * 0.45)) * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 220)]
            d.line(pts, fill=c, width=1)
    img.alpha_composite(layer)


def patent_drawing(height):
    """The public-domain 1904 patent sheet, as ink on our paper."""
    src = Image.open(os.path.join(HERE, 'source', 'magie_patent_748626_1904.png')).convert('L')
    src = src.resize((int(src.size[0] * height / src.size[1]), height), Image.LANCZOS)
    a = 255 - np.asarray(src, float)
    a = np.clip((a - 25) * 1.4, 0, 255)
    layer = Image.new('RGBA', src.size, INK + (0,))
    layer.putalpha(Image.fromarray(a.astype(np.uint8)))
    return layer


def paste_shadowed(img, obj, xy, offset=(10, 14), blur=12, alpha=90):
    a = obj.split()[-1]
    sh = Image.new('RGBA', obj.size, (40, 28, 15, 0)); sh.putalpha(a.point(lambda v: v * alpha // 255))
    pad = blur * 3
    big = ink_layer((obj.size[0] + 2 * pad, obj.size[1] + 2 * pad)); big.alpha_composite(sh, (pad, pad))
    big = big.filter(ImageFilter.GaussianBlur(blur))
    img.alpha_composite(big, (int(xy[0] + offset[0] - pad), int(xy[1] + offset[1] - pad)))
    img.alpha_composite(obj, (int(xy[0]), int(xy[1])))


def pawn(img, cx, base_y, h=70, color=BRASS):
    layer = ink_layer(img.size); d = ImageDraw.Draw(layer)
    dark = tuple(int(v * 0.55) for v in color); light = tuple(min(255, int(v * 1.35)) for v in color)
    w = h * 0.62
    d.ellipse([cx - w / 2, base_y - h * 0.16, cx + w / 2, base_y + h * 0.06], fill=dark + (255,))
    d.polygon([(cx - w * 0.42, base_y - h * 0.08), (cx + w * 0.42, base_y - h * 0.08), (cx + w * 0.15, base_y - h * 0.62), (cx - w * 0.15, base_y - h * 0.62)], fill=color + (255,))
    d.ellipse([cx - h * 0.2, base_y - h, cx + h * 0.2, base_y - h * 0.6], fill=color + (255,), outline=dark + (255,), width=2)
    d.line([(cx - w * 0.05, base_y - h * 0.55), (cx - w * 0.22, base_y - h * 0.12)], fill=light + (200,), width=3)
    d.ellipse([cx - h * 0.1, base_y - h * 0.93, cx - h * 0.02, base_y - h * 0.85], fill=light + (220,))
    sh = ink_layer(img.size); ImageDraw.Draw(sh).ellipse([cx - w * 0.6, base_y - 6, cx + w * 0.75, base_y + 12], fill=(0, 0, 0, 70))
    img.alpha_composite(sh.filter(ImageFilter.GaussianBlur(4)))
    img.alpha_composite(layer)


# ---------------------------------------------------------------- the board, in the manner of the patent
SQUARES = ['WAGES\n$200', 'MEDITER-\nRANEAN\n$60', 'COMMUNITY\nCHEST', 'BALTIC\nAVE.\n$60', 'INCOME\nTAX\n$200', 'READING\nR.R.\n$200',
           'ORIENTAL\nAVE.\n$100', 'CHANCE', 'VERMONT\nAVE.\n$100', 'CONNECT-\nICUT\n$120', 'JAIL', 'ST.\nCHARLES\n$140', 'ELECTRIC\nCO.\n$150',
           'STATES\nAVE.\n$140', 'VIRGINIA\nAVE.\n$160', 'PENNA.\nR.R.\n$200', 'ST.\nJAMES\n$180', 'COMMUNITY\nCHEST', 'TENNES-\nSEE\n$180',
           'NEW\nYORK\n$200', 'PARKING', 'KENTUCKY\nAVE.\n$220', 'CHANCE', 'INDIANA\nAVE.\n$220', 'ILLINOIS\nAVE.\n$240', 'B. & O.\nR.R.\n$200',
           'ATLANTIC\nAVE.\n$260', 'VENTNOR\nAVE.\n$260', 'WATER\nWORKS\n$150', 'MARVIN\nGARDENS\n$280', 'GO TO JAIL', 'PACIFIC\nAVE.\n$300',
           'NO.\nCAROLINA\n$300', 'COMMUNITY\nCHEST', 'PENNA.\nAVE.\n$320', 'SHORT\nLINE\n$200', 'CHANCE', 'PARK\nPLACE\n$350', 'LUXURY\nTAX\n$100',
           'BOARD-\nWALK\n$400']
GRP = {1: 'brown', 3: 'brown', 6: 'lightblue', 8: 'lightblue', 9: 'lightblue', 11: 'pink', 13: 'pink', 14: 'pink', 16: 'orange',
       18: 'orange', 19: 'orange', 21: 'red', 23: 'red', 24: 'red', 26: 'yellow', 27: 'yellow', 29: 'yellow', 31: 'green', 32: 'green',
       34: 'green', 37: 'darkblue', 39: 'darkblue'}


def cell_box(i, X, Y, S, c):
    w = (S - 2 * c) / 9
    if i == 0: return (X + S - c, Y + S - c, X + S, Y + S), 0
    if i < 10: return (X + S - c - i * w, Y + S - c, X + S - c - (i - 1) * w, Y + S), 0
    if i == 10: return (X, Y + S - c, X + c, Y + S), 0
    if i < 20: return (X, Y + S - c - (i - 10) * w, X + c, Y + S - c - (i - 11) * w), -90
    if i == 20: return (X, Y, X + c, Y + c), 0
    if i < 30: return (X + c + (i - 21) * w, Y, X + c + (i - 20) * w, Y + c), 180
    if i == 30: return (X + S - c, Y, X + S, Y + c), 0
    return (X + S - c, Y + c + (i - 31) * w, X + S, Y + c + (i - 30) * w), 90


def draw_board(img, X, Y, S, highlight=None):
    c = S * 0.135
    d = ImageDraw.Draw(img)
    hf = font('hand', int(S * 0.0145))
    for i in range(40):
        (x0, y0, x1, y1), ang = cell_box(i, X, Y, S, c)
        g = GRP.get(i)
        if g:  # colour band on the inner edge, as a watercolour wash
            bw = c * 0.24
            band = {0: (x0, y0, x1, y0 + bw), -90: (x1 - bw, y0, x1, y1), 180: (x0, y1 - bw, x1, y1), 90: (x0, y0, x0 + bw, y1)}[ang]
            wash(img, band, GROUP[g], alpha=175 if (highlight is None or g == highlight) else 70, seed=i)
        if i in (0, 10, 20, 30):
            continue
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        off = {0: (0, c * 0.1), -90: (-c * 0.1, 0), 180: (0, -c * 0.1), 90: (c * 0.1, 0)}[ang] if g else (0, 0)
        text(img, (cx + off[0], cy + off[1]), SQUARES[i], hf, fill=INK, anchor='mm', angle=ang, align='center', spacing=2)
    # corners, in the patent's manner
    k = c
    # start: a globe, as in Magie's corner
    gx, gy, gr = X + S - k / 2, Y + S - k / 2, k * 0.40
    d.ellipse([gx - gr, gy - gr, gx + gr, gy + gr], outline=INK + (220,), width=2)
    d.ellipse([gx - gr * 0.78, gy - gr * 0.78, gx + gr * 0.78, gy + gr * 0.78], outline=INK + (150,), width=1)
    text(img, (gx, gy - gr * 0.18), 'WAGES', font('hand', int(S * 0.019)), anchor='mm')
    text(img, (gx, gy + gr * 0.25), '$200', font('hand', int(S * 0.019)), anchor='mm')
    text(img, (gx, gy - gr * 0.6), 'LABOR', font('hand', int(S * 0.0105)), anchor='mm')
    # jail: a barred box in the corner, "just visiting" round the edge
    jx0, jy0, jx1, jy1 = X + k * 0.36, Y + S - k + 3, X + k - 3, Y + S - k * 0.36
    for j in range(6):
        xx = jx0 + (jx1 - jx0) * (j + 0.5) / 6
        wobble_line(d, [(xx, jy0 + 4), (xx, jy1 - 4)], width=2, amp=0.4, seed=40 + j)
    plate = ink_layer((int(jx1 - jx0 - 14), int(S * 0.028))); ImageDraw.Draw(plate).rectangle([0, 0, plate.size[0] - 1, plate.size[1] - 1], fill=PAPER + (255,), outline=INK + (200,))
    img.alpha_composite(plate, (int(jx0 + 7), int((jy0 + jy1) / 2 - plate.size[1] / 2)))
    text(img, ((jx0 + jx1) / 2, (jy0 + jy1) / 2), 'IN JAIL', font('hand', int(S * 0.0145)), anchor='mm')
    d.line([(jx0, jy0), (jx0, jy1), (jx1, jy1)], fill=INK + (220,), width=2)
    text(img, (X + k * 0.62, Y + S - k * 0.17), 'JUST VISITING', font('hand', int(S * 0.0125)), anchor='mm')
    text(img, (X + k / 2, Y + k / 2), 'PUBLIC\nPARK\n(free\nparking)', font('hand', int(S * 0.016)), anchor='mm', align='center')
    d.ellipse([X + k * 0.12, Y + k * 0.12, X + k * 0.88, Y + k * 0.88], outline=INK + (200,), width=2)
    text(img, (X + S - k / 2, Y + k / 2), 'NO\nTRESPASS-\nING\nGO TO JAIL', font('hand', int(S * 0.0135)), anchor='mm', align='center')
    d.ellipse([X + S - k * 0.9, Y + k * 0.1, X + S - k * 0.1, Y + k * 0.9], outline=INK + (200,), width=3)
    # the grid, hand-inked
    lines = [[(X, Y), (X + S, Y), (X + S, Y + S), (X, Y + S), (X, Y)],
             [(X + c, Y + c), (X + S - c, Y + c), (X + S - c, Y + S - c), (X + c, Y + S - c), (X + c, Y + c)]]
    w = (S - 2 * c) / 9
    for i in range(10):
        lines.append([(X + c + i * w, Y), (X + c + i * w, Y + c)])
        lines.append([(X + c + i * w, Y + S - c), (X + c + i * w, Y + S)])
        lines.append([(X, Y + c + i * w), (X + c, Y + c + i * w)])
        lines.append([(X + S - c, Y + c + i * w), (X + S, Y + c + i * w)])
    for n, l in enumerate(lines):
        wobble_line(d, l, width=3 if n < 2 else 2, seed=n)
    return c


# ---------------------------------------------------------------- frames
def frame_title():
    img = paper(seed=1)
    guilloche_frame(img)
    pd = patent_drawing(860)
    pd = pd.rotate(2.2, expand=True, resample=Image.BICUBIC)
    img.alpha_composite(pd, (120, 100))
    for tx, ty, ang in ((96, 70, 35), (628, 92, -35)):  # tape on the sheet's top corners
        t = ink_layer((150, 44)); ImageDraw.Draw(t).rectangle([0, 0, 150, 44], fill=(232, 220, 190, 175))
        img.alpha_composite(t.rotate(ang, expand=True), (tx, ty))
    text(img, (1010, 205), 'A STUDY OF THE LANDLORD’S GAME', font('sc', 34), fill=SEPIA)
    text(img, (1004, 262), 'THE DECISIONS\nYOU CONTROL', font('deco', 100), fill=INK, spacing=12)
    d = ImageDraw.Draw(img)
    wobble_line(d, [(1010, 545), (1760, 545)], width=3, seed=9)
    text(img, (1010, 575), 'You can’t choose where you land.\nYou can choose what you do next.', font('serif', 46), fill=INK, spacing=12)
    text(img, (1010, 760), '13,400,000 GAMES  ·  ATLANTIC CITY, N.J.\nAFTER L. J. MAGIE, PATENT No. 748,626 (1904)', font('type', 28), fill=SEPIA, spacing=10)
    stamp(img, (1590, 925), ['3 WINS FOR', 'EVERY 2'], angle=-9, size=50)
    return img


def frame_board():
    img = paper(seed=2)
    S = 900; X, Y = 110, 90
    draw_board(img, X, Y, S, highlight=None)
    # centre annotations, as on a patent sheet
    text(img, (X + S / 2, Y + S * 0.50), 'Fig. 1.', font('serifb', 64), anchor='mm')
    text(img, (X + S / 2, Y + S * 0.60), 'THE BOARD AS PLAYED', font('sc', 30), fill=SEPIA, anchor='mm')
    text(img, (X + S / 2, Y + S * 0.67), 'squares named for the streets of\nAtlantic City, New Jersey', font('serif', 26), anchor='mm', align='center')
    # a pawn on Illinois with a draftsman's leader line into the open centre
    (x0, y0, x1, y1), _ = cell_box(24, X, Y, S, S * 0.135)
    pawn(img, (x0 + x1) / 2, y0 + 72, h=62)
    d = ImageDraw.Draw(img)
    lx = (x0 + x1) / 2
    wobble_line(d, [(lx, y1 + 8), (lx, Y + S * 0.255), (lx + 40, Y + S * 0.255)], width=2, seed=77)
    text(img, (lx + 48, Y + S * 0.228), 'ILLINOIS AVE.\n3.19% of rolls end here', font('hand', 22), fill=STAMP)
    # right-hand panel: patent header and notes
    px = 1110
    text(img, (px, 110), 'No. 13,400,000.', font('serifb', 38))
    text(img, (1840, 110), 'SIMULATED 2026.', font('serifb', 38), anchor='ra')
    text(img, ((px + 1840) / 2, 190), 'THE PROPERTY GAME.', font('serifb', 38), anchor='mm')
    text(img, ((px + 1840) / 2, 240), 'DECISIONS OF THE PLAYER.', font('serif', 30), anchor='mm')
    notes = ('1. To buy, or to pass.\n2. To trade for a full set.\n3. To collect the railroads.\n'
             '4. To build: where, how far, and with\n    how much cash held back.\n5. To leave jail, or to stay.')
    text(img, (px + 30, 330), notes, font('serif', 34), spacing=18)
    text(img, (px, 760), 'Witnesses', font('script', 54))
    text(img, (px + 30, 835), 'The Typical Player', font('script', 44), fill=SEPIA)
    text(img, (px + 30, 900), 'The Playbook Player', font('script', 44), fill=SEPIA)
    text(img, (px + 520, 760), 'Rules', font('script', 54))
    text(img, (px + 470, 840), 'official, with auctions;\nendgame deals when\nthe board locks up', font('hand', 24), fill=SEPIA)
    return img


def frame_ledger():
    img = paper(base=(240, 233, 212), seed=3)
    layer = ink_layer(img.size); d = ImageDraw.Draw(layer)
    for y in [240 + k * 93 for k in range(9)]:  # ledger ruling, one rule under each row
        d.line([(80, y), (1840, y)], fill=(120, 150, 170, 110), width=1)
    for x in (150, 560, 1180, 1460):
        d.line([(x, 160), (x, 1040)], fill=(60, 120, 90, 120), width=2)
    d.line([(152, 160), (152, 1040)], fill=(170, 60, 50, 140), width=2)
    img.alpha_composite(layer)
    text(img, (170, 70), 'LEDGER  —  DECISION No. 4', font('scb', 40), fill=NOTE_GREEN)
    text(img, (170, 125), 'Spend it on houses, or keep some back?', font('serif', 34))
    hdr = font('sc', 26)
    text(img, (170, 190), 'Cash kept back', hdr, fill=SEPIA)
    text(img, (580, 190), 'Games won, per hundred', hdr, fill=SEPIA)
    text(img, (1200, 190), 'First to go broke', hdr, fill=SEPIA)
    rows = [('$0', 25.1, '25.1%'), ('$50', 23.6, '25.5%'), ('$100', 22.4, '25.8%'), ('$200', 19.3, '26.7%'),
            ('$300', 16.3, '27.8%'), ('$500', 10.9, '29.9%'), ('$1,000', 3.2, '33.5%')]
    tf = font('type', 40)
    for k, (cash, win, out) in enumerate(rows):
        y = 250 + k * 93
        text(img, (180, y), cash, tf)
        bx0, bx1 = 590, 590 + win / 26 * 480
        hatch(img, (bx0, y + 6, bx1, y + 44), spacing=6, angle=60, alpha=210 if k == 0 else 150)
        dd = ImageDraw.Draw(img); dd.rectangle([bx0, y + 6, bx1, y + 44], outline=INK + (230,), width=2)
        text(img, (bx1 + 16, y), '%.1f' % win, tf)
        text(img, (1210, y), out, tf, fill=STAMP if k >= 4 else INK)
    text(img, (1480, 300), 'keeping cash back\nmakes you MORE\nlikely to go broke\nfirst, not less', font('hand', 34), fill=STAMP, spacing=10)
    stamp(img, (1640, 820), ['SPEND IT'], angle=-11, size=86)
    text(img, (170, 1000), 'One player against three typical players, 100,000 games a line.', font('type', 22), fill=SEPIA)
    return img


def railroad_ticket(name, seed):
    w, h = 270, 440
    t = paper(w, h, base=(226, 205, 160), seed=seed)
    d = ImageDraw.Draw(t)
    d.rectangle([10, 10, w - 10, h - 10], outline=INK + (230,), width=3)
    d.rectangle([20, 20, w - 20, h - 20], outline=INK + (180,), width=1)
    text(t, (w / 2, 62), 'GOOD FOR ONE PASSAGE', font('sc', 17), fill=SEPIA, anchor='mm')
    text(t, (w / 2, 165), name, font('fat', 36), anchor='mm', align='center', spacing=6)
    text(t, (w / 2, 290), 'ATLANTIC CITY', font('type', 22), anchor='mm')
    text(t, (w / 2, 325), 'RENT $25 · 50 · 100 · 200', font('type', 17), anchor='mm', fill=SEPIA)
    text(t, (w / 2, 385), 'No. %05d' % (seed * 7919 % 99999), font('type', 22), anchor='mm', fill=STAMP)
    for y in range(30, h - 30, 18):  # perforated edge
        d.ellipse([w - 8, y, w + 2, y + 8], fill=(0, 0, 0, 0))
    d.ellipse([w / 2 - 12, 226, w / 2 + 12, 250], fill=(239, 228, 204, 255), outline=INK + (180,), width=1)  # ticket punch
    return t


def frame_railroads():
    img = paper(seed=4)
    text(img, (110, 70), 'DECISION No. 3  —  THE RAILROADS', font('scb', 44), fill=NOTE_GREEN)
    for k, nm in enumerate(['READING\nR.R.', 'PENNA.\nR.R.', 'B. & O.\nR.R.', 'SHORT\nLINE']):
        t = railroad_ticket(nm, 11 + k).rotate([-5, 3, -2, 6][k], expand=True, resample=Image.BICUBIC)
        paste_shadowed(img, t, (90 + k * 238, 190 + [0, 30, 10, 40][k]))
    # telegram
    tg = paper(820, 360, base=(236, 226, 196), seed=9)
    d = ImageDraw.Draw(tg)
    text(tg, (40, 30), 'TELEGRAM', font('fat', 52), fill=INK)
    text(tg, (430, 48), 'ATLANTIC CITY N.J.  1935', font('type', 24), fill=SEPIA)
    d.line([(30, 110), (790, 110)], fill=INK + (200,), width=2)
    strips = ['WILL GIVE YOU KENTUCKY AVE STOP', 'YOU FINISH THE REDS STOP', 'I GET SHORT LINE AND ALL FOUR', 'RAILROADS STOP DEAL']
    for k, s in enumerate(strips):
        strip = ink_layer((760, 46)); ImageDraw.Draw(strip).rectangle([0, 0, 760, 46], fill=(250, 246, 232, 255))
        img_s = strip; text(img_s, (12, 6), s, font('type', 30))
        tg.alpha_composite(img_s, (30, 130 + k * 54))
    tg = tg.rotate(-3, expand=True, resample=Image.BICUBIC)
    paste_shadowed(img, tg, (1010, 600))
    text(img, (1150, 200), 'Buying railroads on their own\nalmost never gets you all four.\n\nTrading for the fourth does:\nall four in more than 1 game in 4.', font('serif', 36), spacing=12)
    stamp(img, (520, 860), ['ONLY IF', 'YOU TRADE'], angle=-7, size=62)
    return img


def frame_endcard():
    img = paper(base=(236, 229, 210), seed=5)
    guilloche_frame(img, inset=30, band=56)
    text(img, (W / 2, 150), 'THE PLAYBOOK', font('deco', 110), anchor='mm')
    text(img, (W / 2, 240), 'CERTIFICATE OF BETTER DECISIONS', font('sc', 34), fill=SEPIA, anchor='mm')
    rules = ['Land on it?  Buy it, unless it would leave you nearly broke.',
             'At auction, bid up to one and a half times the price.',
             'Make every fair set-for-set swap.  Don’t haggle.  When it stalls, deal.',
             'Want the railroads?  Trade for the last one, or don’t bother.',
             'Spend your cash on houses, and go to hotels.  Owe money?  Mortgage first.',
             'Two sets?  Cheaper houses first.  Never one house on everything.',
             'Leave jail early.  Stay in once anyone has houses.']
    f = font('serif', 36); nf = font('serifb', 36)
    for k, r in enumerate(rules):
        y = 320 + k * 78
        text(img, (330, y), '%s.' % 'I II III IV V VI VII'.split()[k], nf, fill=NOTE_GREEN, anchor='ra')
        text(img, (360, y), r, f)
    # seal
    layer = ink_layer(img.size); d = ImageDraw.Draw(layer)
    cx, cy, R = 1620, 860, 112
    for k in range(36):
        a = k * math.pi / 18
        d.polygon([(cx + R * math.cos(a), cy + R * math.sin(a)), (cx + (R + 18) * math.cos(a + 0.087), cy + (R + 18) * math.sin(a + 0.087)),
                   (cx + R * math.cos(a + 0.175), cy + R * math.sin(a + 0.175))], fill=STAMP + (230,))
    d.ellipse([cx - R, cy - R, cx + R, cy + R], fill=STAMP + (230,))
    d.ellipse([cx - R + 14, cy - R + 14, cx + R - 14, cy + R - 14], outline=(240, 220, 200, 200), width=2)
    img.alpha_composite(roughen(layer, 0.04, 21))
    text(img, (cx, cy - 12), '3 : 2', font('fat', 72), fill=(246, 234, 214), anchor='mm')
    text(img, (cx, cy + 50), 'WINS', font('scb', 30), fill=(246, 234, 214), anchor='mm')
    text(img, (300, 850), 'Same dice.', font('script', 64))
    text(img, (300, 935), 'Three wins for every two of theirs. A year of weekly game nights: champion two years in three.', font('type', 24), fill=SEPIA)
    return img


def thumbnail():
    img = paper(1280, 720, seed=6)
    S = 1500
    sub = paper(S + 200, S + 200, seed=7)
    draw_board(sub, 100, 100, S, highlight='darkblue')
    (x0, y0, x1, y1), _ = cell_box(39, 100, 100, S, S * 0.135)
    pawn(sub, x0 - 80, y1 - 25, h=120)
    crop = sub.crop((100 + S - 660, 100 + S - 760, 100 + S + 40, 100 + S + 40)).rotate(3, resample=Image.BICUBIC)
    paste_shadowed(img, crop, (-20, -60), offset=(14, 14))
    text(img, (735, 60), 'SAVE IT', font('fat', 118), fill=(120, 110, 95))
    d = ImageDraw.Draw(img)
    wobble_line(d, [(722, 150), (1195, 112)], fill=STAMP, width=12, amp=1.5, seed=3)
    stamp(img, (975, 390), ['SPEND IT'], angle=-8, size=108)
    text(img, (742, 585), '3 wins for every 2', font('deco', 54))
    return img


def tally(img, x, y, count, h=46, gap=13, group_gap=26, color=INK, seed=0):
    """Hand-drawn tally marks: groups of four strokes crossed by a fifth."""
    d = ImageDraw.Draw(img)
    k = 0
    for g in range(count // 5 + 1):
        n = min(5, count - g * 5)
        if n <= 0:
            break
        gx = x + g * (4 * gap + group_gap)
        for j in range(min(n, 4)):
            wobble_line(d, [(gx + j * gap, y), (gx + j * gap + 2, y + h)], fill=color, width=3, amp=0.6, seed=seed + k); k += 1
        if n == 5:
            wobble_line(d, [(gx - 6, y + h * 0.8), (gx + 3 * gap + 8, y + h * 0.2)], fill=color, width=3, amp=0.6, seed=seed + k); k += 1
    return x + (count // 5) * (4 * gap + group_gap) + (count % 5) * gap


def frame_game_nights():
    img = paper(base=(240, 233, 212), seed=8)
    layer = ink_layer(img.size); d = ImageDraw.Draw(layer)
    for y in [300 + k * 120 for k in range(5)]:
        d.line([(80, y), (1840, y)], fill=(120, 150, 170, 110), width=1)
    d.line([(152, 170), (152, 1040)], fill=(170, 60, 50, 140), width=2)
    img.alpha_composite(layer)
    text(img, (170, 60), 'GAME NIGHT LEDGER', font('scb', 44), fill=NOTE_GREEN)
    text(img, (170, 118), 'One hundred game nights, the same four players. You follow the playbook.', font('serif', 32))
    rows = [('YOU', 33, STAMP), ('MOM', 22, INK), ('DAD', 22, INK), ('JORDAN', 23, INK)]
    for k, (who, n, col) in enumerate(rows):
        y = 200 + k * 120
        text(img, (180, y + 22), who, font('type', 44), fill=col)
        end = tally(img, 450, y + 18, n, color=col, seed=100 * k)
        text(img, (end + 40, y + 22), str(n), font('type', 48), fill=col)
    text(img, (180, 720), 'Three wins for every two of theirs.', font('serifb', 44))
    text(img, (180, 800), 'Play every week for a year and you finish as family champion', font('serif', 36))
    text(img, (180, 850), 'two years in three. Each of them: about one year in eleven.', font('serif', 36))
    stamp(img, (1590, 900), ['FAMILY', 'CHAMPION'], angle=-8, size=64)
    text(img, (180, 1000), 'Wins per 100 games: 32.6 vs 22.3, 22.5, 22.5. Champion = most wins outright over 52 weekly games (64.8% vs 8.5-8.9% each).',
         font('type', 20), fill=SEPIA)
    return img


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    for name, fn in (('01_title', frame_title), ('02_board', frame_board), ('03_ledger_spend_or_save', frame_ledger),
                     ('04_railroads_telegram', frame_railroads), ('05_end_card', frame_endcard), ('06_thumbnail', thumbnail),
                     ('07_game_night_ledger', frame_game_nights)):
        fn().convert('RGB').save(os.path.join(OUT, name + '.png'))
        print('wrote', name)
