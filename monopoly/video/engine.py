"""Render engine for the Monopoly video: scenes draw PIL frames, ffmpeg encodes them.

A scene knows its narration (one paragraph of script/narration_elevenlabs.txt) and its length. The length comes from
the voice-over file audio/<scene>.mp3 when it exists; until then it is estimated at 155 words a minute. Visual cues are
placed by phrase: cue("Lizzie Magie") is the moment that phrase starts, estimated from its word position, so the same
code re-times itself when the real audio arrives.
"""
import math
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'art'))
import style_frames as sf  # noqa: E402  (paper, ink, stamps, board, banknotes, fonts)

W, H, FPS = 1920, 1080, 30
WPM = 155
AUDIO = os.path.join(HERE, 'audio')
OUT = os.path.join(HERE, 'out')
SCRIPT = os.path.join(HERE, 'script', 'narration_elevenlabs.txt')
SCENE_FILES = ['01_cold_open', '02_the_promise', '03_how_we_tested_it', '04_buy_or_pass', '05_trading', '06_railroads',
               '07a_spend_or_save', '07b_how_far_to_build', '07c_which_set_first', '07d_one_set_or_spread', '07e_owe_money',
               '08_jail', '09_the_computers_player', '10_the_payoff', '11_the_playbook']


def narration(scene):
    paras = open(SCRIPT).read().strip().split('\n\n')
    return paras[SCENE_FILES.index(scene)]


def audio_length(scene):
    for ext in ('mp3', 'wav', 'm4a'):
        p = os.path.join(AUDIO, '%s.%s' % (scene, ext))
        if os.path.exists(p):
            out = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', p],
                                 capture_output=True, text=True, check=True).stdout
            return float(out), p
    return None, None


# ---------------------------------------------------------------- easing and helpers
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def smooth(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def ease_out(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def ramp(t, t0, dur):
    """0 before t0, 1 after t0 + dur, smooth in between."""
    return smooth((t - t0) / dur) if dur > 0 else float(t >= t0)


def lerp(a, b, x):
    return a + (b - a) * x


class Scene:
    lead_in = 0.4    # silence before the narration starts
    tail = 0.8       # hold after it ends

    def __init__(self, name):
        self.name = name
        self.text = narration(name)
        self.words = self.text.split()
        real, self.audio = audio_length(name)
        self.speech = real if real else len(self.words) / WPM * 60
        self.dur = self.lead_in + self.speech + self.tail
        self.grain = [np.random.default_rng(k).normal(0, 3.0, (H // 2, W // 2)).astype(np.float32) for k in range(6)]

    def cue(self, phrase, offset=0.0):
        """Seconds into the scene when `phrase` starts being spoken (estimated from word position)."""
        flat = ' '.join(self.words).lower()
        i = flat.find(phrase.lower())
        if i < 0:
            raise KeyError('phrase not in narration: %r' % phrase)
        k = len(flat[:i].split())
        return self.lead_in + self.speech * k / len(self.words) + offset

    def captions(self):
        """Sentence-sized caption chunks with start and end times."""
        import re
        parts = [p.strip() for p in re.split(r'(?<=[.?!])\s+', self.text) if p.strip()]
        out, k = [], 0
        for p in parts:
            n = len(p.split())
            t0 = self.lead_in + self.speech * k / len(self.words)
            t1 = self.lead_in + self.speech * (k + n) / len(self.words)
            out.append((t0, t1, p)); k += n
        return out

    def frame(self, t):  # pragma: no cover - each scene draws its own
        raise NotImplementedError

    # ------------------------------------------------------------ finishing: grain, weave, vignette
    def finish(self, img, i):
        a = np.asarray(img.convert('RGB'), np.float32)
        g = self.grain[i % len(self.grain)]
        g = np.repeat(np.repeat(g, 2, 0), 2, 1)
        a += g[..., None]
        out = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
        dx, dy = int(round(math.sin(i * 0.37) * 0.8)), int(round(math.cos(i * 0.29) * 0.8))  # gate weave
        if dx or dy:
            out = out.transform(out.size, Image.AFFINE, (1, 0, dx, 0, 1, dy), resample=Image.BILINEAR)
        return out

    def draw_captions(self, img, t):
        for t0, t1, s in self.captions():
            if t0 <= t < t1:
                f = sf.font('serif', 38)
                d = ImageDraw.Draw(img)
                lines, line = [], ''
                for w in s.split():
                    if d.textlength(line + ' ' + w, font=f) > 1500 and line:
                        lines.append(line); line = w
                    else:
                        line = (line + ' ' + w).strip()
                lines.append(line)
                txt = '\n'.join(lines)
                bb = d.multiline_textbbox((W / 2, H - 70), txt, font=f, anchor='md', align='center', spacing=8)
                band = Image.new('RGBA', img.size, (0, 0, 0, 0))
                ImageDraw.Draw(band).rounded_rectangle([bb[0] - 24, bb[1] - 14, bb[2] + 24, bb[3] + 16], 10, fill=(20, 16, 12, 170))
                img.alpha_composite(band)
                ImageDraw.Draw(img).multiline_text((W / 2, H - 70), txt, font=f, fill=(246, 236, 214, 255), anchor='md',
                                                  align='center', spacing=8)
                break

    def render(self, path=None, captions=False):
        os.makedirs(OUT, exist_ok=True)
        path = path or os.path.join(OUT, self.name + ('_preview' if captions else '') + '.mp4')
        n = int(round(self.dur * FPS))
        cmd = ['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '%dx%d' % (W, H), '-r', str(FPS), '-i', '-']
        if self.audio:
            cmd += ['-itsoffset', str(self.lead_in), '-i', self.audio, '-map', '0:v', '-map', '1:a', '-c:a', 'aac', '-b:a', '256k']
        cmd += ['-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p', '-t', '%.3f' % self.dur, path]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for i in range(n):
            t = i / FPS
            img = self.frame(t).convert('RGBA')
            if captions:
                self.draw_captions(img, t)
            p.stdin.write(self.finish(img, i).tobytes())
        p.stdin.close(); p.wait()
        return path


# ---------------------------------------------------------------- shared drawing pieces
class Camera:
    """A view onto a large world image: keyframes of (time, centre x, centre y, zoom); zoom 1 = world pixels 1:1."""

    def __init__(self, world, keys):
        self.world = world.convert('RGB')
        self.half = self.world.resize((self.world.size[0] // 2, self.world.size[1] // 2), Image.LANCZOS)
        self.keys = sorted(keys)

    def at(self, t):
        ks = self.keys
        if t <= ks[0][0]:
            return ks[0][1:]
        for (t0, *a), (t1, *b) in zip(ks, ks[1:]):
            if t <= t1:
                x = smooth((t - t0) / (t1 - t0)) if t1 > t0 else 1
                return tuple(lerp(u, v, x) for u, v in zip(a, b))
        return ks[-1][1:]

    def view(self, t):
        cx, cy, z = self.at(t)
        WW, WH = self.world.size
        w, h = min(W / z, WW), min(H / z, WH)
        cx, cy = clamp(cx, w / 2, WW - w / 2), clamp(cy, h / 2, WH - h / 2)   # keep the view on the world
        box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
        if z < 0.9:  # zoomed out: sample the half-size copy
            return self.half.resize((W, H), Image.BILINEAR, box=tuple(v / 2 for v in box)).convert('RGBA')
        return self.world.resize((W, H), Image.BILINEAR, box=box).convert('RGBA')


def txt(img, xy, s, fnt, fill=sf.INK, anchor='la', angle=0, alpha=255, spacing=4, align='left'):
    """Fast text: draw into a layer only as big as the text, then composite (sf.text uses a full-frame layer)."""
    if angle:
        return sf.text(img, xy, s, fnt, fill=fill, anchor=anchor, angle=angle, alpha=alpha, spacing=spacing, align=align)
    d = ImageDraw.Draw(img)
    bb = d.multiline_textbbox(xy, s, font=fnt, anchor=anchor, spacing=spacing, align=align)
    x0, y0 = int(math.floor(bb[0])) - 2, int(math.floor(bb[1])) - 2
    w, h = int(math.ceil(bb[2])) - x0 + 4, int(math.ceil(bb[3])) - y0 + 4
    if w <= 0 or h <= 0:
        return
    layer = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(layer).multiline_text((xy[0] - x0, xy[1] - y0), s, font=fnt, fill=tuple(fill) + (alpha,), anchor=anchor,
                                         spacing=spacing, align=align)
    img.alpha_composite(layer, (max(0, x0), max(0, y0)), (max(0, -x0), max(0, -y0)))


_STAMPS = {}


def stamp_layer(center, lines, angle, size, seed, color=sf.STAMP):
    """A landed stamp as a small layer plus its top-left corner, cached."""
    key = (tuple(lines), angle, size, seed, color)
    if key not in _STAMPS:
        big = Image.new('RGBA', (W, H), (0, 0, 0, 0))
        sf.stamp(big, (W // 2, H // 2), lines, angle=angle, size=size, seed=seed, color=color)
        bb = big.getbbox()
        _STAMPS[key] = (big.crop(bb), (bb[0] - W // 2, bb[1] - H // 2))
    lay, (dx, dy) = _STAMPS[key]
    return lay, (int(center[0] + dx), int(center[1] + dy))


_SLIPS = {}


def slip(img, xy, lines, t, t0, fnt=None, cps=30, width=760):
    """A typed paper slip pasted onto the frame: the slip drops in, then the text types itself."""
    if t < t0:
        return
    fnt = fnt or sf.font('type', 32)
    key = (tuple(lines), width)
    if key not in _SLIPS:
        h = 34 + 46 * len(lines)
        card = sf.paper(width, h, base=(244, 238, 222), seed=len(lines) + width)
        ImageDraw.Draw(card).rectangle([0, 0, width - 1, h - 1], outline=sf.INK + (60,), width=1)
        _SLIPS[key] = card
    card = _SLIPS[key].copy()
    shown, budget = [], (t - t0 - 0.15) * cps
    for ln in lines:
        k = int(clamp(budget, 0, len(ln))); budget -= len(ln)
        shown.append(ln[:k])
    for i, ln in enumerate(shown):
        if ln:
            txt(card, (22, 18 + 46 * i), ln, fnt)
    a = ease_out((t - t0) / 0.25)
    lay = card.rotate(-1.2, expand=True, resample=Image.BICUBIC)
    if a < 1:
        arr = np.asarray(lay.split()[-1], np.float32) * a
        lay.putalpha(Image.fromarray(arr.astype(np.uint8)))
    sf.paste_shadowed(img, lay, (xy[0], xy[1] + (1 - a) * 12), offset=(6, 8), blur=6, alpha=int(80 * a))


def fade(img, a, color=(18, 14, 10)):
    if a >= 1:
        return img
    base = Image.new('RGBA', img.size, color + (255,))
    return Image.blend(base, img.convert('RGBA'), clamp(a))


def typewriter(img, xy, s, fnt, t, t0, cps=28, fill=sf.INK, **kw):
    """Reveal `s` one character at a time from t0."""
    n = int(clamp((t - t0) * cps, 0, len(s)))
    if n > 0:
        txt(img, xy, s[:n], fnt, fill=fill, **kw)


def stamp_in(img, t, t0, center, lines, angle=-8, size=64, seed=5):
    """A rubber stamp landing: drops from slightly larger and lighter over 0.18 s."""
    if t < t0:
        return
    x = clamp((t - t0) / 0.18)
    lay, (px, py) = stamp_layer(center, lines, angle, size, seed)
    if x < 1:
        s = 1.25 - 0.25 * x
        w, h = lay.size
        lay = lay.resize((int(w * s), int(h * s)), Image.BILINEAR)
        px, py = int(center[0] - (center[0] - px) * s), int(center[1] - (center[1] - py) * s)
        a = np.asarray(lay.split()[-1], np.float32) * x
        lay = lay.copy(); lay.putalpha(Image.fromarray(a.astype(np.uint8)))
    img.alpha_composite(lay, (max(0, px), max(0, py)), (max(0, -px), max(0, -py)))


def shake(img, t, t0, amp=6, dur=0.25):
    if t0 <= t < t0 + dur:
        k = 1 - (t - t0) / dur
        dx, dy = amp * k * math.sin((t - t0) * 90), amp * k * math.cos((t - t0) * 70)
        return img.transform(img.size, Image.AFFINE, (1, 0, dx, 0, 1, dy), resample=Image.BILINEAR)
    return img


def die_face(n, size=120, seed=0):
    """An ivory die, ink pips, drawn in the period style."""
    s = size
    im = Image.new('RGBA', (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([3, 3, s - 4, s - 4], radius=s // 6, fill=(240, 232, 212, 255), outline=sf.INK + (255,), width=4)
    pos = {1: [(0.5, 0.5)], 2: [(0.27, 0.27), (0.73, 0.73)], 3: [(0.27, 0.27), (0.5, 0.5), (0.73, 0.73)],
           4: [(0.27, 0.27), (0.73, 0.27), (0.27, 0.73), (0.73, 0.73)],
           5: [(0.27, 0.27), (0.73, 0.27), (0.5, 0.5), (0.27, 0.73), (0.73, 0.73)],
           6: [(0.27, 0.25), (0.73, 0.25), (0.27, 0.5), (0.73, 0.5), (0.27, 0.75), (0.73, 0.75)]}[n]
    r = s * 0.085
    for x, y in pos:
        d.ellipse([x * s - r, y * s - r, x * s + r, y * s + r], fill=sf.INK + (255,))
    return im
