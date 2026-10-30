"""Captions in the black band below the picture: one line of plain words saying what is going on, faded in and out.
Words in [brackets] are drawn in the light's colour (slowmo.CYAN), so light reads as one thread through the film.
  cues = [(first_frame, end_frame, "text with [light] in it"), ...]; frame8 = draw(frame8, *at(cues, i))"""
import re
import numpy as np
from PIL import Image, ImageDraw
import comp as C, slowmo as SM

WHITE = (255, 255, 255)
Y_MID = C.TOP + C.PH + (C.H - C.TOP - C.PH) // 2      # the middle of the band below the picture


def _runs(text):
    """[(words, colour)]: bracketed runs in cyan"""
    out = []
    for part in re.split(r"(\[[^\]]*\])", text):
        if not part: continue
        out.append((part[1:-1], SM.CYAN) if part.startswith("[") else (part, WHITE))
    return out


def draw(frame8, text, alpha=1.0, size=38, y=Y_MID):
    if not text or alpha <= 0: return frame8
    im = Image.fromarray(frame8).convert("RGBA")
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    f = C.font(size, "SemiBold")
    runs = _runs(text)
    w = sum(d.textlength(t, font=f) for t, _ in runs)
    if w > C.W - 120:                                    # too long for one line at this size
        size = int(size * (C.W - 120) / w); f = C.font(size, "SemiBold")
        w = sum(d.textlength(t, font=f) for t, _ in runs)
    x = (C.W - w) / 2
    a = int(255 * alpha)
    for t, col in runs:
        d.text((x, y - size * 0.62), t, font=f, fill=col + (a,))
        x += d.textlength(t, font=f)
    return np.asarray(Image.alpha_composite(im, ov).convert("RGB"))


def at(cues, i, fade=7):
    """the caption showing at frame i, and its alpha (fading in and out over `fade` frames)"""
    for k0, k1, text in cues:
        if k0 <= i < k1:
            return text, min(1.0, (i - k0 + 1) / fade, (k1 - i) / fade)
    return None, 0.0
