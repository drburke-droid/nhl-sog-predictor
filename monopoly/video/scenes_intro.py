"""Scenes 01-02: the patent cold open and the promise.

    python3 scenes_intro.py 01_cold_open [--captions]
"""
import math
import os
import sys

import numpy as np
from PIL import Image

from engine import (FPS, H, W, Camera, Scene, clamp, die_face, ease_out, fade, lerp, ramp, sf, shake, slip, smooth,
                    stamp_in, txt, typewriter)

PATENT_SRC = (1805, 2809)  # pixel size of the patent sheet as scanned


class ColdOpen(Scene):
    """The 1904 patent on a desk; the camera visits the date, Magie's signature, the board and her jail corner."""

    def __init__(self):
        super().__init__('01_cold_open')
        WW, WH = 3840, 2160
        self.world = sf.paper(WW, WH, seed=21)
        sheet_h = 2040
        self.k = sheet_h / PATENT_SRC[1]
        pd = sf.patent_drawing(sheet_h)
        self.ox, self.oy = (WW - pd.size[0]) // 2, 60
        sheet = sf.paper(pd.size[0] + 60, pd.size[1] + 40, base=(242, 234, 214), seed=22)
        sheet.alpha_composite(pd, (30, 10))
        sf.paste_shadowed(self.world, sheet, (self.ox - 30, self.oy - 10), offset=(14, 18), blur=18, alpha=110)
        P = self.p
        full = (WW / 2, WH / 2, 0.5)
        self.cam = Camera(self.world, [
            (0.0, *P(1300, 2470), 1.35),                              # Magie's signature as her name is spoken
            (self.cue('was granted') - 0.2, *P(1300, 2470), 1.5),
            (self.cue('for a board game'), *P(1300, 150), 1.55),      # up to "PATENTED JAN. 5, 1904"
            (self.cue('She called it') - 0.1, *P(1240, 170), 1.6),
            (self.cue('She called it') + 0.9, *full),
            (self.cue('She designed it') + 0.8, *full),
            (self.cue('while everyone else') + 0.4, *P(300, 2030), 1.35),
            (self.cue('Thirty years later') + 0.6, *P(300, 830), 1.4),
            (self.cue('You know it as') + 0.2, *P(300, 830), 1.45),
            (self.dur - 0.3, *full),
        ])

    def p(self, x, y):
        """Patent-sheet pixel -> world pixel."""
        return self.ox + x * self.k, self.oy + y * self.k

    def frame(self, t):
        img = self.cam.view(t).convert('RGBA')
        if t < self.cue('Thirty years later'):
            slip(img, (70, 70), ['U.S. PATENT No. 748,626', 'GRANTED JANUARY 5, 1904'], t, self.cue('was granted') - 0.1, width=560)
        if self.cue('She called it') < t < self.cue('Thirty years later'):
            slip(img, (70, 230), ['THE LANDLORD’S GAME', 'BY LIZZIE J. MAGIE'], t, self.cue('She called it') + 0.2, width=560)
        if t >= self.cue('You know it as') + 0.6:
            a = ramp(t, self.cue('You know it as') + 0.6, 0.5)
            txt(img, (W / 2, H / 2 + 330), 'You know it as Monopoly.', sf.font('serif', 56), fill=sf.INK, anchor='mm', alpha=int(255 * a))
        return fade(img, ramp(t, 0, 1.2))

    def extra_sfx(self):
        ev = []
        for t0, lines in ((self.cue('was granted') - 0.1, ['U.S. PATENT No. 748,626', 'GRANTED JANUARY 5, 1904']),
                          (self.cue('She called it') + 0.2, ['THE LANDLORD’S GAME', 'BY LIZZIE J. MAGIE'])):
            ev.append((t0, 'tap'))
            ev += [(t0 + 0.15 + i / 30, 'key') for i, ch in enumerate(''.join(lines)) if ch.strip()]
        return ev


class Promise(Scene):
    """Title card; the orange squares everyone talks about; the dice; the five decisions; the stamp."""

    def __init__(self):
        super().__init__('02_the_promise')
        self.title = sf.frame_title(with_stamp=False)   # style frame 01; its stamp lands live at the end
        self.base = sf.paper(seed=31)
        S = 1500
        self.board = sf.paper(S + 200, S + 200, seed=32)
        sf.draw_board(self.board, 100, 100, S, highlight='orange')
        # the left column from New York down to St. James, turned upright so the names read normally
        crop = self.board.crop((92, 300, 306, 792)).rotate(90, expand=True, resample=Image.BICUBIC)
        self.orange = sf.paper(seed=31)
        big = crop.resize((int(crop.size[0] * 1.7), int(crop.size[1] * 1.7)), Image.LANCZOS)
        sf.paste_shadowed(self.orange, big, ((W - big.size[0]) // 2, 120))
        sf.text(self.orange, (W / 2, 690), 'The colour everyone tells you to buy.', sf.font('serif', 54), anchor='mm')
        sf.text(self.orange, (W / 2, 760), 'Orange: landed on more than any other set.', sf.font('hand', 38), fill=sf.SEPIA, anchor='mm')
        self.choose = sf.paper(seed=33)
        sf.text(self.choose, (560, 140), 'WHAT YOU DO CHOOSE', sf.font('scb', 46), fill=sf.NOTE_GREEN)
        self.faces = [die_face(n, 150) for n in range(1, 7)]
        self.decisions = [('Whether to buy', '1.  To buy, or to pass.'),
                          ('When to trade', '2.  To trade for a full set.'),
                          ('Whether to chase the railroads', '3.  To collect the railroads.'),
                          ('When to build', '4.  To build, and how much cash to keep.'),
                          ('Whether to stay in jail', '5.  To leave jail, or to stay.')]

    def frame(self, t):
        c_orange, c_dice, c_choose = self.cue('Buy the orange'), self.cue("But you don't get"), self.cue('What you do choose')
        c_win = self.cue('three games for every two')
        if t < c_orange - 0.4:
            return fade(self.title.copy(), ramp(t, 0, 0.6))
        if t < c_choose:
            # the orange squares, then the dice tumbling across them
            img = self.orange.copy()
            a = ramp(t, c_orange - 0.4, 0.6)
            if t >= c_dice:
                x = clamp((t - c_dice) / 1.6)
                for k in range(2):
                    face = self.faces[(int(t * 9) + 3 * k) % 6] if x < 1 else self.faces[[2, 4][k]]
                    ang = (1 - ease_out(x)) * (540 + 200 * k)
                    px = lerp(2000, 820 + k * 260, ease_out(x))
                    py = 470 + k * 30 - 160 * abs(math.sin(x * math.pi * 2.5)) * (1 - x)
                    dimg = face.rotate(ang, expand=True, resample=Image.BICUBIC)
                    sf.paste_shadowed(img, dimg, (px - dimg.size[0] / 2, py - dimg.size[1] / 2), offset=(8, 12), blur=8)
                if t >= self.cue('The dice do that'):
                    txt(img, (W / 2, 880), 'You don’t choose where you land.', sf.font('serifb', 56), anchor='mm',
                            alpha=int(255 * ramp(t, self.cue('The dice do that'), 0.4)))
            return fade(img, a) if t < c_orange + 0.3 else img
        # the five decisions, typed one by one
        img = self.choose.copy()
        for k, (phrase, line) in enumerate(self.decisions):
            t0 = self.cue(phrase)
            typewriter(img, (580, 250 + k * 100), line, sf.font('serif', 56), t, t0 - 0.1, cps=34)
        if t >= self.cue('So we tested'):
            txt(img, (580, 760), 'Same dice.  Same board.  Tested, one decision at a time.', sf.font('hand', 38), fill=sf.SEPIA,
                    alpha=int(255 * ramp(t, self.cue('So we tested'), 0.5)))
        stamp_in(img, t, c_win + 0.2, (1620, 950), ['3 WINS FOR', 'EVERY 2'], angle=-9, size=60)
        return shake(img, t, c_win + 0.35)

    def extra_sfx(self):
        ev = [(self.cue("But you don't get"), 'rattle'), (self.cue('three games for every two') + 0.3, 'thud')]
        for phrase, line in self.decisions:
            t0 = self.cue(phrase) - 0.1
            ev += [(t0 + i / 34, 'key') for i, ch in enumerate(line) if ch.strip()]
        return ev


SCENES = {'01_cold_open': ColdOpen, '02_the_promise': Promise}

if __name__ == '__main__':
    name = sys.argv[1]
    cap = '--captions' in sys.argv
    print(SCENES[name]().render(captions=cap))
