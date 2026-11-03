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
from kit import F, Counter, Fn, Obj, PagedScene, Txt, Typed, main, patent_world, pawn_layer

PATENT_SRC = (1805, 2809)  # pixel size of the patent sheet as scanned


class ColdOpen(PagedScene):
    """The hook ($300, thirteen million games), then the 1904 patent on the desk, with two slow camera moves."""

    def __init__(self):
        super().__init__('01_cold_open')
        c = self.cue
        # the hook
        p = self.page(sf.paper(seed=20))
        p.add(Typed(c('Five decisions'), (W / 2, 120), 'FIVE DECISIONS', F('deco', 110), anchor='mm', cps=22))
        for k in range(3):
            note = sf.banknote(100, (40, 92, 70), 50 + k)
            p.add(Obj(c('worth three hundred dollars') + 0.25 * k, note, (110 + k * 40, 200 + k * 90), how='drop', angle=[-9, 4, -3][k]))
        p.add(Typed(c('worth three hundred dollars') + 0.6, (1040, 200), '$300', F('deco', 230), fill=sf.STAMP, cps=10),
              Txt(c('every single game'), (1060, 470), 'every single game', F('hand', 48), fill=sf.SEPIA))
        for k in range(4):
            col = (150, 52, 40) if k == 0 else sf.BRASS
            t0 = c('Get them right') + (0 if k == 0 else 0.6 + 0.15 * k)
            p.add(Obj(t0, pawn_layer(120, col), (240 + k * 300, 760), how='drop'),
                  Txt(t0 + 0.2, (312 + k * 300, 945), '$1,800' if k == 0 else '$1,500', F('type', 40), fill=sf.STAMP if k == 0 else sf.INK,
                      anchor='mm'))
        p.add(Txt(c('more than everyone else'), (312, 1000), 'YOU', F('sc', 30), fill=sf.STAMP, anchor='mm'),
              Txt(c('We played more than'), (1660, 760), 'games played', F('sc', 32), fill=sf.SEPIA, anchor='mm'),
              Counter(c('We played more than'), c('to find them') + 0.4, (1660, 830), 0, 13_400_000, lambda v: '{:,}'.format(int(v)),
                      F('type', 60), fill=sf.STAMP, anchor='mm'))

        # the patent
        world, P = patent_world()
        full = (world.size[0] / 2, world.size[1] / 2 + 20, 0.5)
        t0 = c('And the game itself') - 0.1
        head = P(1300, 160)
        self.cam = Camera(world, [(t0, *full), (c('In nineteen-oh-four') + 0.3, *full[:2], 0.53),
                                  (c('was granted') + 0.4, *head, 1.25), (c('She called it') + 0.1, *head, 1.32),
                                  (c('She designed it') - 0.2, *full[:2], 0.54), (self.dur, *full[:2], 0.58)])
        p = self.page(sf.paper(seed=21), t0, enter='fade', zoom=0)
        p.add(Fn(t0 - 0.01, lambda img, t, a: img.paste(self.cam.view(t), (0, 0))))
        self.slips = [(c('was granted') - 0.1, ['U.S. PATENT No. 748,626', 'GRANTED JANUARY 5, 1904'], (70, 70)),
                      (c('She called it') + 0.2, ['THE LANDLORD’S GAME', 'BY LIZZIE J. MAGIE'], (70, 230))]
        for ts, lines, xy in self.slips:
            p.add(Fn(ts, lambda img, t, a, ts=ts, lines=lines, xy=xy: t < c('Thirty years later') and slip(img, xy, lines, t, ts, width=560)))
        t_k = c('You know it as') + 0.4

        def known(img, t, a):
            al = clamp((t - t_k) / 0.5)
            img.alpha_composite(Image.new('RGBA', (W, 130), (239, 228, 204, int(225 * al))), (0, H - 200))
            txt(img, (W / 2, H - 135), 'You know it as Monopoly.', F('serif', 64), anchor='mm', alpha=int(255 * al))
        p.add(Fn(t_k, known))

    def extra_sfx(self):
        ev = []
        for t0, lines, _ in self.slips:
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
    main(SCENES)
