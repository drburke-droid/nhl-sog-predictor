"""Scenes 08-11: jail, the computer's player, the payoff, the playbook.

    python3 scenes_c.py 10_the_payoff [--sheet | --frame 12.5 | --captions]
"""
import math

import numpy as np
from PIL import Image, ImageDraw

from engine import H, W, Camera, clamp, die_face, ease_out, lerp, sf, smooth, txt
from kit import (F, Bar, Counter, Fn, Grid100, Line, Obj, PagedScene, Stamp, Telegram, Txt, Typed, decision_card, deed,
                 hotel_elevation, house_elevation, ledger_paper, ledger_row, main, note_card, paste, pawn_layer, ring, shadowed, tick,
                 with_alpha)
from kit import patent_world
from scenes_b import houses_row, swatch

SEP, RED, GREEN, G = sf.SEPIA, sf.STAMP, sf.NOTE_GREEN, sf.GROUP


def plain(seed):
    return sf.paper(seed=seed)


def bars_layer(n=9, h=H, seed=0):
    """Prison bars, inked, across the whole frame."""
    lay = sf.ink_layer((W, h))
    d = ImageDraw.Draw(lay)
    for i in range(n):
        x = 120 + i * (W - 240) / (n - 1)
        sf.wobble_line(d, [(x, 60), (x, h - 60)], width=16, amp=0.7, seed=seed + i)
    for y in (140, h - 140):
        sf.wobble_line(d, [(80, y), (W - 80, y)], width=14, amp=0.7, seed=seed + 50 + y)
    return lay


def trophy(h=46, color=sf.BRASS):
    w = int(h * 0.8)
    lay = sf.ink_layer((w, h))
    d = ImageDraw.Draw(lay)
    dark = tuple(int(v * 0.55) for v in color)
    d.pieslice([w * 0.12, -h * 0.25, w * 0.88, h * 0.62], 0, 180, fill=color + (255,), outline=dark + (255,), width=2)
    d.arc([0, h * 0.05, w * 0.32, h * 0.4], 90, 270, fill=dark + (255,), width=2)
    d.arc([w * 0.68, h * 0.05, w, h * 0.4], 270, 90, fill=dark + (255,), width=2)
    d.rectangle([w * 0.44, h * 0.6, w * 0.56, h * 0.82], fill=color + (255,), outline=dark + (255,), width=1)
    d.rectangle([w * 0.26, h * 0.82, w * 0.74, h - 1], fill=dark + (255,))
    return lay


# ====================================================================== 08
class Jail(PagedScene):
    def __init__(self):
        super().__init__('08_jail')
        c = self.cue
        self.page(decision_card(5, 'JAIL'))

        p = self.page(plain(151), c("You're in jail") - 0.2)
        bars = bars_layer(seed=3)
        p.add(Obj(c("You're in jail") + 0.1, pawn_layer(220, (150, 52, 40)), (W / 2 - 132, 330), how='drop', shadow=None))

        def draw_bars(img, t, a, t0=c("You're in jail") + 0.2):
            e = ease_out((t - t0) / 0.7)
            hgt = int(H * e)
            if hgt > 0:
                paste(img, bars.crop((0, 0, W, hgt)), (0, 0))
        p.add(Fn(c("You're in jail") + 0.2, draw_bars))
        opt = lambda lines: note_card(lines, F('type', 40), w=560, seed=len(lines[0]))
        p.add(Obj(c('Pay fifty dollars'), opt(['PAY $50', 'and get out']), (150, 700), how='left', angle=-3),
              Obj(c('or stay and roll'), opt(['STAY', 'and roll for doubles']), (1210, 700), how='right', angle=3))

        # Magie's own jail corner
        world, P = patent_world()
        full = (world.size[0] / 2, world.size[1] / 2, 0.5)
        t0 = c('Jail was there') - 0.2
        cam = Camera(world, [(t0, *full), (c('board has a corner') - 0.3, *full), (c('that reads') + 0.2, *P(285, 860), 1.6),
                             (c('Go to Jail') + 1.4, *P(285, 860), 1.7)])
        p = self.page(plain(152), t0, zoom=0)
        p.add(Fn(t0 - 0.01, lambda img, t, a: img.paste(cam.view(t), (0, 0))))
        from engine import slip
        p.add(Fn(c("Lizzie Magie's") - 0.2, lambda img, t, a: slip(img, (70, 70), ['L. J. MAGIE, PATENT No. 748,626', 'THE BOARD OF 1904'], t,
                                                                     c("Lizzie Magie's") - 0.2, width=640)))
        p.add(Fn(c('No Trespassing'), lambda img, t, a: slip(img, (1150, 860), ['“NO TRESPASSING.', '  GO TO JAIL.”'], t,
                                                             c('No Trespassing'), fnt=F('type', 40), width=640)))

        # the rule: leave early, stay late
        p = self.page(plain(153), c("Here's the rule") - 0.3)
        p.add(Txt(c("Here's the rule"), (W / 2, 90), 'THE RULE', F('sc', 40), fill=SEP, anchor='mm'))
        p.add(Txt(c('Early in the game'), (500, 190), 'EARLY IN THE GAME', F('scb', 44), fill=GREEN, anchor='mm'),
              Txt(c("There's property to buy"), (500, 260), 'property still for sale', F('hand', 38), fill=SEP, anchor='mm'),
              Obj(c("There's property to buy") + 0.2, deed('VERMONT AVE.', 100, 'lightblue', seed=7).resize((420, 262), Image.LANCZOS),
                  (290, 310), how='drop', angle=-3),
              Stamp(c('game, get out') + 0.4, (500, 690), ['GET OUT'], angle=-8, size=70))
        p.add(Txt(c('Once anyone has houses'), (1420, 190), 'ONCE ANYONE HAS HOUSES', F('scb', 44), fill=GREEN, anchor='mm'),
              Obj(c('Once anyone has houses') + 0.2, houses_row(3, h=150, gap=18, seed=4), (1170, 300), how='drop'),
              Stamp(c('stay in.') + 0.1, (1420, 690), ['STAY IN'], angle=-6, size=70))
        p.add(Txt(c('You still collect'), (1420, 800), 'you still collect your rent', F('hand', 36), fill=SEP, anchor='mm'),
              Txt(c('and every turn inside'), (1420, 855), 'and can’t land on their hotels', F('hand', 36), fill=SEP, anchor='mm'))
        p.add(Typed(c('That takes you from'), (150, 820), 'games won .......... 26.5%  (typical 25.1%)', F('type', 34)),
              Typed(c('and it makes you less likely'), (150, 880), 'first to go broke ... 23.0%  (typical 25.1%)', F('type', 34)))

        # timing is what matters
        p = self.page(ledger_paper(154, rules=[420, 590, 760], cols=()), c('Always staying') - 0.2)
        p.add(Txt(c('Always staying') + 0.2, (170, 90), 'JAIL: GAMES WON (%)', F('scb', 42), fill=GREEN))
        ledger_row(p, c('Always staying') + 0.2, 340, 'Always stay', 25.1, 24, x_bar=900, fnt=F('type', 40))
        ledger_row(p, c('or always rolling once'), 510, 'Always roll once, then pay', 25.0, 24, x_bar=900, fnt=F('type', 40))
        ledger_row(p, c("It's the timing") - 0.4, 680, 'Leave early, stay once\nthere are houses', 26.5, 24, x_bar=900, fnt=F('type', 40),
                   wash=G['green'])
        p.add(Txt(c('gains you nothing') + 0.1, (180, 870), 'Always staying, or always rolling once: nothing gained.', F('hand', 38), fill=SEP),
              Stamp(c("It's the timing") + 0.3, (1600, 900), ['TIMING'], angle=-8, size=70))


# ====================================================================== 09
class ComputersPlayer(PagedScene):
    def __init__(self):
        super().__init__('09_the_computers_player')
        c = self.cue
        p = self.page(plain(161))
        p.add(Txt(0.4, (W / 2, 90), 'One more test: let the computer invent a player.', F('serif', 48), anchor='mm'),
              Txt(1.0, (W / 2, 150), 'Fig. 3.  TRIAL AND ERROR', F('sc', 30), fill=SEP, anchor='mm'))
        r = np.random.default_rng(5)
        rows_y = [250, 430, 610, 900]
        x0, gap, bw = 150, 68, 46
        winners = [sorted(r.choice(24, 6, replace=False)) for _ in rows_y]
        t_start = c('Start with twenty-four')
        t_keep = c('keep the winners')
        t_mix = c('mix and tweak')
        t_rep = c('and repeat')

        def tree(img, t, a):
            d = ImageDraw.Draw(img)
            for ri, y in enumerate(rows_y):
                tr = t_start + (0 if ri == 0 else (t_mix + 0.4 * (ri - 1) - t_start if ri < 3 else t_rep + 0.8 - t_start))
                if t < tr:
                    continue
                n = int(clamp((t - tr) / 0.8) * 24)
                for k in range(n):
                    x = x0 + k * gap
                    d.rectangle([x, y, x + bw, y + bw], fill=(244, 238, 222, 255), outline=sf.INK + (220,), width=2)
                if ri < 3 and t >= t_keep + 0.3 * ri:
                    for k in winners[ri]:
                        x = x0 + k * gap + bw / 2
                        pts = ring(x, y + bw / 2, 36, 36)
                        d.line(pts, fill=RED + (220,), width=3)
                if ri < 2 and t >= t_mix + 0.4 * ri:
                    for k in range(24):
                        a_, b_ = winners[ri][k % 6], winners[ri][(k * 5 + 1) % 6]
                        for src in (a_, b_):
                            d.line([(x0 + src * gap + bw / 2, y + bw + 4), (x0 + k * gap + bw / 2, rows_y[ri + 1] - 4)],
                                   fill=sf.INK + (45,), width=1)
            if t >= t_rep:
                for dy in (-26, 0, 26):
                    d.ellipse([W / 2 - 5, 770 + dy - 5, W / 2 + 5, 770 + dy + 5], fill=SEP + (230,))
            for ri, lab in enumerate(['generation 1', 'generation 2', 'generation 3']):
                if t >= t_start + (0 if ri == 0 else t_mix + 0.4 * (ri - 1) - t_start):
                    txt(img, (x0 - 20, rows_y[ri] + bw / 2), str(ri + 1), F('type', 30), anchor='rm', fill=SEP)
        p.add(Fn(t_start, tree))
        p.add(Counter(t_rep + 0.6, self.cue('It ended up') - 0.4, (x0 - 20, rows_y[3] + bw / 2), 4, 25, lambda v: '%d' % v, F('type', 30),
                      fill=RED, anchor='rm'),
              Txt(t_rep + 0.6, (W / 2, 1010), 'keep the winners · mix their habits · tweak · repeat, 25 times', F('hand', 34), fill=SEP, anchor='mm'))

        p = self.page(plain(162), c('It ended up') - 0.2)
        p.add(Txt(c('It ended up') + 0.2, (150, 100), 'THE COMPUTER’S PLAYER', F('scb', 46), fill=GREEN),
              Txt(c('It ended up') + 0.4, (150, 170), 'the same habits you’ve just heard:', F('serif', 40)))
        habits = ['buys what it lands on', 'bids up at auction', 'makes every fair set-for-set swap', 'spends its cash on houses',
                  'leaves jail early, stays in late']
        for k, hb in enumerate(habits):
            t0 = c('It ended up') + 0.6 + 0.35 * k
            p.add(Typed(t0, (230, 250 + k * 70), hb, F('type', 40)))
            tick(p, t0 + 0.1, 170, 254 + k * 70, size=28, seed=k)
        p.add(Txt(c('Plus one'), (1150, 250), 'Plus one:', F('deco', 70), fill=RED),
              Txt(c('It picked two favourite'), (1150, 360), 'two favourite colours', F('serif', 48)),
              Obj(c('light blue and orange') - 0.1, swatch('lightblue', 280, 120, seed=1, label='light blue'), (1150, 440), how='drop'),
              Obj(c('light blue and orange') + 0.3, swatch('orange', 280, 120, seed=2, label='orange'), (1470, 440), how='drop'),
              Txt(c('and went hard for them'), (1150, 600), 'and went hard for them', F('hand', 38), fill=SEP),
              Typed(c('It wins about thirty-six'), (230, 700), '36%', F('deco', 170), fill=RED, cps=10))


# ====================================================================== 10
class Payoff(PagedScene):
    def __init__(self):
        super().__init__('10_the_payoff')
        c = self.cue
        p = self.page(plain(171))
        p.add(Typed(0.4, (W / 2, 110), 'So put it all together.', F('serif', 56), anchor='mm'),
              Obj(c('One player who makes') - 0.1, pawn_layer(220, (150, 52, 40)), (W / 2 - 132, 220), how='drop'),
              Txt(c('One player who makes') + 0.3, (W / 2, 560), 'THE PLAYBOOK PLAYER', F('scb', 50), fill=RED, anchor='mm'),
              Txt(c('every one of these decisions'), (W / 2, 630), 'every decision made well', F('serif', 44), anchor='mm'),
              Txt(c('and never picks'), (W / 2, 690), 'no favourite colour', F('serif', 44), anchor='mm'))
        p.add(Obj(c('Same dice as everyone'), die_face(5, 130), (W / 2 - 170, 800), how='drop', angle=-12),
              Obj(c('Same dice as everyone') + 0.15, die_face(2, 130), (W / 2 + 40, 810), how='drop', angle=9))

        # the price: handing the bank $250
        p = self.page(plain(172), c('How much is that worth') - 0.2)
        p.add(Typed(c('How much is that worth'), (1010, 120), 'How much is that worth?', F('serif', 52)))
        notes = [(100, (40, 92, 70), -12, (70, 150)), (100, (40, 92, 70), -4, (150, 330)), (50, (86, 60, 40), 6, (110, 520))]
        t_take = c('We gave a typical player')
        for k, (v, ink, ang, xy) in enumerate(notes):
            n = sf.banknote(v, ink, 20 + k)
            p.add(Obj(t_take + 0.3 * k, n, xy, how='left', angle=ang, dur=0.5))
        p.add(Txt(c('We gave a typical player') + 0.2, (1010, 205), 'extra cash a typical player needs to keep up:', F('hand', 34), fill=SEP),
              Typed(c('It took three hundred'), (1000, 245), '$300', F('deco', 200), fill=RED, cps=8))
        p.add(Txt(c('It took three hundred') + 0.8, (1010, 500), 'of free cash at the start of every game', F('serif', 48)),
              Stamp(c('It took three hundred') + 1.4, (430, 900), ['WORTH', '$300'], angle=-8, size=66),
              Txt(c('Even without auctions'), (1010, 640), 'Without auctions: about $200,', F('hand', 44), fill=SEP),
              Txt(c('the same as passing Go'), (1010, 705), 'the same as passing Go.', F('hand', 44), fill=SEP))

        # game nights
        p = self.page(ledger_paper(173, rules=[300 + k * 120 for k in range(5)], top=170), c('Play these decisions well') - 0.2)
        p.add(Txt(c('Play these decisions well') + 0.2, (170, 60), 'GAME NIGHT LEDGER', F('scb', 44), fill=GREEN),
              Txt(c('Play these decisions well') + 0.4, (170, 118), 'A hundred game nights, the same four players. You follow the playbook.', F('serif', 32)))
        rows = [('YOU', 33, RED), ('MOM', 22, sf.INK), ('DAD', 22, sf.INK), ('JORDAN', 23, sf.INK)]
        t_you = c('and you win about')
        t_them = c('while each of the people')
        for k, (who, n, col) in enumerate(rows):
            y = 200 + k * 120
            t0 = t_you if k == 0 else t_them + 0.2 * k
            p.add(Txt(t0 - 0.3, (180, y + 22), who, F('type', 44), fill=col))
            p.add(Fn(t0, lambda img, t, a, y=y, n=n, col=col, t0=t0, k=k: self.tallies(img, t, y, n, col, t0, k)))
        p.add(Txt(c("That's about forty-five") - 0.1, (180, 720), '45% more wins than each of them.', F('serifb', 48)),
              Stamp(c("That's about forty-five") + 1.0, (1560, 860), ['45% MORE', 'WINS'], angle=-8, size=72))

        # a year of game nights: trophies
        p = self.page(plain(174), c('And it compounds') - 0.2)
        p.add(Txt(c('And it compounds') + 0.2, (W / 2, 80), 'And it compounds.', F('serif', 52), anchor='mm'),
              Typed(c('Play once a week'), (W / 2, 150), 'A year of weekly game nights with the same family.  100 years of them:',
                    F('type', 32), anchor='mm', fill=SEP))
        tro = trophy(48)
        shelf = [('YOU', 65, RED), ('MOM', 9, sf.INK), ('DAD', 9, sf.INK), ('JORDAN', 9, sf.INK)]
        t_you = c('The player with the playbook')
        t_oth = c('Each of the others')
        for k, (who, n, col) in enumerate(shelf):
            y = 250 + k * 190 if k else 250
            y = [230, 560, 720, 880][k]
            t0 = t_you if k == 0 else t_oth + 0.25 * k

            def draw(img, t, a, y=y, n=n, t0=t0, who=who, col=col, k=k):
                txt(img, (150, y + 30), who, F('type', 42), fill=col)
                m = int(clamp((t - t0) / (2.4 if n > 20 else 0.8)) * n)
                for i in range(m):
                    paste(img, tro, (420 + (i % 26) * 52, y + (i // 26) * 66))
                if m == n:
                    txt(img, (420 + min(n, 26) * 52 + 20, y + 30), str(n), F('type', 44), fill=col, anchor='lm')
            p.add(Fn(t0, draw))
            d = Fn(t0 - 0.2, lambda img, t, a, y=y: ImageDraw.Draw(img).line([(400, y + 56 + (130 if y == 230 else 0)),
                                                                              (1820, y + 56 + (130 if y == 230 else 0))],
                                                                             fill=SEP + (200,), width=4))
            p.add(d)
        p.add(Txt(c('champion two years out of three'), (1820, 450), 'champion two years in three', F('hand', 40), fill=RED, anchor='ra'),
              Txt(c('About one year in eleven'), (1820, 1010), 'each of the others: about one year in eleven', F('hand', 38), fill=SEP, anchor='ra'))

        # same dice
        p = self.page(plain(175), c('Same dice.') - 0.3)
        p.add(Txt(c('Same dice.'), (W / 2, 330), 'Same dice.', F('script', 110), anchor='mm'),
              Txt(c('No favourite colour'), (W / 2, 520), 'No favourite colour.', F('script', 110), anchor='mm'),
              Txt(c('Just better decisions'), (W / 2, 720), 'Just better decisions.', F('script', 110), fill=RED, anchor='mm'))

    def tallies(self, img, t, y, n, col, t0, k):
        m = int(clamp((t - t0) / 1.6) * n)
        if m:
            end = sf.tally(img, 450, y + 18, m, color=col, seed=100 * k)
            if m == n:
                txt(img, (end + 40, y + 22), str(n), F('type', 48), fill=col)


# ====================================================================== 11
def certificate_bg():
    img = sf.paper(base=(236, 229, 210), seed=5)
    sf.guilloche_frame(img, inset=30, band=56)
    txt(img, (W / 2, 150), 'THE PLAYBOOK', F('deco', 110), anchor='mm')
    txt(img, (W / 2, 240), 'CERTIFICATE OF BETTER DECISIONS', F('sc', 34), fill=SEP, anchor='mm')
    return img


def seal():
    lay = sf.ink_layer((300, 300))
    d = ImageDraw.Draw(lay)
    cx = cy = 150
    R = 112
    for k in range(36):
        a = k * math.pi / 18
        d.polygon([(cx + R * math.cos(a), cy + R * math.sin(a)), (cx + (R + 18) * math.cos(a + 0.087), cy + (R + 18) * math.sin(a + 0.087)),
                   (cx + R * math.cos(a + 0.175), cy + R * math.sin(a + 0.175))], fill=RED + (230,))
    d.ellipse([cx - R, cy - R, cx + R, cy + R], fill=RED + (230,))
    d.ellipse([cx - R + 14, cy - R + 14, cx + R - 14, cy + R - 14], outline=(240, 220, 200, 200), width=2)
    lay = sf.roughen(lay, 0.04, 21)
    txt(lay, (cx, cy - 14), '+45%', F('fat', 66), fill=(246, 234, 214), anchor='mm')
    txt(lay, (cx, cy + 48), 'MORE WINS', F('scb', 26), fill=(246, 234, 214), anchor='mm')
    return lay


class Playbook(PagedScene):
    def __init__(self):
        super().__init__('11_the_playbook')
        c = self.cue
        p = self.page(certificate_bg())
        rules = [('One.', 'Land on it?  Buy it, unless it would leave you nearly broke.'),
                 ('Two.', 'At auction, bid up to one and a half times the price.'),
                 ('Three.', 'Make every fair set-for-set trade.  Don’t haggle.  When it stalls, deal.'),
                 ('Four.', 'Want the railroads?  Trade for the last one, or don’t bother.'),
                 ('Five.', 'Spend your cash on houses, and go to hotels.  Owe money?  Mortgage first.'),
                 ('Six.', 'Two sets?  Cheaper houses first.  Never one house on everything.'),
                 ('Seven.', 'Get out of jail early.  Stay in once anyone has houses.')]
        nf, f = F('serifb', 36), F('serif', 36)
        for k, (ph, rule) in enumerate(rules):
            y = 320 + k * 78
            t0 = c(ph)
            p.add(Txt(t0, (330, y), '%s.' % 'I II III IV V VI VII'.split()[k], nf, fill=GREEN, anchor='ra'),
                  Typed(t0 + 0.3, (360, y), rule, f, cps=40))
        t_end = c('Stay in once anyone') + 0.2
        p.add(Obj(t_end, seal(), (1470, 710), how='drop', shadow=(6, 8, 6, 80)),
              Txt(t_end + 0.3, (300, 870), 'Same dice.', F('script', 64)),
              Txt(t_end + 0.6, (300, 950), '45% more wins than each of them. A year of weekly game nights: champion two years in three.',
                  F('type', 24), fill=SEP))

        # over the 1904 patent
        world, P = patent_world()
        full = (world.size[0] / 2, world.size[1] / 2, 0.5)
        t0 = c('Lizzie Magie made') - 0.3
        cam = Camera(world, [(t0, *P(1300, 2470), 1.3), (c('to show that') + 0.6, *P(1300, 2470), 1.4), (c('Now you know') - 0.2, *full),
                             (self.dur, world.size[0] / 2, world.size[1] / 2, 0.52)])
        p = self.page(plain(181), t0, enter='fade', zoom=0)
        p.add(Fn(t0 - 0.01, lambda img, t, a: img.paste(cam.view(t), (0, 0))))

        def banner(img, t, a, s, t1, y, fnt, fill=sf.INK, t2=None):
            if t < t1 or (t2 is not None and t >= t2):
                return
            al = clamp((t - t1) / 0.5)
            band = Image.new('RGBA', (W, 120), (239, 228, 204, int(225 * al)))
            img.alpha_composite(band, (0, y - 60))
            txt(img, (W / 2, y), s, fnt, fill=fill, anchor='mm', alpha=int(255 * al))
        p.add(Fn(t0, lambda img, t, a: banner(img, t, a, 'to show that the landlord always wins', c('to show that'), 980, F('serif', 52),
                                              t2=c('Now you know'))))
        p.add(Fn(t0, lambda img, t, a: banner(img, t, a, 'Now you know how to be the landlord.', c('Now you know'), 980, F('deco', 64), RED)))
        p.add(Fn(t0, lambda img, t, a: banner(img, t, a, 'Which of these does your family get wrong?', c('Which of these'), 110, F('serif', 52))))
        p.add(Txt(c('Which of these'), (W - 40, H - 22), 'Not affiliated with or endorsed by Hasbro. MONOPOLY is a trademark of Hasbro.',
                  F('type', 20), fill=SEP, anchor='rb'))


SCENES = {'08_jail': Jail, '09_the_computers_player': ComputersPlayer, '10_the_payoff': Payoff, '11_the_playbook': Playbook}

if __name__ == '__main__':
    main(SCENES)
