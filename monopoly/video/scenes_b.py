"""Scenes 07a-07e: the building decisions.

    python3 scenes_b.py 07c_which_set_first [--sheet | --frame 12.5 | --captions]
"""
import math

from PIL import Image, ImageDraw

from engine import H, W, clamp, ease_out, sf, txt
from kit import (F, Bar, Counter, Fn, Line, Obj, PagedScene, Stamp, Telegram, Txt, Typed, decision_card, deed, hotel_elevation,
                 house_elevation, ledger_paper, ledger_row, main, note_card, paste, ring, shadowed, tick, with_alpha)

SEP, RED, GREEN, G = sf.SEPIA, sf.STAMP, sf.NOTE_GREEN, sf.GROUP


def plain(seed):
    return sf.paper(seed=seed)


def houses_row(n, h=64, gap=8, seed=0):
    """n little house elevations side by side."""
    one = house_elevation(h, seed=seed)
    lay = sf.ink_layer((n * (one.size[0] + gap), one.size[1]))
    for k in range(n):
        lay.alpha_composite(one, (k * (one.size[0] + gap), 0))
    return lay


def swatch(group, w=150, h=70, seed=0, label=None):
    s = sf.ink_layer((w, h))
    sf.wash(s, (0, 0, w, h), G[group], alpha=200, seed=seed)
    ImageDraw.Draw(s).rectangle([0, 0, w - 1, h - 1], outline=sf.INK + (180,), width=2)
    if label:
        txt(s, (w / 2, h / 2), label, F('sc', 22), anchor='mm', fill=(250, 244, 230) if group in ('darkblue', 'green', 'red', 'brown') else sf.INK)
    return s


SETS = {'brown': ['MEDITERRANEAN', 'BALTIC'], 'lightblue': ['ORIENTAL', 'VERMONT', 'CONNECTICUT'],
        'pink': ['ST. CHARLES', 'STATES', 'VIRGINIA'], 'orange': ['ST. JAMES', 'TENNESSEE', 'NEW YORK'],
        'red': ['KENTUCKY', 'INDIANA', 'ILLINOIS'], 'yellow': ['ATLANTIC', 'VENTNOR', 'MARVIN GDNS.'],
        'green': ['PACIFIC', 'N. CAROLINA', 'PENNSYLVANIA'], 'darkblue': ['PARK PLACE', 'BOARDWALK']}
PRICES = {'ORIENTAL': 100, 'VERMONT': 100, 'CONNECTICUT': 120, 'ST. JAMES': 180, 'TENNESSEE': 180, 'NEW YORK': 200,
          'PACIFIC': 300, 'N. CAROLINA': 300, 'PENNSYLVANIA': 320}


def lots(group, n_each, w=120, h=150, gap=10, hotel=False, seed=0):
    """A row of building lots tinted in a group's colour, each with n houses (or a hotel) drawn on it."""
    k = len(SETS[group])
    lay = sf.ink_layer((k * (w + gap), h + 10))
    d = ImageDraw.Draw(lay)
    for i in range(k):
        x = i * (w + gap)
        sf.wash(lay, (x, h - 34, x + w, h), G[group], alpha=190, seed=seed + i)
        d.rectangle([x, 0, x + w, h], outline=sf.INK + (170,), width=2)
        if hotel:
            ho = hotel_elevation(96, seed=i)
            lay.alpha_composite(ho, (int(x + (w - ho.size[0]) / 2), int(h - 40 - ho.size[1])))
        elif n_each:
            hs = houses_row(n_each, h=int(min(44, (w - 10) / n_each / 1.05)), gap=2, seed=i)
            lay.alpha_composite(hs, (int(x + (w - hs.size[0]) / 2), int(h - 40 - hs.size[1])))
    return lay


# ====================================================================== 07a
class SpendOrSave(PagedScene):
    def __init__(self):
        super().__init__('07a_spend_or_save')
        c = self.cue
        self.page(decision_card(4, 'TO BUILD'))

        p = self.page(plain(101), c('And the first question') - 0.2)
        p.add(Typed(c('And the first question') + 0.3, (W / 2, 110), 'The question every player asks:', F('serif', 50), anchor='mm'))
        p.add(Obj(c('spend all my cash'), house_elevation(300, seed=3), (270, 230), how='drop'),
              Txt(c('spend all my cash') + 0.2, (420, 690), 'Spend it all\non houses?', F('deco', 60), anchor='mm', align='center', spacing=10))
        stack = sf.ink_layer((620, 420))
        for k in range(3):
            n = sf.banknote(100, (40, 92, 70), 40 + k).rotate((k - 1) * 6, expand=True, resample=Image.BICUBIC)
            stack.alpha_composite(n, (10 + k * 12, 10 + k * 40))
        p.add(Obj(c('or keep some back'), stack, (1180, 200), how='drop'),
              Txt(c('or keep some back') + 0.2, (1490, 690), 'Or keep\nsome back?', F('deco', 60), fill=SEP, anchor='mm', align='center',
                  spacing=10))
        p.add(Obj(c("in case I land"), hotel_elevation(230, seed=5), (830, 200), how='down'),
              Txt(c("in case I land") + 0.2, (W / 2, 490), 'someone’s hotel', F('hand', 34), fill=RED, anchor='mm'),
              Txt(c('Keeping a cushion'), (W / 2, 890), 'A cushion feels responsible.', F('serif', 46), anchor='mm'),
              Txt(c('So we gave players'), (W / 2, 975), 'So we tried cushions of every size.', F('hand', 36), fill=SEP, anchor='mm'))

        # the ledger: wins, then first to go broke
        p = self.page(ledger_paper(102, rules=[330 + 110 * k for k in range(5)], cols=(470, 1160, 1480)), c('Keep nothing back') - 0.3)
        p.add(Txt(c('Keep nothing back'), (170, 70), 'LEDGER  —  SPEND IT, OR KEEP SOME BACK?', F('scb', 40), fill=GREEN),
              Txt(c('Keep nothing back'), (170, 200), 'Cash kept back', F('sc', 26), fill=SEP),
              Txt(c('Keep nothing back'), (500, 200), 'Games won (%)', F('sc', 26), fill=SEP))
        rows = [('$0', 25.1, '25.1%', 'Keep nothing back'), ('$100', 22.4, '25.8%', 'Keep a hundred'),
                ('$300', 16.3, '27.8%', 'Three hundred'), ('$500', 10.9, '29.9%', 'Five hundred'), ('$1,000', 3.2, '33.5%', 'A thousand dollars')]
        t_broke = c('The more cash you keep')
        for k, (cash, v, out, ph) in enumerate(rows):
            y = 250 + 110 * k
            p.add(Typed(c(ph), (190, y), cash, F('type', 42)))
            p.add(Bar(c(ph) + 0.25, (510, y - 2), v * 20, h=42, label='%.1f' % v, fnt=F('type', 40), label_fill=RED if v < 20 else sf.INK))
            p.add(Typed(t_broke + 0.35 * k, (1200, y), out, F('type', 42), fill=RED if k >= 2 else sf.INK))
        p.add(Txt(c('Surely the cushion'), (1190, 200), 'First to go broke', F('sc', 26), fill=SEP),
              Txt(c("It doesn't.") - 0.1, (1510, 250), 'It doesn’t\nkeep you safe.', F('hand', 38), fill=RED, spacing=8),
              Line(t_broke + 0.4, [(1440, 270), (1440, 730)], width=5, dur=1.6),
              Line(t_broke + 2.0, [(1420, 700), (1440, 735), (1460, 700)], width=5, dur=0.2),
              Txt(t_broke + 1.0, (1510, 420), 'more cash kept\nback, MORE likely\nto go broke first', F('hand', 36), fill=RED, spacing=8))

        # the smartest-sounding version
        p = self.page(plain(103), c('We even tried') - 0.2)
        card = note_card(['ALWAYS KEEP ENOUGH CASH', 'TO PAY THE BIGGEST RENT', 'ON THE BOARD'], F('type', 46), w=760, seed=11)
        p.add(Obj(c('We even tried') + 0.3, card, (140, 160), how='drop', angle=-2),
              Obj(c('the biggest rent') - 0.3, hotel_elevation(300, seed=9), (1170, 120), how='drop'),
              Txt(c('the biggest rent') + 0.2, (1330, 520), 'Boardwalk with a hotel: $2,000', F('hand', 36), fill=SEP, anchor='mm'),
              Typed(c('That player wins'), (180, 640), '13%', F('deco', 160), fill=RED, cps=10),
              Txt(c('That player wins') + 0.8, (190, 830), 'about half a fair share', F('serif', 46)))

        # why: houses earn, cash doesn't
        p = self.page(plain(104), c("Here's why") - 0.2)
        p.add(Txt(c("Here's why") + 0.2, (W / 2, 100), 'Here’s why.', F('serif', 56), anchor='mm'),
              Obj(c('Houses are what bring'), houses_row(3, h=200, gap=24, seed=1), (110, 230), how='drop'),
              Txt(c('Houses are what bring') + 0.3, (420, 520), 'Houses bring the money in.', F('serif', 46), anchor='mm'))
        note = sf.banknote(100, (40, 92, 70), 44).rotate(-4, expand=True, resample=Image.BICUBIC)
        p.add(Obj(c('Cash in your hand'), note, (1150, 260), how='drop'),
              Txt(c('Cash in your hand') + 0.3, (1430, 560), 'Cash in hand earns nothing.', F('serif', 46), anchor='mm'))
        p.add(Typed(c('And if you do get hit'), (W / 2, 760), 'Get hit?  Mortgage, or sell houses back to the bank.', F('type', 42), anchor='mm'),
              Txt(c('And if you do get hit') + 1.5, (W / 2, 850), 'You can always raise the money when you need it.', F('hand', 38), fill=SEP,
                  anchor='mm'))

        # spend it; mortgage loose property to build
        p = self.page(plain(105), c('So spend it') - 0.3)
        p.add(Stamp(c('So spend it') + 0.1, (420, 240), ['SPEND IT'], angle=-9, size=110))
        bal = deed('BALTIC AVE.', 60, 'brown', seed=8).resize((460, 288), Image.LANCZOS)
        p.add(Obj(c('Better still'), bal, (130, 520), how='left', angle=3),
              Stamp(c("you aren't building on") - 0.1, (360, 670), ['MORTGAGED'], angle=-6, size=50),
              Line(c('and spend that too') - 0.2, [(640, 660), (900, 660)], width=6, dur=0.4),
              Line(c('and spend that too') + 0.2, [(875, 640), (905, 660), (875, 680)], width=6, dur=0.2),
              Obj(c('and spend that too'), houses_row(2, h=150, gap=16, seed=6), (940, 520), how='drop'))
        p.add(Txt(c('That alone takes you'), (1180, 160), 'Mortgage to build:', F('sc', 40), fill=SEP),
              Typed(c('from twenty-five percent'), (1180, 230), '25%', F('deco', 150), cps=10),
              Txt(c('to twenty-eight') - 0.1, (1440, 290), '→', F('serif', 120), fill=SEP),
              Typed(c('to twenty-eight'), (1560, 230), '28%', F('deco', 150), fill=GREEN, cps=10))


# ====================================================================== 07b
class HowFar(PagedScene):
    def __init__(self):
        super().__init__('07b_how_far_to_build')
        c = self.cue
        p = self.page(ledger_paper(111, rules=(), margin=0))
        p.add(Typed(0.4, (W / 2, 90), 'How far should you build?', F('serif', 56), anchor='mm'),
              Txt(0.8, (W / 2, 160), 'games won (%), building only that far', F('sc', 28), fill=SEP, anchor='mm'))
        base, scale = 960, 21
        steps = [(1, 0.1, 'One house'), (2, 4.4, 'Two houses'), (3, 19.6, 'Three houses'), (4, 23.2, 'Four,'), (5, 25.1, 'Hotels,')]
        d_base = Fn(0.4, lambda img, t, a: ImageDraw.Draw(img).line([(120, base), (1800, base)], fill=sf.INK + (220,), width=3))
        p.add(d_base)
        for k, (n, v, ph) in enumerate(steps):
            x = 270 + k * 345
            t0 = c(ph)
            hgt = max(6, v * scale)
            bar = sf.ink_layer((170, int(hgt)))
            sf.hatch(bar, (0, 0, 170, int(hgt)), spacing=7, angle=60, alpha=200 if n < 5 else 120)
            if n == 5:
                sf.wash(bar, (0, 0, 170, int(hgt)), G['green'], alpha=160, seed=3)
            ImageDraw.Draw(bar).rectangle([0, 0, 169, int(hgt) - 1], outline=sf.INK + (230,), width=2)
            p.add(Fn(t0 + 0.2, lambda img, t, a, bar=bar, x=x, t0=t0: self.grow(img, t, bar, x, base, t0 + 0.2)))
            icon = hotel_elevation(130, seed=k) if n == 5 else houses_row(n, h=60 if n < 4 else 52, gap=4, seed=k)
            p.add(Obj(t0, icon, (x + 85 - icon.size[0] / 2, base - hgt - icon.size[1] - 70), how='drop'))
            p.add(Typed(t0 + 0.5, (x + 85, base - hgt - 40), 'almost 0' if v < 1 else '%.1f' % v, F('type', 40), anchor='mm',
                        fill=RED if v < 20 else (GREEN if n == 5 else sf.INK)))
            p.add(Txt(t0, (x + 85, base + 40), ['1 house', '2 houses', '3 houses', '4 houses', 'hotels'][k], F('sc', 30), anchor='mm', fill=SEP))
        x3 = 270 + 2 * 345 + 85
        p.add(Line(c('the third house is the sweet spot'), ring(x3, base - 19.6 * scale - 100, 150, 110), width=4, dur=0.6),
              Txt(c('the third house is the sweet spot') + 0.4, (x3 - 420, 300), '“the sweet spot”?', F('hand', 40), fill=RED, anchor='mm'),
              Txt(c("It's usually the biggest"), (x3 - 420, 362), 'Illinois Ave. rent:\n2 houses $300, 3 houses $750', F('type', 26), fill=SEP,
                  anchor='mm', align='center', spacing=6),
              Stamp(c('Go to hotels') + 0.1, (560, 560), ['GO TO', 'HOTELS'], angle=-8, size=64))


    def grow(self, img, t, bar, x, base, t0):
        e = ease_out((t - t0) / 0.6)
        h = int(bar.size[1] * e)
        if h > 0:
            paste(img, bar.crop((0, bar.size[1] - h, bar.size[0], bar.size[1])), (x, base - h))


# ====================================================================== 07c
class WhichFirst(PagedScene):
    def __init__(self):
        super().__init__('07c_which_set_first')
        c = self.cue
        p = self.page(plain(121))
        p.add(Typed(0.4, (W / 2, 100), 'The question people really argue about.', F('serif', 52), anchor='mm'))
        for k, nm in enumerate(SETS['orange']):
            dk = deed(nm + ' ' + ('PL.' if nm == 'ST. JAMES' else 'AVE.'), PRICES[nm], 'orange', seed=60 + k).resize((330, 206), Image.LANCZOS)
            p.add(Obj(c('You own two full sets') + 0.15 * k, dk, (120 + k * 70, 260 + k * 150), how='left', angle=-4 + 3 * k))
        for k, nm in enumerate(SETS['green']):
            dk = deed(nm + ' AVE.', PRICES[nm], 'green', seed=70 + k).resize((330, 206), Image.LANCZOS)
            p.add(Obj(c('You own two full sets') + 0.3 + 0.15 * k, dk, (1330 + k * 70, 260 + k * 150), how='right', angle=4 - 3 * k))
        p.add(Txt(c('Which one gets') - 0.1, (W / 2, 520), 'Which one\nfirst?', F('deco', 96), anchor='mm', align='center', spacing=14))

        # the setup and the 28 pairs
        p = self.page(plain(122), c('We set up that exact') - 0.2)
        slip = note_card(['THE SET-UP', 'middle of the game', 'every property already bought', '$1,000 each', 'you hold two full sets'],
                         F('type', 38), w=640, seed=12)
        p.add(Obj(c('We set up that exact') + 0.3, slip, (110, 200), how='drop', angle=-2))
        order = ['brown', 'lightblue', 'pink', 'orange', 'red', 'yellow', 'green', 'darkblue']
        gx, gy, cs = 1050, 230, 88
        t_grid = c('Then we tried every pair')
        for i, g in enumerate(order):
            p.add(Obj(t_grid - 0.4, swatch(g, cs - 8, 30, seed=i), (gx + i * cs + 4, gy - 44), how='fade', shadow=None),
                  Obj(t_grid - 0.4, swatch(g, 30, cs - 8, seed=i + 9), (gx - 44, gy + i * cs + 4), how='fade', shadow=None))
        pairs = [(i, j) for i in range(8) for j in range(i + 1, 8)]

        def grid(img, t, a):
            d = ImageDraw.Draw(img)
            for i in range(8):
                for j in range(8):
                    d.rectangle([gx + j * cs, gy + i * cs, gx + (j + 1) * cs, gy + (i + 1) * cs], outline=sf.INK + (110,), width=1)
            n = int(clamp((t - t_grid - 0.3) / 2.8) * 28)
            for (i, j) in pairs[:n]:
                x0, y0 = gx + j * cs, gy + i * cs
                d.line([(x0 + 18, y0 + 46), (x0 + 36, y0 + 66), (x0 + 70, y0 + 22)], fill=RED + (220,), width=5)
        p.add(Fn(t_grid - 0.4, grid))
        p.add(Counter(t_grid + 0.3, t_grid + 3.1, (gx + 4 * cs, gy + 8 * cs + 60), 0, 28, lambda v: '%d pairs' % round(v), F('type', 46),
                      anchor='mm'))

        # the rule and the pecking order
        p = self.page(plain(123), c('The rule is simple') - 0.2)
        p.add(Txt(c('The rule is simple') + 0.2, (W / 2, 90), 'THE RULE', F('sc', 40), fill=SEP, anchor='mm'),
              Typed(c('Build the set with'), (W / 2, 170), 'Build the set with the cheaper houses first.', F('serifb', 58), anchor='mm'))
        tiers = [(['lightblue', 'orange'], 'Light blue', 'beat everything', '$50 · $100 a house'),
                 (['pink'], 'Then pink', '', '$100 a house'),
                 (['red', 'yellow'], 'Red and yellow', 'a coin flip', '$150 a house'),
                 (['darkblue', 'green'], 'so are dark blue', 'a coin flip', '$200 a house')]
        for r, (gs, ph, note, cost) in enumerate(tiers):
            y = 290 + r * 150
            t0 = c(ph)
            for k, g in enumerate(gs):
                p.add(Obj(t0 + 0.12 * k, swatch(g, 220, 100, seed=r * 3 + k, label=g.replace('lightblue', 'light blue').replace('darkblue', 'dark blue')),
                          (360 + k * 250, y), how='drop'))
            if len(gs) == 2 and r > 0:
                p.add(Txt(t0 + 0.4, (585, y + 50), '=', F('serifb', 60), anchor='mm', fill=SEP))
            p.add(Txt(t0 + 0.3, (900, y + 30), cost, F('type', 34), fill=SEP))
            if note:
                p.add(Txt(t0 + 0.5, (1240, y + 26), note, F('hand', 42), fill=RED if r == 0 else SEP))
            p.add(Txt(t0, (200, y + 50), str(r + 1) + '.', F('serifb', 50), anchor='mm', fill=GREEN))
        p.add(Obj(c('The odd one out is brown'), swatch('brown', 220, 100, seed=40, label='brown'), (360, 890), how='drop'),
              Txt(c('The odd one out is brown') + 0.3, (640, 880), 'the odd one out  ($50 a house)', F('hand', 38), fill=RED),
              Txt(c('Its rents are so low'), (640, 935), 'Rents so low it only goes first ahead of dark blue or green, and only just.',
                  F('serif', 32)))

        # getting it wrong
        p = self.page(ledger_paper(124, rules=[470, 650], cols=()), c('And getting it wrong') - 0.2)
        p.add(Txt(c('And getting it wrong') + 0.3, (170, 90), 'GETTING IT WRONG IS EXPENSIVE', F('scb', 44), fill=GREEN),
              Txt(c('Holding orange and green'), (170, 190), 'You hold orange and green.', F('serif', 48)),
              Txt(c('Holding orange and green') + 0.3, (900, 290), 'Games won (%)', F('sc', 26), fill=SEP))
        p.add(Obj(c('Build orange first') - 0.1, swatch('orange', 120, 60, seed=2), (170, 370), how='drop', shadow=None),
              Obj(c('Build green first') - 0.1, swatch('green', 120, 60, seed=3), (170, 550), how='drop', shadow=None))
        ledger_row(p, c('Build orange first'), 380, 'orange first', 84.3, 9.5, x_label=320, x_bar=900, fnt=F('type', 44), wash=G['orange'])
        ledger_row(p, c('Build green first'), 560, 'green first', 63.3, 9.5, x_label=320, x_bar=900, fnt=F('type', 44), wash=G['green'])
        p.add(Txt(c('Same two sets'), (170, 760), 'Same two sets.', F('script', 80)),
              Txt(c('Same dice'), (900, 760), 'Same dice.', F('script', 80)),
              Txt(c('Same two sets') + 0.4, (170, 920), 'Build orange first: 21 percentage points more wins.', F('type', 32), fill=RED))


# ====================================================================== 07d
class OneOrSpread(PagedScene):
    def __init__(self):
        super().__init__('07d_one_set_or_spread')
        c = self.cue
        p = self.page(plain(131))
        p.add(Typed(0.4, (W / 2, 100), 'Hotels on one set, or a house on everything?', F('serif', 52), anchor='mm'))
        lx, rx, y0 = 140, 1010, 300
        p.add(Txt(0.9, (lx + 380, y0 - 50), 'ONE SET UP FIRST', F('sc', 36), fill=SEP, anchor='mm'),
              Txt(1.2, (rx + 380, y0 - 50), 'A HOUSE ON EVERYTHING', F('sc', 36), fill=SEP, anchor='mm'))
        p.add(Obj(1.0, lots('lightblue', 0, hotel=True, w=230, h=200, seed=1), (lx, y0), how='drop'),
              Obj(1.2, lots('red', 0, w=230, h=200, seed=4), (lx, y0 + 240), how='drop'),
              Obj(1.4, lots('lightblue', 1, w=230, h=200, seed=1), (rx, y0), how='drop'),
              Obj(1.6, lots('red', 1, w=230, h=200, seed=4), (rx, y0 + 240), how='drop'))
        p.add(Line(c('Never a house on everything'), [(rx - 20, y0 - 10), (rx + 760, y0 + 460)], width=10, dur=0.3),
              Line(c('Never a house on everything') + 0.3, [(rx - 20, y0 + 460), (rx + 760, y0 - 10)], width=10, dur=0.3),
              Stamp(c('was never the best') + 0.2, (rx + 380, 860), ['BEST IN', '0 OF 28'], angle=-8, size=58))
        p.add(Typed(c('On average'), (lx, 820), 'Average games won:', F('type', 34)),
              Typed(c('On average') + 0.8, (lx + 30, 880), 'the wrong set first .... 67.6%', F('type', 34)),
              Typed(c('On average') + 1.8, (lx + 30, 930), 'a house on everything ... 65.8%', F('type', 34), fill=RED))

        p = self.page(plain(132), c('Build one set up first') - 0.2)
        p.add(Typed(c('Build one set up first'), (W / 2, 110), 'Build one set up first.', F('serifb', 60), anchor='mm'))
        cheap = note_card(['CHEAP SET', '(light blue)', 'straight to hotels'], F('type', 44), w=720, seed=21)
        dear = note_card(['EXPENSIVE SET', '(red, yellow, dark blue)', 'three houses on both sets,', 'then the hotels'], F('type', 44), w=780, seed=22)
        p.add(Obj(c("If it's a cheap set"), cheap, (110, 250), how='drop', angle=-2),
              Obj(c("If it's a cheap set") + 0.6, lots('lightblue', 0, hotel=True, w=200, h=180, seed=1), (150, 600), how='drop'))
        p.add(Obj(c("If it's an expensive one"), dear, (1010, 250), how='drop', angle=2),
              Obj(c('get three houses') + 0.2, lots('red', 3, w=200, h=180, seed=4), (1050, 640), how='drop'),
              Obj(c('then add the hotels') - 0.1, lots('yellow', 3, w=200, h=180, seed=7), (1050, 840), how='drop'))


# ====================================================================== 07e
class OweMoney(PagedScene):
    def __init__(self):
        super().__init__('07e_owe_money')
        c = self.cue
        p = self.page(plain(141))
        bill = note_card(['RENT DUE', 'ILLINOIS AVE., WITH HOTEL', '$1,100', 'PAYABLE AT ONCE'], F('type', 44), w=700, seed=31)
        p.add(Txt(0.4, (W / 2, 90), 'One last building question.', F('serif', 50), anchor='mm'),
              Obj(c('You land on a hotel'), bill, (110, 180), how='drop', angle=-3),
              Stamp(c("and you can't pay"), (560, 440), ['CAN’T PAY'], angle=-10, size=60))
        p.add(Typed(c('Do you sell houses'), (1000, 200), 'Sell houses?', F('deco', 70)),
              Typed(c('or mortgage?'), (1000, 300), 'Or mortgage?', F('deco', 70)))
        loose = [('BALTIC AVE.', 60, 'brown'), ('ORIENTAL AVE.', 100, 'lightblue'), ('PACIFIC AVE.', 300, 'green')]
        for k, (nm, pr, g) in enumerate(loose):
            dk = deed(nm, pr, g, seed=80 + k).resize((330, 206), Image.LANCZOS)
            t0 = c('Mortgage first') + 0.1 + 0.35 * k
            p.add(Obj(t0 - 0.3, dk, (100 + k * 300, 600 + (k % 2) * 30), how='up', angle=-3 + 3 * k),
                  Stamp(t0 + 0.2, (265 + k * 300, 705 + (k % 2) * 30), ['MORTGAGED'], angle=-8, size=36, seed=k + 3))
        p.add(Obj(c('Sell houses last') - 0.1, houses_row(3, h=150, gap=18, seed=2), (1120, 560), how='drop'),
              Txt(c('Sell houses last') + 0.2, (1390, 800), 'keep the houses standing', F('hand', 36), fill=SEP, anchor='mm'))
        p.add(Typed(c('Sell first, and you drop'), (1000, 860), 'sell houses first .. 19.1%', F('type', 38), fill=RED),
              Typed(c("and you're more likely"), (1000, 920), 'first to go broke .. 28.5%', F('type', 38), fill=RED),
              Line(c('Those houses are the only'), ring(1390, 640, 330, 120), width=4, dur=0.6),
              Stamp(c('the only thing paying you back') + 0.2, (1700, 330), ['MORTGAGE', 'FIRST'], angle=-8, size=56))


SCENES = {'07a_spend_or_save': SpendOrSave, '07b_how_far_to_build': HowFar, '07c_which_set_first': WhichFirst,
          '07d_one_set_or_spread': OneOrSpread, '07e_owe_money': OweMoney}

if __name__ == '__main__':
    main(SCENES)
