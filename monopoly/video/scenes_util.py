"""The utilities spin-off: are the Electric Company and the Water Works worth buying, and what would fix them?

    VIDEO=utilities python3 scenes_util.py u04_worthless [--sheet | --frame T]
"""
import math
import os

from PIL import Image, ImageDraw

from engine import H, W, clamp, die_face, ease_out, sf, txt
from kit import F, Counter, Fn, Grid100, Line, Obj, PagedScene, Stamp, Txt, Typed, card, ledger_paper, ledger_row, main, paste, pawn_layer, ring, \
    tick, with_alpha
from twenties import EXTRA_CROP, GOLD, NIGHT, PHOTOS, caption_bar, deco_opener, deco_title, photo_bg, photo_page, Plaque

SEP, RED, GREEN, INK = sf.SEPIA, sf.STAMP, sf.NOTE_GREEN, sf.INK
STEEL = (96, 104, 112)

# period photos (art/source/utilities); a page falls back to a dark Deco plate while one is missing
UTIL_PHOTOS = os.path.join(os.path.dirname(PHOTOS), 'utilities')
PHOTO = {'title': ['power_washington_dynamos_c1922.jpg'], 'price': ['water_cincinnati_filtration_plant_c1905.jpg'],
         'rent': ['power_detroit_edison_plant_c1905.jpg'], 'end': ['lights_use_electric_light_sign_1913.jpg']}
EXTRA_CROP.update({'power_washington_dynamos_c1922.jpg': (0.07, 0.02, 0.02, 0.07), 'lights_use_electric_light_sign_1913.jpg': (0.0, 0.09, 0.03, 0.08)})
CAPTION = {'title': 'DYNAMOS, c. 1920', 'price': 'CINCINNATI WATER WORKS, c. 1905', 'rent': 'DETROIT EDISON PLANT, c. 1905'}
FOCUS = {'title': (0.5, 0.55), 'price': (0.5, 0.5), 'rent': (0.55, 0.45), 'end': (0.5, 0.4)}


def photo_name(key):
    for n in PHOTO[key]:
        if os.path.exists(os.path.join(UTIL_PHOTOS, n)):
            return os.path.join('..', 'utilities', n)
    return None


def plain(seed):
    return sf.paper(seed=seed)


def night_plate(seed=0):
    img = Image.new('RGBA', (W, H), NIGHT + (255,))
    paper = sf.paper(seed=seed).convert('RGBA')
    return Image.blend(img, paper, 0.08)


def opener(scene, kicker, title, key, caption=None, t0=0, focus=(0.5, 0.45)):
    """A section title on a period photo, or on a dark plate until the photo is in."""
    name = photo_name(key)
    if name:
        return deco_opener(scene, 0, title, name, caption or CAPTION.get(key), crop_focus=FOCUS.get(key, (0.5, 0.5)), kicker=kicker, t0=t0,
                           focus=focus)
    p = scene.page(night_plate(7 + len(title)), t0, enter='fade', zoom=0.04)
    lines = title.count('\n') + 1
    p.add(Plaque(t0 + 0.15, deco_title(kicker, title, w=1180, h=330 + 110 * lines), (W / 2, H / 2 - 30)))
    return p


# ---------------------------------------------------------------- props, drawn in our own style (not the game's cards)
def bulb(d, cx, cy, r, col=INK):
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=col + (230,), width=4)
    d.rectangle([cx - r * 0.45, cy + r * 0.85, cx + r * 0.45, cy + r * 1.45], outline=col + (230,), width=4)
    for k in range(3):
        y = cy + r * (1.0 + 0.16 * k)
        d.line([(cx - r * 0.45, y), (cx + r * 0.45, y)], fill=col + (200,), width=2)
    d.line([(cx - r * 0.3, cy + r * 0.85), (cx - r * 0.2, cy), (cx, cy + r * 0.25), (cx + r * 0.2, cy), (cx + r * 0.3, cy + r * 0.85)],
           fill=col + (230,), width=3)
    for a in range(-150, -20, 26):  # rays
        t = math.radians(a)
        d.line([(cx + 1.35 * r * math.cos(t), cy + 1.35 * r * math.sin(t)), (cx + 1.7 * r * math.cos(t), cy + 1.7 * r * math.sin(t))],
               fill=col + (200,), width=3)


def tap(d, cx, cy, r, col=INK):
    d.rectangle([cx - r * 1.4, cy - r * 0.25, cx + r * 0.2, cy + r * 0.25], outline=col + (230,), width=4)        # pipe
    d.rectangle([cx - r * 0.25, cy - r * 0.8, cx + r * 0.15, cy - r * 0.25], outline=col + (230,), width=4)       # stem
    d.line([(cx - r * 0.75, cy - r * 0.8), (cx + r * 0.65, cy - r * 0.8)], fill=col + (230,), width=5)              # handle
    d.arc([cx - r * 0.1, cy - r * 0.25, cx + r * 0.9, cy + r * 0.75], 270, 90, fill=col + (230,), width=4)          # spout
    d.polygon([(cx + r * 0.45, cy + r * 0.95), (cx + r * 0.3, cy + r * 1.3), (cx + r * 0.45, cy + r * 1.45), (cx + r * 0.6, cy + r * 1.3)],
              outline=col + (230,), fill=(120, 150, 170, 160))                                                       # drop


def utility_deed(name, kind, seed=0, w=640, h=400):
    """A deed of sale for a utility, with an inked bulb or tap."""
    c = sf.paper(w, h, base=(238, 230, 206), seed=seed)
    d = ImageDraw.Draw(c)
    d.rectangle([12, 12, w - 12, h - 12], outline=INK + (200,), width=2)
    d.rectangle([20, 20, w - 20, h - 20], outline=INK + (120,), width=1)
    sf.hatch(c, (20, 20, 44, h - 20), spacing=5, angle=60, color=STEEL, alpha=170)
    txt(c, (w / 2 + 12, 58), 'Deed of Sale', F('script', 54), anchor='mm')
    txt(c, (w / 2 + 12, 112), 'ATLANTIC CITY, NEW JERSEY', F('sc', 20), fill=SEP, anchor='mm')
    (bulb if kind == 'electric' else tap)(d, w - 92, 186, 30)
    txt(c, (w / 2 - 20, 182), name, F('fat', 40), anchor='mm')
    txt(c, (w / 2 + 12, 248), 'for the consideration of', F('serif', 24), fill=SEP, anchor='mm')
    txt(c, (w / 2 + 12, 296), '$150', F('type', 48), anchor='mm')
    d.line([(90, h - 56), (w / 2 - 20, h - 56)], fill=INK + (160,), width=1)
    d.line([(w / 2 + 50, h - 56), (w - 60, h - 56)], fill=INK + (160,), width=1)
    txt(c, (90, h - 50), 'buyer', F('serif', 18), fill=SEP)
    txt(c, (w / 2 + 50, h - 50), 'the bank', F('serif', 18), fill=SEP)
    return c


def chance_card(w=760, h=430, seed=3):
    c = card(w, h, base=(240, 226, 196), seed=seed)
    d = ImageDraw.Draw(c)
    d.rectangle([16, 16, w - 16, h - 16], outline=(170, 96, 40, 200), width=3)
    txt(c, (w / 2, 80), 'CHANCE', F('deco', 74), fill=(170, 96, 40), anchor='mm')
    txt(c, (w / 2, 200), 'Advance to the nearest utility.', F('serif', 40), anchor='mm')
    txt(c, (w / 2, 262), 'If it is owned, roll the dice', F('serif', 34), fill=SEP, anchor='mm')
    txt(c, (w / 2, 306), 'and pay the owner ten times the roll.', F('serif', 34), fill=SEP, anchor='mm')
    bulb(d, 90, 360, 22, (170, 96, 40))
    tap(d, w - 100, 350, 22, (170, 96, 40))
    return c


def price_tag(text, w=240, h=104, seed=0, col=INK):
    c = sf.ink_layer((w, h))
    paper = sf.paper(w, h, base=(244, 236, 214), seed=seed).convert('RGBA')
    m = Image.new('L', (w, h), 0)
    ImageDraw.Draw(m).polygon([(0, h / 2), (h * 0.35, 0), (w, 0), (w, h), (h * 0.35, h)], fill=255)
    paper.putalpha(m)
    c.alpha_composite(paper)
    d = ImageDraw.Draw(c)
    d.polygon([(0, h / 2), (h * 0.35, 0), (w - 1, 0), (w - 1, h - 1), (h * 0.35, h - 1)], outline=INK + (200,), width=2)
    d.ellipse([h * 0.22, h / 2 - 9, h * 0.22 + 18, h / 2 + 9], outline=INK + (200,), width=2)
    txt(c, (w / 2 + 22, h / 2), text, F('type', 46), fill=col, anchor='mm')
    return c


def boxes(n, filled, cols=10, size=46, gap=10, colour=RED):
    rows = (n + cols - 1) // cols
    c = sf.ink_layer((cols * (size + gap), rows * (size + gap)))
    d = ImageDraw.Draw(c)
    for k in range(n):
        x, y = (k % cols) * (size + gap), (k // cols) * (size + gap)
        if k < filled:
            d.rectangle([x, y, x + size, y + size], fill=colour + (200,))
        d.rectangle([x, y, x + size, y + size], outline=INK + (200,), width=2)
    return c


def banner(img, t, t1, text, y, fnt, fill=INK, t2=None):
    if t < t1 or (t2 is not None and t >= t2):
        return
    al = clamp((t - t1) / 0.5)
    img.alpha_composite(Image.new('RGBA', (W, 120), (239, 228, 204, int(225 * al))), (0, y - 60))
    txt(img, (W / 2, y), text, fnt, fill=fill, anchor='mm', alpha=int(255 * al))


# ====================================================================== u01
class ColdOpen(PagedScene):
    def __init__(self):
        super().__init__('u01_cold_open')
        c = self.cue
        p = self.page(plain(301))
        p.add(Typed(0.3, (W / 2, 110), 'TWO SQUARES YOU SHOULD NEVER BUY?', F('deco', 74), anchor='mm', cps=28))
        p.add(Obj(c('The Electric Company'), utility_deed('ELECTRIC CO.', 'electric', 4), (180, 300), how='drop', angle=-4),
              Obj(c('the Water Works'), utility_deed('WATER WORKS', 'water', 5), (1060, 320), how='drop', angle=3))
        p = self.page(ledger_paper(302, rules=[460, 640], cols=()), c('I played more than') - 0.2)
        p.add(Txt(c('I played more than'), (W / 2, 90), 'games played', F('sc', 34), fill=SEP, anchor='mm'),
              Counter(c('I played more than'), c('to prove it') + 0.6, (W / 2, 170), 0, 5_200_000, lambda v: '{:,}'.format(int(v)),
                      F('type', 84), fill=RED, anchor='mm'),
              Txt(c('Skip them every'), (820, 300), 'Games won (%)', F('sc', 26), fill=SEP))
        ledger_row(p, c('Skip them every'), 380, 'Never buys a utility', 25.5, 24, fnt=F('type', 38), wash=sf.GROUP['green'])
        ledger_row(p, c('than the player who buys'), 560, 'Buys them', 25.1, 24, fnt=F('type', 38))
        p.add(Txt(c('you win a little more'), (170, 760), 'Skipping them wins a little more often.', F('hand', 44), fill=RED))
        p = self.page(plain(303), c('So I asked a different') - 0.2)
        p.add(Typed(c('What would it take'), (W / 2, 190), 'What would it take to fix them?', F('serif', 66), anchor='mm', cps=30))
        p.add(Obj(c('A lower price'), price_tag('$1??', w=300, seed=2), (300, 420), how='drop', angle=-3),
              Txt(c('A lower price') + 0.3, (520, 600), 'a lower price?', F('hand', 46), anchor='mm'),
              Obj(c('Or a bigger rent'), self.multiplier_card(), (1080, 380), how='drop', angle=3),
              Txt(c('Or a bigger rent') + 0.3, (1300, 600), 'a bigger rent?', F('hand', 46), anchor='mm'),
              Stamp(c('I tested both') + 0.2, (W / 2, 820), ['TESTED BOTH'], angle=-7, size=64))
        opener(self, 'A MONOPOLY STUDY  ·  5,200,000 GAMES', 'FIX THE\nUTILITIES', 'title', t0=c('And the fix turned out') - 0.2)

    @staticmethod
    def multiplier_card():
        c = card(440, 240, seed=9)
        txt(c, (220, 70), 'RENT', F('sc', 34), fill=SEP, anchor='mm')
        txt(c, (220, 150), '?× the roll', F('type', 56), anchor='mm')
        return c


# ====================================================================== u02
class TheRules(PagedScene):
    def __init__(self):
        super().__init__('u02_the_rules')
        c = self.cue
        p = self.page(plain(311))
        p.add(Txt(0.3, (140, 70), 'HOW THEY WORK', F('scb', 46), fill=GREEN),
              Obj(c('Each utility costs') - 0.3, utility_deed('ELECTRIC CO.', 'electric', 6), (120, 160), how='drop', angle=-3),
              Txt(c('Instead of a fixed rent'), (140, 620), 'No fixed rent: it depends on the roll', F('hand', 40), fill=SEP),
              Txt(c('Instead of a fixed rent') + 0.4, (140, 672), 'that brought the player there.', F('hand', 40), fill=SEP),
              Typed(c('Own one utility'), (900, 190), 'Own one:   4 × the roll', F('serif', 54)),
              Typed(c('Own both'), (900, 280), 'Own both:  10 × the roll', F('serif', 54)))
        t_dice = c('On an average roll')
        p.add(Obj(t_dice, die_face(3, 120, 1), (900, 420), how='drop', angle=-10),
              Obj(t_dice + 0.15, die_face(4, 120, 2), (1050, 430), how='drop', angle=8),
              Txt(t_dice + 0.4, (1210, 470), '= 7 on average', F('serif', 50)),
              Typed(c('that\'s twenty-eight'), (900, 640), 'one:   4 × 7 = $28', F('type', 50)),
              Typed(c('or seventy'), (900, 720), 'both: 10 × 7 = $70', F('type', 50), fill=RED))
        p = self.page(plain(312), c("There's also a Chance card") - 0.2)
        p.add(Obj(c("There's also a Chance card"), chance_card(), (120, 170), how='drop', angle=-3),
              Stamp(c('always pays ten times') + 0.2, (520, 790), ['ALWAYS 10 ×'], angle=-6, size=60),
              Txt(c('Each utility gets landed'), (1000, 170), 'Each one is landed on', F('serif', 48)),
              Txt(c('Each utility gets landed') + 0.3, (1000, 235), 'about once every 30 turns.', F('serif', 48)),
              Obj(c('once every thirty') + 0.2, boxes(30, 1), (1000, 340), how='fade', shadow=None),
              Txt(c('once every thirty') + 0.8, (1000, 540), 'one turn in thirty', F('hand', 40), fill=RED))


# ====================================================================== u03
class TheTest(PagedScene):
    def __init__(self):
        super().__init__('u03_the_test')
        c = self.cue
        p = self.page(plain(321))
        p.add(Txt(0.3, (170, 70), 'THE TEST', F('scb', 46), fill=GREEN),
              Typed(c('played by the official'), (170, 150), 'The official rules, played the way people actually play.', F('serif', 42)))
        brass, oxblood = sf.BRASS, (150, 52, 40)
        t1, t2 = c('one player follows a rule'), c('the other three play')
        for k in range(4):
            t0 = t1 if k == 0 else t2 + 0.25 * k
            p.add(Obj(t0, pawn_layer(150, oxblood if k == 0 else brass), (300 + k * 360, 280), how='drop'),
                  Txt(t0 + 0.2, (300 + k * 360 + 90, 560), 'FOLLOWS A RULE' if k == 0 else 'TYPICAL', F('sc', 30),
                      fill=RED if k == 0 else SEP, anchor='mm'))
        p.add(Txt(c('who buys whatever'), (170, 680), 'A typical player buys whatever they land on.', F('hand', 44), fill=SEP))
        p = self.page(plain(322), c('A typical player wins') - 0.2)
        p.add(Grid100(c('A typical player wins'), (300, 230), 25, dur=1.4, size=40, gap=8, wash=SEP),
              Typed(c('wins about twenty-five') + 0.2, (900, 260), '25%', F('deco', 140)),
              Txt(c('wins about twenty-five') + 0.8, (906, 430), 'of games, for a typical player', F('serif', 46)),
              Typed(c('Every result'), (906, 560), '100,000 games per result', F('type', 46), fill=RED),
              Txt(c('a hundred thousand games') + 0.6, (906, 640), '5,200,000 games in all', F('type', 34), fill=SEP))


# ====================================================================== u04
class Worthless(PagedScene):
    def __init__(self):
        super().__init__('u04_worthless')
        c = self.cue
        p = self.page(ledger_paper(331, rules=[440, 640], cols=()))
        p.add(Txt(0.3, (170, 80), 'TEST ONE: SKIP THEM', F('scb', 44), fill=GREEN),
              Txt(0.5, (820, 220), 'Games won (%)', F('sc', 26), fill=SEP))
        ledger_row(p, c('A player who never buys'), 300, 'Never buys a utility', 25.5, 24, fnt=F('type', 38), wash=sf.GROUP['green'])
        ledger_row(p, c('A player who buys them'), 500, 'Buys them', 25.1, 24, fnt=F('type', 38))
        p.add(Stamp(c('Skipping them is slightly') + 0.3, (1460, 800), ['SKIPPING', 'WINS'], angle=-8, size=62))
        p = self.page(ledger_paper(332, rules=[350 + 140 * k for k in range(4)], cols=()), c('Test two') - 0.1)
        p.add(Txt(c('Test two') + 0.2, (170, 70), 'TEST TWO: A FREE GIFT', F('scb', 44), fill=GREEN),
              Typed(c('What if I just gave'), (170, 150), 'Handed over free, before the first roll.  Games won (%):', F('serif', 38)))
        sc, xb = 9, 700
        rows = [("You'd win about thirty-eight", 'Both utilities', 38.2, RED, None),
                ('the two browns', 'The browns', 58.5, INK, sf.GROUP['brown']),
                ('All four railroads', 'All four railroads', 87.1, INK, None),
                ('The oranges', 'The oranges', 98.6, INK, sf.GROUP['orange'])]
        for k, (ph, lab, v, col, wash) in enumerate(rows):
            ledger_row(p, c(ph), 280 + 140 * k, lab, v, sc, x_bar=xb, fnt=F('type', 38), fill=col, wash=wash)
        x25 = xb + 25.1 * sc
        p.add(Line(c("You'd win about thirty-eight") - 0.3, [(x25, 250), (x25, 760)], width=3, dur=0.4),
              Txt(c("You'd win about thirty-eight") - 0.2, (x25, 785), 'a typical player: 25', F('hand', 30), fill=SEP, anchor='mm'),
              Typed(c('Dollar for dollar') + 0.3, (170, 880), 'Per $100 of printed price:  utilities 4  ·  railroads 8  ·  browns 28',
                    F('type', 32), fill=SEP),
              Stamp(c('the worst thing on the board') + 0.2, (1500, 930), ['WORST ON', 'THE BOARD'], angle=-7, size=56))


# ====================================================================== u05
class Ceiling(PagedScene):
    def __init__(self):
        super().__init__('u05_the_ceiling')
        c = self.cue
        p = self.page(ledger_paper(341, rules=[400, 560, 720], cols=()))
        p.add(Txt(0.3, (170, 70), 'ROUNDS TO EARN BACK THE PRICE', F('scb', 44), fill=GREEN),
              Typed(c('Per dollar'), (170, 150), 'Per dollar, their rent isn’t bad.', F('serif', 46)),
              Txt(c('Against three opponents'), (820, 250), 'rounds, against three opponents (fewer is better)', F('sc', 24), fill=SEP))
        sc = 14
        ledger_row(p, c('both utilities earn back'), 330, 'Both utilities', 22, sc, fnt=F('type', 38), wash=sf.GROUP['green'], fmt='%.0f')
        ledger_row(p, c('Two railroads take'), 490, 'Two railroads', 32, sc, fnt=F('type', 38), fmt='%.0f')
        ledger_row(p, c('A colour set with no houses'), 650, 'A colour set, no houses', 62, sc, fnt=F('type', 38), fmt='%.0f')
        p = self.page(plain(342), c('The problem is the ceiling') - 0.2)
        p.add(Txt(c('The problem is the ceiling') + 0.1, (170, 70), 'THE CEILING', F('scb', 46), fill=GREEN),
              Txt(c('The problem is the ceiling') + 0.3, (170, 145), 'Rent per round, against three opponents', F('serif', 40)))
        sc, xb = 3.2, 600
        rows = [('Both together bring in', 'Both utilities', 13.5, RED, None, 0),
                ('A colour set with hotels', 'Light blue, hotels', 139, INK, sf.GROUP['lightblue'], 0),
                ('A colour set with hotels', 'Orange, hotels', 304, INK, sf.GROUP['orange'], 0.6),
                ('A colour set with hotels', 'Green, hotels', 365, INK, sf.GROUP['green'], 1.2)]
        for k, (ph, lab, v, col, wash, dt) in enumerate(rows):
            ledger_row(p, c(ph) + dt, 260 + 150 * k, lab, v, sc, x_bar=xb, fnt=F('type', 36), fill=col, wash=wash, fmt='$%.2f' if v < 20 else '$%.0f')
        xc = xb + 13.5 * sc + 6
        p.add(Line(c("You can't build") + 0.1, [(xc + 30, 330), (xc + 30, 790)], width=4, dur=0.5),
              Txt(c("You can't build") + 0.4, (xc + 46, 790), 'the ceiling: no houses, ever', F('hand', 34), fill=RED),
              Txt(c('never grow up'), (170, 900), 'The utilities never grow up.', F('serif', 54)),
              Stamp(c("You can't build") + 0.6, (1560, 900), ['NO HOUSES'], angle=-8, size=62))


# ====================================================================== u06
class FixPrice(PagedScene):
    def __init__(self):
        super().__init__('u06_fix_the_price')
        c = self.cue
        opener(self, 'FIX  ONE', 'CUT THE\nPRICE', 'price')
        p = self.page(ledger_paper(351, rules=[360 + 170 * k for k in range(4)], cols=()), c('Keep the rent the same') - 0.2)
        p.add(Txt(c('Keep the rent the same'), (170, 70), 'FIX ONE: CUT THE PRICE', F('scb', 44), fill=GREEN),
              Txt(c('Keep the rent the same') + 0.3, (170, 145), 'Rent stays 4× and 10× the roll.', F('serif', 40), fill=SEP))
        rows = [(c('Keep the rent the same') + 0.6, '$150', 'today: skipping them wins', INK),
                (c('a hundred and twenty dollars each'), '$120', 'buying them finally beats skipping them', GREEN),
                (c('At about eighty-five'), '$85', 'as good as the railroads, per dollar', GREEN),
                (c('Even if they were free'), 'FREE', 'still under 40% of games', RED)]
        for k, (t0, tag, note, col) in enumerate(rows):
            y = 250 + 170 * k
            p.add(Obj(t0, price_tag(tag, seed=10 + k, col=col), (170, y), how='left', angle=0, shadow=(6, 8, 6, 70)),
                  Typed(t0 + 0.3, (470, y + 30), note, F('serif', 46), fill=col if col != INK else SEP))
        p.add(Stamp(c('A lower price can') + 0.2, (1500, 960), ['PRICE CAN’T', 'FIX IT'], angle=-7, size=56))


# ====================================================================== u07
class FixRent(PagedScene):
    def __init__(self):
        super().__init__('u07_fix_the_rent')
        c = self.cue
        opener(self, 'FIX  TWO', 'RAISE THE\nRENT', 'rent')
        p = self.page(ledger_paper(361, rules=[320 + 125 * k for k in range(5)], cols=(1240,)), c('At six and fifteen') - 0.4)
        p.add(Txt(c('At six and fifteen') - 0.3, (170, 60), 'FIX TWO: RAISE THE RENT', F('scb', 44), fill=GREEN),
              Txt(c('At six and fifteen') - 0.2, (170, 130), 'Rent: one / both', F('sc', 26), fill=SEP),
              Txt(c('At six and fifteen') - 0.2, (560, 130), 'both utilities, handed over free: games won (%)', F('sc', 26), fill=SEP))
        sc, xb = 6.4, 560
        rows = [(c('At six and fifteen') - 0.2, '4× / 10×', 38.2, 'today', INK, None),
                (c('At six and fifteen'), '6× / 15×', 43.9, 'skipping them starts to hurt', INK, None),
                (c('At eight and twenty'), '8× / 20×', 49.5, 'as good as railroads, per $', GREEN, sf.GROUP['green']),
                (c('At twelve and thirty'), '12× / 30×', 59.9, 'worth a colour set (browns)', INK, None),
                (c('thirty-two and eighty'), '32× / 80×', 87.3, 'worth all four railroads', INK, None)]
        for k, (t0, lab, v, note, col, wash) in enumerate(rows):
            y = 200 + 125 * k
            ledger_row(p, t0, y, lab, v, sc, x_bar=xb, x_label=170, fnt=F('type', 40), fill=col if col != GREEN else INK, wash=wash)
            p.add(Txt(t0 + 0.6, (1270, y + 6), note, F('hand', 34), fill=col if col != INK else SEP))
        p.add(Line(c("double today's rent") + 0.2, ring(390, 470, 250, 62), width=4, dur=0.6),
              Stamp(c("That's eight times") + 0.2, (1500, 930), ['8 × TODAY’S', 'RENT'], angle=-7, size=54))


# ====================================================================== u08
def certificate():
    img = sf.paper(base=(236, 229, 210), seed=7)
    sf.guilloche_frame(img, inset=30, band=56)
    txt(img, (W / 2, 160), 'THE DOUBLE-RENT RULE', F('deco', 96), anchor='mm')
    txt(img, (W / 2, 250), 'A HOUSE RULE FOR THE ELECTRIC COMPANY AND THE WATER WORKS', F('sc', 30), fill=SEP, anchor='mm')
    return img


class TheFix(PagedScene):
    def __init__(self):
        super().__init__('u08_the_fix')
        c = self.cue
        p = self.page(certificate())
        lines = [(c('Keep the price at'), 'Price:', '$150 each, as printed.'),
                 (c('eight times the dice'), 'Own one:', '8 × the roll.'),
                 (c('twenty times with both'), 'Own both:', '20 × the roll.'),
                 (c('twenty times with both') + 0.8, 'Chance card:', '20 × the roll.')]
        for k, (t0, a, b) in enumerate(lines):
            y = 350 + 95 * k
            p.add(Txt(t0, (720, y), a, F('serifb', 48), fill=GREEN, anchor='ra'), Typed(t0 + 0.2, (750, y), b, F('serif', 48), cps=30))
        p.add(Stamp(c('easy to remember') + 0.3, (1480, 840), ['HOUSE', 'RULE'], angle=-9, size=72),
              Obj(c('two dead squares') + 0.2, utility_deed('ELECTRIC CO.', 'electric', 11).resize((352, 220), Image.LANCZOS), (250, 700), how='drop', angle=-6),
              Obj(c('two dead squares') + 0.45, utility_deed('WATER WORKS', 'water', 12).resize((352, 220), Image.LANCZOS), (640, 715), how='drop', angle=4))
        p = self.page(plain(372), c('Until the rules change') - 0.2)
        p.add(Txt(c('Until the rules change') + 0.1, (170, 80), 'IN A NORMAL GAME', F('scb', 46), fill=GREEN))
        todo = [(c("Don't chase"), 'Don’t chase the utilities.'),
                (c("Don't bid them up"), 'Don’t bid them up at auction.'),
                (c('And if the money'), 'Money better spent on houses?  Skip them.'),
                (c("It won't cost you"), 'It costs you nothing.')]
        for k, (t0, line) in enumerate(todo):
            y = 220 + 120 * k
            p.add(Typed(t0, (250, y), line, F('serif', 52), cps=30))
            tick(p, t0 + 0.8, 170, y + 6, size=40, seed=k)
        end = photo_name('end')
        t0 = c('Would you play') - 0.2
        p = self.page(photo_bg(end, FOCUS['end'], darken=0.2) if end else night_plate(19), t0, enter='fade', zoom=0.05, focus=(0.0, 0.3))
        p.add(Fn(t0, lambda img, t, a: banner(img, t, c('Would you play'), 'Would you play the double-rent rule?', 900, F('deco', 64), RED)),
              Txt(c('Tell me in the comments'), (60, H - 80), 'Not affiliated with or endorsed by Hasbro. MONOPOLY is a trademark of Hasbro.',
                  F('type', 22), fill=(226, 216, 196), anchor='lb'))


SCENES = {'u01_cold_open': ColdOpen, 'u02_the_rules': TheRules, 'u03_the_test': TheTest, 'u04_worthless': Worthless,
          'u05_the_ceiling': Ceiling, 'u06_fix_the_price': FixPrice, 'u07_fix_the_rent': FixRent, 'u08_the_fix': TheFix}

if __name__ == '__main__':
    main(SCENES)
