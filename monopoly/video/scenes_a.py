"""Scenes 03-06: how we tested it, buy or pass, trading, railroads.

    python3 scenes_a.py 04_buy_or_pass [--sheet | --frame 12.5 | --captions]
"""
import math

import numpy as np
from PIL import Image, ImageDraw

from engine import H, W, clamp, ease_out, lerp, sf, smooth, txt
from twenties import deco_opener, photo_page
from kit import (F, Bar, Counter, Fn, Grid100, Line, Obj, PagedScene, Stamp, Telegram, Txt, Typed, board_layer,
                 deed, fig_page, ledger_paper, ledger_row, main, note_card, padlock, paste, pawn_layer,
                 ring, shadowed, tick, with_alpha)

SEP = sf.SEPIA
RED = sf.STAMP
GREEN = sf.NOTE_GREEN


def plain(seed):
    return sf.paper(seed=seed)


# ====================================================================== 03
class HowWeTested(PagedScene):
    def __init__(self):
        super().__init__('03_how_we_tested_it')
        c = self.cue
        BX, BY, S = 100, 80, 920
        self.board = board_layer(S)  # corner at (10, 10) inside the layer
        a = np.asarray(self.board.split()[-1], np.float32)
        yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
        cx = cy = 10 + S / 2
        self.ang = (np.degrees(np.arctan2(yy - cy, xx - cx)) - 45) % 360  # 0 at the start corner, rising round the board
        self.board_a = a

        # A: the board inks itself; the official rules are ticked off
        p = self.page(plain(61))
        t_ink = 0.3

        def ink_board(img, t, al, ox=BX - 10, oy=BY - 10):
            f = clamp((t - t_ink) / 3.2)
            if f >= 1:
                paste(img, self.board, (ox, oy)); return
            lay = self.board.copy()
            lay.putalpha(Image.fromarray((self.board_a * (self.ang <= f * 360)).astype(np.uint8)))
            paste(img, lay, (ox, oy))
        p.add(Fn(t_ink, ink_board))
        px = 1120
        p.add(Txt(0.5, (px, 110), 'Fig. 1.', F('serifb', 64)),
              Txt(0.9, (px, 195), 'THE GAME, PLAYED BY A COMPUTER', F('sc', 30), fill=SEP),
              Typed(c('official rules') - 0.6, (px, 280), 'By the official rules:', F('serif', 44)))
        items = [('The real board', 'the real board'), ('the real cards', 'the real cards'), ('auctions', 'auctions'),
                 ('mortgages', 'mortgages'), ("and the bank's", 'the bank’s 32 houses\n   and 12 hotels')]
        for k, (ph, s) in enumerate(items):
            y = 370 + k * 72
            p.add(Typed(c(ph), (px + 70, y), s, F('serif', 40), spacing=10))
            tick(p, c(ph) + 0.15, px + 10, y + 4, seed=k)

        # B: how people actually play (a 1922 game night), then a set for a set
        photo_page(self, 'games_women_mah_jong_1922.jpg', c('Then I taught it') - 0.2, 'GAME NIGHT, 1922', focus=(0.45, 0.6))
        p = self.page(plain(62), c('Everyone typically buys') - 0.35)
        p.add(Txt(c('Then I taught it') + 0.3, (160, 90), 'HOW PEOPLE ACTUALLY PLAY', F('scb', 48), fill=GREEN),
              Typed(c('Everyone typically buys'), (160, 190), '1.  Buy what you land on, if you can afford it.', F('serif', 44)),
              Typed(c('And nobody hands'), (160, 268), '2.  Nobody hands you the card that finishes your set,', F('serif', 44)),
              Typed(c('unless you hand them'), (160, 330), '     unless you hand them the card that finishes theirs.', F('serif', 44)))
        d_red = deed('KENTUCKY AVE.', 220, 'red', seed=3).resize((544, 340), Image.LANCZOS)
        d_org = deed('ST. JAMES PLACE', 180, 'orange', seed=4).resize((544, 340), Image.LANCZOS)
        L_red, _ = shadowed(d_red.rotate(-3, expand=True, resample=Image.BICUBIC))
        L_org, _ = shadowed(d_org.rotate(2, expand=True, resample=Image.BICUBIC))
        t_in, t_swap = c('And nobody hands') + 0.3, c('unless you hand them') + 0.6

        def swap(img, t, al):
            e = smooth((t - t_swap) / 1.1)
            for lay, x0, x1, lab in ((L_red, 150, 1230, 'yours: finishes their reds'), (L_org, 1230, 150, 'theirs: finishes your oranges')):
                a = ease_out((t - t_in) / 0.4)
                x = lerp(x0, x1, e)
                y = 430 - 90 * math.sin(math.pi * e) * (1 if x0 < x1 else -0.4)
                paste(img, with_alpha(lay, a), (x, y))
                if e < 0.05:
                    txt(img, (x0 + 300, 850), lab, F('hand', 34), fill=SEP, anchor='mm', alpha=int(255 * a))
        p.add(Fn(t_in, swap))
        p.add(Stamp(t_swap + 1.2, (W / 2, 700), ['A SET', 'FOR A SET'], angle=-7, size=56))

        # C: the board locks up; the endgame deals begin
        p = self.page(plain(63), c("If you've played") - 0.2)
        p.add(Fn(-1, lambda img, t, al: paste(img, self.board, (W / 2 - self.board.size[0] / 2, BY - 10))))
        lock = padlock(200)
        p.add(Obj(c('the board locks up'), lock, (W / 2 - 100, 330), how='drop'))
        p.add(Stamp(c('Nobody can make') + 0.3, (W / 2, 660), ['STALLED'], angle=-9, size=60))
        p.add(Counter(c('drags') - 0.6, c('So when that') + 0.5, (W / 2, 770), 31, 58, lambda v: 'ROUND %d' % v, F('type', 34),
                      fill=SEP, anchor='mm'))
        tg = Telegram(c('They start dealing'), (980, 470), ['I’ll buy your Kentucky Ave.', 'for $550 cash.',
                                                          'It finishes my reds.'], w=880, angle=-2, how='right', wps=6)
        p.add(tg, Stamp(c('for a price') + 0.2, (1640, 880), ['SOLD'], angle=-10, size=70))

        # D: one player makes the decision; three typical players
        p = self.page(plain(64), c('To test a decision') - 0.2)
        brass, oxblood = sf.BRASS, (150, 52, 40)
        for k in range(4):
            col = oxblood if k == 0 else brass
            p.add(Obj(c('To test a decision') + 0.4 + (0 if k == 0 else 1.0 + 0.25 * k), pawn_layer(140, col), (330 + k * 340, 120), how='drop'))
            lab = 'MAKES THE DECISION' if k == 0 else 'TYPICAL'
            p.add(Txt(c('To test a decision') + 0.6 + (0 if k == 0 else 1.0 + 0.25 * k), (330 + k * 340 + 84, 395), lab, F('sc', 28),
                      fill=RED if k == 0 else SEP, anchor='mm'))
        p.add(Line(c('one player makes it') + 0.5, ring(414, 200, 140, 150), width=4, dur=0.6))
        gx, gy = 330, 450
        p.add(Grid100(c('At a four-player'), (gx, gy + 20), 25, dur=1.6, size=40, gap=8, wash=sf.SEPIA))
        p.add(Typed(c('wins about twenty-five') + 0.2, (900, 520), '25%', F('deco', 130), fill=sf.INK),
              Txt(c('wins about twenty-five') + 0.8, (906, 680), 'of games, for a typical player', F('serif', 46)),
              Txt(c('Going first helps'), (906, 760), 'goes first: 28%     goes last: 22%', F('type', 40), fill=SEP),
              Txt(c('so every player I tested'), (906, 830), 'so every tested player takes every seat', F('hand', 34), fill=SEP))

        # E: the games, counted on ticker tape
        p = self.page(plain(65), c('Every result') - 0.2)
        tape = self.ticker_tape()
        p.add(Typed(c('Every result') + 0.3, (W / 2, 190), 'EVERY RESULT:', F('sc', 44), fill=SEP, anchor='mm'),
              Typed(c('a hundred thousand') - 0.1, (W / 2, 330), '100,000 GAMES', F('deco', 120), anchor='mm', cps=20))
        p.add(Fn(c('Every result'), lambda img, t, al: self.draw_tape(img, t, tape)))
        p.add(Txt(c('More than thirteen') - 0.2, (W / 2, 820), 'IN ALL', F('sc', 40), fill=SEP, anchor='mm'),
              Counter(c('More than thirteen') - 0.2, self.dur - 0.6, (W / 2, 930), 0, 13_400_000, lambda v: '{:,} games'.format(int(v)),
                      F('type', 76), fill=RED, anchor='mm'))

    def ticker_tape(self):
        r = np.random.default_rng(7)
        s = '   '.join('GAME %s  WON BY PLAYER %d  IN %d TURNS' % ('{:,}'.format(13_399_000 + k), r.integers(1, 5), r.integers(60, 400))
                       for k in range(14))
        f = F('type', 34)
        w = int(ImageDraw.Draw(sf.ink_layer((4, 4))).textlength(s, font=f)) + 80
        tape = sf.paper(w, 76, base=(246, 240, 222), seed=66)
        d = ImageDraw.Draw(tape)
        d.line([(0, 2), (w, 2)], fill=sf.INK + (90,), width=2); d.line([(0, 73), (w, 73)], fill=sf.INK + (90,), width=2)
        txt(tape, (40, 38), s, f, anchor='lm')
        lay, _ = shadowed(tape, offset=(6, 10), blur=8, alpha=80)
        return lay

    def draw_tape(self, img, t, tape):
        x = -((t - self.cue('Every result')) * 260) % (tape.size[0] - W - 100)
        paste(img, tape, (-x - 40, 470))


# ====================================================================== 04
class BuyOrPass(PagedScene):
    def __init__(self):
        super().__init__('04_buy_or_pass')
        c = self.cue
        deco_opener(self, 1, 'TO BUY,\nOR TO PASS', 'jazz_stutz_roadster_1920.jpg', 'A NEW STUTZ ROADSTER, 1920', focus=(0.4, 0.6))

        # the deed on the desk
        p = self.page(plain(71), c('You land on') - 0.1)
        dd = deed('KENTUCKY AVE.', 220, 'red', seed=3)
        p.add(Obj(c('You land on') + 0.5, dd, (230, 300), how='drop', angle=-3))
        p.add(Txt(c('Buy it, or pass') - 0.1, (1110, 330), 'Buy it?', F('deco', 96)),
              Txt(c('or pass') - 0.1, (1110, 470), 'Or pass?', F('deco', 96), fill=sf.SEPIA))
        note = sf.banknote(100, (40, 92, 70), 31).rotate(8, expand=True, resample=Image.BICUBIC)
        p.add(Obj(c('protect your cash'), note, (1080, 650), how='right'))

        # the cushion ladder
        p = self.page(ledger_paper(72, rules=[300 + 110 * k for k in range(6)], cols=(1420,)), c('A player who passes') - 0.2)
        p.add(Txt(c('A player who passes') + 0.2, (170, 70), 'PASS IF BUYING WOULD LEAVE YOU WITH LESS THAN ...', F('scb', 38), fill=GREEN),
              Txt(c('A player who passes') + 0.2, (170, 200), 'Cash left after buying', F('sc', 26), fill=SEP),
              Txt(c('A player who passes') + 0.2, (640, 200), 'Games won (%)', F('sc', 26), fill=SEP))
        sc = 24
        rows = [('Always buy', 25.1, c('A player who passes')), ('$100', 25.3, c('Keep a hundred')),
                ('$200', 25.2, c('or two in hand')), ('$300', 24.6, c('But the more careful')), ('$500', 22.7, c('wins about twenty-three'))]
        for k, (lab, v, t0) in enumerate(rows):
            ledger_row(p, t0, 240 + 110 * k, lab, v, sc, fill=RED if v < 24 else sf.INK, x_bar=640)
        p.add(Txt(c('you lose nothing'), (1520, 330), 'a hundred or two:\nno cost', F('hand', 34), fill=SEP, spacing=8),
              Line(c('the more it costs') - 0.4, [(1470, 330), (1470, 720)], width=5, dur=0.6),
              Line(c('the more it costs') + 0.2, [(1450, 690), (1470, 725), (1490, 690)], width=5, dur=0.2),
              Txt(c('the more it costs'), (1520, 560), 'more careful,\nfewer wins', F('hand', 34), fill=RED, spacing=8))

        # the card you never pass on
        p = self.page(plain(73), c("And there's one card") - 0.2)
        kd = deed('KENTUCKY AVE.', 220, 'red', seed=3).resize((480, 300), Image.LANCZOS)
        idd = deed('INDIANA AVE.', 220, 'red', seed=5).resize((480, 300), Image.LANCZOS)
        il = deed('ILLINOIS AVE.', 240, 'red', seed=6).resize((480, 300), Image.LANCZOS)
        p.add(Obj(c("And there's one card") + 0.3, kd, (130, 200), how='drop', angle=-3),
              Obj(c("And there's one card") + 0.6, idd, (640, 190), how='drop', angle=2),
              Txt(c("And there's one card") + 0.3, (640, 110), 'YOURS', F('sc', 34), fill=SEP, anchor='mm'))
        p.add(Obj(c('the one that finishes a set'), il, (1220, 260), how='right', angle=-4),
              Stamp(c('the one that finishes a set') + 0.9, (1460, 420), ['BUY IT'], angle=-10, size=74),
              Txt(c('or prevents someone else') + 0.1, (1220, 620), '... or the one that stops\nsomeone else finishing theirs.',
                  F('serif', 40), spacing=10))
        bal = deed('BALTIC AVE.', 60, 'brown', seed=8).resize((420, 262), Image.LANCZOS)
        p.add(Obj(c('If you have to mortgage') - 0.1, bal, (200, 640), how='left', angle=3),
              Stamp(c('to buy it, do it') - 0.2, (410, 780), ['MORTGAGED'], angle=-6, size=48),
              Txt(c('do it') + 0.2, (700, 760), 'mortgage something else to pay for it', F('hand', 34), fill=RED))

        # every property is a trading chip
        p = self.page(plain(74), c('Why does buying') - 0.2)
        hand = [('ORIENTAL AVE.', 100, 'lightblue'), ('ST. CHARLES PL.', 140, 'pink'), ('NEW YORK AVE.', 200, 'orange'),
                ('ATLANTIC AVE.', 260, 'yellow'), ('PACIFIC AVE.', 300, 'green')]
        for k, (nm, pr, g) in enumerate(hand):
            dk = deed(nm, pr, g, seed=20 + k).resize((400, 250), Image.LANCZOS)
            ang = (k - 2) * 9
            p.add(Obj(c('Because every property') - 0.3 + 0.18 * k, dk, (420 + k * 170, 470 + abs(k - 2) * 40), how='up', angle=-ang))
        p.add(Typed(c('Why does buying'), (W / 2, 140), 'Why does buying matter so much?', F('serif', 54), anchor='mm'),
              Txt(c('every property is a trading chip'), (W / 2, 260), 'Every property is a trading chip.', F('deco', 76), fill=RED, anchor='mm'),
              Txt(c("You can't make a deal"), (W / 2, 1000), 'You can’t make a deal with cards you don’t have.', F('hand', 40), fill=SEP,
                  anchor='mm'))

        # passed: to auction
        p = self.page(plain(75), c('Now, in the official') - 0.2)
        p.add(Obj(c('Now, in the official') + 0.3, dd, (200, 330), how='drop', angle=-3),
              Stamp(c("doesn't just sit") - 0.2, (520, 520), ['PASSED'], angle=-12, size=70))
        p.add(Obj(c('It goes to auction') - 0.2, self.broadside(), (1000, 90), how='drop', angle=2))
        p.add(Typed(c('how high do you bid'), (200, 900), 'How high do you bid?', F('serif', 54), fill=sf.INK))

        # the bid ladder
        p = self.page(ledger_paper(76, rules=[360 + 150 * k for k in range(4)], cols=()), c('Stop at the printed') - 0.2)
        p.add(Txt(c('Stop at the printed') + 0.3, (170, 80), 'AT AUCTION, BID UP TO ...', F('scb', 42), fill=GREEN),
              Txt(c('Stop at the printed') + 0.3, (820, 220), 'Games won (%)', F('sc', 26), fill=SEP))
        sc = 24
        ledger_row(p, c('Lowball at half'), 280, 'half the price', 24.1, sc, fill=RED, fnt=F('type', 40))
        ledger_row(p, c('Stop at the printed'), 430, 'the printed price', 25.1, sc, fnt=F('type', 40))
        ledger_row(p, c('Bid up to one'), 580, '1.5 times the price', 26.5, sc, fnt=F('type', 40), wash=sf.GROUP['green'])
        p.add(Txt(c('and you\'re a typical player') - 0.3, (180, 485), 'what a typical player does', F('hand', 30), fill=SEP),
              Stamp(c('twenty-six and a half') + 0.6, (1560, 780), ['BID UP TO', '1.5 TIMES'], angle=-8, size=72))

        # no auctions
        p = self.page(plain(77), c('And if your family skips') - 0.2)
        rule = note_card(['HOUSE RULE:', 'no auctions. A property', 'nobody buys stays with the bank.'], F('type', 38), w=780, seed=7)
        p.add(Obj(c('And if your family skips') + 0.3, rule, (150, 160), how='drop', angle=-2))
        p.add(Obj(c('because the property') - 0.2, dd, (1080, 170), how='drop', angle=3),
              Stamp(c('just sits there unsold') - 0.1, (1400, 400), ['UNSOLD'], angle=-8, size=62))
        p.add(Typed(c('passing costs you a little less'), (170, 640), 'Pass if left with under $500:', F('type', 38)),
              Typed(c('passing costs you a little less') + 0.9, (230, 720), 'with auctions .......... 22.7', F('type', 38)),
              Typed(c('passing costs you a little less') + 1.8, (230, 790), 'without auctions ....... 23.2', F('type', 38)),
              Txt(c('instead of going to somebody'), (170, 900), 'Nobody else picks it up at a bargain.', F('hand', 38), fill=SEP))

    def broadside(self):
        w, h = 760, 860
        b = sf.paper(w, h, base=(232, 220, 190), seed=78)
        d = ImageDraw.Draw(b)
        d.rectangle([14, 14, w - 14, h - 14], outline=sf.INK + (220,), width=4)
        txt(b, (w / 2, 120), 'AUCTION!', F('fat', 130), anchor='mm')
        d.line([(60, 210), (w - 60, 210)], fill=sf.INK + (200,), width=3)
        txt(b, (w / 2, 270), 'TO BE SOLD TO THE HIGHEST BIDDER', F('sc', 30), anchor='mm')
        txt(b, (w / 2, 380), 'KENTUCKY\nAVENUE', F('deco', 96), anchor='mm', align='center', spacing=6)
        txt(b, (w / 2, 520), 'ATLANTIC CITY, N.J.', F('sc', 30), fill=SEP, anchor='mm')
        d.line([(60, 570), (w - 60, 570)], fill=sf.INK + (200,), width=3)
        txt(b, (w / 2, 640), 'LISTED AT $220', F('type', 44), anchor='mm')
        txt(b, (w / 2, 720), 'BIDDING OPEN TO ALL PLAYERS', F('sc', 26), fill=SEP, anchor='mm')
        txt(b, (w / 2, 790), 'TERMS: CASH', F('fat', 40), fill=RED, anchor='mm')
        return b


# ====================================================================== 05
class Trading(PagedScene):
    def __init__(self):
        super().__init__('05_trading')
        c = self.cue
        deco_opener(self, 2, 'TO TRADE', 'money_curb_brokers_wall_street_1920.jpg', 'THE CURB MARKET, WALL STREET, c. 1920')

        # never trading
        p = self.page(plain(81), c('This is the one') - 0.2)
        p.add(Typed(c('This is the one'), (120, 90), 'The one that surprised me most.', F('serif', 50)),
              Txt(c("It's how much"), (120, 170), 'Not that trading helps. How much.', F('hand', 40), fill=SEP))
        tg = Telegram(c('This is the one') + 0.6, (110, 300), ['My St. James for your Ventnor?', 'We both get a set.'],
                      w=780, angle=-2, wps=3)
        p.add(tg, Stamp(c('refuses every trade') + 0.2, (500, 560), ['REFUSED'], angle=-12, size=76))
        p.add(Txt(c('wins about five'), (1060, 300), 'Never trades', F('sc', 36), fill=SEP),
              Grid100(c('wins about five'), (1060, 360), 5, dur=0.4, size=34, gap=8, wash=sf.STAMP),
              Typed(c('Five.') - 0.1, (1520, 360), '5', F('deco', 200), fill=RED, cps=10),
              Txt(c('Five.') + 0.3, (1530, 600), 'percent', F('serif', 44)),
              Txt(c('A fifth of a fair'), (1060, 820), 'A fifth of a fair share.', F('serif', 48)),
              Txt(c('Nothing else in this'), (1060, 900), 'a typical player: 25%', F('hand', 36), fill=SEP))

        # take the fair swap; don't haggle
        p = self.page(plain(82), c('So when the deal') - 0.2)
        t1 = Telegram(c('So when the deal') + 0.3, (90, 110), ['My St. James for your Ventnor?', 'A set for a set.'],
                      w=780, angle=-2, wps=8)
        p.add(t1, Typed(c('take it.') - 0.2, (130, 560), 'REPLY:  Deal.', F('type', 44), fill=GREEN),
              Stamp(c('take it.') + 0.4, (700, 520), ['ACCEPTED'], angle=-8, size=58),
              Txt(c('at the printed prices'), (130, 650), 'any difference settled at the printed prices', F('hand', 36), fill=SEP))
        t2 = Telegram(c('ruin a trade is to haggle') - 0.2, (1000, 140), ['Your set is better than mine.', 'Throw in $200 and', 'it’s a deal.'],
                      w=830, angle=2, wps=6, how='right')
        p.add(t2, Txt(c('A player who does that') + 0.2, (1040, 640), 'The haggler:', F('sc', 36), fill=SEP),
              Typed(c('wins about twenty-one'), (1040, 700), '21%, instead of 25%', F('type', 46), fill=RED),
              Line(c('The other player walks') + 0.2, [(1040, 190), (1800, 470)], width=8, dur=0.3),
              Line(c('The other player walks') + 0.5, [(1040, 470), (1800, 190)], width=8, dur=0.3),
              Txt(c('neither of you gets a set'), (1040, 800), 'They walk away. Nobody gets a set.', F('hand', 38), fill=SEP),
              Stamp(c('neither of you gets a set') + 0.8, (1560, 930), ['NO DEAL'], angle=-7, size=60))

        # waiting and holding out
        p = self.page(ledger_paper(83, rules=[360, 560, 760], cols=()), c('What about waiting') - 0.2)
        p.add(Txt(c('What about waiting') + 0.3, (170, 90), 'OTHER WAYS TO TRADE', F('scb', 42), fill=GREEN),
              Txt(c('What about waiting') + 0.3, (840, 210), 'Games won (%)', F('sc', 26), fill=SEP))
        sc = 24
        ledger_row(p, c('What about waiting') + 0.5, 280, 'Typical: take every fair swap', 25.1, sc, fnt=F('type', 34), x_bar=840)
        ledger_row(p, c('What about waiting') + 1.0, 480, 'Wait until you have the\ncash to build', 25.1, sc, fnt=F('type', 34), x_bar=840,
                   bar_t=c('It makes no difference') - 0.3)
        ledger_row(p, c('Holding out for'), 680, 'Hold out for the better set', 25.6, sc, fnt=F('type', 34), x_bar=840,
                   bar_t=c('It barely helps') - 0.3)
        p.add(Txt(c('It makes no difference') + 0.4, (1560, 470), 'no difference', F('hand', 36), fill=RED),
              Txt(c('It barely helps') + 0.4, (1560, 670), 'barely', F('hand', 36), fill=RED))

        # the endgame deal
        p = self.page(plain(84), c('And remember the endgame') - 0.2)
        p.add(Txt(c('And remember the endgame') + 0.3, (120, 80), 'THE ENDGAME: THE BOARD HAS LOCKED UP', F('sc', 36), fill=SEP),
              Obj(c('And remember the endgame') + 0.5, padlock(150), (1640, 60), how='drop'))
        t3 = Telegram(c('Someone offers to buy') - 0.2, (110, 160), ['I’ll buy your Kentucky Ave.', 'for $550 cash.', 'It finishes my reds.'],
                      w=880, angle=-2, wps=7)
        p.add(t3, Txt(c('Every instinct') + 0.4, (1140, 220), 'NO!', F('fat', 150), fill=sf.SEPIA),
              Line(c('Say yes') - 0.3, [(1120, 330), (1360, 210)], width=10, dur=0.3),
              Typed(c('Say yes') + 0.1, (150, 620), 'REPLY:  Yes. Deal.', F('type', 48), fill=GREEN),
              Stamp(c('Say yes') + 0.9, (920, 520), ['SOLD'], angle=-10, size=70))
        p.add(Txt(c('Players who refuse') + 0.1, (1140, 450), 'Wins / first to go broke:', F('sc', 34), fill=SEP),
              Typed(c('Players who refuse') + 0.6, (1140, 510), 'refuse .. 24.2% / 28.7%', F('type', 40), fill=RED),
              Typed(c('Players who refuse') + 1.4, (1140, 570), 'sell .... 25.1% / 25.1%', F('type', 40)))
        notes = sf.banknote(100, (40, 92, 70), 33).rotate(-5, expand=True, resample=Image.BICUBIC)
        p.add(Obj(c('The cash lets you build'), notes, (150, 760), how='left', dur=0.5),
              Obj(c('on what you') + 0.2, house_small(), (790, 700), how='drop'),
              Stamp(c('the deal is the move') - 0.1, (1500, 880), ['TAKE THE', 'DEAL'], angle=-8, size=62))


def house_small():
    from kit import house_elevation
    return house_elevation(230, seed=4)


# ====================================================================== 06
class Railroads(PagedScene):
    def __init__(self):
        super().__init__('06_railroads')
        c = self.cue
        deco_opener(self, 3, 'THE RAILROADS', 'rail_penn_station_new_york.jpg', 'PENNSYLVANIA STATION, NEW YORK', focus=(0.55, 0.4))
        names = ['READING\nR.R.', 'PENNA.\nR.R.', 'B. & O.\nR.R.', 'SHORT\nLINE']
        tickets = [sf.railroad_ticket(nm, 11 + k) for k, nm in enumerate(names)]

        p = self.page(plain(91), c('The railroad player') - 0.2)
        for k, tk in enumerate(tickets):
            p.add(Obj(c('The railroad player') + 0.2 + 0.25 * k, tk, (330 + k * 330, 170 + [0, 30, 10, 40][k]), how='drop',
                      angle=[-5, 3, -2, 6][k]))
        p.add(Typed(c('collect two hundred'), (W / 2, 760), 'ALL FOUR:  $200 EVERY VISIT', F('type', 56), anchor='mm'),
              Txt(c('Does it work?'), (W / 2, 900), 'Does it work?', F('hand', 64), fill=RED, anchor='mm'))

        rr_rows = [('Buys them when it lands\non them', 25.1, '4%'), ('Chases them: mortgages,\nbids double', 25.1, '4%'),
                   ('Trades for the last one', 27.0, '28%')]

        def rr_ledger(p, ts, reveal_third, t_pct):
            p.add(Txt(ts[0] - 1.5, (150, 80), 'THREE RAILROAD PLAYERS', F('scb', 42), fill=GREEN),
                  Txt(ts[0] - 1.5, (760, 210), 'Games won (%)', F('sc', 26), fill=SEP),
                  Txt(ts[0] - 1.5, (1520, 210), 'Owns all four', F('sc', 26), fill=SEP))
            for k, (lab, v, pct) in enumerate(rr_rows):
                if k == 2 and reveal_third is None:
                    continue
                t0 = ts[k] if k < 2 else reveal_third
                y = 280 + k * 200
                ledger_row(p, t0, y, lab, v, 24, fnt=F('type', 34), x_bar=760, wash=sf.GROUP['green'] if k == 2 else None)
                p.add(Typed(t_pct[k], (1560, y), pct, F('type', 44), fill=RED if k < 2 else GREEN))

        p = self.page(ledger_paper(92, rules=[440, 640, 840], cols=(1480,)), c('I tried three') - 0.2)
        rr_ledger(p, [c('A player who just buys'), c('A player who chases')], None, [c('in only about four percent'), c('in only about four percent') + 0.3, 0])
        p.add(Txt(c('no better than typical') + 0.2, (760, 360), 'typical: 25.1', F('hand', 30), fill=SEP),
              Txt(c('still no better') + 0.2, (760, 560), 'still 25.1', F('hand', 30), fill=SEP),
              Txt(c('Either way, buying gets'), (150, 760), 'Buying alone almost never gets you all four.', F('serif', 44)),
              Line(c('in only about four percent') + 0.6, ring(1590, 400, 90, 110), width=4, dur=0.5))

        # the trade for the fourth
        p = self.page(plain(93), c('The version that works') - 0.2)
        for k, tk in enumerate(tickets[:3]):
            p.add(Obj(-1, tk, (90 + k * 200, 120 + [0, 30, 10][k]), how='drop', angle=[-5, 3, -2][k]))
        p.add(Obj(c('You get your four'), tickets[3], (690, 150), how='right', angle=6, dur=0.6))
        tg = Telegram(c('When someone is holding') + 0.2, (1010, 140), ['I’ll give you Kentucky Ave.', 'You finish your reds.',
                                                                      'I get Short Line and', 'all four railroads. Deal?'], w=830,
                      angle=-3, wps=7, how='right')
        p.add(tg, Txt(c('They get a set'), (1040, 680), 'They get a set.', F('serif', 52)),
              Txt(c('You get your four'), (1040, 760), 'You get your four railroads.', F('serif', 52)),
              Txt(c('The version that works') + 0.3, (110, 700), 'The version that works\nis the one that trades.', F('deco', 60), spacing=14))

        p = self.page(ledger_paper(92, rules=[440, 640, 840], cols=(1480,)), c('Do that, and') - 0.2)
        rr_ledger(p, [-10, -10], c("You'll win about twenty"), [-10, -10, c('own all four in about')])
        p.add(Txt(c('where you do hold all four') + 0.2, (150, 900), 'Holding all four: wins 53% of those games.', F('serif', 44)),
              Stamp(c('So, go for the railroads') + 0.4, (1560, 900), ['ONLY IF', 'YOU TRADE'], angle=-8, size=60),
              Txt(c("Otherwise, don't bother"), (150, 990), 'Otherwise, don’t bother.', F('hand', 40), fill=RED))


SCENES = {'03_how_we_tested_it': HowWeTested, '04_buy_or_pass': BuyOrPass, '05_trading': Trading, '06_railroads': Railroads}

if __name__ == '__main__':
    main(SCENES)
