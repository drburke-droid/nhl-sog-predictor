import json, re, sys
D = sys.argv[1]
S = json.load(open(D + '/slides.json'))
TITLES = {'01_cold_open': 'Cold open', '02_the_promise': 'The promise', '03_how_we_tested_it': 'How I tested it',
          '04_buy_or_pass': 'Decision 1: Buy or pass', '05_trading': 'Decision 2: Trading', '06_railroads': 'Decision 3: The railroads',
          '07a_spend_or_save': 'Decision 4a: Spend or save', '07b_how_far_to_build': 'Decision 4b: How far to build',
          '07c_which_set_first': 'Decision 4c: Two sets, which first?', '07d_one_set_or_spread': 'Decision 4d: One set, or a house on everything?',
          '07e_owe_money': 'Decision 4e: Owe money?', '08_jail': 'Decision 5: Jail', '09_the_computers_player': 'The computer’s own player',
          '10_the_payoff': 'The payoff', '11_the_playbook': 'The playbook'}
EXTRA = {
 ('01_cold_open', 2): ['The 1904 patent drawing on the desk; the camera pushes in on the header, then pulls back to the whole board'],
 ('01_cold_open', 3): ['Photo: THE BOARDWALK, ATLANTIC CITY, N.J.'],
 ('02_the_promise', 1): ['Title card: A STUDY OF THE LANDLORD’S GAME', 'THE DECISIONS YOU CONTROL', 'You can’t choose where you land. You can choose what you do next.', '13,400,000 GAMES · ATLANTIC CITY, N.J.', 'AFTER L. J. MAGIE, PATENT No. 748,626 (1904)'],
 ('02_the_promise', 2): ['The orange squares, enlarged', 'The colour everyone tells you to buy.', 'Orange: landed on more than any other set.', 'Two dice tumble across'],
 ('03_how_we_tested_it', 2): ['Photo: GAME NIGHT, 1922 (four women playing mah-jong)'],
 ('04_buy_or_pass', 1): ['Photo: A NEW STUTZ ROADSTER, 1920', 'Plaque: DECISION ONE / TO BUY, OR TO PASS'],
 ('04_buy_or_pass', 2): ['Deed of Sale: KENTUCKY AVE., $220'],
 ('05_trading', 1): ['Photo: THE CURB MARKET, WALL STREET, c. 1920', 'Plaque: DECISION TWO / TO TRADE'],
 ('06_railroads', 1): ['Photo: PENNSYLVANIA STATION, NEW YORK', 'Plaque: DECISION THREE / THE RAILROADS'],
 ('06_railroads', 2): ['Four railway tickets: READING R.R., PENNA. R.R., B. & O. R.R., SHORT LINE'],
 ('07a_spend_or_save', 1): ['Photo: HOTEL TRAYMORE, ATLANTIC CITY, 1920', 'Plaque: DECISION FOUR / TO BUILD'],
 ('08_jail', 1): ['Photo: PROHIBITION RAID, NEW YORK, 1921', 'Plaque: DECISION FIVE / JAIL'],
 ('08_jail', 2): ['A pawn; ink prison bars drop over it', 'PAY $50 and get out', 'STAY and roll for doubles'],
 ('10_the_payoff', 2): ['Three $100 banknotes slide in'],
 ('10_the_payoff', 4): ['Rows of trophies; snapshot: ATLANTIC CITY, 1922 (the first Miss America)'],
 ('10_the_payoff', 5): ['Photo: mah-jong at the bathing beach, 1924'],
 ('11_the_playbook', 1): ['Certificate: THE PLAYBOOK / CERTIFICATE OF BETTER DECISIONS', 'Wax seal: +45% MORE WINS'],
 ('11_the_playbook', 2): ['Fade to the 1904 patent; the camera pulls back from Magie’s signature to the whole board'],
}
OVERRIDE = {
 ('07c_which_set_first', 1): 'Orange and green deeds · The question people really argue about. · Which one / first?',
 ('07c_which_set_first', 2): 'A set-up card and a grid ticking off every pair of sets · 28 pairs',
 ('07c_which_set_first', 3): 'THE RULE · Build the set with the cheaper houses first. · 1. light blue $50, orange $100 a house: beat everything · 2. pink, $100 a house · 3. red = yellow, $150 a house: a coin flip · 4. dark blue = green, $200 a house · brown: the odd one out ($50 a house) · Rents so low it only goes first ahead of dark blue or green, and only just.',
 ('07d_one_set_or_spread', 2): 'Build one set up first. · Two cards: a cheap set going straight to hotels; an expensive set with three houses on both sets, then hotels',
 ('07e_owe_money', 1): 'One last building question. · Rent bill for Illinois Ave. with a hotel, $1,100 · CAN’T PAY · Sell houses? · Or mortgage? · MORTGAGED · keep the houses standing · sell houses first .. 19.1% · first to go broke .. 28.5% · MORTGAGE FIRST',
 ('08_jail', 3): 'The camera moves to the jail corner of the 1904 patent board · L. J. MAGIE, PATENT No. 748,626 · THE BOARD OF 1904 · “NO TRESPASSING. · GO TO JAIL.”',
 ('09_the_computers_player', 1): 'One more test: let the computer invent a player. · Fig. 3. TRIAL AND ERROR · A family tree: rows of 24 boxes, winners ringed, generations 1 to 25 · keep the winners · mix their habits · tweak · repeat, 25 times',
 ('10_the_payoff', 1): 'So put it all together. · A red pawn · THE PLAYBOOK PLAYER · every decision made well · no favourite colour · Two dice',
 ('10_the_payoff', 2): 'Three $100 banknotes slide in · How much is that worth? · extra cash a typical player needs to keep up: · $300 · of free cash at the start of every game · stamp WORTH $300 · Without auctions: about $200, · the same as passing Go.',
 ('10_the_payoff', 3): 'GAME NIGHT LEDGER · A hundred game nights, the same four players. You follow the playbook. · Tally marks: YOU 33 · MOM 22 · DAD 22 · JORDAN 23 · 45% more wins than each of them. · stamp 45% MORE WINS',
 ('10_the_payoff', 4): 'And it compounds. · A year of weekly game nights with the same family. 100 years of them: · Trophies: YOU 65 · MOM 9 · DAD 9 · JORDAN 9 · champion two years in three · each of the others: about one year in eleven · Snapshot: ATLANTIC CITY, 1922 (the first Miss America)',
 ('11_the_playbook', 1): 'Certificate: THE PLAYBOOK / CERTIFICATE OF BETTER DECISIONS · I. Land on it? Buy it, unless it would leave you nearly broke. · II. At auction, bid up to one and a half times the price. · III. Make every fair set-for-set trade. Don’t haggle. When it stalls, deal. · IV. Want the railroads? Trade for the last one, or don’t bother. · V. Spend your cash on houses, and go to hotels. Owe money? Mortgage first. · VI. Two sets? Cheaper houses first. Never one house on everything. · VII. Get out of jail early. Stay in once anyone has houses. · Footer: Same dice. 45% more wins than each of them. A year of weekly game nights: champion two years in three. · Wax seal: +45% MORE WINS',
}
def mmss(t):
    m, s = divmod(int(t), 60); return '%d:%02d' % (m, s)
def clean(xs):
    out = []
    norm = lambda s: re.sub(r'[\d,]+', '#', s)
    for s in xs:
        s = s.replace('\n', ' / ')
        s = re.sub(r'\s{2,}', ' ', s)
        if re.fullmatch(r'[\d,]+( games)?', s) and any(norm(o) == norm(s) and o != s for o in xs[xs.index(s.replace(' / ', '\n')) + 1:] if True):
            continue
        out.append(s)
    # drop counter values that a later value replaces
    final = []
    for i, s in enumerate(out):
        if re.fullmatch(r'[\d,]+( games)?', s) and any(re.fullmatch(r'[\d,]+( games)?', o) and o.endswith(s[-6:] if False else '') and o != s and norm(o) .startswith('#') and o.split()[-1:] == s.split()[-1:] for o in out[i + 1:]):
            continue
        final.append(s)
    return final
scenes, txt = [], ['THE MONOPOLY DECISIONS YOU CONTROL: NARRATION BY SLIDE', 'Times are from the 11:42 cut.', '']
for k, name in enumerate(TITLES):
    sl = [s for s in S if s['scene'] == name]
    head = '%d. %s (%s)' % (k + 1, TITLES[name], mmss(sl[0]['start']))
    md = ['## ' + head]
    txt += ['=' * 70, head.upper() if False else head, '=' * 70, '']
    for s in sl:
        on = EXTRA.get((name, s['n']), []) + clean(s['onscreen'])
        if (name, s['n']) in OVERRIDE:
            on = [OVERRIDE[(name, s['n'])]]
        tag = 'Slide %d.%d · %s–%s' % (k + 1, s['n'], mmss(s['start']), mmss(s['end']))
        md += ['### ' + tag, '**Narration:** ' + s['narration'], '**On screen:** ' + (' · '.join(on) if on else '(picture only)')]
        txt += ['[' + tag + ']', 'ON SCREEN: ' + (' · '.join(on) if on else '(picture only)'), 'NARRATION: ' + s['narration'], '']
        s['tag'] = tag
    scenes.append('\n\n'.join(md))
open(D + '/sections.json', 'w').write(json.dumps(scenes))
open('script/narration_by_slide.txt', 'w').write('\n'.join(txt))
json.dump(S, open(D + '/slides2.json', 'w'), indent=1)
for i, sc in enumerate(scenes):
    print('----- %d (%d chars)' % (i, len(sc)))
