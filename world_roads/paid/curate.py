"""The documented salaries used before each league's own series begins:  python curate.py  -> curated.csv

research/*.csv hold every figure the research turned up (each with its source and a quote), conflicts included.
SPEC below picks one figure per player and season, and spreads a contract's average over the seasons it covered. The
rules: a figure from contract documents or a contemporary report (AP, UPI, the New York Times, the Los Angeles Times,
as quoted) or a biography beats the 2010 APBR forum compilation of star contracts, which lists terms without sources;
the compilation is used where nothing else is found; a figure from one secondary source with no contract terms
(Ernie DiGregorio's reported $500,000, 1973-74) is left out; a season no source covers is left empty rather than
guessed. Each row keeps its source's URL and quote, and a note on what it overrode.
"""
import os, csv

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "research")

# (league file, player, research year_end, research salary, first season, last season, note)
# a season is the calendar year it ends (1967-68 -> 1968)
NBA = [
    ("George Mikan", 1947, 12000, 1947, 1947, ""), ("Tom King", 1947, 16500, 1947, 1947, "also front-office pay"),
    ("Joe Fulks", 1947, 8000, 1947, 1947, ""), ("Bob Davies", 1947, 12500, 1947, 1950, "4 years, $50,000"),
    ("George Mikan", 1948, 12500, 1948, 1948, "over a reported $25,000"), ("George Mikan", 1950, 20000, 1950, 1950, ""),
    ("George Mikan", 1954, 35000, 1954, 1954, ""), ("Bob Pettit", 1955, 11000, 1955, 1955, ""),
    ("Bob Cousy", 1957, 25000, 1957, 1957, ""), ("Bill Russell", 1957, 24000, 1957, 1957, "over APBR's $19,500"),
    ("Elgin Baylor", 1959, 20000, 1959, 1959, ""), ("Bob Cousy", 1960, 30000, 1960, 1960, ""),
    ("Wilt Chamberlain", 1960, 30000, 1960, 1960, "over APBR's $60,000"), ("Wilt Chamberlain", 1961, 65000, 1961, 1961, ""),
    ("Oscar Robertson", 1961, 33333, 1961, 1963, "3 years, $100,000"),
    ("Wilt Chamberlain", 1961, 83333, 1962, 1963, "APBR: 3 years from 1960-61, $250,000"),
    ("Elgin Baylor", 1962, 50000, 1962, 1966, "5 years, $250,000"), ("Bill Russell", 1964, 40000, 1964, 1964, ""),
    ("Wilt Chamberlain", 1964, 100000, 1964, 1965, "APBR: 3 years, $300,000"),
    ("Wilt Chamberlain", 1966, 100000, 1966, 1967, "$100,000 a year for three seasons; 1967-68 replaced"),
    ("Bill Russell", 1966, 100001, 1966, 1966, ""), ("Bill Russell", 1967, 125001, 1967, 1967, "player and coach"),
    ("Oscar Robertson", 1967, 100000, 1967, 1967, ""), ("Elgin Baylor", 1967, 70000, 1967, 1967, ""),
    ("Wilt Chamberlain", 1968, 250000, 1968, 1968, ""), ("Jerry West", 1968, 100000, 1968, 1968, ""),
    ("Bill Bradley", 1968, 125000, 1968, 1971, "4 years, $500,000 (reported)"),
    ("Oscar Robertson", 1968, 100000, 1968, 1970, "3 years, $300,000"),
    ("Wilt Chamberlain", 1969, 250000, 1969, 1971, "APBR: 5 years, $1.25 million, pre-tax (Cherry: $250,000 after taxes)"),
    ("Bill Russell", 1969, 200000, 1969, 1969, "player and coach"),
    ("Rick Barry", 1969, 75000, 1969, 1971, "APBR: 3 years, $225,000; plus a share of the club"),
    ("Lew Alcindor (Kareem Abdul-Jabbar)", 1970, 280000, 1970, 1972, "5 years, $1.4 million, renegotiated 1972"),
    ("Elgin Baylor", 1970, 100000, 1970, 1971, "2 years, $200,000"),
    ("Oscar Robertson", 1971, 175000, 1971, 1973, "3 years, $525,000"),
    ("Spencer Haywood", 1971, 250000, 1971, 1976, "6 years, $1.5 million"),
    ("Artis Gilmore", 1972, 150000, 1972, 1976, "Pluto, Loose Balls (over a reported 10 years, $2.5 million)"),
    ("Wilt Chamberlain", 1972, 400000, 1972, 1973, "APBR: 2 years, $800,000, which its compiler queried"),
    ("Kareem Abdul-Jabbar", 1973, 350004, 1973, 1973, "over APBR's 4 years, $2 million"),
    ("Jerry West", 1973, 300000, 1973, 1974, "2 years, $600,000"),
    ("John Havlicek", 1973, 250000, 1973, 1976, "4 years, $1 million"),
    ("Walt Frazier", 1973, 340000, 1973, 1975, "5 years, $1.7 million"),
    ("Kareem Abdul-Jabbar", 1974, 378000, 1974, 1974, ""), ("Oscar Robertson", 1974, 325000, 1974, 1974, ""),
    ("Julius Erving", 1974, 350000, 1974, 1976, "8 years at a reported $350,000 a year (over APBR's 5 years, $2 million)"),
    ("Kareem Abdul-Jabbar", 1975, 399996, 1975, 1975, ""),
    ("Bill Walton", 1975, 400000, 1975, 1979, "5 years, $2 million"),
    ("Marvin Barnes", 1975, 300000, 1975, 1976, "7 years, $2.1 million"),
    ("Moses Malone", 1975, 200000, 1975, 1976, "5 years, $1 million (over APBR's 4 years, $565,000)"),
    ("Kareem Abdul-Jabbar", 1976, 475000, 1976, 1980, "contract documents, 5 years"),
    ("George McGinnis", 1976, 533333, 1976, 1981, "6 years, $3.2 million"),
    ("David Thompson", 1976, 500000, 1976, 1978, "6 years, $3 million (AP), over APBR's 3 years, $1.35 million"),
    ("Julius Erving", 1977, 500000, 1977, 1979, "6 years, $3 million (over APBR's $3.5 million)"),
    ("Artis Gilmore", 1977, 366667, 1977, 1979, "3 years, $1.1 million"),
    ("Pete Maravich", 1978, 600000, 1978, 1980, "APBR: 5 years, $3 million; retired 1980"),
    ("Bob McAdoo", 1978, 500000, 1978, 1981, "4 years, $2 million"),
    ("David Thompson", 1979, 750000, 1979, 1983, "5 years, $3.75 million (APBR; the NYT headline agrees)"),
    ("Rick Barry", 1979, 500000, 1979, 1980, "2 years, $1 million"),
    ("Bill Walton", 1980, 1000000, 1980, 1984, "$1 million a year (7 years, $7 million; UPI: 5 years, $5 million)"),
    ("Moses Malone", 1980, 1050000, 1980, 1982, "APBR: 3 years, $3.15 million (a 2013 retrospective says $2 million a year)"),
    ("Julius Erving", 1980, 600000, 1980, 1980, ""), ("Larry Bird", 1980, 650000, 1980, 1983, "5 years, $3.25 million"),
    ("Magic Johnson", 1980, 600000, 1980, 1984, "over APBR's 5 years, $2.5 million"),
    ("Julius Erving", 1981, 875000, 1981, 1984, "APBR: 4 years, $3.5 million (UPI: nearly $1 million)"),
    ("Kareem Abdul-Jabbar", 1981, 1000000, 1981, 1982, "'makes $1 million a year' (August 1981)"),
    ("Moses Malone", 1983, 2200000, 1983, 1983, "6 years, $13.2 million"), ("Moses Malone", 1984, 2160000, 1984, 1984, ""),
    ("Larry Bird", 1984, 2140000, 1984, 1984, ""), ("Kareem Abdul-Jabbar", 1984, 1650000, 1984, 1984, ""),
    ("Ralph Sampson", 1984, 1000000, 1984, 1984, "over a reported $800,000 base"),
]
ADDED = [   # figures checked directly, not in the research files: (row as in research, first, last, note)
    (dict(league="NBA", player="Pete Maravich", team="Atlanta Hawks", year_end=1971, salary_usd=380000,
          basis="contract average", source_url="https://en.wikipedia.org/wiki/Pete_Maravich",
          source_quote="the $1.9 million contract that Maravich received from the team"),
     1971, 1975, "$1.9 million over 5 years (APBR: $1.5 million; a 1970 annual: $2 million)"),
]
SPECS = {"nba_pre1985.csv": NBA}


def main():
    out = []
    for fname, spec in SPECS.items():
        rows = list(csv.DictReader(open(os.path.join(R, fname), encoding="utf8")))
        for player, y, sal, a, b, note in spec:
            hit = [r for r in rows if r["player"] == player and int(r["year_end"]) == y and abs(float(r["salary_usd"]) - sal) < 1]
            assert len(hit) >= 1, (fname, player, y, sal)
            r = hit[0]
            for yr in range(a, b + 1):
                out.append(dict(league=r["league"], player=player.split(" (")[-1].rstrip(")") if "(" in player else player,
                                team=r["team"], year_end=yr, salary_usd=sal, basis=r["basis"] if a == b else "contract average",
                                source_url=r["source_url"], source_quote=r["source_quote"], notes=note))
    for r, a, b, note in ADDED:
        for yr in range(a, b + 1):
            out.append(dict(r, year_end=yr, notes=note))
    keys = {}
    for r in out:
        k = (r["player"], r["year_end"])
        assert k not in keys, f"two figures for {k}"
        keys[k] = r
    cols = ["league", "player", "team", "year_end", "salary_usd", "basis", "source_url", "source_quote", "notes"]
    with open(os.path.join(HERE, "curated.csv"), "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(sorted(out, key=lambda r: (r["year_end"], -float(r["salary_usd"]))))
    print(f"curated.csv: {len(out)} rows")


if __name__ == "__main__":
    main()
