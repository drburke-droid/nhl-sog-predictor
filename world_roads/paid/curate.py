"""The documented salaries used before each league's own series begins:  python curate.py  -> curated.csv

research/*.csv hold every figure the research turned up (each with its source and a quote), conflicts included.
SPEC below picks one figure per player and season, and spreads a contract's average over the seasons it covered. The
rules: a figure from contract documents or a contemporary report (AP, UPI, the New York Times, the Los Angeles Times,
as quoted) or a biography beats the 2010 APBR forum compilation of star contracts, which lists terms without sources;
the compilation is used where nothing else is found; a figure from one secondary source with no contract terms
(Ernie DiGregorio's reported $500,000, 1973-74) is left out; a season no source covers is left empty rather than
guessed. A signing or playoff bonus is spread over its contract, as the leagues' own series do (Sergei Fedorov's $28
million of 1997-98 counts as six years of up to $38 million), and NFL cash-paid figures, which put a whole bonus in
one season, are not used. Each row keeps its source's URL and quote, and a note on what it overrode.
"""
import os, re, csv, html
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "research")

# (player, research year_end, research salary, first season, last season, note[, the figure the same source gives
# for this season]); a season is the calendar year it ends (1967-68 -> 1968; NFL seasons are their own year)
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
NHL = [
    ("Howie Morenz", 1924, 3500, 1924, 1926, "3 years at $3,500 (other accounts: $2,500 or C$1,800)"),
    ("Lionel Conacher", 1926, 7500, 1926, 1926, ""), ("Dunc Munro", 1926, 7500, 1926, 1926, ""),
    ("Howie Morenz", 1929, 10000, 1929, 1929, "C$10,000"), ("Howie Morenz", 1934, 9200, 1934, 1934, "C$11,500 (HZP's conversion)"),
    ("Eddie Shore", 1939, 7000, 1939, 1939, "the NHL maximum"), ("Maurice Richard", 1947, 9000, 1947, 1947, ""),
    ("Jean Beliveau", 1954, 21000, 1954, 1958, "5 years, $105,000"),
    ("Gordie Howe", 1958, 20000, 1958, 1958, "base; about $8,000 more in bonuses"),
    ("Gordie Howe", 1967, 40000, 1967, 1967, "plus up to $10,000 in bonuses"), ("Gordie Howe", 1970, 45000, 1970, 1970, ""),
    ("Bobby Baun", 1970, 100000, 1970, 1970, "'over $100,000' (another account: $67,000)"),
    ("Gordie Howe", 1971, 100000, 1971, 1971, "2 years at $100,000; he retired in 1971"),
    ("Stan Mikita", 1973, 100000, 1971, 1971, "contract papers", 90000),
    ("Stan Mikita", 1973, 100000, 1972, 1972, "contract papers", 95000),
    ("Stan Mikita", 1973, 100000, 1973, 1973, "contract papers"),
    ("Bobby Orr", 1972, 200000, 1972, 1976, "5 years at $200,000"),
    ("Phil Esposito", 1972, 100000, 1972, 1975, "'over $100,000' a season"),
    ("Phil Esposito", 1976, 200000, 1976, 1979, "4 years at $200,000"),
    ("Derek Sanderson", 1973, 530000, 1973, 1973, "WHA; 5 years, $2.65 million; bought out after 8 games"),
    ("Gordie Howe", 1974, 250000, 1974, 1977, "WHA; 4 years, $1 million"),
    ("Marcel Dionne", 1976, 300000, 1976, 1980, "5 years, $1.5 million"),
    ("Bobby Orr", 1977, 600000, 1977, 1979, "5 years, $3 million, paid over 30 years; he played 26 games"),
    ("Wayne Gretzky", 1979, 250000, 1979, 1979, "WHA; 7-year personal-services contract, US$1.75 million"),
    ("Marcel Dionne", 1981, 600000, 1981, 1986, "6 years, $3.6 million"),
    ("Wayne Gretzky", 1982, 417000, 1982, 1982, "C$500,000 (HZP's conversion)"),
    ("Wayne Gretzky", 1986, 878400, 1986, 1986, "C$1.2 million (HZP's conversion)"),
    ("Wayne Gretzky", 1989, 2000000, 1989, 1989, "base salary, 8-year deal"),
    ("Mario Lemieux", 1990, 2000000, 1990, 1990, "NHLPA list, base"), ("Wayne Gretzky", 1990, 1720000, 1990, 1990, "NHLPA list, base"),
    ("Wayne Gretzky", 1991, 3000000, 1991, 1991, ""), ("Wayne Gretzky", 1992, 3000000, 1992, 1992, ""),
    ("Wayne Gretzky", 1989, 2000000, 1993, 1993, "the 8-year deal's $3 million years", 3000000),
    ("Mario Lemieux", 1993, 6000000, 1993, 1993, "7 years, $42 million, by his account"),
    ("Mario Lemieux", 1994, 7000000, 1994, 1994, "'about $7 million a year'"),
    ("Mario Lemieux", 1993, 6000000, 1995, 1996, "7 years, $42 million"), ("Mario Lemieux", 1997, 11000000, 1997, 1997, ""),
    ("Eric Lindros", 1993, 3500000, 1993, 1998, "6 years, $21 million"),
    ("Wayne Gretzky", 1994, 8500000, 1994, 1996, "3 years, $25.5 million (UPI; HZP lists a $3 million NHL salary for 1993-94)"),
    ("Mark Messier", 1995, 6000000, 1995, 1995, "'close to $6 million' (full-season rate)"),
    ("Mark Messier", 1997, 6000000, 1997, 1997, ""),
    ("Pavel Bure", 1995, 4900000, 1995, 1999, "5 years, $24.5 million"),
    ("Keith Tkachuk", 1996, 6000000, 1996, 1996, "first season of a front-loaded offer sheet"),
    ("Wayne Gretzky", 1997, 4000000, 1997, 1998, "2 years, $8 million"),
    ("Jaromir Jagr", 1998, 5100000, 1998, 1998, ""), ("Jaromir Jagr", 1998, 5100000, 1999, 1999, "", 4750000),
    ("Jaromir Jagr", 2000, 9500000, 2000, 2000, "scheduled, 6 years, $48 million"),
    ("Paul Kariya", 1999, 8500000, 1999, 1999, ""),
    ("Paul Kariya", 2000, 10666667, 2000, 2002, "3 years, $32 million (AP: $10 million in 2000-01)"),
    ("Paul Kariya", 2003, 10000000, 2003, 2003, ""), ("Dominik Hasek", 1999, 8666667, 1999, 2001, "3 years, $26 million"),
    ("Pavel Bure", 2000, 9500000, 2000, 2003, "5 years, $47.5 million"),
    ("Pavel Bure", 2004, 10000000, 2004, 2004, "injured all season"), ("Peter Forsberg", 2001, 10000000, 2001, 2001, ""),
    ("Keith Tkachuk", 2001, 8300000, 2001, 2001, ""),
    ("Keith Tkachuk", 2003, 10000000, 2003, 2004, "4 years, $40 million (the 2004-05 lockout and a rollback followed)"),
    ("Jaromir Jagr", 2002, 11000000, 2002, 2004, "7 years, $77 million"),
    ("Alexei Yashin", 2002, 8750000, 2002, 2004, "10 years, $87.5 million"),
    ("Chris Pronger", 2002, 9500000, 2002, 2004, "3 years, $29.5 million"), ("Teemu Selanne", 2002, 9500000, 2002, 2002, ""),
    ("Sergei Fedorov", 2004, 10000000, 2004, 2004, "5 years, $40 million"),
]
NFL = [
    ("Red Grange", 1925, 100000, 1925, 1925, "salary and a share of the gate, barnstorming tours included (his autobiography: about $125,000)"),
    ("Benny Friedman", 1929, 10000, 1929, 1929, ""), ("Bronko Nagurski", 1930, 5000, 1930, 1930, ""),
    ("Bronko Nagurski", 1931, 4500, 1931, 1931, ""), ("Sammy Baugh", 1937, 8000, 1937, 1937, ""),
    ("Byron 'Whizzer' White", 1938, 15800, 1938, 1938, ""), ("Otto Graham", 1946, 7500, 1946, 1946, "AAFC"),
    ("Frankie Albert", 1946, 10000, 1946, 1946, "AAFC"), ("Bob Waterfield", 1946, 20000, 1946, 1948, "3 years at $20,000"),
    ("Charley Trippi", 1947, 25000, 1947, 1950, "4 years, $100,000 (plus a first-year bonus of $25,000)"),
    ("Bill Dudley", 1947, 20000, 1947, 1949, "3 years at $20,000"), ("Johnny Lujack", 1948, 17000, 1948, 1948, ""),
    ("Steve Van Buren", 1949, 15000, 1949, 1949, ""), ("Johnny Lujack", 1951, 20000, 1951, 1951, ""),
    ("Otto Graham", 1955, 25000, 1955, 1955, ""),
    ("Billy Cannon", 1960, 33000, 1960, 1962, "AFL; 3 years at $33,000 plus a $10,000 signing bonus"),
    ("Jim Brown", 1965, 65000, 1965, 1965, "signed contract rider"),
    ("Joe Namath", 1965, 142333, 1965, 1967, "AFL; 3 years, $427,000, bonus included"),
    ("Donny Anderson", 1966, 237000, 1966, 1968, "3 years, $711,000 by his account (another report: $600,000)"),
    ("Roger Staubach", 1972, 75000, 1972, 1974, "3 years at about $75,000"),
    ("Johnny Unitas", 1973, 250000, 1973, 1974, "2 years, close to $500,000"),
    ("Larry Csonka", 1975, 466667, 1975, 1975, "WFL; 3 years, $1.4 million; the league folded in October"),
    ("Paul Warfield", 1975, 300000, 1975, 1975, "WFL; 3 years, $900,000; the league folded in October"),
    ("O.J. Simpson", 1976, 833333, 1976, 1978, "3 years, $2.5 million (another account: $733,000 a year)"),
    ("Ricky Bell", 1977, 240000, 1977, 1981, "5 years, $1.2 million"), ("Earl Campbell", 1978, 233333, 1978, 1983, "6 years, $1.4 million"),
    ("Herschel Walker", 1983, 1400000, 1983, 1983, "USFL; 3 years, $4.2 million (another account: $1.5 million a year)"),
    ("Herschel Walker", 1984, 1500000, 1984, 1985, "USFL"),
    ("Steve Young", 1984, 2400000, 1984, 1985, "USFL; cash paid, a $2.5 million signing bonus included (the '$40 million' deal was mostly a deferred annuity)"),
    ("Warren Moon", 1984, 1100000, 1984, 1988, "5 years, $5.5 million (an AP caption: $6 million)"),
    ("Doug Flutie", 1985, 1400000, 1985, 1985, "USFL; 5 years, $7 million"), ("Jim Kelly", 1986, 1600000, 1986, 1989, "5 years, $8 million"),
    ("Herschel Walker", 1986, 1000000, 1986, 1989, "5 years, $5 million"),
    ("Brian Bosworth", 1987, 1100000, 1987, 1989, "10 years, $11 million; his career ended in 1989"),
    ("Randall Cunningham", 1989, 2600000, 1989, 1993, "AP: an average $2.6 million (NBC Sports: $3.6 million)"),
    ("Warren Moon", 1989, 2000000, 1989, 1993, "5 years, $10 million"),
    ("Joe Montana", 1990, 4000000, 1990, 1990, "base salary"), ("Herschel Walker", 1990, 2250000, 1990, 1990, "base salary"),
    ("Jim Kelly", 1990, 1400000, 1990, 1990, "base salary"),
]
ADDED += [
    (dict(league="WHA", player="Bobby Hull", team="Winnipeg Jets", year_end=1973, salary_usd=275000, basis="contract average",
          source_url="https://en.wikipedia.org/wiki/Bobby_Hull",
          source_quote="a contract worth $1.75 million over 10 years plus a $1 million signing bonus"),
     1973, 1980, "$2.75 million over 10 years, bonus included; he retired in 1980 (another account: $1 million bonus, $1 million over 4 years)"),
    (dict(league="NHL", player="Joe Sakic", team="Colorado Avalanche", year_end=1998, salary_usd=7000000, basis="contract average",
          source_url="https://en.wikipedia.org/wiki/Joe_Sakic",
          source_quote="a three-year, $21 million offer sheet with the New York Rangers"),
     1998, 2000, "3 years, $21 million, matched by Colorado ($17 million of it, a $15 million bonus included, paid in 1997-98)"),
    (dict(league="NHL", player="Sergei Fedorov", team="Detroit Red Wings", year_end=1998, salary_usd=6333333, basis="contract average",
          source_url="https://en.wikipedia.org/wiki/Sergei_Fedorov", source_quote="worth up to $38 million, including bonuses"),
     1998, 2003, "up to $38 million over six years (Sports Illustrated), matched by Detroit; $28 million of it paid in 1997-98"),
    (dict(league="NFL", player="O.J. Simpson", team="Buffalo Bills", year_end=1972, salary_usd=53750,
          basis="contract average (4 years, $215,000 salary; excl. $100,000 loan, incentives, annuity)",
          source_url="https://en.wikipedia.org/wiki/1969_Buffalo_Bills_season", source_quote="Simpson signs a four-year contract worth $215,000."),
     1972, 1972, "career research: the fourth season of the 1969 contract (careers_nfl.csv has 1969-71 from the same source)"),
]
SPECS = {"nba_pre1985.csv": NBA, "nhl_pre2006.csv": NHL, "nfl_pre2011.csv": NFL}

# The career research (research/careers_*.csv): whole careers season by season, for the career race, one figure per
# player and season chosen by the same rules. A season the selection above already has keeps that figure unless the
# research corrects it (CORRECT); a row in DROP is left out, for the reason given.
CAREERS = ["careers_nba.csv", "careers_nfl.csv", "careers_nhl.csv"]
CORRECT = {
    ("Eric Lindros", 1998): "the revised two-year, $16 million deal of Dec 1997 paid $7.5 million in 1997-98 (NYT), not the $3.5 million average",
    ("Phil Esposito", 1977): "THN (Feb 1976): $375,000 a year with three years left, over the $200,000 used before",
    ("Phil Esposito", 1978): "Globe and Mail: $325,000 for 1977-78, over the $200,000 used before",
    ("Phil Esposito", 1979): "THN (Feb 1976): $375,000 a year with three years left, over the $200,000 used before",
    ("Wayne Gretzky", 1997): "AP's salary list (Sept 1996): $5,047,500, over the $4 million average used before",
    ("Wayne Gretzky", 1998): "THN: $6.5 million after the Oct 1997 renegotiation, over the $4 million average used before",
    ("Joe Namath", 1965): "Phil Berger's 1969 biography itemizes the $427,000: $25,000 a season for four seasons (1965-68) and a "
                          "$200,000 bonus were his pay; $127,000 went to a lawyer, a car and scouting jobs for relatives",
    ("Joe Namath", 1966): "Phil Berger's 1969 biography itemizes the $427,000 (see 1965)",
    ("Joe Namath", 1967): "Phil Berger's 1969 biography itemizes the $427,000 (see 1965)",
    ("Warren Moon", 1991): "the NFLPA's salary survey: $2,883,000, over the $2 million average of his 1989 deal",
    ("Warren Moon", 1992): "the NFLPA's salary survey: $3,563,000, over the $2 million average of his 1989 deal",
    ("Rick Barry", 1971): "Jim O'Brien, ABA All-Stars (1972): $165,000 a year with the Nets; the $75,000 used before was his Oaks rate",
    ("Walt Frazier", 1975): "Pro Sports (March 1975): $250,000, over the APBR compilation's $340,000 average",
    ("Bob McAdoo", 1981): "UPI (March 1981): $542,000, over the $500,000 used before",
}
DROP = {
    ("Wayne Gretzky", 1987): "an outdated 1979 schedule, superseded by the 1982 contract; the season is interpolated instead",
    ("Jerry West", 1966): "a magazine writer's own estimate, not a reported figure; the season is interpolated instead",
}
DROP.update({("Maurice Richard", y): "Canadian dollars before 1971, when the Federal Reserve's H.10 rates begin"
             for y in (1943, 1944, 1951, 1954, 1955, 1956, 1957, 1958, 1959, 1960)})
UNPAID = {   # seasons of the selection above the research found were not paid
    ("Bobby Orr", 1978): "sat out 1977-78 and refused his salary; he 'never cashed a Chicago paycheque' (Montreal Gazette, Nov 1978)",
}


def fx():
    """Canadian dollars per US dollar, each calendar year's average of the Federal Reserve's H.10 noon rates (1971 on)"""
    by = defaultdict(list)
    for f in ("dat89", "dat96", "dat00"):
        h = open(os.path.join(HERE, "data", "fx", f"{f}_ca.htm"), encoding="utf8", errors="replace").read()
        for r in re.findall(r"<tr[^>]*>(.*?)</tr>", h, re.S):
            t = html.unescape(re.sub(r"<[^>]+>", "", r)).replace("\xa0", " ")
            m = re.search(r"(\d{1,2})-([A-Za-z]{3})-(\d{2})\s+([\d.]+)", t)
            if m:
                yy = int(m.group(3))
                by[(1900 if yy >= 50 else 2000) + yy].append(float(m.group(4)))
    return {y: sum(v) / len(v) for y, v in by.items()}


def main():
    out = []
    for fname, spec in SPECS.items():
        rows = list(csv.DictReader(open(os.path.join(R, fname), encoding="utf8")))
        for player, y, sal, a, b, note, *own in spec:
            hit = [r for r in rows if r["player"] == player and int(r["year_end"]) == y and abs(float(r["salary_usd"]) - sal) < 1]
            assert len(hit) >= 1, (fname, player, y, sal)
            r = hit[0]
            for yr in range(a, b + 1):         # own: a figure the same source gives for another season
                out.append(dict(league=r["league"], player=player.split(" (")[-1].rstrip(")") if "(" in player else player,
                                team=r["team"].split(" / ")[-1], year_end=yr, salary_usd=own[0] if own else sal,
                                basis=r["basis"] if a == b and not own else "contract average" if not own else "salary",
                                source_url=r["source_url"], source_quote=r["source_quote"], notes=note))
    for r, a, b, note in ADDED:
        for yr in range(a, b + 1):
            out.append(dict(r, year_end=yr, notes=note))
    keys = {}
    for r in out:
        k = (r["player"], r["year_end"])
        assert k not in keys, f"two figures for {k}"
        keys[k] = r
    for k, why in UNPAID.items():
        out.remove(keys.pop(k))
    rates = fx()
    added = 0
    for fname in CAREERS:
        p = os.path.join(R, fname)
        for r in csv.DictReader(open(p, encoding="utf8")) if os.path.exists(p) else []:
            k = (r["player"], int(r["year_end"]))
            if k in DROP or (k in keys and k not in CORRECT) or not r["salary_usd"].strip():
                continue
            sal, basis = float(r["salary_usd"]), r["basis"]
            if basis.startswith("C$"):                # a season's rate: the average of the two calendar years it spans
                rate = (rates[k[1] - 1] + rates[k[1]]) / 2
                sal, basis = round(sal / rate), f"{basis}; C${sal:,.0f} at C${rate:.4f} per US$ (Federal Reserve H.10)"
            row = dict(league=r["league"], player=r["player"], team=r["team"].split(" / ")[-1], year_end=k[1],
                       salary_usd=sal, basis=basis, source_url=r["source_url"],
                       source_quote=r["source_quote"], notes=("career research; " + CORRECT[k] + "; " if k in CORRECT else "career research; ") + r["notes"])
            if k in keys:
                out.remove(keys[k])
            out.append(row)
            keys[k] = row
            added += 1
    cols = ["league", "player", "team", "year_end", "salary_usd", "basis", "source_url", "source_quote", "notes"]
    with open(os.path.join(HERE, "curated.csv"), "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(sorted(out, key=lambda r: (r["year_end"], -float(r["salary_usd"]))))
    print(f"curated.csv: {len(out)} rows ({added} from the career research)")


if __name__ == "__main__":
    main()
