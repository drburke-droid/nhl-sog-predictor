# Video A review: the biggest single-game outliers in team sports

Model: `topk.py` (curve through each season's top 5 other games, ±15 seasons; length-adjusted games; league adjustment; counts treated as counts; clear of the field by one step). 4,364 season-leading games in 18 categories; 15 clear their field.

Numbers are in the athlete's own season: the curve and step are what a card would show; *best other* is the best game by anyone else on the same graph, converted to that season's scoring rate.

## The list

| # | Game | Sport | Raw | Length | Curve | Step | Score | Clear by | Best other on the graph | In top 10 of 16 versions |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Wilt Chamberlain, points in a game, 1962-03-02 | NBA | 100 | regulation | 50.3 | 5.52 | **8.91** | 5.15 | Pete Maravich 68 (1977) = 71.6 | 16 |
| 2 | Kobe Bryant, points in a game, 2006-01-22 | NBA | 81 | regulation | 50.9 | 4.38 | **6.75** | 2.62 | David Robinson 71 (1994) = 69.5 | 16 |
| 3 | Norm Van Brocklin, passing yards in a game, 1951-09-28 | NFL | 554 | regulation | 312.0 | 41.24 | **5.86** | 1.99 | Y.A. Tittle 505 (1962) = 472.0 | 16 (one-sided window) |
| 4 | Jim Bottomley, runs batted in, one game, 1924-09-16 | MLB | 12 | 9.0 innings batted | 6.7 | 0.83 | **5.78** | 2.01 | Tony Lazzeri 11 (1936) = 10.3 | 14 |
| 5 | Cloyce Box, receiving yards in a game, 1950-12-03 | NFL | 302 | regulation | 170.1 | 25.06 | **5.24** | 1.04 | Billy Howton 257 (1956) = 276.0 | 14 (one-sided window) |
| 6 | Bob Cousy, assists in a game, 1959-02-27 | NBA | 28 | regulation | 14.5 | 2.60 | **5.02** | 1.09 | Guy Rodgers 28 (1963) = 25.2 | 8 (one-sided window) |
| 7 | Mark Whiten, runs batted in, one game, 1993-09-07 | MLB | 12 | 9.0 innings batted | 7.3 | 0.85 | **4.90** | 2.42 | Alex Rodriguez 10 (2005) = 9.9 | 16 |
| 8 | Darryl Sittler, points in a game, 1976 | NHL | 10 | regulation | 6.1 | 0.72 | **4.75** | 2.56 | Tom Bladon 8 (1978) = 8.2 | 12 |
| 9 | Bert Olmstead, points in a game, 1954 | NHL | 8 | regulation | 4.6 | 0.65 | **4.52** | 1.94 | Maurice Richard 8 (1945) = 6.7 | 5 |
| 10 | Derrick Thomas, sacks in a game, 1990-11-11 | NFL | 7 | regulation | 4.0 | 0.59 | **4.33** | 1.91 | Fred Dean 6 (1983) = 5.9 | 12 |
| 11 | Sam Gagner, points in a game, 2012 | NHL | 8 | regulation | 5.1 | 0.60 | **4.05** | 1.62 | Jaromir Jagr 7 (2003) = 7.0 | 6 |
| 12 | Joe Namath, passing yards in a game, 1972-09-24 | NFL | 496 | regulation | 356.0 | 34.64 | **4.03** | 1.28 | Y.A. Tittle 505 (1962) = 451.5 | 3 |
| 13 | Billy Taylor, assists in a game, 1947 | NHL | 7 | regulation | 3.9 | 0.67 | **3.89** | 2.08 | Syl Apps 5 (1937) = 5.6 | 2 |
| 14 | Rennie Stennett, hits in a game, 1975-09-16 | MLB | 7 | 9.0 innings batted | 5.1 | 0.39 | **3.67** | 2.34 | Bob Oliver 6 (1969) = 6.1 | 1 |
| 15 | Mike Schmidt, home runs in a game, 1976-04-17 | MLB | 4 | 10.0 innings batted | 2.4 | 0.44 | **1.60** | 1.35 | Bob Horner 4 (1986) = 3.0 | 0 |

## Where the well-known games land

| Game | Raw | Length | Score | Rank among all season-best games | Clear by | Best other on the graph | Verdict |
|---|---|---|---|---|---|---|---|
| Wilt's 100, the NBA record | 100 | regulation | 8.91 | 1 | 5.15 | Pete Maravich 68 (1977) = 71.6 | **#1** |
| Kobe's 81 | 81 | regulation | 6.75 | 2 | 2.62 | David Robinson 71 (1994) = 69.5 | **#2** |
| Doncic's 73 | 73 | regulation | 2.41 | 110 | 0.08 | Devin Booker 70 (2017) = 72.6 | out: another game on its graph comes within a step |
| Wilt's 55 rebounds, the record | 55 | n/a | 4.26 | 12 | 0.82 | Bill Russell 51 (1960) = 51.0 | out: another game on its graph comes within a step |
| Skiles's 30 assists, the record | 30 | regulation | 3.81 | 23 | 0.55 | Kevin Porter 29 (1978) = 28.7 | out: another game on its graph comes within a step |
| Cousy's 28 assists | 28 | regulation | 5.02 | 6 | 1.09 | Guy Rodgers 28 (1963) = 25.2 | **#6** |
| Klay Thompson's 14 threes, the record | 14 | regulation | 2.36 | 116 | -1.52 | Donyell Marshall 12 (2005) = 16.0 | out: another game on its graph is as good or better |
| Sittler's 10 points, the record | 10 | regulation | 4.75 | 8 | 2.56 | Tom Bladon 8 (1978) = 8.2 | **#8** |
| Joe Malone's 7 goals, the record | 7 | regulation | 1.59 | 276 | -0.45 | Red Green 5 (1925) = 7.5 | out: another game on its graph is as good or better |
| Red Berenson's 6 goals (1968-69) | 6 | regulation | 3.43 | 33 | 0.34 | Bernie Geoffrion 5 (1955) = 5.8 | out: another game on its graph comes within a step |
| Billy Taylor's 7 assists, the record | 7 | regulation | 3.89 | 22 | 2.08 | Syl Apps 5 (1937) = 5.6 | **#13** |
| Bottomley's 12 RBI | 12 | 9.0 innings batted | 5.78 | 4 | 2.01 | Tony Lazzeri 11 (1936) = 10.3 | **#4** |
| Whiten's 12 RBI (and 4 HR) | 12 | 9.0 innings batted | 4.90 | 7 | 2.42 | Alex Rodriguez 10 (2005) = 9.9 | **#7** |
| Stennett's 7-for-7 | 7 | 9.0 innings batted | 3.67 | 24 | 2.34 | Bob Oliver 6 (1969) = 6.1 | **#14** |
| Schmidt's 4 HR (10 innings) | 4 | 10.0 innings batted | 1.60 | 275 | 1.35 | Bob Horner 4 (1986) = 3.0 | **#15** |
| Hamilton's 4 HR | 4 | 9.0 innings batted | 1.40 | 326 | 0.17 | Mike Cameron 4 (2002) = 3.9 | out: another game on its graph comes within a step |
| Kerry Wood's 20 K | 20 | 9.0 innings pitched | 2.53 | 98 | -0.67 | Roger Clemens 20 (1986) = 21.1 | out: another game on its graph is as good or better |
| Clemens's 20 K | 20 | 9.0 innings pitched | 2.73 | 76 | -0.30 | Nolan Ryan 19 (1974) = 20.5 | out: another game on its graph is as good or better |
| Scherzer's 20 K | 20 | 9.0 innings pitched | 3.53 | 29 | -1.39 | Randy Johnson 20 (2001) = 21.9 | out: another game on its graph is as good or better |
| Feller's 18 K | 18 | 9.0 innings pitched | 3.91 | 19 | 0.02 | Dizzy Dean 17 (1933) = 18.0 | out: another game on its graph comes within a step |
| Adrian Peterson's 296, the record | 296 | regulation | 3.92 | 18 | 0.64 | Jamal Lewis 295 (2003) = 281.0 | out: another game on its graph comes within a step |
| Calvin Johnson's 329 | 329 | regulation | 4.28 | 11 | 0.59 | Jimmy Smith 291 (2000) = 313.4 | out: another game on its graph comes within a step |
| Burrow's 525 | 525 | regulation | 2.38 | 113 | -0.34 | Drew Brees 510 (2006) = 536.0 | out: another game on its graph is as good or better |
| Peyton Manning's 7 TD passes | 7 | regulation | 1.89 | 206 | 0.00 | Nick Foles 7 (2013) = 7.0 | out: another game on its graph is as good or better |
| Umenyiora's 6 sacks | 6 | regulation | 2.89 | 65 | 0.16 | Adrian Clayborn 6 (2017) = 5.9 | out: another game on its graph comes within a step |
| Van Brocklin's 554, the record | 554 | regulation | 5.86 | 3 | 1.99 | Y.A. Tittle 505 (1962) = 472.0 | **#3** |
| Warren Moon's 527 | 527 | regulation | 3.21 | 40 | 0.33 | Dan Marino 521 (1988) = 515.2 | out: another game on its graph comes within a step |
| Esiason's 522 (overtime) | 522 | regulation | 3.04 | 50 | -0.51 | Warren Moon 527 (1990) = 539.3 | out: another game on its graph is as good or better |
| Flipper Anderson's 336 (overtime), the record | 336 | overtime (length not recorded) | 3.39 | 35 | -0.44 | Stephone Paige 309 (1985) = 310.5 | out: another game on its graph is as good or better |
| Stephone Paige's 309 | 309 | regulation | 3.90 | 21 | 0.44 | Flipper Anderson 336 (1989) = 297.2 | out: another game on its graph comes within a step |
| Cloyce Box's 302 | 302 | regulation | 5.24 | 5 | 1.04 | Billy Howton 257 (1956) = 276.0 | **#5** |
| Walter Payton's 275 | 275 | regulation | 3.24 | 38 | -0.12 | Cookie Gilchrist 243 (1963) = 278.2 | out: another game on its graph is as good or better |
| O.J. Simpson's 273 | 273 | regulation | 2.87 | 67 | -0.58 | Cookie Gilchrist 243 (1963) = 288.6 | out: another game on its graph is as good or better |
| Derrick Thomas's 7 sacks, the record | 7 | regulation | 4.33 | 10 | 1.91 | Fred Dean 6 (1983) = 5.9 | **#10** |
| Adrian Burk's 7 TD passes | 7 | regulation | 2.99 | 56 | -0.14 | Joe Kapp 7 (1969) = 7.1 | out: another game on its graph is as good or better |
| Blanda's 7 TD passes (AFL) | 7 | regulation | 2.25 | 130 | -0.37 | Joe Namath 6 (1972) = 7.3 | out: another game on its graph is as good or better |
| Y.A. Tittle's 7 TD passes | 7 | regulation | 2.03 | 177 | -1.08 | Bob Griese 6 (1977) = 8.0 | out: another game on its graph is as good or better |
| Joe Kapp's 7 TD passes | 7 | regulation | 2.07 | 170 | -0.70 | Bob Griese 6 (1977) = 7.6 | out: another game on its graph is as good or better |
