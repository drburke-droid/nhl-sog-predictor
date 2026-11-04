# Video A review: the biggest single-game outliers in team sports

Model: `topk.py` (curve through each season's top 5 other games, ±15 seasons; length-adjusted games; league adjustment; counts treated as counts; clear of the field by one step). 3,908 season-leading games in 18 categories; 12 clear their field.

Numbers are in the athlete's own season: the curve and step are what a card would show; *best other* is the best game by anyone else on the same graph, converted to that season's scoring rate.

## The list

| # | Game | Sport | Raw | Length | Curve | Step | Score | Clear by | Best other on the graph | In top 10 of 16 versions |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Wilt Chamberlain, points in a game, 1962-03-02 | NBA | 100 | regulation | 50.2 | 5.51 | **8.94** | 5.23 | Elgin Baylor 71 (1961) = 71.2 | 16 |
| 2 | Kobe Bryant, points in a game, 2006-01-22 | NBA | 81 | regulation | 50.9 | 4.39 | **6.74** | 2.57 | David Robinson 71 (1994) = 69.7 | 16 |
| 3 | Jim Bottomley, runs batted in, one game, 1924-09-16 | MLB | 12 | 9.0 innings batted | 6.7 | 0.82 | **5.85** | 1.88 | Tony Lazzeri 11 (1936) = 10.5 | 14 |
| 4 | Bob Cousy, assists in a game, 1959-02-27 | NBA | 28 | regulation | 14.5 | 2.61 | **4.96** | 1.02 | Guy Rodgers 28 (1963) = 25.3 | 4 (one-sided window) |
| 5 | Mark Whiten, runs batted in, one game, 1993-09-07 | MLB | 12 | 9.0 innings batted | 7.3 | 0.85 | **4.91** | 2.42 | Alex Rodriguez 10 (2005) = 9.9 | 16 |
| 6 | Darryl Sittler, points in a game, 1976 | NHL | 10 | regulation | 6.1 | 0.72 | **4.74** | 2.55 | Tom Bladon 8 (1978) = 8.2 | 14 |
| 7 | Bert Olmstead, points in a game, 1954 | NHL | 8 | regulation | 4.6 | 0.65 | **4.49** | 1.87 | Maurice Richard 8 (1945) = 6.8 | 6 |
| 8 | John Roche, three-pointers in a game, 1982-01-09 | NBA | 8 | regulation | 4.3 | 0.75 | **4.31** | 1.05 | Rick Barry 8 (1980) = 7.2 | 4 (one-sided window) |
| 9 | Sam Gagner, points in a game, 2012 | NHL | 8 | regulation | 5.1 | 0.60 | **4.04** | 1.63 | Jaromir Jagr 7 (2003) = 7.0 | 15 |
| 10 | Billy Taylor, assists in a game, 1947 | NHL | 7 | regulation | 3.9 | 0.67 | **3.88** | 2.11 | Syl Apps 5 (1937) = 5.6 | 6 |
| 11 | Rennie Stennett, hits in a game, 1975-09-16 | MLB | 7 | 9.0 innings batted | 5.1 | 0.39 | **3.68** | 2.36 | Bob Oliver 6 (1969) = 6.1 | 7 |
| 12 | Mike Schmidt, home runs in a game, 1976-04-17 | MLB | 4 | 10.0 innings batted | 2.4 | 0.44 | **1.57** | 1.29 | Bob Horner 4 (1986) = 3.0 | 2 |

## Where the well-known games land

| Game | Raw | Length | Score | Rank among all season-best games | Clear by | Best other on the graph | Verdict |
|---|---|---|---|---|---|---|---|
| Wilt's 100, the NBA record | 100 | regulation | 8.94 | 1 | 5.23 | Elgin Baylor 71 (1961) = 71.2 | **#1** |
| Kobe's 81 | 81 | regulation | 6.74 | 2 | 2.57 | David Robinson 71 (1994) = 69.7 | **#2** |
| Doncic's 73 | 73 | regulation | 2.43 | 85 | 0.16 | Devin Booker 70 (2017) = 72.2 | out: another game on its graph comes within a step |
| Wilt's 55 rebounds, the record | 55 | n/a | 4.29 | 10 | 0.82 | Bill Russell 51 (1960) = 51.0 | out: another game on its graph comes within a step |
| Skiles's 30 assists, the record | 30 | regulation | 3.81 | 18 | 0.55 | Kevin Porter 29 (1978) = 28.8 | out: another game on its graph comes within a step |
| Cousy's 28 assists | 28 | regulation | 4.96 | 4 | 1.02 | Guy Rodgers 28 (1963) = 25.3 | **#4** |
| Klay Thompson's 14 threes, the record | 14 | regulation | 2.44 | 81 | -1.19 | Donyell Marshall 12 (2005) = 15.5 | out: another game on its graph is as good or better |
| Sittler's 10 points, the record | 10 | regulation | 4.74 | 6 | 2.55 | Tom Bladon 8 (1978) = 8.2 | **#6** |
| Joe Malone's 7 goals, the record | 7 | regulation | 1.09 | 360 | -1.34 | Art Gagne 4 (1928) = 8.8 | out: another game on its graph is as good or better |
| Red Berenson's 6 goals (1968-69) | 6 | regulation | 3.43 | 28 | 0.37 | Bernie Geoffrion 5 (1955) = 5.8 | out: another game on its graph comes within a step |
| Billy Taylor's 7 assists, the record | 7 | regulation | 3.88 | 17 | 2.11 | Syl Apps 5 (1937) = 5.6 | **#10** |
| Bottomley's 12 RBI | 12 | 9.0 innings batted | 5.85 | 3 | 1.88 | Tony Lazzeri 11 (1936) = 10.5 | **#3** |
| Whiten's 12 RBI (and 4 HR) | 12 | 9.0 innings batted | 4.91 | 5 | 2.42 | Alex Rodriguez 10 (2005) = 9.9 | **#5** |
| Stennett's 7-for-7 | 7 | 9.0 innings batted | 3.68 | 21 | 2.36 | Bob Oliver 6 (1969) = 6.1 | **#11** |
| Schmidt's 4 HR (10 innings) | 4 | 10.0 innings batted | 1.57 | 230 | 1.29 | Bob Horner 4 (1986) = 3.0 | **#12** |
| Hamilton's 4 HR | 4 | 9.0 innings batted | 1.41 | 263 | 0.17 | Mike Cameron 4 (2002) = 3.9 | out: another game on its graph comes within a step |
| Kerry Wood's 20 K | 20 | 9.0 innings pitched | 2.57 | 72 | -0.55 | Roger Clemens 20 (1986) = 20.9 | out: another game on its graph is as good or better |
| Clemens's 20 K | 20 | 9.0 innings pitched | 2.81 | 56 | -0.13 | Nolan Ryan 19 (1974) = 20.2 | out: another game on its graph is as good or better |
| Scherzer's 20 K | 20 | 9.0 innings pitched | 3.57 | 24 | -1.15 | Randy Johnson 20 (2001) = 21.5 | out: another game on its graph is as good or better |
| Feller's 18 K | 18 | 9.0 innings pitched | 3.96 | 15 | 0.14 | Dizzy Dean 17 (1933) = 17.8 | out: another game on its graph comes within a step |
| Adrian Peterson's 296, the record | 296 | regulation | 3.71 | 19 | 0.41 | Jamal Lewis 295 (2003) = 285.8 | out: another game on its graph comes within a step |
| Calvin Johnson's 329 | 329 | regulation | 4.36 | 8 | 0.90 | Jimmy Smith 291 (2000) = 305.1 | out: another game on its graph comes within a step |
| Burrow's 525 | 525 | regulation | 2.34 | 93 | -0.06 | Drew Brees 510 (2006) = 526.8 | out: another game on its graph is as good or better |
| Peyton Manning's 7 TD passes | 7 | regulation | 1.93 | 153 | 0.00 | Nick Foles 7 (2013) = 7.0 | out: another game on its graph is as good or better |
| Umenyiora's 6 sacks | 6 | regulation | 2.90 | 51 | 0.19 | Adrian Clayborn 6 (2017) = 5.9 | out: another game on its graph comes within a step |
