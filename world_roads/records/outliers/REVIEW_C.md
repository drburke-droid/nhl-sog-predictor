# Video C review: the biggest season-long outliers in team sports

Model: the season film's locked model (E3_main_env: best of everyone else in the 15 seasons either side, quadratic trend, league adjustment, the athlete's own seasons left out), team sports only, plus the rule used in all three films: a season must be at least 1 step clear of every other season on its graph. 2,607 season-leading performances in 22 categories; 39 clear their field.

Reproduction: 2129 of 2184 scores match the published model to 0.005 (NBA 269 of 270, NHL 246 of 252, MLB 1113 of 1151, NFL 501 of 511; top 30 of the old list: 30 of 30). The published model used the follow-up's deeper season lists (top 40, stable IDs), which can't be rebuilt without the original download cache; the original top-10 lists with the same identity fixes stand in for them, so a few seasons whose comparison runs past rank 10 or through a merged name score differently (all listed at the end).

Curve, step and *best other* are in the athlete's own season's units (*best other* is the best season by anyone else on the same graph, converted to that season's league rate).

## The film's five: in the top ten in at least 12 of the 16 versions

Every season below was considered; these are the ones the versions agree on (three-quarters or more), in score order. 3 of them are in the top ten of all 16.

| # | Season | Score | In top 10 of 16 versions |
|---|---|---|---|
| 1 | Wilt Chamberlain, points per game, 1961-62 (50.4) | 7.05 | 16 |
| 2 | Joe DiMaggio, hitting streak, 1941 (56) | 7.01 | 16 |
| 3 | Wayne Gretzky, assists in a season, 1985-86 (163) | 6.46 | 16 |
| 4 | Dennis Rodman, rebounds per game, 1992-93 (18.3) | 5.68 | 13 |
| 5 | Ichiro Suzuki, hits in a season, 2004 (262) | 5.27 | 15 |

## Considered, not in the film: the rest of the top ten by score

Each was in the top ten of fewer than 12 of the 16 versions (8 with the league adjustment, 8 without; windows of 7, 10, 15 and 20 seasons; straight or curved trend). What kept it out in the others:

- **Dazzy Vance, strikeouts by a pitcher, 1924 (262)**: #4 by score (5.83), top ten in 8 of 16. It rests on the league adjustment: strikeouts were scarce in 1924, but without that adjustment the big raw totals of 1904-1915 (Waddell 349, Johnson 313) win, and a 20-season window reaches Waddell's 1904 even with it. Out: Rube Waddell 349 (1904) within a step or ahead in 4 (adjusted or not, 20-season window); Walter Johnson 313 (1910) within a step or ahead in 2 (unadjusted, 15-season window); Grover Alexander 241 (1915) within a step or ahead in 1 (unadjusted, 10-season window, straight trend); below the top ten in 1 (unadjusted, 7-season window, straight trend; ranked #15).
- **Babe Ruth, home runs in a season, 1920 (54)**: #6 by score (5.56), top ten in 6 of 16. It rests on the league adjustment: without it, the home-run boom Ruth started catches him (Gehrig, Wilson's 56 in 1930, Foxx's 58 in 1932); with it, a 7-season window ranks it 12th or 19th. Out: Jimmie Foxx 58 (1932) within a step or ahead in 4 (unadjusted, 15/20-season windows); Lou Gehrig 47 (1927) within a step or ahead in 2 (unadjusted, 7-season window); Hack Wilson 56 (1930) within a step or ahead in 2 (unadjusted, 10-season window); below the top ten in 2 (adjusted, 7-season window; ranked #12, #19).
- **Bob Feller, strikeouts by a pitcher, 1946 (348)**: #8 by score (5.03), top ten in 9 of 16. Always clear of its own era, but with 7- or 10-season windows it ranks 12th to 19th, and the widest window without adjustment reaches Koufax's 382 in 1965. Out: Sandy Koufax 382 (1965) within a step or ahead in 2 (unadjusted, 20-season window); below the top ten in 5 (adjusted or not, 7/10-season windows; ranked #12, #14, #14, #15, #19).
- **Maury Wills, stolen bases in a season, 1962 (104)**: #9 by score (4.78), top ten in 9 of 16. Without the league adjustment, the steals boom that followed him wins (Brock's 118 in 1974, Henderson's 130 in 1982); with 7-season windows it ranks 11th to 21st. Out: Lou Brock 118 (1974) within a step or ahead in 2 (unadjusted, 15-season window); Rickey Henderson 130 (1982) within a step or ahead in 2 (unadjusted, 20-season window); below the top ten in 3 (adjusted or not, 7-season window; ranked #11, #18, #21).
- **Ted Williams, batting average, 1941 (.406)**: #10 by score (4.77), top ten in 3 of 16. Without the league adjustment, the high averages of the 1920s and 30s (Hornsby's .424, Terry's .401) come within a step or ahead; with it, only the 15- and 20-season windows put it in the ten. Out: Bill Terry .401 (1930) within a step or ahead in 2 (unadjusted, 15-season window); Rogers Hornsby .424 (1924) within a step or ahead in 2 (unadjusted, 20-season window); below the top ten in 9 (adjusted or not, 7/10/20-season windows; ranked #11, #13, #14, #16, #16, #23, #24, #24, #27).

## The list

| # | Season | Sport | Value | Curve | Step | Score | Published score | Clear by | Best other on the graph | In top 10 of 16 versions |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Wilt Chamberlain, points per game, 1961-62 | NBA | 50.4 | 31.79 | 2.64 | **7.05** | 7.0479 | 3.94 | George Mikan 28.3 (1948-49) = 40 | 16 |
| 2 | Joe DiMaggio, hitting streak, 1941 | MLB | 56.0 | 28.13 | 3.98 | **7.01** | 7.0112 | 4.78 | Tommy Holmes 37.0 (1945) = 37 | 16 |
| 3 | Wayne Gretzky, assists in a season, 1985-86 | NHL | 163.0 | 92.82 | 10.9 | **6.46** | 6.4657 | 2.80 | Bobby Orr 102.0 (1970-71) = 132.6 | 16 |
| 4 | Dazzy Vance, strikeouts by a pitcher, 1924 | MLB | 262.0 | 153.8 | 18.6 | **5.83** | 5.8304 | 2.47 | Walter Johnson 313.0 (1910) = 216.2 | 8 |
| 5 | Dennis Rodman, rebounds per game, 1992-93 | NBA | 18.3 | 13.36 | 0.87 | **5.68** | 5.6783 | 1.57 | Moses Malone 17.6 (1978-79) = 16.94 | 13 |
| 6 | Babe Ruth, home runs in a season, 1920 | MLB | 54.0 | 22.96 | 5.59 | **5.56** | 5.5571 | 2.97 | Gavy Cravath 24.0 (1915) = 37.4 | 6 |
| 7 | Ichiro Suzuki, hits in a season, 2004 | MLB | 262.0 | 218.9 | 8.18 | **5.27** | 5.2704 | 2.63 | Jose Altuve 225.0 (2014) = 240.5 | 15 |
| 8 | Bob Feller, strikeouts by a pitcher, 1946 | MLB | 348.0 | 205.2 | 28.4 | **5.03** | 5.0307 | 2.44 | Van Mungo 238.0 (1936) = 278.7 | 9 |
| 9 | Maury Wills, stolen bases in a season, 1962 | MLB | 104.0 | 55.12 | 10.2 | **4.78** | 4.7802 | 2.25 | Lou Brock 118.0 (1974) = 80.95 | 9 |
| 10 | Ted Williams, batting average, 1941 | MLB | 0.406 | 0.3575 | 0.0102 | **4.77** | 4.7726 | 3.07 | Stan Musial 0.376 (1948) = 0.3748 | 3 |
| 11 | Dan Marino, touchdown passes in a season, 1984 | NFL | 48.0 | 31.86 | 3.39 | **4.76** | 4.7588 | 2.13 | Brett Favre 39.0 (1996) = 40.79 | 8 |
| 12 | Tim Keefe, earned run average, 1880 | MLB | 0.86 | 1.467 | 0.194 | **4.62** | 4.6216 | 2.09 | Old Hoss Radbourn 1.38 (1884) = 1.058 | 2 (one-sided window) |

## What the rule removes from the old top ten (team sports)

| Season | Score | Clear by | Best other on the graph |
|---|---|---|---|
| Stephen Curry, three-pointers made, 2015-16 | 5.69 | 0.68 | Ray Allen 269.0 (2005-06) = 383.3 |

## Where the well-known seasons land

| Season | Value | Score | Rank among all team seasons | Clear by | Best other on the graph | Verdict |
|---|---|---|---|---|---|---|
| Wilt's 50.4 points a game | 50.4 | 7.05 | 1 | 3.94 | George Mikan 28.3 (1948-49) = 40 | **#1** |
| Curry's 402 threes | 402.0 | 5.69 | 5 | 0.68 | Ray Allen 269.0 (2005-06) = 383.3 | out: another season on its graph comes within a step |
| Rodman's 18.3 rebounds | 18.3 | 5.68 | 6 | 1.57 | Moses Malone 17.6 (1978-79) = 16.94 | **#5** |
| Jordan's 37.1 points a game | 37.1 | 4.26 | 16 | 0.28 | Bob McAdoo 34.5 (1974-75) = 36.64 | out: another season on its graph comes within a step |
| Stockton's 14.5 assists | 14.5 | 2.36 | 80 | 0.86 | Steve Nash 11.5 (2004-05) = 13.45 | out: another season on its graph comes within a step |
| Wilt's 27.2 rebounds | 27.2 | 2.03 | not the athlete best | 0.15 | Bill Russell 24.7 (1963-64) = 26.92 | out: another season on its graph comes within a step |
| Gretzky's 163 assists | 163.0 | 6.46 | 3 | 2.80 | Bobby Orr 102.0 (1970-71) = 132.6 | **#3** |
| Gretzky's 92 goals | 92.0 | 1.90 | not the athlete best | -0.72 | Phil Esposito 76.0 (1970-71) = 99.51 | out: another season on its graph is as good or better |
| Orr's 102 assists | 102.0 | 4.56 | 14 | -3.09 | Wayne Gretzky 163.0 (1985-86) = 125.4 | out: another season on its graph is as good or better |
| Esposito's 76 goals | 76.0 | 3.69 | 25 | 0.91 | Wayne Gretzky 92.0 (1981-82) = 70.27 | out: another season on its graph comes within a step |
| Lemieux's 199 points | 199.0 | 1.44 | not the athlete best | -0.24 | Wayne Gretzky 215.0 (1985-86) = 204 | out: another season on its graph is as good or better |
| Howe's 95 points | 95.0 | 4.13 | 17 | 1.80 | Bill Cowley 64.0 (1940-41) = 83.77 | in, below the top ten |
| Babe Dye's 38 goals | 38.0 | 4.00 | 19 | 2.07 | Joe Malone 44.0 (1917-18) = 32.13 | in, below the top ten |
| DiMaggio's 56-game streak | 56.0 | 7.01 | 2 | 4.78 | Tommy Holmes 37.0 (1945) = 37 | **#2** |
| Ruth's 54 home runs | 54.0 | 5.56 | 7 | 2.97 | Gavy Cravath 24.0 (1915) = 37.4 | **#6** |
| Bonds's 73 home runs | 73.0 | 2.00 | 118 | -0.28 | Mark McGwire 70.0 (1998) = 74.88 | out: another season on its graph is as good or better |
| Ichiro's 262 hits | 262.0 | 5.27 | 8 | 2.63 | Jose Altuve 225.0 (2014) = 240.5 | **#7** |
| Vance's 262 strikeouts | 262.0 | 5.83 | 4 | 2.47 | Walter Johnson 313.0 (1910) = 216.2 | **#4** |
| Feller's 348 strikeouts | 348.0 | 5.03 | 9 | 2.44 | Van Mungo 238.0 (1936) = 278.7 | **#8** |
| Ryan's 383 strikeouts | 383.0 | 3.29 | 35 | 1.32 | J.R. Richard 313.0 (1979) = 343.7 | in, below the top ten |
| Wills's 104 steals | 104.0 | 4.78 | 10 | 2.25 | Lou Brock 118.0 (1974) = 80.95 | **#9** |
| Henderson's 130 steals | 130.0 | 3.01 | 45 | -0.44 | Lou Brock 118.0 (1974) = 136.2 | out: another season on its graph is as good or better |
| Ted Williams's .406 | 0.406 | 4.77 | 11 | 3.07 | Stan Musial 0.376 (1948) = 0.3748 | **#10** |
| Gibson's 1.12 ERA | 1.123 | 4.06 | 18 | 2.02 | Dean Chance 1.649 (1964) = 1.331 | in, below the top ten |
| Hack Wilson's 191 RBI | 191.0 | 0.33 | 506 | -1.44 | Lou Gehrig 185.0 (1931) = 214.1 | out: another season on its graph is as good or better |
| McLain's 31 wins | 31.0 | 3.77 | 23 | 1.55 | Don Newcombe 27.0 (1956) = 28.4 | in, below the top ten |
| Marino's 48 TD passes | 48.0 | 4.76 | 12 | 2.13 | Brett Favre 39.0 (1996) = 40.79 | in, below the top ten |
| Jim Brown's 1,863 yards | 1863.0 | 3.90 | 22 | 1.05 | O.J. Simpson 2003.0 (1973) = 1669 | in, below the top ten |
| Dickerson's 2,105 yards | 2105.0 | 2.07 | 112 | -0.69 | Barry Sanders 2053.0 (1997) = 2263 | out: another season on its graph is as good or better |
| Hutson's 1,211 yards | 1211.0 | 3.34 | 34 | 0.60 | Jim Benton 1067.0 (1945) = 1111 | out: another season on its graph comes within a step |
| Peyton Manning's 5,477 yards | 5477.0 | 1.11 | not the athlete best | -0.88 | Drew Brees 5069.0 (2008) = 5714 | out: another season on its graph is as good or better |
| Night Train Lane's 14 interceptions | 14.0 | 1.66 | 156 | 0.38 | Dan Sandifer 13.0 (1948) = 13.45 | out: another season on its graph comes within a step |

## Scores that differ from the published model

| Season | Score here | Published |
|---|---|---|
| Old Hoss Radbourn, strikeouts by a pitcher, 1884 | -0.112 | 2.277 |
| Honus Wagner, stolen bases in a season, 1908 | -0.561 | 1.389 |
| Harry Davis, runs batted in, 1906 | -0.928 | 0.788 |
| Amos Rusie, wins by a pitcher, 1894 | 0.338 | 1.819 |
| Paul Hines, runs batted in, 1878 | -0.408 | 0.784 |
| Carl Hubbell, wins by a pitcher, 1936 | 0.182 | 1.202 |
| Harry Newman, passing yards in a season, 1933 | 0.797 | 1.720 |
| Milt Schmidt, assists in a season, 1940 | -0.626 | 0.290 |
| Hal Newhouser, earned run average, 1946 | -0.016 | 0.882 |
| Walter Johnson, wins by a pitcher, 1913 | 1.435 | 2.221 |
| Ed Delahanty, runs batted in, 1893 | -0.103 | 0.682 |
| Pete Rose, batting average, 1968 | 0.239 | 0.903 |
| Hal Trosky, runs batted in, 1936 | -0.770 | -0.137 |
| Ryan Howard, runs batted in, 2006 | -0.213 | 0.401 |
| Cy Seymour, batting average, 1905 | 0.610 | 1.186 |
| Lady Baldwin, strikeouts by a pitcher, 1886 | -1.200 | -0.637 |
| Nikita Kucherov, points in a season, 2025 | -0.667 | -0.121 |
| Nikita Kucherov, points in a season, 2024 | 1.295 | 1.833 |
| Chris Sale, earned run average, 2024 | -1.496 | -0.960 |
| Guy Lafleur, points in a season, 1978 | -0.838 | -0.356 |
| Sandy Koufax, strikeouts by a pitcher, 1966 | 0.619 | 1.081 |
| Ronald Acuña Jr., stolen bases in a season, 2023 | 1.273 | 1.710 |
| Lou Brock, hitting streak, 1971 | -0.105 | 0.321 |
| Tommy Bond, wins by a pitcher, 1877 | 1.103 | 1.499 |
| Dante Bichette, runs batted in, 1995 | -0.636 | -0.262 |
| Jimmy Ryan, home runs in a season, 1888 | -0.379 | -0.028 |
| Ted Kluszewski, home runs in a season, 1954 | 0.467 | 0.815 |
| Mickey Mantle, home runs in a season, 1956 | 0.216 | 0.554 |
| Sid Luckman, touchdown passes in a season, 1943 | 2.234 | 2.563 |
| Babe Parilli, passing yards in a season, 1964 | -0.393 | -0.090 |
| Joe Burrow, touchdown passes in a season, 2024 | 0.190 | 0.492 |
| Ty Cobb, hitting streak, 1917 | 1.636 | 1.931 |
| Tim Keefe, earned run average, 1888 | -0.326 | -0.039 |
| Whit Merrifield, hits in a season, 2018 | -1.010 | -0.724 |
| Cap Anson, batting average, 1881 | 1.374 | 1.642 |
| Dan Marino, passing yards in a season, 1985 | -0.734 | -0.488 |
| Mickey Lolich, strikeouts by a pitcher, 1971 | -0.170 | 0.063 |
| Johnny Pesky, hitting streak, 1947 | -0.469 | -0.281 |
| Bill Cook, points in a season, 1933 | -0.707 | -0.525 |
| Adrian Peterson, rushing yards in a season, 2008 | 0.060 | -0.122 |
| Ty Cobb, stolen bases in a season, 1909 | 1.129 | 1.294 |
| Kiki Cuyler, hitting streak, 1926 | -0.746 | -0.603 |
| Chris Davis, home runs in a season, 2013 | 1.548 | 1.686 |
| Pedro Martínez, wins by a pitcher, 1999 | 0.703 | 0.814 |
| Walter Johnson, strikeouts by a pitcher, 1912 | 1.330 | 1.440 |
| Nels Stewart, goals in a season, 1926 | 1.166 | 1.273 |
| Tobin Rote, touchdown passes in a season, 1956 | -0.179 | -0.080 |
| Jimmie Foxx, home runs in a season, 1932 | 0.530 | 0.614 |
| Ernie Banks, home runs in a season, 1958 | -0.206 | -0.123 |
| Neil Johnston, points per game, 1955 | -1.122 | -1.042 |
| José Canseco, home runs in a season, 1988 | -0.016 | 0.039 |
| Tommy Davis, runs batted in, 1962 | 1.009 | 1.039 |
| Ty Law, interceptions in a season, 2005 | 1.295 | 1.265 |
| Adrian Peterson, rushing yards in a season, 2012 | 2.378 | 2.355 |
| Adrian Peterson, rushing yards in a season, 2015 | -0.397 | -0.416 |
