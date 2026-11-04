# Video C review: the biggest season-long outliers in team sports

Model: the season film's locked model (E3_main_env: best of everyone else in the 15 seasons either side, quadratic trend, league adjustment, the athlete's own seasons left out), team sports only, plus the rule used in all three films: a season must be at least 1 step clear of every other season on its graph. 2,299 season-leading performances in 22 categories; 37 clear their field.

Reproduction: 1638 of 1906 scores match the published model to 0.005 (NBA 269 of 270, NHL 246 of 252, MLB 1113 of 1151, NFL 10 of 233; top 30 of the old list: 28 of 30). The published model used the follow-up's deeper season lists (top 40, stable IDs), which can't be rebuilt without the original download cache; the original top-10 lists with the same identity fixes stand in for them, so a few seasons whose comparison runs past rank 10 or through a merged name score differently (all listed at the end).

Curve, step and *best other* are in the athlete's own season's units (*best other* is the best season by anyone else on the same graph, converted to that season's league rate).

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
| 11 | Tim Keefe, earned run average, 1880 | MLB | 0.86 | 1.467 | 0.194 | **4.62** | 4.6216 | 2.09 | Old Hoss Radbourn 1.38 (1884) = 1.058 | 2 (one-sided window) |
| 12 | Pete Rose, hitting streak, 1978 | MLB | 44.0 | 25.98 | 4.16 | **4.33** | 4.3337 | 1.20 | Paul Molitor 39.0 (1987) = 39 | 11 |

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
| Stockton's 14.5 assists | 14.5 | 2.36 | 71 | 0.86 | Steve Nash 11.5 (2004-05) = 13.45 | out: another season on its graph comes within a step |
| Wilt's 27.2 rebounds | 27.2 | 2.03 | not the athlete best | 0.15 | Bill Russell 24.7 (1963-64) = 26.92 | out: another season on its graph comes within a step |
| Gretzky's 163 assists | 163.0 | 6.46 | 3 | 2.80 | Bobby Orr 102.0 (1970-71) = 132.6 | **#3** |
| Gretzky's 92 goals | 92.0 | 1.90 | not the athlete best | -0.72 | Phil Esposito 76.0 (1970-71) = 99.51 | out: another season on its graph is as good or better |
| Orr's 102 assists | 102.0 | 4.56 | 13 | -3.09 | Wayne Gretzky 163.0 (1985-86) = 125.4 | out: another season on its graph is as good or better |
| Esposito's 76 goals | 76.0 | 3.69 | 23 | 0.91 | Wayne Gretzky 92.0 (1981-82) = 70.27 | out: another season on its graph comes within a step |
| Lemieux's 199 points | 199.0 | 1.44 | not the athlete best | -0.24 | Wayne Gretzky 215.0 (1985-86) = 204 | out: another season on its graph is as good or better |
| Howe's 95 points | 95.0 | 4.13 | 17 | 1.80 | Bill Cowley 64.0 (1940-41) = 83.77 | in, below the top ten |
| Babe Dye's 38 goals | 38.0 | 4.00 | 19 | 2.07 | Joe Malone 44.0 (1917-18) = 32.13 | in, below the top ten |
| DiMaggio's 56-game streak | 56.0 | 7.01 | 2 | 4.78 | Tommy Holmes 37.0 (1945) = 37 | **#2** |
| Ruth's 54 home runs | 54.0 | 5.56 | 7 | 2.97 | Gavy Cravath 24.0 (1915) = 37.4 | **#6** |
| Bonds's 73 home runs | 73.0 | 2.00 | 106 | -0.28 | Mark McGwire 70.0 (1998) = 74.88 | out: another season on its graph is as good or better |
| Ichiro's 262 hits | 262.0 | 5.27 | 8 | 2.63 | Jose Altuve 225.0 (2014) = 240.5 | **#7** |
| Vance's 262 strikeouts | 262.0 | 5.83 | 4 | 2.47 | Walter Johnson 313.0 (1910) = 216.2 | **#4** |
| Feller's 348 strikeouts | 348.0 | 5.03 | 9 | 2.44 | Van Mungo 238.0 (1936) = 278.7 | **#8** |
| Ryan's 383 strikeouts | 383.0 | 3.29 | 32 | 1.32 | J.R. Richard 313.0 (1979) = 343.7 | in, below the top ten |
| Wills's 104 steals | 104.0 | 4.78 | 10 | 2.25 | Lou Brock 118.0 (1974) = 80.95 | **#9** |
| Henderson's 130 steals | 130.0 | 3.01 | 43 | -0.44 | Lou Brock 118.0 (1974) = 136.2 | out: another season on its graph is as good or better |
| Ted Williams's .406 | 0.406 | 4.77 | 11 | 3.07 | Stan Musial 0.376 (1948) = 0.3748 | **#10** |
| Gibson's 1.12 ERA | 1.123 | 4.06 | 18 | 2.02 | Dean Chance 1.649 (1964) = 1.331 | in, below the top ten |
| Hack Wilson's 191 RBI | 191.0 | 0.33 | 433 | -1.44 | Lou Gehrig 185.0 (1931) = 214.1 | out: another season on its graph is as good or better |
| McLain's 31 wins | 31.0 | 3.77 | 22 | 1.55 | Don Newcombe 27.0 (1956) = 28.4 | in, below the top ten |
| Marino's 48 TD passes | - | - | - | - | - | not scored (not that season's leader, or no league rate) |
| Jim Brown's 1,863 yards | - | - | - | - | - | not scored (not that season's leader, or no league rate) |
| Dickerson's 2,105 yards | - | - | - | - | - | not scored (not that season's leader, or no league rate) |
| Hutson's 1,211 yards | - | - | - | - | - | not scored (not that season's leader, or no league rate) |
| Peyton Manning's 5,477 yards | 5477.0 | 1.11 | not the athlete best | -0.86 | Drew Brees 5069.0 (2008) = 5714 | out: another season on its graph is as good or better |
| Night Train Lane's 14 interceptions | 14.0 | 1.65 | 140 | 0.57 | Lindon Crow 11.0 (1956) = 13.13 | out: another season on its graph comes within a step |

## Scores that differ from the published model

| Season | Score here | Published |
|---|---|---|
| Otto Graham, passing yards in a season, 1952 | 4.313 | 0.582 |
| Old Hoss Radbourn, strikeouts by a pitcher, 1884 | -0.112 | 2.277 |
| Honus Wagner, stolen bases in a season, 1908 | -0.561 | 1.389 |
| Otto Graham, passing yards in a season, 1953 | 1.887 | 0.021 |
| Harry Davis, runs batted in, 1906 | -0.928 | 0.788 |
| Amos Rusie, wins by a pitcher, 1894 | 0.338 | 1.819 |
| Paul Hines, runs batted in, 1878 | -0.408 | 0.784 |
| Bobby Layne, passing yards in a season, 1950 | 0.143 | -1.036 |
| Elroy Hirsch, receiving yards in a season, 1951 | 2.572 | 1.461 |
| Johnny Unitas, touchdown passes in a season, 1959 | 2.350 | 3.453 |
| Bobby Layne, touchdown passes in a season, 1951 | 1.778 | 0.733 |
| Carl Hubbell, wins by a pitcher, 1936 | 0.182 | 1.202 |
| George Ratterman, touchdown passes in a season, 1950 | 0.650 | -0.314 |
| Joe Perry, rushing yards in a season, 1953 | 1.114 | 0.193 |
| Milt Schmidt, assists in a season, 1940 | -0.626 | 0.290 |
| Hal Newhouser, earned run average, 1946 | -0.016 | 0.882 |
| Joe Perry, rushing yards in a season, 1954 | 1.076 | 0.186 |
| Marion Motley, rushing yards in a season, 1950 | -0.433 | -1.250 |
| Walter Johnson, wins by a pitcher, 1913 | 1.435 | 2.221 |
| Ed Delahanty, runs batted in, 1893 | -0.103 | 0.682 |
| Billy Atkins, interceptions in a season, 1961 | -0.473 | -1.181 |
| Jim Brown, rushing yards in a season, 1958 | 3.381 | 2.697 |
| Dan Towler, rushing yards in a season, 1952 | 0.334 | -0.341 |
| Pete Rose, batting average, 1968 | 0.239 | 0.903 |
| Eddie Price, rushing yards in a season, 1951 | 0.209 | -0.449 |
| Goose Gonsoulin, interceptions in a season, 1960 | 0.705 | 0.057 |
| Billy Howton, receiving yards in a season, 1956 | 1.930 | 1.290 |
| Hal Trosky, runs batted in, 1936 | -0.770 | -0.137 |
| Ryan Howard, runs batted in, 2006 | -0.213 | 0.401 |
| Billy Wade, passing yards in a season, 1958 | -0.038 | 0.568 |
| James Hasty, interceptions in a season, 1999 | -1.710 | -2.295 |
| Sam Madison, interceptions in a season, 1999 | -1.710 | -2.295 |
| Troy Vincent, interceptions in a season, 1999 | -1.710 | -2.295 |
| Rod Woodson, interceptions in a season, 1999 | -1.710 | -2.295 |
| Cy Seymour, batting average, 1905 | 0.610 | 1.186 |
| Lady Baldwin, strikeouts by a pitcher, 1886 | -1.200 | -0.637 |
| Tobin Rote, touchdown passes in a season, 1955 | -0.285 | -0.843 |
| Nikita Kucherov, points in a season, 2025 | -0.667 | -0.121 |
| Nikita Kucherov, points in a season, 2024 | 1.295 | 1.833 |
| Chris Sale, earned run average, 2024 | -1.496 | -0.960 |
| Jamal Lewis, rushing yards in a season, 2003 | 1.423 | 0.914 |
| Tom Brady, passing yards in a season, 2005 | -1.576 | -1.080 |
| Guy Lafleur, points in a season, 1978 | -0.838 | -0.356 |
| Ricky Williams, rushing yards in a season, 2002 | 0.434 | -0.031 |
| Sandy Koufax, strikeouts by a pitcher, 1966 | 0.619 | 1.081 |
| Tom Fears, receiving yards in a season, 1950 | -0.028 | -0.479 |
| Jim Brown, rushing yards in a season, 1959 | 1.827 | 1.387 |
| Ronald Acuña Jr., stolen bases in a season, 2023 | 1.273 | 1.710 |
| Edgerrin James, rushing yards in a season, 2000 | 0.351 | -0.083 |
| T.J. Watt, sacks in a season, 2023 | 0.164 | -0.268 |
| Pete Pihos, receiving yards in a season, 1955 | -0.437 | -0.864 |
| Lou Brock, hitting streak, 1971 | -0.105 | 0.321 |
| Sonny Jurgensen, passing yards in a season, 1961 | 1.540 | 1.117 |
| T.J. Watt, sacks in a season, 2021 | 2.087 | 1.666 |
| Jim Finks, touchdown passes in a season, 1952 | -0.633 | -1.044 |
| Jimmy Smith, receiving yards in a season, 1999 | -0.006 | 0.391 |
| Tommy Bond, wins by a pitcher, 1877 | 1.103 | 1.499 |
| Xavien Howard, interceptions in a season, 2020 | 2.053 | 2.431 |
| Tom Brady, touchdown passes in a season, 2002 | -1.392 | -1.770 |
| Trey Hendrickson, sacks in a season, 2024 | -0.171 | -0.548 |
| Dante Bichette, runs batted in, 1995 | -0.636 | -0.262 |
| Calvin Johnson, receiving yards in a season, 2011 | 0.791 | 0.433 |
| Rick Casares, rushing yards in a season, 1956 | -0.430 | -0.782 |
| Jimmy Ryan, home runs in a season, 1888 | -0.379 | -0.028 |
| Calvin Johnson, receiving yards in a season, 2012 | 3.216 | 2.866 |
| Bobby Thomason, touchdown passes in a season, 1953 | 0.069 | -0.280 |
| Ted Kluszewski, home runs in a season, 1954 | 0.467 | 0.815 |
| Dexter McCleon, interceptions in a season, 2000 | -0.572 | -0.918 |
| Darren Sharper, interceptions in a season, 2000 | -0.572 | -0.918 |
| Mickey Mantle, home runs in a season, 1956 | 0.216 | 0.554 |
| Jim Brown, rushing yards in a season, 1957 | -0.234 | -0.563 |
| Edgerrin James, rushing yards in a season, 1999 | -0.307 | -0.620 |
| Alan Ameche, rushing yards in a season, 1955 | -0.821 | -1.125 |
| Ty Cobb, hitting streak, 1917 | 1.636 | 1.931 |
| Tim Keefe, earned run average, 1888 | -0.326 | -0.039 |
| Whit Merrifield, hits in a season, 2018 | -1.010 | -0.724 |
| Jimmy Patton, interceptions in a season, 1958 | 1.466 | 1.180 |
| Jason Taylor, sacks in a season, 2002 | 0.496 | 0.781 |
| Brian Kelly, interceptions in a season, 2002 | -0.698 | -0.983 |
| Rod Woodson, interceptions in a season, 2002 | -0.698 | -0.983 |
| Spec Sanders, interceptions in a season, 1950 | 0.029 | 0.314 |
| Shaun Alexander, rushing yards in a season, 2005 | 0.823 | 0.540 |
| Y.A. Tittle, touchdown passes in a season, 1955 | -0.649 | -0.931 |
| Tobin Rote, touchdown passes in a season, 1956 | 0.201 | -0.080 |
| Nick Bosa, sacks in a season, 2022 | -0.035 | -0.313 |
| Cap Anson, batting average, 1881 | 1.374 | 1.642 |
| Charley Hennigan, receiving yards in a season, 1961 | 2.061 | 1.808 |
| Drew Brees, passing yards in a season, 2006 | -0.227 | 0.017 |
| Ja'Marr Chase, receiving yards in a season, 2024 | 0.228 | -0.011 |
| Daunte Culpepper, passing yards in a season, 2004 | 0.261 | 0.498 |
| Dwight Freeney, sacks in a season, 2004 | -0.817 | -0.581 |
| Priest Holmes, rushing yards in a season, 2001 | -0.858 | -1.092 |
| Mickey Lolich, strikeouts by a pitcher, 1971 | -0.170 | 0.063 |
| Raymond Berry, receiving yards in a season, 1957 | -1.186 | -0.955 |
| Otto Schnellbacher, interceptions in a season, 1951 | -0.596 | -0.366 |
| Billy Howton, receiving yards in a season, 1952 | 0.539 | 0.310 |
| Peyton Manning, passing yards in a season, 2003 | -0.393 | -0.165 |
| Dick Lane, interceptions in a season, 1954 | -0.847 | -1.075 |
| Derrick Burgess, sacks in a season, 2005 | -0.900 | -0.672 |
| Adrian Peterson, rushing yards in a season, 2008 | 0.094 | -0.122 |
| Drew Brees, passing yards in a season, 2012 | 0.930 | 0.716 |
| Drew Brees, passing yards in a season, 2011 | 2.574 | 2.361 |
| Michael Strahan, sacks in a season, 2001 | 2.433 | 2.644 |
| Curtis Martin, rushing yards in a season, 2004 | -0.692 | -0.902 |
| T.J. Watt, sacks in a season, 2020 | -0.720 | -0.929 |
| Cooper Kupp, receiving yards in a season, 2021 | 1.432 | 1.229 |
| Kurt Warner, passing yards in a season, 2001 | 1.523 | 1.320 |
| Johnny Unitas, touchdown passes in a season, 1957 | 1.267 | 1.466 |
| Tyreek Hill, receiving yards in a season, 2023 | 0.585 | 0.386 |
| Andre Johnson, receiving yards in a season, 2008 | 0.688 | 0.493 |
| Raymond Berry, receiving yards in a season, 1959 | -0.822 | -0.630 |
| David Boston, receiving yards in a season, 2001 | 0.089 | 0.278 |
| Justin Jefferson, receiving yards in a season, 2022 | 0.889 | 0.700 |
| Johnny Pesky, hitting streak, 1947 | -0.469 | -0.281 |
| Jim Brown, rushing yards in a season, 1960 | 1.412 | 1.226 |
| Bill Cook, points in a season, 1933 | -0.707 | -0.525 |
| Del Shofner, receiving yards in a season, 1958 | -0.425 | -0.245 |
| Peyton Manning, touchdown passes in a season, 2004 | 2.135 | 2.315 |
| Kerby Joseph, interceptions in a season, 2024 | 0.768 | 0.945 |
| Andre Johnson, receiving yards in a season, 2009 | 0.258 | 0.087 |
| Peyton Manning, passing yards in a season, 2000 | -0.124 | 0.044 |
| Tony Parrish, interceptions in a season, 2003 | 0.079 | -0.088 |
| Brian Russell, interceptions in a season, 2003 | 0.079 | -0.088 |
| Ty Cobb, stolen bases in a season, 1909 | 1.129 | 1.294 |
| Dak Prescott, touchdown passes in a season, 2023 | -0.457 | -0.620 |
| Michael Strahan, sacks in a season, 2003 | 1.105 | 1.259 |
| Torry Holt, receiving yards in a season, 2000 | 0.378 | 0.528 |
| Torry Holt, receiving yards in a season, 2003 | 1.594 | 1.738 |
| Ed Reed, interceptions in a season, 2008 | 0.927 | 1.071 |
| Kiki Cuyler, hitting streak, 1926 | -0.746 | -0.603 |
| Stephon Gilmore, interceptions in a season, 2019 | -1.387 | -1.245 |
| Anthony Harris, interceptions in a season, 2019 | -1.387 | -1.245 |
| Tre'Davious White, interceptions in a season, 2019 | -1.387 | -1.245 |
| Ty Law, interceptions in a season, 2005 | 1.123 | 1.265 |
| Deltha O'Neal, interceptions in a season, 2005 | 1.123 | 1.265 |
| George Blanda, touchdown passes in a season, 1961 | 1.302 | 1.442 |
| Chris Davis, home runs in a season, 2013 | 1.548 | 1.686 |
| Tom Brady, touchdown passes in a season, 2007 | 2.250 | 2.388 |
| Steve Beuerlein, passing yards in a season, 1999 | -0.256 | -0.121 |
| DeMarcus Ware, sacks in a season, 2008 | 1.910 | 2.042 |
| Shaquil Barrett, sacks in a season, 2019 | 0.049 | -0.081 |
| Jim Brown, rushing yards in a season, 1961 | 1.610 | 1.482 |
| Joe Burrow, touchdown passes in a season, 2024 | 0.366 | 0.492 |
| LaDainian Tomlinson, rushing yards in a season, 2006 | 0.144 | 0.025 |
| Charles Woodson, interceptions in a season, 2011 | -1.171 | -1.289 |
| Eric Weddle, interceptions in a season, 2011 | -1.171 | -1.289 |
| Kyle Arrington, interceptions in a season, 2011 | -1.171 | -1.289 |
| DeMarcus Ware, sacks in a season, 2010 | -1.071 | -0.957 |
| Tua Tagovailoa, passing yards in a season, 2023 | -0.902 | -1.016 |
| Peyton Manning, touchdown passes in a season, 2013 | 2.683 | 2.796 |
| Daron Bland, interceptions in a season, 2023 | 0.421 | 0.534 |
| Patrick Mahomes, passing yards in a season, 2022 | 1.959 | 1.847 |
| Pedro Martínez, wins by a pitcher, 1999 | 0.703 | 0.814 |
| Walter Johnson, strikeouts by a pitcher, 1912 | 1.330 | 1.440 |
| Adrian Burk, touchdown passes in a season, 1954 | -0.048 | -0.158 |
| Peyton Manning, touchdown passes in a season, 2006 | -0.631 | -0.740 |
| Dean Derby, interceptions in a season, 1959 | -1.570 | -1.677 |
| Milt Davis, interceptions in a season, 1959 | -1.570 | -1.677 |
| Don Shinnick, interceptions in a season, 1959 | -1.570 | -1.677 |
| Nels Stewart, goals in a season, 1926 | 1.166 | 1.273 |
| Kurt Warner, touchdown passes in a season, 1999 | 0.776 | 0.671 |
| Norm Van Brocklin, passing yards in a season, 1954 | -0.978 | -0.874 |
| Aaron Rodgers, touchdown passes in a season, 2020 | 0.670 | 0.566 |
| Trevon Diggs, interceptions in a season, 2021 | 2.183 | 2.286 |
| Shawne Merriman, sacks in a season, 2006 | -0.570 | -0.470 |
| La'Roi Glover, sacks in a season, 2000 | -0.333 | -0.235 |
| Carson Palmer, touchdown passes in a season, 2005 | -0.498 | -0.592 |
| Josh Gordon, receiving yards in a season, 2013 | -0.161 | -0.251 |
| Aaron Donald, sacks in a season, 2018 | 0.551 | 0.462 |
| Steve Smith, receiving yards in a season, 2005 | 0.323 | 0.408 |
| Jimmie Foxx, home runs in a season, 1932 | 0.530 | 0.614 |
| Adrian Peterson, rushing yards in a season, 2012 | 2.271 | 2.355 |
| Jack Butler, interceptions in a season, 1957 | 0.583 | 0.499 |
| Milt Davis, interceptions in a season, 1957 | 0.583 | 0.499 |
| Ernie Banks, home runs in a season, 1958 | -0.206 | -0.123 |
| Justin Simmons, interceptions in a season, 2022 | -1.716 | -1.634 |
| Minkah Fitzpatrick, interceptions in a season, 2022 | -1.716 | -1.634 |
| Chauncey Gardner-Johnson, interceptions in a season, 2022 | -1.716 | -1.634 |
| Tariq Woolen, interceptions in a season, 2022 | -1.716 | -1.634 |
| Neil Johnston, points per game, 1955 | -1.122 | -1.042 |
| Muhsin Muhammad, receiving yards in a season, 2004 | -1.452 | -1.371 |
| Saquon Barkley, rushing yards in a season, 2024 | 1.260 | 1.340 |
| Johnny Unitas, touchdown passes in a season, 1958 | -1.252 | -1.330 |
| Deshaun Watson, passing yards in a season, 2020 | -0.848 | -0.922 |
| Philip Rivers, passing yards in a season, 2010 | -0.700 | -0.774 |
| Will Sherman, interceptions in a season, 1955 | 0.343 | 0.416 |
| Pete Pihos, receiving yards in a season, 1953 | -0.537 | -0.609 |
| Jim Finks, passing yards in a season, 1955 | -0.736 | -0.666 |
| Rich Gannon, passing yards in a season, 2002 | 0.163 | 0.231 |
| LeSean McCoy, rushing yards in a season, 2013 | -0.292 | -0.225 |
| Johnny Unitas, passing yards in a season, 1960 | 0.444 | 0.507 |
| Reggie Wayne, receiving yards in a season, 2007 | -0.338 | -0.399 |
| Jack Christiansen, interceptions in a season, 1957 | 0.599 | 0.542 |
| Antonio Cromartie, interceptions in a season, 2007 | 0.712 | 0.768 |
| Drew Brees, passing yards in a season, 2008 | 2.447 | 2.391 |
| Chris Johnson, rushing yards in a season, 2009 | 1.404 | 1.460 |
| Jack Christiansen, interceptions in a season, 1953 | -0.195 | -0.140 |
| José Canseco, home runs in a season, 1988 | -0.016 | 0.039 |
| Ed Reed, interceptions in a season, 2010 | -0.264 | -0.316 |
| Daunte Culpepper, touchdown passes in a season, 2000 | -0.230 | -0.281 |
| Stefon Diggs, receiving yards in a season, 2020 | -1.286 | -1.336 |
| Lamar Jackson, touchdown passes in a season, 2019 | -0.741 | -0.789 |
| Ed Reed, interceptions in a season, 2004 | 0.361 | 0.317 |
| Robert Mathis, sacks in a season, 2013 | -0.282 | -0.239 |
| Tom Brady, touchdown passes in a season, 2010 | -0.566 | -0.523 |
| Michael Thomas, receiving yards in a season, 2019 | 0.217 | 0.175 |
| Ronde Barber, interceptions in a season, 2001 | 0.722 | 0.681 |
| Anthony Henry, interceptions in a season, 2001 | 0.722 | 0.681 |
| Maurice Jones-Drew, rushing yards in a season, 2011 | -0.698 | -0.658 |
| Richard Sherman, interceptions in a season, 2013 | -0.402 | -0.441 |
| Glover Quin, interceptions in a season, 2014 | -0.737 | -0.776 |
| Kyle Fuller, interceptions in a season, 2018 | -0.573 | -0.534 |
| Damontae Kazee, interceptions in a season, 2018 | -0.573 | -0.534 |
| Xavien Howard, interceptions in a season, 2018 | -0.548 | -0.511 |
| Champ Bailey, interceptions in a season, 2006 | 1.022 | 1.058 |
| Asante Samuel, interceptions in a season, 2006 | 1.022 | 1.058 |
| J.J. Watt, sacks in a season, 2015 | -0.332 | -0.297 |
| Jared Allen, sacks in a season, 2007 | -0.687 | -0.652 |
| Josh Jacobs, rushing yards in a season, 2022 | -0.612 | -0.579 |
| Simeon Rice, sacks in a season, 1999 | -0.477 | -0.510 |
| Brett Favre, touchdown passes in a season, 2003 | -0.498 | -0.531 |
| Adrian Peterson, rushing yards in a season, 2015 | -0.383 | -0.416 |
| T.Y. Hilton, receiving yards in a season, 2016 | -1.681 | -1.712 |
| Tommy Davis, runs batted in, 1962 | 1.009 | 1.039 |
| Drew Brees, touchdown passes in a season, 2009 | -0.741 | -0.771 |
| Drew Brees, passing yards in a season, 2015 | -1.329 | -1.357 |
| Patrick Mahomes, touchdown passes in a season, 2018 | 1.145 | 1.117 |
| Tom Brady, passing yards in a season, 2021 | 0.736 | 0.708 |
| Reggie Nelson, interceptions in a season, 2015 | 0.145 | 0.117 |
| Marcus Peters, interceptions in a season, 2015 | 0.145 | 0.117 |
| Vic Beasley, sacks in a season, 2016 | -0.741 | -0.714 |
| Julio Jones, receiving yards in a season, 2015 | 1.067 | 1.094 |
| Lindon Crow, interceptions in a season, 1956 | 1.098 | 1.124 |
| Arian Foster, rushing yards in a season, 2010 | -0.558 | -0.532 |
| J.J. Watt, sacks in a season, 2012 | 0.941 | 0.916 |
| Derrick Henry, rushing yards in a season, 2020 | 2.062 | 2.086 |
| Jared Allen, sacks in a season, 2011 | 1.554 | 1.530 |
| Jonathan Taylor, rushing yards in a season, 2021 | 0.615 | 0.638 |
| Christian McCaffrey, rushing yards in a season, 2023 | -0.578 | -0.601 |
| LaDainian Tomlinson, rushing yards in a season, 2007 | -1.139 | -1.162 |
| Kareem Hunt, rushing yards in a season, 2017 | -1.312 | -1.334 |
| Chandler Jones, sacks in a season, 2017 | -0.590 | -0.568 |
| Jameis Winston, passing yards in a season, 2019 | 0.240 | 0.218 |
| Tom Brady, passing yards in a season, 2017 | -0.913 | -0.934 |
| Antonio Brown, receiving yards in a season, 2014 | 0.237 | 0.257 |
| Tim Jennings, interceptions in a season, 2012 | 0.653 | 0.672 |
| Justin Houston, sacks in a season, 2014 | 1.363 | 1.345 |
| Julio Jones, receiving yards in a season, 2018 | -0.146 | -0.165 |
| Andrew Luck, touchdown passes in a season, 2014 | -0.472 | -0.453 |
| Bob Boyd, receiving yards in a season, 1954 | -0.337 | -0.320 |
| Ben Roethlisberger, passing yards in a season, 2018 | -0.044 | -0.060 |
| Bill Groman, receiving yards in a season, 1960 | 1.028 | 1.043 |
| Casey Hayward, interceptions in a season, 2016 | -0.343 | -0.358 |
| Russell Wilson, touchdown passes in a season, 2017 | -0.941 | -0.956 |
| DeMarco Murray, rushing yards in a season, 2014 | 1.239 | 1.254 |
| Matt Schaub, passing yards in a season, 2009 | -0.067 | -0.080 |
| Dick Lane, interceptions in a season, 1952 | 1.647 | 1.659 |
| Al Dorow, touchdown passes in a season, 1960 | -0.542 | -0.529 |
| Drew Brees, passing yards in a season, 2016 | 0.492 | 0.504 |
| Brandon Lloyd, receiving yards in a season, 2010 | -0.984 | -0.995 |
| Antonio Brown, receiving yards in a season, 2017 | -0.341 | -0.351 |
| Chad Johnson, receiving yards in a season, 2006 | -1.116 | -1.107 |
| Philip Rivers, touchdown passes in a season, 2008 | -0.320 | -0.327 |
| Aaron Rodgers, touchdown passes in a season, 2016 | -0.210 | -0.204 |
| Darren Sharper, interceptions in a season, 2009 | 0.105 | 0.111 |
| Charles Woodson, interceptions in a season, 2009 | 0.105 | 0.111 |
| Asante Samuel, interceptions in a season, 2009 | 0.105 | 0.111 |
| Jairus Byrd, interceptions in a season, 2009 | 0.105 | 0.111 |
