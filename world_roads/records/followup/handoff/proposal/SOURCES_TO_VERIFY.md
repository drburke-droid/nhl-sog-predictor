# Source audit and retrieval targets

These links document where the earlier experiment drew external-era summaries and where Claude should verify or replace them with more authoritative series. No scraped source cache is included.

- Original inputs: see `original_project/AUDIT.md` and `original_project/docs/ACCURACY.md`. Source pipeline output from 2026 snapshot; primary raw cached pages omitted.
- NHL historical scoring: Hockey-Reference https://www.hockey-reference.com/leagues/stats.html ; external tabulation https://github.com/johngunerli/gretzky-mcdavid-era/blob/master/league_avgs_clean.csv . Verify season endings, 2025-26, and handling of 2004-05 cancellation.
- NBA points/team-game, attempts, free throw attempts: Basketball-Reference https://www.basketball-reference.com/leagues/NBA_stats_per_game.html ; historical tabulation https://github.com/YonahAFeld/NBA-League-Dashboard/blob/main/data/Avg%20league%20stats%20per%20game.csv . Verify 2025-26 source row separately; identify estimated vs actual pace.
- NBA historical franchise changes: https://en.wikipedia.org/wiki/Expansion_of_the_NBA ; prefer NBA historical reference pages year-by-year.
- NHL expansion/merger franchise histories: https://www.nhl.com/news/nhl-expansion-history-281005106 ; prefer per-season league tables for actual active team count.
- NBA 1961-62 individual/team production: https://www.basketball-reference.com/leagues/NBA_1962_per_game.html , https://www.basketball-reference.com/teams/PHW/1962.html . Distinguish team pace from league pace and playing minutes.
- NHL adjusted stats methodology (comparison only, do not assume adopted): https://www.hockey-reference.com/about/adjusted_stats.html .
- Baseball-Reference https://www.baseball-reference.com/ ; Pro Football Reference https://www.pro-football-reference.com/ ; World Athletics https://worldathletics.org/ ; World Aquatics https://www.worldaquatics.com/ .

Ask whether data permit full qualified player-season distributions and denominators for each category. Document historical exclusions and changes in membership, eligibility, games played, qualification rules, player role, scorekeeping and recording, and all source timestamps.
