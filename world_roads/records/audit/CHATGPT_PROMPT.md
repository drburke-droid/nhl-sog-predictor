# Suggested prompt for the audit

Paste this with the zip attached (use a model with code execution so it can unzip and run the script).

---

I'm attaching a zip with a sports-statistics analysis that ranks the ten greatest individual seasons across
athletics, swimming, MLB, NBA, NHL and NFL. Each season-leading performance is measured against everyone
else's season bests in the 15 seasons before and after it. Please audit it independently and critically.

1. Unzip it and read `AUDIT.md` first. It describes the data, the method, the results and the known
   limitations.
2. Run `python audit_recompute.py` and confirm that it reproduces the published top 10 and the pipeline's
   scores. Then run it with `--window 7`, `--window 10`, `--window 20`, `--flat` and `--no-shift`.
3. Check the method:
   - Is a trend line through the window's season bests a sound way to handle a changing sport?
   - Is z (or the Gumbel "once in N seasons") a fair way to compare different kinds of statistics, such as
     per-game averages, season totals, hitting streaks and race times?
   - How much do one-sided windows at the ends of the data, the swimming suit term, and the number of
     categories per sport bias the list?
   - Are there better alternatives (for example robust regression, a different tail model, or
     per-category standardisation)? Show what they would change.
4. Check the data: spot-check season leaders in `inputs/season_lists.csv` against sources you know
   (Baseball-Reference, Basketball-Reference, Hockey-Reference, Pro-Football-Reference, World Athletics, World
   Aquatics). Look for wrong values, missing seasons, duplicate or misspelt names (which would break the
   "leave the performer out" step), wrong games-per-season values, and seasons that should or should not be
   marked as cut short.
5. Check the claims in `AUDIT.md` and on the page (`outputs/report.html`) against the data.
6. Report your findings ranked by how much they would change the top 10. For each one, give the evidence,
   whether it is an error or a judgment call, and a concrete fix.
