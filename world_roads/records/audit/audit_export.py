"""Export everything the greatest-seasons ranking uses into plain CSV, for an independent audit.

    python audit/audit_export.py        # from world_roads/records; writes audit/out/inputs/*.csv

Needs the local cache (the pipeline's downloads). The files it writes need nothing else: audit_recompute.py
rebuilds the ranking from them with numpy alone.

inputs/categories.csv     one row per category: what is measured, how, and the record
inputs/season_lists.csv   each season's top ten per category (raw value and performance), with the schedule
                          (games per team) for totals and whether the season is left out as cut short
inputs/record_lines.csv   every step of every record line drawn on the page
inputs/pipeline_era.csv   the pipeline's own score for every season-leading performance at windows 7, 10, 15
                          and 20, to compare against the recomputation
"""
import os, sys, csv, math

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.chdir(HERE)
import bigfour as B
import analysis as A

OUT = os.path.join(HERE, "audit", "out", "inputs")
TOP = 10


def perf_rule(c):
    if c["sport"] == "Athletics":
        ev = c["event"]
        return f"speed = {A.RUN_DIST[ev]} m / time (s)" if ev in A.RUN_DIST else "the mark itself (m)"
    if c["sport"] == "Swimming":
        return f"speed = {c['event'].split()[0]} m / time (s)"
    return "minus the ERA (lower is better)" if c["kind"] == "era" else "the stat itself"


def main():
    os.makedirs(OUT, exist_ok=True)
    cats = B.categories()
    B.HOLDERS.update({B.norm(nm): nm for c in cats if c["sport"] == "Swimming" for _, _, nm in c["record_line"]})
    with open(os.path.join(OUT, "categories.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["category", "sport", "title", "performance", "higher_raw_is_better", "is_total_scaled_by_schedule",
                    "first_season", "last_season", "performances_count_from", "short_seasons_left_out",
                    "level_shift_seasons", "cutoff_raw", "cutoff_performance", "today_seasons",
                    "record_raw", "record_holder", "record_season"])
        for c in cats:
            ys = sorted(c["seasons"])
            rv, rn, ry = c["record"]
            w.writerow([c["id"], c["sport"], c["title"], perf_rule(c), c["higher"], bool(c["counting"] and c["games"]),
                        ys[0], ys[-1], 1960 if c["sport"] == "Athletics" else ys[0],
                        " ".join(map(str, sorted(c["short"]))), " ".join(map(str, sorted(B.SHIFTS.get(c["sport"], ())))),
                        "" if c["cutoff"] is None else c["cutoff"],
                        "" if c["cutoff"] is None else repr(float(c["perf"](c["cutoff"]))),
                        f"{min(c['today'])}-{max(c['today'])}", rv, rn, ry])
    with open(os.path.join(OUT, "season_lists.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["category", "season", "rank", "name_raw", "name_display", "raw_value", "performance",
                    "games_per_team", "short_season"])
        for c in cats:
            for y in sorted(c["seasons"]):
                g = c["games"](y) if c["counting"] and c["games"] else ""
                for i, (n, v) in enumerate(c["seasons"][y][:TOP]):
                    w.writerow([c["id"], y, i + 1, n, B.disp(n, c["sport"]), v, repr(float(c["perf"](v))),
                                g, y in c["short"]])
    with open(os.path.join(OUT, "record_lines.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["category", "year_fraction", "raw_value", "holder"])
        for c in cats:
            for x, v, n in c["record_line"]:
                w.writerow([c["id"], round(x, 3), v, n])
    with open(os.path.join(OUT, "pipeline_era.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["window", "category", "season", "name_raw", "raw_value", "z", "T", "seasons_compared"])
        for win in sorted(set(B.ALT_WINDOWS) | {B.WINDOW}):
            for c in cats:
                for e in B.era_adjusted(c, win):
                    w.writerow([win, c["id"], e["year"], e["name"], e["value"], repr(float(e["z"])),
                                "" if e["T"] is None else (repr(float(e["T"])) if math.isfinite(e["T"]) else "inf"), e["n"]])
            print("window", win, "done", flush=True)
    for p in sorted(os.listdir(OUT)):
        print(p, sum(1 for _ in open(os.path.join(OUT, p), encoding="utf-8")) - 1, "rows")


if __name__ == "__main__":
    main()
