"""Gather every result into one file for the findings page, and build the page.

    python report_data.py      -> data/report.json, report.html (report_page.html with the data inlined)
"""
import os, json, math, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")


def main():
    us = json.load(open(os.path.join(D, "us_stats.json")))
    baer = json.load(open(os.path.join(D, "baer_stats.json")))
    stems = np.load(os.path.join(D, "baer_stems_rows.npy"))
    named = np.load(os.path.join(D, "baer_named.npy"))
    sy = json.load(open(os.path.join(HERE, "..", "rivers", "data", "all_rivers.json")))
    out = {}
    # 1. junction angles against aridity, at each measuring distance
    out["junctions"] = {}
    for k, dist in (("a250", 250), ("a500", 500), ("a1000", 1000)):
        j = us["junctions"][k]
        out["junctions"][str(dist)] = {
            "n": j["n"], "spearman": round(j["spearman"], 3),
            "bins": [{"ai": [b["ai"][0], b["ai"][1]], "mid": math.sqrt(b["ai"][0] * b["ai"][1]), "n": b["n"],
                      "mean": round(b["mean"], 1), "lo": round(b["ci"][0], 1), "hi": round(b["ci"][1], 1)} for b in j["bins"]],
            "controls": {name: {"dry": round(c["dry_mean"], 1), "wet": round(c["wet_mean"], 1), "n_dry": c["n_dry"], "n_wet": c["n_wet"]}
                         for name, c in j["controls"].items()}}
    # 2. river length over straight-line distance, by size: US main stems and the world's big systems
    p = us["paths"]
    out["us_stems"] = {"n": p["n"], "km": round(p["km"]), "ratio_of_totals": round(p["ratio_of_totals"], 3), "median": round(p["median"], 3),
                       "near_pi": p["near_pi"], "above_pi": p["above_pi"],
                       "by_length": [{"len": r["len_km"], "n": r["n"], "median": round(r["median"], 3), "ratio": round(r["ratio_of_totals"], 3),
                                      "near_pi": r["near_pi"]} for r in p["by_length"]],
                       "curve": [{"straight_km": round(10 ** r[0], 2), "ratio": round(10 ** r[1] / 10 ** r[0], 3), "n": r[2]} for r in p["len_vs_straight"]]}
    L = np.array([r["length_km"] for r in sy]); S = np.array([r["straight_km"] for r in sy])
    rows = []
    for lo, hi in ((100, 200), (200, 500), (500, 1000), (1000, 2000), (2000, 8000)):
        m = (L >= lo) & (L < hi)
        rows.append({"len": [lo, hi], "n": int(m.sum()), "median": round(float(np.median(L[m] / S[m])), 3),
                     "ratio": round(float(L[m].sum() / S[m].sum()), 3), "near_pi": float(np.mean(np.abs(L[m] / S[m] - math.pi) < 0.2))})
    out["world_systems"] = {"n": len(L), "ratio_of_totals": round(float(L.sum() / S.sum()), 3), "median": round(float(np.median(L / S)), 3),
                            "near_pi": float(np.mean(np.abs(L / S - math.pi) < 0.2)), "by_length": rows}
    # 3. Hack's law
    h = us["hack"]
    out["hack"] = {"n": h["n"], "h": round(h["h"], 3), "c": round(h["c"], 2),
                   "local": [{"area": [10 ** r["log10_area"][0], 10 ** r["log10_area"][1]], "h": round(r["h"], 3)} for r in h["local"]],
                   "curve": [{"area": round(10 ** r[0], 3), "len": round(10 ** r[1], 3), "p10": round(10 ** r[2], 3), "p90": round(10 ** r[3], 3), "n": r[4]}
                             for r in h["curve"]]}
    # 4. Baer's law: every OSM line by latitude, and the big rivers
    out["baer_lines"] = {"lines": baer["lines"], "km": round(baer["km"]),
                         "bands": {str(sc): [{"band": b["band"], "share": round(b["share_left"] * 100, 3), "lo": round(b["ci"][0] * 100, 3),
                                              "hi": round(b["ci"][1] * 100, 3), "km": round(b["km"]), "lines": b["lines"]} for b in baer[f"bands_{sc}"]]
                                   for sc in (250, 1000)},
                         "north": {sc: baer[f"north_{sc}"] for sc in (250, 1000)}, "south": {sc: baer[f"south_{sc}"] for sc in (250, 1000)}}
    lat, lon = stems[:, 0], stems[:, 10]
    rng = np.random.default_rng(11)

    def share(m, col=6):
        l, r = stems[m, col], stems[m, col + 1]
        idx = np.nonzero(m)[0]
        bs = []
        for _ in range(1000):
            k = rng.choice(idx, len(idx))
            bs.append(stems[k, col].sum() / (stems[k, col].sum() + stems[k, col + 1].sum()))
        return {"share": round(float(l.sum() / (l.sum() + r.sum())) * 100, 2), "lo": round(float(np.percentile(bs, 2.5)) * 100, 2),
                "hi": round(float(np.percentile(bs, 97.5)) * 100, 2), "rivers": int(m.sum()), "mostly_left": round(float(np.mean(l > r)) * 100, 1)}
    groups = [("Southern hemisphere", lat <= -5), ("Tropics north (5–25°N)", (lat >= 5) & (lat < 25)), ("25–45°N", (lat >= 25) & (lat < 45)),
              ("45–60°N", (lat >= 45) & (lat < 60)), ("60–80°N", lat >= 60)]
    regions = [("North America", (lat >= 5) & (lon < -50)), ("Europe", (lat >= 5) & (lon >= -25) & (lon < 40)),
               ("Russia and Siberia", (lon >= 40) & (lat >= 45)), ("North, without Russia", (lat >= 5) & ~((lon >= 40) & (lat >= 45))),
               ("All northern", lat >= 5), ("All southern", lat <= -5)]
    out["baer_big"] = {"rivers": len(stems), "scale_km": 5, "by_latitude": [dict(name=n, **share(m)) for n, m in groups],
                       "by_region": [dict(name=n, **share(m)) for n, m in regions]}
    # 5. flow direction of the world's systems
    src = np.array([r["source"] for r in sy]); mou = np.array([r["mouth"] for r in sy])
    az = (np.degrees(np.arctan2((mou[:, 0] - src[:, 0]) * np.cos(np.radians((mou[:, 1] + src[:, 1]) / 2)), mou[:, 1] - src[:, 1])) + 360) % 360
    out["flow_direction"] = {d: round(float(np.mean(((az >= a0) | (az < a1)) if a0 > a1 else ((az >= a0) & (az < a1)))) * 100, 1)
                             for d, a0, a1 in (("north", 315, 45), ("east", 45, 135), ("south", 135, 225), ("west", 225, 315))}
    # totals
    nfl = 0
    for f in glob.glob(os.path.join(D, "nhd", "out", "*.npz")):
        nfl += len(np.load(f)["hack"])
    out["totals"] = {"us_flowlines": nfl, "us_units": len(glob.glob(os.path.join(D, "nhd", "out", "*.npz")))}
    json.dump(out, open(os.path.join(D, "report.json"), "w"), indent=1)
    page = open(os.path.join(HERE, "report_page.html")).read()
    data = json.dumps(out, separators=(",", ":")).replace("</", "<\\/")
    open(os.path.join(HERE, "report.html"), "w").write(page.replace("__DATA__", data))
    print(json.dumps({k: (v if k in ("totals", "flow_direction") else "...") for k, v in out.items()}, indent=1))
    print(json.dumps(out["baer_big"], indent=0)[:1500])


if __name__ == "__main__":
    main()
