"""Run every model variant and write the follow-up outputs.

    python analysis/run_all.py            # from followup/; about 15-25 minutes (bootstrap B=300 x 2 models)
    python analysis/run_all.py --quick    # B=40, for a smoke test

Outputs (outputs/):
  rankings_sensitivity.csv   every athlete-season (one row per sport, athlete, season), the original score and
                             every labelled variant's z and rank (rank among athlete-seasons, and the page's
                             rank with each athlete's best season only)
  top10_by_variant.csv       each variant's top ten (one per athlete)
  bootstrap_<model>.csv      P(top ten) and 5-95% rank range per athlete, moving-block bootstrap
  spec_ensemble.csv          share of a pre-declared set of specifications that put each athlete in the top ten
  calibration_fits.csv       per-category GPD tail fits (raw and pooled shape)
  gamma_estimates.csv        environment elasticities, raw and partially pooled
"""
import csv, math, os, sys, json
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gsmodel as G                                       # noqa: E402
import context as X                                       # noqa: E402
import calibrate as K                                     # noqa: E402
import env_field_tests as T                               # noqa: E402

OUT = os.path.join(os.path.dirname(HERE), "outputs")
QUICK = "--quick" in sys.argv
NOBOOT = "--no-bootstrap" in sys.argv          # keep the existing bootstrap_*.csv (e.g. after a calibration-only change)


# ------------------------------------------------------------------ estimated environment elasticities
def pooled_gamma(D):
    """DerSimonian-Laird pooling of the field-median elasticities across league categories (shrinks noisy
    categories such as NBA rebounds toward the common value)"""
    el = T.elasticity(D)
    g = np.array([r["elasticity_field_median"] for r in el])
    se = np.array([max(r["se_field"], 1e-3) for r in el])
    w = 1 / se ** 2
    mu_fe = (w * g).sum() / w.sum()
    Q = float((w * (g - mu_fe) ** 2).sum())
    tau2 = max(0.0, (Q - (len(g) - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    ws = 1 / (se ** 2 + tau2)
    mu = float((ws * g).sum() / ws.sum())
    out, rows = {}, []
    for r, gi, si in zip(el, g, se):
        post = (gi / si ** 2 + mu / max(tau2, 1e-9)) / (1 / si ** 2 + 1 / max(tau2, 1e-9))
        out[r["category"]] = float(post)
        rows.append({**r, "gamma_pooled": round(float(post), 3), "pooled_mean": round(mu, 3), "tau2": round(tau2, 4)})
    return out, rows


def field_coeffs(D, ctx_env):
    """the sport-level expansion effect from field_size_test, in each category's comparison units:
    coefficient c such that the comparison value is moved by c*ln(Nref/N) (used as 'estimated field')"""
    fs = T.field_size_test(D)
    pooled = {r["sport"]: r for r in fs if r["category"] == "POOLED"}
    E = X.env_series()
    TM = X.teams()
    coef = {}
    for cat in X.ENV_MAP:
        m = D.meta[cat]
        vals = {}
        for y, lst in D.seasons[cat].items():
            if y in m["short"] or len(lst) < 2 or y not in E[cat]:
                continue
            e = lst[1]
            v = (E[cat][y] / (-e.perf)) if cat == "mlb_era" else (e.perf / E[cat][y])
            vals[y] = v
        ys = sorted(y for y in vals if y - 1 in vals)
        if len(ys) < 10:
            continue
        sd = float(np.std([vals[y] - vals[y - 1] for y in ys], ddof=1))
        sp = m["sport"]
        coef[cat] = {y: pooled[sp]["slope_sd_units_per_dlnN"] * sd for y in D.seasons[cat]} if sp in pooled else {}
    return coef


# ------------------------------------------------------------------ variants
def variants(D0, D2):
    gam, grows = pooled_gamma(D2)
    ratio = X.make(D2, env="ratio")
    power = X.make(D2, env="power", gamma=gam)
    V = []
    add = lambda key, label, data, spec, ctx=None, group="": V.append((key, label, data, spec, ctx, group))
    add("A0_original", "Original published (OLS +/-15, name-based exclusion, first-listed leader)", D0, G.Spec(), None, "baseline")
    add("A1_corrected", "Corrected data (stable IDs, logged repairs), every tied leader scored", D2, G.Spec(ties="all"), None, "data")
    for w in (7, 10, 20):
        add(f"A1_w{w}", f"Corrected, window +/-{w}", D2, G.Spec(ties="all", window=w), None, "window")
    add("A1_flat", "Corrected, flat window (no trend)", D2, G.Spec(ties="all", trend="flat"), None, "trend")
    add("A1_huber", "Corrected, robust (Huber) trend and MAD scale", D2, G.Spec(ties="all", trend="huber"), None, "trend")
    add("A2_main", "MAIN peer dominance: corrected, +/-15, quadratic trend (OLS near the data's ends) - CV choice within +/-15",
        D2, G.Spec(ties="all", trend="quad_safe"), None, "main")
    add("A2_w7", "Corrected, +/-7 quadratic (best cross-validated overall)", D2, G.Spec(ties="all", trend="quad_safe", window=7), None, "window")
    add("A2_twosided", "MAIN, ranking only seasons with at least 5 comparison seasons on each side", D2,
        G.Spec(ties="all", trend="quad_safe"), "TWOSIDED", "edges")
    add("A1_noshift", "Corrected, no swimming suit term", D2, G.Spec(ties="all", shift=False), None, "trend")
    add("B1_allseasons", "ALL-SEASONS rarity: only the tested season left out (athlete's other seasons kept)",
        D2, G.Spec(ties="all", exclusion="tested"), None, "estimand B")
    add("B2_allseasons_main", "All-seasons rarity with the main trend", D2, G.Spec(ties="all", exclusion="tested", trend="quad_safe"),
        None, "estimand B")
    add("E1_env_ratio", "Big four: value / league per-team-game rate (elasticity 1)", D2, G.Spec(ties="all", env="ratio"), ratio, "environment")
    add("E2_env_power", "Big four: value / league rate^gamma, gamma estimated and partially pooled", D2,
        G.Spec(ties="all", env="power"), power, "environment")
    add("E3_main_env", "MAIN + estimated environment (quadratic +/-15, gamma pooled)", D2,
        G.Spec(ties="all", env="power", trend="quad_safe"), power, "main+env")
    for s in (0.25, 0.5, 1.0):
        ctxf = X.make(D2, env="ratio", field=s)
        add(f"F_ratio_field{int(s * 100)}", f"Big four: ratio environment + proposed field offset at {int(s * 100)}% (beta*ln(Nref/N))",
            D2, G.Spec(ties="all", env="ratio", field=s), ctxf, "field (handoff model, extended to MLB/NFL)")
    ctxn = X.make(D2, env="ratio", field=1.0)
    add("F_ratio_field100_normal", "Big four: ratio environment + field offset with normal-tail growth", D2,
        G.Spec(ties="all", env="ratio", field=1.0, field_tail="normal"), ctxn, "field (handoff model, extended to MLB/NFL)")
    ctxe = X.make(D2, env="ratio")
    ctxe.beta = field_coeffs(D2, ctxe)
    add("F_ratio_field_estimated", "Big four: ratio environment + estimated expansion effect (sport-pooled)", D2,
        G.Spec(ties="all", env="ratio", field=1.0), ctxe, "field (estimated)")
    # individual sports: participation depth (value / median of the season's ranks 2-10)
    dctx = G.Context(env={c: {y: G.depth_level(l) for y, l in D2.seasons[c].items()} for c in D2.meta
                          if D2.meta[c]["sport"] in ("Athletics", "Swimming")},
                     env_cats=[c for c in D2.meta if D2.meta[c]["sport"] in ("Athletics", "Swimming")])
    Dd = G.Data("v2")                     # 2020's pandemic-thinned fields make a depth ratio meaningless
    for c in Dd.meta:
        if Dd.meta[c]["sport"] in ("Athletics", "Swimming"):
            Dd.meta[c]["short"] = Dd.meta[c]["short"] | {2020}
    add("D1_depth", "Athletics and swimming: value / median of the season's ranks 2-10 (participation depth; 2020 left out)", Dd,
        G.Spec(ties="all", env="depth", trend="quad_safe"), dctx, "depth/technology (individual sports)")
    # technology: a separate level for running events from 2020 (plated track spikes; World Athletics shoe rules)
    D3 = G.Data("v2")
    run = ("100m", "200m", "400m", "800m", "1500m", "110mH", "100mH", "400mH")
    for c in D3.meta:
        if c.startswith("ath_") and c.split("_", 2)[2] in run:
            D3.meta[c]["shift"] = set(range(2020, 2027))
    add("T1_spikes", "Athletics running events: separate level for 2020-26 (spike technology), main trend", D3,
        G.Spec(ties="all", trend="quad_safe"), None, "depth/technology (individual sports)")
    return V, gam, grows


def ranks(recs):
    allr = G.athlete_seasons(recs)
    page = G.by_athlete(allr)
    pr = {(r["sport"], r["id"]): i + 1 for i, r in enumerate(page)}
    out = {}
    for i, r in enumerate(allr):
        k = (r["sport"], r["id"], r["year"])
        best = pr.get((r["sport"], r["id"])) if page[pr[(r["sport"], r["id"])] - 1]["year"] == r["year"] else None
        out[k] = {"z": r["z"], "rank_all": i + 1, "rank_page": best, "cat": r["cat"], "name": r["name"], "raw": r["raw"]}
    return out, page


def main():
    os.makedirs(OUT, exist_ok=True)
    D0, D2 = G.Data("original"), G.Data("v2")
    V, gam, grows = variants(D0, D2)
    with open(os.path.join(OUT, "gamma_estimates.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(grows[0].keys()))
        w.writeheader()
        w.writerows(grows)
    res, pages, recs_by = {}, {}, {}
    for key, label, data, spec, ctx, group in V:
        if ctx == "TWOSIDED":
            recs = [r for r in G.score(data, spec) if not r["edge"]]
        else:
            recs = G.score(data, spec, ctx=ctx)
        recs_by[key] = recs
        res[key], pages[key] = ranks(recs)
        print(f"{key:28} {len(recs):5} scores; top: " + ", ".join(f"{r['name']} {r['year']} {r['z']:.2f}" for r in pages[key][:5]), flush=True)
    # simple category standardisation of the main model: (z - median) / (1.4826 MAD) over each category's leaders
    by = defaultdict(list)
    for r in recs_by["A2_main"]:
        by[r["cat"]].append(r["z"])
    med = {c: float(np.median(v)) for c, v in by.items()}
    mad = {c: 1.4826 * float(np.median(np.abs(np.array(v) - med[c]))) or 1.0 for c, v in by.items()}
    std = [{**r, "z": (r["z"] - med[r["cat"]]) / mad[r["cat"]]} for r in recs_by["A2_main"]]
    res["A2_main_catstd"], pages["A2_main_catstd"] = ranks(std)
    V.append(("A2_main_catstd", "MAIN standardised within category: (z - category median) / (1.4826 x MAD)", None, None, None, "calibration"))
    # calibration layer on the main model (and the original, for comparison)
    cal_rows = []
    for key in ("A0_original", "A2_main", "E3_main_env"):
        cal, summ = K.calibrate(recs_by[key])
        ntot = len(cal)
        with open(os.path.join(OUT, f"calibrated_{key}.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["athlete", "sport", "season", "category", "value", "z", "p_category_leader",
                        "minus_log10_p", "expected_count_in_all_category_seasons", "one_sided_window"])
            for r in sorted(cal, key=lambda r: -r["cal"])[:300]:
                w.writerow([r["name"], r["sport"], r["year"], r["cat"], r["raw"], round(r["z"], 3), f"{r['p_cal']:.3g}",
                            round(r["cal"], 3), f"{r['p_cal'] * ntot:.3g}", r["edge"]])
        res[key + "_cal"], pages[key + "_cal"] = ranks([{**r, "z": r["cal"]} for r in cal])
        V.append((key + "_cal", f"{key} tail-calibrated: -log10 P(a season leader of the category exceeds z), GPD with pooled shape",
                  None, None, None, "calibration"))
        for c, ftt in summ["fits"].items():
            cal_rows.append({"model": key, "category": c, "threshold_z": round(ftt["u"], 3), "leaders": ftt["n"],
                             "exceedances": ftt["k"], "mean_excess_scale": round(ftt["scale"], 3),
                             "scale_pooled": round(ftt["scale_pooled"], 3), "gpd_xi_bounded": round(ftt["xi_hat"], 3),
                             "gpd_xi_se": round(ftt["se"], 3), "gpd_xi_pooled": round(ftt["xi_pooled"], 3),
                             "gpd_xi_pooled_mean": round(summ["xi_pooled_mean"], 3), "gpd_xi_tau2": round(summ["tau2"], 4)})
    with open(os.path.join(OUT, "calibration_fits.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cal_rows[0].keys()))
        w.writeheader()
        w.writerows(cal_rows)
    # sensitivity table: every athlete-season in any variant
    keys = set()
    for k in res:
        keys |= set(res[k])
    base = res["A0_original"]
    disp = {}
    for k in res:
        for kk, v in res[k].items():
            disp.setdefault(kk, v)
    order = sorted(keys, key=lambda k: -max((res[v][k]["z"] for v in res if k in res[v] and not v.endswith("_cal")), default=-9))
    cols = ["sport", "athlete", "season", "category", "value"]
    for key in res:
        cols += [f"{key}_z", f"{key}_rank_all", f"{key}_rank_page"]
    with open(os.path.join(OUT, "rankings_sensitivity.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for k in order:
            d = disp[k]
            row = [k[0], d["name"], k[2], d["cat"], d["raw"]]
            for key in res:
                v = res[key].get(k)
                row += ([round(v["z"], 4), v["rank_all"], v["rank_page"] or ""] if v else ["", "", ""])
            w.writerow(row)
    with open(os.path.join(OUT, "variants.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "group", "label"])
        for key, label, *_rest in V:
            w.writerow([key, _rest[-1], label])
    with open(os.path.join(OUT, "top10_by_variant.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variant", "rank", "athlete", "sport", "season", "category", "value", "score"])
        for key in res:
            for i, r in enumerate(pages[key][:10]):
                w.writerow([key, i + 1, r["name"], r["sport"], r["year"], r["cat"], r["raw"], round(r["z"], 3)])
    # specification ensemble (pre-declared): windows x trends x environment, corrected data, estimand A
    ens = defaultdict(int)
    n_spec = 0
    power = X.make(D2, env="power", gamma=gam)
    for wdw in (7, 10, 15, 20):
        for tr in ("ols", "quad_safe"):
            for env in ("none", "power"):
                recs = G.score(D2, G.Spec(ties="all", window=wdw, trend=tr, env=env), ctx=power if env == "power" else None)
                page = G.by_athlete(G.athlete_seasons(recs))
                n_spec += 1
                for r in page[:10]:
                    ens[(r["sport"], r["id"], r["name"])] += 1
    with open(os.path.join(OUT, "spec_ensemble.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["athlete", "sport", "specs_in_top10", "specs", "share"])
        for (sp, i, nm), c in sorted(ens.items(), key=lambda kv: -kv[1]):
            w.writerow([nm, sp, c, n_spec, round(c / n_spec, 3)])
    # bootstrap of the main model and of the corrected original-form model
    B = 40 if QUICK else 300
    for key, spec, ctx in (() if NOBOOT else (("A2_main", G.Spec(ties="all", trend="quad_safe"), None),
                           ("A1_corrected", G.Spec(ties="all"), None),
                           ("E3_main_env", G.Spec(ties="all", env="power", trend="quad_safe"), power))):
        hits, rk, zs, nb = K.bootstrap(D2, spec, ctx=ctx, B=B)
        names = {}
        for r in pages[key][:200]:
            names[(r["sport"], r["id"])] = (r["name"], r["year"], r["cat"], r["z"])
        rows = []
        for k, (nm, yr, cat, z) in names.items():
            rr = rk.get(k, [])
            full = rr + [201] * (nb - len(rr))
            rows.append({"athlete": nm, "sport": k[0], "season": yr, "category": cat, "z_point": round(z, 3),
                         "p_top10": round(hits.get(k, 0) / nb, 3),
                         "rank_p05": int(np.percentile(full, 5)), "rank_median": int(np.percentile(full, 50)),
                         "rank_p95": int(np.percentile(full, 95)) if np.percentile(full, 95) <= 200 else ">200",
                         "z_p05": round(float(np.percentile(zs[k], 5)), 2) if zs.get(k) else "",
                         "z_p95": round(float(np.percentile(zs[k], 95)), 2) if zs.get(k) else ""})
        rows.sort(key=lambda r: -r["z_point"])
        with open(os.path.join(OUT, f"bootstrap_{key}.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("bootstrap", key, "done", flush=True)
    if not NOBOOT:
        json.dump({"B": B, "block_length": 3, "variants": [v[0] for v in V]}, open(os.path.join(OUT, "run_meta.json"), "w"))


if __name__ == "__main__":
    main()
