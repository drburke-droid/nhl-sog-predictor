"""Run every model variant and write the follow-up outputs.

    python analysis/run_all.py            # from followup/; about 60-80 minutes (B=1000 bootstraps, model-averaged B=800)
    python analysis/run_all.py --quick    # B=40, for a smoke test

Outputs (outputs/):
  rankings_sensitivity.csv   every athlete-season (one row per sport, athlete, season), the original score and
                             every labelled variant's z and rank (rank among athlete-seasons, and the page's
                             rank with each athlete's best season only)
  top10_by_variant.csv       each variant's top ten (one per athlete)
  bootstrap_<model>.csv      top-ten FREQUENCY among block-bootstrap refits (conditional on the model), MC SE,
                             5-95% rank range; bootstrap_model_averaged.csv also draws the specification
  calibrated_<model>.csv     pooled-exponential tail p, its 90% interval, E = p x N scored records, Poisson P(>=1)
  calibration_sensitivity.csv  threshold (q) x family (exponential, GPD) for each model's top seasons
  calibration_tail_coverage.csv  leave-one-out check of each tail family
  empirical_null_top30.csv   model-light reference from withheld comparison seasons
  identity_crosswalk_log.csv original name keys that collapse to one stable ID, and records removed in v2
  spec_ensemble.csv          share of a pre-declared set of specifications that put each athlete in the top ten
  calibration_fits.csv       per-category exponential tail scales (raw, pooled) and GPD shape diagnostics
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


# ------------------------------------------------------------------ identity crosswalk (review round 2, issue 1)
REMOVED_IN_V2 = {("nba_apg", 1947), ("nba_apg", 1948), ("ath_M_PV", 2027)}   # C7 definition break, C2 quarantine
INCOMPLETE = {("Swimming", 2026): "calendar year to 2 Nov 2026; the long-course season may not be complete"}


def crosswalk():
    """(category, season, original name key) -> stable v2 athlete_id, via the original rank of each row"""
    orig = {(r["category"], int(r["season"]), int(r["rank"])): G.norm(r["name_raw"])
            for r in csv.DictReader(open(os.path.join(G.DATA, "original", "season_lists.csv"), encoding="utf-8"))}
    xw = {}
    for r in csv.DictReader(open(os.path.join(G.DATA, "v2", "season_lists_v2.csv"), encoding="utf-8")):
        if r["orig_rank"]:
            k = (r["category"], int(r["season"]), int(r["orig_rank"]))
            if k in orig:
                xw[(r["category"], int(r["season"]), orig[k])] = r["athlete_id"]
    return xw


def ranks(recs, xwalk=None):
    """-> {(sport, stable id, season): entry}, page list, collisions. Ranks are computed with the variant's own
    identities (so the original variant reproduces the published ranking); only the join key is mapped."""
    allr = G.athlete_seasons(recs)
    page = G.by_athlete(allr)
    pr = {(r["sport"], r["id"]): i + 1 for i, r in enumerate(page)}
    out, collisions = {}, []
    for i, r in enumerate(allr):
        sid, status = r["id"], "v2"
        if xwalk is not None:
            if (r["cat"], r["year"], r["id"]) in xwalk:
                sid, status = xwalk[(r["cat"], r["year"], r["id"])], "matched"
            else:
                sid, status = "legacy:" + r["id"], "unmapped"
            if (r["cat"], r["year"]) in REMOVED_IN_V2:
                status = "removed_in_v2"
        k = (r["sport"], sid, r["year"])
        best = pr[(r["sport"], r["id"])] if page[pr[(r["sport"], r["id"])] - 1]["year"] == r["year"] else None
        e = {"z": r["z"], "rank_all": i + 1, "rank_page": best, "cat": r["cat"], "name": r["name"], "raw": r["raw"],
             "status": status, "edge": r.get("edge", False), "orig_id": r["id"]}
        if k in out:
            collisions.append((k, out[k]["name"], out[k]["z"], r["name"], r["z"]))
            out[k]["status"] = "collapsed"
            if out[k]["z"] >= r["z"]:
                continue
            e["status"] = "collapsed"
        out[k] = e
    return out, page, collisions


def write_csv(name, rows):
    with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def poisson_any(E):
    return 1 - math.exp(-E)


def main():
    os.makedirs(OUT, exist_ok=True)
    D0, D2 = G.Data("original"), G.Data("v2")
    V, gam, grows = variants(D0, D2)
    write_csv("gamma_estimates.csv", grows)
    XW = crosswalk()
    res, pages, recs_by, collide = {}, {}, {}, {}
    for key, label, data, spec, ctx, group in V:
        if ctx == "TWOSIDED":
            recs = [r for r in G.score(data, spec) if not r["edge"]]
        else:
            recs = G.score(data, spec, ctx=ctx)
        recs_by[key] = recs
        res[key], pages[key], collide[key] = ranks(recs, XW if data is D0 else None)
        print(f"{key:28} {len(recs):5} scores; top: " + ", ".join(f"{r['name']} {r['year']} {r['z']:.2f}" for r in pages[key][:5]), flush=True)
    # simple category standardisation of the main model: (z - median) / (1.4826 MAD) over each category's leaders
    by = defaultdict(list)
    for r in recs_by["A2_main"]:
        by[r["cat"]].append(r["z"])
    med = {c: float(np.median(v)) for c, v in by.items()}
    mad = {c: 1.4826 * float(np.median(np.abs(np.array(v) - med[c]))) or 1.0 for c, v in by.items()}
    std = [{**r, "z": (r["z"] - med[r["cat"]]) / mad[r["cat"]]} for r in recs_by["A2_main"]]
    res["A2_main_catstd"], pages["A2_main_catstd"], collide["A2_main_catstd"] = ranks(std)
    V.append(("A2_main_catstd", "MAIN standardised within category: (z - category median) / (1.4826 x MAD)", None, None, None, "calibration"))
    # calibration (review round 2, issue 2): pooled-EXPONENTIAL tail is the calibrated ranking; GPD is a sensitivity
    cal_rows, sens_rows, cov_rows = [], [], []
    for key in ("A0_original", "A2_main", "E3_main_env"):
        cal, summ = K.calibrate(recs_by[key], family="exponential")
        N = summ["n_scored"]
        top = sorted(cal, key=lambda r: -r["cal"])
        rows = []
        for i, r in enumerate(top[:300]):
            E = r["p_cal"] * N
            lo = hi = ""
            if i < 30 and not QUICK:
                lo, hi = K.p_interval(recs_by[key], r["cat"], r["z"], B=300)
            rows.append({"athlete": r["name"], "sport": r["sport"], "season": r["year"], "category": r["cat"], "value": r["raw"],
                         "z": round(r["z"], 3), "p_exponential_tail": f"{r['p_cal']:.3g}",
                         "p_90pct_interval": f"{lo:.2g}-{hi:.2g}" if lo != "" else "",
                         "minus_log10_p": round(r["cal"], 3), "scored_records_N": N,
                         "expected_count_E_equals_p_times_N": f"{E:.3g}",
                         "poisson_P_at_least_one": f"{poisson_any(E):.2f}",
                         "one_sided_window": r["edge"],
                         "incomplete_season": INCOMPLETE.get((r["sport"], r["year"]), "")})
        write_csv(f"calibrated_{key}.csv", rows)
        res[key + "_cal"], pages[key + "_cal"], collide[key + "_cal"] = ranks([{**r, "z": r["cal"]} for r in cal],
                                                                               XW if key == "A0_original" else None)
        V.append((key + "_cal", f"{key} calibrated with a pooled-EXPONENTIAL category tail (sensitivity model): "
                                f"-log10 P(a season leader of the category exceeds z); GPD shapes are diagnostics only",
                  None, None, None, "calibration (exponential tail)"))
        for c, ftt in summ["fits"].items():
            cal_rows.append({"model": key, "category": c, "threshold_z": round(ftt["u"], 3), "leaders": ftt["n"],
                             "exceedances": ftt["k"], "exp_mean_excess_scale": round(ftt["scale"], 3),
                             "exp_scale_pooled": round(ftt["scale_pooled"], 3), "gpd_xi_bounded_diagnostic": round(ftt["xi_hat"], 3),
                             "gpd_xi_se": round(ftt["se"], 3), "gpd_xi_pooled_diagnostic": round(ftt["xi_pooled"], 3),
                             "gpd_xi_pooled_mean": round(summ["xi_pooled_mean"], 3), "gpd_xi_tau2": round(summ["tau2"], 4)})
        # threshold x family sensitivity for the top seasons of this model
        lead = [(r["cat"], r["year"], r["name"]) for r in top[:12]]
        for q in (0.60, 0.65, 0.70, 0.75, 0.80, 0.85):
            for fam in ("exponential", "gpd"):
                c2, s2 = K.calibrate(recs_by[key], q=q, family=fam)
                d = {(r["cat"], r["year"], r["name"]): r for r in c2}
                for k in lead:
                    r = d[k]
                    E = r["p_cal"] * s2["n_scored"]
                    sens_rows.append({"model": key, "athlete": k[2], "season": k[1], "category": k[0], "q": q, "family": fam,
                                      "p": "beyond fitted support" if r["beyond_support"] else f"{r['p_cal']:.3g}",
                                      "expected_count": "" if r["beyond_support"] else f"{E:.3g}",
                                      "poisson_P_at_least_one": "" if r["beyond_support"] else f"{poisson_any(E):.2f}"})
        for fam in ("exponential", "gpd"):
            cov_rows += [{"model": key, **x} for x in K.tail_coverage(recs_by[key], family=fam)]
    write_csv("calibration_fits.csv", cal_rows)
    write_csv("calibration_sensitivity.csv", sens_rows)
    write_csv("calibration_tail_coverage.csv", cov_rows)
    # empirical null (model-light reference) for the main model's top seasons
    nul = K.empirical_null(D2, G.Spec(ties="all", trend="quad_safe"))
    distinct = len({(c, s) for c, s, _, _ in nul})
    nrows = []
    for r in pages["A2_main"][:30]:
        ex = {(c, s) for c, s, pid, e in nul if e >= r["z"] and pid != r["id"]}
        nrows.append({"athlete": r["name"], "season": r["year"], "category": r["cat"], "z": round(r["z"], 3),
                      "null_category_seasons_at_or_above_z_other_athletes": len(ex),
                      "distinct_null_category_seasons": distinct,
                      "empirical_share": f"{len(ex) / distinct:.2g}",
                      "examples": "; ".join(f"{c} {s}" for c, s in sorted(ex)[:4])})
    write_csv("empirical_null_top30.csv", nrows)
    # sensitivity table: every athlete-season in any variant, keyed by STABLE identity
    keys = set()
    for k in res:
        keys |= set(res[k])
    disp, status = {}, {}
    for k in res:
        for kk, v in res[k].items():
            disp.setdefault(kk, v)
            if v["status"] in ("collapsed", "removed_in_v2", "unmapped"):
                status[kk] = v["status"]
    for kk in keys:
        if kk not in status:
            inA0 = kk in res["A0_original"]
            inA1 = kk in res["A1_corrected"]
            status[kk] = "matched" if inA0 and inA1 else ("original_only" if inA0 else "corrected_only")
    order = sorted(keys, key=lambda k: (-max((res[v][k]["z"] for v in res if k in res[v] and not v.endswith("_cal")), default=-9),
                                        k[0], str(k[1]), k[2]))
    main_edge = {k: v["edge"] for k, v in res["A2_main"].items()}
    cols = ["sport", "athlete", "season", "athlete_id", "id_status", "category", "value", "one_sided_window_main",
            "incomplete_season"]
    for key in res:
        cols += [f"{key}_z", f"{key}_rank_all", f"{key}_rank_page"]
    with open(os.path.join(OUT, "rankings_sensitivity.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for k in order:
            d = disp[k]
            row = [k[0], d["name"], k[2], k[1], status[k], d["cat"], d["raw"], main_edge.get(k, ""),
                   INCOMPLETE.get((k[0], k[2]), "")]
            for key in res:
                v = res[key].get(k)
                row += ([round(v["z"], 4), v["rank_all"], v["rank_page"] or ""] if v else ["", "", ""])
            w.writerow(row)
    xrows = [{"variant": key, "sport": k[0], "season": k[2], "stable_id": k[1], "name_a": a, "z_a": round(za, 4),
              "name_b": b, "z_b": round(zb, 4)} for key in ("A0_original",) for (k, a, za, b, zb) in collide[key]]
    xrows += [{"variant": "A0_original", "sport": k[0], "season": k[2], "stable_id": k[1], "name_a": v["name"],
               "z_a": round(v["z"], 4), "name_b": "(removed in v2)", "z_b": ""}
              for k, v in res["A0_original"].items() if v["status"] in ("removed_in_v2", "unmapped")]
    write_csv("identity_crosswalk_log.csv", xrows)
    both = sum(1 for k in keys if k in res["A0_original"] and k in res["A1_corrected"])
    print(f"join: {len(keys)} athlete-seasons; {both} with original and corrected scores; "
          f"{len(collide['A0_original'])} collapses; {sum(1 for v in res['A0_original'].values() if v['status'] == 'removed_in_v2')} removed", flush=True)
    assert both > 4000, "original and corrected scores are not aligned"
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
    SPECS = []
    for wdw in (7, 10, 15, 20):
        for tr in ("ols", "quad_safe"):
            for env in ("none", "power"):
                sp_ = G.Spec(ties="all", window=wdw, trend=tr, env=env)
                cx = power if env == "power" else None
                SPECS.append((sp_, cx))
                page = G.by_athlete(G.athlete_seasons(G.score(D2, sp_, ctx=cx)))
                n_spec += 1
                for r in page[:10]:
                    ens[(r["sport"], r["id"], r["name"])] += 1
    with open(os.path.join(OUT, "spec_ensemble.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["athlete", "sport", "specs_in_top10", "specs", "share"])
        for (sp, i, nm), c in sorted(ens.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][2])):
            w.writerow([nm, sp, c, n_spec, round(c / n_spec, 3)])
    # bootstrap (review round 2, item 4): a conditional frequency, with Monte Carlo SE
    B = 40 if QUICK else 1000
    def boot_rows(key, hits, rk, zs, nb):
        rows = []
        for r in pages[key][:200]:
            k = (r["sport"], r["id"])
            rr = rk.get(k, [])
            full = rr + [201] * (nb - len(rr))
            f_ = hits.get(k, 0) / nb
            p95 = np.percentile(full, 95)
            rows.append({"athlete": r["name"], "sport": k[0], "season": r["year"], "category": r["cat"], "z_point": round(r["z"], 3),
                         "top10_frequency_in_refits": round(f_, 3), "monte_carlo_se": round(math.sqrt(f_ * (1 - f_) / nb), 3),
                         "refits": nb, "rank_p05": int(np.percentile(full, 5)), "rank_median": int(np.percentile(full, 50)),
                         "rank_p95": int(p95) if p95 <= 200 else ">200",
                         "z_p05": round(float(np.percentile(zs[k], 5)), 2) if zs and zs.get(k) else "",
                         "z_p95": round(float(np.percentile(zs[k], 95)), 2) if zs and zs.get(k) else ""})
        return rows
    if not NOBOOT:
        for key, spec, ctx in (("A2_main", G.Spec(ties="all", trend="quad_safe"), None),
                               ("E3_main_env", G.Spec(ties="all", env="power", trend="quad_safe"), power),
                               ("A1_corrected", G.Spec(ties="all"), None)):
            hits, rk, zs, nb = K.bootstrap(D2, spec, ctx=ctx, B=B if key != "A1_corrected" else max(40, B // 2))
            write_csv(f"bootstrap_{key}.csv", boot_rows(key, hits, rk, zs, nb))
            print("bootstrap", key, "done", flush=True)
        hits, rk, nb = K.model_averaged_bootstrap(D2, SPECS, B=40 if QUICK else 800)
        rows = []
        for k in sorted(set(hits) | set(rk), key=lambda k: -hits.get(k, 0))[:60]:
            rr = rk.get(k, [])
            full = rr + [201] * (nb - len(rr))
            f_ = hits.get(k, 0) / nb
            nm = next((r["name"] for key in res for r in pages[key][:300] if (r["sport"], r["id"]) == k), k[1])
            rows.append({"athlete": nm, "sport": k[0], "top10_frequency_over_specs_and_refits": round(f_, 3),
                         "monte_carlo_se": round(math.sqrt(f_ * (1 - f_) / nb), 3), "draws": nb,
                         "rank_p05": int(np.percentile(full, 5)), "rank_median": int(np.percentile(full, 50)),
                         "rank_p95": int(np.percentile(full, 95)) if np.percentile(full, 95) <= 200 else ">200"})
        write_csv("bootstrap_model_averaged.csv", rows)
        print("model-averaged bootstrap done", flush=True)
        json.dump({"B": B, "B_A1": max(40, B // 2), "B_model_averaged": 40 if QUICK else 800, "block_length": 3,
                   "specs_in_ensemble": n_spec, "variants": [v[0] for v in V]},
                  open(os.path.join(OUT, "run_meta.json"), "w"))


if __name__ == "__main__":
    main()
