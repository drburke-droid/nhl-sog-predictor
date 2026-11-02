"""Figures for top10_robustness.md (PNG, light background).

    python analysis/plots.py        # from followup/, after run_all.py; writes figures/*.png

Palette: validated categorical slots blue #2a78d6, orange #eb6834, aqua #1baf7a (aqua below 3:1 contrast, so
every aqua mark is labelled directly); sequential = one blue hue light to dark; ink #0b0b0b / #52514e.
"""
import csv, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
F = os.path.dirname(HERE)
OUT = os.path.join(F, "outputs")
FIG = os.path.join(F, "figures")
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True})


def read(name):
    return list(csv.DictReader(open(os.path.join(OUT, name), encoding="utf-8")))


def fig_rank_heatmap():
    V = ["A0_original", "A1_corrected", "A2_main", "A2_twosided", "A2_w7", "A1_flat", "B1_allseasons",
         "E2_env_power", "E3_main_env", "A2_main_cal", "E3_main_env_cal", "F_ratio_field100", "F_ratio_field_estimated"]
    lab = ["Original", "Corrected data", "MAIN (quad. +/-15)", "MAIN, two-sided only", "+/-7 (CV best)", "Flat window",
           "All-seasons rarity", "Env. (estimated)", "MAIN + env.", "MAIN calibrated", "MAIN + env. calibrated",
           "Field offset 100%", "Field (estimated)"]
    top = read("top10_by_variant.csv")
    names = []
    for v in V:
        for r in top:
            if r["variant"] == v and (r["athlete"], r["sport"]) not in names:
                names.append((r["athlete"], r["sport"]))
    S = read("rankings_sensitivity.csv")
    best = {}
    for r in S:
        for v in V:
            k = (r["athlete"], r["sport"])
            rp = r.get(f"{v}_rank_page")
            if rp:
                best[(k, v)] = min(int(rp), best.get((k, v), 999))
    M = np.full((len(names), len(V)), np.nan)
    for i, k in enumerate(names):
        for j, v in enumerate(V):
            if (k, v) in best:
                M[i, j] = best[(k, v)]
    order = np.argsort(np.nanmean(np.where(np.isnan(M), 60, np.minimum(M, 60)), axis=1))
    M, names = M[order], [names[i] for i in order]
    fig, ax = plt.subplots(figsize=(11, 0.33 * len(names) + 2.2))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#0d3b73", "#2a78d6", "#a9c8ef", "#eef4fc"])
    show = np.where(np.isnan(M), 60, np.minimum(M, 60))
    ax.imshow(show, cmap=cmap, vmin=1, vmax=60, aspect="auto")
    for i in range(len(names)):
        for j in range(len(V)):
            t = "–" if np.isnan(M[i, j]) else (str(int(M[i, j])) if M[i, j] <= 99 else "99+")
            ax.text(j, i, t, ha="center", va="center", fontsize=8, color="white" if show[i, j] <= 12 else INK)
    ax.set_xticks(range(len(V)))
    ax.set_xticklabels(lab, rotation=40, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f"{n} ({s})" for n, s in names], fontsize=8.5)
    ax.grid(False)
    ax.set_title("Rank of each athlete's best season under each model (1 = greatest; darker = higher)", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_rank_heatmap.png"), dpi=150)
    plt.close(fig)


def fig_bootstrap(key="E3_main_env", title="MAIN + environment"):
    rows = read(f"bootstrap_{key}.csv")[:25]
    fig, ax = plt.subplots(figsize=(9, 7.5))
    for i, r in enumerate(rows):
        hi = 200 if r["rank_p95"] == ">200" else int(r["rank_p95"])
        lo, md = int(r["rank_p05"]), int(r["rank_median"])
        ax.plot([lo, min(hi, 60)], [i, i], color=BLUE, lw=2, solid_capstyle="round")
        if hi > 60:
            ax.annotate("", xy=(62, i), xytext=(58, i), arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.5))
        ax.plot(md, i, "o", color=BLUE, ms=6, mec=SURF, mew=1.5)
        ax.text(64, i, f"P(top 10) {float(r['p_top10']):.0%}", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r['athlete']} {r['season']} ({r['sport']})" for r in rows], fontsize=8.5)
    ax.invert_yaxis()
    ax.axvline(10.5, color=ORANGE, lw=1.2, ls="--")
    ax.text(10.8, len(rows) - 0.5, "top-ten line", color=INK2, fontsize=8.5)
    ax.set_xlim(0, 76)
    ax.set_xlabel("rank of the athlete's best season (median and 5-95% range over block-bootstrap replicates)")
    ax.set_title(f"{title}: how sure is the order?", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"fig2_bootstrap_{key}.png"), dpi=150)
    plt.close(fig)


def fig_nba_denominators():
    L = list(csv.DictReader(open(os.path.join(F, "data", "context", "nba_league.csv"))))
    y = np.array([int(r["season"]) for r in L])
    f = lambda k: np.array([float(r[k]) if r[k] not in ("", "None") else np.nan for r in L])
    pts, pm, lo, hi = f("pts_per_team_g"), f("pace_team_mean"), f("pace_team_min"), f("pace_team_max")
    so = f("fga_per_team_g") + 0.44 * f("fta_per_team_g")
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.fill_between(y, lo, hi, color=BLUE, alpha=0.15, lw=0)
    ax.plot(y, pm, color=BLUE, lw=2, label="pace: mean of estimated team paces (band = slowest to fastest team)")
    ax.plot(y, pts, color=ORANGE, lw=2, label="league points per team-game")
    ax.plot(y, so, color=AQUA, lw=2, label="shot opportunities, FGA + 0.44 FTA per team-game")
    ax.annotate("1961-62: mean of team paces 126.2;\nPhiladelphia (Wilt's team) 131.1", xy=(1962, 131.1), xytext=(1977, 128),
                fontsize=8.5, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.axvline(1974, color=INK2, lw=0.8, ls=":")
    ax.text(1974.6, 70, "league pace published from 1973-74\n(turnovers and offensive rebounds recorded)", fontsize=8, color=INK2)
    ax.text(y[-1] + 0.5, pts[-1], "points", color=INK, fontsize=8.5, va="center")
    ax.text(y[-1] + 0.5, so[-1] - 3, "shot opps.", color=INK, fontsize=8.5, va="center")
    ax.set_xlim(1946, 2034)
    ax.set_ylabel("per team per game")
    ax.legend(loc="lower left", fontsize=8.5, frameon=False)
    ax.set_title("Three NBA scoring-environment denominators (Basketball-Reference)", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_nba_denominators.png"), dpi=150)
    plt.close(fig)


def fig_babe_dye():
    import gsmodel as G
    import context as X
    D = G.Data("original")
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for label, spec, kw, col in (("scoring only", G.Spec(env="ratio"), dict(env="ratio"), BLUE),
                                 ("scoring + field offset 100%", G.Spec(env="ratio", field=1.0), dict(env="ratio", field=1.0), ORANGE)):
        ctx = X.make(D, sports=("NHL",), **kw)
        r = next(r for r in G.score(D, spec, ctx=ctx, cats=["nhl_goals"], details=True) if r["year"] == 1925)
        s = np.array([p[0] for p in r["pts"]], float)
        v = np.array([p[1] for p in r["pts"]], float)
        A = np.column_stack([np.ones_like(s), s - 1925])
        c = np.linalg.lstsq(A, v, rcond=None)[0]
        ax.plot(s, v, "o", color=col, ms=5, mec=SURF, mew=1)
        ax.plot(s, A @ c, color=col, lw=1.5)
        ax.plot(1925, r["x"], "D", color=col, ms=8, mec=INK, mew=0.8)
        ax.text(1925.4, r["x"], f"Dye: z = {r['z']:.2f} ({label}; scale {r['scale']:.2f})", fontsize=8.5, va="center", color=INK)
    ax.set_xlabel("season (ending year)")
    ax.set_ylabel("goals / league goals per team-game\n(+ field offset where applied)")
    ax.set_title("Babe Dye 1924-25: the field offset raises his z by shrinking the comparators' scatter", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_babe_dye.png"), dpi=150)
    plt.close(fig)


def fig_elasticity():
    rows = read("gamma_estimates.csv")
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for i, r in enumerate(rows):
        g, se, p = float(r["elasticity_field_median"]), float(r["se_field"]), float(r["gamma_pooled"])
        ax.plot([g - 1.96 * se, g + 1.96 * se], [i, i], color=BLUE, lw=2, solid_capstyle="round")
        ax.plot(g, i, "o", color=BLUE, ms=6, mec=SURF, mew=1.2)
        ax.plot(p, i, "D", color=ORANGE, ms=5, mec=SURF, mew=1)
    ax.axvline(1, color=INK2, lw=0.8, ls=":")
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["category"] for r in rows], fontsize=8.5)
    ax.invert_yaxis()
    ax.plot([], [], "o", color=BLUE, label="estimate and 95% interval (first differences of logs)")
    ax.plot([], [], "D", color=ORANGE, label="partially pooled value used by the environment models")
    ax.set_xlabel("elasticity of the field (median of ranks 2-10) to the league rate; 1 = proportional")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.09), ncol=1, fontsize=8.5, frameon=False)
    ax.set_title("Does the field move with the league's scoring environment?", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig5_elasticity.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    fig_nba_denominators()
    fig_babe_dye()
    if os.path.exists(os.path.join(OUT, "gamma_estimates.csv")):
        fig_elasticity()
    if os.path.exists(os.path.join(OUT, "top10_by_variant.csv")):
        fig_rank_heatmap()
    for k, t in (("E3_main_env", "MAIN + environment"), ("A2_main", "MAIN (no environment adjustment)")):
        if os.path.exists(os.path.join(OUT, f"bootstrap_{k}.csv")):
            fig_bootstrap(k, t)
    print(sorted(os.listdir(FIG)))
