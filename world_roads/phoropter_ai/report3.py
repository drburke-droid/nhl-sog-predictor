"""Write RESULTS_CLINIC.md from data/results_f.json and data/stages_f.json: every number in the clinic video."""
import os, json

HERE = os.path.dirname(os.path.abspath(__file__))


def pct(x):
    return f"{100 * x:.1f}%"


def mmss(m):
    a, b = divmod(int(round(m * 60)), 60)
    return f"{a}:{b:02d}"


def ar_section(G):
    m = G["meta"]
    L = ["# Results: the clinic video (autorefractor start)", "",
         f"{m['test_patients']:,} new simulated patients (seed {m['seed']}). Every exam starts at the patient's "
         "autorefractor reading (env3.ar_pair). The AI is the final network after "
         f"{m['exams']:,} practice patients (`runs/f4/ckpt/final.pt`), choosing its most likely button at each step. "
         "The textbook script starts from the same reading. Net points = points for the glasses minus 4 per minute "
         "of chair time. See CLINIC.md.", "",
         "| Examiner | Net points | Points for the glasses | Chair time | Buttons | SE within 0.25 D | Cylinder within "
         "0.25 D | Whole Rx within 0.25 D | Exact (<0.13 D) | Happy | Rx check + review | Over-minused, under 40 |",
         "|---|" + "---:|" * 11]
    for name, key in (("Autorefractor reading alone", "ar"), ("Textbook script", "textbook"), ("AI", "ai")):
        s = G[key]["summary"]
        L.append(f"| {name} | {s['net']:+.1f} | {s['points']:+.1f} | {mmss(s['minutes'])} | {s['steps']:.0f} | "
                 f"{pct(s['se_within_025'])} | {pct(s['cyl_within_025'])} | {pct(s['vec_within_025'])} | {pct(s['exact'])} | "
                 f"{pct(s['happy'])} | {pct(s['recheck'] + s['review'])} | {pct(s['over_minus_young'])} |")
    L += ["", "Astigmatism of 0.75 D or more, eyes within 0.25 D by orientation:", "",
          "| Examiner | With the rule | Against the rule | Oblique |", "|---|---:|---:|---:|"]
    for name, key in (("Autorefractor reading alone", "ar"), ("Textbook script", "textbook"), ("AI", "ai")):
        o = G[key]["orient"]
        L.append(f"| {name} | {pct(o['wtr_within'])} | {pct(o['atr_within'])} | {pct(o['obl_within'])} |")
    L += ["", "Repeatability (the same 2,000 patients twice, fresh answers; spherical equivalent):", "",
          "| Examiner | Agree within 0.25 D | Within 0.50 D |", "|---|---:|---:|"]
    for name, key in (("Textbook script", "textbook"), ("AI", "ai")):
        r = G[key]["repeat"]
        L.append(f"| {name} | {pct(r['within_025'])} | {pct(r['within_050'])} |")
    L += ["", "Habits per exam (mean):", "", "| | AI | Textbook |", "|---|---:|---:|"]
    for gname in G["ai"]["habits"]["mean_count"]:
        L.append(f"| {gname} | {G['ai']['habits']['mean_count'][gname]:.1f} | {G['textbook']['habits']['mean_count'][gname]:.1f} |")
    L.append(f"| cross-cylinder axis questions | {G['ai']['jcc_axis']:.1f} | {G['textbook']['jcc_axis']:.1f} |")
    return L


def main():
    R = json.load(open(os.path.join(HERE, "data", "results_f.json")))
    st = json.load(open(os.path.join(HERE, "data", "stages_f.json")))
    m = R["meta"]
    L = ["# Results: the clinic video", "",
         f"{m['test_patients']:,} new simulated patients (seed {m['seed']}), every exam starting at plano, the axis dial "
         "where the last patient left it. The AI is the final network after "
         f"{m['exams']:,} practice patients (`runs/f2/ckpt/{m['ckpt']}`), choosing its most likely button at each step. "
         "Error is optics.rx_error; SE is the spherical equivalent. Points are clinic points (clinic.py); net = points "
         "for the glasses minus 4 per minute of chair time. See CLINIC.md.", "",
         "| Examiner | Net points | Points for the glasses | Chair time | Buttons | SE within 0.25 D | SE within 0.50 D | "
         "Cylinder within 0.25 D | Whole Rx within 0.25 D | Exact (<0.13 D) | Over-minused, under 40 | Over-plussed |",
         "|---|" + "---:|" * 11]
    for name, key in (("Plano for everyone", "plano"), ("Textbook script", "textbook"), ("AI", "ai")):
        s = R[key]["summary"]
        L.append(f"| {name} | {s['net']:+.1f} | {s['points']:+.1f} | {mmss(s['minutes'])} | {s['steps']:.0f} | "
                 f"{pct(s['se_within_025'])} | {pct(s['se_within_050'])} | {pct(s['cyl_within_025'])} | "
                 f"{pct(s['vec_within_025'])} | {pct(s['exact'])} | {pct(s['over_minus_young'])} | {pct(s['over_plus'])} |")
    L += ["", "Outcome chances (mean over patients) and one sampled outcome per patient:", "",
          "| Examiner | Happy | Annoyed | Rx check | Bad review | Walked out | Sampled: happy / annoyed / Rx check / review / walked out |",
          "|---|---:|---:|---:|---:|---:|---|"]
    for name, key in (("Plano for everyone", "plano"), ("Textbook script", "textbook"), ("AI", "ai")):
        s = R[key]["summary"]
        L.append(f"| {name} | {pct(s['happy'])} | {pct(s['annoyed'])} | {pct(s['recheck'])} | {pct(s['review'])} | "
                 f"{pct(s['timeout'])} | {' / '.join(str(x) for x in R[key]['outcomes'])} |")
    L += ["", "## Repeatability (the same 2,000 patients twice, fresh answers; spherical equivalent, both eyes)", "",
          "| Examiner | Agree within 0.25 D | Within 0.50 D | 95% range of the difference |", "|---|---:|---:|---|"]
    for name, key in (("Textbook script", "textbook"), ("AI", "ai")):
        r = R[key]["repeat"]
        L.append(f"| {name} | {pct(r['within_025'])} | {pct(r['within_050'])} | {r['p025']:+.2f} to {r['p975']:+.2f} D |")
    L += ["", "Published, real optometrists: about 80% within 0.25 D (Goss & Grosvenor 1996; Shah et al. 2009: 81%). "
              "See PUBLISHED.md.", "",
          "## Habits per exam (mean)", "", "| | AI | Textbook |", "|---|---:|---:|"]
    Ha, Ht = R["ai"]["habits"], R["textbook"]["habits"]
    for g in Ha["mean_count"]:
        L.append(f"| {g} | {Ha['mean_count'][g]:.1f} | {Ht['mean_count'][g]:.1f} |")
    for i, lab in enumerate(("right eye", "left eye", "both eyes")):
        L.append(f"| minutes: {lab} | {Ha['stage_minutes'][i]:.2f} | {Ht['stage_minutes'][i]:.2f} |")
    L.append(f"| switches between sphere and cylinder work | {Ha['sph_cyl_switches']:.1f} | {Ht['sph_cyl_switches']:.1f} |")
    T1 = R["term1"]
    L += ["", "## The shortcut (end of practice term 1, 1,630,473 patients)", "",
          "| Clinic | Net points | SE within 0.25 D | Axis turns per exam | Cylinders at 180° | With-the-rule within 0.25 D | "
          "Against-the-rule | Oblique |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for lab, k in (("Dial always at 180° (as practised)", "dial180"), ("Dial where the last patient left it", "random_dial")):
        x = T1[k]
        L.append(f"| {lab} | {x['summary']['net']:+.1f} | {pct(x['summary']['se_within_025'])} | {x['axis_turns']:.2f} | "
                 f"{pct(x['cyl_at_180'])} | {pct(x['wtr_within'])} | {pct(x['atr_within'])} | {pct(x['obl_within'])} |")
    L += ["", "## Practice checkpoints (600 test patients each, sampling buttons as it practised)", "",
          "| Patients practised | Term | Net points | Chair time | Happy | SE within 0.25 D | Whole Rx within 0.25 D | "
          "Plano Rx | 1-or-2 | Chart | Cross-cyl | Axis | Balance |", "|---:|---:|" + "---:|" * 11]
    for r in st:
        h = r["habits"]
        L.append(f"| {r['exams']:,} | {r['term']} | {r['net']:+.1f} | {mmss(r['minutes'])} | {pct(r['happy'])} | {pct(r['se'])} | "
                 f"{pct(r['vec'])} | {pct(r['plano_rx'])} | {h['sphere 1-or-2']:.1f} | {h['chart reading']:.1f} | "
                 f"{h['cross-cylinder']:.1f} | {h['axis']:.1f} | {h['balance']:.1f} |")
    g = os.path.join(HERE, "data", "results_g.json")
    if os.path.exists(g):
        G = json.load(open(g))
        L = ar_section(G) + ["", "---", "", "# Earlier: the plano-start comparison (after practice term 2)", ""] + L[1:]
    open(os.path.join(HERE, "RESULTS_CLINIC.md"), "w").write("\n".join(L) + "\n")
    print("wrote RESULTS_CLINIC.md")


if __name__ == "__main__":
    main()
