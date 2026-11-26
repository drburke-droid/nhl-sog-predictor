"""Write RESULTS.md from data/results_*.json (every number quoted in the video)."""
import os, json

HERE = os.path.dirname(os.path.abspath(__file__))


def pct(x):
    return f"{100 * x:.1f}%"


def main():
    runs = {k: json.load(open(os.path.join(HERE, "data", f"results_{k}.json")))
            for k in ("a", "b") if os.path.exists(os.path.join(HERE, "data", f"results_{k}.json"))}
    L = ["# Results", "", "5,000 simulated test patients (seed 2026), the same patients for every examiner. "
         "Error = dioptric distance between power vectors. Exact = under 0.13 D. Over-minused = spherical "
         "equivalent at least 0.25 D more minus than the truth.", ""]
    L += ["| Examiner | Within 0.25 D | Exact | Mean error (D) | Moves per exam | Over-minused, under 40 | Over-plussed |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    A = runs["a"]
    rows = [("Autorefractor alone", A["autoref"]["summary"]), ("Patient-pleaser script", A["pleaser"]["summary"]),
            ("Textbook script", A["textbook"]["summary"]), ("AI, attempt 1", A["ai"]["summary"])]
    if "b" in runs:
        rows.append(("AI, attempt 2", runs["b"]["ai"]["summary"]))
    for name, s in rows:
        L.append(f"| {name} | {pct(s['within_025'])} | {pct(s['exact'])} | {s['mean_err']:.3f} | {s['steps']:.1f} | "
                 f"{pct(s['over_minus_young'])} | {pct(s['over_plus'])} |")
    for k, R in runs.items():
        label = {"a": "Attempt 1", "b": "Attempt 2"}[k]
        st = R["ai"]["strategy"]
        L += ["", f"## {label}", "", f"Checkpoint: {R['ai']['meta']}", "",
              f"Questions per exam: {st['questions']:.1f}. Lens changes per exam: {st['lens_moves']:.1f}.", "",
              "| Move | Mean uses per exam | Exams using it |", "|---|---:|---:|"]
        for m, v in st["mean_count"].items():
            L.append(f"| {m} | {v:.2f} | {pct(st['used_pct'][m])} |")
        L += ["", "By age:", "", "| Age | n | Within 0.25 D | Exact | Mean ΔSE (D) | Over-minused | Over-plussed |",
              "|---|---:|---:|---:|---:|---:|---:|"]
        for g in R["ai"]["by_age"]:
            L.append(f"| {g['lo']}–{g['hi'] - 1} | {g['n']} | {pct(g['within'])} | {pct(g['exact'])} | "
                     f"{g['mean_dM']:+.3f} | {pct(g['over_minus'])} | {pct(g['over_plus'])} |")
        b = R["ai"]["bias"]
        L += ["", f"Over-minused under 40, patients who strongly prefer minus (bias > 0.13): "
                  f"{pct(b['over_minus_young_high_bias'])}; who barely do (bias < 0.07): "
                  f"{pct(b['over_minus_young_low_bias'])}. Textbook script: "
                  f"{pct(R['textbook']['bias']['over_minus_young_high_bias'])} / "
                  f"{pct(R['textbook']['bias']['over_minus_young_low_bias'])}.", "",
              "Checkpoints during training (2,000 of the same test patients):", "",
              "| Checkpoint | Practice exams | Within 0.25 D | Exact | Moves |", "|---|---:|---:|---:|---:|"]
        for s in R["stages"]:
            L.append(f"| {s['ckpt']} | {s['practice_exams']:,} | {pct(s['within_025'])} | {pct(s['exact'])} | "
                     f"{s['steps']:.1f} |")
    open(os.path.join(HERE, "RESULTS.md"), "w").write("\n".join(L) + "\n")
    print("wrote RESULTS.md")


if __name__ == "__main__":
    main()
