"""Write RESULTS.md from data/results_*.json: every number quoted in the video."""
import os, json

HERE = os.path.dirname(os.path.abspath(__file__))


def pct(x):
    return f"{100 * x:.1f}%"


def load(name):
    p = os.path.join(HERE, "data", f"results_{name}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def table(rows, both=False):
    head = ("| Examiner | Within 0.25 D | Exact | Mean error (D) | Moves per exam | Over-minused, under 40 | "
            "Over-plussed |" + (" Both eyes within 0.25 D |" if both else ""))
    out = [head, "|---|" + "---:|" * (6 + both)]
    for name, s in rows:
        line = (f"| {name} | {pct(s['within_025'])} | {pct(s['exact'])} | {s['mean_err']:.3f} | {s['steps']:.1f} | "
                f"{pct(s['over_minus_young'])} | {pct(s['over_plus'])} |")
        if both:
            line += f" {pct(s['both_within'])} |" if "both_within" in s else " |"
        out.append(line)
    return out


def main():
    A, B, Bp, C, Cp, D = (load(n) for n in ("a", "b", "b_plano", "c", "c_plano", "d"))
    L = ["# Results", "",
         "Simulated test patients drawn independently of training (seed 2026). Every examiner in a table saw the same "
         "patients. Error is optics.rx_error: the length of (spherical-equivalent error, leftover cylinder), with "
         "the leftover cylinder in full dioptres. Exact means under 0.13 D. Over-minused means the spherical "
         "equivalent is at least 0.25 D more minus than the truth. See ACCURACY.md for every assumption.", ""]
    if A:
        L += ["## One eye, whole-line chart, starting from the autorefractor (attempts 1 and 2)", ""]
        rows = [("Autorefractor reading alone", A["autoref"]["summary"]),
                ("Script: patient-pleaser", A["pleaser"]["summary"]),
                ("Script: textbook", A["textbook"]["summary"]),
                ("AI, attempt 1 (1,002,464 exams)", A["ai"]["summary"])]
        if B:
            rows.append(("AI, attempt 2 (2,002,626 exams)", B["ai"]["summary"]))
        L += table(rows)
        L += ["", "Attempt 1 on the half-weight-cylinder score it was trained on (the plain power-vector distance):", ""]
        L += table([("Script: textbook", A["textbook"]["summary_half"]), ("AI, attempt 1", A["ai"]["summary_half"])])
    if C:
        L += ["", "## One eye, letter chart (attempt 3)", "", "Starting from the autorefractor:", ""]
        L += table([("Autorefractor reading alone", C["autoref"]["summary"]),
                    ("Script: textbook (letter chart)", C["textbook"]["summary"]),
                    ("AI, attempt 3", C["ai"]["summary"])])
    if Cp:
        L += ["", "No autorefractor (starting at plano, up to 80 moves):", ""]
        rows = [("Prescribe nothing (plano)", Cp["start"]["summary"]),
                ("Script: textbook from scratch (letter chart)", Cp["textbook"]["summary"])]
        if Bp:
            rows.append(("AI, attempt 2 (never practised without one)", Bp["ai"]["summary"]))
        rows.append(("AI, attempt 3", Cp["ai"]["summary"]))
        L += table(rows)
    if D:
        L += ["", "## Two eyes, binocular balance, line-size-dependent judgement (attempt 4)", "",
              "Within, exact, mean error and over-minus are per eye (both eyes pooled).", ""]
        L += table([("Autorefractor reading alone", D["autoref"]["summary"]),
                    ("Script: textbook", D["textbook"]["summary"]),
                    ("Script: the optometrist's routine", D["mine"]["summary"]),
                    ("AI, attempt 4", D["ai"]["summary"])], both=True)
        L += ["", "Moves used per exam (mean):", "", "| Move | AI, attempt 4 | Textbook | Optometrist's routine |",
              "|---|---:|---:|---:|"]
        for m, v in D["ai"]["habits"]["mean_count"].items():
            t = D["textbook"]["habits"]["mean_count"][m]
            o = D["mine"]["habits"]["mean_count"][m]
            if max(v, t, o) >= 0.01:
                L.append(f"| {m} | {v:.2f} | {t:.2f} | {o:.2f} |")
    for name, R in (("attempt 1", A), ("attempt 3", C)):
        if not R or "strategy" not in R["ai"]:
            continue
        st = R["ai"]["strategy"]
        L += ["", f"## Habits, {name}", "", "| Move | Mean uses per exam | Exams using it |", "|---|---:|---:|"]
        for m, v in st["mean_count"].items():
            if v >= 0.005:
                L.append(f"| {m} | {v:.2f} | {pct(st['used_pct'][m])} |")
        L += ["", "Checkpoints during training (2,000 of the test patients):", "",
              "| Checkpoint | Practice exams | Within 0.25 D | Exact | Moves |", "|---|---:|---:|---:|---:|"]
        for s in R["stages"]:
            L.append(f"| {s['ckpt']} | {s['practice_exams']:,} | {pct(s['within_025'])} | {pct(s['exact'])} | "
                     f"{s['steps']:.1f} |")
    open(os.path.join(HERE, "RESULTS.md"), "w").write("\n".join(L) + "\n")
    print("wrote RESULTS.md")


if __name__ == "__main__":
    main()
