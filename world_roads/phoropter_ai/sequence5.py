"""Turn an exam (a record5 trace) into a short list of named steps, stage by stage, for the side-by-side.

Step names a clinician would use: find the sphere, fine-tune the sphere, astigmatism check, cylinder axis
(cross-cylinder), spin the axis, cylinder power (cross-cylinder), balance the two eyes, both eyes: fine-tune.
"""
import env5 as E5
import ui

STAGES = ["RIGHT EYE  ·  left eye covered", "LEFT EYE  ·  right eye covered", "BOTH EYES OPEN"]
KIND_NAME = {"sph_big": "Find the sphere (big steps)", "sph_fine": "Fine-tune the sphere",
             "fog": "Fog, then clear (best sphere)", "chk": "Astigmatism check (180/90, 45/135)",
             "axis": "Cylinder AXIS (cross-cylinder)", "spin": "Spin the axis, patient says stop",
             "power": "Cylinder POWER (cross-cylinder)", "cyl": "Add or remove cylinder",
             "turn": "Turn the axis", "balance": "Balance the two eyes (prism)", "ou": "Both eyes: fine-tune the sphere",
             "ou_fog": "Fog both eyes", "ou_clear": "Clear the fog, both eyes together",
             "rx_no": "Final check: “Still blurry!”"}
KIND_GROUP = {"sph_big": "sphere", "sph_fine": "sphere", "fog": "sphere", "ou": "sphere", "chk": "jcc", "axis": "jcc",
              "power": "jcc", "spin": "axis", "turn": "axis", "cyl": "cyl", "balance": "prism", "ou_fog": "sphere",
              "ou_clear": "sphere", "rx_no": "rx"}


def kind_of(a, ph):
    if a in (E5.SP100, E5.SM100):
        return "sph_big"
    if a in (E5.SP25, E5.SM25, E5.CMP_P, E5.CMP_M, E5.READ, E5.READ_LINE):
        return "ou" if ph == 2 else "sph_fine"
    if a in (E5.OU_P, E5.OU_M):
        return "ou"
    if a in (E5.CHK_180, E5.CHK_45):
        return "chk"
    if a == E5.JCC_AX:
        return "axis"
    if a == E5.JCC_PW:
        return "power"
    if a == E5.SPIN:
        return "spin"
    if a in (E5.AXP2, E5.AXM2, E5.AXP5, E5.AXM5, E5.AXP15, E5.AXM15):
        return "turn"
    if a in (E5.CYL_MORE, E5.CYL_LESS):
        return "cyl"
    if a in (E5.BALANCE, E5.R_P, E5.L_P):
        return "balance"
    return None                                     # line size, next, prescribe


def steps_of(tr):
    """[(stage index, kind, n_buttons, n_questions, before, after, extra)] in order."""
    out = []
    cur = None
    ph = 0
    for s in tr["steps"]:
        a = s["a"]
        if a == E5.NEXT:
            ph = (ph + 1) % 3
            cur = None
            continue
        if a == E5.RX:
            if s.get("code", -1) in (7, 8, 9):            # refused: "it's still blurry"
                cur = dict(stage=s["phase_before"], kind="rx_no", n=1, q=1, before=s["before"], after=s["after"],
                           acts=[a], code=s["code"])
                out.append(cur)
                cur = None
            continue
        k = kind_of(a, s["phase_before"])
        if k is None:
            if cur is not None:
                cur["n"] += 1
            continue
        # merge rules: turning the axis belongs to an axis refinement; cylinder clicks to a power refinement or a
        # check; sphere quarter steps inside a power refinement keep the spherical equivalent
        if cur is not None:
            if (cur["kind"] == "axis" and k == "turn") or (cur["kind"] == "power" and k in ("cyl", "sph_fine")
                                                          and a not in (E5.READ, E5.READ_LINE, E5.CMP_P, E5.CMP_M)) \
                    or (cur["kind"] == "chk" and k == "cyl") or (cur["kind"] == "spin" and k == "turn") \
                    or (cur["kind"] == "sph_big" and a in (E5.READ, E5.READ_LINE)):
                k = cur["kind"]
            if cur["kind"] == "turn" and k == "axis":
                cur["kind"] = "axis"
            if cur["kind"] == "cyl" and k == "power":
                cur["kind"] = "power"
            if k == "sph_big" and cur["kind"] == "sph_fine" and all(x in (E5.READ, E5.READ_LINE) for x in cur["acts"]):
                cur["kind"] = "sph_big"
        if cur is None or k != cur["kind"] or s["phase_before"] != cur["stage"]:
            cur = dict(stage=s["phase_before"], kind=k, n=0, q=0, before=s["before"], after=s["after"], acts=[])
            out.append(cur)
        cur["n"] += 1
        cur["acts"].append(a)
        cur["after"] = s["after"]
        if a in (E5.READ, E5.READ_LINE, E5.CMP_P, E5.CMP_M, E5.JCC_AX, E5.JCC_PW, E5.CHK_180, E5.CHK_45, E5.BALANCE,
                 E5.SPIN):
            cur["q"] += 1
    # a run of quarter steps that starts by adding plus is a fog-then-clear
    for st in out:
        lens = [x for x in st["acts"] if x in (E5.SP25, E5.SM25, E5.OU_P, E5.OU_M)]
        if st["kind"] == "sph_fine" and lens[:2] == [E5.SP25, E5.SP25]:
            st["kind"] = "fog"
        if st["kind"] == "ou" and lens:
            plus = sum(x == E5.OU_P for x in lens)
            if lens[:2] == [E5.OU_P, E5.OU_P] and plus == len(lens):
                st["kind"] = "ou_fog"
            elif sum(x == E5.OU_M for x in lens) > plus:
                st["kind"] = "ou_clear"
    return out


def detail(st):
    e = min(st["stage"], 1)
    S, C, ax = st["after"][e]
    k = st["kind"]
    q = st["q"]
    qs = f"{q} question{'s' if q != 1 else ''}" if q else f"{st['n']} click{'s' if st['n'] != 1 else ''}"
    if k in ("sph_big", "sph_fine", "fog"):
        return f"{qs}  →  sphere {ui.fmt_d(S)}"
    if k in ("chk",):
        return f"{qs}  →  " + (f"cylinder at {int(round(ax)) % 180 or 180}°" if C < 0 else "no astigmatism found")
    if k in ("axis", "spin", "turn"):
        return f"{qs}  →  axis {int(round(ax)) % 180 or 180}°"
    if k in ("power", "cyl"):
        return f"{qs}  →  cylinder {ui.fmt_d(C, False) if C < 0 else '0.00'}"
    if k == "rx_no":
        return "  →  " + {7: "right eye", 8: "left eye", 9: "both eyes"}[st["code"]]
    if k in ("balance", "ou", "ou_fog", "ou_clear"):
        (S0, _, _), (S1, _, _) = st["after"]
        return f"{qs}  →  R {ui.fmt_d(S0)}, L {ui.fmt_d(S1)}"
    return qs


if __name__ == "__main__":
    import sys, json
    tr = json.load(open(sys.argv[1]))
    for st in steps_of(tr):
        print(STAGES[st["stage"]][:10], KIND_NAME[st["kind"]], "|", detail(st))
