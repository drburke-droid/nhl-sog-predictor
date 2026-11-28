"""The video's step legend: every phoropter button belongs to one numbered step of the textbook (Borish)
sequence, so any exam, the textbook's or the AI's, can be written as a row of step numbers.

    1  Cover an eye (or open both)
    2  Lens power: stronger or weaker, and read the chart
    3  "1 or 2?": a touch more or less power
    4  Is there astigmatism? (cross-cylinder check)
    5  Astigmatism angle: "1 or 2?"
    6  Astigmatism strength: "1 or 2?"
    7  Balance the two eyes
    8  Prescribe
    S  Spin the angle: "say stop!" (not part of the textbook sequence)
"""
import env5 as E5

LEGEND = [("1", "Cover an eye (or open both)"),
          ("2", "Lens power: stronger or weaker, read the chart"),
          ("3", "“1 or 2?”  a touch more or less power"),
          ("4", "Is there astigmatism?  (cross-cylinder check)"),
          ("5", "Astigmatism angle:  “1 or 2?”"),
          ("6", "Astigmatism strength:  “1 or 2?”"),
          ("7", "Balance the two eyes"),
          ("8", "Prescribe"),
          ("S", "Spin the angle: “say stop!”  (not in the textbook)")]
STEP_COL = {"1": (150, 160, 180), "2": (74, 144, 255), "3": (120, 190, 255), "4": (240, 90, 90),
            "5": (255, 120, 120), "6": (255, 160, 120), "7": (245, 160, 50), "8": (250, 205, 90),
            "S": (180, 130, 255)}

STEP = {}
for a in (E5.NEXT,):
    STEP[a] = "1"
for a in (E5.SP25, E5.SM25, E5.SP100, E5.SM100, E5.OU_P, E5.OU_M, E5.R_P, E5.L_P, E5.READ, E5.READ_LINE, E5.LINE_SM,
          E5.LINE_LG):
    STEP[a] = "2"
for a in (E5.CMP_P, E5.CMP_M):
    STEP[a] = "3"
for a in (E5.CHK_180, E5.CHK_45):
    STEP[a] = "4"
for a in (E5.JCC_AX, E5.AXP2, E5.AXM2, E5.AXP5, E5.AXM5, E5.AXP15, E5.AXM15):
    STEP[a] = "5"
for a in (E5.JCC_PW, E5.CYL_MORE, E5.CYL_LESS):
    STEP[a] = "6"
STEP[E5.BALANCE] = "7"
STEP[E5.RX] = "8"
STEP[E5.SPIN] = "S"
assert len(STEP) == E5.N_ACT3


def steps(actions):
    """Every button as its step number."""
    return [STEP[int(a)] for a in actions]


def collapse(actions):
    """Steps with repeats merged: 2 2 2 3 3 -> 2 3."""
    out = []
    for s in steps(actions):
        if not out or out[-1] != s:
            out.append(s)
    return out


if __name__ == "__main__":
    import json, sys
    for n in sys.argv[1:]:
        tr = json.load(open(f"data/traces/{n}.json"))
        acts = [st["a"] for st in tr["steps"]]
        print(n, len(acts), " ".join(collapse(acts)))
