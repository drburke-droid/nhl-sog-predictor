"""Buttons and colours for the version-5 clinic exam (env5.py). Drawing helpers come from clinic_ui.py."""
from clinic_ui import *                       # noqa: F401,F403  (draw_phoropter, colours, OUT_COL, ...)
from clinic_ui import GROUP_COL
import env5 as E5

GROUP_NAME5 = {"sphere": "SPHERE: lens power", "cyl": "CYLINDER: astigmatism power", "axis": "AXIS: turn the cylinder",
               "jcc": "CROSS-CYLINDER: questions about the cylinder", "prism": "PRISM: compare the two eyes",
               "occ": "OCCLUDER: which eye", "chart": "PROJECTOR CHART", "rx": "FINISH"}

# (index, chip line 1, chip line 2, plain label, what it does, group)
BUTTONS = [
    (E5.SP25, "+0.25", "sphere", "Sphere +0.25", "a quarter step more plus", "sphere"),
    (E5.SM25, "−0.25", "sphere", "Sphere −0.25", "a quarter step more minus", "sphere"),
    (E5.SP100, "+1.00", "sphere", "Sphere +1.00", "a big jump more plus", "sphere"),
    (E5.SM100, "−1.00", "sphere", "Sphere −1.00", "a big jump more minus", "sphere"),
    (E5.CMP_P, "1 or 2?", "with +0.25?", "Sphere: “1 or 2?” (+0.25)", "clearer with +0.25 (2) or without (1)?", "sphere"),
    (E5.CMP_M, "1 or 2?", "with −0.25?", "Sphere: “1 or 2?” (−0.25)", "clearer with −0.25 (2) or without (1)?", "sphere"),
    (E5.OU_P, "+0.25", "both eyes", "Both eyes +0.25", "more plus on both eyes together", "sphere"),
    (E5.OU_M, "−0.25", "both eyes", "Both eyes −0.25", "more minus on both eyes together", "sphere"),
    (E5.R_P, "+0.25", "right eye", "Right eye +0.25", "more plus on the right eye only", "sphere"),
    (E5.L_P, "+0.25", "left eye", "Left eye +0.25", "more plus on the left eye only", "sphere"),
    (E5.CYL_MORE, "−0.25", "cylinder", "Cylinder −0.25", "more astigmatism correction", "cyl"),
    (E5.CYL_LESS, "+0.25", "cylinder", "Cylinder +0.25", "less astigmatism correction", "cyl"),
    (E5.AXP2, "+2°", "axis", "Axis +2°", "turn the cylinder a touch", "axis"),
    (E5.AXM2, "−2°", "axis", "Axis −2°", "", "axis"),
    (E5.AXP5, "+5°", "axis", "Axis +5°", "turn it a bit", "axis"),
    (E5.AXM5, "−5°", "axis", "Axis −5°", "", "axis"),
    (E5.AXP15, "+15°", "axis", "Axis +15°", "turn it a lot", "axis"),
    (E5.AXM15, "−15°", "axis", "Axis −15°", "", "axis"),
    (E5.SPIN, "Spin", "say stop!", "Spin the axis", "the patient says stop when the letters are sharpest", "axis"),
    (E5.CHK_180, "Check", "180 or 90?", "Astigmatism check: 180 or 90?",
     "cross-cylinder at 180 and at 90: is there astigmatism, and which way?", "jcc"),
    (E5.CHK_45, "Check", "45 or 135?", "Astigmatism check: 45 or 135?", "the same at 45 and 135", "jcc"),
    (E5.JCC_AX, "Axis?", "which way?", "Cylinder AXIS: which way to turn?",
     "cross-cylinder either side of the axis: “1 or 2?” says which way", "jcc"),
    (E5.JCC_PW, "Power?", "more or less?", "Cylinder POWER: more or less?",
     "cross-cylinder along the axis: “1 or 2?” says more or less cylinder", "jcc"),
    (E5.BALANCE, "Top or", "bottom?", "Balance: top or bottom?", "prism splits the chart: right eye top, left bottom",
     "prism"),
    (E5.NEXT, "Next eye", "R → L → both", "Next eye", "cover one eye: right, then left, then both open, then back to the right",
     "occ"),
    (E5.READ, "Read the", "smallest line", "“Read the smallest line”", "how far down the chart can they read?", "chart"),
    (E5.READ_LINE, "Read", "this line", "“Read this line”", "how many letters on this line?", "chart"),
    (E5.LINE_SM, "Smaller", "letters", "Smaller line", "show smaller letters", "chart"),
    (E5.LINE_LG, "Bigger", "letters", "Bigger line", "show bigger letters", "chart"),
    (E5.RX, "Final check", "prescribe", "Final check, then prescribe",
     "the patient reads with each eye: both clear, these are the glasses; if not, they say which eye is blurry", "rx"),
]
BTN = {b[0]: b for b in BUTTONS}
assert len(BUTTONS) == E5.N_ACT3, (len(BUTTONS), E5.N_ACT3)


def col_of(a):
    return GROUP_COL[BTN[a][5]]
