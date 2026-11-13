"""Fill the interactive page's DATA block from board.py and results/ -> web/plinko_lab.html

    cd plinko && python3 web/build_lab.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import board as B          # noqa: E402
import galton              # noqa: E402
import physics as P        # noqa: E402

here = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(here)
st = json.load(open(os.path.join(root, "results/strategy.json")))
zf = np.load(os.path.join(root, "results/kernel_final.npz"))
K = zf["K"]
real = json.load(open(os.path.join(root, "data/real_outcomes.json")))

rng = np.random.default_rng(2026)
example = np.bincount(rng.choice(9, size=1000, p=K[4] / K[4].sum()), minlength=9)

r6 = lambda a: [round(float(x), 6) for x in a]
data = dict(
    geom=dict(pitch=B.SLOT_PITCH, row=B.ROW_SPACING, chip=B.CHIP_DIAMETER, peg=B.PEG_DIAMETER,
              div_t=B.DIVIDER_THICKNESS, y_div=B.last_row_y() + B.DIVIDER_TOP_BELOW_LAST_ROW,
              y_floor=B.last_row_y() + B.FLOOR_BELOW_LAST_ROW, rows=B.N_ROWS),
    segs=[r6(s) for s in B.segments()],
    pegs=[r6(p) for p in B.pegs()[0]],
    walls=[[r6(p) for p in B.wall_polyline(-1)], [r6(p) for p in B.wall_polyline(1)]],
    tuned=dict(P.DEFAULT, e=float(zf["e"]), sx=P.RELEASE["sx"]),
    kernel=[r6(row) for row in K],
    coin=[r6(row) for row in galton.kernel()],
    example=example.tolist(),
    footer=(f"Geometry measured from broadcast frames and scaled by the chip; no official drawings are public. "
            f"Bounciness tuned so the model reproduces {real['totals']['chips']:,} chips logged by "
            f"<a href=\"http://tpirstats.com/Season54/Pricing/Plinko.htm\" target=\"_blank\" rel=\"noopener\">tpirstats.com</a> "
            f"(seasons 44 to 54). Coin-flip model as in Biesterfeld, <i>Journal of Statistics Education</i> 9(3), 2001. "
            f"The example batch on load is a sample from the model, not show data. Independent analysis, not affiliated "
            f"with any TV show or network."),
)
tpl = open(os.path.join(here, "plinko_lab.template.html")).read()
html = tpl.replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
open(os.path.join(here, "plinko_lab.html"), "w").write(html)
print("wrote web/plinko_lab.html", len(html), "bytes")
