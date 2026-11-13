"""Shot list for the chaos-theory cut ("Plinko and chaos").

    python3 prepare_chaos.py   -> out/anim3d_chaos/shots.json, paths.npz

Reuses the Blender plates and most chip passes of the earlier cut (`chips_from`, with `frame_offset` to start
partway into a pass; `hold_chips` holds the first or last rendered frame when a shot outlasts its pass). New chip
passes: the hook (a winning drop, then its replay one atom away), heat (that drop alone, at half speed, while
the narration follows its energy into heat) and warm (the drop and its exact replay on a warmed board, together).
Shot lengths follow the narration (results/narration_cues.json, from narration.py).
"""
import json
import math
import os

import numpy as np

import board as B

W = B.SLOT_PITCH
OUT = "out/anim3d_chaos"


def main():
    os.makedirs(OUT, exist_ok=True)
    h = np.load("results/chaos2_hook.npz")
    w = np.load("results/chaos2_warm.npz")
    # the warm winner again with its impact flags (column 4), for the heat shot's glows; must match bit for bit
    import chaos2 as C2
    o = json.load(open("results/chaos2_onevar.json"))
    _, rec = C2.trajectory_scaled(B.drop_x(5) + o["pair_release_m"], 0.0, fps=60)
    assert np.array_equal(rec[:, :4], w["win"]), "warm path moved"
    np.savez_compressed(f"{OUT}/paths.npz", hook_win=h["win"], hook_replay=h["replay"], warm_win=w["win"],
                        warm_replay=w["replay"], warm_win_full=rec)
    T_win = float(h["win"][-1, 0])
    front = [[0, 0.88, 0.50, 0.50]]
    top = [[0, 0.86, 0.5, 0.52]]
    cues = json.load(open("results/narration_cues.json"))
    need = lambda n, base: max(base, math.ceil(2 * (cues[n]["need"] + 0.3)) / 2) if n in cues else base
    heat_fall = cues["heat"]["lines"][2]["phrases"][1][0]          # "As it falls, ..."
    T_warm = float(w["win"][-1, 0])
    d_det, d_out = need("deterministic", 11.0), need("outro", 10.0)
    shots = [
        dict(name="hook", cam="front", dur=need("hook", 17.0), crop=front, hold_chips=True,
             chips=[dict(path="hook_win", t0=1.0, appear=0.3, hold=True, linger=99, face="chip_face"),
                    dict(path="hook_replay", t0=7.2, appear=6.7, hold=True, linger=99, face="chip_blue_face")]),
        dict(name="deterministic", cam="hero", dur=d_det, chips_from="open", hold_chips=True,
             crop=[[0, 1.0, 0.50, 0.50], [11.0, 0.80, 0.40, 0.47]], chips=[]),      # as its chip pass
        dict(name="twins", cam="front", dur=need("twins", 14.0), crop=front, chips_from="twins", hold_chips=True,
             chips=[]),
        dict(name="chaos", cam="hero", dur=need("chaos", 40.0), crop=[[0, 0.80, 0.40, 0.47]], chips_from="none",
             chips=[]),
        dict(name="mechanism", cam="top", dur=need("mechanism", 18.0), crop=top, chips_from="zoom",
             hold_chips=True, chips=[]),
        dict(name="horizon", cam="top", dur=need("horizon", 14.0), crop=top, chips_from="zoom", hold_chips=True,
             chips=[]),
        dict(name="crux", cam="front", dur=need("crux", 28.0), crop=front, chips=[]),
        dict(name="islands", cam="top", dur=need("islands", 37.0), crop=top, chips_from="none", chips=[]),
        dict(name="heat", cam="front", dur=need("heat", 40.0), crop=front, slowmo=0.5, hold_chips=True,
             render_window=[heat_fall - 0.2, heat_fall + T_warm / 0.5 + 0.6],
             chips=[dict(path="warm_win", t0=heat_fall, appear=0.0, hold=True, linger=99, face="chip_face")]),
        dict(name="warm", cam="front", dur=need("warm", 17.0), crop=front, slowmo=0.5,
             chips=[dict(path="warm_win", t0=2.5, appear=0.3, hold=True, linger=99, face="chip_face",
                         depth_offset=0.0004),
                    dict(path="warm_replay", t0=2.5, appear=0.3, hold=True, linger=99, face="chip_blue_face")]),
        dict(name="realworld", cam="front", dur=need("realworld", 20.0), crop=front, chips=[]),
        dict(name="statistics", cam="front", dur=need("statistics", 24.0), crop=front, chips_from="rain",
             frame_offset=120, hold_chips=True, chips=[]),
        dict(name="outro", cam="hero", dur=d_out, chips_from="outro", hold_chips=True,
             crop=[[0, 0.80, 0.40, 0.47], [10.0, 1.0, 0.50, 0.50]], chips=[]),
    ]
    meta = dict(fps=30, hz=60, shots=shots, hook_t_win=T_win)
    json.dump(meta, open(f"{OUT}/shots.json", "w"), indent=1)
    print([(s["name"], s["dur"]) for s in shots], "total", sum(s["dur"] for s in shots))


if __name__ == "__main__":
    main()
