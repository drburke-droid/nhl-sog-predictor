"""Shot list for the chaos-theory cut ("Plinko and chaos").

    python3 prepare_chaos.py   -> out/anim3d_chaos/shots.json, paths.npz

Reuses the Blender plates and most chip passes of the earlier cut (`chips_from`, with `frame_offset` to start
partway into a pass). Two shots need new chip passes: the hook (a winning drop, then its replay one atom away)
and warm (a winning drop and its exact replay on a board warmed by one drop's heat, dropped together).
"""
import json
import os

import numpy as np

import board as B

W = B.SLOT_PITCH
OUT = "out/anim3d_chaos"


def main():
    os.makedirs(OUT, exist_ok=True)
    h = np.load("results/chaos2_hook.npz")
    w = np.load("results/chaos2_warm.npz")
    np.savez_compressed(f"{OUT}/paths.npz", hook_win=h["win"], hook_replay=h["replay"], warm_win=w["win"],
                        warm_replay=w["replay"])
    T_win = float(h["win"][-1, 0])
    front = [[0, 0.88, 0.50, 0.50]]
    top = [[0, 0.86, 0.5, 0.52]]
    shots = [
        dict(name="hook", cam="front", dur=17.0, crop=front,
             chips=[dict(path="hook_win", t0=1.0, appear=0.3, hold=True, linger=99, face="chip_face"),
                    dict(path="hook_replay", t0=7.2, appear=6.7, hold=True, linger=99, face="chip_blue_face")]),
        dict(name="deterministic", cam="hero", dur=11.0, chips_from="open",
             crop=[[0, 1.0, 0.50, 0.50], [11.0, 0.80, 0.40, 0.47]], chips=[]),
        dict(name="twins", cam="front", dur=14.0, crop=front, chips_from="twins", chips=[]),
        dict(name="mechanism", cam="top", dur=18.0, crop=top, chips_from="zoom", chips=[]),
        dict(name="horizon", cam="top", dur=14.0, crop=top, chips_from="zoom", chips=[]),
        dict(name="crux", cam="front", dur=28.0, crop=front, chips=[]),
        dict(name="islands", cam="top", dur=37.0, crop=top, chips_from="none", chips=[]),
        dict(name="warm", cam="front", dur=17.0, crop=front, slowmo=0.5,
             chips=[dict(path="warm_win", t0=2.5, appear=0.3, hold=True, linger=99, face="chip_face",
                         depth_offset=0.0004),
                    dict(path="warm_replay", t0=2.5, appear=0.3, hold=True, linger=99, face="chip_blue_face")]),
        dict(name="realworld", cam="front", dur=20.0, crop=front, chips=[]),
        dict(name="statistics", cam="front", dur=24.0, crop=front, chips_from="rain", frame_offset=120, chips=[]),
        dict(name="outro", cam="hero", dur=10.0, chips_from="outro",
             crop=[[0, 0.80, 0.40, 0.47], [10.0, 1.0, 0.50, 0.50]], chips=[]),
    ]
    meta = dict(fps=30, hz=60, shots=shots, hook_t_win=T_win)
    json.dump(meta, open(f"{OUT}/shots.json", "w"), indent=1)
    print([(s["name"], s["dur"]) for s in shots], "total", sum(s["dur"] for s in shots))


if __name__ == "__main__":
    main()
