"""The film's three camera setups (world coordinates, metres). Shared by the plate and chip renders."""
import scene as S

PLATE_RES = (2880, 1620)        # plates are 1.5x the output, so the compositor can push in up to 1.5x
OUT_RES = (1920, 1080)


def drop_gap_world():
    return S.local_to_world(S.board_to_local(5 * S.W, -0.02, 0.03))


CAMS = {
    "hero": dict(lens=32, loc=(-2.55, -6.9, 2.0), target=(0.25, -0.2, 1.72), fstop=5.6),
    "front": dict(lens=35, loc=(0.0, -7.2, 1.45), target=(0.0, -0.25, 1.45), fstop=8.0, shift=(-0.15, 0.0)),
    "top": dict(lens=75, loc=None, target=None, fstop=2.4),
}


def setup(name):
    c = dict(CAMS[name])
    if name == "top":
        g = drop_gap_world()
        c["target"] = (g.x, g.y, g.z)
        c["loc"] = (g.x - 0.30, g.y - 1.05, g.z + 0.22)
    cam, tgt = S.camera(name=f"cam_{name}", lens=c["lens"], loc=c["loc"], target=c["target"], fstop=c["fstop"])
    cam.data.shift_x, cam.data.shift_y = c.get("shift", (0.0, 0.0))
    return cam, tgt, c
