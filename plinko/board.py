"""Plinko board geometry, prize values and physical parameters.

Everything is in SI units (metres, seconds, radians) unless a name says otherwise.
See ACCURACY.md for where each number comes from and how sure we are of it.

Coordinates: x runs left to right across the board, y runs DOWN the board (gravity is +y).
Slot centres sit at x = 1..9 slot pitches, so the side walls sit at x = 0.5 and 9.5 pitches.
Peg row r (r = 0..12) sits at y = r * ROW_SPACING.
"""
import math
import numpy as np

# ---------------------------------------------------------------- measured / estimated geometry
# Slot pitch: the horizontal distance between neighbouring slot centres (= peg spacing within a row).
# Measured from a near-frontal close-up of the slots (2017 broadcast frame): chip diameter / slot pitch
# = 95 px / 129.7 px = 0.73. With the chip at ~13 cm across, the pitch is ~17.8 cm (7.0 in).
CHIP_DIAMETER = 0.130            # m (5.1 in). Hand-scaled 4K frame: 13.5 cm; replicas sold as 5 in.
CHIP_THICKNESS = 0.019           # m (3/4 in), replica spec; does not enter the 2D dynamics
SLOT_PITCH = CHIP_DIAMETER / 0.73  # m  ~0.178 m = 7.0 in
# Pegs sit on an equilateral triangular lattice (the plexiglass cover is cut as equilateral triangles;
# the close-up frame shows a foreshortened 0.77 ratio, consistent with the board leaning back plus a
# camera looking slightly up). Row spacing = pitch * sqrt(3)/2.
ROW_SPACING = SLOT_PITCH * math.sqrt(3) / 2   # m  ~0.154 m = 6.1 in
PEG_DIAMETER = 0.0095            # m (3/8 in metal pin), estimated from frames
N_ROWS = 13                      # rows 0..12 (Howard CC paper: "thirteen rows of pegs"; counted on frames)
N_SLOTS = 9
# Slot dividers: thin metal plates directly below the 8 pegs of the bottom row.
DIVIDER_TOP_BELOW_LAST_ROW = 0.81 * SLOT_PITCH   # m, from the close-up frame
FLOOR_BELOW_LAST_ROW = 1.23 * SLOT_PITCH         # m, slot floor
DIVIDER_THICKNESS = 0.004        # m
BOARD_TILT_DEG = 10.0            # board leans back; chip slides on the backboard (estimated from side shots)

# ---------------------------------------------------------------- prizes
# Daytime values since 15 Oct 1998 (centre was $5,000 from 1983 to 1998).
SLOT_VALUES = np.array([100, 500, 1000, 0, 10000, 0, 1000, 500, 100], dtype=np.float64)
SLOT_VALUES_1983 = np.array([100, 500, 1000, 0, 5000, 0, 1000, 500, 100], dtype=np.float64)
MAX_CHIPS = 5

G = 9.81


def pegs():
    """All peg centres as an (N, 2) array in metres, plus the row index of each."""
    pts, rows = [], []
    for r in range(N_ROWS):
        y = r * ROW_SPACING
        if r % 2 == 0:   # 8 pegs between slot centres: x = 1.5 .. 8.5 pitches
            xs = [k + 0.5 for k in range(1, 9)]
        else:            # 7 pegs over slot centres 2..8; the wall teeth stand in for 1 and 9
            xs = list(range(2, 9))
        for xk in xs:
            pts.append((xk * SLOT_PITCH, y))
            rows.append(r)
    return np.array(pts), np.array(rows)


def wall_polyline(side):
    """Sawtooth side wall as a list of (x, y) vertices, top to bottom. side = -1 (left) or +1 (right).

    The wall's flat line is at x = 0.5 (9.5) pitches; at every odd row a triangular tooth reaches in to
    x = 1 (9), i.e. to the slot-1 (slot-9) centre line, exactly where a peg would otherwise be. The tooth
    faces follow the lattice directions (60 degrees to horizontal)."""
    base = 0.5 if side < 0 else 9.5
    apex = 1.0 if side < 0 else 9.0
    top = -3.0 * SLOT_PITCH
    pts = [(base * SLOT_PITCH, top)]
    for r in range(N_ROWS):
        x = (apex if r % 2 == 1 else base) * SLOT_PITCH
        pts.append((x, r * ROW_SPACING))
    pts.append((base * SLOT_PITCH, last_row_y() + FLOOR_BELOW_LAST_ROW))
    return pts


def last_row_y():
    return (N_ROWS - 1) * ROW_SPACING


def divider_segments():
    y0 = last_row_y() + DIVIDER_TOP_BELOW_LAST_ROW
    y1 = last_row_y() + FLOOR_BELOW_LAST_ROW
    return [((k + 0.5) * SLOT_PITCH, y0, (k + 0.5) * SLOT_PITCH, y1) for k in range(1, 9)]


def segments():
    """Every straight obstacle as rows of (x0, y0, x1, y1): both walls, then the 8 dividers."""
    segs = []
    for side in (-1, 1):
        p = wall_polyline(side)
        for a, b in zip(p[:-1], p[1:]):
            segs.append((a[0], a[1], b[0], b[1]))
    segs += divider_segments()
    return np.array(segs, dtype=np.float64)


def drop_x(position):
    """Release x for drop position 1..9 (directly above the slot of the same number)."""
    return position * SLOT_PITCH


def summary():
    pg, _ = pegs()
    inch = 0.0254
    field_h = last_row_y()
    return {
        "chip_diameter_m": CHIP_DIAMETER, "chip_diameter_in": CHIP_DIAMETER / inch,
        "slot_pitch_m": SLOT_PITCH, "slot_pitch_in": SLOT_PITCH / inch,
        "row_spacing_m": ROW_SPACING, "row_spacing_in": ROW_SPACING / inch,
        "playfield_width_m": 9 * SLOT_PITCH, "playfield_width_in": 9 * SLOT_PITCH / inch,
        "peg_rows": N_ROWS, "pegs": int(len(pg)), "wall_teeth_per_side": (N_ROWS - 1) // 2,
        "top_row_to_bottom_row_m": field_h, "top_row_to_bottom_row_in": field_h / inch,
        "top_row_to_slot_floor_m": field_h + FLOOR_BELOW_LAST_ROW,
        "top_row_to_slot_floor_in": (field_h + FLOOR_BELOW_LAST_ROW) / inch,
        "peg_gap_m": SLOT_PITCH - PEG_DIAMETER, "chip_to_gap_ratio": CHIP_DIAMETER / (SLOT_PITCH - PEG_DIAMETER),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(summary(), indent=1))
