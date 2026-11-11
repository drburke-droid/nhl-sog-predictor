"""
Starting arrangements for the "satisfying blending" bowls: squares and 16x16
pixel art, laid over the bowl like a stencil.  Each letter is one powder:

  R paprika red   Y turmeric yellow   G matcha green   B butterfly-pea blue
  W sugar white   N cocoa brown       P beetroot pink  K charcoal

    python bowl_sim.py --tool whisk --load art:face --grain 1.5
"""
import math

import numpy as np

LETTERS = "RYGBWNPK"            # index = position in build_scene.PALETTE

FACE = """
BBBBBBBBBBBBBBBB
BBBBBYYYYYYBBBBB
BBBYYYYYYYYYYBBB
BBYYYYYYYYYYYYBB
BBYYYKKYYKKYYYBB
BYYYYKKYYKKYYYYB
BYYYYKKYYKKYYYYB
BYYYYYYYYYYYYYYB
BYYYYYYYYYYYYYYB
BYYKYYYYYYYYKYYB
BYYKKYYYYYYKKYYB
BBYYKKKKKKKKYYBB
BBYYYKKKKKKYYYBB
BBBYYYYYYYYYYBBB
BBBBBYYYYYYBBBBB
BBBBBBBBBBBBBBBB
"""

TREE = """
BBBBBBBBBBBBBBBB
BBBBBBGGGGBBBBBB
BBBBGGGGGGGGBBBB
BBBGGGRGGGGGGBBB
BBGGGGGGGGRGGGBB
BBGGGGGGGGGGGGBB
BBGGRGGGGGGGGGBB
BBGGGGGGGRGGGGBB
BBBGGGGGGGGGGBBB
BBBBGGGNNGGGBBBB
BBBBBBBNNBBBBBBB
BBBBBBBNNBBBBBBB
BBBBBBBNNBBBBBBB
YYYYYYNNNNYYYYYY
YYYYYYYYYYYYYYYY
YYYYYYYYYYYYYYYY
"""

HEART = """
WWWWWWWWWWWWWWWW
WWWWWWWWWWWWWWWW
WWWRRRRWWRRRRWWW
WWRRRRRRRRRRRRWW
WRRPPRRRRRRRRRRW
WRRPRRRRRRRRRRRW
WRRRRRRRRRRRRRRW
WRRRRRRRRRRRRRRW
WWRRRRRRRRRRRRWW
WWWRRRRRRRRRRWWW
WWWWRRRRRRRRWWWW
WWWWWRRRRRRWWWWW
WWWWWWRRRRWWWWWW
WWWWWWWRRWWWWWWW
WWWWWWWWWWWWWWWW
WWWWWWWWWWWWWWWW
"""

HOUSE = """
BBBBBBBBBBBBBBBB
BBBBBBBBBBBBBBBB
BBBBBBBRRBBBBBBB
BBBBBBRRRRBBBBBB
BBBBBRRRRRRBBBBB
BBBBRRRRRRRRBBBB
BBBRRRRRRRRRRBBB
BBRRRRRRRRRRRRBB
BBBWWWWWWWWWWBBB
BBBWKKWWWWNNWBBB
BBBWKKWWWWNNWBBB
BBBWWWWWWWNNWBBB
BBBWWWWWWWNNWBBB
GGGGGGGGGGGGGGGG
GGGGGGGGGGGGGGGG
GGGGGGGGGGGGGGGG
"""

FLOWER = """
WWWWWWWWWWWWWWWW
WWWWWWPPPPWWWWWW
WWWWWPPPPPPWWWWW
WWWPPPPPPPPPPWWW
WWPPPPPYYPPPPPWW
WWPPPPYYYYPPPPWW
WWPPPPYYYYPPPPWW
WWWPPPPYYPPPPWWW
WWWWWPPPPPPWWWWW
WWWWWWPPPPWWWWWW
WWWWWWWGGWWWWWWW
WWWWGGWGGWWWWWWW
WWWWWGGGGWGGWWWW
WWWWWWWGGGGWWWWW
WWWWWWWGGWWWWWWW
WWWWWWWGGWWWWWWW
"""


# Original 8-bit-style art, drawn for this film (no existing game's sprites).
ROCKET = """
KKKKKKKKKKKKKKKK
KKKKKKKRRKKKKWKK
KKWKKKRRRRKKKKKK
KKKKKKRRRRKKKKKK
KKKKKWWWWWWKKKKK
KKKKKWWBBWWKKKKK
KWKKKWWBBWWKKKKK
KKKKKWWWWWWKKKWK
KKKKKWWWWWWKKKKK
KKKKRWWWWWWRKKKK
KKKRRWWWWWWRRKKK
KKKRRKYYYYKRRKKK
KKKKKKYYYYKKKKKK
KKWKKKKYYKKKKKKK
KKKKKKKYYKKKWKKK
KKKKKKKKKKKKKKKK
"""

ALIEN = """
KKKKKKKKKKKKKKKK
KKKKGKKKKKKGKKKK
KKKKKGKKKKGKKKKK
KKKKKGGGGGGKKKKK
KKKKGGGGGGGGKKKK
KKKGGGWWWWGGGKKK
KKKGGWWWWWWGGKKK
KKKGGWWKKWWGGKKK
KKKGGWWKKWWGGKKK
KKKGGGWWWWGGGKKK
KKKGGGGGGGGGGKKK
KKKGGPGGGGPGGKKK
KKKKGGPPPPGGKKKK
KKKKGGKKKKGGKKKK
KKKPPPKKKKPPPKKK
KKKKKKKKKKKKKKKK
"""

FLOPPY = """
YYYYYYYYYYYYYYYY
YYYYYYYYYYYYYYYY
YYBBBBBBBBBBYYYY
YYBBKKKKKKBBBYYY
YYBBKKKKWKBBBBYY
YYBBKKKKWKBBBBYY
YYBBKKKKKKBBBBYY
YYBBBBBBBBBBBBYY
YYBBBBBBBBBBBBYY
YYBWWWWWWWWWWBYY
YYBWWWWWWWWWWBYY
YYBWRRRRRRRRWBYY
YYBWWWWWWWWWWBYY
YYBWWWWWWWWWWBYY
YYYYYYYYYYYYYYYY
YYYYYYYYYYYYYYYY
"""

POTION = """
BBBBBBBBBBBBBBBB
BBBBBBNNNNBBBBBB
BBBBBBNNNNBBBBBB
BBBBBBWWWWBBBBBB
BBBBBBWWWWBBBBBB
BBBBBWWWWWWBBBBB
BBBBWWWWWWWWBBBB
BBBWWPPPPPPWWBBB
BBWWPPPPPPPPWWBB
BBWPPPWPPPPPPWBB
BBWPPWPPPPPPPWBB
BBWPPPPPPPPPPWBB
BBWWPPPPPPPPWWBB
BBBWWPPPPPPWWBBB
BBBBWWWWWWWWBBBB
BBBBBBBBBBBBBBBB
"""


# The channel's own logo, read off its 11 x 7 grid: a question mark and a solid block.
LOGO = """
.###..#####
#...#.#####
....#.#####
...#..#####
..#...#####
......#####
..#...#####
"""

# The YouTube icon (a trademark of Google LLC), shown to point at the platform.
YOUTUBE = """
WWWWWWWWWWWWWWWW
WWWWWWWWWWWWWWWW
WWWWWWWWWWWWWWWW
WWRRRRRRRRRRRRWW
WRRRRRRRRRRRRRRW
WRRRRRWRRRRRRRRW
WRRRRRWWRRRRRRRW
WRRRRRWWWRRRRRRW
WRRRRRWWWRRRRRRW
WRRRRRWWRRRRRRRW
WRRRRRWRRRRRRRRW
WRRRRRRRRRRRRRRW
WWRRRRRRRRRRRRWW
WWWWWWWWWWWWWWWW
WWWWWWWWWWWWWWWW
WWWWWWWWWWWWWWWW
"""


def _logo():
    rows = LOGO.strip().split("\n")
    g = np.full((16, 16), LETTERS.index("K"))
    r0, c0 = (16 - len(rows)) // 2, (16 - len(rows[0])) // 2
    for i, row in enumerate(rows):
        for j, ch in enumerate(row):
            if ch == "#":
                g[r0 + i, c0 + j] = LETTERS.index("G")
    return g


def _grid(art):
    rows = [r for r in art.strip().split("\n")]
    assert len(rows) == 16 and all(len(r) == 16 for r in rows), [len(r) for r in rows]
    return np.array([[LETTERS.index(c) for c in r] for r in rows])


def _checker():
    """Classic two-tone chessboard, 8 x 8 squares."""
    g = np.zeros((16, 16), int)
    for i in range(16):
        for j in range(16):
            g[i, j] = LETTERS.index("K" if ((i // 2) + (j // 2)) % 2 else "W")
    return g


def _squares():
    """Squares in six colours; no two neighbours alike."""
    g = np.zeros((16, 16), int)
    for i in range(16):
        for j in range(16):
            g[i, j] = ((i // 2) + 2 * (j // 2)) % 6
    return g


def _rainbow():
    g = np.zeros((16, 16), int)
    bands = [(6.5, 7.7, "R"), (5.4, 6.5, "Y"), (4.3, 5.4, "G"), (3.2, 4.3, "B"), (2.1, 3.2, "P")]
    for i in range(16):
        for j in range(16):
            c = "G" if i >= 12 else "W"
            if i < 12:
                dist = math.hypot(i - 11.5, j - 7.5)
                for lo, hi, col in bands:
                    if lo <= dist < hi:
                        c = col
            g[i, j] = LETTERS.index(c)
    return g


ART = {
    "checkerboard": ("Checkerboard", _checker()),
    "squares": ("Squares", _squares()),
    "face": ("Smiley", _grid(FACE)),
    "heart": ("Heart", _grid(HEART)),
    "tree": ("Apple tree", _grid(TREE)),
    "house": ("House", _grid(HOUSE)),
    "flower": ("Flower", _grid(FLOWER)),
    "rainbow": ("Rainbow", _rainbow()),
    "rocket": ("Rocket", _grid(ROCKET)),
    "alien": ("Alien", _grid(ALIEN)),
    "floppy": ("Floppy disk", _grid(FLOPPY)),
    "potion": ("Potion", _grid(POTION)),
    "logo": ("Our logo", _logo()),
    "youtube": ("YouTube", _grid(YOUTUBE)),
}


def paint(name, x, y, R):
    """Powder index for grains at (x, y) in a bowl of radius R."""
    g = ART[name][1]
    n = g.shape[0]
    j = np.clip(((x + R) / (2 * R) * n).astype(int), 0, n - 1)
    i = np.clip(((R - y) / (2 * R) * n).astype(int), 0, n - 1)        # row 0 is the far side of the bowl
    return g[i, j].astype(np.int32)
