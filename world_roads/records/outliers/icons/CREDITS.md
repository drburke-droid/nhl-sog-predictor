# Icon models

The four chart icons are renders of 3D models supplied for this film as Blender files (`baseballbat.blend`,
`hockey.blend`, `basketball.blend`, `baseball.blend`). `render_icons.py` opens each with scripts disabled and renders
it with Cycles: an orthographic camera, three soft area lights, a transparent background, and AgX tone mapping. The
results are cropped to the object.

| Sprite | Model | View |
|---|---|---|
| `bat.png` | Simple Baseball Bat | side on, 10° above |
| `puck.png` | Realistic Ice Hockey Puck | 22° above, so stacks read as a pile |
| `basketball.png` | Basket ball | 60° around, 25° above |
| `baseball.png` | baseball_01 | 90° around, 20° above (the curved seam) |

The models come from BlenderKit (the puck carries a "BLENDER KIT" brand text, unreadable at chart size). The film's
owner confirmed on 3 November 2026 that their use here is permitted.
