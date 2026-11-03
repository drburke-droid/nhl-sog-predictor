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

The hockey model carries a "BLENDER KIT" brand text on the puck's side, which suggests the models come from
BlenderKit; check each model's licence there before publishing. The text is unreadable at chart size.
