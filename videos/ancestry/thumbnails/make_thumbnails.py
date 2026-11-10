"""The migration film's thumbnails (v2): photographs of the model, one small caption at most (a-d), plus one
big-type contender for the click test (e).

    blender -b every_grain_of_us.blend -P ../../../.claude/skills/sandglass/scripts/thumb_shot.py -- \
        shots.json renders 50 96 ../../../ancestry_sand          # five 1920x1080 stills, ~4 min each on CPU
    python3 make_thumbnails.py renders                            # -> youtube_thumbnail_a..e.png + feed_preview.jpg
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", ".claude", "skills", "sandglass", "scripts"))
import thumb_finish as F
from PIL import Image, ImageDraw, ImageFont

R = sys.argv[1] if len(sys.argv) > 1 else "renders"
OUT = os.path.join(HERE, "..")

THUMBS = [   # file, render, finish, caption (text, xy) or None, paired title
    ("youtube_thumbnail_a_leaving.png", "exit", dict(sat=0.95, vignette=0.34, exposure=1.08, halation=0.14),
     ("c. 55,000 years ago", (60, 658)), "Where everyone came from"),
    ("youtube_thumbnail_b_1903.png", "streams", dict(sat=0.95, vignette=0.30, exposure=1.04),
     ("1903", (60, 658)), "I simulated 200,000 years of human migration with sand"),
    ("youtube_thumbnail_c_today.png", "today", dict(sat=0.95, vignette=0.30, exposure=1.04),
     ("1 grain = about 56,000 people", (60, 658)), "Every person alive, as 145,000 grains of sand"),
    ("youtube_thumbnail_d_first.png", "first", dict(sat=0.95, vignette=0.30, exposure=1.06),
     None, "200,000 years of people, one grain at a time"),
]

made = []
for name, src, kw, cap, title in THUMBS:
    im = F.photo(F.load(os.path.join(R, src + ".png")), **kw)
    if cap:
        F.label(im, cap[1], cap[0], F.mono(36))
    im.save(os.path.join(OUT, name), optimize=True)
    made.append((im, title))
    print("wrote", name)

# E: the big-type contender, built to win a click test: one promise in four words, set huge in condensed
# caps over a darkened band of empty table; the accent is the teal of the flying sand, not stock yellow.
im = F.photo(F.load(os.path.join(R, "bigtext.png")), sat=1.0, vignette=0.26, exposure=1.08, halation=0.12)
im = F.shade(im, top=330, strength=0.72)
b = F.label(im, (50, 8), "200,000 YEARS", F.headline(196), fill=(250, 248, 242), shadow=0.75)
F.label(im, (54, b[3] + 6), "IN 4 MINUTES", F.headline(98), fill=(132, 236, 216), shadow=0.75)
im.save(os.path.join(OUT, "youtube_thumbnail_e_bigtext.png"), optimize=True)
made.insert(0, (im, "I simulated human migration with 145,000 grains of sand"))
print("wrote youtube_thumbnail_e_bigtext.png")

# how they look in a phone feed (thumbnail ~360 px wide) with the paired title; plain layout, no site branding
TW, TH, PAD = 400, 225, 28
font = ImageFont.truetype("/usr/share/fonts/opentype/inter/Inter-SemiBold.otf", 17)
rows = (len(made) + 1) // 2
sheet = Image.new("RGB", (PAD + 2 * (TW + PAD), PAD + rows * (TH + 70)), (15, 15, 15))
d = ImageDraw.Draw(sheet)
for i, (im, title) in enumerate(made):
    x, y = PAD + (i % 2) * (TW + PAD), PAD + (i // 2) * (TH + 70)
    sheet.paste(im.resize((TW, TH), Image.LANCZOS), (x, y))
    words, line, lines = title.split(), "", []
    for w in words:
        if d.textlength(line + " " + w, font=font) > TW and line:
            lines.append(line); line = w
        else:
            line = (line + " " + w).strip()
    lines.append(line)
    for k, l in enumerate(lines[:2]):
        d.text((x, y + TH + 10 + k * 22), l, font=font, fill=(236, 236, 236))
sheet.save(os.path.join(HERE, "feed_preview.jpg"), quality=90)
print("wrote feed_preview.jpg")
