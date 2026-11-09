"""How the thumbnails read where people meet them: a phone feed (about 360 px wide) and the sidebar
(168 px), next to thumbnails in the common gold-on-dark style.

    python3 feed_check.py OUT.png THUMB1 THUMB2 THUMB3 --vs OTHER1 OTHER2 ...
"""
import sys
from PIL import Image, ImageDraw, ImageFont

out = sys.argv[1]
args = sys.argv[2:]
vs_i = args.index("--vs") if "--vs" in args else len(args)
mine, others = args[:vs_i], args[vs_i + 1:]
F = ImageFont.truetype("/usr/share/fonts/opentype/inter/Inter-Medium.otf", 15)
FB = ImageFont.truetype("/usr/share/fonts/opentype/inter/Inter-SemiBold.otf", 17)
BG = (15, 15, 15)

# a dark-mode feed: a column of phone-width cards, ours mixed in among the usual style
feed = []
for i, p in enumerate(mine):
    feed.append((p, True))
    if i < len(others):
        feed.append((others[i], False))
cw, ch = 360, 202
col = Image.new("RGB", (cw + 40, 40 + len(feed) * (ch + 46)), BG)
d = ImageDraw.Draw(col)
y = 20
for p, ours in feed:
    im = Image.open(p).convert("RGB").resize((cw, ch), Image.LANCZOS)
    col.paste(im, (20, y))
    d.text((20, y + ch + 8), "this video" if ours else "the usual style", font=F, fill=(230, 230, 230) if ours else (130, 130, 130))
    y += ch + 46

# the sidebar: tiny
sw, sh = 168, 94
side = Image.new("RGB", (sw + 40, 40 + len(mine) * (sh + 30)), BG)
d2 = ImageDraw.Draw(side)
for i, p in enumerate(mine):
    side.paste(Image.open(p).convert("RGB").resize((sw, sh), Image.LANCZOS), (20, 20 + i * (sh + 30)))

W = col.width + side.width + 40
Hh = max(col.height, side.height) + 40
sheet = Image.new("RGB", (W, Hh), BG)
ImageDraw.Draw(sheet).text((20, 8), "phone feed, actual size", font=FB, fill=(200, 200, 200))
ImageDraw.Draw(sheet).text((col.width + 40, 8), "sidebar", font=FB, fill=(200, 200, 200))
sheet.paste(col, (0, 30))
sheet.paste(side, (col.width + 20, 30))
sheet.save(out)
print("wrote", out)
