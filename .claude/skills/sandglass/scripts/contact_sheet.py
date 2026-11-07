"""Tile preview frames into one labelled contact sheet.

    python3 contact_sheet.py OUT.jpg COLUMNS IMG[:LABEL] [IMG[:LABEL] ...]
"""
import sys
from PIL import Image, ImageDraw

out, cols = sys.argv[1], int(sys.argv[2])
items = [(a.split(":", 1) + [""])[:2] for a in sys.argv[3:]]
ims = [Image.open(p).convert("RGB") for p, _ in items]
w, h = ims[0].size
rows = (len(ims) + cols - 1) // cols
sheet = Image.new("RGB", (w * cols, h * rows))
for i, (im, (_, label)) in enumerate(zip(ims, items)):
    im = im.resize((w, h))
    if label:
        d = ImageDraw.Draw(im)
        d.rectangle((0, 0, 8 + 7 * len(label), 22), fill=(0, 0, 0))
        d.text((5, 5), label, fill=(255, 220, 150))
    sheet.paste(im, ((i % cols) * w, (i // cols) * h))
sheet.save(out, quality=88)
print(out, sheet.size)
