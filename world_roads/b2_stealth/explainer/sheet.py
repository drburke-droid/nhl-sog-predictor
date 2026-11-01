import sys, glob, os
from PIL import Image, ImageDraw
files = sorted(glob.glob(sys.argv[1] + '/f_*.png')); out = sys.argv[2]; os.makedirs(out, exist_ok=True)
for k in range(0, len(files), 4):
    sheet = Image.new('RGB', (1920, 1080), 'white')
    for j, f in enumerate(files[k:k+4]):
        im = Image.open(f).convert('RGB').resize((956, 538))
        x, y = (j % 2) * 964, (j // 2) * 542
        sheet.paste(im, (x, y)); ImageDraw.Draw(sheet).text((x + 8, y + 8), os.path.basename(f), fill='yellow')
    sheet.save(f'{out}/sheet_{k//4:02d}.png')
