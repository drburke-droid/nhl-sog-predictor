"""One entry per slide (page) of every scene: time in the cut, narration spoken over it, text drawn on it, a thumbnail.

    python3 slides_extract.py <dir>    # then: python3 slides_text.py <dir>  -> script/narration_by_slide.txt
"""
import importlib, json, os, re, subprocess, sys
sys.path.insert(0, '.')
from PIL import ImageDraw
from engine import OUT, SCENE_FILES
from kit import TR, PagedScene
from render import module_for

DST = sys.argv[1]
X = 0.5
REC = []
_orig = ImageDraw.ImageDraw.text
def rec_text(self, xy, text, *a, **k):
    if isinstance(text, str) and text.strip():
        REC.append(text.strip())
    return _orig(self, xy, text, *a, **k)
ImageDraw.ImageDraw.text = rec_text

def probe(path):
    return float(subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=duration', '-of', 'csv=p=0', path],
                                capture_output=True, text=True, check=True).stdout.strip())
durs = [probe(os.path.join(OUT, s + '.mp4')) for s in SCENE_FILES]
starts = [sum(d - X for d in durs[:k]) for k in range(len(durs))]

def clean(strings, final):
    norm = lambda s: re.sub(r'\d', '#', s)
    fk = {norm(s) for s in final}
    keep = []
    for s in strings:
        if norm(s) in fk and s not in final:
            continue
        if any(o != s and o.startswith(s) for o in strings):
            continue
        if s not in keep:
            keep.append(s)
    return keep

out = []
for k, name in enumerate(SCENE_FILES):
    sc = importlib.import_module(module_for(name)).SCENES[name]()
    words = sc.words
    wt = [sc.word_time(i) for i in range(len(words))]
    if isinstance(sc, PagedScene):
        bounds = [(p.t0 if i else 0.0, sc.pages[i + 1].t0 if i + 1 < len(sc.pages) else sc.dur) for i, p in enumerate(sc.pages)]
        draw = lambda i, t: sc.draw_page(i, t)
        first = lambda i: (bounds[i][0] + (TR if i else 0) + 0.05)
    else:  # the promise: title, the orange squares and dice, the five decisions
        c = sc.cue
        cuts = [0.0, c('Buy the orange') - 0.4, c('What you do choose'), sc.dur]
        bounds = list(zip(cuts[:-1], cuts[1:]))
        draw = lambda i, t: sc.frame(t)
        first = lambda i: bounds[i][0] + 0.6
    for i, (a, b) in enumerate(bounds):
        ws = [w for w, t in zip(words, wt) if (a <= t < b) or (i == 0 and t < a)]
        t_end = b - 0.12
        samples = [first(i) + (t_end - first(i)) * f for f in (0, 0.25, 0.5, 0.75)] + [t_end]
        seen, final = [], []
        for j, t in enumerate(samples):
            REC.clear()
            img = draw(i, max(a, min(t, t_end)))
            seen += REC
            if j == len(samples) - 1:
                final = list(REC)
                img.convert('RGB').resize((960, 540)).save(os.path.join(DST, '%s_%02d.jpg' % (name, i + 1)), quality=82)
        out.append({'scene': name, 'n': i + 1, 'start': starts[k] + a, 'end': starts[k] + b, 'narration': ' '.join(ws),
                    'onscreen': clean(seen, final), 'image': '%s_%02d.jpg' % (name, i + 1)})
    print(name, len(bounds), 'slides', flush=True)
json.dump(out, open(os.path.join(DST, 'slides.json'), 'w'), indent=1)
print(len(out), 'slides in all')
