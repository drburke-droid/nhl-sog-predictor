"""Portraits for the single-day film: the same compliant fetch as ../../outliers/photos.py (article page for the
infobox image, Commons file page for licence and author, a standard-size thumbnail), for the athletes the season
film did not need. Wilt Chamberlain, Bob Beamon, Jan Železný and Katie Ledecky reuse the season film's files.

    python photos_sg.py fetch      # -> ../../cache/singlegame/photos_raw/ and meta.json
    python photos_sg.py crop       # -> photos/<key>.png (4:5), photos/cut/<key>.png (background removed), credits.json
"""
import json, os, re, sys, urllib.parse
HERE = os.path.dirname(os.path.abspath(__file__))
OUTL = os.path.join(HERE, "..", "..", "outliers")
sys.path.insert(0, OUTL)
import photos as P                                         # noqa: E402

P.RAW = os.path.join(HERE, "..", "..", "cache", "singlegame", "photos_raw")


def curl_get(url):
    """the same polite fetch (cached, >= 5 s apart, descriptive User-Agent), through curl"""
    import hashlib, subprocess, time
    cache = os.path.join(P.RAW, "cache")
    os.makedirs(cache, exist_ok=True)
    cp = os.path.join(cache, hashlib.sha1(url.encode()).hexdigest())
    if os.path.exists(cp) and os.path.getsize(cp) > 0:
        return open(cp, "rb").read()
    image = "upload.wikimedia.org" in url
    for attempt in range(4):
        gap = 60.0 if image else 5.0                         # the image servers allow only a request now and then
        wait = gap - (time.time() - (_last_img[0] if image else P._last[0]))
        if wait > 0:
            time.sleep(wait)
        r = subprocess.run(["curl", "-sS", "-L", "-m", "120", "-A", P.UA, "-o", cp, "-w", "%{http_code}", url],
                           capture_output=True, text=True)
        P._last[0] = time.time()
        if image:
            _last_img[0] = time.time()
        if r.stdout.strip() == "200":
            return open(cp, "rb").read()
        if os.path.exists(cp):
            os.remove(cp)
        if r.stdout.strip() != "429" or attempt == 1:
            raise RuntimeError(f"HTTP {r.stdout.strip()} for {url}")
        print("  429; waiting 120 s", flush=True)
        time.sleep(120)


_last_img = [0.0]
P.get = curl_get
NEW = {"bryant": ("Kobe Bryant", "Kobe_Bryant"), "rojas": ("Yulimar Rojas", "Yulimar_Rojas"),
       "oldfield": ("Brian Oldfield", "Brian_Oldfield"), "bubka": ("Sergey Bubka", "Sergey_Bubka"),
       "marchand": ("Léon Marchand", "Léon_Marchand"), "vetter": ("Johannes Vetter", "Johannes_Vetter")}


def fetch():
    for rnd in range(3):                                     # skip what the image servers refuse; come back for it
        missing = fetch_once()
        if not missing:
            return
        print("still missing:", missing, "- trying again in 5 minutes", flush=True)
        import time
        time.sleep(300)


def fetch_once():
    os.makedirs(P.RAW, exist_ok=True)
    mp = os.path.join(P.RAW, "meta.json")
    meta = json.load(open(mp)) if os.path.exists(mp) else {}
    missing = []
    for key, (name, title) in NEW.items():
        if key in meta and os.path.exists(meta[key].get("path", "")):
            continue
        try:
            one(key, name, title, meta, mp)
        except RuntimeError as e:
            print(key, "skipped for now:", e, flush=True)
            missing.append(key)
    return missing


def one(key, name, title, meta, mp):
    if True:
        art = P.get("https://en.wikipedia.org/wiki/" + urllib.parse.quote(title)).decode("utf-8")
        box = re.search(r'<table class="infobox.*?</table>', art, re.S)
        cands = re.findall(r'<a href="(?:https://en\.wikipedia\.org)?/wiki/File:([^"]+)"[^>]*>\s*<img[^>]+src="([^"]+)"',
                           box.group(0) if box else art)
        if not cands or "/wikipedia/commons/" not in cands[0][1]:
            cat = P.get("https://commons.wikimedia.org/wiki/Category:" + urllib.parse.quote(title)).decode("utf-8")
            files = [urllib.parse.unquote(f).replace("_", " ") for f in re.findall(r'href="/wiki/File:([^"]+\.(?:jpe?g|png))"', cat, re.I)]
            if not files:
                print(key, "no free image on Wikipedia or in its Commons category", flush=True)
                return
            print(key, "from the Commons category:", files[:4], flush=True)
            cands = [(files[0], "/wikipedia/commons/")]
        fname = urllib.parse.unquote(cands[0][0]).replace("_", " ")
        m = P.commons_meta(fname)
        url = P.standard_thumb(m["original"], 500)          # a standard size the image servers serve to us
        path = os.path.join(P.RAW, key + (os.path.splitext(urllib.parse.urlparse(url).path)[1].lower() or ".jpg"))
        data = P.get(url)
        open(path, "wb").write(data)
        m.update(name=name, downloaded=url, path=path)
        meta[key] = m
        json.dump(meta, open(mp, "w"), indent=1, ensure_ascii=False)
        print(key, "|", fname, "|", m["license"], "|", m["author"][:50], "|", m["date"][:25], flush=True)


if __name__ == "__main__" and sys.argv[1] == "fetch":
    fetch()


CHOICE = {"bryant": (0.36, 0.0, 0.55), "rojas": (0.05, 0.0, 0.90), "oldfield": (0.15, 0.0, 0.70),
          "bubka": (0.08, 0.0, 0.84), "marchand": (0.05, 0.0, 0.85), "vetter": (0.05, 0.0, 0.90)}


def crop():
    """4:5 greyscale crops (face discs, credits page), rembg cut-outs (cards), credits.json"""
    from PIL import Image
    meta = json.load(open(os.path.join(P.RAW, "meta.json")))
    os.makedirs(os.path.join(HERE, "photos", "cut"), exist_ok=True)
    P.DST = os.path.join(HERE, "photos")
    cpath = os.path.join(HERE, "photos", "credits.json")
    credits = json.load(open(cpath)) if os.path.exists(cpath) else {}
    from rembg import remove, new_session
    sess = new_session("u2net_human_seg")
    for key, (fl, ft, fw) in CHOICE.items():
        if key not in meta:
            continue
        m = meta[key]
        im = Image.open(m["path"]).convert("RGB")
        W_, H_ = im.size
        w = fw * W_
        h = w * P.OUT_H / P.OUT_W
        l, t = fl * W_, ft * H_
        if t + h > H_:
            t = max(0, H_ - h)
        box = tuple(int(round(v)) for v in (l, t, l + w, t + h))
        from PIL import ImageOps, ImageEnhance, ImageDraw
        g = ImageOps.grayscale(im.crop(box)).resize((P.OUT_W, P.OUT_H), Image.LANCZOS)
        g = ImageEnhance.Brightness(ImageOps.autocontrast(g, cutoff=0.5)).enhance(0.92).convert("RGB")
        mask = Image.new("L", (P.OUT_W, P.OUT_H), 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, P.OUT_W - 1, P.OUT_H - 1), radius=18, fill=255)
        out = Image.new("RGB", (P.OUT_W, P.OUT_H), P.BG_RGB)
        out.paste(g, (0, 0), mask)
        out.save(os.path.join(HERE, "photos", key + ".png"))
        remove(im, session=sess, post_process_mask=True).save(os.path.join(HERE, "photos", "cut", key + ".png"))
        author = re.sub(r"(Unknown author\s*)+", "Unknown author", m["author"]).strip() or "Unknown author"
        lic = m["license"] or "see file page"
        credits[key] = {"athlete": m["name"], "file": m["file"], "page": m["page"], "author": author, "license": lic,
                        "license_url": m.get("license_url", ""), "date": m["date"], "source": m.get("source", ""),
                        "changes": "cropped, background removed (rembg), toned in team colours",
                        "short": f"Photo: {author}, {lic}, via Wikimedia Commons"}
        print(key, box, credits[key]["short"], flush=True)
    json.dump(credits, open(cpath, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__" and sys.argv[1] == "crop":
    crop()
