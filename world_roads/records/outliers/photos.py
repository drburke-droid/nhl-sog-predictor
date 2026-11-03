"""Portraits of the ten athletes (and Katie Ledecky, the method example) from Wikimedia Commons.

    python photos.py fetch     # candidates -> out/photos_raw/, licence and author -> out/photos_raw/meta.json
    python photos.py crop      # chosen crops -> photos/<key>.jpg (greyscale, 4:5) and photos/credits.json

Only free files hosted on Commons are used. Wikimedia's robots.txt disallows /w/api.php for generic agents, so this
reads the article page (/wiki/<title>) for the infobox image, and the Commons file page (/wiki/File:...) for the
licence and author, then downloads the image from upload.wikimedia.org. Requests are sequential, 2 s apart, with a
descriptive User-Agent.
"""
import json, os, re, sys, time, html, urllib.parse, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "out", "photos_raw")
DST = os.path.join(HERE, "photos")
UA = "OutliersFilm/1.0 (https://github.com/drburke-droid/nhl-sog-predictor; portraits for a data video) python-urllib"

ATHLETES = {  # key: (display name, en.wikipedia title)
    "ichiro": ("Ichiro Suzuki", "Ichiro_Suzuki"),
    "ruth": ("Babe Ruth", "Babe_Ruth"),
    "rodman": ("Dennis Rodman", "Dennis_Rodman"),
    "curry": ("Stephen Curry", "Stephen_Curry"),
    "vance": ("Dazzy Vance", "Dazzy_Vance"),
    "zelezny": ("Jan Železný", "Jan_Železný"),
    "gretzky": ("Wayne Gretzky", "Wayne_Gretzky"),
    "dimaggio": ("Joe DiMaggio", "Joe_DiMaggio"),
    "chamberlain": ("Wilt Chamberlain", "Wilt_Chamberlain"),
    "beamon": ("Bob Beamon", "Bob_Beamon"),
    "ledecky": ("Katie Ledecky", "Katie_Ledecky"),
}
_last = [0.0]


def get(url):
    """sequential, >= 5 s apart, cached on disk; on HTTP 429 wait as told (Retry-After) and try again"""
    import hashlib, urllib.error
    cache = os.path.join(RAW, "cache")
    os.makedirs(cache, exist_ok=True)
    cp = os.path.join(cache, hashlib.sha1(url.encode()).hexdigest())
    if os.path.exists(cp):
        return open(cp, "rb").read()
    for attempt in range(4):
        wait = 5.0 - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            _last[0] = time.time()
            open(cp, "wb").write(data)
            return data
        except urllib.error.HTTPError as e:
            _last[0] = time.time()
            if e.code != 429 or attempt == 3:
                raise
            ra = e.headers.get("Retry-After")
            delay = float(ra) if ra and ra.isdigit() else 30.0 * (attempt + 1)
            print(f"  429, waiting {delay:.0f} s", flush=True)
            time.sleep(delay)


def text_of(fragment):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", fragment))).strip()


def commons_meta(fname):
    """licence, author, original URL from the Commons file page"""
    page = get("https://commons.wikimedia.org/wiki/File:" + urllib.parse.quote(fname)).decode("utf-8").replace("&#95;", "_")
    lic = re.findall(r'<span class="licensetpl_short"[^>]*>(.*?)</span>', page, re.S)
    lic = [text_of(l) for l in lic if text_of(l)]
    links = [text_of(l) for l in re.findall(r'<span class="licensetpl_link"[^>]*>(.*?)</span>', page, re.S)]
    pd = "Public domain" if re.search(r'class="[^"]*licensetpl[^"]*"[^>]*>.*?public domain', page, re.S | re.I) else None
    m = re.search(r'id="fileinfotpl_aut"[^>]*>.*?</td>\s*<td[^>]*>(.*?)</td>', page, re.S)
    author = text_of(m.group(1)) if m else ""
    m = re.search(r'id="fileinfotpl_date"[^>]*>.*?</td>\s*<td[^>]*>(.*?)</td>', page, re.S)
    date = text_of(m.group(1))[:60] if m else ""
    m = re.search(r'id="fileinfotpl_src"[^>]*>.*?</td>\s*<td[^>]*>(.*?)</td>', page, re.S)
    source = text_of(m.group(1))[:160] if m else ""
    m = re.search(r'<div class="fullImageLink"[^>]*>\s*<a href="([^"]+)"', page)
    orig = ("https:" + m.group(1)) if m and m.group(1).startswith("//") else (m.group(1) if m else None)
    others = re.findall(r'<a href="((?:https:)?//(?:upload|thumb)\.wikimedia\.org/wikipedia/commons/thumb/[^"]+)" class="mw-thumbnail-link"', page)
    return {"file": fname, "license": lic[0] if lic else pd, "licenses_all": lic, "license_url": links[0] if links else "", "author": author, "date": date,
            "source": source, "original": orig, "thumbs": others,
            "page": "https://commons.wikimedia.org/wiki/File:" + urllib.parse.quote(fname.replace(" ", "_"))}


def fetch():
    os.makedirs(RAW, exist_ok=True)
    meta = {}
    for key, (name, title) in ATHLETES.items():
        art = get("https://en.wikipedia.org/wiki/" + urllib.parse.quote(title)).decode("utf-8")
        box = re.search(r'<table class="infobox.*?</table>', art, re.S)
        cands = re.findall(r'<a href="(?:https://en\.wikipedia\.org)?/wiki/File:([^"]+)"[^>]*>\s*<img[^>]+src="([^"]+)"', box.group(0) if box else art)
        if not cands:
            print(key, "no infobox image")
            continue
        fname = urllib.parse.unquote(cands[0][0]).replace("_", " ")
        if "/wikipedia/commons/" not in cands[0][1]:
            print(key, "infobox image is not on Commons (likely non-free):", fname)
            meta[key] = {"name": name, "file": fname, "skip": "not on Commons"}
            continue
        m = commons_meta(fname)
        m["thumbs"] = [("https:" + t) if t.startswith("//") else t for t in m["thumbs"]]
        url = next((t for t in m["thumbs"] if re.search(r"/(9\d\d|1[0-2]\d\d)px-", t)), None) or m["original"]
        ext = os.path.splitext(urllib.parse.urlparse(url).path)[1].lower() or ".jpg"
        path = os.path.join(RAW, key + ext)
        open(path, "wb").write(get(url))
        m.update(name=name, downloaded=url, path=path)
        meta[key] = m
        print(key, "|", fname, "|", m["license"], "|", m["author"][:60], "|", os.path.getsize(path) // 1000, "kB")
    json.dump(meta, open(os.path.join(RAW, "meta.json"), "w"), indent=1, ensure_ascii=False)


ALTERNATES = {  # era photos considered alongside the infobox image (Commons category pages, /wiki/Category:...)
    "beamon_a": "Bob Beamon 1968.jpg",
    "rodman_a": "Dennis Rodman-Jackson Citizen Patriot-3 February 1989.jpg",
    "gretzky_c": "President Ronald Reagan Greeting Wayne Gretzky at a Luncheon for National Hockey League All Stars - DPLA - 4615e44f8e6f1fff3d3b2a7f12d38637.jpg",
    "gretzky_a": "GretzkyEDM.png", "gretzky_b": "Wgretz.jpg",
    "zelezny_a": "Jan Zelezny.jpg",
}


def fetch_alternates():
    mp = os.path.join(RAW, "meta.json")
    meta = json.load(open(mp))
    for key, fname in ALTERNATES.items():
        if key in meta and os.path.exists(meta[key].get("path", "")) and os.path.getsize(meta[key]["path"]) > 0:
            continue
        m = commons_meta(fname)
        m["thumbs"] = [("https:" + t) if t.startswith("//") else t for t in m["thumbs"]]
        url = next((t for t in m["thumbs"] if re.search(r"/(9\d\d|1[0-2]\d\d)px-", t)), None) or m["original"]
        ext = os.path.splitext(urllib.parse.urlparse(url).path)[1].lower() or ".jpg"
        path = os.path.join(RAW, key + ext)
        data = get(url)
        open(path, "wb").write(data)
        m.update(name=ATHLETES[key.split("_")[0]][0], downloaded=url, path=path)
        meta[key] = m
        json.dump(meta, open(mp, "w"), indent=1, ensure_ascii=False)           # saved after every file
        print(key, "|", fname[:60], "|", m["license"], "|", m["author"][:50], "|", m["date"][:30], flush=True)


# which file to use for each athlete, and the 4:5 crop as fractions of the image (left, top, width)
CHOICE = {
    "ichiro": ("ichiro", (0.10, 0.00, 0.66)), "ruth": ("ruth", (0.22, 0.03, 0.46)),
    "rodman": ("rodman", (0.06, 0.00, 0.86)), "curry": ("curry", (0.10, 0.00, 0.80)),
    "vance": ("vance", (0.00, 0.00, 1.00)), "zelezny": ("zelezny", (0.16, 0.00, 0.84)),
    "gretzky": ("gretzky", (0.00, 0.04, 1.00)), "dimaggio": ("dimaggio", (0.06, 0.00, 0.88)),
    "chamberlain": ("chamberlain", (0.06, 0.00, 0.88)), "beamon": ("beamon_a", (0.13, 0.10, 0.64)),
    "ledecky": ("ledecky", (0.20, 0.04, 0.62)),
}
OUT_W, OUT_H = 480, 600
BG_RGB = (15, 18, 21)                                         # look.BG, #0f1215


def crop():
    from PIL import Image, ImageDraw, ImageOps, ImageEnhance
    meta = json.load(open(os.path.join(RAW, "meta.json")))
    os.makedirs(DST, exist_ok=True)
    credits = {}
    for key, (src, (fl, ft, fw)) in CHOICE.items():
        m = meta[src]
        im = Image.open(m["path"]).convert("RGB")
        W_, H_ = im.size
        w = fw * W_
        h = w * OUT_H / OUT_W
        l, t = fl * W_, ft * H_
        if t + h > H_:
            t = max(0, H_ - h)
        box = tuple(int(round(v)) for v in (l, t, l + w, t + h))
        g = ImageOps.grayscale(im.crop(box)).resize((OUT_W, OUT_H), Image.LANCZOS)
        g = ImageOps.autocontrast(g, cutoff=0.5)
        g = ImageEnhance.Brightness(g).enhance(0.92).convert("RGB")
        mask = Image.new("L", (OUT_W, OUT_H), 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, OUT_W - 1, OUT_H - 1), radius=18, fill=255)
        out = Image.new("RGB", (OUT_W, OUT_H), BG_RGB)
        out.paste(g, (0, 0), mask)
        out.save(os.path.join(DST, key + ".png"))
        author = re.sub(r"(Unknown author\s*)+", "Unknown author", m["author"]).strip() or "Unknown author"
        credits[key] = {"athlete": m["name"], "file": m["file"], "page": m["page"], "author": author,
                        "license": m["license"], "license_url": m.get("license_url", ""), "date": m["date"],
                        "source": m.get("source", ""), "changes": "cropped, converted to greyscale, contrast adjusted",
                        "short": f"Photo: {author}, {m['license']}, via Wikimedia Commons" if m["license"] not in ("Public domain", "CC0")
                                 else f"Photo: {author if author != 'Unknown author' else 'unknown'}, {m['license'].lower() if m['license'] == 'Public domain' else m['license']}, via Wikimedia Commons"}
        print(key, box, credits[key]["short"])
    json.dump(credits, open(os.path.join(DST, "credits.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    {"fetch": fetch, "alternates": fetch_alternates, "crop": crop}[sys.argv[1]]()
