"""Portraits for the three companion films (A: single games, B: athletics and swimming, C: five seasons), for the
athletes the earlier films didn't need. The same compliant fetch as ../singlegame/film/photos_sg.py (article infobox
image, or the Commons category when the infobox image isn't on Commons; licence and author from the Commons file page;
a standard 500 px thumbnail; image requests a minute apart). An athlete with no freely licensed photo gets none.

    python photos3.py fetch [key ...]   # -> ../cache/three/photos_raw/ and meta.json
    python photos3.py crop [key ...]    # -> photos/<key>.png (4:5), photos/cut/<key>.png, photos/credits.json
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "singlegame", "film"))
sys.argv = [sys.argv[0], "noop"] + sys.argv[1:]            # photos_sg reads its command from argv[1]
import photos_sg as S                                       # noqa: E402
sys.argv = [sys.argv[0]] + sys.argv[2:]

S.P.RAW = os.path.join(HERE, "..", "cache", "three", "photos_raw")
S.HERE = HERE                                               # crops go to ./photos
S.NEW = {"feller": ("Bob Feller", "Bob_Feller"), "wills": ("Maury Wills", "Maury_Wills"),
         "williams": ("Ted Williams", "Ted_Williams"), "duplantis": ("Armand Duplantis", "Armand_Duplantis"),
         "mcintosh": ("Summer McIntosh", "Summer_McIntosh"), "bolt": ("Usain Bolt", "Usain_Bolt"),
         "vanbrocklin": ("Norm Van Brocklin", "Norm_Van_Brocklin"), "bottomley": ("Jim Bottomley", "Jim_Bottomley"),
         "box": ("Cloyce Box", "Cloyce_Box"), "whiten": ("Mark Whiten", "Mark_Whiten"),
         "sittler": ("Darryl Sittler", "Darryl_Sittler"), "olmstead": ("Bert Olmstead", "Bert_Olmstead"),
         "dthomas": ("Derrick Thomas", "Derrick_Thomas")}
S.ONLY = sys.argv[2:] or None
S.CHOICE = {                                                # crop boxes (left, top, width as fractions), set after a look
    "williams": (0.0, 0.0, 1.0), "duplantis": (0.08, 0.0, 0.84), "mcintosh": (0.37, 0.0, 0.45),
    "bolt": (0.01, 0.0, 0.98), "vanbrocklin": (0.1, 0.0, 0.9),
}

FREE = ("CC BY", "CC0", "Public domain")                    # GFDL-only files are passed over (the licence text must travel)


def alternate(key, prefer=()):
    """the first freely licensed (CC BY, CC BY-SA, CC0, public domain) photo in the athlete's Commons category,
    preferring file names that contain any of `prefer`"""
    import json, re, urllib.parse
    name, title = S.NEW[key]
    cat = S.P.get("https://commons.wikimedia.org/wiki/Category:" + urllib.parse.quote(title)).decode("utf-8")
    files = [urllib.parse.unquote(f).replace("_", " ") for f in re.findall(r'href="/wiki/File:([^"]+\.(?:jpe?g|png))"', cat, re.I)]
    files = sorted(dict.fromkeys(files), key=lambda f: -sum(p.lower() in f.lower() for p in prefer))
    for fname in files[:12]:
        m = S.P.commons_meta(fname)
        if not (m["license"] or "").startswith(FREE):
            print("  skip", fname[:60], m["license"], flush=True)
            continue
        url = S.P.standard_thumb(m["original"], 500)
        path = os.path.join(S.P.RAW, key + (os.path.splitext(urllib.parse.urlparse(url).path)[1].lower() or ".jpg"))
        open(path, "wb").write(S.P.get(url))
        mp = os.path.join(S.P.RAW, "meta.json")
        meta = json.load(open(mp))
        m.update(name=name, downloaded=url, path=path)
        meta[key] = m
        json.dump(meta, open(mp, "w"), indent=1, ensure_ascii=False)
        print(key, "|", fname, "|", m["license"], "|", m["author"][:50], flush=True)
        return
    print(key, "no free alternate found", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "alt":
        alternate(sys.argv[2], tuple(sys.argv[3:]))
    else:
        {"fetch": S.fetch, "crop": S.crop}[sys.argv[1]]()
