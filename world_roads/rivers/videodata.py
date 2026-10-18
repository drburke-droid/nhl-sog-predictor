"""What the video draws: the 1,000 rivers' main stems (simplified for the globe, and full detail for close-ups),
their straight lines (great circles from source to mouth), English display names.

    python videodata.py   -> data/vid.npz, data/vid.json
"""
import os, sys, json, re
import numpy as np
import shapely

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
sys.path.insert(0, HERE)

# the names a viewer knows (OpenStreetMap names are local)
KNOWN = {"نهر النيل": "Nile", "Chang Jiang": "Yangtze", "Irtysh": "Ob–Irtysh", "Río Paraná": "Paraná",
         "Saint Lawrence River": "St. Lawrence", "La Loire": "Loire", "La Seine": "Seine", "Дон": "Don",
         "塔里木河": "Tarim", "Mississippi River": "Mississippi–Missouri", "Murray River": "Murray–Darling",
         "Amazon River": "Amazon", "Yenisei River": "Yenisei", "Yellow River": "Yellow River", "Lena River": "Lena",
         "Mekong River": "Mekong", "Niger River": "Niger", "Mackenzie River": "Mackenzie", "ኦሞ / Omo": "Omo",
         "Hongshui River": "Xi (Pearl)", "Rio Tocantins": "Tocantins", "Rio São Francisco": "São Francisco",
         "Río Bravo / Rio Grande": "Rio Grande", "Amudarya": "Amu Darya", "Kolyma River": "Kolyma",
         "Мессояха": "Messoyakha", "Сыалах": "Syalakh", "Чёрная": "Chyornaya", "Сёяха": "Seyakha", "Ярудей": "Yarudey",
         "Торғай": "Torgay", "Кулунда": "Kulunda", "Алазея": "Alazeya", "Чукочья": "Chukochya",
         "Riu Xúquer": "Júcar", "Sungai Sembakung": "Sembakung", "Sungai Walennae": "Walennae", "Sungai Barumun": "Barumun",
         "Sungai Tungkal": "Tungkal", "Salo Lariang": "Lariang", "Batang Baram": "Baram"}
CYR = dict(zip("абвгдеёжзийклмнопрстуфхцчшщъыьэюяғқңөұүһә",
               ["a", "b", "v", "g", "d", "e", "yo", "zh", "z", "i", "y", "k", "l", "m", "n", "o", "p", "r", "s", "t", "u",
                "f", "kh", "ts", "ch", "sh", "shch", "", "y", "", "e", "yu", "ya", "g", "k", "n", "o", "u", "u", "h", "a"]))


def latin(s):
    if all(ord(c) < 0x250 or c in "–—’'" for c in s):
        return s
    if any("Ѐ" <= c <= "ӿ" for c in s):
        out = []
        for c in s:
            lo = c.lower()
            t = CYR.get(lo, c)
            out.append(t.capitalize() if c != lo and t else t)
        return "".join(out)
    from unidecode import unidecode
    return unidecode(s).strip()


def display(r):
    n = r["name"]
    if n in KNOWN:
        return KNOWN[n]
    n = latin(n)
    for suf in (" River", " river", " Rivier", " Creek"):
        if n.endswith(suf) and len(n) > len(suf) + 2:
            n = n[: -len(suf)] if suf != " Creek" else n
    for pre in ("Río ", "Rio ", "Rivière ", "Sungai ", "Riu ", "Oued ", "Wadi "):
        if n.startswith(pre) and pre not in ("Oued ", "Wadi "):
            n = n[len(pre):]
    return n


def gc_points(a, b, n=96):
    from systems import xyz
    pa, pb = xyz(np.array([a[0]]), np.array([a[1]]))[0], xyz(np.array([b[0]]), np.array([b[1]]))[0]
    th = np.arccos(np.clip(pa @ pb, -1, 1))
    u = np.linspace(0, 1, n)[:, None]
    p = (np.sin((1 - u) * th) * pa + np.sin(u * th) * pb) / max(np.sin(th), 1e-12)
    return np.c_[np.degrees(np.arctan2(p[:, 1], p[:, 0])), np.degrees(np.arcsin(np.clip(p[:, 2], -1, 1)))]


def main():
    R = json.load(open(os.path.join(DATA, "rivers.json")))
    z = np.load(os.path.join(DATA, "stems.npz"))
    sxy, soff = z["xy"].astype(np.float64), z["off"]
    g_xy, g_off, f_xy, f_off, c_xy = [], [0], [], [0], []
    meta = []
    for r in R:
        ll = sxy[soff[r["sys_index"]]:soff[r["sys_index"] + 1]]
        g = shapely.get_coordinates(shapely.simplify(shapely.LineString(ll), 0.01))   # ~1 km, for the globe
        g_xy.append(g.astype(np.float32)); g_off.append(g_off[-1] + len(g))
        f_xy.append(ll.astype(np.float32)); f_off.append(f_off[-1] + len(ll))
        c_xy.append(gc_points(r["source"], r["mouth"]).astype(np.float32))
        meta.append(dict(rank=r["rank"], name=r["name"], display=display(r), S=r["sinuosity"], L=r["length_km"],
                         D=r["straight_km"], M=r["meander"], C=r["wander"], grad=r["gradient_m_per_km"],
                         low=r["lowland_share"], lat=r["mouth_lat"], continent=r["continent"], terminus=r["terminus"],
                         source=r["source"], mouth=r["mouth"], lake_km=r["lake_km"], gap_km=r["gap_km"]))
    np.savez(os.path.join(DATA, "vid.npz"), g_xy=np.concatenate(g_xy), g_off=np.array(g_off), f_xy=np.concatenate(f_xy),
             f_off=np.array(f_off), c_xy=np.stack(c_xy))
    json.dump(meta, open(os.path.join(DATA, "vid.json"), "w"), ensure_ascii=False, indent=0)
    print(f"{len(meta)} rivers; globe vertices {g_off[-1]:,}, full {f_off[-1]:,}")
    near = sorted([m for m in meta if abs(m["S"] - np.pi) < 0.2], key=lambda m: abs(m["S"] - np.pi))
    print("near pi:", ", ".join(f"{m['display']} {m['S']:.2f}" for m in near))
    print("first 20:", ", ".join(m["display"] for m in meta[:20]))


if __name__ == "__main__":
    main()
