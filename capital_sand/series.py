"""Every candidate fortune as one time series, from all sources, in nominal $M and in today's money.

Sources, oldest first:
  research/gilded/anchors.csv    1900-1986 estimates (contemporary lists, estates, reconstructions)
  research/forbes87/values.csv   Forbes-reported figures 1987-2001 (world list, Forbes 400, Forbes Global)
  modern.load()                  Forbes annual lists 2002-2023, Forbes real-time daily 2019-2026

    P = build(dl_dir, research_dir)      # {key: dict(name, t=array, nominal=array, today=array, ...)}

Families that Forbes listed as one entry are split evenly among the members named in FAMILY, the way Forbes
itself split them when it listed them individually. Between data points a fortune is interpolated
log-linearly (steady growth), so a fortune that doubled over 4 years grows smoothly rather than in steps.
"""
import csv, os
import numpy as np
import gdp, modern

FAMILY = {   # Forbes family entries -> the individuals we track (equal shares)
    "walton family": ["helen walton", "rob walton", "john walton", "jim walton", "alice walton"],
    "karl theo albrecht": ["karl albrecht", "theo albrecht"],
    "mars family": ["forrest mars jr", "john mars", "jacqueline mars"],
    "forrest edward mars jr john franklyn mars jacqueline mars": ["forrest mars jr", "john mars", "jacqueline mars"],
}


def _dec(y, m):
    return int(y) + ((int(m) if m else 7) - 0.5) / 12


def build(dl, research):
    S = modern.load(dl)
    pts = {}                                 # key -> {t: (nominal $M, source)}

    def add(key, name, t, v, src):
        d = pts.setdefault(key, {"name": name, "pts": {}})
        d["pts"][round(t, 4)] = (v, src)

    for key, p in S.items():
        for t, v in p["annual"].items():
            add(key, p["name"], t, v, "Forbes World's Billionaires")
        if p["daily"] is not None:
            t, v = p["daily"]
            step = max(1, len(t) // 400)       # ~weekly is plenty for a 4-minute film
            idx = np.unique(np.r_[np.arange(0, len(t), step), np.argmax(v), len(t) - 1])
            for i in idx:
                add(key, p["name"], float(t[i]), float(v[i]), "Forbes Real-Time Billionaires")
    f87 = os.path.join(research, "forbes87", "values.csv")
    if os.path.exists(f87):
        for r in csv.DictReader(open(f87)):
            try:
                v = float(r["net_worth_busd"]) * 1000.0
            except ValueError:
                continue
            key = modern.norm(r["name"]); t = _dec(r["year"], r.get("month_if_known"))
            members = FAMILY.get(key)
            src = "Forbes " + {"world": "World's Billionaires", "forbes400": "400", "other": "(other report)"}.get(r["list"], r["list"])
            if members:
                for m in members:
                    add(m, m.title(), t, v / len(members), src + " (family entry, split evenly)")
            else:
                add(key, r["name"], t, v, src)
    gil = os.path.join(research, "gilded", "anchors.csv")
    if os.path.exists(gil):
        for r in csv.DictReader(open(gil)):
            vals = [r.get(k) for k in ("net_worth_musd", "low_musd", "high_musd")]
            try:
                v = float(vals[0]) if vals[0] else np.sqrt(float(vals[1]) * float(vals[2]))
            except (TypeError, ValueError):
                continue
            add(modern.norm(r["name"]), r["name"], int(r["year"]) + 0.5, v, r.get("kind", "estimate") + ": " + r.get("source", "")[:80])
    out = {}
    for key, d in pts.items():
        t = np.array(sorted(d["pts"]))
        v = np.array([d["pts"][x][0] for x in t])
        meta = S.get(key, {})
        out[key] = dict(name=meta.get("name", d["name"]), t=t, nominal=v, today=v * gdp.scale(t),
                        src=[d["pts"][x][1] for x in t], industry=sorted(meta.get("industry", [])),
                        country=meta.get("country"), deceased=meta.get("deceased", False))
    return out


def at(p, t):
    """A fortune in today's money at decimal year t (log-linear between points; 0 outside its span)."""
    tt, v = p["t"], np.maximum(p["nominal"], 1e-3)
    if t < tt[0] or t > tt[-1]:
        return 0.0
    return float(np.exp(np.interp(t, tt, np.log(v))) * gdp.scale(t))
