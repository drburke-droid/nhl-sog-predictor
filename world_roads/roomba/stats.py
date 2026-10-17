"""Every number the video and its description quote, in one place.

    python stats.py   -> data/stats.json, and a Markdown summary on stdout
"""
import os, json
import numpy as np
import cover as CV

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
AU = 149_597_870.7


def hm(s):
    h, m = divmod(int(round(s / 60)), 60)
    return f"{h} h {m:02d} min"


def main():
    cal = json.load(open(os.path.join(DATA, "calgary_cover.json")))
    many = json.load(open(os.path.join(DATA, "calgary_cover_many.json")))
    meta = json.load(open(os.path.join(DATA, "usa_meta.json")))
    usa = json.load(open(os.path.join(DATA, "usa_cover.json")))["walks"]
    mp = os.path.join(DATA, "usa_cover_more.json")
    if os.path.exists(mp):
        usa += json.load(open(mp))["walks"]
    assert all(w["done"] for w in usa)
    s = np.array([w["seconds"] for w in usa])
    out = dict(
        calgary=dict(network_km=cal["network_km"], all_km=cal["total_km"], segments=cal["segments"],
                     nonback={k: v for k, v in many["pct_nonback"].items()}, n_nonback=len(many["seconds_nonback"]),
                     plain={k: v for k, v in many["pct_plain"].items()}, n_plain=len(many["seconds_plain"])),
        usa=dict(roads=meta["roads"], junctions=meta["junctions"], km=meta["km"], once_s=meta["km"] / CV.C_KMS,
                 start=meta["start_lonlat"], n=len(s), median_s=float(np.median(s)), min_s=float(s.min()), max_s=float(s.max()),
                 mean_x_network=float(np.mean([w["x_network"] for w in usa])),
                 walks=[dict(seed=w["seed"], seconds=w["seconds"], km=w["km"], x_network=w["x_network"], last=w["last"]) for w in usa]),
        planets_light_s={nm: au * AU / CV.C_KMS for nm, au in (("Saturn", 9.537), ("Uranus", 19.19), ("Neptune", 30.07))},
        planets_km={nm: au * AU for nm, au in (("Saturn", 9.537), ("Uranus", 19.19))})
    json.dump(out, open(os.path.join(DATA, "stats.json"), "w"), indent=1)
    c, u = out["calgary"], out["usa"]
    print(f"Calgary: {c['network_km']:,.0f} km, {c['segments']:,} segments")
    print(f"  Roomba photon, {c['n_nonback']} runs: median {c['nonback']['median']:.2f} s "
          f"(10-90%: {c['nonback']['p10']:.2f}-{c['nonback']['p90']:.2f} s; {c['nonback']['min']:.2f}-{c['nonback']['max']:.2f} s)")
    print(f"  plain random walk, {c['n_plain']} runs: median {c['plain']['median']:.1f} s "
          f"(10-90%: {c['plain']['p10']:.1f}-{c['plain']['p90']:.1f} s; {c['plain']['min']:.1f}-{c['plain']['max']:.1f} s)")
    print(f"USA: {u['km']:,.0f} km, {u['roads']:,} roads, {u['junctions']:,} junctions; once over = {u['once_s']:.1f} s of light")
    print(f"  {u['n']} runs: median {hm(u['median_s'])}, {hm(u['min_s'])} to {hm(u['max_s'])}; average {u['mean_x_network']:.0f}x the network")
    print("\n| Run | Light time | Distance | × network |\n|---:|---:|---:|---:|")
    for w in sorted(u["walks"], key=lambda w: w["seed"]):
        print(f"| {w['seed']} | {hm(w['seconds'])} | {w['km'] / 1e9:.2f} billion km | {w['x_network']:.0f}× |")


if __name__ == "__main__":
    main()
