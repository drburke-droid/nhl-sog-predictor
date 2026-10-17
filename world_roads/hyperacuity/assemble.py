"""Join the hyperacuity sections (cross-fades), and add the score with countdown ticks and answer bells.

    python assemble.py              -> out/hyperacuity.mp4 (4K)
    python assemble.py render       -> render every section first (4 at a time)
"""
import os, sys, json, subprocess
import numpy as np
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "flights"))
sys.path.insert(0, os.path.join(ROOT, "signal"))
OUT = os.path.join(HERE, "out")
SECS = [("hook", "scenes"), ("title", "scenes"), ("rules", "scenes"), ("rounds", "rounds"), ("results", "scenes"),
        ("scale", "explain"), ("aa", "explain"), ("retina", "explain"), ("straight", "scenes"), ("bent", "scenes"),
        ("circle", "scenes"), ("tilt", "scenes"), ("tilt_diag", "scenes"), ("oblique", "scenes"), ("caliper", "scenes"),
        ("amsler", "scenes"), ("amblyopia", "scenes"), ("outro", "scenes")]
XF = 0.5
# final audio: trim, then a peak limiter at -3 dBFS; AAC at 320k keeps the true peak under -1 dBTP
AF = "volume=-1.2dB,alimiter=limit=0.708:attack=3:release=60:level=false"


def section(name, mod):
    m = __import__(mod)
    if mod == "rounds":
        return m.Rounds()
    return m.SECTIONS[name]()


def render_all():
    def one(nm_mod):
        nm, mod = nm_mod
        log = open(os.path.join(OUT, f"{nm}.log"), "w")
        r = subprocess.run([sys.executable, f"{mod}.py", nm] if mod != "rounds" else [sys.executable, "rounds.py"],
                           cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
        print(f"  {nm}: exit {r.returncode}", flush=True)
        return r.returncode
    order = sorted(SECS, key=lambda s: -section(*s).dur)       # longest first
    with ThreadPoolExecutor(4) as ex:
        codes = list(ex.map(one, order))
    assert not any(codes), "a section failed"


def dur(p):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p],
                                capture_output=True, text=True, check=True).stdout)


def sfx(N, sr, events):
    """ticks (a soft wooden click) and bells (two partials, slow decay)"""
    out = np.zeros((2, N), np.float32)
    for t, kind in events:
        i0 = int(t * sr)
        if kind == "tick":
            n = int(0.05 * sr)
            tt = np.arange(n) / sr
            s = (np.sin(2 * np.pi * 1850 * tt) * 0.6 + np.sin(2 * np.pi * 920 * tt) * 0.4) * np.exp(-tt / 0.008) * 0.16
        else:
            n = int(1.4 * sr)
            tt = np.arange(n) / sr
            s = (np.sin(2 * np.pi * 1318.5 * tt) + 0.5 * np.sin(2 * np.pi * 1975.5 * tt) * np.exp(-tt / 0.25)) \
                * np.exp(-tt / 0.45) * np.minimum(1, tt / 0.003) * 0.13
        m = min(n, N - i0)
        if m > 0:
            out[:, i0:i0 + m] += s[:m]
    return out


def main():
    secs = [s for s, _ in SECS]
    paths = [os.path.join(OUT, f"{s}.mp4") for s in secs]
    ds = [dur(p) for p in paths]
    off = [0.0]
    for d in ds[:-1]:
        off.append(off[-1] + d - XF)
    total = off[-1] + ds[-1]
    O = dict(zip(secs, off))
    for s, o, d in zip(secs, off, ds):
        print(f"  {s:10s} {o:7.2f} + {d:6.2f}")
    print(f"total {total:.2f} s ({int(total // 60)}:{total % 60:04.1f})", flush=True)
    vid = os.path.join(OUT, "_video.mp4")
    args = ["ffmpeg", "-y", "-v", "error"]
    for p in paths:
        args += ["-i", p]
    fc, last = [], "0:v"
    for k in range(1, len(secs)):
        fc.append(f"[{last}][{k}:v]xfade=transition=fade:duration={XF}:offset={off[k]:.3f}[v{k}]")
        last = f"v{k}"
    args += ["-filter_complex", ";".join(fc), "-map", f"[{last}]", "-c:v", "libx264", "-preset", "medium", "-crf", "16",
             "-pix_fmt", "yuv420p", "-r", "30", vid]
    subprocess.run(args, check=True)
    # ---- music: calm under the tests, lifting for the results and the ending
    import score as SC
    events = []
    for s, mod in SECS:
        sec = section(s, mod)
        events += [(O[s] + t, k) for t, k in getattr(sec, "sfx", [])]
    r0 = O["rounds"]
    energy = [(0, 0.3), (O["title"] + 2.6, 0.3), (O["title"] + 2.65, 0.55), (O["rules"], 0.45), (r0, 0.35),
              (O["results"] - 0.5, 0.6), (O["results"], 0.7), (O["scale"], 0.45), (O["retina"], 0.5), (O["straight"], 0.5),
              (O["bent"], 0.38), (O["oblique"], 0.5), (O["caliper"], 0.55), (O["amsler"], 0.32), (O["outro"], 0.6),
              (total, 0.0)]
    impacts = [(O["title"] + 2.6, 0.9), (O["results"], 0.8), (O["scale"] + 31.6, 0.6), (O["outro"], 0.7)]
    risers = [(O["title"] + 0.6, O["title"] + 2.6), (O["results"] - 2.0, O["results"])]
    sc = SC.Score(total, energy, impacts, risers, transpose_at=O["scale"], final_at=O["outro"] + 0.5, intro_hit=False)
    sc.compose()
    mix = sc.mix() * 0.8
    mix = mix + sfx(mix.shape[1], SC.SR, events)
    wav = os.path.join(OUT, "score.wav")
    SC.write(mix, wav)
    final = os.path.join(OUT, "hyperacuity.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", vid, "-i", wav, "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                    "-af", AF, "-c:a", "aac", "-b:a", "320k", "-shortest", "-movflags", "+faststart", final], check=True)
    os.remove(vid)
    json.dump({"sections": {s: [O[s], d] for s, d in zip(secs, ds)}, "total": total}, open(os.path.join(OUT, "timeline.json"), "w"), indent=1)
    print("wrote", final)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "render":
        render_all()
    main()
