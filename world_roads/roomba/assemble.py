"""Join the Roomba-photon sections (cross-fades) and add the score.

    python assemble.py              -> out/roomba_photon.mp4
    python assemble.py render       -> render every section first (2 at a time; the USA ones need ~3-5 GB each)
    python assemble.py short        -> out/roomba_short.mp4 (after: python short.py)
"""
import os, sys, json, subprocess
import numpy as np
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "flights"))
sys.path.insert(0, os.path.join(ROOT, "signal"))
OUT = os.path.join(HERE, "out")
SECS = ["room", "rule", "nyc_intro", "nyc_run", "usa_intro", "usa_run", "last", "runs", "payoff"]
XF = 0.5
# final audio: trim, then a peak limiter at -3 dBFS; AAC at 320k keeps the true peak under -1 dBTP
AF = "volume=-1.2dB,alimiter=limit=0.708:attack=3:release=60:level=false"


def render_all(names=SECS, jobs=2):
    def one(nm):
        log = open(os.path.join(OUT, f"{nm}.log"), "w")
        r = subprocess.run([sys.executable, "video.py", nm], cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
        print(f"  {nm}: exit {r.returncode}", flush=True)
        return r.returncode
    heavy = ["usa_run", "last", "usa_intro", "nyc_run"]
    order = [n for n in heavy if n in names] + [n for n in names if n not in heavy]
    with ThreadPoolExecutor(jobs) as ex:
        codes = list(ex.map(one, order))
    assert not any(codes), "a section failed"


def dur(p):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p],
                                capture_output=True, text=True, check=True).stdout)


def bells(N, sr, times):
    """a soft bell (two partials, slow decay) at each time"""
    out = np.zeros((2, N), np.float32)
    n = int(1.6 * sr)
    tt = np.arange(n) / sr
    s = (np.sin(2 * np.pi * 1318.5 * tt) + 0.5 * np.sin(2 * np.pi * 1975.5 * tt) * np.exp(-tt / 0.25)) \
        * np.exp(-tt / 0.5) * np.minimum(1, tt / 0.003) * 0.14
    for t in times:
        i0 = int(t * sr)
        m = min(n, N - i0)
        if m > 0:
            out[:, i0:i0 + m] += s[:m]
    return out


def main():
    import video as V
    paths = [os.path.join(OUT, f"{s}.mp4") for s in SECS]
    ds = [dur(p) for p in paths]
    off = [0.0]
    for d in ds[:-1]:
        off.append(off[-1] + d - XF)
    total = off[-1] + ds[-1]
    O = dict(zip(SECS, off))
    for s, o, d in zip(SECS, off, ds):
        print(f"  {s:10s} {o:7.2f} + {d:6.2f}")
    print(f"total {total:.2f} s ({int(total // 60)}:{total % 60:04.1f})", flush=True)
    vid = os.path.join(OUT, "_video.mp4")
    args = ["ffmpeg", "-y", "-v", "error"]
    for p in paths:
        args += ["-i", p]
    fc, last = [], "0:v"
    for k in range(1, len(SECS)):
        fc.append(f"[{last}][{k}:v]xfade=transition=fade:duration={XF}:offset={off[k]:.3f}[v{k}]")
        last = f"v{k}"
    args += ["-filter_complex", ";".join(fc), "-map", f"[{last}]", "-c:v", "libx264", "-preset", "medium", "-crf", "16",
             "-pix_fmt", "yuv420p", "-r", "30", vid]
    subprocess.run(args, check=True)
    # ---- music: light for the room, building through each run, a hush for the zoom, the biggest hit on the last road
    import score as SC
    nyc_go = O["nyc_intro"] + 10.0                                  # "at the speed of light"
    nyc_end = O["nyc_run"] + V.CityRun("run").T_end
    arrive = O["last"] + V.Last.ARRIVE
    usa_go = O["usa_intro"] + 5.0
    energy = [(0, 0.32), (O["rule"], 0.38), (O["nyc_intro"], 0.42), (nyc_go - 0.05, 0.42), (nyc_go, 0.68),
              (O["nyc_run"] + 30, 0.78), (nyc_end - 5, 0.5), (nyc_end, 0.85), (O["usa_intro"], 0.35), (usa_go - 0.05, 0.35),
              (usa_go, 0.7), (O["usa_run"] + 30, 0.8), (O["last"], 0.85), (O["last"] + 1.0, 0.3), (arrive - 0.05, 0.3),
              (arrive, 1.0), (O["runs"], 0.6), (O["payoff"], 0.5), (O["payoff"] + 17.0, 0.55), (total, 0.0)]
    impacts = [(O["rule"], 0.5), (nyc_go, 0.8), (nyc_end, 0.9), (usa_go, 0.8), (arrive, 1.3), (O["payoff"] + 7.2, 0.6),
               (O["payoff"] + 17.3, 0.6)]
    risers = [(nyc_go - 4.0, nyc_go), (nyc_end - 4.0, nyc_end), (O["usa_intro"] + 2.5, usa_go), (arrive - 6.0, arrive)]
    sc = SC.Score(total, energy, impacts, risers, transpose_at=arrive, final_at=O["payoff"] + 17.3, intro_hit=False)
    sc.compose()
    mix = sc.mix() * 0.8
    mix = mix + bells(mix.shape[1], SC.SR, [nyc_end, arrive])
    wav = os.path.join(OUT, "score.wav")
    SC.write(mix, wav)
    final = os.path.join(OUT, "roomba_photon.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", vid, "-i", wav, "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                    "-af", AF, "-c:a", "aac", "-b:a", "320k", "-shortest", "-movflags", "+faststart", final], check=True)
    os.remove(vid)
    json.dump({"sections": {s: [O[s], d] for s, d in zip(SECS, ds)}, "total": total}, open(os.path.join(OUT, "timeline.json"), "w"), indent=1)
    print("wrote", final)


def mux_short(vid, final, go, arrive, end):
    """a Short's own score: a lift at `go`, the hit and bell at `arrive`, the end card at `end`"""
    import score as SC
    total = dur(vid)
    energy = [(0, 0.45), (go - 0.05, 0.45), (go, 0.75), (arrive - 6, 0.85), (arrive - 1.0, 0.4), (arrive - 0.05, 0.4),
              (arrive, 1.0), (end, 0.7), (total, 0.0)]
    impacts = [(go, 0.8), (arrive, 1.3), (end, 0.6)]
    risers = [(max(0.3, go - 2.0), go), (arrive - 4.0, arrive)]
    sc = SC.Score(total, energy, impacts, risers, transpose_at=arrive, final_at=end + 0.3, intro_hit=False)
    sc.compose()
    mix = sc.mix() * 0.8
    mix = mix + bells(mix.shape[1], SC.SR, [arrive])
    wav = final.replace(".mp4", ".wav")
    SC.write(mix, wav)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", vid, "-i", wav, "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                    "-af", AF, "-c:a", "aac", "-b:a", "320k", "-shortest", "-movflags", "+faststart", final], check=True)
    print("wrote", final)


def short():
    """the vertical USA Short"""
    import short as S
    import video as V
    mux_short(os.path.join(OUT, "roomba_short_video.mp4"), os.path.join(OUT, "roomba_short.mp4"),
              go=S.HOOK, arrive=S.LAST0 + V.Last.ARRIVE, end=S.END)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "short":
        short()
        sys.exit()
    if len(sys.argv) > 1 and sys.argv[1] == "render":
        render_all(sys.argv[2:] or SECS)
    main()
