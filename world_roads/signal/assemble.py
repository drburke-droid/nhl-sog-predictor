"""Join the long video's sections (cross-fades) and score it.

    python assemble.py      -> out/long/signal_fires_full.mp4
"""
import os, sys, json, subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "flights"))
OUT = os.path.join(HERE, "out", "long")
SECS = ["hook", "history", "fpv", "physics", "rules", "journey", "tour", "outro"]
XF = 0.6


def path(name):
    return os.path.join(HERE, "out", "fpv.mp4") if name == "fpv" else os.path.join(OUT, f"{name}.mp4")


def dur(p):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p],
                                capture_output=True, text=True, check=True).stdout)


def main():
    ds = [dur(path(s)) for s in SECS]
    off = [0.0]
    for d in ds[:-1]:
        off.append(off[-1] + d - XF)
    total = off[-1] + ds[-1]
    O = dict(zip(SECS, off))
    for s, o, d in zip(SECS, off, ds):
        print(f"  {s:8s} {o:7.2f} + {d:6.2f}")
    print(f"total {total:.2f} s ({int(total // 60)}:{total % 60:04.1f})", flush=True)
    # ---- video
    vid = os.path.join(OUT, "_video.mp4")
    args = ["ffmpeg", "-y", "-v", "error"]
    for s in SECS:
        args += ["-i", path(s)]
    fc, last = [], "0:v"
    for k in range(1, len(SECS)):
        lab = f"v{k}"
        fc.append(f"[{last}][{k}:v]xfade=transition=fade:duration={XF}:offset={off[k]:.3f}[{lab}]")
        last = lab
    args += ["-filter_complex", ";".join(fc), "-map", f"[{last}]", "-c:v", "libx264", "-preset", "slow", "-crf", "18",
             "-pix_fmt", "yuv420p", "-r", "30", vid]
    subprocess.run(args, check=True)
    # ---- music
    import score as SC
    from long_globe import Journey, Tour
    J, T = Journey(), Tour()
    oj, ot = O["journey"], O["tour"]
    bering = [oj + J.t_lit[i] for i in range(1, J.n_out) if J.country[i - 1] == "Russia" and J.country[i].startswith("USA")]
    fpv1, fpv2 = O["fpv"] + 16.0, O["fpv"] + 29.6
    energy = [(0, 0.62), (O["history"], 0.62), (O["history"] + 1.0, 0.26), (O["fpv"], 0.26), (fpv1 - 0.1, 0.3), (fpv1, 0.5),
              (fpv2 - 0.1, 0.5), (fpv2, 0.6), (O["physics"] - 1, 0.6), (O["physics"], 0.3), (O["rules"], 0.48),
              (oj, 0.76), (oj + J.t_dead - 0.05, 0.84), (oj + J.t_dead, 0.24), (oj + J.t_ships, 0.24), (oj + J.t_ships + 0.5, 0.88),
              (oj + J.t_loop, 0.95), (oj + J.t_loop + 3, 0.5), (ot, 0.4), (ot + T.INTRO - 0.1, 0.4), (ot + T.INTRO, 0.84),
              (ot + T.t_end - 20, 0.9), (ot + T.t_end, 1.0), (O["outro"], 1.0), (total, 0.0)]
    impacts = [(0.0, 0.8), (O["history"] + 21.4, 0.6), (fpv1, 0.9), (fpv2, 0.9), (O["physics"] + 27.0, 0.6), (oj, 0.8)] + \
              [(b, 1.0) for b in bering] + [(oj + J.t_dead, 1.2), (oj + J.t_ships + 0.5, 0.9), (oj + J.t_loop, 1.3),
                                           (ot + T.INTRO, 1.0), (ot + T.t_end, 1.2)]
    risers = [(fpv1 - 2.5, fpv1), (oj + J.t_dead - 2.5, oj + J.t_dead), (oj + J.t_ships - 2.0, oj + J.t_ships + 0.5),
              (oj + J.t_loop - 3.0, oj + J.t_loop), (ot + T.INTRO - 3.0, ot + T.INTRO), (ot + T.t_end - 3.0, ot + T.t_end)]
    sc = SC.Score(total, energy, impacts, risers, transpose_at=ot + T.INTRO, final_at=O["outro"] + 0.5)
    sc.compose()
    wav = os.path.join(OUT, "score_full.wav")
    SC.write(sc.mix(), wav)
    final = os.path.join(OUT, "signal_fires_full.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", vid, "-i", wav, "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                    "-af", "volume=-1.2dB", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", final],
                   check=True)                                      # headroom: AAC overshoots the -1.5 dB true peak
    json.dump({"sections": dict(zip(SECS, zip(off, ds))), "total": total}, open(os.path.join(OUT, "timeline.json"), "w"), indent=1)
    print("wrote", final)


if __name__ == "__main__":
    main()
