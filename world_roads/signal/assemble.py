"""Join a long video's sections (cross-fades) and score it.

    python assemble.py v1   -> out/long/signal_fires_world.mp4  (one fire around the world)
    python assemble.py v2   -> out/long/signal_fires_ai.mp4     (if an AI took over every phone: warn every capital by fire)
"""
import os, sys, json, subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "flights"))
OUT = os.path.join(HERE, "out", "long")
VIDEOS = {"v1": (["hook", "history", "fpv", "physics", "rules", "journey", "outro"], "signal_fires_world"),
          "v2": (["cold", "fpv2", "plan", "wave", "outro2"], "signal_fires_ai")}
XF = 0.6


def path(name):
    return os.path.join(HERE, "out", "fpv.mp4") if name == "fpv" else os.path.join(OUT, f"{name}.mp4")


def dur(p):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p],
                                capture_output=True, text=True, check=True).stdout)


def fpv2():
    """Video 2's first-person sequence: fpv.mp4 up to the rise, then the rise with Video 2's caption."""
    from fpv import ALT_FROM, FPS
    out = path("fpv2")
    tail = os.path.join(HERE, "out", "fpv_alt_tail.mp4")
    if os.path.exists(out) and os.path.getmtime(out) > os.path.getmtime(tail):
        return
    n = int(round(ALT_FROM * FPS))
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path("fpv"), "-i", tail, "-filter_complex",
                    f"[0:v]trim=end_frame={n},setpts=PTS-STARTPTS[a];[1:v]setpts=PTS-STARTPTS[b];[a][b]concat=n=2:v=1[v]",
                    "-map", "[v]", "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p", "-r", "30", out],
                   check=True)


def static(N, t1, sr):
    """Video 2's NO SIGNAL: band-passed TV static that cuts out at t1."""
    from scipy.signal import butter, sosfilt
    n = int(t1 * sr)
    rng = np.random.default_rng(7)
    x = sosfilt(butter(2, [300, 6000], "bp", fs=sr, output="sos"), rng.standard_normal((2, n)))
    crackle = (rng.random((2, n)) > 0.9993) * rng.standard_normal((2, n)) * 6
    env = np.minimum(1, np.arange(n) / (0.05 * sr)) * (0.8 + 0.2 * np.sin(np.arange(n) / sr * 2 * np.pi * 7))
    out = np.zeros((2, N), np.float32)
    out[:, :n] = (x * 0.09 + crackle * 0.05) * env
    return out


def timeline(secs):
    ds = [dur(path(s)) for s in secs]
    off = [0.0]
    for d in ds[:-1]:
        off.append(off[-1] + d - XF)
    total = off[-1] + ds[-1]
    for s, o, d in zip(secs, off, ds):
        print(f"  {s:8s} {o:7.2f} + {d:6.2f}")
    print(f"total {total:.2f} s ({int(total // 60)}:{total % 60:04.1f})", flush=True)
    return dict(zip(secs, off)), ds, total


def cues_v1(O, total):
    from long_globe import Journey
    J = Journey()
    oj = O["journey"]
    bering = [oj + J.t_lit[i] for i in range(1, J.n_out) if J.country[i - 1] == "Russia" and J.country[i].startswith("USA")]
    fpv1, fpv2_ = O["fpv"] + 16.0, O["fpv"] + 29.6
    energy = [(0, 0.62), (O["history"], 0.62), (O["history"] + 1.0, 0.26), (O["fpv"], 0.26), (fpv1 - 0.1, 0.3), (fpv1, 0.5),
              (fpv2_ - 0.1, 0.5), (fpv2_, 0.6), (O["physics"] - 1, 0.6), (O["physics"], 0.3), (O["rules"], 0.48),
              (oj, 0.76), (oj + J.t_dead - 0.05, 0.84), (oj + J.t_dead, 0.24), (oj + J.t_ships, 0.24), (oj + J.t_ships + 0.5, 0.88),
              (oj + J.t_loop, 0.95), (oj + J.t_loop + 3, 0.7), (O["outro"], 0.7), (O["outro"] + 8.3, 0.85), (total, 0.0)]
    impacts = [(0.0, 0.8), (O["history"] + 21.4, 0.6), (fpv1, 0.9), (fpv2_, 0.9), (O["physics"] + 27.0, 0.6), (oj, 0.8)] + \
              [(b, 1.0) for b in bering] + [(oj + J.t_dead, 1.2), (oj + J.t_ships + 0.5, 0.9), (oj + J.t_loop, 1.3),
                                           (O["outro"] + 8.3, 0.9)]
    risers = [(fpv1 - 2.5, fpv1), (oj + J.t_dead - 2.5, oj + J.t_dead), (oj + J.t_ships - 2.0, oj + J.t_ships + 0.5),
              (oj + J.t_loop - 3.0, oj + J.t_loop)]
    return dict(energy=energy, impacts=impacts, risers=risers, transpose_at=oj + J.t_ships + 0.5, final_at=O["outro"] + 0.5)


def cues_v2(O, total):
    from long_globe import Wave
    T = Wave()
    ow = O["wave"] - T.LEAD                                 # the wave's own clock starts before the section
    fire0, last = ow + T.INTRO, ow + T.t_end
    fpv1, fpv2_ = O["fpv2"] + 16.0, O["fpv2"] + 29.6
    energy = [(0, 0.0), (T.GLITCH, 0.0), (T.GLITCH + 0.05, 0.5), (O["fpv2"], 0.5), (O["fpv2"] + 1.0, 0.26),
              (fpv1 - 0.1, 0.3), (fpv1, 0.5), (fpv2_ - 0.1, 0.5), (fpv2_, 0.6), (O["plan"], 0.5), (fire0 - 0.1, 0.5),
              (fire0, 0.84), (last - 20, 0.92), (last, 1.0), (last + 5, 0.55), (last + 10.4, 0.8), (O["outro2"] + 8.3, 0.9),
              (total, 0.0)]
    impacts = [(T.GLITCH, 1.1), (fpv1, 0.9), (fpv2_, 0.9), (O["plan"], 0.6), (fire0, 1.1)] + \
              [(ow + T.tk[c["hop"]], 0.55) for _, c in T.callouts] + [(last, 1.3), (O["outro2"] + 8.3, 0.9)]
    risers = [(fpv1 - 2.5, fpv1), (fire0 - 3.0, fire0), (last - 3.0, last)]
    return dict(energy=energy, impacts=impacts, risers=risers, transpose_at=fire0, final_at=O["outro2"] + 0.5,
                glitch=T.GLITCH)


def main(which):
    secs, stem = VIDEOS[which]
    if "fpv2" in secs:
        fpv2()
    O, ds, total = timeline(secs)
    # ---- video
    vid = os.path.join(OUT, f"_video_{which}.mp4")
    args = ["ffmpeg", "-y", "-v", "error"]
    for s in secs:
        args += ["-i", path(s)]
    fc, last = [], "0:v"
    off = [O[s] for s in secs]
    for k in range(1, len(secs)):
        lab = f"v{k}"
        fc.append(f"[{last}][{k}:v]xfade=transition=fade:duration={XF}:offset={off[k]:.3f}[{lab}]")
        last = lab
    args += ["-filter_complex", ";".join(fc), "-map", f"[{last}]", "-c:v", "libx264", "-preset", "slow", "-crf", "18",
             "-pix_fmt", "yuv420p", "-r", "30", vid]
    subprocess.run(args, check=True)
    # ---- music
    import score as SC
    cue = (cues_v1 if which == "v1" else cues_v2)(O, total)
    sc = SC.Score(total, cue["energy"], cue["impacts"], cue["risers"], transpose_at=cue["transpose_at"], final_at=cue["final_at"],
                  intro_hit=which == "v1")
    sc.compose()
    mix = sc.mix()
    if "glitch" in cue:
        mix = mix + static(mix.shape[1], cue["glitch"], SC.SR)
    wav = os.path.join(OUT, f"score_{which}.wav")
    SC.write(mix, wav)
    final = os.path.join(OUT, f"{stem}.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", vid, "-i", wav, "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                    "-af", "volume=-1.2dB", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", final],
                   check=True)                                      # headroom: AAC overshoots the -1.5 dB true peak
    os.remove(vid)
    json.dump({"sections": {s: [O[s], d] for s, d in zip(secs, ds)}, "total": total},
              open(os.path.join(OUT, f"timeline_{which}.json"), "w"), indent=1)
    print("wrote", final)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "v1")
