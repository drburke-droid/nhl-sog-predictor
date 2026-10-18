"""Join the rivers sections (cross-fades) and add the score.

    python assemble.py              -> out/rivers_pi.mp4
    python assemble.py render       -> render every section first (2 at a time)
"""
import os, sys, json, subprocess
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "flights"))
sys.path.insert(0, os.path.join(ROOT, "signal"))
OUT = os.path.join(HERE, "out")
SECS = ["open", "einstein", "stolum", "test", "verdict", "near", "common", "end"]
XF = 0.5
# final audio: trim, then a peak limiter at -3 dBFS; AAC at 320k keeps the true peak under -1 dBTP
AF = "volume=-1.2dB,alimiter=limit=0.708:attack=3:release=60:level=false"


def render_all(names=SECS, jobs=2):
    def one(nm):
        log = open(os.path.join(OUT, f"render_{nm}.log"), "w")
        r = subprocess.run([sys.executable, "video.py", nm], cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
        print(f"  {nm}: exit {r.returncode}", flush=True)
        return r.returncode
    with ThreadPoolExecutor(jobs) as ex:
        codes = list(ex.map(one, names))
    assert not any(codes), "a section failed"


def dur(p):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p],
                                capture_output=True, text=True, check=True).stdout)


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
    # ---- music: the pi theme (music.py), cut to the picture
    import score as SC
    import music
    T = V.Test()
    marks = {"rest": O["test"] + T.t_rest, "notpi": O["test"] + T.t_hold + 0.8, "final": O["end"] + 13.0}
    mix = music.main_track(O, total, marks)
    wav = os.path.join(OUT, "score.wav")
    SC.write(mix, wav)
    final = os.path.join(OUT, "rivers_pi.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", vid, "-i", wav, "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                    "-af", AF, "-c:a", "aac", "-b:a", "320k", "-shortest", "-movflags", "+faststart", final], check=True)
    os.remove(vid)
    json.dump({"sections": {s: [O[s], d] for s, d in zip(SECS, ds)}, "total": total, "marks": marks},
              open(os.path.join(OUT, "timeline.json"), "w"), indent=1)
    print("wrote", final)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "render":
        render_all(sys.argv[2:] or SECS)
    main()
