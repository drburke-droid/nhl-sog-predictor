"""Join the sections (cross-fades) and add the score.

    python assemble.py render [names...]   -> render sections (4 at a time), then assemble
    python assemble.py                     -> ../media/rebuild/win_now_or_tear_it_down.mp4 from rendered sections
"""
import os, sys, json, subprocess
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
sys.path.insert(0, HERE)
import video as V      # noqa: E402

SECS = V.ORDER
XF = V.XF
FINAL = os.path.join(os.path.dirname(HERE), "media", "rebuild", "win_now_or_tear_it_down.mp4")
# trim, then a peak limiter at -3 dBFS; AAC at 320k keeps the true peak under -1 dBTP
AF = "volume=-1.2dB,alimiter=limit=0.708:attack=3:release=60:level=false"


def render_all(names, jobs=4):
    def one(nm):
        log = open(os.path.join(OUT, f"{nm}.log"), "w")
        r = subprocess.run([sys.executable, "video.py", nm], cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
        print(f"  {nm}: exit {r.returncode}", flush=True)
        return r.returncode
    heavy = ["teardown", "shifted", "hook", "titles"]
    order = [n for n in heavy if n in names] + [n for n in names if n not in heavy]
    os.makedirs(OUT, exist_ok=True)
    with ThreadPoolExecutor(jobs) as ex:
        codes = list(ex.map(one, order))
    assert not any(codes), "a section failed"


def dur(p):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p],
                                capture_output=True, text=True, check=True).stdout)


def offsets():
    ds = [V.SECTIONS[s].dur for s in SECS]
    off = [0.0]
    for d in ds[:-1]:
        off.append(off[-1] + d - XF)
    return dict(zip(SECS, off)), off[-1] + ds[-1]


def main():
    import music
    import score as SC
    paths = [os.path.join(OUT, f"{s}.mp4") for s in SECS]
    O, total = offsets()
    for s in SECS:
        assert abs(dur(os.path.join(OUT, f"{s}.mp4")) - V.SECTIONS[s].dur) < 0.05, s
        print(f"  {s:9s} {O[s]:7.2f}")
    print(f"total {total:.2f} s ({int(total // 60)}:{total % 60:04.1f})", flush=True)
    vid = os.path.join(OUT, "_video.mp4")
    args = ["ffmpeg", "-y", "-v", "error"]
    for p in paths:
        args += ["-i", p]
    fc, last = [], "0:v"
    for k in range(1, len(SECS)):
        fc.append(f"[{last}][{k}:v]xfade=transition=fade:duration={XF}:offset={O[SECS[k]]:.3f}[v{k}]")
        last = f"v{k}"
    args += ["-filter_complex", ";".join(fc), "-map", f"[{last}]", "-c:v", "libx264", "-preset", "slow", "-crf", "18",
             "-tune", "animation", "-pix_fmt", "yuv420p", "-r", "30", vid]
    subprocess.run(args, check=True)
    wav = os.path.join(OUT, "score.wav")
    SC.write(music.main_track(O, total), wav)
    os.makedirs(os.path.dirname(FINAL), exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", vid, "-i", wav, "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                    "-af", AF, "-c:a", "aac", "-b:a", "320k", "-shortest", "-movflags", "+faststart", FINAL], check=True)
    os.remove(vid)
    json.dump({"sections": {s: [O[s], V.SECTIONS[s].dur] for s in SECS}, "total": total}, open(os.path.join(OUT, "timeline.json"), "w"), indent=1)
    print("wrote", FINAL)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "render":
        render_all(sys.argv[2:] or SECS)
    main()
