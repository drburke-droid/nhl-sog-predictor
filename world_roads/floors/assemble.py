"""Join the film's sections with short cross-fades and add the score.

    python assemble.py        -> out/video/every_floor_light_speed.mp4 (+ timeline.json)

The score reuses the Roomba film's synthesized theme (roomba/music.py, instruments from flights/score.py), cut to
this film's picture: a quiet build under the slowed-down opening, a drop when the gold bursts out of the Empire
State Building, hits when the building, New York and the planet are done, a hush before the U-turn and its boom,
and the home chord with the docking chime at the end. No samples.
"""
import os, sys, json, subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "flights"))
sys.path.insert(0, os.path.join(ROOT, "roomba"))
sys.path.insert(0, HERE)                     # first, so `video` is this film's, not the Roomba film's
OUT = os.path.join(HERE, "out", "video")
XF = 0.5
AF = "volume=-1.2dB,alimiter=limit=0.708:attack=3:release=60:level=false"
FINAL = os.path.join(OUT, "every_floor_light_speed.mp4")


def dur(p):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p],
                                capture_output=True, text=True, check=True).stdout)


def track(O, total):
    import music as MU
    import video as V
    SUS = MU.SUS
    T = MU.Track(total)
    h, e, n = O["hook"], O["esb"], O["nyc"]
    H = V.Hook
    # the opening: a heartbeat under the billion-times slow motion, a lift as the floors start to fly by,
    # the drop when it speeds up and pulls out, the hit on the question
    T.section(h, h + H.FLOORS0, "suspense", prog=SUS)
    T.riser(h + 2.6, h + H.FLOORS0)
    T.impact(h + H.FLOORS0, 0.6)
    T.section(h + H.FLOORS0, h + H.DROP, "tension", prog=SUS)
    T.riser(h + H.DROP - 2.2, h + H.DROP)
    T.section(h + H.DROP, h + H.EARTH, "full")
    T.impact(h + H.DROP, 0.9); T.crash(h + H.DROP, 1.0)
    T.section(h + H.EARTH, O["room"], "tension", prog=SUS)
    T.riser(h + H.EARTH, h + H.Q)
    T.impact(h + H.Q, 1.1); T.crash(h + H.Q, 0.8)
    # the rules: light, then mellow
    T.section(O["room"], O["rules"], "light")
    T.section(O["rules"], e, "mellow")
    # the Empire State Building, then New York
    hit = e + V.ESB.T1
    T.section(e, hit, "tension", prog=SUS)
    T.riser(hit - 3.0, hit)
    T.impact(hit, 1.0); T.crash(hit, 1.0); T.bell(hit, 81)
    T.section(hit, n, "triumph")
    done = n + V.NYCRun.END
    T.section(n, done, "full")
    T.riser(done - 3.0, done)
    T.impact(done, 1.2); T.crash(done, 1.0); T.bell(done, 83)
    T.section(done, O["earth"], "triumph")
    # Earth: a key change for the planet, one long build to the last floor, then a breath for the distance
    E = O["earth"]
    last = E + V.Sweep.E1 + V.Sweep.E2
    T.section(E, last, "full2", key=2)
    T.impact(E, 0.8)
    T.riser(last - 3.5, last)
    T.impact(last, 1.3); T.crash(last, 1.0); T.bell(last, 85)
    T.section(last, O["far"], "triumph", key=2)
    T.section(O["far"], O["dust"], "mellow", key=2)
    T.section(O["dust"], O["uturn"], "light")
    T.chime(O["dust"] + 0.5, 0, 0.6)
    # the U-turn: a hush, the boom, then the burning planet and home
    boom = O["uturn"] + V.UTurn.HIT
    T.section(O["uturn"], boom, "suspense", prog=SUS)
    T.riser(boom - 2.4, boom)
    T.impact(boom, 1.8); T.crash(boom, 1.0)
    T.section(boom + 0.3, O["spotless"], "tension", prog=SUS)
    end = O["spotless"] + 8.6
    T.section(O["spotless"], end, "mellow")
    T.ending(end)
    return T.mixdown()


def main():
    import video as V
    import score as SC
    secs = V.FILM
    # the Earth sweep is one continuous shot rendered in two parts: join them with a hard cut (no cross-fade)
    e1, e2, ej = (os.path.join(OUT, f"{n}.mp4") for n in ("earth1", "earth2", "earth"))
    if not os.path.exists(ej) or os.path.getmtime(ej) < max(os.path.getmtime(e1), os.path.getmtime(e2)):
        lst = os.path.join(OUT, "earth.txt")
        open(lst, "w").write(f"file '{e1}'\nfile '{e2}'\n")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", ej], check=True)
    paths = [os.path.join(OUT, f"{s}.mp4") for s in secs]
    ds = [dur(p) for p in paths]
    off = [0.0]
    for d in ds[:-1]:
        off.append(off[-1] + d - XF)
    total = off[-1] + ds[-1]
    O = dict(zip(secs, off))
    for s, o, d in zip(secs, off, ds):
        print(f"  {s:9s} {o:7.2f} + {d:6.2f}")
    print(f"total {total:.2f} s ({int(total // 60)}:{total % 60:04.1f})", flush=True)
    vid = os.path.join(OUT, "_video.mp4")
    args = ["ffmpeg", "-y", "-v", "error"]
    for p in paths:
        args += ["-i", p]
    fc, last = [], "0:v"
    for k in range(1, len(secs)):
        fc.append(f"[{last}][{k}:v]xfade=transition=fade:duration={XF}:offset={off[k]:.3f}[v{k}]")
        last = f"v{k}"
    args += ["-filter_complex", ";".join(fc), "-map", f"[{last}]", "-c:v", "libx264", "-preset", "medium", "-crf", "17",
             "-pix_fmt", "yuv420p", "-r", "30", vid]
    subprocess.run(args, check=True)
    mix = track(O, total)
    wav = os.path.join(OUT, "score.wav")
    SC.write(mix, wav)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", vid, "-i", wav, "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                    "-af", AF, "-c:a", "aac", "-b:a", "320k", "-shortest", "-movflags", "+faststart", FINAL], check=True)
    os.remove(vid)
    json.dump({"sections": {s: [O[s], d] for s, d in zip(secs, ds)}, "total": total},
              open(os.path.join(OUT, "timeline.json"), "w"), indent=1)
    print("wrote", FINAL)


if __name__ == "__main__":
    main()
