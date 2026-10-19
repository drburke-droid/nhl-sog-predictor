"""Join the rendered sections with crossfades, lay the score under them, and write the final video (HEVC:
the desk's moving film grain makes H.264 four times larger for the same quality).

    python assemble.py
"""
import os, sys, json, re, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "out")
SECS = ["open", "eye", "retina", "back", "smaller", "how", "screen", "numbers", "uses", "close"]
XF = 1.0                                         # crossfade between sections (s)
FINAL = os.path.join(OUT, "vernier_4k_master.mp4")


def timeline():
    """section start times (video time) and the total length, from the sections' own durations"""
    import film
    ds = [film.SECTIONS[s]().dur for s in SECS]
    off = [0.0]
    for d in ds[:-1]:
        off.append(off[-1] + d - XF)
    return dict(zip(SECS, off)), off[-1] + ds[-1]


def dur(p):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p],
                       capture_output=True, text=True, check=True)
    return float(r.stdout)


def loudnorm(src, dst, I=-16.0, TP=-1.5, LRA=11.0):
    """two-pass EBU R128 normalisation (linear gain when the range allows)"""
    r = subprocess.run(["ffmpeg", "-hide_banner", "-i", src, "-af", f"loudnorm=I={I}:TP={TP}:LRA={LRA}:print_format=json",
                        "-f", "null", "-"], capture_output=True, text=True, check=True)
    m = json.loads(re.findall(r"\{[^{}]*\}", r.stderr)[-1])
    af = (f"loudnorm=I={I}:TP={TP}:LRA={LRA}:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
          f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-af", af, "-ar", "48000", "-c:a", "pcm_s16le", dst], check=True)
    return m


def main(crf=24):
    O, total = timeline()
    paths = [os.path.join(OUT, f"{s}.mp4") for s in SECS]
    for s, p in zip(SECS, paths):
        print(f"  {s:8s} {O[s]:7.2f} + {dur(p):6.2f}")
    print(f"total {total:.2f} s ({int(total // 60)}:{total % 60:04.1f})", flush=True)
    # the score
    import music
    mix, ev = music.render(O, total)
    raw = os.path.join(OUT, "score_raw.wav")
    music.write(mix, raw)
    wav = os.path.join(OUT, "score.wav")
    m = loudnorm(raw, wav)
    print("score loudness in", m["input_i"], "LUFS; notes", len(ev), flush=True)
    # the picture, crossfaded, with the score
    args = ["ffmpeg", "-y", "-v", "error"]
    for p in paths:
        args += ["-i", p]
    args += ["-i", wav]
    fc, last = [], "0:v"
    for k in range(1, len(SECS)):
        fc.append(f"[{last}][{k}:v]xfade=transition=fade:duration={XF}:offset={O[SECS[k]]:.3f}[v{k}]")
        last = f"v{k}"
    args += ["-filter_complex", ";".join(fc), "-map", f"[{last}]", "-map", f"{len(SECS)}:a",
             "-c:v", "libx265", "-preset", "medium", "-crf", str(crf), "-x265-params", "log-level=error", "-tag:v", "hvc1",
             "-pix_fmt", "yuv420p", "-r", "30",
             "-c:a", "aac", "-b:a", "256k", "-shortest", "-movflags", "+faststart", FINAL]
    subprocess.run(args, check=True)
    json.dump({"sections": {s: O[s] for s in SECS}, "total": total}, open(os.path.join(OUT, "timeline.json"), "w"), indent=1)
    print("wrote", FINAL, f"{os.path.getsize(FINAL) / 1e6:.1f} MB")


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:]))
