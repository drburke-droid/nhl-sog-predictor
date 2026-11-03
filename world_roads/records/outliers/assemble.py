"""The 10 Biggest Statistical Outliers in Sports History: join the rendered sections with short crossfades, lay the score under them, write the final video.

    python assemble.py [crf]
"""
import os, sys, json, re, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "out")
import film
SECS = list(film.ORDER)
XF = 0.6                                         # crossfade between sections (s)
FINAL = os.path.join(OUT, "outliers_final.mp4")


def dur(p):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p],
                       capture_output=True, text=True, check=True)
    return float(r.stdout)


def timeline():
    """section start times (video time) and total length, from the sections' own durations"""
    import film
    ds = [film.SECTIONS[s]().dur for s in SECS]
    off = [0.0]
    for d in ds[:-1]:
        off.append(off[-1] + d - XF)
    return dict(zip(SECS, off)), off[-1] + ds[-1]


def loudnorm(src, dst, I=-16.0, TP=-1.5, LRA=11.0):
    """two-pass EBU R128 normalisation (linear gain when the range allows)"""
    r = subprocess.run(["ffmpeg", "-hide_banner", "-i", src, "-af", f"loudnorm=I={I}:TP={TP}:LRA={LRA}:print_format=json",
                        "-f", "null", "-"], capture_output=True, text=True, check=True)
    m = json.loads(re.findall(r"\{[^{}]*\}", r.stderr)[-1])
    af = (f"loudnorm=I={I}:TP={TP}:LRA={LRA}:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
          f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-af", af, "-ar", "48000", "-c:a", "pcm_s16le", dst], check=True)
    return m


def main(crf=18):
    O, total = timeline()
    paths = [os.path.join(OUT, f"{s}.mp4") for s in SECS]
    for s, p in zip(SECS, paths):
        print(f"  {s:7s} {O[s]:7.2f} + {dur(p):6.2f}")
    print(f"total {total:.2f} s ({int(total // 60)}:{total % 60:04.1f})", flush=True)
    import score as music
    mix = music.render(O, total)
    raw = os.path.join(OUT, "score_raw.wav")
    music.write(mix, raw)
    wav = os.path.join(OUT, "score.wav")
    m = loudnorm(raw, wav)
    print("score loudness in", m["input_i"], "LUFS", flush=True)
    args = ["ffmpeg", "-y", "-v", "error"]
    for p in paths:
        args += ["-i", p]
    args += ["-i", wav]
    fc, last = [], "[0:v]"
    acc = dur(paths[0])
    for i in range(1, len(paths)):
        off = acc - XF
        lab = f"[v{i}]"
        fc.append(f"{last}[{i}:v]xfade=transition=fade:duration={XF}:offset={off:.3f}{lab}")
        last = lab
        acc = off + dur(paths[i])
    args += ["-filter_complex", ";".join(fc), "-map", last, "-map", f"{len(paths)}:a",
             "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p",
             "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
             "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", FINAL]
    subprocess.run(args, check=True)
    print(FINAL, f"{os.path.getsize(FINAL) / 1e6:.1f} MB", f"{dur(FINAL):.2f} s")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 18)
