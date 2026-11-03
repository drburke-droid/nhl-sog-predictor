"""The 10 Biggest Single-Day Outliers: join the rendered sections with 0.6 s crossfades, lay in the orchestral score
(../../outliers/orchestra.py, with this film's timeline), normalise to -16 LUFS, write out/single_day_final.mp4.

    python sd_assemble.py [crf]
"""
import os, sys, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "outliers"))
import sdfilm                                               # noqa: E402
import assemble as A                                        # noqa: E402  (the season film's helpers)
import orchestra                                            # noqa: E402

OUT = os.path.join(HERE, "out")
SECS = list(sdfilm.ORDER)
FINAL = os.path.join(OUT, "single_day_final.mp4")
orchestra.OUT = OUT
orchestra.CLOSE_FINALE, orchestra.CLOSE_DUR = 33.0, sdfilm.Close.dur


def timeline():
    ds = [sdfilm.SECTIONS[s]().dur for s in SECS]
    off = [0.0]
    for d in ds[:-1]:
        off.append(off[-1] + d - A.XF)
    return dict(zip(SECS, off)), off[-1] + ds[-1]


def main(crf=18):
    O, total = timeline()
    paths = [os.path.join(OUT, f"{s}.mp4") for s in SECS]
    print(f"total {total:.2f} s ({int(total // 60)}:{total % 60:04.1f})", flush=True)
    S = orchestra.build(O)
    if S.dropped:
        raise SystemExit(f"{len(S.dropped)} notes out of range")
    mix = orchestra.render(O, total)
    voice = os.path.join(OUT, "narration.wav")
    if os.path.exists(voice):                                # narrate.py output: the voice over a dipped score
        import narrate
        mix = narrate.duck_and_mix(mix, voice)
        print("narration mixed in", flush=True)
    raw = os.path.join(OUT, "score_raw.wav")
    orchestra.write(mix, raw)
    wav = os.path.join(OUT, "score.wav")
    m = A.loudnorm(raw, wav)
    print("score loudness in", m["input_i"], "LUFS", flush=True)
    args = ["ffmpeg", "-y", "-v", "error"]
    for p in paths:
        args += ["-i", p]
    args += ["-i", wav]
    fc, last = [], "[0:v]"
    acc = A.dur(paths[0])
    for i in range(1, len(paths)):
        off = acc - A.XF
        fc.append(f"{last}[{i}:v]xfade=transition=fade:duration={A.XF}:offset={off:.3f}[v{i}]")
        last = f"[v{i}]"
        acc = off + A.dur(paths[i])
    args += ["-filter_complex", ";".join(fc), "-map", last, "-map", f"{len(paths)}:a", "-c:v", "libx264", "-preset", "slow",
             "-crf", str(crf), "-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709",
             "-colorspace", "bt709", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", FINAL]
    subprocess.run(args, check=True)
    print(FINAL, f"{os.path.getsize(FINAL) / 1e6:.1f} MB", f"{A.dur(FINAL):.2f} s")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 18)
