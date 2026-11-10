"""Master the score and mix the narration over it.

    python mix_audio.py SCORE.mp3 LINES_DIR OUT_DIR      # LINES_DIR holds line_00.mp3 ... (one per narration line)

Writes OUT_DIR/every_grain_of_us_score.wav (the score, -16 LUFS, exactly the film's length),
narration_lily.wav (the voice alone, -16 LUFS), every_grain_of_us_score_with_narration.wav (the voice
over the score, which dips about 10 dB under each line), MP3s of the two mixes, and narration_lily.srt.
A quiet opening in the score is lifted first (ride_intro) so it is heard under the voice.
Each line starts at its time in narration_script.json, or 0.4 s after the previous line ends if later.
Each voice clip is trimmed to its speech and faded in and out (clean_line), so none clicks.
"""
import json, os, subprocess, sys
import numpy as np
import story as S

SR = 48000
HERE = os.path.dirname(os.path.abspath(__file__))


def run(*a):
    subprocess.run(list(a), check=True)


def loudnorm(src, dst, lra=11):
    """Two-pass EBU R128 normalisation to -16 LUFS, peaks at -1.5 dBTP, padded/trimmed to the film."""
    out = subprocess.run(["ffmpeg", "-hide_banner", "-i", src, "-af", f"loudnorm=I=-16:TP=-1.5:LRA={lra}:print_format=json",
                          "-f", "null", "-"], capture_output=True, text=True).stderr
    m = json.loads(out[out.rindex("{"):out.rindex("}") + 1])
    meas = (f"measured_I={m['input_i']}:measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}:"
            f"measured_thresh={m['input_thresh']}:offset={m['target_offset']}")
    run("ffmpeg", "-v", "error", "-y", "-i", src, "-af", f"loudnorm=I=-16:TP=-1.5:LRA={lra}:{meas}:linear=true",
        "-ar", str(SR), "-c:a", "pcm_s16le", dst + ".ln.wav")
    run("ffmpeg", "-v", "error", "-y", "-i", dst + ".ln.wav", "-af", "apad", "-t", str(S.DURATION),
        "-c:a", "pcm_s16le", dst)                          # exactly the film's length
    os.remove(dst + ".ln.wav")


def decode(path, ch):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", str(ch), "-ar", str(SR), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, ch)


def clean_line(a):
    """Trim a generated voice clip to its speech. The voice generator sometimes leaves a stray click or a
    short high chirp at the very end of a clip, after the line has fallen silent: anything less than 0.3 s
    long that follows the last silence of 150 ms or more is silenced. Then a 5 ms fade-in and a 30 ms
    fade-out, so no clip starts or stops on a non-zero sample (which clicks)."""
    x = a[:, 0].copy()
    w = int(0.01 * SR)
    n = len(x) // w
    quiet = 20 * np.log10(np.sqrt((x[:n * w].reshape(n, w) ** 2).mean(1)) + 1e-9) < -55
    run, last = 0, None
    for i in range(n):                                     # the last silent stretch of 15+ frames
        run = run + 1 if quiet[i] else 0
        if run >= 15:
            last = (i - run + 1, i)
    end = (last[0] + 3) * w if last is not None and (n - 1 - last[1]) * w < 0.3 * SR else len(x)
    fi, fo = int(0.005 * SR), int(0.03 * SR)
    x[:fi] *= np.linspace(0, 1, fi)
    x[end - fo:end] *= np.linspace(1, 0, fo)
    x[end:] = 0.0                                          # silenced, not cut: the line keeps its length and timing
    return x[:, None]


def ride_intro(mus, until=36.5, start_db=-6.0, floor_db=-4.0, rise=3.0, max_gain_db=18.0):
    """Lift a quiet opening so it is heard under the narration. A generated score can open 15-20 dB below
    its body (the epic take 3 did, for its first 30 s), and ducked under the voice it then vanishes.
    Where the music's short-term level (3 s RMS) falls further below the body's median level than a floor
    rising from start_db (at 0 s) to floor_db (by `rise` s), raise it to that floor, with a smooth gain;
    the gain hands back to 0 dB by `until` (the end of the first section). The music then starts at
    full presence on the first frame (the score's leading silence is trimmed before: audio_sources/README.md)."""
    mono = mus.mean(1)
    hop, win = int(0.25 * SR), int(3.0 * SR)
    c = np.arange(0, len(mono), hop)
    pw = np.convolve(mono.astype(np.float64) ** 2, np.ones(win) / win, mode="same")[c]
    env = 10 * np.log10(pw + 1e-12)
    t = c / SR
    body = np.median(env[(t > until) & (t < 150)])
    target = body + np.interp(t, [0, rise], [start_db, floor_db])
    gdb = np.clip(target - env, 0, max_gain_db) * np.interp(t, [until - 6, until], [1, 0])
    k = 8                                                  # smooth over ~2 s: no pumping
    kern = np.hanning(2 * k + 1); gdb = np.convolve(np.pad(gdb, k, mode="edge"), kern / kern.sum(), mode="valid")
    g = 10 ** (np.interp(np.arange(len(mono)) / SR, t, gdb) / 20)
    print(f"intro ride: body {body:.1f} dB, lift up to {gdb.max():.1f} dB, back to 0 dB by {until:.1f} s")
    return (mus * g[:, None]).astype(np.float32)


def main(score, lines_dir, out):
    os.makedirs(out, exist_ok=True)
    N = int(round(S.DURATION * SR))
    j = lambda n: os.path.join(out, n)
    music_wav = j("every_grain_of_us_score.wav")
    tmp = j("_tmp.wav")
    run("ffmpeg", "-v", "error", "-y", "-i", score, "-af", f"afade=t=out:st={S.DURATION - 1.5}:d=1.5", "-ar", str(SR), tmp)
    loudnorm(tmp, music_wav)
    # a quiet opening lifted so it is heard; after the normalisation, so the rest of the score is untouched
    ride_intro(decode(music_wav, 2)).tofile(j("_ride.f32"))
    run("ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", j("_ride.f32"),
        "-c:a", "pcm_s16le", music_wav)
    os.remove(j("_ride.f32"))
    L = json.load(open(os.path.join(HERE, "narration_script.json")))["lines"]
    vo = np.zeros((N, 1), np.float32)
    spans, end, srt = [], 0.0, []
    ts = lambda x: f"{int(x // 3600):02d}:{int(x // 60 % 60):02d}:{int(x % 60):02d},{int(round(x * 1000)) % 1000:03d}"
    for i, (t, text) in enumerate(L):
        a = clean_line(decode(os.path.join(lines_dir, f"line_{i:02d}.mp3"), 1))
        start = max(t, end + 0.4)
        s = int(start * SR)
        vo[s:s + len(a)] += a[:N - s]
        end = start + len(a) / SR
        spans.append((start, end))
        srt.append(f"{i + 1}\n{ts(start)} --> {ts(end)}\n{text.replace('...', '…')}\n")
        print(f"line {i:2d}  {start:6.1f}-{end:6.1f}  {text[:50]}")
    open(j("narration_lily.srt"), "w").write("\n".join(srt))
    np.repeat(vo, 2, 1).tofile(j("_vo.f32"))
    run("ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", j("_vo.f32"), tmp)
    loudnorm(tmp, j("narration_lily.wav"))
    g = np.full(N, 0.85, np.float32)                        # the music bed dips under each line
    for s0, s1 in spans:
        g[int((s0 - 0.3) * SR):int((s1 + 0.6) * SR)] = 0.27
    k = int(0.35 * SR)
    kern = np.hanning(2 * k + 1).astype(np.float32)
    g = np.convolve(np.pad(g, k, mode="edge"), kern / kern.sum(), mode="valid")[:N]
    mus = decode(music_wav, 2)
    mus = np.pad(mus, ((0, max(0, N - len(mus))), (0, 0)))[:N]
    (mus * g[:, None]).astype(np.float32).tofile(j("_bed.f32"))
    run("ffmpeg", "-v", "error", "-y", "-i", j("narration_lily.wav"), "-f", "f32le", "-ar", str(SR), "-ac", "2",
        "-i", j("_bed.f32"), "-filter_complex", "[0:a][1:a]amix=inputs=2:normalize=0", tmp)
    loudnorm(tmp, j("every_grain_of_us_score_with_narration.wav"), lra=14)
    for n in ("every_grain_of_us_score", "every_grain_of_us_score_with_narration"):
        run("ffmpeg", "-v", "error", "-y", "-i", j(n + ".wav"), "-c:a", "libmp3lame", "-b:a", "320k", j(n + ".mp3"))
    for n in ("_tmp.wav", "_vo.f32", "_bed.f32"):
        os.remove(j(n))


if __name__ == "__main__":
    main(*sys.argv[1:4])
