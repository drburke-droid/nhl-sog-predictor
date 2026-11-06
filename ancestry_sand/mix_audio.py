"""Master the score and mix the narration over it.

    python mix_audio.py SCORE.mp3 LINES_DIR OUT_DIR      # LINES_DIR holds line_00.mp3 ... (one per narration line)

Writes OUT_DIR/every_grain_of_us_score.wav (the score, -16 LUFS, exactly the film's length),
narration_lily.wav (the voice alone, -16 LUFS), every_grain_of_us_score_with_narration.wav (the voice
over the score, which dips about 10 dB under each line), MP3s of the two mixes, and narration_lily.srt.
Each line starts at its time in narration_script.json, or 0.4 s after the previous line ends if later.
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


def main(score, lines_dir, out):
    os.makedirs(out, exist_ok=True)
    N = int(round(S.DURATION * SR))
    j = lambda n: os.path.join(out, n)
    music_wav = j("every_grain_of_us_score.wav")
    tmp = j("_tmp.wav")
    run("ffmpeg", "-v", "error", "-y", "-i", score, "-af", f"afade=t=out:st={S.DURATION - 1.5}:d=1.5", "-ar", str(SR), tmp)
    loudnorm(tmp, music_wav)
    L = json.load(open(os.path.join(HERE, "narration_script.json")))["lines"]
    vo = np.zeros((N, 1), np.float32)
    spans, end, srt = [], 0.0, []
    ts = lambda x: f"{int(x // 3600):02d}:{int(x // 60 % 60):02d}:{int(x % 60):02d},{int(round(x * 1000)) % 1000:03d}"
    for i, (t, text) in enumerate(L):
        a = decode(os.path.join(lines_dir, f"line_{i:02d}.mp3"), 1)
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
