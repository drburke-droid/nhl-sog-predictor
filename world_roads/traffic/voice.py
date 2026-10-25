"""Mix recorded narration into the score, the music ducked under the voice (assemble.py calls mix()).

Clips live in ../media/traffic/voice/ (VOICE_DIR=... for another folder), named by section from VOICEOVER.md:
    hook.mp3                 one clip for the whole section, starting at its first cue
    hook_01.mp3, hook_02...  one clip per line, each starting at its own cue (exact timing)
.wav, .m4a, .aac, .flac and .ogg work too. With no clips the score is left as it is.
"""
import os, re, subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.environ.get("VOICE_DIR", os.path.join(os.path.dirname(HERE), "media", "traffic", "voice"))
EXT = (".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg")
DUCK_DB = -11.0              # music under the voice
VOICE_RMS = 0.16             # the voice's level while speaking (the score peaks at 0.89)


def decode(path, sr):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).astype(np.float64)


def clips(O):
    """(start s, label, path) for every clip found, placed by the script's cues"""
    import voiceover as VO
    if not os.path.isdir(DIR):
        return []
    files = {os.path.splitext(f)[0]: os.path.join(DIR, f) for f in sorted(os.listdir(DIR)) if f.lower().endswith(EXT)}
    rows = VO.numbered()
    out = []
    for sec in O:
        mine = [r for r in rows if r[0] == sec]
        per_line = [(f"{sec}_{k:02d}", cue) for _, k, cue, _ in mine if f"{sec}_{k:02d}" in files]
        if per_line:
            out += [(O[sec] + cue, name, files[name]) for name, cue in per_line]
        elif sec in files and mine:
            out.append((O[sec] + mine[0][2], sec, files[sec]))
    unknown = set(files) - {c[1] for c in out}
    for u in sorted(unknown):
        print(f"  voice: {u} doesn't match a section or line in VOICEOVER.md, skipped")
    return out


def mix(music, O, total, sr):
    """music: (2, n) score. Returns it with the narration added and the music ducked under it."""
    cl = clips(O)
    if not cl:
        return music
    n = music.shape[1]
    v = np.zeros(n)
    ends = {}
    for start, name, path in cl:
        x = decode(path, sr)
        act = np.abs(x) > 0.02 * max(np.abs(x).max(), 1e-9)
        rms = np.sqrt(np.mean(x[act] ** 2)) if act.any() else 1.0
        x = x * (VOICE_RMS / rms)
        i0 = int(round(start * sr))
        seg = x[:max(0, n - i0)]
        v[i0:i0 + len(seg)] += seg
        ends[name] = (start, start + len(x) / sr)
        print(f"  voice: {name:10s} {start:7.2f}-{start + len(x) / sr:7.2f} s")
    names = sorted(ends, key=lambda k: ends[k][0])
    for a, b in zip(names[:-1], names[1:]):          # warn when one clip runs into the next
        if ends[a][1] > ends[b][0] + 0.05:
            print(f"  voice: {a} runs {ends[a][1] - ends[b][0]:.2f} s into {b}")
    # ducking: a 100 Hz envelope of the voice, fast attack, slow release
    hop = sr // 100
    env = np.sqrt(np.convolve(v ** 2, np.ones(hop) / hop, "same")[::hop])
    on = (env > 0.1 * VOICE_RMS).astype(float)
    g = np.zeros_like(on)
    for k in range(1, len(on)):                       # attack 80 ms, release 600 ms
        a = 0.12 if on[k] > g[k - 1] else 0.016
        g[k] = g[k - 1] + a * (on[k] - g[k - 1])
    duck = 1 - (1 - 10 ** (DUCK_DB / 20)) * np.interp(np.arange(n) / hop, np.arange(len(g)), g)
    out = music * duck + v[None, :]
    peak = np.abs(out).max()
    return out * (0.89 / peak) if peak > 0.89 else out
