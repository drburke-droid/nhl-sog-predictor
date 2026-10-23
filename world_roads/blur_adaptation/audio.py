"""Score + voiceover mix for the blur-adaptation video parts.

Music reuses the felt-piano / pad / room instruments from the Light Speed vernier score
(ls_music/vernier_music.py). Voice lines come from tts.py (Kokoro, bf_emma).

    python3 audio.py part1        # writes part1/mix.wav and muxes part1/part1_final.mp4
    python3 audio.py part2        # same for part2 (needs part2/timeline.json from part2.py)
"""
from paths import WORK  # noqa: E402
import os
import sys
import json
import math
import subprocess
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly, lfilter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "vernier"))
import music as VM                                  # noqa: E402

SR = VM.SR
BASE = WORK + ""
CH = dict(VM.CH)
CH["Dm9"] = (["D2", "A2"], ["A4", "C5", "E5", "F5", "A5"], ["D3", "A3", "E4", "F4"])
CH["Fadd9"] = (["F2", "C3"], ["G4", "A4", "C5", "F5", "G5"], ["F3", "C4", "G4", "A4"])


# ------------------------------------------------------------------ music
def bell(f0, vel=0.6, dur=3.0, seed=0):
    """a soft glass/bell chime: inharmonic sine partials with fast-then-slow decay"""
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    out = np.zeros(n, np.float32)
    for ratio, amp, tau in ((1.0, 1.0, 1.4), (2.01, 0.45, 0.9), (2.76, 0.3, 0.6), (5.4, 0.12, 0.25), (8.93, 0.05, 0.12)):
        out += amp * np.sin(2 * math.pi * f0 * ratio * t) * np.exp(-t / tau)
    att = np.clip(t / 0.004, 0, 1)
    return (out * att * vel * 0.5).astype(np.float32)


class Score:
    def __init__(self, total, seed=11):
        self.total = total
        self.n = int((total + 4.0) * SR)
        self.L = np.zeros(self.n, np.float32)
        self.R = np.zeros(self.n, np.float32)
        self.P = np.zeros(self.n, np.float32)
        self.FX = np.zeros(self.n, np.float32)
        self.rng = np.random.default_rng(seed)
        self.k = 0

    def note(self, t, name, d, v):
        x = VM.piano(VM.hz(name), d, v, seed=self.k)
        self.k += 1
        i = int(t * SR)
        x = x[: max(0, self.n - i)]
        pan = np.clip((VM.midi(name) - 60) / 40, -0.35, 0.35)
        self.L[i:i + len(x)] += x * math.sqrt(0.5 - pan)
        self.R[i:i + len(x)] += x * math.sqrt(0.5 + pan)

    def pad(self, t0, t1, chord, gain=1.0):
        for j, name in enumerate(CH[chord][2]):
            x = VM.pad_tone(VM.hz(name), t1 - t0 + 1.5, seed=j + int(t0 * 10))
            i = int(t0 * SR)
            x = x[: max(0, self.n - i)]
            self.P[i:i + len(x)] += x * gain

    def chime(self, t, names=("A5", "E6"), vel=0.55):
        for k, nm in enumerate(names):
            x = bell(VM.hz(nm), vel * (1 - 0.3 * k), seed=k)
            i = int((t + 0.03 * k) * SR)
            x = x[: max(0, self.n - i)]
            self.FX[i:i + len(x)] += x

    def section(self, t0, t1, chords, dens, lh=True, pad=True, pad_gain=1.0):
        """chords evenly over [t0, t1): left-hand root/fifth, pad, sparse right-hand melody"""
        step = (t1 - t0) / len(chords)
        prev = VM.midi("C5")
        for k, c in enumerate(chords):
            a, b = t0 + k * step, t0 + (k + 1) * step
            lhn, rh, _ = CH[c]
            if lh:
                self.note(a, lhn[0], min(step, 6.0), 0.30)
                if step > 3.0:
                    self.note(a + min(1.5, step / 3), lhn[1], min(step - 1.5, 5.0), 0.20)
            if pad:
                self.pad(a, b, c, pad_gain)
            g = a + 1.0
            while g < b - 0.6:
                if self.rng.random() < dens:
                    cand = [VM.midi(n) for n in rh]
                    w = np.array([math.exp(-abs(m - prev) / 3.0) * (0.4 if m == prev else 1.0) for m in cand])
                    m = int(self.rng.choice(cand, p=w / w.sum()))
                    self.note(g + self.rng.normal(0, 0.012), rh[cand.index(m)],
                              float(self.rng.choice([1.0, 1.5, 2.0, 3.0])), float(self.rng.uniform(0.26, 0.44)))
                    prev = m
                    g += float(self.rng.choice([1.0, 1.5, 2.0]))
                else:
                    g += 0.5

    def ostinato(self, t0, t1, step, names, vel=0.20):
        """a soft repeated note on every picture change (keeps the adaptation block steady)"""
        t, k = t0, 0
        while t < t1 - 0.05:
            self.note(t, names[k % len(names)], step * 0.9, vel)
            t += step
            k += 1

    def rolled(self, t, names, d, v=0.36):
        for k, nm in enumerate(names):
            self.note(t + 0.11 * k, nm, d, v - 0.025 * k)

    def render(self, fade_out=3.0):
        n = self.n
        P = self.P * (0.10 / (np.abs(self.P).max() + 1e-9))
        dry = np.stack([self.L + P * 0.95, self.R + P * 1.05], 1)
        ir = VM.reverb_ir()
        wet = np.stack([fftconvolve(dry[:, c], ir[:, c])[:n] for c in range(2)], 1)
        mix = dry * 0.78 + wet * 0.55
        fx = np.stack([self.FX, self.FX], 1)
        fxw = np.stack([fftconvolve(self.FX, ir[:, c])[:n] for c in range(2)], 1)
        t = np.arange(n) / SR
        env = np.clip(t / 0.6, 0, 1) * np.clip((self.total + 0.3 - t) / fade_out, 0, 1)
        mix = mix / (np.abs(mix).max() + 1e-9) * 0.7
        return (mix * env[:, None])[: int(self.total * SR)], ((fx * 0.8 + fxw * 0.5) * env[:, None])[: int(self.total * SR)]


# ------------------------------------------------------------------ voice + mix
def voice_track(lines, vodir, total):
    v = np.zeros(int(total * SR), np.float32)
    for ln in lines:
        x, sr = sf.read(os.path.join(vodir, f"{ln['id']}.wav"), dtype="float32")
        x = resample_poly(x, SR, sr).astype(np.float32)
        i = int(ln["start"] * SR)
        x = x[: max(0, len(v) - i)]
        v[i:i + len(x)] += x
    # gentle presence: a touch of high-shelf lift via first-difference blend, then normalize to -3 dBFS
    v = v + 0.15 * np.concatenate([[0], np.diff(v)])
    return (v / (np.abs(v).max() + 1e-9) * 0.70).astype(np.float32)


def duck_env(v, depth_db=-11.0, attack=0.06, release=0.6):
    a = np.abs(v)
    a = lfilter([1 - math.exp(-1 / (0.02 * SR))], [1, -math.exp(-1 / (0.02 * SR))], a)
    on = (a > 0.01).astype(np.float32)
    g = np.empty_like(on)
    ka, kr = math.exp(-1 / (attack * SR)), math.exp(-1 / (release * SR))
    s = 0.0
    for i in range(0, len(on), 64):                        # block-wise envelope follower
        tgt = on[i:i + 64].max()
        k = ka if tgt > s else kr
        s = tgt + (s - tgt) * (k ** 64)
        g[i:i + 64] = s
    return 10 ** (depth_db * g / 20)


def mix(music, fx, voice, music_db=-9.0):
    total = min(len(music), len(voice))
    music, fx, voice = music[:total], fx[:total], voice[:total]
    g = duck_env(voice)[:, None]
    out = music * (10 ** (music_db / 20)) * g + fx * 0.5 + voice[:, None]
    return (out / max(1.0, np.abs(out).max() / 0.92)).astype(np.float32)


def mux(video, wav, out):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-i", wav, "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-ar", "48000", "-movflags", "+faststart",
                    "-shortest", out], check=True)


# ------------------------------------------------------------------ parts
def part1():
    d = f"{BASE}/part1"
    lines = json.load(open(f"{d}/lines.json"))
    starts = {"p1_16": 134.0, "p1_17": 137.7}
    for ln in lines:
        ln["start"] = starts.get(ln["id"], ln["start"])
    total = 142.9
    S = Score(total, seed=5)
    # intro + rules: calm, open
    S.section(0.0, 22.0, ["F", "Bb", "Dm", "Csus"], 0.14)
    # adaptation: steady pad, a soft note on every picture change
    S.section(22.0, 82.0, ["Dm9", "Bb", "Dm9", "Bb", "Gm", "Csus"], 0.0, lh=True)
    S.ostinato(22.0, 82.0, 1.5, ["A4", "A4", "C5", "A4"], 0.16)
    tests = [82.0, 98.7, 115.4]
    for k, t in enumerate(tests):
        S.chime(t)
        S.pad(t, t + 5.5, "Fadd9", 0.9)
        if k < 2:
            a = t + 6.2
            S.section(a, a + 10.5, ["Dm9", "Bb"], 0.0)
            S.ostinato(a, a + 10.5, 1.5, ["A4", "A4", "C5", "A4"], 0.16)
    # reveal + control + end: warm resolution
    S.rolled(120.9, ["F2", "C3", "A3", "G4", "C5", "F5"], 6.0)
    S.section(120.9, 136.9, ["Fadd9", "Bb", "F/A", "Csus"], 0.18, lh=False)
    S.rolled(136.9, ["F2", "C3", "A3", "E4", "G4", "C5"], 6.0, 0.32)
    S.pad(136.9, 142.9, "F", 1.0)
    music, fx = S.render(fade_out=2.5)
    voice = voice_track(lines, f"{d}/vo", total)
    out = mix(music, fx, voice)
    sf.write(f"{d}/mix.wav", out, SR, subtype="PCM_16")
    mux(f"{BASE}/blur/demo1_v2.mp4", f"{d}/mix.wav", f"{d}/part1_final.mp4")
    print("part1 ok", f"{d}/part1_final.mp4")


def part2():
    d = f"{BASE}/part2"
    tl = json.load(open(f"{d}/timeline.json"))
    total = tl["total"]
    lines = [{"id": s["id"], "start": s["vo_start"]} for s in tl["scenes"] if s.get("vo")]
    S = Score(total, seed=23)
    for sec in tl["music"]:
        S.section(sec["t0"], sec["t1"], sec["chords"], sec["dens"], lh=sec.get("lh", True))
    for c in tl.get("chimes", []):
        S.chime(c)
    if tl.get("final"):
        S.rolled(tl["final"], ["F2", "C3", "A3", "E4", "G4", "C5"], total - tl["final"], 0.34)
    music, fx = S.render(fade_out=3.0)
    voice = voice_track(lines, f"{d}/vo", total)
    out = mix(music, fx, voice)
    sf.write(f"{d}/mix.wav", out, SR, subtype="PCM_16")
    mux(f"{d}/part2_video.mp4", f"{d}/mix.wav", f"{d}/part2_final.mp4")
    print("part2 ok", f"{d}/part2_final.mp4")


if __name__ == "__main__":
    {"part1": part1, "part2": part2}[sys.argv[1]]()
