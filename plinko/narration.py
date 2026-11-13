"""Voice-over for the chaos cut, read by ElevenLabs' "Lily" voice.

    python3 narration.py lines     # synthesise every phrase (cached), trim, time each shot -> results/narration_cues.json
    python3 narration.py check     # spectrogram sheet of every line's ending -> out/narration/tails.png
    python3 narration.py mix       # -> out/narration/track.wav, out/plinko_chaos_narrated.mp4
    python3 narration.py script    # -> ../videos/copy/plinko_chaos_NARRATION.md

A line is a list of phrases separated by pause marks, "[p0.8]" = 0.8 s of silence. Each phrase is synthesised on
its own (with its neighbours as context, and a fixed seed), cut 70 ms after its last sound with a short fade, and
any short burst that follows a gap after the voice has stopped is dropped: that removes the clicks and squeals the
model sometimes leaves at the end. A line starts at a fixed time in its shot, or "+g": g seconds after the
previous line ends. Shot lengths then come from the narration (prepare_chaos.py reads results/narration_cues.json).
Numbers are spelled the way they should be read and match the figures on screen. The API key is supplied by the
environment's proxy for api.elevenlabs.io; nothing here stores one.
"""
import hashlib
import json
import os
import re
import subprocess
import sys

import numpy as np

VOICE = "pFZP5JQG7iQjIQuC4Bku"          # Lily (British, warm, educational)
MODEL = "eleven_multilingual_v2"
SETTINGS = dict(stability=0.55, similarity_boost=0.75, style=0.15, use_speaker_boost=True, speed=1.0)
SR = 44100
OUT = "out/narration"
CUES = "results/narration_cues.json"
TAIL = 1.2                               # silence kept after a shot's last line

SCRIPT = [
    ("hook", 0.4, "This is Plinko, from The Price Is Right. [p0.5] And this drop wins ten thousand dollars."),
    ("hook", 6.8, "Now, the same drop again. [p0.4] Same spot, matched to the width of an atom."),
    ("hook", 12.4, "Same start. [p0.8] Different finish. [p1.1] That's chaos."),

    ("deterministic", 0.5, "Here's the strange part. [p0.5] There are no dice inside this board. [p0.4] Every bounce "
                           "follows Newton's laws of motion, [p0.2] so the same start should always give the same "
                           "finish."),
    ("deterministic", "+0.6", "If the replay lost, [p0.3] its start can't have been quite the same."),

    ("twins", 0.4, "Here are two chips, released a thousandth of a millimetre apart. [p0.4] They fall together, "
                   "[p0.3] then split."),
    ("twins", 7.4, "One wins ten thousand dollars. [p0.4] Its twin wins a hundred."),
    ("twins", "+0.8", "A difference far too small to see, [p0.3] and a completely different result."),

    ("chaos", 0.6, "This is the heart of chaos theory. [p0.5] It studies systems that follow exact rules, with no "
                   "randomness at all, [p0.3] yet still can't be predicted for long."),
    ("chaos", "+0.8", "It began with the weather. [p0.4] In 1961, the meteorologist Edward Lorenz reran a "
                      "simulation, [p0.2] typing in point five zero six, instead of point five zero six one two "
                      "seven. [p0.4] A difference of about one part in four thousand. [p0.4] Two months into the "
                      "simulated weather, the two forecasts had nothing in common."),
    ("chaos", "+0.8", "Here are two runs of Lorenz's equations, starting almost exactly together. [p0.4] For a "
                      "while, they move as one. [p0.5] Then they part, [p0.3] and never agree again."),
    ("chaos", "+0.8", "Chaos theory calls this sensitive dependence on initial conditions. [p0.4] Lorenz called "
                      "it the butterfly effect."),
    ("chaos", "+0.7", "A Plinko board is a perfect example."),

    ("mechanism", 0.5, "So why is Plinko chaotic? [p0.4] Because the pins are round."),
    ("mechanism", "+0.8", "Bounce two chips off a flat wall, a hair apart, [p0.3] and they leave a hair apart. "
                          "[p0.3] A flat wall faces the same way everywhere."),
    ("mechanism", "+0.8", "A round pin is different. [p0.3] Its surface faces a slightly different way at every "
                          "point. [p0.4] So two chips that hit it a hair apart meet slopes tilted at different "
                          "angles, [p0.3] and bounce off in different directions."),
    ("mechanism", "+0.6", "The further they travel, the further apart they get. [p0.3] Then the next pin does it "
                          "again."),
    ("mechanism", "+0.8", "In the simulator, each bounce multiplies the gap by about two point two. [p0.4] Over a "
                          "whole drop, about thirty-six bounces, [p0.2] that's nearly a trillion times."),
    ("mechanism", "+0.7", "Doubling, over and over: [p0.3] that's exponential growth, [p0.3] the engine of chaos."),

    ("horizon", 0.5, "So how far ahead can you see? [p0.5] Know the start to a millimetre, and you can predict "
                     "about three bounces. [p0.4] To the width of a single atom, [p0.2] just twenty-four."),
    ("horizon", "+0.7", "To call a whole drop, [p0.2] you'd need the start to within the size of an atomic "
                        "nucleus."),

    ("crux", 0.5, "Here's the real test. [p0.5] Find a drop that wins, from anywhere across the top, [p0.3] then "
                  "replay it from exactly the same spot, as precisely as you like."),
    ("crux", "+0.8", "Match it to a micron, and it wins again just sixteen percent of the time, [p0.3] barely more "
                     "than a fresh drop. [p0.5] Match it to a single atom: [p0.3] thirty-nine percent. [p0.4] Still "
                     "worse than a coin flip."),
    ("crux", "+0.8", "More precision always helps. [p0.6] But certainty never arrives."),

    ("islands", 0.5, "On the show, you can let go anywhere across the top, [p0.2] at the pins or up to a foot "
                     "above. [p0.4] Each dot here is one of those spots. [p0.3] The gold ones win."),
    ("islands", 10.4, "Take one line through the middle, [p0.2] and two winning spots on it. [p0.3] Now zoom in, "
                      "a hundred times closer with each row."),
    ("islands", 20.6, "Down to the size of a virus, [p0.2] both look random."),
    ("islands", 24.6, "Only within a few atoms does spot two turn solid gold: [p0.3] an island. [p0.4] Spot one "
                      "never settles: [p0.3] a knife edge."),
    ("islands", "+0.8", "And one win can't tell you which kind you've found."),

    ("heat", 0.5, "Now let's make the start perfect, exact to the last atom, [p0.3] and change just one thing: "
                  "[p0.4] the temperature of the board."),
    ("heat", "+0.7", "Where would the heat come from? [p0.4] The chips themselves."),
    ("heat", "+0.6", "At the top, a chip holds about five joules of energy, just from its height. [p0.3] As it "
                     "falls, that becomes speed. [p0.3] And every time it hits a pin, some of that speed turns "
                     "into heat: [p0.2] in the pin, the board, and the chip."),
    ("heat", "+0.6", "When it stops, [p0.2] all five joules have become heat."),
    ("heat", "+0.8", "Spread through the board, that warms it by less than a ten-thousandth of a degree. [p0.4] "
                     "But warm wood and steel expand. [p0.3] The board grows by about a nanometre, [p0.2] the "
                     "width of ten atoms, [p0.3] and every pin shifts with it."),
    ("heat", "+0.8", "A nanometre is nothing, [p0.3] until the bounces double it, about twenty times over. [p0.4] "
                     "Then it's a centimetre: [p0.3] a different slot."),

    ("warm", 0.4, "So drop the same chip again, from exactly the same spot, on the warmer board. [p0.4] Gold is "
                  "the first drop. [p0.2] Blue is the replay."),
    ("warm", 8.2, "They fall as one, [p0.3] until the shifted pins catch up with them."),
    ("warm", 12.6, "Ten thousand dollars [p0.5] becomes zero."),

    ("realworld", 0.5, "Try it with hundreds of winning drops. [p0.3] Warm the board by just a billionth of a "
                       "degree, [p0.2] and six replays in a hundred already miss. [p0.4] One drop's heat [p0.2] "
                       "spoils more than half."),
    ("realworld", "+0.8", "And every drop warms the board. [p0.4] So in practice, a replay is barely better than a "
                          "fresh drop."),

    ("statistics", 0.5, "So one drop can't be predicted. [p0.5] But four hundred can."),
    ("statistics", 5.5, "Chaos scrambles every single path, [p0.3] yet the pile they make comes out the same, drop "
                        "after drop."),
    ("statistics", 15.2, "It matches the show's real chips, [p0.3] and it's why the middle still pays best."),

    ("outro", 0.3, "So next time you see Plinko on The Price Is Right, [p0.3] drop it in the middle, [p0.3] and let "
                   "chaos decide."),
    ("outro", "+0.9", "Deterministic. [p0.7] Unpredictable. [p0.9] That's chaos."),
]


def phrases(text):
    """'a [p0.5] b' -> [('a', 0.5), ('b', 0.0)]"""
    parts = re.split(r"\s*\[p(\d*\.?\d+)\]\s*", text.strip())
    out = []
    for k in range(0, len(parts), 2):
        out.append((parts[k], float(parts[k + 1]) if k + 1 < len(parts) else 0.0))
    return out


def plain(text):
    return re.sub(r"\s*\[p\d*\.?\d+\]\s*", " ", text).strip()


def synth(path, text, prev, nxt, seed):
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return
    body = dict(text=text, model_id=MODEL, voice_settings=SETTINGS, seed=seed)
    if prev:
        body["previous_text"] = prev
    if nxt:
        body["next_text"] = nxt
    tmp = path + ".part"
    r = subprocess.run(["curl", "-sS", "-m", "120", "-X", "POST",
                        f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE}?output_format=mp3_44100_128",
                        "-H", "Content-Type: application/json", "-d", json.dumps(body), "-o", tmp,
                        "-w", "%{http_code}"], capture_output=True, text=True)
    if r.stdout.strip() != "200":
        raise SystemExit(f"ElevenLabs returned {r.stdout.strip()}: {open(tmp, errors='replace').read()[:400]}")
    os.replace(tmp, path)


def decode(path):
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
                       capture_output=True, check=True)
    return np.frombuffer(r.stdout, np.float32).copy()


def trim(a):
    """Cut leading silence and everything after the voice ends. Frames of 10 ms; voice = within 42 dB of the
    loudest frame. A final burst shorter than 0.35 s that follows a gap of 0.18 s or more is not speech."""
    hop = SR // 100
    n = len(a) // hop
    rms = np.sqrt(np.mean(a[:n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    on = db > max(db.max() - 42.0, -55.0)
    segs, k = [], 0
    while k < n:
        if on[k]:
            j = k
            while j < n and on[j]:
                j += 1
            if segs and k - segs[-1][1] < 6:               # gaps under 60 ms are inside a word
                segs[-1][1] = j
            else:
                segs.append([k, j])
            k = j
        else:
            k += 1
    while len(segs) > 1 and segs[-1][1] - segs[-1][0] < 35 and segs[-1][0] - segs[-2][1] >= 18:
        segs.pop()
    s0 = max(0, segs[0][0] * hop - int(0.03 * SR))
    s1 = min(len(a), segs[-1][1] * hop + int(0.07 * SR))
    out = a[s0:s1].copy()
    fi, fo = int(0.012 * SR), int(0.09 * SR)
    out[:fi] *= np.linspace(0, 1, fi)
    out[-fo:] *= 0.5 * (1 + np.cos(np.linspace(0, np.pi, fo)))
    return out


def build_line(i, shot, text):
    """Synthesise and trim each phrase, join them with the pauses: one float array per line."""
    ph = phrases(text)
    pieces, marks, t = [], [], 0.0
    for k, (p, pause) in enumerate(ph):
        prev = " ".join(x for x, _ in ph[:k]) or None
        nxt = " ".join(x for x, _ in ph[k + 1:]) or None
        key = json.dumps([VOICE, MODEL, SETTINGS, p, prev, nxt], sort_keys=True)
        h = hashlib.sha1(key.encode()).hexdigest()[:10]
        path = f"{OUT}/raw/{i:02d}_{k}_{shot}_{h}.mp3"
        synth(path, p, prev, nxt, seed=1000 + 37 * i + k)
        a = trim(decode(path))
        pieces.append(a)
        marks.append((round(t, 3), round(t + len(a) / SR, 3)))
        t += len(a) / SR
        if pause > 0:
            pieces.append(np.zeros(int(pause * SR), np.float32))
            t += int(pause * SR) / SR
    return np.concatenate(pieces), marks


def wav_write(path, a):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-", path],
                   input=a.astype(np.float32).tobytes(), check=True)


def lines():
    os.makedirs(f"{OUT}/raw", exist_ok=True)
    cues, prev_end, last_shot = {}, 0.0, None
    for i, (shot, start, text) in enumerate(SCRIPT):
        a, marks = build_line(i, shot, text)
        path = f"{OUT}/line_{i:02d}.wav"
        wav_write(path, a)
        d = len(a) / SR
        if shot != last_shot:
            prev_end, last_shot = 0.0, shot
        t0 = prev_end + float(start[1:]) if isinstance(start, str) else float(start)
        if t0 < prev_end + 0.2:
            print(f"  ! line {i} ({shot}) starts {prev_end + 0.2 - t0:.2f}s too early; pushed back")
            t0 = prev_end + 0.2
        cues.setdefault(shot, dict(lines=[], need=0.0))
        cues[shot]["lines"].append(dict(i=i, t0=round(t0, 3), t1=round(t0 + d, 3), path=path, text=plain(text),
                                        phrases=[(round(t0 + a0, 3), round(t0 + a1, 3)) for a0, a1 in marks]))
        cues[shot]["need"] = round(t0 + d + TAIL, 2)
        prev_end = t0 + d
        print(f"{i:2d} {shot:13s} {t0:6.2f}-{t0 + d:6.2f}  ({d:4.1f}s)  {plain(text)[:60]}")
    json.dump(cues, open(CUES, "w"), indent=1)
    print({k: v["need"] for k, v in cues.items()})
    return cues


def check():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cues = json.load(open(CUES))
    ls = [ln for v in cues.values() for ln in v["lines"]]
    cols = 6
    fig, axs = plt.subplots((len(ls) + cols - 1) // cols, cols, figsize=(24, 3 * ((len(ls) + cols - 1) // cols)))
    for ln, ax in zip(ls, axs.flat):
        a = decode(ln["path"])[-int(1.0 * SR):]
        ax.specgram(a + 1e-9, NFFT=512, Fs=SR, noverlap=384, cmap="magma", vmin=-140, vmax=-30)
        ax.set_ylim(0, 16000)
        ax.set_title(f"{ln['i']} ...{ln['text'][-26:]}", fontsize=9)
    for ax in list(axs.flat)[len(ls):]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"{OUT}/tails.png", dpi=55)
    print("wrote", f"{OUT}/tails.png")


def shot_starts():
    shots = json.load(open("out/anim3d_chaos/shots.json"))["shots"]
    t, out = 0.0, {}
    for s in shots:
        out[s["name"]] = (t, s["dur"])
        t += s["dur"]
    return out, t


def mix():
    cues = json.load(open(CUES))
    starts, total = shot_starts()
    track = np.zeros(int(total * SR) + SR, np.float32)
    for shot, v in cues.items():
        t_shot, dur = starts[shot]
        for ln in v["lines"]:
            a = decode(ln["path"])
            if ln["t1"] > dur + 0.05:
                print(f"  ! line {ln['i']} runs {ln['t1'] - dur:.2f}s past the end of {shot}")
            k = int(round((t_shot + ln["t0"]) * SR))
            track[k:k + len(a)] += a[:len(track) - k]
    track = track[:int(total * SR)]
    wav_write(f"{OUT}/raw_track.wav", track)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", f"{OUT}/raw_track.wav", "-af",
                    "highpass=f=70,loudnorm=I=-16:TP=-1.5:LRA=11", "-ar", "48000", "-ac", "2", f"{OUT}/track.wav"],
                   check=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", "out/plinko_chaos.mp4", "-i", f"{OUT}/track.wav",
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
                    "-movflags", "+faststart", "out/plinko_chaos_narrated.mp4"], check=True)
    print("wrote out/plinko_chaos_narrated.mp4")


def script_md():
    cues = json.load(open(CUES))
    starts, total = shot_starts()
    names = {"hook": "Same start, different finish", "deterministic": "No dice inside",
             "twins": "Two chips a thousandth of a millimetre apart", "chaos": "What chaos theory is",
             "mechanism": "Why round pins make chaos", "horizon": "How far ahead can you see?",
             "crux": "Replaying a winning drop", "islands": "Every place you could let go",
             "heat": "Where the heat comes from", "warm": "Perfect start, one thing changed",
             "realworld": "How much warmer is too warm?", "statistics": "One drop vs four hundred",
             "outro": "Deterministic, and unpredictable"}
    out = ["# Narration: Plinko and chaos (\"Same start, different finish\")", "",
           f"Voice: ElevenLabs \"Lily\" ({VOICE}), model `{MODEL}`. Built by `plinko/narration.py`; each line "
           "starts at the time shown (film time). \"…\" marks a deliberate pause.", ""]
    for shot, (t_shot, dur) in starts.items():
        if shot not in cues:
            continue
        out += ["", f"## {int(t_shot // 60)}:{int(t_shot % 60):02d} {names.get(shot, shot)}", ""]
        for ln in cues[shot]["lines"]:
            t = t_shot + ln["t0"]
            txt = re.sub(r"\s*\[p\d*\.?\d+\]\s*", " … ", SCRIPT[ln["i"]][2]).strip()
            out.append(f"* **{int(t // 60)}:{t % 60:04.1f}** {txt}")
    open("../videos/copy/plinko_chaos_NARRATION.md", "w").write("\n".join(out) + "\n")
    print("wrote ../videos/copy/plinko_chaos_NARRATION.md")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "lines"
    {"lines": lines, "check": check, "mix": mix, "script": script_md}[what]()
