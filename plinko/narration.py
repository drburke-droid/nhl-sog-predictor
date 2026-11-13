"""Voice-over for the chaos cut, read by ElevenLabs' "Lily" voice.

    python3 narration.py lines     # synthesise each line (cached), report timing against the shots
    python3 narration.py mix       # -> out/narration/track.wav, out/plinko_chaos_narrated.mp4
    python3 narration.py script    # -> ../videos/copy/plinko_chaos_NARRATION.md

Each line is placed at a time within its shot (seconds from the shot's start) so it lands with the picture.
Numbers are spelled the way they should be read, and match the figures on screen (results/chaos2_*.json).
The API key is supplied by the environment's proxy for api.elevenlabs.io; nothing here stores one.
"""
import hashlib
import json
import os
import subprocess
import sys

VOICE = "pFZP5JQG7iQjIQuC4Bku"          # Lily (British, warm, educational)
MODEL = "eleven_multilingual_v2"
SETTINGS = dict(stability=0.55, similarity_boost=0.75, style=0.15, use_speaker_boost=True, speed=1.0)
OUT = "out/narration"
SHOTS = json.load(open("out/anim3d_chaos/shots.json"))["shots"]

SCRIPT = [
    ("hook", 0.4, "This is Plinko, from The Price Is Right. And this drop wins ten thousand dollars."),
    ("hook", 6.6, "Now the same drop again, from the same spot, to the width of an atom."),
    ("hook", 12.0, "Same start. Different finish. That's chaos theory."),
    ("deterministic", 0.4, "There are no dice in here. Every bounce follows Newton's laws. So if the replay lost, "
                           "its start wasn't quite the same."),
    ("twins", 0.4, "Two chips, a thousandth of a millimetre apart. They fall together, then split."),
    ("twins", 6.0, "One wins ten thousand dollars. The other, a hundred."),
    ("twins", 10.8, "Chaos theory calls this the butterfly effect."),
    ("mechanism", 0.4, "Why so touchy? Round pins. Paths that hit a pin almost together leave it further apart."),
    ("mechanism", 7.0, "Each bounce more than doubles the gap. Over a whole drop, that's nearly a trillion times."),
    ("mechanism", 13.6, "That exponential growth is the signature of chaos."),
    ("horizon", 0.4, "How far ahead can you see? Know the start to a millimetre: three bounces. To a single atom: "
                     "just twenty-four."),
    ("horizon", 8.6, "A whole drop would take the precision of an atomic nucleus."),
    ("crux", 0.4, "Here's the real test. Find a drop that wins, then replay it from exactly the same spot, as "
                  "precisely as you like."),
    ("crux", 9.2, "Match it to a micron, and it wins again just sixteen percent of the time, barely more than a "
                  "fresh drop. Match it to a single atom: thirty-nine percent. Still worse than a coin flip."),
    ("crux", 22.6, "More precision always helps. But certainty never arrives."),
    ("islands", 0.4, "On the show, you can let go anywhere across the top, at the pins or up to a foot above. Each "
                     "dot is one of those spots. Gold ones win."),
    ("islands", 9.8, "Take one line through the middle, and two winning spots on it. Now zoom in, a hundred times "
                     "closer each row."),
    ("islands", 20.0, "Down to the size of a virus, both look random."),
    ("islands", 24.4, "Only within a few atoms does spot two turn solid gold: an island. Spot one never settles: a "
                      "knife edge."),
    ("islands", 33.6, "One win can't tell you which kind you've found."),
    ("warm", 0.4, "Now make the start perfect, and change just one thing: the board's temperature."),
    ("warm", 6.6, "The first drop's own heat warms it by less than a ten-thousandth of a degree."),
    ("warm", 12.3, "Ten thousand dollars becomes zero."),
    ("realworld", 0.4, "Warm the board, and replay hundreds of winning drops exactly. A billionth of a degree already "
                       "spoils six in a hundred. One drop's heat spoils more than half."),
    ("realworld", 12.6, "And every drop warms the board. So in practice, a replay is barely better than a fresh drop."),
    ("statistics", 0.5, "So one drop can't be predicted. But four hundred can."),
    ("statistics", 5.0, "Chaos scrambles every single path, yet the pile they make comes out the same, drop after "
                        "drop."),
    ("statistics", 15.2, "It matches the show's real chips, and it's why the middle still pays best."),
    ("outro", 0.3, "Next time you see Plinko on The Price Is Right, drop it in the middle, and let chaos decide."),
    ("outro", 5.6, "Deterministic. Unpredictable. That's chaos."),
]


def shot_starts():
    t, out = 0.0, {}
    for s in SHOTS:
        out[s["name"]] = (t, s["dur"])
        t += s["dur"]
    return out, t


def line_path(i, shot, text, prev, nxt):
    key = json.dumps([VOICE, MODEL, SETTINGS, text, prev, nxt], sort_keys=True)
    h = hashlib.sha1(key.encode()).hexdigest()[:10]
    return f"{OUT}/{i:02d}_{shot}_{h}.mp3"


def synth(path, text, prev, nxt):
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return
    body = dict(text=text, model_id=MODEL, voice_settings=SETTINGS)
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
        raise SystemExit(f"ElevenLabs returned {r.stdout.strip()}: {open(tmp).read()[:400]}")
    os.replace(tmp, path)


def duration(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                       capture_output=True, text=True)
    return float(r.stdout)


def lines():
    os.makedirs(OUT, exist_ok=True)
    starts, total = shot_starts()
    rows = []
    for i, (shot, off, text) in enumerate(SCRIPT):
        same = [s for s in SCRIPT if s[0] == shot]
        k = same.index((shot, off, text))
        prev = same[k - 1][2] if k > 0 else None
        nxt = same[k + 1][2] if k + 1 < len(same) else None
        p = line_path(i, shot, text, prev, nxt)
        synth(p, text, prev, nxt)
        t0 = starts[shot][0] + off
        d = duration(p)
        rows.append(dict(i=i, shot=shot, path=p, t0=t0, dur=d, end=t0 + d, text=text,
                         shot_end=starts[shot][0] + starts[shot][1]))
    ok = True
    for a, b in zip(rows, rows[1:] + [None]):
        nxt_t = b["t0"] if b else total
        flag = ""
        if a["end"] > nxt_t - 0.25:
            flag = f"  <-- overlaps next line by {a['end'] - nxt_t + 0.25:.2f}s"
            ok = False
        elif a["end"] > a["shot_end"] - 0.1:
            flag = f"  (runs {a['end'] - a['shot_end'] + 0.1:.2f}s into the next shot)"
        print(f"{a['i']:2d} {a['shot']:13s} {a['t0']:6.1f}-{a['end']:6.1f} ({a['dur']:4.1f}s){flag}")
    json.dump(rows, open(f"{OUT}/timing.json", "w"), indent=1)
    print("all lines fit" if ok else "SOME LINES OVERLAP")
    return rows, total


def mix():
    rows, total = lines()
    inputs, filt = [], []
    for k, r in enumerate(rows):
        inputs += ["-i", r["path"]]
        ms = int(round(r["t0"] * 1000))
        filt.append(f"[{k}:a]aresample=48000,adelay={ms}|{ms},apad[a{k}]")
    filt.append("".join(f"[a{k}]" for k in range(len(rows))) +
                f"amix=inputs={len(rows)}:normalize=0:duration=longest,atrim=0:{total},"
                f"loudnorm=I=-16:TP=-1.5:LRA=11[out]")
    subprocess.run(["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", ";".join(filt), "-map", "[out]",
                    "-ar", "48000", "-ac", "2", f"{OUT}/track.wav"], check=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", "out/plinko_chaos.mp4", "-i", f"{OUT}/track.wav",
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
                    "-movflags", "+faststart", "out/plinko_chaos_narrated.mp4"], check=True)
    print("wrote out/plinko_chaos_narrated.mp4")


def script_md():
    starts, total = shot_starts()
    names = {"hook": "Same start, different finish", "deterministic": "No dice inside",
             "twins": "Two chips a thousandth of a millimetre apart", "mechanism": "Why round pins make chaos",
             "horizon": "How far ahead can you see?", "crux": "Replaying a winning drop",
             "islands": "Every place you could let go", "warm": "Perfect start, one thing changed",
             "realworld": "How much warmer is too warm?", "statistics": "One drop vs four hundred",
             "outro": "Deterministic, and unpredictable"}
    out = ["# Narration: Plinko and chaos (\"Same start, different finish\")", "",
           f"Voice: ElevenLabs \"Lily\" ({VOICE}), model `{MODEL}`. Built by `plinko/narration.py`; each line "
           "starts at the time shown (film time) so it lands with the picture.", ""]
    last = None
    for shot, off, text in SCRIPT:
        if shot != last:
            t0 = starts[shot][0]
            out += ["", f"## {int(t0 // 60)}:{int(t0 % 60):02d} {names[shot]}", ""]
            last = shot
        t = starts[shot][0] + off
        out.append(f"* **{int(t // 60)}:{t % 60:04.1f}** {text}")
    open("../videos/copy/plinko_chaos_NARRATION.md", "w").write("\n".join(out) + "\n")
    print("wrote ../videos/copy/plinko_chaos_NARRATION.md")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "lines"
    {"lines": lines, "mix": mix, "script": script_md}[what]()
