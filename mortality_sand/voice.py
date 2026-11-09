"""The voiceover: a few words on each great catastrophe as it happens (inputs/narration.csv).

    python voice.py tts            # speak every line (ElevenLabs, cached in data/vo/) -> data/vo/lines.json
    python build_data.py           # the clock makes room for each line (build_clock: dwell)
    python voice.py mix [video]    # the narration track, mixed onto the film -> out/<film>_narrated.mp4

Each line in inputs/narration.csv starts at a calendar year (`at`), or during the title ("title") or the
end card ("end"). The clock (build_data.py) slows where a line would not finish before the next one
begins, so the voice never has to hurry and never talks over itself.

The voice is ElevenLabs' "Lily" (British, female; VOICE_ID). The API key is read from the environment
(ELEVEN_LABS_API or ELEVENLABS_API_KEY).
"""
import csv, hashlib, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
VO = os.path.join(DATA, "vo")
INP = os.path.join(HERE, "inputs")
OUT = os.path.join(HERE, "out")

VOICE_ID = "pFZP5JQG7iQjIQuC4Bku"       # Lily (British, female)
MODEL = "eleven_multilingual_v2"
SETTINGS = dict(stability=0.55, similarity_boost=0.8, style=0.15, use_speaker_boost=True, speed=0.95)
GAP = 0.7                                 # seconds of quiet the clock keeps after each line
LEAD = 0.15                               # a line starts this long after its moment
TITLE_AT = 0.9                            # video second a "title" line starts
END_AT = 1.2                              # seconds into the end card an "end" line starts


def read_lines():
    out = []
    for r in csv.DictReader(open(os.path.join(INP, "narration.csv"), encoding="utf-8")):
        t = r["text"].strip()
        if not t or r["id"].startswith("#"):
            continue
        at = r["at"].strip()
        out.append(dict(id=r["id"].strip(), at=at if at in ("title", "end") else float(at), text=t))
    return out


def _key(text):
    h = hashlib.sha1(json.dumps([VOICE_ID, MODEL, SETTINGS, text], sort_keys=True).encode()).hexdigest()
    return h[:14]


def duration(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                       capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


def speak(text, path):
    import urllib.request
    key = os.environ.get("ELEVEN_LABS_API") or os.environ.get("ELEVENLABS_API_KEY") or ""
    req = urllib.request.Request(
        f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}?output_format=mp3_44100_128",
        data=json.dumps(dict(text=text, model_id=MODEL, voice_settings=SETTINGS)).encode(),
        headers={"xi-api-key": key, "Content-Type": "application/json", "Accept": "audio/mpeg"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    tmp = path + ".part"
    open(tmp, "wb").write(data)
    os.replace(tmp, path)


def cmd_tts():
    os.makedirs(VO, exist_ok=True)
    lines = read_lines()
    for L in lines:
        f = os.path.join(VO, _key(L["text"]) + ".mp3")
        if not os.path.exists(f):
            print("speaking:", L["id"], "-", L["text"][:70], flush=True)
            speak(L["text"], f)
        L["file"] = os.path.relpath(f, HERE)
        L["dur"] = round(duration(f), 3)
    json.dump(lines, open(os.path.join(VO, "lines.json"), "w"), indent=1)
    tot = sum(L["dur"] for L in lines)
    print(f"{len(lines)} lines, {tot:.0f} s of voice -> data/vo/lines.json")


def load_lines():
    p = os.path.join(VO, "lines.json")
    return json.load(open(p)) if os.path.exists(p) else []


def starts():
    """Each line's start on the film's clock (video seconds)."""
    import story as S
    import sand as Sd
    end0 = S.CLOCK["s"][-1]
    out = []
    for L in load_lines():
        if L["at"] == "title":
            s = TITLE_AT
        elif L["at"] == "end":
            s = end0 + END_AT
        else:
            s = Sd.s_at(L["at"]) + LEAD
        out.append((s, L))
    return out


def srt_time(s):
    ms = int(round(s * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def cmd_mix(video=None):
    import story as S
    film = os.path.join(OUT, "two_thousand_years_of_catastrophe.mp4")
    video = video or film
    st = starts()
    dur = S.DURATION
    # subtitles too
    with open(os.path.join(OUT, "narration.srt"), "w", encoding="utf-8") as f:
        for i, (s, L) in enumerate(st, 1):
            f.write(f"{i}\n{srt_time(s)} --> {srt_time(s + L['dur'])}\n{L['text']}\n\n")
    args = ["ffmpeg", "-y", "-v", "error", "-i", video]
    filt = []
    for i, (s, L) in enumerate(st, 1):
        args += ["-i", os.path.join(HERE, L["file"])]
        ms = int(round(s * 1000))
        filt.append(f"[{i}:a]aresample=48000,adelay={ms}|{ms},apad[a{i}]")
    mix = "".join(f"[a{i}]" for i in range(1, len(st) + 1))
    filt.append(f"{mix}amix=inputs={len(st)}:normalize=0:duration=longest,"
                f"atrim=0:{dur:.3f},loudnorm=I=-18:TP=-1.5:LRA=11[vo]")
    out = os.path.splitext(video)[0] + "_narrated.mp4"
    args += ["-filter_complex", ";".join(filt), "-map", "0:v", "-map", "[vo]", "-c:v", "copy", "-c:a", "aac",
             "-b:a", "192k", "-shortest", "-movflags", "+faststart", out]
    subprocess.run(args, check=True)
    print("wrote", out, "and out/narration.srt")
    # overlaps (should be none: the clock makes room)
    for (s0, a), (s1, b) in zip(st, st[1:]):
        if s0 + a["dur"] > s1 - 0.2:
            print(f"  overlap: {a['id']} ends {s0 + a['dur']:.1f} s, {b['id']} starts {s1:.1f} s")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "tts"
    if cmd == "tts":
        cmd_tts()
    elif cmd == "mix":
        cmd_mix(sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        sys.exit(__doc__)
