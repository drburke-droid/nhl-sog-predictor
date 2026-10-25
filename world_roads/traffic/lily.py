"""Record the narration with ElevenLabs' Lily voice:  python lily.py [--dry-run] [section ...]
-> ../media/traffic/voice/<section>_<nn>.mp3, one clip per line of VOICEOVER.md, then python assemble.py mixes them in.

Needs ELEVENLABS_API_KEY in the environment (a key limited to text to speech is enough). Each line is sent with
the lines either side as context, so the reading flows across clips. Clips are cached by their text and settings:
a re-run only records lines that changed. A clip too long for its slot is sped up a little (at most 12%,
pitch kept); anything that would need more is reported, to trim in voiceover.py.
"""
import os, sys, json, hashlib, subprocess, urllib.request, urllib.error
import voiceover as VO
import video as V

VOICE_ID = "pFZP5JQG7iQjIQuC4Bku"          # Lily - Velvety Actress (ElevenLabs premade: British, informative)
MODEL = "eleven_multilingual_v2"
SETTINGS = dict(stability=0.5, similarity_boost=0.75, style=0.1, use_speaker_boost=True, speed=1.0)
OUT = os.path.join(V.L.MEDIA, "voice")
MAX_TEMPO = 1.12
API = "https://api.elevenlabs.io/v1/text-to-speech/{v}?output_format=mp3_44100_128"


def tts(text, prev, nxt, key):
    body = dict(text=text, model_id=MODEL, voice_settings=SETTINGS)
    if prev:
        body["previous_text"] = prev
    if nxt:
        body["next_text"] = nxt
    req = urllib.request.Request(API.format(v=VOICE_ID), data=json.dumps(body).encode(), method="POST",
                                 headers={"xi-api-key": key, "Content-Type": "application/json", "Accept": "audio/mpeg"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:300]
            if e.code in (429, 500, 502, 503) and attempt < 3:
                import time
                time.sleep(2 ** (attempt + 1))
                continue
            raise SystemExit(f"ElevenLabs refused the request ({e.code}): {msg}")


def duration(path):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                                capture_output=True, text=True, check=True).stdout)


def fit_clip(path, slot):
    """speed a clip up (pitch kept) if it overruns its slot; returns (seconds, tempo applied)"""
    d = duration(path)
    if d <= slot:
        return d, 1.0
    tempo = min(MAX_TEMPO, d / slot)
    tmp = path + ".tmp.mp3"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path, "-filter:a", f"atempo={tempo:.4f}", "-b:a", "128k", tmp], check=True)
    os.replace(tmp, path)
    return duration(path), tempo


def main(args):
    dry = "--dry-run" in args
    only = [a for a in args if not a.startswith("--")]
    VO.check()
    rows, O = VO.fit()
    if only:
        rows = [r for r in rows if r[0] in only]
    chars = sum(len(r[3]) for r in rows)
    print(f"{len(rows)} lines, {chars:,} characters ({MODEL}, voice {VOICE_ID})")
    key = os.environ.get("ELEVENLABS_API_KEY")
    if dry:
        return
    if not key:
        raise SystemExit("ELEVENLABS_API_KEY is not set: add it to the environment's secrets, then start a new session")
    os.makedirs(OUT, exist_ok=True)
    cache_p = os.path.join(OUT, "cache.json")
    cache = json.load(open(cache_p)) if os.path.exists(cache_p) else {}
    allrows = VO.fit()[0]
    report = []
    for sec, k, cue, txt, end, nxt, ok in rows:
        j = next(i for i, r in enumerate(allrows) if r[0] == sec and r[1] == k)
        prev = allrows[j - 1][3] if j > 0 and allrows[j - 1][0] == sec else None
        after = allrows[j + 1][3] if j + 1 < len(allrows) and allrows[j + 1][0] == sec else None
        name = f"{sec}_{k:02d}"
        path = os.path.join(OUT, name + ".mp3")
        h = hashlib.sha1(json.dumps([txt, prev, after, VOICE_ID, MODEL, SETTINGS]).encode()).hexdigest()
        slot = nxt - cue - 0.1
        if cache.get(name) != h or not os.path.exists(path):
            open(path, "wb").write(tts(txt, prev, after, key))
            cache[name] = h
            json.dump(cache, open(cache_p, "w"), indent=1)
            got = "recorded"
        else:
            got = "cached"
        d, tempo = fit_clip(path, slot)
        flag = "" if d <= slot + 0.05 else f"  ⚠ {d - slot:.1f} s over: trim this line"
        report.append((name, d, slot, tempo, flag))
        print(f"  {name:11s} {got:8s} {d:5.1f} s of {slot:5.1f}" + (f"  sped up ×{tempo:.2f}" if tempo > 1 else "") + flag)
    over = [r for r in report if r[4]]
    print(f"done: {len(report)} clips in {OUT}" + (f"; {len(over)} still too long" if over else ""))


if __name__ == "__main__":
    main(sys.argv[1:])
