"""Lily's lines to WAV, one file per line (a line already said in the same words is kept).
  ttsenv/bin/python -I vo_tts.py lines.json outdir      (lines.json: {"id": "text", ...}; vo_script.py writes it)
Two voices: by default Kokoro-82M's bf_lily, offline (model files in /home/user/tts_models); with an ElevenLabs key in
the environment (ELEVENLABS_API_KEY, or any variable named for it), ElevenLabs' Lily (the premade voice of that name, eleven_multilingual_v2), into its own outdir."""
import json, os, sys, urllib.error, urllib.request
import numpy as np
import soundfile as sf

M = "/home/user/tts_models"
SR_EL = 24000


def kokoro():
    from kokoro_onnx import Kokoro
    voices = np.load(f"{M}/voices-v1.0.bin", allow_pickle=False)
    assert "bf_lily" in voices.files, voices.files
    k = Kokoro(f"{M}/kokoro-v1.0.onnx", f"{M}/voices-v1.0.bin")
    speed = float(os.environ.get("VO_SPEED", "0.95"))
    return lambda text: k.create(text, voice="bf_lily", speed=speed, lang="en-gb")


def elevenlabs(key):
    def call(path, body=None):
        req = urllib.request.Request("https://api.elevenlabs.io" + path, data=None if body is None else
                                     json.dumps(body).encode(), headers={"xi-api-key": key, "Content-Type":
                                     "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:                  # say why (quota, permissions): the body has no key
            raise SystemExit(f"ElevenLabs {e.code} on {path.split('?')[0]}: {e.read()[:400]!r}")
    voices = json.loads(call("/v1/voices"))["voices"]
    vid = next(v["voice_id"] for v in voices if v["name"].split(" ")[0].lower() == "lily")
    settings = {"stability": 0.55, "similarity_boost": 0.75, "style": 0.15, "use_speaker_boost": True}

    def say(text):
        pcm = call(f"/v1/text-to-speech/{vid}?output_format=pcm_{SR_EL}",
                   {"text": text, "model_id": "eleven_multilingual_v2", "voice_settings": settings})
        return np.frombuffer(pcm, "<i2").astype(np.float32) / 32768, SR_EL
    return say


def api_key():
    """ElevenLabs' key, under whatever name the environment gives it"""
    names = [n for n in ["ELEVENLABS_API_KEY"] + sorted(os.environ)
             if os.environ.get(n) and ("ELEVEN" in n.upper() or n.upper() == "XI_API_KEY")]
    names.sort(key=lambda n: not os.environ[n].startswith("sk_"))     # a key, not a key's ID, first
    return os.environ[names[0]] if names else None


def main(lines_path, out):
    key = api_key()
    say = elevenlabs(key) if key else kokoro()
    os.makedirs(out, exist_ok=True)
    for name, text in json.load(open(lines_path)).items():
        path = f"{out}/{name}.wav"
        if os.path.exists(path) and os.path.exists(path + ".txt") and open(path + ".txt").read() == text:
            continue
        samples, sr = say(text)
        sf.write(path, samples, sr)
        open(path + ".txt", "w").write(text)
        print(name, f"{len(samples) / sr:.2f} s", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
