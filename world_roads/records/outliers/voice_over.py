"""Give a finished film a voice without re-rendering a frame: take its score from the finished video, voice its
narration script with ElevenLabs, dip the score under the voice, and put the new soundtrack on the existing video.
Needs only ffmpeg, numpy and scipy.

    ELEVENLABS_API_KEY=... python voice_over.py season    # videos/sports/biggest_statistical_outliers_10m07s.mp4
    ELEVENLABS_API_KEY=... python voice_over.py day       # videos/sports/biggest_single_day_outliers_9m49s.mp4

Writes <name>_voiced.mp4 next to the original. Voice: ELEVENLABS_VOICE_ID (default c6SfcYrb2t09NHXiT80T, the chosen
voice). Clips are cached in out/voice/ (season) or ../singlegame/film/out/voice/ (day), so a re-run with an edited
script only pays for changed lines.
"""
import json, os, re, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REC = os.path.dirname(HERE)
VIDEOS = os.path.normpath(os.path.join(REC, "..", "..", "videos", "sports"))
sys.path.insert(0, HERE)
import narrate                                               # noqa: E402

FILMS = {"season": (os.path.join(HERE, "VO_SCRIPT.md"), os.path.join(HERE, "out"),
                    "biggest_statistical_outliers_10m07s.mp4"),
         "day": (os.path.join(REC, "singlegame", "film", "VO_SCRIPT.md"), os.path.join(REC, "singlegame", "film", "out"),
                 "biggest_single_day_outliers_9m49s.mp4")}


def duration(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                       capture_output=True, text=True, check=True)
    return float(r.stdout)


def soundtrack(video, total):
    """the finished video's own score, as float32 stereo at 48 kHz"""
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", video, "-vn", "-ac", "2", "-ar", "48000", "-f", "f32le", "-"],
                       capture_output=True, check=True)
    x = np.frombuffer(r.stdout, np.float32).reshape(-1, 2)
    n = int(total * 48000)
    return np.pad(x, ((0, max(0, n - len(x))), (0, 0)))[:n].copy()


def loudnorm(src, dst, I=-16.0, TP=-1.5, LRA=11.0):
    """two-pass EBU R128 normalisation"""
    r = subprocess.run(["ffmpeg", "-hide_banner", "-i", src, "-af", f"loudnorm=I={I}:TP={TP}:LRA={LRA}:print_format=json",
                        "-f", "null", "-"], capture_output=True, text=True, check=True)
    m = json.loads(re.findall(r"\{[^{}]*\}", r.stderr)[-1])
    af = (f"loudnorm=I={I}:TP={TP}:LRA={LRA}:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
          f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-af", af, "-ar", "48000", "-c:a", "pcm_s16le", dst], check=True)


def main(which):
    script, out, name = FILMS[which]
    video = os.path.join(VIDEOS, name)
    total = duration(video)
    os.makedirs(out, exist_ok=True)
    narrate.build(script, out, total)                          # needs ELEVENLABS_API_KEY for lines not yet cached
    mix = narrate.duck_and_mix(soundtrack(video, total), os.path.join(out, "narration.wav"))
    from scipy.io import wavfile
    raw, wav = os.path.join(out, "voiced_raw.wav"), os.path.join(out, "voiced.wav")
    wavfile.write(raw, 48000, (np.clip(mix, -1, 1) * 32767).astype(np.int16))
    loudnorm(raw, wav)
    dst = video.replace(".mp4", "_voiced.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", video, "-i", wav, "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", dst], check=True)
    print(dst, f"{duration(dst):.1f} s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "season")
