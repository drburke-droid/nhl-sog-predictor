"""Give a finished film a voice without re-rendering a frame: rebuild its score (deterministic), voice its narration
script with ElevenLabs, dip the score under the voice, and put the new soundtrack on the existing video.

    ELEVENLABS_API_KEY=... python voice_over.py season    # videos/sports/biggest_statistical_outliers_10m07s.mp4
    ELEVENLABS_API_KEY=... python voice_over.py day       # videos/sports/biggest_single_day_outliers_9m49s.mp4

Writes <name>_voiced.mp4 next to the original. Voice: ELEVENLABS_VOICE_ID (default c6SfcYrb2t09NHXiT80T, the chosen voice); clips are cached in
out/voice/ (season) or ../singlegame/film/out/voice/ (day), so a re-run with an edited script only pays for changed lines.
"""
import os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
REC = os.path.dirname(HERE)
VIDEOS = os.path.join(os.path.dirname(REC), "..", "videos", "sports")
sys.path.insert(0, HERE)


def main(which):
    import orchestra, narrate
    import assemble as A
    if which == "season":
        O, total = A.timeline()
        out = os.path.join(HERE, "out")
        script = os.path.join(HERE, "VO_SCRIPT.md")
        video = os.path.join(VIDEOS, "biggest_statistical_outliers_10m07s.mp4")
    else:
        sys.path.insert(0, os.path.join(REC, "singlegame", "film"))
        import sd_assemble as SD
        O, total = SD.timeline()
        out = SD.OUT
        script = os.path.join(REC, "singlegame", "film", "VO_SCRIPT.md")
        video = os.path.join(VIDEOS, "biggest_single_day_outliers_9m49s.mp4")
    os.makedirs(out, exist_ok=True)
    orchestra.OUT = out
    narrate.build(script, out, total)                          # needs ELEVENLABS_API_KEY for lines not yet cached
    mix = narrate.duck_and_mix(orchestra.render(O, total), os.path.join(out, "narration.wav"))
    raw, wav = os.path.join(out, "voiced_raw.wav"), os.path.join(out, "voiced.wav")
    orchestra.write(mix, raw)
    A.loudnorm(raw, wav)
    dst = video.replace(".mp4", "_voiced.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", video, "-i", wav, "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", dst], check=True)
    print(dst)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "season")
