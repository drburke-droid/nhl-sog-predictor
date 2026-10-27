"""Score and picture together -> ../media/fantasy/fantasy_best_seasons.mp4

    python assemble.py [render [jobs]]          the synthesized march (music.py), with the picture rendered first if asked
    python assemble.py --audio track.wav        a recording of your own instead (a licensed library track, say): it is
                                                loudness-matched (-14 LUFS), cut or padded to the film's length and
                                                faded out over its last 3 s
"""
import os, sys, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import video as V      # noqa: E402

FINAL = os.path.join(os.path.dirname(HERE), "media", "fantasy", "fantasy_best_seasons.mp4")
AF = "volume=-1.2dB,alimiter=limit=0.708:attack=3:release=60:level=false"   # peaks under -3 dBFS before AAC


def main():
    from orchestra import SC
    args = sys.argv[1:]
    wav = os.path.join(V.OUT, "score.wav")
    if args[:1] == ["--audio"]:
        V.setup()
        T = V.total()
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", args[1], "-af",
                        f"apad,atrim=0:{T:.3f},afade=t=out:st={T - 3:.3f}:d=3,loudnorm=I=-14:TP=-1.5:LRA=11", "-ar", "48000",
                        wav], check=True)
    else:
        import music
        if args[:1] == ["render"]:
            V.render(int(args[1]) if len(args) > 1 else 4)
        SC.write(music.track(), wav)
    os.makedirs(os.path.dirname(FINAL), exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", os.path.join(V.OUT, "race.mp4"), "-i", wav, "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-af", AF, "-c:a", "aac", "-b:a", "320k", "-shortest", "-movflags", "+faststart", FINAL], check=True)
    print("wrote", FINAL)


if __name__ == "__main__":
    main()
