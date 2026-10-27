"""Score and picture together:  python assemble.py [render [jobs]]  -> ../media/fantasy/fantasy_vorp_race.mp4"""
import os, sys, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import video as V      # noqa: E402

FINAL = os.path.join(os.path.dirname(HERE), "media", "fantasy", "fantasy_vorp_race.mp4")
AF = "volume=-1.2dB,alimiter=limit=0.708:attack=3:release=60:level=false"   # peaks under -3 dBFS before AAC


def main():
    import music
    import score as SC
    if len(sys.argv) > 1 and sys.argv[1] == "render":
        V.render(int(sys.argv[2]) if len(sys.argv) > 2 else 4)
    wav = os.path.join(V.OUT, "score.wav")
    SC.write(music.track(), wav)
    os.makedirs(os.path.dirname(FINAL), exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", os.path.join(V.OUT, "race.mp4"), "-i", wav, "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-af", AF, "-c:a", "aac", "-b:a", "320k", "-shortest", "-movflags", "+faststart", FINAL], check=True)
    print("wrote", FINAL)


if __name__ == "__main__":
    main()
