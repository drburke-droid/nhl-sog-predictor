"""Score and picture together:  python assemble.py [--career] [render [jobs]]
-> ../media/paid/best_paid_athletes.mp4 (the yearly race), best_paid_careers.mp4 (--career)"""
import os, sys, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import video as V      # noqa: E402

FINAL = {"race": "best_paid_athletes.mp4", "careers": "best_paid_careers.mp4"}
AF = "volume=-1.2dB,alimiter=limit=0.708:attack=3:release=60:level=false"   # peaks under -3 dBFS before AAC


def main():
    import music
    import score as SC
    args = sys.argv[1:]
    if args and args[0] == "--career":
        V.configure("career")
        args = args[1:]
    film = V.MODE["film"]
    if args and args[0] == "render":
        V.render(int(args[1]) if len(args) > 1 else 4)
    wav = os.path.join(V.OUT, f"{film}_score.wav")
    SC.write(music.track(), wav)
    final = os.path.join(os.path.dirname(HERE), "media", "paid", FINAL[film])
    os.makedirs(os.path.dirname(final), exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", os.path.join(V.OUT, f"{film}.mp4"), "-i", wav, "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-af", AF, "-c:a", "aac", "-b:a", "320k", "-shortest", "-movflags", "+faststart", final], check=True)
    print("wrote", final)


if __name__ == "__main__":
    main()
