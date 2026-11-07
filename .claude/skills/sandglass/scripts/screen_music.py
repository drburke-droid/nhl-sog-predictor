"""Screen generated score takes: mean loudness per story section and any dropouts.

    python3 screen_music.py take1.mp3 take2.mp3 ... [--sections "0:37.4:origins,37.4:83.7:spread,..."]

A good take gets louder toward the climax (the flight), never dips there, and keeps sounding to the
last second. Dropouts are 1 s or more below -38 dB.
"""
import subprocess, sys
import numpy as np

DEFAULT = [(0, 37.4, "origins"), (37.4, 83.7, "spread"), (83.7, 131.6, "farm/voyage"), (131.6, 160, "1500-modern"),
           (160, 171, "descent"), (171, 216, "flight"), (216, 222, "end")]
args = sys.argv[1:]
beats = DEFAULT
if "--sections" in args:
    i = args.index("--sections")
    beats = [(float(a), float(b), n) for a, b, n in (s.split(":") for s in args[i + 1].split(","))]
    args = args[:i] + args[i + 2:]
for f in args:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", f, "-ac", "1", "-ar", "8000", "-f", "f32le", "-"],
                         capture_output=True).stdout
    x = np.frombuffer(raw, np.float32); sr = 8000
    if not len(x):
        print(f, "empty or not audio (an error response?)"); continue
    w = sr // 4
    db = np.array([20 * np.log10(max(np.sqrt(np.mean(x[i * w:(i + 1) * w] ** 2)), 1e-6)) for i in range(len(x) // w)])
    runs, r = [], 0
    for i, v in enumerate(db < -38):
        r = r + 1 if v else 0
        if r == 4:
            runs.append(round((i - 3) / 4, 1))
    print(f"{f}: {len(x) / sr:.1f}s  dropouts at {runs}")
    print("   " + "  ".join(f"{n} {np.mean(db[int(a * 4):int(b * 4)]):.0f}" for a, b, n in beats))
    print("   last 8 s by second: " + " ".join(f"{v:.0f}" for v in
          [20 * np.log10(max(np.sqrt(np.mean(x[int(t * sr):int((t + 1) * sr)] ** 2)), 1e-6)) for t in
           range(max(0, int(len(x) / sr) - 8), int(len(x) / sr))]))
