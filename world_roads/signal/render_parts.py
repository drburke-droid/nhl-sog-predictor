"""Render one long-video section in parallel chunks and join them without re-encoding.

    python render_parts.py wave 4     -> out/long/wave.mp4
"""
import os, sys, subprocess, time
import numpy as np
from multiprocessing import Pool
from longform import OUT, W, H, FPS


def chunk(args):
    name, k, n = args
    from long_globe import SECTIONS
    sec = SECTIONS[name]()
    N = int(round(sec.dur * FPS))
    a, b = N * k // n, N * (k + 1) // n
    path = os.path.join(OUT, f"_{name}.part{k}.mp4")
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                          "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p", path],
                         stdin=subprocess.PIPE)
    for i in range(a, b):
        p.stdin.write(np.ascontiguousarray(sec.frame(i / FPS)).tobytes())
    p.stdin.close(); p.wait()
    return path


def main(name, n):
    t0 = time.time()
    with Pool(n) as pool:
        parts = pool.map(chunk, [(name, k, n) for k in range(n)])
    lst = os.path.join(OUT, f"_{name}.txt")
    open(lst, "w").write("".join(f"file '{os.path.basename(q)}'\n" for q in parts))
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
                    os.path.join(OUT, f"{name}.mp4")], check=True)
    for q in parts + [lst]:
        os.remove(q)
    print(f"wrote {name}.mp4 in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 4)
