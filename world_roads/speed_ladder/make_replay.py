"""Composite every replay frame (in parallel), join test clip + replay, add the sound -> /home/user/lot/out/test2_sound.mp4"""
import os, sys, subprocess, multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUT = "/home/user/lot/out"


def work(rng):
    import replay_comp as RC, replay_time as T, cv2
    tl = T.timeline()
    for idx in rng:
        t, slow = tl[idx]
        cv2.imwrite(f"{RC.OUT}/r{idx:04d}.png", RC.frame(idx, t, slow, 0.5 / (T.FPS * slow))[..., ::-1])


if __name__ == "__main__":
    import replay_time as T
    n = len(T.timeline())
    jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    with mp.Pool(jobs) as p:
        p.map(work, [range(k, n, jobs) for k in range(jobs)])
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "25", "-i", f"{OUT}/replay/r%04d.png", "-c:v", "libx264",
                    "-crf", "16", "-pix_fmt", "yuv420p", f"{OUT}/replay.mp4"], check=True)
    with open(f"{OUT}/concat.txt", "w") as f:
        f.write(f"file '{OUT}/test1.mp4'\nfile '{OUT}/replay.mp4'\n")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", f"{OUT}/concat.txt", "-c", "copy",
                    f"{OUT}/test2.mp4"], check=True)
    dur = (145 + n) / 25.0
    subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "sound.py"), str(dur)], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{OUT}/test2.mp4", "-i", f"{OUT}/test2.wav", "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "192k", "-shortest", f"{OUT}/test2_sound.mp4"], check=True)
    print("done", n, "replay frames")
