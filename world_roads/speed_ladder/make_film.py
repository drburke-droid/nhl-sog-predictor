"""The whole film: a title over the empty road, the five machines in order of speed (each shot opens on the empty road,
so they cut straight together), the finale's end card. One sound mix: the shots' tracks at levels that keep the
escalation (a 12 hp engine is quiet next to a sonic boom), each limited so nothing clips.
  python make_film.py -> /home/user/lot/out/history_fastest_machines.mp4"""
import cv2, numpy as np, os, subprocess, sys, wave
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import comp as C, fly_sound as S

OUT = "/home/user/lot/out"
SHOTS = [("wright", 0.45), ("spitfire", 0.8), ("x1", 1.0), ("x15", 1.0), ("parker", 1.0)]   # name, level in the mix
TITLE = ("What if history's fastest machines", "flew past you?")
WHITE = (255, 255, 255)


def read_wav(path):
    with wave.open(path) as f:
        x = np.frombuffer(f.readframes(f.getnframes()), "<i2").reshape(-1, 2).astype(np.float64) / 32767
    return x


def title_frames(seconds=4.0):
    st = C.props("/home/user/lot/render/fly_wright/static.npz")
    nf = int(seconds * 25)
    frames = []
    for n in range(nf):
        img = C.over_props(C.plate_loop(n), C.plate_match(st, 300 + n))
        img = C.grain(C.bloom(img), 300 + n)
        pic = (C.srgb(img) * 255 + 0.5).astype(np.uint8)
        o = np.zeros((C.H, C.W, 3), np.uint8); o[C.TOP:C.TOP + C.PH] = pic
        a = min(1.0, max(0.0, (n - 8) / 12), (nf - 1 - n) / 10)
        pil = Image.fromarray(o).convert("RGBA")
        ov = Image.new("RGBA", pil.size, (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
        for i, line in enumerate(TITLE):
            f = C.font(66, "Bold")
            w = d.textlength(line, font=f)
            x, y = (C.W - w) / 2, C.TOP + 120 + i * 82
            d.text((x + 2, y + 3), line, font=f, fill=(0, 0, 0, int(120 * a)))
            d.text((x, y), line, font=f, fill=WHITE + (int(255 * a),))
        frames.append(np.asarray(Image.alpha_composite(pil, ov).convert("RGB")))
    return frames


if __name__ == "__main__":
    import shutil
    fr = f"{OUT}/frames_film"
    shutil.rmtree(fr, ignore_errors=True); os.makedirs(fr)
    k = 0
    tf = title_frames()
    for im in tf:
        cv2.imwrite(f"{fr}/f{k:05d}.png", im[..., ::-1]); k += 1
    sr = S.SR
    audio = [S.wind(int(len(tf) / 25 * sr), np.random.default_rng(1), 0.035)]
    for name, level in SHOTS:
        src = f"{OUT}/frames_{name}"
        files = sorted(f for f in os.listdir(src) if f.startswith("f"))
        for f in files:
            os.link(f"{src}/{f}", f"{fr}/f{k:05d}.png")
            k += 1
        a = read_wav(f"{OUT}/fly_{name}.wav")
        n = int(len(files) / 25 * sr)
        a = np.pad(a, ((0, max(0, n - len(a))), (0, 0)))[:n]
        audio.append(a * level)
    mix = np.concatenate(audio)
    mix = np.tanh(mix * 1.1) / 1.1 * 0.95
    S.write(f"{OUT}/film.wav", mix)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "25", "-i", f"{fr}/f%05d.png", "-i", f"{OUT}/film.wav",
                    "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest",
                    f"{OUT}/history_fastest_machines.mp4"], check=True)
    print("frames", k, "seconds", k / 25)
