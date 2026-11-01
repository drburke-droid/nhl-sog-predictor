"""What every flyby shot shares: paths, the picture (2.39:1 letterboxed in 1920x1080, as in the speed-ladder
flybys), colour conversions, the speed label, writing frames to video."""
import os, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

WORK = os.environ.get("B2_WORK", "/home/user/lot/b2")
FPS = 30
W, H = 1920, 1080
PIC_H = 804                       # 2.39:1 picture, black bands above and below
BAND = (H - PIC_H) // 2
MPH = 0.44704                     # m/s per mph

FONT_DIRS = ["/usr/share/fonts", "/usr/local/share/fonts", os.path.expanduser("~/.fonts")]


def font(name, size):
    for d in FONT_DIRS:
        for root, _, files in os.walk(d):
            for f in files:
                if f.lower() == name.lower():
                    return ImageFont.truetype(os.path.join(root, f), size)
    return ImageFont.load_default()


def to_lin(a):
    """sRGB 0..1 -> linear"""
    a = np.clip(a, 0, 1)
    return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)


def to_srgb(a):
    a = np.clip(a, 0, 1)
    return np.where(a <= 0.0031308, a * 12.92, 1.055 * a ** (1 / 2.4) - 0.055)


def load_rgba16(path):
    """a Blender PNG (straight alpha, Standard view transform) -> linear premultiplied RGB, alpha"""
    im = Image.open(path)
    a = np.asarray(im).astype(np.float32)
    a /= 65535.0 if a.max() > 255 else 255.0
    rgb, al = a[..., :3], a[..., 3:4]
    return to_lin(rgb) * al, al


def label(img, big, small, x=60, y_from_bottom=60):
    """the speed in the corner, as in the speed-ladder flybys: '300 mph' over 'B-2 Spirit, 1989'"""
    d = ImageDraw.Draw(img)
    fb, fs = font("InterDisplay-Bold.otf", 52), font("Inter-Medium.otf", 24)
    yb = BAND + PIC_H - y_from_bottom - 34
    for dx, dy in ((2, 2),):
        d.text((x + dx, yb - 52 + dy), big, font=fb, fill=(0, 0, 0, 90))
    d.text((x, yb - 52), big, font=fb, fill=(255, 255, 255))
    d.text((x, yb + 8), small, font=fs, fill=(235, 238, 242))
    return img


def frame_out(pic_srgb_u8, big=None, small=None):
    """picture (PIC_H x W) -> the 1920x1080 frame with bands and the label"""
    out = Image.new("RGB", (W, H), (0, 0, 0))
    out.paste(Image.fromarray(pic_srgb_u8), (0, BAND))
    if big:
        label(out, big, small)
    return out


class Video:
    """frames -> an H.264 file through ffmpeg"""
    def __init__(self, path, fps=FPS, crf=16):
        self.p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                   "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "medium",
                                   "-crf", str(crf), "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)

    def write(self, img):
        self.p.stdin.write(np.asarray(img.convert("RGB"), dtype=np.uint8).tobytes())

    def close(self):
        self.p.stdin.close(); self.p.wait()
