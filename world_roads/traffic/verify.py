"""Probe the master with ffprobe:  python verify.py [path]   (default ../media/traffic/three_routes_one_eta.mp4)

Asserts H.264 High / yuv420p, 1920x1080, 30/1 fps, the frame count and duration of the timeline, and an
AAC stereo track of the same length.
"""
import os, sys, json, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import assemble as A    # noqa: E402

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(HERE), "media", "traffic", "three_routes_one_eta.mp4")
pr = json.loads(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-show_format", "-of", "json", path],
                               capture_output=True, text=True, check=True).stdout)
v = next(s for s in pr["streams"] if s["codec_type"] == "video")
a = next(s for s in pr["streams"] if s["codec_type"] == "audio")
_, total = A.offsets()
frames = int(v["nb_read_frames"])
print(f"{path}\n  video {v['codec_name']} {v.get('profile')} {v['pix_fmt']} {v['width']}x{v['height']} {v['r_frame_rate']} fps, {frames} frames")
print(f"  audio {a['codec_name']} {a['channels']} ch {a['sample_rate']} Hz, {float(a['duration']):.2f} s")
print(f"  duration {float(pr['format']['duration']):.2f} s (timeline {total:.2f} s), {int(pr['format']['size']) / 1e6:.1f} MB")
assert v["codec_name"] == "h264" and v["pix_fmt"] == "yuv420p" and (v["width"], v["height"]) == (1920, 1080)
assert v["r_frame_rate"] == "30/1"
assert abs(frames - round(total * 30)) <= 1, frames
assert a["codec_name"] == "aac" and a["channels"] == 2
assert abs(float(a["duration"]) - total) < 0.1 and abs(float(pr["format"]["duration"]) - total) < 0.1
print("OK")
