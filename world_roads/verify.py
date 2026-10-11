"""Probe the rendered videos and assert the acceptance-critical properties."""
import json, subprocess, sys, os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                        "stream=codec_name,profile,pix_fmt,width,height,r_frame_rate,nb_read_frames,duration:format=duration,size",
                        "-of", "json", path], capture_output=True, text=True, check=True)
    return json.loads(r.stdout)


if __name__ == "__main__":
    full = os.path.join(OUT, "world_roads_light_speed.mp4")
    for p in sys.argv[1:] or [full, os.path.join(OUT, "world_roads_preview.mp4")]:
        if not os.path.exists(p):
            print("missing:", p); continue
        j = probe(p); s = j["streams"][0]; f = j["format"]
        print(os.path.basename(p), json.dumps({**s, "container_duration": f["duration"], "MB": round(int(f["size"]) / 1e6, 1)}))
        if p == full:
            assert s["codec_name"] == "h264" and s["pix_fmt"] == "yuv420p"
            assert (s["width"], s["height"]) == (1920, 1080) and s["r_frame_rate"] == "30/1"
            assert int(s["nb_read_frames"]) == 8400, s["nb_read_frames"]
            assert abs(float(f["duration"]) - 280.0) < 0.1, f["duration"]
            print("  OK: 8400 frames @ 30 fps = 280.0 s, 1920x1080 H.264 yuv420p")
