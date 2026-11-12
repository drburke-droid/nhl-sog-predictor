"""
Build the master .blend that assembles the finished film in Blender's Video
Sequencer: the rendered frames of every shot, the overlay (scores, charts,
cards) and the music.  Open it and press Render Animation.

    python make_film.py plates --scale 3            # overlay videos, score, frame map -> blender/
    python make_master.py                           # -> blender/the_mix_off_MASTER.blend

The overlay is two ordinary videos (see make_film.stage_plates):
    channel 1  black
    channel 2  rendered frames        frames/<shot>/f_####.png
    channel 3  overlay_multiply.mp4   blend: Multiply   (how much picture shows through)
    channel 4  overlay_add.mp4        blend: Add        (the overlay itself)
    channel 5  score.m4a
"""
import argparse
import json
import os
import sys

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(HERE, "blender"), help="folder holding the plates; the master is saved here")
    ap.add_argument("--scale", type=float, default=3.0, help="1 = 1280x720, 1.5 = 1080p, 3 = 4K; must match the plates")
    ap.add_argument("--pic-width", type=int, help="width the matches are rendered at (default: the film's width)")
    ap.add_argument("--full-width", type=int, help="width the closing shots are rendered at (default: the film's width)")
    ap.add_argument("--frames", default="frames", help="frames folder, relative to the master")
    ap.add_argument("--subdir", default="", help="folder inside each shot's frames folder (the preview renders use 'frames')")
    ap.add_argument("--name", default="the_mix_off_MASTER.blend")
    ap.add_argument("--out-name", default="the_mix_off_4K")
    a = ap.parse_args(argv)

    W, H, PIC_H = (int(round(v * a.scale)) for v in (1280, 720, 560))
    pics = json.load(open(os.path.join(a.dir, "film_pics.json")))
    tl = json.load(open(os.path.join(a.dir, "film_timeline.json")))
    full = {s[0] for s in tl["segments"] if s[0].startswith("X")}          # closing shots fill the frame

    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.name = "The Mix-Off"
    r = sc.render
    r.resolution_x, r.resolution_y, r.resolution_percentage = W, H, 100
    r.fps = tl["fps"]
    sc.frame_start, sc.frame_end = 1, len(pics)
    sc.view_settings.view_transform = "Standard"         # the frames already carry their look
    sc.view_settings.look = "None"
    sc.sequencer_colorspace_settings.name = "sRGB"
    se = sc.sequence_editor_create()
    strips = se.strips if hasattr(se, "strips") else se.sequences
    master = os.path.join(a.dir, a.name)
    bpy.ops.wm.save_as_mainfile(filepath=master)          # so that // paths resolve

    base = strips.new_effect("black", "COLOR", 1, frame_start=1, length=len(pics))       # blends need something opaque underneath
    base.color = (0, 0, 0)

    # ---- channel 2: the rendered pictures, one strip per run of frames from the same shot ----
    runs, k = [], 0
    while k < len(pics):
        if pics[k] is None:
            k += 1
            continue
        j = k
        while j < len(pics) and pics[j] is not None and pics[j][0] == pics[k][0]:
            j += 1
        runs.append((k, j))
        k = j
    for n, (k0, k1) in enumerate(runs):
        key = pics[k0][0]
        folder = f"//{a.frames}/{key}/" + (a.subdir + "/" if a.subdir else "")
        st = strips.new_image(f"{n + 1:02d} {key}", folder + f"f_{pics[k0][1]:04d}.png", 2, k0 + 1)
        for q in range(k0 + 1, k1):
            st.elements.append(f"f_{pics[q][1]:04d}.png")
        st.blend_type = "ALPHA_OVER"
        st.transform.scale_x = st.transform.scale_y = W / ((a.full_width if key in full else a.pic_width) or W)
        st.transform.offset_y = 0 if key in full else (H - PIC_H) / 2      # matches: the picture sits at the top
        st.color_tag = "COLOR_04"

    mul = strips.new_movie("overlay multiply", "//overlay_multiply.mp4", 3, 1)
    mul.blend_type = "MULTIPLY"
    add = strips.new_movie("overlay add", "//overlay_add.mp4", 4, 1)
    add.blend_type = "ADD"
    snd = strips.new_sound("score", "//score.m4a", 5, 1)
    snd.volume = 1.0

    r.use_sequencer = True
    r.filepath = "//" + a.out_name
    try:
        r.image_settings.media_type = "VIDEO"            # Blender 5: video is a media type
    except Exception:
        pass
    r.image_settings.file_format = "FFMPEG"
    r.ffmpeg.format = "MPEG4"
    r.ffmpeg.codec = "H264"
    r.ffmpeg.constant_rate_factor = "HIGH"
    r.ffmpeg.ffmpeg_preset = "GOOD"
    r.ffmpeg.gopsize = 12
    r.ffmpeg.audio_codec = "AAC"
    r.ffmpeg.audio_bitrate = 256
    r.ffmpeg.audio_mixrate = 48000
    try:
        r.image_settings.color_mode = "RGB"
    except Exception:
        pass
    bpy.ops.wm.save_mainfile()
    print(f"saved {master}: {len(pics)} frames, {len(runs)} picture strips, {W}x{H}")


if __name__ == "__main__":
    main()
