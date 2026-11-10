"""A small .blend that only assembles the finished film: the rendered 4K frames, the HUD overlay, the title,
the sound and (optionally) the short captions, with every path pointing at one folder on the user's PC.

    blender -b -P blender_compose.py -- [--dir "C:\\Users\\me\\Downloads\\film\\render4k"] [--out compose.blend]
        [--hud-parts "path/hud_part1.mov@0,path/hud_part2.mov@2220,..."]   (the HUD in pieces, each under 100 MB)
        [--streak path/streak_scale.mov]     (the '1 streak = N people' line as its own overlay)
    (default --dir //render4k: everything relative to the .blend, which sits in the project folder)

The folder must hold frame_0000.png ... frame_NNNN.png, hud.mov (+ hud_tail.mov), every_grain_of_us_score_with_narration.wav
and, for the music-only cut, every_grain_of_us_score.wav and captions_4k.mov. The strips are built from the
local copies (out/blender_film, the scratch audio) so their lengths are right, then pointed at that folder.

Channels, bottom to top: 1 frames, 2 HUD, 3 title shade, 4 title (5 subtitle, if story.SUBTITLE), 6 music + narration (on),
7 music only (muted), 8 short captions (muted).
"""
import ntpath, os, sys
import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import story as S

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
arg = lambda name, d=None: argv[argv.index(name) + 1] if name in argv else d
DIR = arg("--dir", "//render4k").rstrip("\\/")
REL = DIR.startswith("//")                 # paths relative to the .blend: frames in DIR, other files beside it
OUT = arg("--out", os.path.join(HERE, "out", "every_grain_of_us_compose.blend"))
LOCAL = dict(hud=arg("--hud", os.path.join(HERE, "out", "blender_film", "hud.mov")),
             hud_tail=arg("--hud-tail", os.path.join(HERE, "out", "blender_film", "hud_tail.mov")),
             captions=arg("--captions", os.path.join(HERE, "out", "blender_film", "captions.mov")),
             mix=arg("--mix"), music=arg("--music"))
FONT_DIR = "/usr/share/fonts/opentype/inter"
FPS, RES = 30, (3840, 2160)
NFRAMES = int(round(S.DURATION * FPS))
win = lambda name: "//" + name if REL else ntpath.join(DIR, name)   # beside the .blend, or in the user's folder

# the Video Editing template: opens straight into the sequencer with a preview
bpy.ops.wm.read_homefile(app_template="Video_Editing")
ed = bpy.context.scene
ed.name = "Every Grain of Us"
ed.render.resolution_x, ed.render.resolution_y = RES
ed.render.resolution_percentage = 100
ed.render.fps, ed.render.fps_base = FPS, 1.0
ed.frame_start, ed.frame_end = 0, NFRAMES - 1
ed.frame_current = 0
ed.view_settings.view_transform = "Standard"
ed.render.use_sequencer = True
ed.render.use_compositing = False
sed = ed.sequence_editor_create()
for s in list(sed.sequences_all):
    sed.sequences.remove(s)

# 1 frames
frames = sed.sequences.new_image("Frames", os.path.join(HERE, "frame_0000.png"), channel=1, frame_start=0)
frames.directory = DIR + "/" if REL else DIR
frames.elements[0].filename = "frame_0000.png"
for f in range(1, NFRAMES):
    frames.elements.append(f"frame_{f:04d}.png")

# 2 HUD (year, legend, labels): hud.mov, and from the world tour on (TAIL) hud_tail.mov, which carries the
# labels along the new camera path; the frames before it are the same in both
TAIL = int(arg("--tail-start", 5415))
PARTS = arg("--hud-parts")       # "local/hud_part1.mov@0,local/hud_part2.mov@2220,...": the HUD in pieces (each under
if PARTS:                        # GitHub's 100 MB file limit), laid end to end on channel 2
    for i, spec in enumerate(PARTS.split(",")):
        path, start = spec.rsplit("@", 1)
        st = sed.sequences.new_movie(f"HUD {i + 1}", path, channel=2, frame_start=int(start))
        st.filepath = win(os.path.basename(path))
        st.blend_type = "ALPHA_OVER"
else:
    hud = sed.sequences.new_movie("HUD", LOCAL["hud"], channel=2, frame_start=0)
    hud.filepath = win("hud.mov")
    hud.blend_type = "ALPHA_OVER"
if not PARTS and os.path.exists(LOCAL["hud_tail"]):
    hud.frame_final_end = TAIL
    tail = sed.sequences.new_movie("HUD (world tour)", LOCAL["hud_tail"], channel=2, frame_start=TAIL)
    tail.filepath = win("hud_tail.mov")
    tail.blend_type = "ALPHA_OVER"

# 9 the '1 streak = N people' line, when the HUD was exported before it was part of the HUD
STREAK = arg("--streak")
if STREAK:
    st = sed.sequences.new_movie("Streak scale", STREAK, channel=9, frame_start=0)
    st.filepath = win(os.path.basename(STREAK))
    st.blend_type = "ALPHA_OVER"

# 3-4 title over a darkened opening; fades out between 4.2 and 5.8 s, as in the film
t_on, t_off = int(4.2 * FPS), int(5.8 * FPS)


def fade_out(strip, a0):
    strip.blend_alpha = a0
    strip.keyframe_insert("blend_alpha", frame=t_on)
    strip.blend_alpha = 0.0
    strip.keyframe_insert("blend_alpha", frame=t_off)


shade = sed.sequences.new_effect("TitleShade", "COLOR", channel=3, frame_start=0, frame_end=t_off + 1)
shade.color = (0, 0, 0)
shade.blend_type = "ALPHA_OVER"
fade_out(shade, 0.55)
INK = (0xe8 / 255, 0xee / 255, 0xf8 / 255, 1.0)
MUTED = (0x93 / 255, 0xa1 / 255, 0xb8 / 255, 1.0)
for name, text, ch, fnt, size, y, colr in (
        ("Title", S.TITLE, 4, "InterDisplay-SemiBold.otf", 150, 0.54, INK),
        ("Subtitle", S.SUBTITLE, 5, "Inter-Regular.otf", 56, 0.47, MUTED)):
    if not text:
        continue
    t = sed.sequences.new_effect(name, "TEXT", channel=ch, frame_start=0, frame_end=t_off + 1)
    t.text = text
    t.font = bpy.data.fonts.load(os.path.join(FONT_DIR, fnt))
    t.font.pack()
    t.font_size = size
    t.color = colr
    t.location = (0.5, y)
    t.align_x, t.align_y = "CENTER", "CENTER"
    t.blend_type = "ALPHA_OVER"
    fade_out(t, 1.0)

# 6-7 sound: the narrated mix (on) and the music alone (muted, for the captions cut)
for name, key, fname, ch, mute in (("Music + narration", "mix", "every_grain_of_us_score_with_narration.wav", 6, False),
                                   ("Music only", "music", "every_grain_of_us_score.wav", 7, True)):
    snd = sed.sequences.new_sound(name, LOCAL[key], channel=ch, frame_start=0)
    snd.sound.filepath = win(fname)
    snd.mute = mute

# 8 short captions (muted: unmute for the music-only cut)
cap = sed.sequences.new_movie("Short captions", LOCAL["captions"], channel=8, frame_start=0)
cap.filepath = win("captions_4k.mov")
cap.blend_type = "ALPHA_OVER"
cap.mute = True

# output: 4K H.264 MP4 with AAC audio, next to the frames' folder
r = ed.render
r.image_settings.file_format = "FFMPEG"
r.ffmpeg.format = "MPEG4"
r.ffmpeg.codec = "H264"
r.ffmpeg.constant_rate_factor = "HIGH"
r.ffmpeg.ffmpeg_preset = "GOOD"
r.ffmpeg.gopsize = FPS
r.ffmpeg.audio_codec = "AAC"
r.ffmpeg.audio_bitrate = 320
r.ffmpeg.audio_channels = "STEREO"
r.ffmpeg.audio_mixrate = 48000
r.filepath = "//every_grain_of_us_4k.mp4" if REL else ntpath.join(ntpath.dirname(DIR), "every_grain_of_us_4k.mp4")
ed.sync_mode = "AUDIO_SYNC"                   # preview plays in time with the sound, dropping frames if needed
ed.use_audio = False      # Playback > Mute off: Blender run headless (-b) saves new scenes with Mute on, so the
                          # timeline would play silently (the rendered MP4 has its sound either way)

bpy.ops.wm.save_as_mainfile(filepath=OUT, compress=True, relative_remap=False)   # keep the Windows paths absolute
print("saved", OUT, "frames", NFRAMES, "dir", DIR)
