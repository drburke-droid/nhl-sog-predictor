"""What the three companion films share: the score (../outliers/orchestra.py, with calm music under any section it
doesn't know and a short finale for a short close), the assembly (0.6 s crossfades, -16 LUFS), stills, and the
countdown-free sections' helpers.
"""
import os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REC = os.path.dirname(HERE)
sys.path.insert(0, REC)
sys.path.insert(0, os.path.join(REC, "outliers"))
import look                                                 # noqa: E402
import assemble as A                                        # noqa: E402  (the season film's helpers: XF, dur, loudnorm)
import orchestra                                            # noqa: E402

VIDEOS = os.path.normpath(os.path.join(REC, "..", "..", "videos", "sports"))


def close_short(S, t0, finale, dur):
    """a riser into a full-orchestra hit as the closing line lands, then the theme to the end of the section"""
    key = orchestra.D
    orchestra.pad(S, key, "V", t0, max(0.5, finale - 0.2), 0.45)
    orchestra.riser(S, key, t0 + finale, min(3.0, finale + 0.5), 1.0)
    orchestra.hit(S, key, "I", t0 + finale, 1.0, 2.0)
    orchestra.theme(S, key, t0 + finale, (dur - finale) / 8, list(range(8)), 1.0, choir=True)


def build_score(O, durs, close_finale, explain=("curve",), minor=()):
    S = orchestra.Score()
    for name, t0 in O.items():
        d = durs[name]
        if name == "hook":
            orchestra.hook(S, t0)
        elif name[0] == "c" and name[1:].isdigit():
            orchestra.card(S, t0, int(name[1:]))
        elif name == "curve":
            orchestra.calm(S, t0, d, events=(16.0, 30.0))
        elif name == "close":
            close_short(S, t0, close_finale, d)
        elif name == "end":
            orchestra.end(S, t0)
        else:
            orchestra.calm(S, t0, d, minor=name in minor)
    return S


def timeline(film):
    ds = [film.SECTIONS[s]().dur for s in film.ORDER]
    off = [0.0]
    for d in ds[:-1]:
        off.append(off[-1] + d - A.XF)
    return dict(zip(film.ORDER, off)), dict(zip(film.ORDER, ds)), off[-1] + ds[-1]


def assemble(film, out_dir, final, close_finale, minor=(), crf=18):
    """join the rendered sections with crossfades, lay in the score, normalise to -16 LUFS -> out_dir/final"""
    import numpy as np
    O, durs, total = timeline(film)
    paths = [os.path.join(out_dir, f"{s}.mp4") for s in film.ORDER]
    print(f"total {total:.2f} s ({int(total // 60)}:{total % 60:04.1f})", flush=True)
    orchestra.OUT = out_dir
    S = build_score(O, durs, close_finale, minor=minor)
    if S.dropped:
        raise SystemExit(f"{len(S.dropped)} notes out of range")
    mid, wavf = os.path.join(out_dir, "score.mid"), os.path.join(out_dir, "score_fluid.wav")
    S.save(mid)
    subprocess.run(["fluidsynth", "-ni", "-q", "-g", "0.45", "-r", str(orchestra.SR), "-R", "1", "-C", "0",
                    "-o", "synth.reverb.room-size=0.82", "-o", "synth.reverb.damp=0.25", "-o", "synth.reverb.width=1.0",
                    "-o", "synth.reverb.level=0.75", "-o", "synth.polyphony=512", "-F", wavf, orchestra.SF2, mid], check=True)
    from scipy.io import wavfile
    sr, x = wavfile.read(wavf)
    x = x.astype(np.float32) / (32768.0 if x.dtype == np.int16 else 1.0)
    n = int(total * orchestra.SR)
    x = x[:n] if len(x) >= n else np.pad(x, ((0, n - len(x)), (0, 0)))
    t = np.arange(n) / orchestra.SR
    x *= np.clip((total - t) / 2.5, 0, 1)[:, None]
    mix = (x / (np.abs(x).max() + 1e-9) * 0.89).astype(np.float32)
    raw, wav = os.path.join(out_dir, "score_raw.wav"), os.path.join(out_dir, "score.wav")
    orchestra.write(mix, raw)
    m = A.loudnorm(raw, wav)
    print("score loudness in", m["input_i"], "LUFS", flush=True)
    args = ["ffmpeg", "-y", "-v", "error"]
    for p in paths:
        args += ["-i", p]
    args += ["-i", wav]
    fc, last = [], "[0:v]"
    acc = A.dur(paths[0])
    for i in range(1, len(paths)):
        off = acc - A.XF
        fc.append(f"{last}[{i}:v]xfade=transition=fade:duration={A.XF}:offset={off:.3f}[v{i}]")
        last = f"[v{i}]"
        acc = off + A.dur(paths[i])
    dst = os.path.join(out_dir, final)
    args += ["-filter_complex", ";".join(fc), "-map", last, "-map", f"{len(paths)}:a", "-c:v", "libx264", "-preset", "slow",
             "-crf", str(crf), "-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709",
             "-colorspace", "bt709", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", dst]
    subprocess.run(args, check=True)
    print(dst, f"{os.path.getsize(dst) / 1e6:.1f} MB", f"{A.dur(dst):.2f} s")
    if os.path.exists(os.path.join(out_dir, "narration.wav")):
        voiced(out_dir, final)
    return dst


BED_DB = {"b": 10.0}            # the voiced mix lowers the score by this much throughout (film B: the music between lines
                                # was as loud as the voice); the 18 dB duck under the voice comes on top


def voiced(out_dir, final, duck_db=18.0, bed_db=None):
    """the voiced copy, from the finished film: the score (score_raw.wav, as assembled) lowered by bed_db, the narration
    on top with the score ducked duck_db under the voice, normalised to -16 LUFS, on the film's own video stream"""
    import numpy as np
    import narrate
    from scipy.io import wavfile
    if bed_db is None:
        bed_db = BED_DB.get(os.path.basename(out_dir.rstrip("/"))[-1], 0.0)
    sr, x = wavfile.read(os.path.join(out_dir, "score_raw.wav"))
    music = (x.astype(np.float32) / 32768.0) * 10 ** (-bed_db / 20.0)
    vmix = narrate.duck_and_mix(music, os.path.join(out_dir, "narration.wav"), duck_db=duck_db)
    vraw, vwav = os.path.join(out_dir, "voiced_raw.wav"), os.path.join(out_dir, "voiced.wav")
    orchestra.write(vmix, vraw)
    A.loudnorm(vraw, vwav)
    dst = os.path.join(out_dir, final)
    vdst = dst.replace(".mp4", "_voiced.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", dst, "-i", vwav, "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", vdst], check=True)
    print(vdst, f"music {bed_db:g} dB down, ducked {duck_db:g} dB under the voice;",
          f"{os.path.getsize(vdst) / 1e6:.1f} MB", f"{A.dur(vdst):.2f} s")
    return vdst


def stills(film, picks, out_dir):
    os.makedirs(os.path.join(out_dir, "stills"), exist_ok=True)
    for name in film.ORDER:
        sec = film.SECTIONS[name]()
        for t in picks.get(name, [33.0] if name[0] == "c" and name[1:].isdigit() else [sec.dur / 2]):
            p = os.path.join(out_dir, "stills", f"{film.ORDER.index(name):02d}_{name}_{t:05.1f}.png")
            look.still(sec, t, p)
            print(p, flush=True)


class Board:
    """how sure the list is: each row's share of the 16 versions of the method that put it in their top ten, one square
    per version. rows: dicts with rank, athlete, achievement, season, z, specs_in_top10, hi (shown in gold once the
    caption about it lands), face (an RGBA face disc). caps: [(t, caption)]."""
    dur = 32.0

    def __init__(self, rows, heading, title_text, caps, gold_at=16.0, n_specs=16, dur=None):
        self.rows, self.heading, self.title_text, self.caps = rows, heading, title_text, caps
        self.gold_at, self.n = gold_at, n_specs
        if dur:
            self.dur = dur

    def draw(self, fig, t):
        import matplotlib
        import film as FM
        from look import W, H, INK, INK2, MUTED, GOLD, BLUE, fade, clamp, text, title
        al = fade(t, 0, self.dur, 0.6, 0.6)
        text(fig, 96, 92, self.heading, size=21, color=MUTED, weight="semibold", alpha=al)
        title(fig, 96, 160, self.title_text, size=58, alpha=al)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        text(fig, 1060, 236, "IN THE TOP TEN OF EACH VERSION", size=17, color=MUTED, weight="semibold", alpha=al)
        text(fig, 940, 236, "SCORE", size=17, color=MUTED, weight="semibold", ha="right", alpha=al)
        fill = clamp((t - 9.0) / 5.0)
        for i, b in enumerate(self.rows):
            y = 280 + i * 60
            a = al * fade(t, 1.0 + 0.15 * i, 99, 0.5, 0)
            hi = b["hi"] and t >= self.gold_at
            col = GOLD if hi else INK
            text(fig, 110, y + 12, f"{b['rank']}", size=24, color=MUTED, ha="right", alpha=a)
            ax.imshow(b["face"], extent=(130, 184, y + 27, y - 27), alpha=a, zorder=5)
            text(fig, 200, y + 10, b["athlete"], size=25, color=col, weight="semibold" if b.get("bold", True) else "regular",
                 alpha=a)
            text(fig, 470, y + 10, f"{b['achievement']}, {b['season']}", size=21, color=INK2, alpha=a)
            text(fig, 940, y + 10, f"{b['z']:.2f}", size=22, color=INK2, ha="right", alpha=a)
            shown = int(round(b["specs_in_top10"] * fill))
            for j in range(self.n):
                on = j < shown
                ax.add_patch(matplotlib.patches.Rectangle((1060 + j * 27, y - 11), 22, 22, lw=0, zorder=4,
                                                          color=(GOLD if hi else BLUE) if on else "#2a2f36",
                                                          alpha=a * (0.95 if on else 0.8)))
            if fill > 0:
                text(fig, 1510, y + 10, f"{shown} of {self.n}", size=21, color=GOLD if hi else INK2, alpha=a * fill)
        cur = [m for t0, m in self.caps if t0 <= t]
        if cur:
            t0 = max(t0 for t0, m in self.caps if t0 <= t)
            FM.para(fig, 96, 905, cur[-1], width=110, size=27, color=INK, alpha=al * fade(t, t0, 99, 0.6, 0))


class ClosingLine:
    """the last words: what the list is and isn't"""
    dur = 22.0
    FINALE = 1.5

    def __init__(self, what="one result", question="Which would you choose?"):
        self.what, self.question = what, question

    def draw(self, fig, t):
        from look import W, INK2, GOLD, fade, text, title
        al = fade(t, 0, self.dur, 0.6, 0.9)
        title(fig, W / 2, 400, "Not a ranking of the greatest athletes.", size=60, ha="center", alpha=al * fade(t, 1.3, 99, 0.8, 0))
        title(fig, W / 2, 500, f"A ranking of how far {self.what} stood", size=60, ha="center", alpha=al * fade(t, 3.0, 99, 0.8, 0))
        title(fig, W / 2, 580, "from everyone else of its time.", size=60, ha="center", color=GOLD,
              alpha=al * fade(t, 3.0, 99, 0.8, 0))
        text(fig, W / 2, 720, self.question, size=36, color=INK2, ha="center", alpha=al * fade(t, 7.5, 99, 0.8, 0))


def credits_page(fig, t, keys, credits, t0=13.0, dur=26.0):
    """photo credits, two columns"""
    import film as FM
    from look import INK, INK2, MUTED, fade, text
    al = fade(t, t0, dur, 0.8, 1.2)
    if al <= 0:
        return
    text(fig, 96, 120, "PHOTOGRAPHS", size=21, color=MUTED, weight="semibold", alpha=al)
    rows = [k for k in keys if credits.get(k)]
    step = 128 if len(rows) <= 10 else 104
    i = 0
    for k in rows:
        cr = credits[k]
        col, row = i % 2, i // 2
        x, y = 96 + col * 880, 190 + row * step
        a = al * fade(t, t0 + 0.3 + 0.12 * i, 99, 0.5, 0)
        try:
            FM.photo(fig, k, x, y, x + 64, y + 80, alpha=a, credit=False)
        except (FileNotFoundError, KeyError):
            pass
        text(fig, x + 88, y + 26, cr["athlete"], size=23, color=INK, weight="semibold", alpha=a)
        lic = cr["license"] if cr["license"] != "Public domain" else "public domain"
        text(fig, x + 88, y + 56, f"{cr['author'][:60]} · {lic}", size=19, color=INK2, alpha=a)
        text(fig, x + 88, y + 80, cr["file"][:72], size=15, color=MUTED, alpha=a)
        i += 1
    FM.para(fig, 96, 930, "All from Wikimedia Commons (commons.wikimedia.org): cropped, backgrounds removed, toned in team or "
                          "national colours. Licences: creativecommons.org/licenses (CC BY, CC BY-SA, CC0) or public domain. "
                          "Full credits and links are in the description. Icons: BlenderKit models.", width=120, size=20,
            color=MUTED, alpha=al)


def sources_page(fig, t, lines, dur=12.5):
    import film as FM
    from look import INK2, MUTED, GOLD, fade, text
    al = fade(t, 0, dur, 0.8, 0.8)
    if al <= 0:
        return
    text(fig, 96, 120, "SOURCES AND METHOD", size=21, color=MUTED, weight="semibold", alpha=al)
    y = 200
    for i, (h, s) in enumerate(lines):
        a = al * fade(t, 0.4 + 0.5 * i, 99, 0.6, 0)
        text(fig, 96, y + 26, h, size=26, color=GOLD, weight="semibold", alpha=a)
        y = FM.para(fig, 330, y, s, width=92, size=25, color=INK2, alpha=a) + 30
