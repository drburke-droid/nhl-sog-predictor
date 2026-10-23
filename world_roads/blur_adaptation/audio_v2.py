"""v2 audio: Lily voice, Part 1 frame holds (more room for the voice), Part 2 layman cut, merge.

    python3 audio_v2.py part1 | part2 | merge
"""
import os
import sys
import json
import subprocess
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audio as AU                                          # noqa: E402

V2 = os.environ.get("PART2_DIR", f"{AU.BASE}/v2")
DEMO_VIDEO = os.environ.get("DEMO_VIDEO", f"{AU.BASE}/blur/demo1_v2.mp4")
FINAL = os.environ.get("FINAL_NAME", "blur_adaptation_v2.mp4")
FPS = 30
CHIME_GAIN = 0.4                 # test chimes ~8 dB quieter than v2's first cut
# Part 1 lines anchored to the ORIGINAL demo timeline; holds (freeze still cards) are sized from Lily's durations
ANCHOR = {"P01": 0.6, "P02": 6.5, "P03": 19.1, "P04": 27.0, "P05": 50.0, "P06": 71.0, "P07": 82.0, "P08": 85.3,
          "P09": 87.9, "P10": 98.7, "P11": 101.9, "P12": 104.4, "P13": 115.4, "P14": 118.6, "P15": 121.2,
          "P16": 131.9, "P17": 137.3}
# (card end time in the original, line that must fit before it, seconds of air after the line, minimum hold)
CARDS = [(6.0, "P01", 0.3, 0.0), (19.0, "P02", 0.3, 0.0), (131.9, "P15", 0.5, 0.0), (136.9, "P16", 0.5, 0.8),
         (142.9, "P17", 1.0, 1.0)]


def _holds():
    d = json.load(open(f"{V2}/vo_p1/durations.json"))
    holds = []
    for end, lid, air, mn in CARDS:
        start = ANCHOR[lid] if lid != "P16" else 131.9
        need = start + d[lid]["seconds"] + air - end
        holds.append((end, round(max(mn, need, 0.0) + 0.05, 2)))
    return holds


HOLDS = _holds()


def sh(t):
    """original demo time -> re-timed time (holds at a card's end shift everything at or after it)"""
    return t + sum(dd for a, dd in HOLDS if t >= a and a < 142.9)


P1_STARTS = {k: round(sh(v) + (0.3 if k == "P16" else 0.0), 3) for k, v in ANCHOR.items()}


def part1_video():
    """re-time the finished demo: freeze the still cards longer (no re-render)"""
    cuts = [0] + [int(round(a * FPS)) for a, _ in HOLDS]
    parts, fc = [], []
    for i, (a, d) in enumerate(HOLDS):
        fc.append(f"[0:v]trim=start_frame={cuts[i]}:end_frame={cuts[i + 1]},setpts=PTS-STARTPTS,"
                  f"tpad=stop_mode=clone:stop_duration={d}[v{i}]")
        parts.append(f"[v{i}]")
    fc.append("".join(parts) + f"concat=n={len(parts)}:v=1:a=0[out]")
    out = f"{V2}/part1_video.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", DEMO_VIDEO, "-filter_complex",
                    ";".join(fc), "-map", "[out]", "-c:v", "libx264", "-preset", "slow", "-crf", "12", "-pix_fmt",
                    "yuv420p", "-r", "30", out], check=True)
    return out


def part1():
    video = part1_video()
    total = 142.9 + sum(d for _, d in HOLDS)
    lines = [{"id": k, "start": v} for k, v in P1_STARTS.items()]
    S = AU.Score(total, seed=5)
    S.section(0.0, sh(22.0), ["F", "Bb", "Dm", "Csus"], 0.14)
    S.section(sh(22.0), sh(82.0), ["Dm9", "Bb", "Dm9", "Bb", "Gm", "Csus"], 0.0)
    S.ostinato(sh(22.0), sh(82.0), 1.5, ["A4", "A4", "C5", "A4"], 0.16)
    for k, t in enumerate([82.0, 98.7, 115.4]):
        t = sh(t)
        S.chime(t)
        S.pad(t, t + 5.5, "Fadd9", 0.9)
        if k < 2:
            a = t + 6.2
            S.section(a, a + 10.5, ["Dm9", "Bb"], 0.0)
            S.ostinato(a, a + 10.5, 1.5, ["A4", "A4", "C5", "A4"], 0.16)
    S.rolled(sh(120.9), ["F2", "C3", "A3", "G4", "C5", "F5"], 6.0)
    S.section(sh(120.9), sh(136.9), ["Fadd9", "Bb", "F/A", "Csus"], 0.18, lh=False)
    S.rolled(sh(136.9), ["F2", "C3", "A3", "E4", "G4", "C5"], 6.0, 0.32)
    S.pad(sh(136.9), total, "F", 1.0)
    music, fx = S.render(fade_out=2.5)
    fx = fx * CHIME_GAIN
    voice = AU.voice_track(lines, f"{V2}/vo_p1", total)
    sf.write(f"{V2}/part1_mix.wav", AU.mix(music, fx, voice), AU.SR, subtype="PCM_16")
    AU.mux(video, f"{V2}/part1_mix.wav", f"{V2}/part1_final.mp4")
    print("part1", total)


def part2():
    tl = json.load(open(f"{V2}/timeline.json"))
    total = tl["total"]
    lines = [{"id": s["id"], "start": s["vo_start"]} for s in tl["scenes"] if s.get("vo")]
    S = AU.Score(total, seed=23)
    for sec in tl["music"]:
        S.section(sec["t0"], sec["t1"], sec["chords"], sec["dens"])
    for c in tl["chimes"]:
        S.chime(c)
    S.rolled(tl["final"], ["F2", "C3", "A3", "E4", "G4", "C5"], total - tl["final"], 0.34)
    music, fx = S.render(fade_out=3.0)
    voice = AU.voice_track(lines, f"{V2}/vo_p2", total)
    sf.write(f"{V2}/part2_mix.wav", AU.mix(music, fx, voice), AU.SR, subtype="PCM_16")
    AU.mux(f"{V2}/part2_video.mp4", f"{V2}/part2_mix.wav", f"{V2}/part2_final.mp4")
    print("part2", total)


def merge():
    lst = f"{V2}/merge.txt"
    open(lst, "w").write(f"file '{V2}/part1_final.mp4'\nfile '{V2}/part2_final.mp4'\n")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
                    f"{V2}/merged_raw.mp4"], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", f"{V2}/merged_raw.mp4", "-c:v", "copy", "-af",
                    "loudnorm=I=-16:TP=-1.5:LRA=11", "-ar", "48000", "-c:a", "aac", "-b:a", "256k", "-movflags",
                    "+faststart", f"{V2}/{FINAL}"], check=True)
    print("merged", f"{V2}/{FINAL}")


if __name__ == "__main__":
    {"part1": part1, "part2": part2, "merge": merge}[sys.argv[1]]()
