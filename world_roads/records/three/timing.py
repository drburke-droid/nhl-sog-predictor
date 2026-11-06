"""Narration timing for a film's review round (film A's script_a.py, made general): the script in chunks, each voiced on
its own (every line cached by its text, so a chunk re-voiced after an edit costs only its changed lines); the hook timed
to its lines (its pictures and music on a longer clock); the other sections keep their length and their lines' offsets
(VO_<F>_body.md); and the whole script written out with positions in the final cut.

A film's script module (script_c.py) gives a Film: its hook lines and when each part of the hook lands, its body file,
its chunks, and its film module. Then:

    python script_c.py status            # what each chunk would cost to voice
    python script_c.py voice opening     # voice one chunk; out_<f>/preview/<n>_<chunk>.mp3 and the chunks spliced
    python opening.py c && python script_c.py    # once all are voiced: timed_<f>.json and VO_<F>.md
"""
import json, math, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "outliers"))
import opening                                              # noqa: E402  (clip_len, LEAD, GAP, TAIL, SAY)

HOOK_FACES, HOOK_LEN = 17.4, 30.0                           # the hook's own clock: the faces arrive, the hook ends


class Film:
    """key ('c'); hook: its lines; hook_at: {hook-clock time: (line, phrase or None)} for 7.2 (the "graded" caption),
    10.0 ("most"), 12.4 ("five") and 18.0 (the faces and "stood clear"); hook_words: the hook's captions; chunks:
    {name: a list of lines, or ('body', [sections])}; title: the script's heading"""

    def __init__(self, key, title, hook, hook_at, hook_words, chunks, cls=None):
        self.key, self.title, self.hook, self.hook_at, self.hook_words, self.chunks = key, title, hook, hook_at, \
            hook_words, chunks
        self.cls = cls          # the class test timed to its lines: {lines, caps, at: {key: (line, phrase)}, pages: [line]}
        self.out = os.path.join(HERE, f"out_{key}")
        self.cache = os.path.join(self.out, "voice")
        self.body_path = os.path.join(HERE, f"VO_{key.upper()}_body.md")
        self.timed_path = os.path.join(HERE, f"timed_{key}.json")


def voice(lines, cache):
    """when each line starts and ends, from LEAD, with GAP between lines (whole tenths)"""
    starts, ends, t = [], [], opening.LEAD
    for ln in lines:
        t = math.ceil(t * 10) / 10
        n = opening.clip_len(ln, cache)
        starts.append(t)
        ends.append(t + 0.15 + n)
        t = t + 0.15 + n + opening.GAP
    return starts, ends


def at_word(line, phrase, start, end):
    """roughly when a phrase is spoken: its share of the line's characters"""
    return round(start + (end - start) * line.index(phrase) / len(line), 2)


def plan_hook(film):
    lines = film.hook
    s, e = voice(lines, film.cache)

    def when(a):
        i, ph = film.hook_at[a]
        return s[i] if not ph else at_word(lines[i], ph, s[i], e[i])
    i5, i18 = film.hook_at[12.4][0], film.hook_at[18.0][0]
    if film.hook_at[12.4][1] is None and i5 < i18:          # the "five" caption stays up while its line is spoken:
        short = e[i5] + 3.3 - s[i18]                        # the faces fly 2.4 s before "stood clear", so wait for it
        if short > 0:
            s = s[:i18] + [round(x + short, 2) for x in s[i18:]]
            e = e[:i18] + [x + short for x in e[i18:]]
    faces = when(18.0) - 0.6                                # "stood clear" lands as its caption does (faces + 0.6)
    dur = max(faces + HOOK_LEN - HOOK_FACES, math.ceil((e[-1] + opening.TAIL) * 10) / 10)
    knots = [(0.0, 0.0), (7.2, when(7.2)), (10.0, when(10.0)), (12.4, when(12.4)), (15.0, faces - 2.4),
             (HOOK_FACES, faces), (HOOK_LEN, faces + HOOK_LEN - HOOK_FACES)]
    if any(b2 <= b1 for (_, b1), (_, b2) in zip(knots, knots[1:])):
        raise SystemExit(f"hook lines too short for its pictures: {knots}")
    if dur > knots[-1][1] + 0.05:
        knots.append((HOOK_LEN + 0.5, dur))
    music = [(0.0, 0.0), (9.6, faces - (HOOK_FACES - 9.6)), (HOOK_LEN, faces + HOOK_LEN - HOOK_FACES),
             (HOOK_LEN + 30.0, faces + HOOK_LEN - HOOK_FACES + 30.0)]
    return {"dur": round(dur, 2), "knots": [[round(a, 2), round(b, 2)] for a, b in knots],
            "music": [[round(a, 2), round(b, 2)] for a, b in music], "starts": s, "ends": e, "words": film.hook_words}


def plan_class(film):
    c = film.cls
    lines = c["lines"]
    s, e = voice(lines, film.cache)
    at = {k: (s[i] if not ph else at_word(lines[i], ph, s[i], e[i])) for k, (i, ph) in c["at"].items()}
    dur = math.ceil((e[-1] + opening.TAIL + 0.5) * 10) / 10
    events = (round(at["star"] + 0.8 + 3.5, 2), round(at["lift"] + 1.0 + 3.5, 2))   # the counter lands on 5, then on 1
    return dict(c.get("extra", {}), dur=dur, at=at, caps=[[s[i], cap] for i, cap in enumerate(c["caps"])],
                pages=[s[i] for i in c.get("pages", [])], starts=s, ends=e, events=events)


def body(film):
    """{section: [(offset, text)]} and {section: heading} from VO_<F>_body.md"""
    s = open(film.body_path, encoding="utf-8").read()
    out, heads = {}, {}
    for m in re.finditer(r"^## (\w+) · ([^\n]+)\n(.*?)(?=^## |\Z)", s, re.S | re.M):
        heads[m.group(1)] = m.group(2).strip()
        out[m.group(1)] = [(float(c.group(1)), " ".join(c.group(2).split()))
                           for c in re.finditer(r"\*\*\+(\d+(?:\.\d+)?)\*\*\s+(.+?)(?=\n\s*\n|\Z)", m.group(3), re.S)]
    return out, heads


def chunk_lines(film, name):
    c = film.chunks[name]
    if isinstance(c, tuple) and c[0] == "body":
        B, _ = body(film)
        return [t for k in c[1] for _, t in B[k]]
    if c == "opening":
        return [ln for sc in opening.FILMS[film.key] for ln in sc[2]]
    return list(c)


def say_for(film):
    import narrate
    narrate.SAY = opening.SAY.get(film.key, {})
    os.makedirs(film.cache, exist_ok=True)
    return narrate


def status(film):
    narrate = say_for(film)
    out = {}
    for name in film.chunks:
        lines = chunk_lines(film, name)
        todo = [ln for ln in lines if not narrate.cached(ln, film.cache)]
        out[name] = todo
        print(f"  {name:8s} {len(lines):3d} lines, {len(todo):3d} to voice, {sum(len(narrate.say(x)) for x in todo):5d} "
              "characters (credits)")
    return out


def voice_chunk(film, name):
    """voice a chunk's missing lines, then out_<f>/preview/<n>_<chunk>.mp3: its lines in order, for listening"""
    import numpy as np
    from scipy.io import wavfile
    narrate = say_for(film)
    lines = chunk_lines(film, name)
    todo = [ln for ln in lines if not narrate.cached(ln, film.cache)]
    spent = sum(len(narrate.say(x)) for x in todo)
    for i, ln in enumerate(todo, 1):
        narrate.tts(ln, film.cache)
        print(f"  {name}: voiced {i}/{len(todo)}", flush=True)
    gap = np.zeros(int(0.7 * narrate.SR), np.float32)
    x = np.concatenate([np.concatenate([narrate.decode(narrate.tts(ln, film.cache)), gap]) for ln in lines])
    d = os.path.join(film.out, "preview")
    os.makedirs(d, exist_ok=True)
    wav = os.path.join(d, f"{name}.wav")
    wavfile.write(wav, narrate.SR, (x / (np.abs(x).max() + 1e-9) * 0.89 * 32767).astype(np.int16))
    mp3 = os.path.join(d, f"{list(film.chunks).index(name) + 1}_{name}.mp3")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", wav, "-b:a", "128k", mp3], check=True)
    os.remove(wav)
    print(f"{name}: {len(todo)} lines voiced ({spent} credits), {len(lines)} in all -> {mp3} ({len(x) / narrate.SR:.0f} s)")
    return mp3


def splice(film):
    """out_<f>/preview/film_<f>_narration_all.mp3: the voiced chunk previews in order, 2 s apart"""
    d = os.path.join(film.out, "preview")
    parts = [p for p in (os.path.join(d, f"{i + 1}_{c}.mp3") for i, c in enumerate(film.chunks)) if os.path.exists(p)]
    out = os.path.join(d, f"film_{film.key}_narration_all.mp3")
    n = len(parts)
    args = ["ffmpeg", "-y", "-v", "error"]
    for p in parts:
        args += ["-i", p]
    args += ["-f", "lavfi", "-t", "2", "-i", "anullsrc=r=48000:cl=mono"]
    chain = "".join(f"[{i}:a]aresample=48000,aformat=channel_layouts=mono[a{i}];" for i in range(n))
    gaps = (f"[{n}:a]asplit={n - 1}" + "".join(f"[g{i}]" for i in range(n - 1)) + ";") if n > 1 else ""
    seq = "".join(f"[a{i}]" + (f"[g{i}]" if i < n - 1 else "") for i in range(n))
    args += ["-filter_complex", chain + gaps + seq + f"concat=n={2 * n - 1}:v=0:a=1[out]", "-map", "[out]",
             "-b:a", "128k", out]
    if n < 2:
        args = ["ffmpeg", "-y", "-v", "error", "-i", parts[0], "-b:a", "128k", out]
    subprocess.run(args, check=True)
    print(out, f"({n} of {len(film.chunks)} chunks)")
    return out


def fit(cues, cache):
    """when each line really starts (a line that would run into the next pushes it, as narrate.build does), and when
    the last one ends"""
    free, real = 0.0, []
    for off, txt in cues:
        st = max(off + 0.15, free + 0.35)
        real.append(st)
        free = st + opening.clip_len(txt, cache)
    return real, free


def mmss(t):
    return f"{int(t // 60)}:{t % 60:04.1f}"


def main(film, module_name, extra=None):
    """timed_<f>.json, then VO_<F>.md from the opening (opening_<f>.json), the hook and the body"""
    say_for(film)
    if not os.environ.get("NARRATE_ESTIMATE"):              # timing the film never spends credits: voice by chunk first
        todo = status(film)
        if any(todo.values()):
            raise SystemExit(f"some chunks aren't voiced: python script_{film.key}.py voice <chunk> "
                             "(or NARRATE_ESTIMATE=1)")
    op = json.load(open(os.path.join(HERE, f"opening_{film.key}.json"), encoding="utf-8"))
    hook = plan_hook(film)
    B, heads = body(film)
    T = dict(extra or {}, hook=hook, estimate=bool(os.environ.get("NARRATE_ESTIMATE")))
    if film.cls:
        T["curve"] = plan_class(film)
    json.dump(T, open(film.timed_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    sys.argv = [sys.argv[0], "x"]
    import importlib
    import common
    mod = importlib.import_module(module_name)               # the timeline, with the new lengths
    O, durs, total = common.timeline(mod)
    L = [f"# Narration script: {film.title}", ""]
    if T["estimate"]:
        L += ["**Timing estimated**: some lines are not voiced yet, so they are timed by their length.", ""]
    L += [f"Timecodes are positions in the final cut ({round(total) // 60}:{round(total) % 60:02d}), in seconds and "
          f"tenths. Written by script_{film.key}.py from opening.py (the opening), the hook's lines and "
          f"VO_{film.key.upper()}_body.md (the rest). Names are spelled normally here; say.json respells a few for the "
          "voice. Never say \"proved\", \"greatest athlete\" or \"certain\".", "",
          "## Opening: why eras matter, and how the score allows for them (0:00)", ""]
    L += [f"**{mmss(t)}** {ln}\n" for t, ln in op["cues"]]
    L += [f"## Hook ({mmss(O['hook'])})", ""]
    L += [f"**{mmss(O['hook'] + st)}** {ln}\n" for st, ln in zip(hook["starts"], film.hook)]
    report = []
    for k in mod.ORDER:
        if k in B:
            L += [f"## {heads[k]} ({mmss(O[k])})", ""]
            L += [f"**{mmss(O[k] + off)}** {txt}\n" for off, txt in B[k]]
            real, end = fit(B[k], film.cache)
            late = [round(r - off - 0.15, 1) for r, (off, _) in zip(real, B[k]) if r - off - 0.15 > 0.3]
            report.append((k, round(end, 1), durs[k], late))
        elif k == "curve" and "curve" in T:
            L += [f"## How the score works: grading on a curve ({mmss(O[k])})", ""]
            L += [f"**{mmss(O[k] + st)}** {ln}\n" for st, ln in zip(T["curve"]["starts"], film.cls["lines"])]
    open(os.path.join(HERE, f"VO_{film.key.upper()}.md"), "w", encoding="utf-8").write("\n".join(L))
    print(f"total {total:.1f} s; hook {hook['dur']} s" + (f", curve {T['curve']['dur']} s" if "curve" in T else "") +
          ("  (ESTIMATED: lines not yet voiced are timed by their length)" if T["estimate"] else ""))
    for k, end, d, late in report:
        flag = "  OVER" if end > d - 0.8 else ""
        print(f"  {k:10s} last line ends {end:5.1f} of {d:.0f} s{flag}" + (f"; pushed late {late}" if late else ""))
    return total


def cli(film, module_name):
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        status(film)
    elif len(sys.argv) > 1 and sys.argv[1] == "voice":
        for c in sys.argv[2:] or list(film.chunks):
            voice_chunk(film, c)
        splice(film)
    elif len(sys.argv) > 1 and sys.argv[1] == "splice":
        splice(film)
    else:
        main(film, module_name)
