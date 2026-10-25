"""The narration, written for ElevenLabs' Lily voice and timed to the video:  python voiceover.py
-> ../media/traffic/VOICEOVER.md (the script, with cue times and a paste-ready block per section)

Each line is (section, cue, text): the cue is seconds from the start of its section, the moment the matching
caption or animation appears. Numbers are spelled the way they should be read. check() fails if the scenario's
numbers no longer match what the script says, so a change to scenario.py cannot leave the voice out of date.

Recorded clips go in ../media/traffic/voice/ (see voice.py): one per section (hook.mp3, light.mp3, ...), or one
per line (hook_01.mp3, hook_02.mp3, ...) for exact timing.
"""
import os, re
import video as V
import assemble as A

WPS = 2.6                    # words per second, a measured narrator's pace
LINES = [
    ("hook", 0.8, "Ever wonder why, when you check your maps app for the best route home in the evening rush, there are always "
                  "two or more completely different routes, with pretty much the exact same time?"),
    ("hook", 14.8, "Why isn't there just one no-brainer route that's by far the best?"),
    ("hook", 20.0, "Here's a real one: Times Square to Bayside, leaving at six p.m."),
    ("hook", 25.4, "Fifty-nine minutes. One hour. One hour and one minute."),
    ("hook", 29.8, "Three completely different ways home, all within two minutes."),
    ("hook", 34.6, "Our simple traffic model does the same thing: two ways home, twenty-nine minutes each."),
    ("hook", 40.8, "Well, you're not alone. And your house isn't in a magic spot."),
    ("hook", 46.2, "It happens on almost every busy trip. Here's why."),

    ("light", 0.4, "In an earlier video, a point of light drove all the world's roads."),
    ("light", 6.6, "No traffic, no red lights: about four minutes, forty seconds."),
    ("light", 11.0, "Down here, we share the road, and that changes everything."),

    ("empty", 0.3, "At three a.m., the same drive home, three ways."),
    ("empty", 6.0, "No traffic, no red lights: real roads, at the speed limit."),
    ("empty", 16.4, "The Midtown Tunnel wins, by five minutes."),
    ("empty", 20.9, "So why are two of them tied at rush hour?"),

    ("curve", 1.0, "Every car slows every other car."),
    ("curve", 5.0, "A little at first."),
    ("curve", 7.0, "Then a lot."),
    ("curve", 9.0, "This is the standard curve traffic planners use. Our capacities are made up."),
    ("curve", 14.6, "And each route has its own curve."),

    ("fill", 3.0, "As more people head home, they all take the Midtown Tunnel at first. It slows down."),
    ("fill", 13.0, "At about twenty-three hundred drivers an hour, it's as slow as the Throgs Neck route is when empty. "
                   "Now drivers take that one too."),
    ("fill", 22.8, "From here on, the two slow down together."),
    ("fill", 27.2, "The Battery Tunnel is still slower, so nobody takes it. On a heavier night it would fill too, "
                   "and then all three would match."),
    ("fill", 37.0, "Two completely different routes home. One E.T.A."),

    ("why", 0.4, "Suppose the Midtown Tunnel showed twenty-five minutes, and the Throgs Neck thirty-six."),
    ("why", 5.4, "Drivers leave the slower route. The quicker one slows down as they arrive; the other speeds up."),
    ("why", 13.2, "They meet in the middle. Any gap left would tempt more drivers, so it closes."),
    ("why", 19.6, "That's Wardrop's principle: no driver can do better by switching alone."),
    ("why", 24.4, "Game theorists call it a Nash equilibrium. Nobody plans it."),

    ("fine", 0.2, "The fine print."),
    ("fine", 1.6, "One: it only works on routes people use. The one nobody takes is never pulled into line."),
    ("fine", 8.8, "Two: it's about one trip, at one time. Leave an hour earlier and it changes."),
    ("fine", 15.4, "Three: equal means close enough. Nobody switches to save thirty seconds, and apps round to the minute."),
    ("fine", 22.4, "And four: it isn't the best outcome for everyone. As we'll see, it can leave everyone slower."),

    ("learn", 0.3, "So how do drivers find this balance? Imagine it hadn't settled yet."),
    ("learn", 5.2, "On Monday, most drivers take the Midtown Tunnel, the quickest on empty roads. It jams: "
                   "forty-two minutes, while the Throgs Neck takes twenty-six."),
    ("learn", 14.6, "People find out they weren't on the quickest route. If half of everyone switches on Tuesday, "
                    "the Throgs Neck jams instead."),
    ("learn", 23.2, "Day after day, the crowd chases yesterday's quickest route, and the jam just moves from road to road."),
    ("learn", 31.0, "Real drivers are calmer. Say one in fifty tries another route each day, and nobody switches to save "
                    "under a minute."),
    ("learn", 39.6, "Then the times creep together, and settle within a minute in about thirteen days."),
    ("learn", 45.6, "They're not exactly equal. But rounded to the minute, both say twenty-nine."),

    ("app", 0.3, "There's a second reason the E.T.A.s match."),
    ("app", 4.2, "There are hundreds of ways home, taking twenty-one to sixty-seven minutes at three a.m."),
    ("app", 13.0, "Your app never shows you hundreds. It offers a few that are quick and genuinely different, like our three."),
    ("app", 21.0, "Slow ones, and near-copies of quicker ones, never make the list. So what you see is already the close calls."),
    ("app", 29.0, "Traffic pulls the routes people use together. The app hides the rest. Both make the E.T.A.s on your "
                  "screen match."),

    ("optimum", 0.4, "Is everyone choosing for themselves the best for everyone?"),
    ("optimum", 5.8, "Not quite. A perfect traffic controller would send some drivers the slow way, so the rest save more."),
    ("optimum", 13.0, "Why? Joining a busy road slows everyone behind you, and nobody counts that cost."),
    ("optimum", 20.0, "Here, the gap is about six percent."),
    ("optimum", 23.4, "Economists call it the price of anarchy. And a new road can make it much bigger."),

    ("myth", 0.3, "Can a new road make everyone slower? Here's a thought experiment."),
    ("myth", 6.0, "First: a bottleneck is where the road narrows, say from four lanes to two."),
    ("myth", 12.0, "In light traffic, two lanes cope, and nobody waits."),
    ("myth", 17.0, "In heavy traffic, more cars arrive than two lanes can pass. A queue builds, and the wait keeps growing."),
    ("myth", 26.0, "Give each route one like that: the Queens-Midtown Tunnel and the Throgs Neck Bridge. The rest never jams."),
    ("myth", 34.0, "With twenty-five hundred drivers an hour, they split so both routes take twenty-seven minutes."),
    ("myth", 41.0, "Now dig a mythical tunnel, straight from one bottleneck to the other. Three minutes, never any traffic."),
    ("myth", 48.0, "At three a.m. it's a real shortcut: fourteen minutes, instead of twenty-one."),
    ("myth", 54.0, "At rush hour, the first driver to try it saves seven minutes. So others follow..."),
    ("myth", 60.0, "...and every one of them goes through both bottlenecks, so both queues grow."),
    ("myth", 71.0, "When the switching stops, every route takes thirty-four minutes. Seven minutes longer, for everyone."),
    ("myth", 78.0, "And nobody can do better by going back: both old routes now share a jammed bottleneck."),
    ("myth", 85.0, "Each switch saved one driver time, but slowed everyone. Close the tunnel, and everyone's home sooner."),
    ("myth", 92.0, "This is Braess's paradox. It isn't the rule: with fewer drivers, the same tunnel helps."),

    ("outro", 0.5, "So, next time your drive home offers two routes with the same E.T.A..."),
    ("outro", 6.8, "...you're not alone."),
    ("outro", 8.6, "Your house isn't in a magic spot. Thousands of drivers, each taking the fastest way they know, pushed "
                   "those times together. And your app only showed you the close calls."),
    ("outro", 20.6, "And beware the shortcut that everyone takes."),
]


def check():
    """the numbers the script speaks, against the scenario"""
    N, m = V.N, V.N["myth"]
    said = {"rush ETA (29)": (V.N["eta_final"][V.USED[0]], 29), "3 a.m. A (21)": (round(V.T0[0]), 21),
            "3 a.m. B (26)": (round(V.T0[1]), 26), "B joins (~2,300)": (round(N["joins"][1], -2), 2300),
            "why start (25, 36)": (tuple(round(x) for x in N["why_start"]["times"][:2]), (25, 36)),
            "Monday (42, 26)": (tuple(round(x) for x in N["learn"]["chase"]["times"][0][:2]), (42, 26)),
            "settles (13 days)": (N["learn"]["calm"]["settle_day"] + 1, 13), "via (21-67)": ((round(N["app"]["via_min"]), round(N["app"]["via_max"])), (21, 67)),
            "gap (~6 %)": (round(100 * (N["poa"] - 1)), 6), "tunnel demand (2,500)": (m["cfg"]["demand"], 2500.0),
            "tunnel before/after (27, 34)": ((round(m["closed"]["avg"]), round(m["opened"]["avg"])), (27, 34)),
            "tunnel at 3 a.m. (14 vs 21)": ((round(m["night"][1]), round(min(m["night"][0], m["night"][2]))), (14, 21)),
            "first driver saves (7)": (round(m["closed"]["avg"] - m["lone"]), 7)}
    bad = {k: v for k, (v, want) in said.items() if v != want}
    assert not bad, f"the script no longer matches the scenario: {bad}"


def est(text):
    """seconds to say a line: words at WPS, plus a beat at each sentence break (not the dots in E.T.A. or a.m.)"""
    return len(text.split()) / WPS + 0.25 * len(re.findall(r"[^.][.?!:;](?=\s|$)", text))


def mmss(t):
    return f"{int(t // 60)}:{t % 60:04.1f}"


def numbered():
    """(section, k, cue, text) with k counting from 1 within each section"""
    out, k = [], {}
    for sec, cue, txt in LINES:
        k[sec] = k.get(sec, 0) + 1
        out.append((sec, k[sec], cue, txt))
    return out


def fit():
    """per line: where it would end, and whether it runs into the next line or past its section"""
    O, total = A.offsets()
    rows = numbered()
    out = []
    for j, (sec, k, cue, txt) in enumerate(rows):
        end = cue + est(txt)
        nxt = rows[j + 1][2] if j + 1 < len(rows) and rows[j + 1][0] == sec else V.SECTIONS[sec].dur - V.XF
        out.append((sec, k, cue, txt, end, nxt, end <= nxt + 0.3))
    return out, O


def write(path=os.path.join(V.L.MEDIA, "VOICEOVER.md")):
    rows, O = fit()
    L = ["# Voiceover script: two routes home, one ETA", "",
         "Written for ElevenLabs' **Lily** voice (a warm, measured British narrator), timed to `two_routes_one_eta.mp4`.",
         "Generated by `traffic/voiceover.py` from the same numbers as the video; edit the script there, not here.", "",
         "**Suggested settings** (a starting point): model Eleven Multilingual v2, which reads the `<break>` tags below;",
         "stability about 50%, similarity about 75%, style 0–15%, speaker boost on, speed 1.0. Generate each section",
         "separately so a retake doesn't shift the rest.", "",
         "**Recording it automatically.** With `ELEVENLABS_API_KEY` set, `python lily.py` records every line below with",
         "Lily (one clip per line, about 5,400 characters of credit), speeds up any clip that overruns its slot by up",
         "to 12%, and `python assemble.py` then mixes them in.", "",
         "**Putting it in the video by hand.** Save each section's audio in `media/traffic/voice/` as `<section>.mp3` (or `.wav`",
         "or `.m4a`), e.g. `hook.mp3`, then run `python assemble.py`. Each clip starts at its section's first cue, and",
         "the music ducks under the voice. For exact timing, record one clip per line instead (`hook_01.mp3`,",
         "`hook_02.mp3`, …) and each line starts on its own cue. The captions stay on screen either way, so the",
         "video still works with the sound off.", "",
         f"Total: {sum(len(t.split()) for *_, t in LINES)} words, about {sum(est(t) for *_, t in LINES) / 60:.1f} minutes of speech "
         f"in a {A.offsets()[1] / 60:.1f}-minute video.", ""]
    for sec in V.ORDER:
        mine = [r for r in rows if r[0] == sec]
        if not mine:
            continue
        d = V.SECTIONS[sec].dur
        L += [f"## {sec}: {mmss(O[sec])}–{mmss(O[sec] + d)} ({d:.1f} s)", "",
              "| # | at | line | ≈ s |", "|---:|---:|---|---:|"]
        for _, k, cue, txt, end, nxt, ok in mine:
            L.append(f"| {k} | {mmss(O[sec] + cue)} | {txt} | {est(txt):.1f}{'' if ok else ' ⚠'} |")
        paste, prev_end = [], mine[0][2]
        for _, k, cue, txt, end, nxt, ok in mine:
            gap = cue - prev_end
            while gap > 0.25:                                  # ElevenLabs takes breaks of up to 3 s
                g = min(3.0, gap)
                paste.append(f'<break time="{g:.1f}s" />')
                gap -= g
            paste.append(txt)
            prev_end = max(end, cue)
        L += ["", "Paste into ElevenLabs:", "", "```", "\n".join(paste), "```", ""]
    warn = [r for r in rows if not r[6]]
    if warn:
        L += ["Lines marked ⚠ may run into the next cue at this pace; speed the voice up slightly, or trim them.", ""]
    open(path, "w").write("\n".join(L))
    return path, warn


if __name__ == "__main__":
    check()
    p, warn = write()
    print("wrote", p)
    for sec, k, cue, txt, end, nxt, ok in warn:
        print(f"  tight: {sec} #{k} ends {end:.1f} s, next cue {nxt:.1f} s")
