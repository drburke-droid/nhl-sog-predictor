"""Original score for the coast film, synthesized from scratch with the flights film's instruments
(../flights/score.py): no samples, no licences.

A minor, 104 BPM, Am-F-C-G. Quiet under the title and the Calgary demo (pad, held bass, a slow arpeggio);
a riser into light speed and an impact when the fish reaches it; full groove for the run, with a breakdown
before the halfway point and a whole-step lift at halfway; it thins out as the fish slows down and resolves on
a major chord when the fish is back in Calgary, ringing under the pull-back and the end card.

    python music.py   -> out/score.wav  (loudness-normalised to -14 LUFS)
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "flights"))
sys.path.insert(0, HERE)
import score as SC  # noqa: E402

SC.BPM = 104.0
SC.BEAT = 60.0 / SC.BPM
SC.BAR = 4 * SC.BEAT
SC.PROG[:] = [(57, True), (53, False), (60, False), (55, False)]      # Am F C G
OUT = os.path.join(HERE, "out")


def main():
    import film
    F = film.Film()
    tl = F.tl
    to_tau = lambda t: tl.ramp_end + (t - tl.km_intro / film.C)        # light-time -> video time in the run
    half = to_tau(tl.total / 2)
    total = tl.end
    ev = F.ev
    print(f"light speed {tl.ramp_end:.2f}  half {half:.2f}  slow-down {tl.outro_start:.2f}  arrive {tl.arrive:.2f}  end {total:.2f}")
    energy = [(0, 0.22), (film.FISH_START, 0.24), (ev["fork"], 0.34), (ev["back"], 0.38), (tl.ramp_end - 5, 0.5),
              (tl.ramp_end - 0.05, 0.62), (tl.ramp_end, 0.88), (half - 11, 0.88), (half - 10.9, 0.5), (half - 0.05, 0.55),
              (half, 0.93), (tl.outro_start - 2, 0.93), (tl.outro_start + 2, 0.62), (tl.outro_start + 6, 0.36),
              (tl.arrive, 0.32)]
    impacts = [(tl.ramp_end, 1.0), (half, 1.0), (tl.arrive, 1.25)]
    risers = [(tl.ramp_end - 6.0, tl.ramp_end), (half - 6.0, half), (tl.arrive - 4.0, tl.arrive)]
    sc = SC.Score(total, energy + [(tl.arrive + 0.01, 0.0), (total, 0.0)], impacts, risers, transpose_at=half,
                  final_at=tl.arrive)
    sc.compose()
    os.makedirs(OUT, exist_ok=True)
    SC.write(sc.mix(), os.path.join(OUT, "score.wav"))
    json.dump({"light_speed": tl.ramp_end, "half": half, "slow_down": tl.outro_start, "arrive": tl.arrive, "end": total},
              open(os.path.join(OUT, "score_cues.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
