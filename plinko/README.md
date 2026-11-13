# Plinko: where should you drop the chip?

A measured, to-scale model of the TV Plinko board, a rigid-body physics simulator tuned against 1,029
real chips from the show, and a 3:21 film plus a 0:58 Short on how to play it.

**The answer.** Drop in the middle. It is the best drop, or tied for best, in every model here. But on a
board that behaves like the real one, the middle three positions are a tie at about $1,829 a chip, and
the far edge still pays about $1,617. The textbook coin-flip model says $2,558 against $780. How you
release the chip (spin, toss, steady or sloppy hand) makes no measurable difference. What matters most is
winning more chips, because each one is worth about $1,800. When unsure on a small prize, the last digit
was the correct one 61% of the time.

See [ACCURACY.md](ACCURACY.md) for every source, measurement and error bar, and for the one open question:
the show's data fit a chaotic board with contestants aiming at the middle, or a coin-flip board with
contestants dropping all over. Drop-position logs would settle it.

## The board (measured from broadcast frames; no official drawings are public)

| | |
|---|---|
| Slots | 9: $100, $500, $1,000, $0, **$10,000**, $0, $1,000, $500, $100 |
| Slot pitch / pin spacing | 17.8 cm (7.0 in) |
| Pin rows | 13, alternating 8 and 7 pins: 98 steel pins; a chip meets 12 of them |
| Row spacing | 15.4 cm (6.1 in), equilateral lattice |
| Side walls | sawtooth, 6 teeth a side standing in for the missing end pins |
| Chip | 13 cm (5.1 in) across, about 3/4 in thick |
| Playfield | 160 cm (63 in) wide; 185 cm (73 in) from the top pin row to the bottom one |

`python3 board.py` prints the full geometry.

## Deliverables

| File | What |
|---|---|
| `../videos/plinko/where_to_drop_a_plinko_chip_3m21s.mp4` | The film: 1920x1080, 30 fps, H.264 High / yuv420p, 6,030 frames = 201.0 s, on-screen captions, no narration |
| `../videos/plinko/where_to_drop_a_plinko_chip_SHORT_58s.mp4` | Vertical cut, 1080x1920, 1,725 frames = 57.5 s |
| `../videos/plinko/thumbnails/` | Thumbnails A (the question over a rain of chips) and B (the twin chips), 1280x720; `thumbs.py` |
| `media/stills/` | Frames from each section |
| `web/plinko_lab.html` | Plinko Drop Lab: the same physics in the browser (also published as a private artifact) |
| `results/*.json, *.npz` | Every number on screen: transfer matrices, fits, sensitivity, chaos |
| `data/real_outcomes.json` | The show's results, seasons 44 to 54 (tpirstats.com), plus the 2000 to 2011 tally |
| `../videos/copy/plinko_YOUTUBE.md` | Titles, description and chapters |

## Run it

```bash
pip install -r requirements.txt     # numpy, numba, pycairo; plus ffmpeg and the Inter font
python3 galton.py                   # the textbook model: prints the exact 9x9 matrix and EVs
python3 physics.py                  # benchmark: 20,000 chips from the middle (~25 s on 4 cores)
python3 analyze.py all              # kernels, fit, final, strategy, sensitivity, technique, chaos (~30 min)
python3 render.py prepare           # animation paths
python3 render.py video --workers 4 # the film -> out/plinko_optimal_play.mp4 (~5 min on 4 cores)
python3 render.py short --workers 4 # the Short -> out/plinko_optimal_play_SHORT.mp4
python3 thumbs.py                   # thumbnails -> media/thumbnails/
python3 web/build_lab.py            # rebuild the interactive page from results/
```

Seeds are fixed and every chip carries its own random stream, so reruns reproduce the same numbers.

## Files

* `board.py`: geometry, prize values, measured constants (with their sources in comments).
* `galton.py`: the coin-flip model as an exact Markov chain (reproduces Biesterfeld 2001: $2,557.91 from
  the middle).
* `physics.py`: numba rigid-body engine; a disc with spin against round pins, sawtooth walls and dividers.
* `analyze.py`: Monte Carlo sweeps, the fit to the show's data, sensitivity and the twin-chip chaos test.
* `draw.py`, `render.py`: cairo drawing and the frame-by-frame film; everything is computed, nothing is
  footage from the show.
* `web/`: the Drop Lab page (a template plus `build_lab.py`, which fills in geometry and results).

Independent analysis; not affiliated with the show or the network.
