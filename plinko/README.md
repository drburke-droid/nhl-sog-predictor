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
| `../videos/plinko/plinko_chaos_same_start_different_finish_narrated_6m03s.mp4` | **The chaos cut (main film)**: 1920x1080, 30 fps, H.264, 10,890 frames = 363.0 s, narrated (ElevenLabs, Lily voice; script `../videos/copy/plinko_chaos_NARRATION.md`). Thumbnails `plinko_chaos_thumb_A/B` |
| `../videos/plinko/where_to_drop_a_plinko_chip_blender_2m55s.mp4` | The Blender edition: 1920x1080, 30 fps, H.264 High / yuv420p, 5,250 frames = 175.0 s. Thumbnails `plinko_blender_thumb_A/B` (`thumbs3d.py`), stills in `media/stills_blender/` |
| `../videos/plinko/where_to_drop_a_plinko_chip_3m21s.mp4` | The film: 1920x1080, 30 fps, H.264 High / yuv420p, 6,030 frames = 201.0 s, on-screen captions, no narration |
| `../videos/plinko/where_to_drop_a_plinko_chip_SHORT_58s.mp4` | Vertical cut, 1080x1920, 1,725 frames = 57.5 s |
| `../videos/plinko/thumbnails/` | Thumbnails A (the question over a rain of chips) and B (the twin chips), 1280x720; `thumbs.py` |
| `media/stills/` | Frames from each section |
| `web/plinko_lab.html` | Plinko Drop Lab: the same physics in the browser (also published as a private artifact) |
| `results/*.json, *.npz` | Every number on screen: transfer matrices, fits, sensitivity, chaos |
| `data/real_outcomes.json` | The show's results, seasons 44 to 54 (tpirstats.com), plus the 2000 to 2011 tally |
| `../videos/copy/plinko_YOUTUBE.md` | Titles, description and chapters |

## The chaos cut: "Same start, different finish" (the main film)

A 6:03 film that uses the board to explain chaos. A winning drop is replayed from the same spot, matched to
the width of an atom, and lands in $500. The film then explains chaos theory (Lorenz's 1961 rerun and the
butterfly attractor), why round pins make chaos (a flat wall against a round pin) (a gap grows about 2.2 times per bounce), the predictability horizon, and the replay
experiment. Players may let go anywhere across the top, at the pins or up to 30 cm (12 in) above them, so the
experiment finds winning release points all over that area and replays each at a given precision:
16% win again at 1 micron (a fresh drop: 13%), 39% at 1 atom, never 100%. A map of every release
point across the top narrows to the middle gap at the pins, and a strip there zooms in on two winning spots, a
hundredfold at a time, to show an island and a knife edge. Then the start is made perfect and only
the board's temperature changes: one drop's own heat (0.00006 C) sends an exact replay of a $10,000 drop to $0,
and a sweep from a ten-billionth of a degree to 1 degree shows how fast replays stop winning;
and how a predictable pile emerges from unpredictable drops. Releases are drawn only from where the chip can
physically be let go (its centre within +-1.92 cm of the gap centre); an earlier 2:54 cut drew them from +-3 cm
and overstated replay odds (ACCURACY.md section 7).

```bash
python3 chaos2.py all          # growth, replay, film data, board temperature, the whole top (~45 min)
python3 narration.py lines     # ElevenLabs voice-over, phrase by phrase (sets each shot's length)
python3 prepare_chaos.py       # shot list; reuses the Blender plates and chip passes
CHIPS_ANIM=out/anim3d_chaos blender -b --factory-startup -P blender/render_chips.py -- hook
CHIPS_ANIM=out/anim3d_chaos blender -b --factory-startup -P blender/render_chips.py -- warm
CHIPS_ANIM=out/anim3d_chaos blender -b --factory-startup -P blender/render_chips.py -- heat
python3 composite_chaos.py segments hook deterministic twins chaos mechanism horizon crux islands heat warm realworld statistics outro
python3 composite_chaos.py assemble   # -> out/plinko_chaos.mp4
python3 narration.py mix              # ElevenLabs voice-over (needs an API key for api.elevenlabs.io) -> out/plinko_chaos_narrated.mp4
python3 thumbs_chaos.py
```

ACCURACY.md section 7 has every number, and a correction: the first Blender cut said micron accuracy was "no
better than random", which holds only for releases within 0.5 mm of dead centre.

## The Blender edition

A second cut of the film is rendered in Blender 4.5 (Cycles): a to-scale studio board built from the same
`board.py` geometry (steel pins, sawtooth walls, the triangular plexiglass cover, slot dividers, LED value
signs, a metal-rimmed chip), on a stage with stairs. The show's logo, sunburst frame and chip artwork are
left out on purpose. An LED bar-chart panel on the pedestal is the film's data display.

There is no GPU here, so full frames would cost about 53 s each on 4 CPU cores (about 37 hours for the film).
Instead it is rendered the way visual-effects shots are:
* **Plates** (`blender/render_plates.py`): each of three static cameras (hero, front, top close-up) renders
  the set once at 2880x1620, plus the plexiglass cover alone as a separate layer.
* **Chip passes** (`blender/render_chips.py`): only the moving chips, every frame, at 1920x1080 with real
  motion blur. Everything else is a holdout, so pins and walls still hide the chips behind them. Only the
  screen box around the chips renders: about 1 s a frame. A 2D push-in on a plate is the same as a longer
  lens with a lens shift, so the chip camera matches each frame's crop exactly.
* **Compositing** (`composite3d.py`): plate crop, contact shadow from the chip silhouette, chips, cover,
  lit LED cells with bloom, and every caption, trail and callout projected through Blender's own camera
  (checked against Blender's projection at startup).

```bash
python3 releasemap.py                       # the micron question: release map and twin tests
python3 blender/make_textures.py            # cover lattice mask and chip face, from board.py
python3 prepare3d.py                        # chip paths at 60 Hz (from physics.py) and the shot list
for c in front hero top; do blender -b --factory-startup -P blender/render_plates.py -- $c 32; done
./blender/run_all_chips.sh textbook measure twins open outro question zoom rain
python3 composite3d.py film --workers 4     # -> out/plinko_blender.mp4
```

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
* `releasemap.py`: could perfect aim pick the slot? Release points scanned from 9 cm down to 1 nm.
* `chaos2.py`, `prepare_chaos.py`, `composite_chaos.py`, `thumbs_chaos.py`, `narration.py`: the chaos cut.
* `prepare3d.py`, `blender/`, `composite3d.py`: the Blender edition (see above).
* `web/`: the Drop Lab page (a template plus `build_lab.py`, which fills in geometry and results).

Independent analysis; not affiliated with the show or the network.
