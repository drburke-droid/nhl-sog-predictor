"""Build every scene once and report cue phrases the narration no longer contains (run after editing the script)."""
import importlib
import sys

import sfx

from engine import SCENE_FILES
from render import module_for

bad = 0
for s in SCENE_FILES:
    try:
        sc = importlib.import_module(module_for(s)).SCENES[s]()
        sfx.events(sc)                       # cues used only by sound effects
        for f in (0.1, 0.5, 0.8, 0.99):      # cues looked up while drawing
            sc.frame(sc.dur * f)
        print('ok   ', s, flush=True)
    except KeyError as e:
        bad += 1
        print('MISS ', s, e, flush=True)
sys.exit(1 if bad else 0)
