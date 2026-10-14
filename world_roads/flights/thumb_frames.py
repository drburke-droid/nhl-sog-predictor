"""Render text-free part-2 frames for thumbnail candidates: python thumb_frames.py 240 280 300"""
import os, sys
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flights_render as FR

R = FR.FlightRenderer()
for T in map(float, sys.argv[1:]):
    R.reset_to(T - 3.0)
    for tt in np.arange(T - 3.0, T + 1e-9, 1 / 30):
        E, _ = R.frame_light(float(tt), 0.88)
    Image.fromarray(R.finish(E, [])).save(os.path.join(FR.OUT, f"thumbbase_{T:g}.png"))
    print(T)
