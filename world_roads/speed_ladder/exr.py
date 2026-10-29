"""Blender's multilayer EXR renders -> compact .npz for the compositor."""
import os
import numpy as np


def save(path, exr, passes=False):
    """Combined and depth always; normal, position and diffuse colour (for relighting) when asked"""
    import OpenEXR
    f = OpenEXR.File(exr)
    ch = {k: np.asarray(c.pixels) for part in f.parts for k, c in part.channels.items()}
    out = {"rgba": ch["ViewLayer.Combined"].astype(np.float16),
           "depth": np.minimum(ch["ViewLayer.Depth.Z"], 6e4).astype(np.float16)}
    if passes:
        out["normal"] = np.stack([ch[f"ViewLayer.Normal.{a}"] for a in "XYZ"], -1).astype(np.float16)
        out["position"] = np.clip(np.stack([ch[f"ViewLayer.Position.{a}"] for a in "XYZ"], -1), -6e4, 6e4).astype(np.float16)
        out["diffcol"] = ch["ViewLayer.Diffuse Color"].astype(np.float16)
    np.savez_compressed(path, **out)
    os.remove(exr)
