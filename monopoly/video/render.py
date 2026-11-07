"""Render scenes in parallel:  python3 render.py [-j 3] [scene-prefix ...]   (no prefixes = every scene)."""
import importlib
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from engine import SCENE_FILES

MODULES = ['scenes_intro', 'scenes_a', 'scenes_b', 'scenes_c', 'scenes_util']


def module_for(scene):
    for m in MODULES:
        try:
            if scene in importlib.import_module(m).SCENES:
                return m
        except ModuleNotFoundError:
            pass
    raise KeyError(scene)


if __name__ == '__main__':
    args = sys.argv[1:]
    j = 3
    if '-j' in args:
        k = args.index('-j'); j = int(args[k + 1]); del args[k:k + 2]
    todo = [s for s in SCENE_FILES if not args or any(s.startswith(a) for a in args)]

    def one(s):
        r = subprocess.run([sys.executable, module_for(s) + '.py', s], capture_output=True, text=True)
        print(s, 'ok' if r.returncode == 0 else 'FAILED\n' + r.stderr[-2000:], flush=True)

    with ThreadPoolExecutor(j) as ex:
        list(ex.map(one, todo))
