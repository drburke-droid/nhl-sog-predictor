"""Find and re-render the film's missing or broken frames (run by fix_and_finish.bat, inside Blender).

    blender -b -y two_thousand_years_of_catastrophe.blend -S Film --python-exit-code 1 -P fix_frames.py

Checks every frame the Film scene writes (render4k/frame_0000.png ... frame_8273.png):
missing, empty (a placeholder left by an interrupted render), cut off (no PNG end marker), or the
wrong size (a draft). Broken frames are deleted, then the Film scene renders, which skips every
frame that is already there. Then it checks again. Exits with code 1 if anything is still missing.

Test options (after --): --last N renders only up to frame N; --percent P renders at P% (draft).
"""
import os, struct, sys, time
import bpy

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
sc = bpy.data.scenes["Film"]
if "--percent" in argv:
    sc.render.resolution_percentage = int(argv[argv.index("--percent") + 1])
if "--samples" in argv:
    sc.cycles.samples = int(argv[argv.index("--samples") + 1])
first = sc.frame_start
last = int(argv[argv.index("--last") + 1]) if "--last" in argv else sc.frame_end
W = sc.render.resolution_x * sc.render.resolution_percentage // 100
H = sc.render.resolution_y * sc.render.resolution_percentage // 100
folder = bpy.path.abspath("//render4k")
PNG_SIG = b"\x89PNG\r\n\x1a\n"
IEND = b"\x00\x00\x00\x00IEND\xaeB`\x82"


def path(f):
    return os.path.join(folder, f"frame_{f:04d}.png")


def problem(f):
    """None if frame f is a whole PNG of the right size, else what is wrong with it."""
    p = path(f)
    if not os.path.exists(p):
        return "missing"
    n = os.path.getsize(p)
    if n == 0:
        return "empty"
    with open(p, "rb") as fh:
        head = fh.read(24)
        fh.seek(max(0, n - 12))
        tail = fh.read(12)
    if head[:8] != PNG_SIG:
        return "not a PNG"
    if tail != IEND:
        return "cut off"
    w, h = struct.unpack(">II", head[16:24])
    if (w, h) != (W, H):
        return f"wrong size {w}x{h}"
    return None


def scan():
    bad = {}
    for f in range(first, last + 1):
        why = problem(f)
        if why:
            bad[f] = why
    return bad


def ranges(frames):
    out, start = [], None
    for f in sorted(frames):
        if start is None:
            start = prev = f
        elif f == prev + 1:
            prev = f
        else:
            out.append((start, prev)); start = prev = f
    if start is not None:
        out.append((start, prev))
    return ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in out)


os.makedirs(folder, exist_ok=True)
total = last - first + 1
print(f"\n=== Checking {total} frames in {folder} (expected {W}x{H}) ===", flush=True)
bad = scan()
if not bad:
    print("All frames are present and whole. Nothing to render.", flush=True)
else:
    kinds = {}
    for why in bad.values():
        kinds[why] = kinds.get(why, 0) + 1
    print(f"{len(bad)} frames to render: " + ", ".join(f"{n} {k}" for k, n in kinds.items()), flush=True)
    print(f"Frames: {ranges(bad)}", flush=True)
    for f, why in bad.items():
        if why != "missing":
            os.remove(path(f))
    print("\n=== Rendering the missing frames (frames already there are skipped) ===", flush=True)
    sc.frame_start, sc.frame_end = first, last
    sc.render.use_overwrite = False
    sc.render.use_placeholder = False
    sc.render.filepath = "//render4k/frame_"
    bpy.context.window_manager  # (background: the operator renders the scene named here)
    t0 = time.time()
    bpy.ops.render.render(animation=True, scene=sc.name)
    print(f"Rendered in {(time.time() - t0) / 60:.1f} minutes.", flush=True)
    bad = scan()
    if bad:
        print(f"\nSTILL MISSING OR BROKEN: {len(bad)} frames: {ranges(bad)}", flush=True)
        sys.exit(1)
    print("\nAll frames are now present and whole.", flush=True)
