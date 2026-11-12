"""Turn on GPU rendering for a command-line render:  blender -b shot_Q1.blend -P use_gpu.py -a
(Inside Blender the same thing is Edit > Preferences > System > Cycles Render Devices.)"""
import bpy

prefs = bpy.context.preferences.addons["cycles"].preferences
for kind in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
    try:
        prefs.compute_device_type = kind
        prefs.get_devices()
    except Exception:
        continue
    gpus = [d for d in prefs.devices if d.type == kind]
    if gpus:
        for d in prefs.devices:
            d.use = d.type == kind
        for scene in bpy.data.scenes:
            scene.cycles.device = "GPU"
        print("Cycles will render on:", ", ".join(d.name for d in gpus), f"({kind})")
        break
else:
    print("No GPU found for Cycles: rendering on the CPU, which will be very slow at this size.")
