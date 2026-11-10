"""
combined_03_render_chunk.py  –  Stage 3: render a range of frames from a template.

    blender -b <SIM>_<mode>_<camera>.blend -P combined_03_render_chunk.py -- START END

(render_chunks.sh runs many of these at once.)  The mode, run directory and
component files are read from the template, which must have been saved by
combined_02_build.py with MAKE_TEMPLATE = True.  Renders go to
renders_<mode>/frame_NNNN.png (fish_ for the fisheye camera, vr360_ for the
360-degree camera, with _3d added for stereo, e.g. vr360_3d_NNNN.png).
"""

import bpy, os, re, sys, json, traceback


# ===========================================================================
# SETTINGS
# ===========================================================================
BODY_ON     = True    # switch components off without rebuilding the template
REST_ON     = True
SMOOTH_BODY = True    # smooth-shade the mesh body
SAMPLES     = 16      # Cycles samples (denoised)


# ===========================================================================
# TEMPLATE
# ===========================================================================
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
if len(argv) != 2:
    raise SystemExit("Usage: blender -b template.blend -P combined_03_render_chunk.py -- START END")
START, END = map(int, argv)

scene = bpy.context.scene
if "COMPONENTS" not in scene or not scene.get("IS_TEMPLATE", False):
    raise SystemExit("Not a template from the current combined_02_build.py with MAKE_TEMPLATE = True")
MODE, RUN_DIR = scene["RENDER_MODE"], scene["RUN_DIR"]
ACTIVE = [c for c in json.loads(scene["COMPONENTS"])
          if {"body": BODY_ON, "rest": REST_ON}[c[0]]]

LABEL, TIMES = scene.get("TIME_LABEL", ""), json.loads(scene.get("FRAME_TIMES_S", "[]"))
cam     = scene.camera.data
PREFIX  = ({"FISHEYE_EQUIDISTANT": "fish", "EQUIRECTANGULAR": "vr360"}[cam.panorama_type]
           if cam.type == "PANO" else "frame") + ("_3d" if scene.render.use_multiview else "")
OUT_DIR = os.path.join(RUN_DIR, f"renders_{MODE.lower()}")
os.makedirs(OUT_DIR, exist_ok=True)
print(f"\nStage 3  {MODE}  frames {START}-{END}  components {[c[3] for c in ACTIVE]}"
      f"\n  -> {OUT_DIR}/{PREFIX}_NNNN.png\n")

body_mat = bpy.data.materials.get("BodyLook")
mist_mat = bpy.data.materials.get("MistLook")
pts_group = bpy.data.objects["PointsTemplate"].modifiers["Pts"].node_group
cut_centre = bpy.data.objects.get("CutCentre")


# ===========================================================================
# CYCLES ON THE GPU
# ===========================================================================
scene.render.engine = "CYCLES"
scene.cycles.device = "GPU"
scene.cycles.samples = SAMPLES
scene.cycles.use_denoising = True
prefs = bpy.context.preferences.addons["cycles"].preferences
for dev_type in ("OPTIX", "CUDA"):
    try:
        prefs.compute_device_type = dev_type; prefs.get_devices(); break
    except Exception:
        pass
for dev in prefs.devices:
    dev.use = dev.type == prefs.compute_device_type


# ===========================================================================
# PER-FRAME OBJECTS
# ===========================================================================
def empty_ply(path):
    """True for a PLY with no vertices (e.g. no rest particles yet, early on),
    which Blender's importer refuses; such a component is simply not drawn."""
    with open(path, "rb") as f:
        return b"\nelement vertex 0\n" in f.read(1024)


def ply_centre(path):
    """The target centre Stage 1 records in a particle file's header (for a cutaway)."""
    with open(path, "rb") as f:
        m = re.search(rb"\ncomment target_centre (\S+) (\S+) (\S+)\n", f.read(1024))
    return tuple(map(float, m.groups())) if m else (0.0, 0.0, 0.0)


def load(kind, path):
    if kind == "mist":
        bpy.ops.object.volume_import(filepath=path, align="WORLD")
        ob = bpy.context.active_object; ob.location = (0, 0, 0)
        ob.data.grids.load()
        ob.data.materials.clear(); ob.data.materials.append(mist_mat)
        return ob
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.wm.ply_import(filepath=path)
    ob = bpy.context.selected_objects[0]
    if kind == "mesh":
        if SMOOTH_BODY:
            for poly in ob.data.polygons: poly.use_smooth = True
        ob.data.materials.clear(); ob.data.materials.append(body_mat)
    else:                                   # particles: material is set in the node group
        ob.modifiers.new("Pts", "NODES").node_group = pts_group
        if cut_centre is not None:          # the cutaway plane passes through the target's centre
            cut_centre.location = ply_centre(path)
    return ob


# ===========================================================================
# RENDER LOOP   (Blender frame = snapshot + 1, as for the camera keyframes)
# ===========================================================================
failed = []
for i in range(START, END + 1):
    objs = []
    try:                                    # one bad frame should not stop the chunk
        for role, kind, sub, pat in ACTIVE:
            path = os.path.join(RUN_DIR, sub, pat.format(i))
            if not os.path.exists(path):
                print(f"  frame {i:04d}: missing {path}")
            elif kind == "mist" or not empty_ply(path):
                objs.append(load(kind, path))
        if objs:
            scene.frame_set(i + 1)
            scene.render.filepath = os.path.join(OUT_DIR, f"{PREFIX}_{i:04d}.png")
            if LABEL:
                t = TIMES[i] if i < len(TIMES) else float("nan")
                scene.render.stamp_note_text = LABEL.format(seconds=t, hours=t / 3600, days=t / 86400)
            bpy.ops.render.render(write_still=True)
    except Exception:
        traceback.print_exc(); print(f"  [FAILED] frame {i:04d}"); failed.append(i)
    for ob in objs:
        data = ob.data
        bpy.data.objects.remove(ob, do_unlink=True)
        (bpy.data.meshes if isinstance(data, bpy.types.Mesh) else bpy.data.volumes).remove(data)
    bpy.data.orphans_purge(do_local_ids=True)   # materials and node group have fake users

if failed:
    print(f"\n{len(failed)} frame(s) failed: {' '.join(map(str, failed))}")
    sys.exit(1)                             # so render_chunks.sh can report this chunk
print("\nChunk complete.")
