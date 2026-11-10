"""
combined_02_build.py  –  Stage 2: build the Blender scene and the template.

    blender -b -P combined_02_build.py                 # TEST_FRAMES
    blender -b -P combined_02_build.py -- 300-319      # frames on the command line

With MAKE_TEMPLATE = False, renders test frames to renders/ and saves a .blend
that still contains them (the one to open in the GUI).  With MAKE_TEMPLATE =
True, builds the mist .vdb files (mist modes), renders TEMPLATE_FRAME as a
check, removes the per-frame objects and saves the template for Stage 3.
A .vdb file is (re)built only if it is missing or out of date: like the
Stage 1 files, each carries a fingerprint (in its file metadata) of the volume
file and VDB settings it was made from.
The template records the mode, run directory and component files, so Stage 3
needs no settings of its own.

All colour is decided here, from the temperature and iron fraction that the
Stage 1 files carry: one node group, HeatLook, colours the body, the
particles and the mist alike.  So the look can be tuned with test renders
alone, without rerunning Stage 1 or rebuilding the .vdb files.

RENDER_MODE must match Stage 1.
Components:  body = mesh (meshes/body_NNNN.ply) or particles (particles/body_NNNN.ply)
             rest = particles (particles/rest_NNNN.ply) or mist (volume/*.npz -> vdb/mist_NNNN.vdb)
"""

import bpy, os, sys, glob, re, math, json, hashlib
import numpy as np
from mathutils import Vector, Euler, Matrix


# ===========================================================================
# PATHS
# ===========================================================================
SIM_NAME   = "pE65_Th35_b60v10_3e6"
BASE_DIR   = "/cosma5/data/dp004/vreke/swift/rssse"
RUN_DIR    = os.path.join(BASE_DIR, SIM_NAME)
# Lengths are in Blender units: the simulation's length unit (R_E here), as set in Stage 1, with
# each frame's centre of mass at the origin.


# ===========================================================================
# WHAT TO BUILD
# ===========================================================================
# "PARTICLES" | "MESH_PARTICLES" | "MESH_MIST" | "PARTICLES_MIST"
RENDER_MODE = "MESH_MIST"
BODY_ON     = True        # include the body in test renders
REST_ON     = True        # include the rest (and build the VDBs, mist modes)

MAKE_TEMPLATE  = False
TEMPLATE_FRAME = 800
TEST_FRAMES    = [0, 200, 400, 600, 800, 1000, 1200, 1800]   # or frames after "--"


# ===========================================================================
# LOOK
# ===========================================================================
# Colour, for body, particles and mist alike.  Every surface is its
# material's own colour, lit by the suns; hot material also glows, in the
# PALETTE colour for its temperature, with a strength rising smoothly (in
# log T) from 0 to 1 across GLOW_K.  So cool crust stays dark and only hot
# material shines.  Colours are display (sRGB) values, as in a colour picker.
#
# Changing with time: GLOW_K, PALETTE_SHIFT and the *_EMIT_GAIN settings may
# each instead be {snapshot frame: value, ...}.  The value then changes
# smoothly between those frames, and stays fixed before the first and after
# the last, e.g. GLOW_K = {0: (3000, 7000), 300: (3000, 7000), 600: (2800, 16000)}.
ROCK_RGB = (0.31, 0.19, 0.14)
IRON_RGB = (0.72, 0.52, 0.16)
GLOW_K   = {0:   (3000.0, 7000.0),     # impact: cracks at 6,000-8,500 K glow fully, crust stays dark
            300: (3000.0, 7000.0),
            600: (2800.0, 16000.0)}    # then this look, held for the rest of the run
PALETTE_SHIFT = 1.0          # glow colour is taken from PALETTE at T x this: > 1 whiter
PALETTE  = [                 # (T_K, R, G, B): glow colour, interpolated in log T
    ( 2500, 0.30, 0.05, 0.01),
    ( 4000, 0.60, 0.10, 0.02),
    ( 6000, 0.85, 0.25, 0.04),
    ( 9000, 1.00, 0.40, 0.08),
    (14000, 1.00, 0.52, 0.14),
    (20000, 1.00, 0.65, 0.25),
    (40000, 1.00, 0.90, 0.70),   # white-hot, reached only with PALETTE_SHIFT > 1 (or the hottest mist)
]

# Mesh body
BODY_EMIT_GAIN   = {0: 1.5, 300: 1.5, 600: 1.0}    # glow strength of the hottest surface

# Particles (body and rest): sphere radius = PARTICLE_RADIUS * radius_factor
# (0.5-1, set by density in Stage 1).  0.004-0.005 keeps the mantle opaque.
PARTICLE_RADIUS  = 0.005
EMIT_GAIN        = 1.0    # glow strength of the hottest dense particles
PARTICLE_ALPHA   = 0.7    # opacity (x the density fade of rest particles)
DEBRIS_EMIT_FRAC = 0.3    # glow fraction for the least dense (radius_factor 0.5)

# Particle colours: "HEAT" (material colour, lit, glowing with temperature, as
# above) or "BODY": a fixed colour for each body and material, to show where
# the impactor's material ends up.
PARTICLE_COLOURS = "HEAT"
BODY_COLOURS = {"target rock":   (0.55, 0.55, 0.58), "target iron":   (0.90, 0.55, 0.15),
                "impactor rock": (0.25, 0.50, 0.90), "impactor iron": (0.85, 0.15, 0.15)}
BODY_GLOW = 0.3           # "BODY" colours: self-glow, so shaded sides and the cut still show

# Cutaway (particles only): remove the particles on the near side of a plane
# through the target's centre, to show the layers inside.  None = no cutaway;
# "CAMERA" = the plane faces the camera, turning with it; or the plane's normal
# in simulation axes, e.g. (0, 0, 1) removes everything with z > 0.
CUTAWAY = None

# Mist
VDB_RES          = 128    # .vdb grid cells per side
MIST_DENSITY     = 0.5    # opacity of the densest mist
MIST_EMIT_GAIN   = 0.05   # glow of the densest, hottest mist
MIST_FADE_RHO    = (1.0, 1000.0)  # kg/m^3: mist opacity and glow rise smoothly (in log
                                 # density) from 0 to full between these, like REST_FADE_RHO
VDB_SMOOTH       = 2      # smoothing passes on the .vdb grids

# Camera
CAMERA_MODE     = "FISHEYE"  # "PERSP" | "ORTHO" | "FISHEYE" (dome) | "VR360" (headset)
# Adjusts PATH for this template: (distance scale, turn about the spin (z) axis in
# degrees, elevation above the collision plane in degrees, or None = as in PATH).
# Each PATH position keeps its distance (x scale) while being turned and raised,
# exactly as camera_clearance.py does, so its clearances apply.  Not used by ORTHO.
CAMERA_VIEW     = (0.23, 180, 20)
#                 (1.0, 0, None)       # flat screen, as PATH
#                 (0.23, 180, 20)        # dome: closer, looking down on the collision
#                 (0.23, 180, 20)      # headset: closer, from the impact side, raised
DOME_TILT_DEG   = 40.0
DOME_AZ_DEG     = 10.0

# Stereo 3D: left and right views in one image, for a 3D screen or headset.
# PERSP and FISHEYE give side-by-side frames, VR360 the top-bottom 360-degree
# format that headset video players expect (not for ORTHO).  Doubles render time.
STEREO          = False
STEREO_DEPTH    = 1 / 30   # eye separation / camera distance: larger = stronger 3D

# Time label, burned into the top-left corner of every frame; None = no label.
# {seconds}, {hours} and {days} are the snapshot time: the number in its file
# name (e.g. ..._80000.hdf5) times SNAPSHOT_NAME_S.  Best with PERSP or ORTHO,
# since that corner lies off the dome (FISHEYE) or behind the viewer (VR360).
TIME_LABEL      = None     # e.g. "  Time = {hours:.1f} h"
TIME_LABEL_SIZE = 32       # font size, pixels
SNAPSHOT_NAME_S = 1.0      # seconds per unit of the number in the snapshot file names


# ===========================================================================
# MODE AND COMPONENTS
# ===========================================================================
MODES   = {"PARTICLES":      ("PARTICLES", "PARTICLES"),
           "MESH_PARTICLES": ("MESH",      "PARTICLES"),
           "MESH_MIST":      ("MESH",      "MIST"),
           "PARTICLES_MIST": ("PARTICLES", "MIST")}
if RENDER_MODE not in MODES:
    raise SystemExit(f"Unknown RENDER_MODE {RENDER_MODE!r}; choose from {list(MODES)}")
BODY_STYLE, REST_STYLE = MODES[RENDER_MODE]

# role, kind, folder, file pattern – stored in the template for Stage 3
COMPONENTS = [
    ("body", "mesh",      "meshes",    "body_{:04d}.ply") if BODY_STYLE == "MESH" else
    ("body", "particles", "particles", "body_{:04d}.ply"),
    ("rest", "mist",      "vdb",       "mist_{:04d}.vdb") if REST_STYLE == "MIST" else
    ("rest", "particles", "particles", "rest_{:04d}.ply"),
]
ACTIVE = [c for c in COMPONENTS if {"body": BODY_ON, "rest": REST_ON}[c[0]]]
MIST   = REST_STYLE == "MIST"
VOL_DIR, VDB_DIR = os.path.join(RUN_DIR, "volume"), os.path.join(RUN_DIR, "vdb")
# <SIM>_<mode>_<camera>.blend, e.g. ..._mesh_mist_persp.blend or ..._mesh_mist_vr360_3d.blend;
# a test run saves ..._test.blend instead, so it never overwrites a template.
CAMERA_NAME = CAMERA_MODE.lower() + ("_3d" if STEREO else "")
BLEND_OUT = os.path.join(RUN_DIR, f"{SIM_NAME}_{RENDER_MODE.lower()}_{CAMERA_NAME}"
                                  f"{'' if MAKE_TEMPLATE else '_test'}.blend")


# ===========================================================================
# FRAMES
# ===========================================================================
def parse_frames(args):
    """'800', '300-319' (inclusive) and '0-2160/100' (with step), in any mix."""
    frames = []
    for a in args:
        a, _, step = a.partition("/")
        lo, _, hi  = a.partition("-")
        frames += list(range(int(lo), int(hi or lo) + 1, int(step or 1)))
    return frames

_cli  = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
if _cli:
    TEST_FRAMES = parse_frames(_cli)

_, _, _bdir, _bpat = COMPONENTS[0]
_found = glob.glob(os.path.join(RUN_DIR, _bdir, _bpat.replace("{:04d}", "[0-9]" * 4)))
if not _found:
    raise SystemExit(f"\nNo Stage 1 body files in {os.path.join(RUN_DIR, _bdir)} – run Stage 1 first.\n")
ALL_FRAMES = sorted(int(re.search(r"(\d{4})\.ply$", f).group(1)) for f in _found)
NFRAMES    = max(ALL_FRAMES) + 1

render_frames = [TEMPLATE_FRAME] if MAKE_TEMPLATE else TEST_FRAMES

# Snapshot time of each frame, from the snapshot file names (frame 0 = ICs, t = 0)
_snap = re.compile(rf"{re.escape(SIM_NAME)}_(\d+)\.hdf5$")
_nums = sorted(int(m.group(1)) for m in (_snap.search(os.path.basename(f))
               for f in glob.glob(os.path.join(RUN_DIR, "snapshots", "*.hdf5"))) if m)
FRAME_TIMES_S = [0.0] + [n * SNAPSHOT_NAME_S for n in _nums]


def time_label(frame):
    t = FRAME_TIMES_S[frame] if frame < len(FRAME_TIMES_S) else float("nan")
    return TIME_LABEL.format(seconds=t, hours=t / 3600, days=t / 86400)
vdb_frames    = ALL_FRAMES if MAKE_TEMPLATE else TEST_FRAMES
print(f"{RENDER_MODE}: {len(ALL_FRAMES)} frames found; "
      f"{'TEMPLATE' if MAKE_TEMPLATE else 'TEST'} run, rendering {render_frames}")


# ===========================================================================
# MIST: PALETTE AND VDBs  (before read_factory_settings wipes the session)
# ===========================================================================
try:
    import openvdb
except ImportError:
    import pyopenvdb as openvdb


def build_vdbs(frames):
    os.makedirs(VDB_DIR, exist_ok=True)

    def downsample(a):
        f = a.shape[0] // VDB_RES; s = VDB_RES * f
        return a[:s, :s, :s].reshape(VDB_RES, f, VDB_RES, f, VDB_RES, f).mean(axis=(1, 3, 5))

    def smooth(a):                  # 7-point average
        for _ in range(VDB_SMOOTH):
            a = (a + sum(np.roll(a, s, ax) for ax in range(3) for s in (1, -1))) / 7.0
        return a

    def grid(arr, name, box):         # VDB_RES^3 cells filling the box, centred on the origin
        g = openvdb.FloatGrid(); s = box / VDB_RES; o = s / 2 - box / 2
        g.copyFromArray(np.ascontiguousarray(arr, dtype=np.float32))
        g.transform = openvdb.createLinearTransform([[s, 0, 0, 0], [0, s, 0, 0], [0, 0, s, 0], [o, o, o, 1]])
        g.gridClass = openvdb.GridClass.FOG_VOLUME
        g.name = name
        return g

    n = 0
    print(f"Checking {len(frames)} VDB files (and building any out of date) ...", flush=True)
    for k, i in enumerate(frames):
        if k and k % 250 == 0:
            print(f"  {k}/{len(frames)} checked, {n} built", flush=True)
        vp, zp = vdb_paths(i)
        if not os.path.exists(zp) or vdb_stamp(vp) == vdb_fingerprint(zp):
            continue
        with np.load(zp) as z:                     # temperature is mass-weighted
            if "box_size" not in z.files:
                raise SystemExit(f"{zp} was made by an older Stage 1: rerun Stage 1 for this mode.")
            d, t = z["density"].astype(np.float32), z["temperature"].astype(np.float32)
            box  = float(z["box_size"])
        rho, rhoT = downsample(d), downsample(d * t)
        rho, rhoT = smooth(rho), smooth(rhoT)
        T = np.where(rho > 0, rhoT / np.maximum(rho, 1e-30), 0.0)
        write_vdb(vp, [grid(rho, "rho", box), grid(T, "T_K", box)], vdb_fingerprint(zp))
        n += 1
    print(f"Built {n} VDB files in {VDB_DIR}; the others were up to date")


def vdb_paths(i):
    return os.path.join(VDB_DIR, f"mist_{i:04d}.vdb"), os.path.join(VOL_DIR, f"volume_{i:04d}.npz")


def vdb_fingerprint(npz):
    """Depends on the volume file's own fingerprint and the VDB settings."""
    try:
        with np.load(npz) as z:
            src = str(z["fingerprint"]) if "fingerprint" in z.files else "unstamped"
    except Exception:
        src = "unreadable"
    key = json.dumps([src, VDB_RES, VDB_SMOOTH, "rho,T_K,centred"])
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def vdb_stamp(path):
    """The fingerprint stored in a .vdb file's metadata; None if missing or unstamped."""
    try:
        return openvdb.readMetadata(path).get("fingerprint")
    except Exception:
        return None


def write_vdb(path, grids, fp):
    """Write via a temporary file, so an interrupted job never leaves a partial file."""
    openvdb.write(path + ".part", grids=grids, metadata={"fingerprint": fp})
    os.replace(path + ".part", path)


if MIST and REST_ON:
    build_vdbs(vdb_frames)


# ===========================================================================
# MATERIALS
# ===========================================================================
def new_material(name):
    mat = bpy.data.materials.new(name); mat.use_nodes = True
    for n in list(mat.node_tree.nodes):
        mat.node_tree.nodes.remove(n)
    return mat, mat.node_tree.nodes, mat.node_tree.links


def math_node(nodes, op, value):
    m = nodes.new("ShaderNodeMath"); m.operation = op; m.inputs[1].default_value = value
    return m


def sock(sockets, name):
    """The visible socket called name (Mix nodes keep hidden ones for other types)."""
    return next(x for x in sockets if x.name == name and x.enabled)


def linear(rgb):
    """Display (sRGB) colour -> the linear values Blender's nodes work in."""
    return tuple(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb)


def animate(socket, setting, value=float):
    """Set socket from a setting that is a constant or {snapshot frame: value};
    value() turns the setting into the socket's number."""
    keys = setting if isinstance(setting, dict) else {0: setting}
    for frame, v in sorted(keys.items()):
        socket.default_value = value(v)
        if len(keys) > 1:                        # Blender frame = snapshot + 1
            socket.keyframe_insert("default_value", frame=frame + 1)


def heat_look():
    """Node group HeatLook: (Temperature K, Iron 0-1) -> Colour, Glow Colour, Glow 0-1."""
    ng = bpy.data.node_groups.new("HeatLook", "ShaderNodeTree")
    for name, kind, io in (("Temperature", "Float", "INPUT"), ("Iron", "Float", "INPUT"),
                           ("Colour", "Color", "OUTPUT"), ("Glow Colour", "Color", "OUTPUT"),
                           ("Glow", "Float", "OUTPUT")):
        ng.interface.new_socket(name, in_out=io, socket_type=f"NodeSocket{kind}")
    N, L = ng.nodes, ng.links
    gin, gout = N.new("NodeGroupInput"), N.new("NodeGroupOutput")

    base = N.new("ShaderNodeMix"); base.data_type = "RGBA"         # rock -> iron
    sock(base.inputs, "A").default_value, sock(base.inputs, "B").default_value = (*linear(ROCK_RGB), 1), (*linear(IRON_RGB), 1)
    L.new(gin.outputs["Iron"], sock(base.inputs, "Factor"))
    L.new(sock(base.outputs, "Result"), gout.inputs["Colour"])

    logT = math_node(N, "LOGARITHM", 10.0); safe = math_node(N, "MAXIMUM", 1.0)
    L.new(gin.outputs["Temperature"], safe.inputs[0]); L.new(safe.outputs[0], logT.inputs[0])

    lo, hi = math.log10(PALETTE[0][0]), math.log10(PALETTE[-1][0])
    pos  = N.new("ShaderNodeMapRange"); pos.clamp = True             # log T -> 0-1 along palette
    animate(pos.inputs["From Min"], PALETTE_SHIFT, lambda f: lo - math.log10(f))
    animate(pos.inputs["From Max"], PALETTE_SHIFT, lambda f: hi - math.log10(f))
    ramp = N.new("ShaderNodeValToRGB"); cr = ramp.color_ramp
    cr.elements.remove(cr.elements[1])
    cr.elements[0].color = (*linear(PALETTE[0][1:]), 1)
    for t, *rgb in PALETTE[1:]:
        cr.elements.new((math.log10(t) - lo) / (hi - lo)).color = (*linear(rgb), 1)
    L.new(logT.outputs[0], pos.inputs["Value"]); L.new(pos.outputs["Result"], ramp.inputs["Fac"])
    L.new(ramp.outputs["Color"], gout.inputs["Glow Colour"])

    glow = N.new("ShaderNodeMapRange"); glow.interpolation_type = "SMOOTHSTEP"
    animate(glow.inputs["From Min"], GLOW_K, lambda r: math.log10(r[0]))
    animate(glow.inputs["From Max"], GLOW_K, lambda r: math.log10(r[1]))
    L.new(logT.outputs[0], glow.inputs["Value"]); L.new(glow.outputs["Result"], gout.inputs["Glow"])
    return ng


def heat_node(N, L, iron=True):
    """A HeatLook node fed from the 'T_K' (and 'iron') attributes.  (Not 'temperature',
    which Cycles reserves for its own smoke simulations.)"""
    g = N.new("ShaderNodeGroup"); g.node_tree = bpy.data.node_groups["HeatLook"]
    for name, socket in (("T_K", "Temperature"), ("iron", "Iron"))[:2 if iron else 1]:
        a = N.new("ShaderNodeAttribute"); a.attribute_name = name
        L.new(a.outputs["Fac"], g.inputs[socket])
    return g


def body_material():
    """Mesh body: its material colour, lit; plus glow from its temperature."""
    mat, N, L = new_material("BodyLook")
    out, hl = N.new("ShaderNodeOutputMaterial"), heat_node(N, L)
    bsdf = N.new("ShaderNodeBsdfPrincipled"); bsdf.inputs["Roughness"].default_value = 0.85
    gain = math_node(N, "MULTIPLY", 1.0); animate(gain.inputs[1], BODY_EMIT_GAIN)
    L.new(hl.outputs["Colour"], bsdf.inputs["Base Color"])
    L.new(hl.outputs["Glow Colour"], bsdf.inputs["Emission Color"])
    L.new(hl.outputs["Glow"], gain.inputs[0]); L.new(gain.outputs[0], bsdf.inputs["Emission Strength"])
    L.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def particle_material():
    """Particles: as the body, partly transparent.  Glow falls to DEBRIS_EMIT_FRAC
    for the least dense; the opacity attribute carries the rest-particle fade."""
    mat, N, L = new_material("ParticleLook")
    out, hl = N.new("ShaderNodeOutputMaterial"), heat_node(N, L)
    al, rf = N.new("ShaderNodeAttribute"), N.new("ShaderNodeAttribute")
    al.attribute_name, rf.attribute_name = "opacity", "radius_factor"
    bsdf = N.new("ShaderNodeBsdfPrincipled"); bsdf.inputs["Roughness"].default_value = 0.9
    alpha = math_node(N, "MULTIPLY", PARTICLE_ALPHA)
    L.new(al.outputs["Fac"], alpha.inputs[0]); L.new(alpha.outputs[0], bsdf.inputs["Alpha"])
    if PARTICLE_COLOURS == "BODY":
        col = body_colour(N, L)
        L.new(col, bsdf.inputs["Base Color"]); L.new(col, bsdf.inputs["Emission Color"])
        bsdf.inputs["Emission Strength"].default_value = BODY_GLOW
        L.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
        return finish_particle_material(mat)
    L.new(hl.outputs["Colour"], bsdf.inputs["Base Color"])
    L.new(hl.outputs["Glow Colour"], bsdf.inputs["Emission Color"])
    w = N.new("ShaderNodeMapRange"); w.clamp = True        # radius_factor 0.5-1 -> FRAC-1
    w.inputs["From Min"].default_value, w.inputs["From Max"].default_value = 0.5, 1.0
    w.inputs["To Min"].default_value,   w.inputs["To Max"].default_value   = DEBRIS_EMIT_FRAC, 1.0
    hot, glow = math_node(N, "MULTIPLY", 1.0), math_node(N, "MULTIPLY", 1.0)
    animate(glow.inputs[1], EMIT_GAIN)
    L.new(rf.outputs["Fac"], w.inputs["Value"]); L.new(w.outputs["Result"], hot.inputs[0])
    L.new(hl.outputs["Glow"], hot.inputs[1]);    L.new(hot.outputs[0], glow.inputs[0])
    L.new(glow.outputs[0], bsdf.inputs["Emission Strength"])
    L.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return finish_particle_material(mat)


def finish_particle_material(mat):
    for attr, val in (("surface_render_method", "DITHERED"), ("blend_method", "HASHED")):
        try: setattr(mat, attr, val)
        except Exception: pass
    return mat


def body_colour(N, L):
    """Colour output from BODY_COLOURS, chosen by the 'iron' and 'impactor' attributes."""
    attr = {}
    for name in ("iron", "impactor"):
        attr[name] = N.new("ShaderNodeAttribute"); attr[name].attribute_name = name

    def mix(fac, a, b):
        m = N.new("ShaderNodeMix"); m.data_type = "RGBA"
        L.new(fac, sock(m.inputs, "Factor"))
        for name, v in (("A", a), ("B", b)):
            if isinstance(v, tuple): sock(m.inputs, name).default_value = (*linear(v), 1)
            else:                    L.new(v, sock(m.inputs, name))
        return sock(m.outputs, "Result")
    c = BODY_COLOURS
    target   = mix(attr["iron"].outputs["Fac"], c["target rock"],   c["target iron"])
    impactor = mix(attr["iron"].outputs["Fac"], c["impactor rock"], c["impactor iron"])
    return mix(attr["impactor"].outputs["Fac"], target, impactor)


def mist_material():
    """Volume: rock-coloured dust, lit, glowing as the body does; opacity and
    glow fade in with density (MIST_FADE_RHO)."""
    mat, N, L = new_material("MistLook")
    out, pv = N.new("ShaderNodeOutputMaterial"), N.new("ShaderNodeVolumePrincipled")
    pv.inputs["Density Attribute"].default_value   = ""      # density is set by the fade
    pv.inputs["Blackbody Intensity"].default_value = 0.0
    hl, dens = heat_node(N, L, iron=False), N.new("ShaderNodeAttribute")
    dens.attribute_name = "rho"
    safe, logr = math_node(N, "MAXIMUM", 1e-6), math_node(N, "LOGARITHM", 10.0)
    fade = N.new("ShaderNodeMapRange"); fade.interpolation_type = "SMOOTHSTEP"
    fade.inputs["From Min"].default_value, fade.inputs["From Max"].default_value = map(math.log10, MIST_FADE_RHO)
    L.new(dens.outputs["Fac"], safe.inputs[0]); L.new(safe.outputs[0], logr.inputs[0])
    L.new(logr.outputs[0], fade.inputs["Value"])

    opacity, glow = math_node(N, "MULTIPLY", MIST_DENSITY), math_node(N, "MULTIPLY", 1.0)
    animate(glow.inputs[1], MIST_EMIT_GAIN)
    hot = math_node(N, "MULTIPLY", 1.0)                              # glow x fade
    L.new(fade.outputs["Result"], opacity.inputs[0]); L.new(opacity.outputs[0], pv.inputs["Density"])
    L.new(hl.outputs["Glow"], hot.inputs[0]);         L.new(fade.outputs["Result"], hot.inputs[1])
    L.new(hot.outputs[0], glow.inputs[0]);            L.new(glow.outputs[0], pv.inputs["Emission Strength"])
    L.new(hl.outputs["Colour"], pv.inputs["Color"])
    L.new(hl.outputs["Glow Colour"], pv.inputs["Emission Color"])
    L.new(pv.outputs["Volume"], out.inputs["Volume"])
    return mat


def particle_nodes(mat):
    """Geometry Nodes: each vertex -> sphere of PARTICLE_RADIUS x radius_factor."""
    ng = bpy.data.node_groups.new("PtsGN", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT",  socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    N, L = ng.nodes, ng.links
    gin, gout, m2p = N.new("NodeGroupInput"), N.new("NodeGroupOutput"), N.new("GeometryNodeMeshToPoints")
    rf = N.new("GeometryNodeInputNamedAttribute"); rf.data_type = "FLOAT"
    rf.inputs["Name"].default_value = "radius_factor"
    r  = N.new("ShaderNodeMath"); r.operation = "MULTIPLY"; r.inputs[1].default_value = PARTICLE_RADIUS
    sm = N.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = mat
    L.new(gin.outputs[0], m2p.inputs["Mesh"]); L.new(rf.outputs["Attribute"], r.inputs[0])
    L.new(r.outputs[0], m2p.inputs["Radius"])
    pts = m2p.outputs["Points"]
    if CUTAWAY:                       # delete points with (position - centre) . normal > 0
        cen = N.new("GeometryNodeObjectInfo"); cen.inputs["Object"].default_value = cut_centre
        centre = cen.outputs["Location"]  # the CutCentre empty: the target's centre this frame
        pos, rel = N.new("GeometryNodeInputPosition"), N.new("ShaderNodeVectorMath")
        rel.operation = "SUBTRACT"
        L.new(pos.outputs["Position"], rel.inputs[0]); L.new(centre, rel.inputs[1])
        if CUTAWAY == "CAMERA":       # normal = camera position - centre
            oi = N.new("GeometryNodeObjectInfo"); oi.name = "CutCamera"
            nrm = N.new("ShaderNodeVectorMath"); nrm.operation = "SUBTRACT"
            L.new(oi.outputs["Location"], nrm.inputs[0]); L.new(centre, nrm.inputs[1])
            normal = nrm.outputs["Vector"]
        else:
            c = N.new("FunctionNodeInputVector"); c.vector = CUTAWAY
            normal = c.outputs["Vector"]
        dot = N.new("ShaderNodeVectorMath"); dot.operation = "DOT_PRODUCT"
        L.new(rel.outputs["Vector"], dot.inputs[0]); L.new(normal, dot.inputs[1])
        near = math_node(N, "GREATER_THAN", 0.0); L.new(dot.outputs["Value"], near.inputs[0])
        cut = N.new("GeometryNodeDeleteGeometry"); cut.domain = "POINT"
        L.new(pts, cut.inputs["Geometry"]); L.new(near.outputs[0], cut.inputs["Selection"])
        pts = cut.outputs["Geometry"]
    L.new(pts, sm.inputs["Geometry"])
    L.new(sm.outputs["Geometry"], gout.inputs[0])
    return ng


# ===========================================================================
# SCENE
# ===========================================================================
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.frame_start, scene.frame_end = 1, NFRAMES

heat_ng = heat_look()
kinds = {k for _, k, _, _ in COMPONENTS}
if "mesh" in kinds:
    body_material()
if "mist" in kinds:
    mist_mat = mist_material()
cut_centre = bpy.data.objects.new("CutCentre", None)   # an empty at the target's centre;
scene.collection.objects.link(cut_centre)              # Stage 3 moves it for each frame
pts_group = particle_nodes(particle_material())

# Hidden one-vertex object carrying the particle Geometry Nodes, for Stage 3
_tm = bpy.data.meshes.new("PointsTemplateMesh"); _tm.from_pydata([(0, 0, 0)], [], [])
template_ob = bpy.data.objects.new("PointsTemplate", _tm)
scene.collection.objects.link(template_ob)
template_ob.modifiers.new("Pts", "NODES").node_group = pts_group
template_ob.hide_viewport = template_ob.hide_render = True
for blk in list(bpy.data.materials) + [pts_group, heat_ng]:
    blk.use_fake_user = True                     # survive Stage 3's orphan purges


def empty_ply(path):
    """True for a PLY with no vertices (e.g. no rest particles yet, early on),
    which Blender's importer refuses; such a component is simply not drawn."""
    with open(path, "rb") as f:
        return b"\nelement vertex 0\n" in f.read(1024)


def ply_centre(path):
    """The target centre Stage 1 records in a particle file's header, or the origin."""
    with open(path, "rb") as f:
        m = re.search(rb"\ncomment target_centre (\S+) (\S+) (\S+)\n", f.read(1024))
    return tuple(map(float, m.groups())) if m else (0.0, 0.0, 0.0)


def import_ply(path, kind):
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.wm.ply_import(filepath=path)
    ob = bpy.context.selected_objects[0]
    if kind == "mesh":
        for poly in ob.data.polygons: poly.use_smooth = True
        ob.data.materials.clear(); ob.data.materials.append(bpy.data.materials["BodyLook"])
    else:
        ob.modifiers.new("Pts", "NODES").node_group = pts_group
    return ob


def keyframe_only_on(ob, frame):
    """Visible on its own Blender frame only (constant interpolation)."""
    for f, hide in [(max(1, frame - 1), True), (frame, False), (frame + 1, True)]:
        ob.hide_viewport = ob.hide_render = hide
        ob.keyframe_insert("hide_viewport", frame=f); ob.keyframe_insert("hide_render", frame=f)
    hold_keys(ob)


def hold_keys(ob):
    """Make all of ob's keyframes constant (no easing between frames)."""
    act = ob.animation_data.action
    curves = list(getattr(act, "fcurves", []) or [])
    for layer in getattr(act, "layers", []):              # Blender 4.4+ layered actions
        for strip in layer.strips:
            for bag in strip.channelbags:
                curves += list(bag.fcurves)
    for fc in curves:
        for kp in fc.keyframe_points:
            kp.interpolation = "CONSTANT"


# Test geometry (one object per frame and component; one shared mist object)
mist_ob = None
for i in render_frames:
    for role, kind, sub, pat in ACTIVE:
        path = os.path.join(RUN_DIR, sub, pat.format(i))
        if not os.path.exists(path):
            print(f"  missing: {path}"); continue
        if kind == "mist":
            if mist_ob is None:
                bpy.ops.object.volume_import(filepath=path, align="WORLD")
                mist_ob = bpy.context.active_object; mist_ob.name = "Mist"
                mist_ob.location = (0, 0, 0)
                mist_ob.data.materials.clear(); mist_ob.data.materials.append(mist_mat)
            continue
        if empty_ply(path):
            continue
        ob = import_ply(path, kind)
        ob.name = f"{role}_{i:04d}"
        keyframe_only_on(ob, i + 1)
        if kind == "particles" and role == "body":      # the cutaway follows the target
            cut_centre.location = ply_centre(path)
            cut_centre.keyframe_insert("location", frame=i + 1); hold_keys(cut_centre)


# ===========================================================================
# LIGHTS AND CAMERA
# ===========================================================================
world = bpy.data.worlds.new("Space"); scene.world = world; world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0, 0, 0, 1)
# Flat cameras: two suns shining from above and to the camera's side of PATH;
# they turn with CAMERA_VIEW's turn, so the lighting seen from the camera stays
# the same.  Panoramas (dome, headset) see bodies in widely different directions,
# where fixed suns would light them from different sides; there a lamp at the
# camera lights everything from the viewer's direction instead.
PANORAMA = CAMERA_MODE in ("FISHEYE", "VR360")
LAMP_STRENGTH = 3.0          # camera lamp: light on the bodies, in sun units (key sun = 4)
turn = Matrix.Rotation(math.radians(CAMERA_VIEW[1]), 3, "Z")
for name, energy, rot in () if PANORAMA else (("KeySun", 4.0, (0.6, 0.0, 0.8)),
                                               ("FillSun", 1.2, (-0.5, 0.0, -2.2))):
    sun = bpy.data.objects.new(name, bpy.data.lights.new(name, "SUN"))
    sun.data.energy, sun.rotation_euler = energy, (turn @ Euler(rot).to_matrix()).to_euler()
    scene.collection.objects.link(sun)

centre = Vector((0, 0, 0))                         # each frame's centre of mass
cam_data = bpy.data.cameras.new("Cam")
cam = bpy.data.objects.new("Cam", cam_data)
scene.collection.objects.link(cam); scene.camera = cam
lamp = None
if PANORAMA:                                   # moves with the camera
    lamp = bpy.data.objects.new("CameraLamp", bpy.data.lights.new("CameraLamp", "POINT"))
    lamp.data.shadow_soft_size = 0.0
    lamp.parent = cam
    scene.collection.objects.link(lamp)
if "CutCamera" in pts_group.nodes:                  # the cutaway plane faces this camera
    pts_group.nodes["CutCamera"].inputs["Object"].default_value = cam


def view_offset(off):
    """A PATH position adjusted by CAMERA_VIEW: scaled, turned about z, raised."""
    scale, turn, elev = CAMERA_VIEW
    o = Vector(off) * scale
    r, az = o.length, math.atan2(o.y, o.x) + math.radians(turn)
    el = math.asin(o.z / r) if elev is None else math.radians(elev)
    return Vector((r * math.cos(el) * math.cos(az), r * math.cos(el) * math.sin(az), r * math.sin(el)))


def aim(offset):
    cam.location = centre + offset
    cam.rotation_euler = (centre - cam.location).to_track_quat("-Z", "Y").to_euler()
    if CAMERA_MODE == "FISHEYE":
        cam.rotation_euler.rotate_axis("X", math.radians(DOME_TILT_DEG))
        cam.rotation_euler.rotate_axis("Y", math.radians(DOME_AZ_DEG))


# Camera path: (Blender frame, camera position relative to the centre, in Blender units);
# Blender eases smoothly between them.  Ortho zooms through ORTHO_SCALES (width
# of view) instead.  Short test sets get the first view.
PATH = [(1,       (3.4, -26.9,  4.5)),
        (150,     (2.4, -19.2,  3.2)),
        (350,     (3.0, -22.5,  4.3)),
        (400,     (3.7, -26.0,  5.5)),
        (500,     (4.0, -28.0,  6.0)),
        (750,     (4.3, -28.8,  6.7)),
        (1000,    (5.5, -35.2,  8.8)),
        (1200,    (6.6, -42.0, 10.5)),
        (NFRAMES, (7.6, -48.0, 12.0))]
ORTHO_SCALES = [(1, 10.9), (150, 6.2), (300, 11.8), (600, 17.9), (1000, 10.0), (1200, 25.7), (1800, 45.7), (NFRAMES, 32.3)]
move_camera = NFRAMES > PATH[-2][0]
cam_data.clip_end   = 100 * CAMERA_VIEW[0] * max(Vector(o).length for _, o in PATH)
cam_data.clip_start = 1e-5 * cam_data.clip_end      # works whatever the length unit

if CAMERA_MODE == "ORTHO":
    cam_data.type = "ORTHO"
    cam.location, cam.rotation_euler = centre + Vector((0, 0, 100)), (0, 0, 0)
    for frm, s in (ORTHO_SCALES if move_camera else ORTHO_SCALES[:1]):
        cam_data.ortho_scale = s
        cam_data.keyframe_insert("ortho_scale", frame=frm)
else:
    if CAMERA_MODE == "PERSP":
        cam_data.type, cam_data.lens = "PERSP", 85
    elif CAMERA_MODE == "FISHEYE":
        cam_data.type, cam_data.panorama_type = "PANO", "FISHEYE_EQUIDISTANT"
        cam_data.fisheye_fov = math.pi
    else:
        cam_data.type, cam_data.panorama_type = "PANO", "EQUIRECTANGULAR"
    for frm, off in (PATH if move_camera else PATH[:1]):
        off = view_offset(off)
        aim(off)
        cam.keyframe_insert("location", frame=frm); cam.keyframe_insert("rotation_euler", frame=frm)
        if lamp:              # same light on the centre however far away the camera is
            lamp.data.energy = LAMP_STRENGTH * 4 * math.pi * off.length ** 2
            lamp.data.keyframe_insert("energy", frame=frm)
        if STEREO:            # keep the 3D depth the same as the camera pulls back
            st = cam_data.stereo; st.convergence_distance = off.length
            st.interocular_distance = STEREO_DEPTH * st.convergence_distance
            st.keyframe_insert("convergence_distance", frame=frm)
            st.keyframe_insert("interocular_distance", frame=frm)

if STEREO:
    scene.render.use_multiview, scene.render.views_format = True, "STEREO_3D"
    st = cam_data.stereo
    st.convergence_mode = "OFFAXIS"               # objects at the centre sit at screen depth
    st.use_spherical_stereo = cam_data.type == "PANO"
    st.use_pole_merge = CAMERA_MODE == "VR360"    # avoid eye strain looking straight up/down
    im = scene.render.image_settings
    im.views_format = "STEREO_3D"
    im.stereo_3d_format.display_mode = "TOPBOTTOM" if CAMERA_MODE == "VR360" else "SIDEBYSIDE"
    im.stereo_3d_format.use_sidebyside_crosseyed = False
    im.stereo_3d_format.use_squeezed_frame = False  # full resolution for each eye


# ===========================================================================
# RENDER SETTINGS AND TEST RENDERS
# ===========================================================================
if CAMERA_MODE in ("FISHEYE", "VR360"):     # panoramas need Cycles: GPU, few samples
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = 4096
    scene.render.resolution_y = 4096 if CAMERA_MODE == "FISHEYE" else 2048
    scene.cycles.samples, scene.cycles.use_denoising = 16, True
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for dev_type in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = dev_type; prefs.get_devices(); break
        except Exception:
            pass
    for dev in prefs.devices:
        dev.use = dev.type == prefs.compute_device_type
    if any(d.use for d in prefs.devices):
        scene.cycles.device = "GPU"
    print(f"Test renders: Cycles on {scene.cycles.device}, 16 samples", flush=True)
else:
    for eng in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        try: scene.render.engine = eng; break
        except Exception: pass
    if CAMERA_MODE == "PERSP":
        scene.render.resolution_x, scene.render.resolution_y = 1280, 720
    else:
        scene.render.resolution_x, scene.render.resolution_y = 1080, 1080
scene.render.fps = 12
scene.render.image_settings.file_format = "PNG"
scene.view_settings.view_transform, scene.view_settings.look = "Standard", "None"

if TIME_LABEL:                         # Blender's "stamp": text burned into the image
    r = scene.render
    for p in r.bl_rna.properties:
        if p.identifier.startswith("use_stamp_"):
            setattr(r, p.identifier, False)
    r.use_stamp, r.use_stamp_note = True, True
    r.stamp_font_size = TIME_LABEL_SIZE
    r.stamp_foreground, r.stamp_background = (1, 1, 1, 1), (0, 0, 0, 0)

test_dir = os.path.join(RUN_DIR, "renders")
os.makedirs(test_dir, exist_ok=True)
for fr in render_frames:
    vdb = os.path.join(VDB_DIR, f"mist_{fr:04d}.vdb")
    if mist_ob is not None:
        mist_ob.hide_render = not os.path.exists(vdb)
        if os.path.exists(vdb):
            mist_ob.data.filepath = vdb; mist_ob.data.grids.load()
    scene.frame_set(fr + 1)                       # Blender frame = snapshot + 1
    scene.render.filepath = os.path.join(test_dir, f"frame_{fr:04d}.png")
    if TIME_LABEL:
        scene.render.stamp_note_text = time_label(fr)
    bpy.ops.render.render(write_still=True)


# ===========================================================================
# SAVE
# ===========================================================================
if MAKE_TEMPLATE:                                  # Stage 3 imports each frame itself
    for ob in list(bpy.data.objects):
        if ob.name.startswith(("body_", "rest_")) or ob.type == "VOLUME":
            bpy.data.objects.remove(ob, do_unlink=True)
    cut_centre.animation_data_clear(); cut_centre.location = (0, 0, 0)

scene["RENDER_MODE"] = RENDER_MODE
scene["RUN_DIR"]     = RUN_DIR
scene["IS_TEMPLATE"] = bool(MAKE_TEMPLATE)
scene["COMPONENTS"]  = json.dumps(COMPONENTS)
scene["TIME_LABEL"]  = TIME_LABEL or ""          # Stage 3 sets each frame's label
scene["FRAME_TIMES_S"] = json.dumps(FRAME_TIMES_S)
bpy.ops.wm.save_as_mainfile(filepath=BLEND_OUT)
print(f"\nSaved {BLEND_OUT}")
