"""
combined_01_make.py  –  Stage 1: SWIFT snapshots → Blender-ready files.

Every particle is either BODY (rho >= SURFACE_RHO, inside the box, not a lone
speck) or REST (everything else), so all the mass is shown exactly once.
RENDER_MODE is <body style>_<rest style>:

  RENDER_MODE       body                         rest
  PARTICLES         particles/body_NNNN.ply      particles/rest_NNNN.ply
  MESH_PARTICLES    meshes/body_NNNN.ply         particles/rest_NNNN.ply
  MESH_MIST         meshes/body_NNNN.ply         volume/volume_NNNN.npz
  PARTICLES_MIST    particles/body_NNNN.ply      volume/volume_NNNN.npz

Each output
depends only on its own style, so e.g. MESH_PARTICLES reuses the meshes of a
MESH_MIST run, and PARTICLES_MIST its volume files.

Up-to-date files are never remade.  Every output carries a fingerprint of
the settings it was made with (a PLY header comment, or an entry in the
.npz); an output is remade only if it is missing or those settings have
changed, so nothing needs switching on or off between runs.

The files carry physics, not colours: each point or mesh vertex has its
temperature (K) and iron fraction, and the volume files hold density
(kg m^-3) and temperature grids.  All colour and glow is decided in Stage 2,
so the look can be changed without rerunning this stage.

Meshes are made with splashsurf (pip install pysplashsurf; Python >= 3.10),
volumes with swiftsimio's render_gas.

Run
---
    python3 combined_01_make.py                   # TEST_FRAMES
    python3 combined_01_make.py 0 215             # frames 0-215 (START END)
    python3 combined_01_make.py 300-319 1600      # ranges and single frames
    python3 combined_01_make.py 0-2160/100        # every 100th frame
    python3 combined_01_make.py all               # every frame

With START END, the IC frame (frame 0) is done only by the chunk starting at
0 or 1.  Set RAYON_NUM_THREADS to limit splashsurf's threads (stage1_slurm.sh
does this); on a login node use e.g.  env RAYON_NUM_THREADS=4 python3 ...
"""

import warnings; warnings.filterwarnings("ignore")
import os, sys, time, glob, re, traceback, json, hashlib
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import h5py


# ===========================================================================
# PATHS
# ===========================================================================
SIM_NAME = "pE65_Th35_b60v10_3e6"
BASE_DIR = "/cosma5/data/dp004/vreke/swift/rssse"
IC_FILE  = os.path.join(BASE_DIR, "init_cond", f"init_{SIM_NAME}.hdf5")
RUN_DIR  = os.path.join(BASE_DIR, SIM_NAME)
DATA_DIR = os.path.join(RUN_DIR, "snapshots")
FILE_FMT = f"{SIM_NAME}_" + "{}.hdf5"

PARTICLES_DIR = os.path.join(RUN_DIR, "particles")
MESH_DIR      = os.path.join(RUN_DIR, "meshes")
VOL_DIR       = os.path.join(RUN_DIR, "volume")


# ===========================================================================
# WHAT TO MAKE
# ===========================================================================
# "PARTICLES" | "MESH_PARTICLES" | "MESH_MIST" | "PARTICLES_MIST"
RENDER_MODE = "PARTICLES"

TEST_FRAMES = [0, 200, 400, 600, 800, 1600, 2160]   # used when no frames are given

# Bump when a change to this code alters the output files, so that they are
# all remade; settings changes are detected automatically.
FORMAT_VERSION = 3


# ===========================================================================
# BOX AND UNITS
# One Blender unit is the simulation's own length unit (read from the
# snapshots), so lengths in Blender read as they do in the simulation, with
# the centre of mass of each frame at the origin.  Only material inside the
# box (side BOX_SIZE) is drawn: choose it to hold everything you want to see
# over the whole run.  Each frame reports how much dense material falls
# outside, as a guide.
# ===========================================================================
BLENDER_UNIT_M = None      # metres per Blender unit; None = the snapshots' length unit
BOX_SIZE       = 35.0      # box side, in Blender units
UNIT_M         = None      # (set at start-up from BLENDER_UNIT_M or the snapshots)


# ===========================================================================
# BODY / REST SPLIT  (SI, kg m^-3)
# ===========================================================================
SURFACE_RHO    = 1000.0  # body = particles at least this dense ...
MIN_NEIGHBOURS = 12       # ... with at least this many body neighbours within
                         #   2 particle radii (drops lone dense specks; 0 = off)


# ===========================================================================
# PARTICLES  (body and rest particle files)
# ===========================================================================
RHO_DEBRIS   = 1000.0          # radius_factor ramps from 0.5 at RHO_DEBRIS ...
RHO_BODY     = 2000.0          # ... to 1.0 at RHO_BODY (Stage 2: size, debris glow)
IRON_ID      = 402             # MaterialID of iron; others are rock
TARGET_MAX_ID = None           # particle IDs 1... - these are the target, higher ones the
                               #  impactor; None = find it from the two bodies in the IC file
REST_FADE_RHO = (5.0, 20.0)    # rest particles fade in (alpha 0 -> 1) between
                               #  these densities; thinner ones are left out


# ===========================================================================
# MESH  (splashsurf)
# The particle radius is that of a sphere holding one particle at SURFACE_RHO,
# so it sets the finest detail.  splashsurf's iso level is a "fluid fraction"
# (~1 inside any group of particles), so the body is chosen by density first.
# ===========================================================================
SPLASH_ISO            = 0.6    # fluid-fraction iso level (0.6 is standard)
SPLASH_SMOOTHING_LEN  = 2.0    # kernel support, in particle radii
SPLASH_CUBE_SIZE      = 0.5    # marching-cubes cell, in particle radii
                               #   (smaller = finer, ~1/CUBE_SIZE^2 more triangles)
SPLASH_SMOOTH_ITERS   = 10     # weighted Laplacian smoothing passes (0 = off)
SPLASH_NORMALS_ITERS  = 10     # smoothing passes on the vertex normals


# ===========================================================================
# MIST  (volume grids)
# ===========================================================================
RESOLUTION    = 256     # grid cells per side (a multiple of Stage 2's VDB_RES)
MIST_MAX_T_K  = 60000.0 # temperatures are stored as float16, so capped here


# ===========================================================================
# MODES
# ===========================================================================
MODES   = {"PARTICLES":      ("PARTICLES", "PARTICLES"),
           "MESH_PARTICLES": ("MESH",      "PARTICLES"),
           "MESH_MIST":      ("MESH",      "MIST"),
           "PARTICLES_MIST": ("PARTICLES", "MIST")}
if RENDER_MODE not in MODES:
    raise SystemExit(f"Unknown RENDER_MODE {RENDER_MODE!r}; choose from {list(MODES)}")
BODY_STYLE, REST_STYLE = MODES[RENDER_MODE]


def output_files(frame):
    """(body file, rest file) for this frame."""
    f = f"{frame:04d}"
    body = os.path.join(MESH_DIR if BODY_STYLE == "MESH" else PARTICLES_DIR, f"body_{f}.ply")
    rest = (os.path.join(VOL_DIR, f"volume_{f}.npz") if REST_STYLE == "MIST"
            else os.path.join(PARTICLES_DIR, f"rest_{f}.ply"))
    return body, rest


# ===========================================================================
# FINGERPRINTS  (which settings each kind of output depends on)
# ===========================================================================
def fingerprints():
    """Fingerprints of the body and rest outputs made with the current settings."""
    split = dict(v=FORMAT_VERSION, box=[BOX_SIZE, UNIT_M],
                 rho=SURFACE_RHO, neighbours=MIN_NEIGHBOURS, iron=IRON_ID)
    size  = [RHO_DEBRIS, RHO_BODY]
    body  = (dict(splash=[SPLASH_ISO, SPLASH_SMOOTHING_LEN, SPLASH_CUBE_SIZE,
                          SPLASH_SMOOTH_ITERS, SPLASH_NORMALS_ITERS])
             if BODY_STYLE == "MESH" else dict(size=size, target=TARGET_MAX_ID, p=2))
    rest  = (dict(res=RESOLUTION, t_max=MIST_MAX_T_K) if REST_STYLE == "MIST"
             else dict(size=size, fade=REST_FADE_RHO, target=TARGET_MAX_ID, p=2))
    h = lambda *x: hashlib.sha1(json.dumps(x, sort_keys=True).encode()).hexdigest()[:16]
    return h(BODY_STYLE, split, body), h(REST_STYLE, split, rest)


def file_fingerprint(path):
    """The fingerprint stored in an output file; None if missing, unreadable or unstamped."""
    try:
        if path.endswith(".npz"):
            with np.load(path) as z:
                return str(z["fingerprint"]) if "fingerprint" in z.files else None
        with open(path, "rb") as f:
            m = re.search(rb"\ncomment fingerprint (\w+)\n", f.read(1024))
        return m and m.group(1).decode()
    except Exception:
        return None


def save(path, write):
    """Write via a temporary file, so an interrupted job never leaves a partial file."""
    with open(path + ".part", "wb") as f:
        write(f)
    os.replace(path + ".part", path)


# ===========================================================================
# SNAPSHOTS AND REFERENCE PROPERTIES
# ===========================================================================
_files         = glob.glob(os.path.join(DATA_DIR, f"{SIM_NAME}_*.hdf5"))
_rx            = re.compile(rf"{re.escape(SIM_NAME)}_(\d+)\.hdf5$")
SNAPSHOT_TIMES = sorted(int(_rx.search(os.path.basename(f)).group(1))
                        for f in _files if _rx.search(os.path.basename(f)))
FRAME_TO_TIME  = {i + 1: t for i, t in enumerate(SNAPSHOT_TIMES)}

REF_IDS = REF_MASSES = REF_MAT_IDS = None


def snapshot_path(frame):
    return IC_FILE if frame == 0 else os.path.join(DATA_DIR, FILE_FMT.format(FRAME_TO_TIME[frame]))


def load_reference_properties():
    """Particle IDs, SI masses and material IDs from the IC file."""
    global REF_IDS, REF_MASSES, REF_MAT_IDS
    if not os.path.exists(IC_FILE):
        raise SystemExit(f"\nCannot find IC file:\n  {IC_FILE}\n")
    with h5py.File(IC_FILE, "r") as f:
        m_si    = f["Units"].attrs["Unit mass in cgs (U_M)"][0] / 1000.0
        ids     = f["PartType0/ParticleIDs"][:]
        masses  = f["PartType0/Masses"][:] * m_si
        mat_ids = f["PartType0/MaterialIDs"][:]
        xyz     = f["PartType0/Coordinates"][:]
    order = np.argsort(ids)
    REF_IDS, REF_MASSES, REF_MAT_IDS = ids[order], masses[order], mat_ids[order]
    pops = ", ".join(f"ID {m}: {n:,}" for m, n in zip(*np.unique(mat_ids, return_counts=True)))
    print(f"Reference particles from {IC_FILE}\n  {len(ids):,} particles ({pops})")
    global TARGET_MAX_ID
    if TARGET_MAX_ID is None:
        TARGET_MAX_ID = find_target_max_id(ids, xyz)
    n_t = int((ids <= TARGET_MAX_ID).sum())
    print(f"  target: IDs up to {TARGET_MAX_ID} ({n_t:,} particles); impactor: {len(ids) - n_t:,} particles")


def find_target_max_id(ids, xyz):
    """Split the two separate bodies of the IC file, and check that the target
    (the body holding the lowest ID) has the contiguous block of lowest IDs."""
    from scipy.cluster.vq import kmeans2
    _, label = kmeans2(xyz.astype(np.float64), 2, minit="++", seed=1)
    target = label == label[np.argmin(ids)]
    if ids[target].max() >= ids[~target].min():
        raise SystemExit("The target's particle IDs are not the lowest block: set TARGET_MAX_ID.")
    return int(ids[target].max())


def mapped_properties(ids):
    """IC masses (SI kg) and material IDs for the given particle IDs."""
    idx   = np.clip(np.searchsorted(REF_IDS, ids), 0, len(REF_IDS) - 1)
    valid = REF_IDS[idx] == ids
    return np.where(valid, REF_MASSES[idx], 0.0), np.where(valid, REF_MAT_IDS[idx], -1)


def centre_of_mass(xyz, m):
    return np.average(xyz, axis=0, weights=m) if m.sum() > 0 else xyz.mean(0)


def read_snapshot(path):
    """SI quantities, and Blender coordinates (centre of mass at the origin)."""
    with h5py.File(path, "r") as f:
        u   = f["Units"].attrs
        U_M, U_L, U_t = (u[k][0] for k in ("Unit mass in cgs (U_M)",
                         "Unit length in cgs (U_L)", "Unit time in cgs (U_t)"))
        xyz = f["PartType0/Coordinates"][:]
        ids = f["PartType0/ParticleIDs"][:]
        rho = f["PartType0/Densities"][:] * (U_M / U_L**3) * 1e3            # kg m^-3
        TK  = f["PartType0/InternalEnergies"][:] * (U_L / U_t)**2 * 1e-7    # K (c_v = 1000 J/kg/K)
    masses, mat_ids = mapped_properties(ids)
    com   = centre_of_mass(xyz, masses)
    world = (xyz - com) * (U_L / 100.0) / UNIT_M
    return dict(world=world, com=com, masses=masses, mat_ids=mat_ids, rho=rho, TK=TK,
                iron=(mat_ids == IRON_ID).astype(np.float64),
                impactor=(ids > TARGET_MAX_ID).astype(np.float64),
                inside=np.all(np.abs(world) <= BOX_SIZE / 2, axis=1),
                u_l_m=U_L / 100.0)


def blender_unit():
    """Metres per Blender unit, and a line describing the units and box."""
    with h5py.File(snapshot_path(min(FRAME_TO_TIME)), "r") as f:
        code_m = f["Units"].attrs["Unit length in cgs (U_L)"][0] / 100.0
        sim    = f["Header"].attrs.get("BoxSize") if "Header" in f else None
    unit = BLENDER_UNIT_M or code_m
    note = f"1 Blender unit = {unit:.4g} m = {unit / 6.371e6:.4g} R_E"
    if sim is not None:
        note += f"; simulation domain {np.atleast_1d(sim)[0] * code_m / unit:.4g} Blender units across"
    return unit, note


# ===========================================================================
# HELPERS
# ===========================================================================
def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ===========================================================================
# BODY SELECTION
# ===========================================================================
def particle_radius_world(snap):
    """Radius of a sphere holding one (median-mass) particle at SURFACE_RHO, in world units."""
    m   = np.median(snap["masses"][snap["masses"] > 0])
    r_m = (3.0 * m / (4.0 * np.pi * SURFACE_RHO)) ** (1.0 / 3.0)
    return r_m / UNIT_M, r_m


def body_mask(snap, r_w):
    """Inside the box, rho >= SURFACE_RHO, and with >= MIN_NEIGHBOURS body neighbours."""
    body = snap["inside"] & (snap["rho"] >= SURFACE_RHO)
    if MIN_NEIGHBOURS > 0 and body.any():
        from scipy.spatial import cKDTree
        pts  = snap["world"][body]
        nthr = int(os.environ.get("RAYON_NUM_THREADS", -1))
        d, _ = cKDTree(pts).query(pts, k=MIN_NEIGHBOURS + 1,
                                  distance_upper_bound=2.0 * r_w, workers=nthr)
        body[np.flatnonzero(body)[~np.isfinite(d[:, -1])]] = False
    return body


# ===========================================================================
# WRITERS
# ===========================================================================
def target_centre(snap):
    """Centre of mass of the target's dense material (all of it, if none is dense)."""
    t = snap["impactor"] == 0
    d = t & (snap["rho"] >= SURFACE_RHO)
    s = d if d.any() else t
    return centre_of_mass(snap["world"][s], snap["masses"][s])


def write_particles(path, snap, sel, alpha, fp):
    """Binary PLY point cloud: x y z, T_K (temperature), iron and impactor (0 or 1),
    opacity, radius_factor (not 'alpha', which Blender's importer takes as part of a
    colour).  The header also records the target's centre, for a cutaway in Stage 2."""
    names = ("x", "y", "z", "T_K", "iron", "impactor", "opacity", "radius_factor")
    arr = np.empty(sel.sum(), dtype=[(n, "<f4") for n in names])
    for k, v in zip("xyz", snap["world"][sel].T): arr[k] = v
    arr["T_K"], arr["iron"], arr["impactor"] = snap["TK"][sel], snap["iron"][sel], snap["impactor"][sel]
    arr["opacity"]       = np.clip(alpha, 0, 1)
    arr["radius_factor"] = 0.5 * (1.0 + smoothstep(RHO_DEBRIS, RHO_BODY, snap["rho"][sel]))
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"comment fingerprint {fp}\n"
              "comment target_centre {:.5g} {:.5g} {:.5g}\n".format(*target_centre(snap))
              + f"element vertex {len(arr)}\n"
              + "".join(f"property float {n}\n" for n in names) + "end_header\n")
    save(path, lambda f: (f.write(header.encode("ascii")), arr.tofile(f)))
    print(f"  → {path}  ({len(arr):,} particles)")


def vertex_normals(V, F):
    """Area-weighted vertex normals from the triangles."""
    V   = V.astype(np.float64)
    fn  = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    acc = np.zeros_like(V)
    for k in range(3):
        np.add.at(acc, F[:, k], fn)
    L = np.linalg.norm(acc, axis=1, keepdims=True)
    return np.where(L > 0, acc / np.maximum(L, 1e-30), [0.0, 0.0, 1.0]).astype(np.float32)


def write_mesh(path, snap, body, r_w, fp):
    """Surface of the body particles with splashsurf; binary PLY with normals,
    temperature (K) and iron fraction interpolated from the particles."""
    import pysplashsurf
    t0 = time.time()
    mwd, _ = pysplashsurf.reconstruction_pipeline(
        np.ascontiguousarray(snap["world"][body], dtype=np.float64),
        attributes_to_interpolate={"temperature": np.ascontiguousarray(snap["TK"][body]),
                                   "iron":        np.ascontiguousarray(snap["iron"][body])},
        particle_radius=r_w, rest_density=1000.0,   # rest_density is irrelevant here
        smoothing_length=SPLASH_SMOOTHING_LEN, cube_size=SPLASH_CUBE_SIZE,
        iso_surface_threshold=SPLASH_ISO, mesh_cleanup=True, mesh_smoothing_weights=True,
        mesh_smoothing_iters=SPLASH_SMOOTH_ITERS or None,
        normals_smoothing_iters=SPLASH_NORMALS_ITERS or None, compute_normals=True)
    V  = np.asarray(mwd.mesh.vertices,  dtype=np.float32)
    F  = np.asarray(mwd.mesh.triangles, dtype=np.int32)
    if len(F) == 0:
        print("  [no surface] splashsurf produced no triangles"); return
    pa = mwd.point_attributes
    N  = np.asarray(pa["normals"], dtype=np.float32) if "normals" in pa else vertex_normals(V, F)
    bad = ~np.isfinite(N).all(axis=1)              # splashsurf leaves ~0.5% NaN normals
    if bad.any():
        N[bad] = vertex_normals(V, F)[bad]
    TK_v = np.asarray(pa["temperature"])

    names = ("x", "y", "z", "nx", "ny", "nz", "T_K", "iron")
    va = np.empty(len(V), dtype=[(n, "<f4") for n in names])
    for k, v in zip(names, np.hstack([V, N]).T): va[k] = v
    va["T_K"], va["iron"] = TK_v, np.clip(np.asarray(pa["iron"]), 0, 1)
    fa = np.empty(len(F), dtype=[("n","u1"), ("v","<i4",(3,))]); fa["n"] = 3; fa["v"] = F
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"comment fingerprint {fp}\n"
              f"element vertex {len(V)}\n"
              + "".join(f"property float {n}\n" for n in names) +
              f"element face {len(F)}\nproperty list uchar int vertex_indices\nend_header\n")
    save(path, lambda f: (f.write(header.encode("ascii")), va.tofile(f), fa.tofile(f)))
    print(f"  → {path}  ({len(F):,} triangles, {time.time()-t0:.0f} s;  surface T/K "
          f"5/50/95%: {np.percentile(TK_v, [5, 50, 95]).round(0)})")


def write_mist(path, snapfile, snap, rest, fp):
    """Density (kg m^-3) and temperature (K) grids of the rest particles (render_gas), .npz."""
    import unyt, swiftsimio as sw
    from swiftsimio.visualisation.volume_render import render_gas
    from swiftsimio.objects import cosmo_array
    from scipy.ndimage import gaussian_filter

    data = sw.load(snapfile)
    if snapfile == IC_FILE:
        data = repair_ic(data, snapfile, sw, cosmo_array)
    cf, cm = data.gas.coordinates.cosmo_factor, data.gas.coordinates.comoving
    try:    units = data.metadata.boxsize.units
    except Exception: units = "code_length"
    c, h   = snap["com"], BOX_SIZE / 2 * UNIT_M / snap["u_l_m"]   # code units
    region = cosmo_array([c[0]-h, c[0]+h, c[1]-h, c[1]+h, c[2]-h, c[2]+h],
                         units=units, cosmo_factor=cf, comoving=cm)
    m = snap["masses"] * rest                       # body particles get zero mass
    data.gas.f_m  = cosmo_array(m,              "kg",            cosmo_factor=cf, comoving=cm)
    data.gas.f_um = cosmo_array(m * snap["TK"], "dimensionless", cosmo_factor=cf, comoving=cm)
    kw   = dict(resolution=RESOLUTION, region=region, parallel=False, periodic=False)
    dens = render_gas(data, project="f_m",  **kw).to(unyt.kg / unyt.m**3).value
    mT   = render_gas(data, project="f_um", **kw).to(1 / unyt.m**3).value
    temp = np.where(dens > 0, mT / np.maximum(dens, 1e-30), 0.0)
    dens, temp = gaussian_filter(dens, 1.0), gaussian_filter(temp, 1.0)

    temp[dens <= 0] = 0.0
    save(path, lambda f: np.savez_compressed(f, fingerprint=np.array(fp),
             box_size=np.float32(BOX_SIZE),
             density=np.maximum(dens, 0).astype(np.float16),             # half precision is
             temperature=np.minimum(temp, MIST_MAX_T_K).astype(np.float16)))  # plenty, and half the disk
    print(f"  → {path}  ({rest.sum():,} mist particles)")


def repair_ic(data, path, sw, cosmo_array):
    """The IC file lacks some fields swiftsimio needs; borrow units from a snapshot."""
    tmpl = sw.load(snapshot_path(1)).gas
    cf, cm = tmpl.coordinates.cosmo_factor, tmpl.coordinates.comoving
    with h5py.File(path, "r") as f:
        if getattr(data.gas, "coordinates", None) is None:
            data.gas.coordinates = cosmo_array(f["PartType0/Coordinates"][:],
                units=tmpl.coordinates.units, cosmo_factor=cf, comoving=cm)
        if getattr(data.gas, "smoothing_lengths", None) is None:
            data.gas.smoothing_lengths = cosmo_array(f["PartType0/SmoothingLengths"][:],
                units=tmpl.smoothing_lengths.units,
                cosmo_factor=tmpl.smoothing_lengths.cosmo_factor,
                comoving=tmpl.smoothing_lengths.comoving)
        if getattr(data.gas, "masses", None) is None:
            m_si = f["Units"].attrs["Unit mass in cgs (U_M)"][0] / 1000.0
            data.gas.masses = cosmo_array(f["PartType0/Masses"][:] * m_si,
                units="kg", cosmo_factor=cf, comoving=cm)
    return data


# ===========================================================================
# ONE FRAME
# ===========================================================================
def process_frame(frame):
    body_file, rest_file = output_files(frame)
    body_fp, rest_fp = fingerprints()
    need_body = file_fingerprint(body_file) != body_fp
    need_rest = file_fingerprint(rest_file) != rest_fp
    if not (need_body or need_rest):
        print(f"[Frame {frame:04d}]  up to date"); return

    path = snapshot_path(frame)
    print(f"\n[Frame {frame:04d}]  {os.path.basename(path)}")
    snap     = read_snapshot(path)
    r_w, r_m = particle_radius_world(snap)
    body     = body_mask(snap, r_w)
    rest     = ~body
    print(f"  T/K 50/90/99%: {np.percentile(snap['TK'], [50, 90, 99]).round(0)}   "
          f"body {body.sum():,}, rest {rest.sum():,} particles")
    dense = snap["rho"] >= SURFACE_RHO
    lost  = snap["masses"][dense & ~snap["inside"]].sum() / max(snap["masses"][dense].sum(), 1e-30)
    if lost > 1e-4:
        print(f"  NOTE: {100 * lost:.2f}% of the dense material is outside the box (BOX_SIZE)")

    if not need_body:
        print(f"  [up to date] {body_file}")
    else:
        if BODY_STYLE == "MESH":
            write_mesh(body_file, snap, body, r_w, body_fp)
        else:
            write_particles(body_file, snap, body, np.ones(body.sum()), body_fp)

    if not need_rest:
        print(f"  [up to date] {rest_file}")
    else:
        if REST_STYLE == "MIST":
            write_mist(rest_file, path, snap, rest, rest_fp)
        else:
            fade = smoothstep(*REST_FADE_RHO, snap["rho"])
            sel  = rest & snap["inside"] & (fade > 0)
            write_particles(rest_file, snap, sel, fade[sel], rest_fp)


# ===========================================================================
# FRAME SELECTION
# ===========================================================================
def parse_frames(args):
    """'800', '300-319' (inclusive), '0-2160/100' (with step) and 'all', in any mix."""
    frames = []
    for a in args:
        if a == "all":
            frames += [0] + sorted(FRAME_TO_TIME); continue
        a, _, step = a.partition("/")
        lo, _, hi  = a.partition("-")
        frames += list(range(int(lo), int(hi or lo) + 1, int(step or 1)))
    return frames


def frames_to_process(argv):
    if len(argv) == 2 and all(a.isdigit() for a in argv):     # START END (chunked runs)
        lo, hi = map(int, argv)
        return ([0] if lo <= 1 else []) + list(range(max(lo, 1), hi + 1))
    return parse_frames(argv) if argv else list(TEST_FRAMES)


# ===========================================================================
# MAIN
# ===========================================================================
if __name__ == "__main__":
    if not os.path.isdir(DATA_DIR):
        raise SystemExit(f"\nCannot find data folder:\n  {DATA_DIR}\n")
    for d in {os.path.dirname(f) for f in output_files(0)}:
        os.makedirs(d, exist_ok=True)

    frames = [f for f in frames_to_process(sys.argv[1:]) if f == 0 or f in FRAME_TO_TIME]

    UNIT_M, units = blender_unit()
    load_reference_properties()
    print(f"{units}\nBox {BOX_SIZE:g} Blender units across.  {len(frames)} frame(s), "
          f"mode {RENDER_MODE}, SURFACE_RHO {SURFACE_RHO:g}\n")

    t0, failed = time.time(), []
    for fr in frames:
        try:
            process_frame(fr)
        except Exception:
            traceback.print_exc()
            print(f"  [FAILED] frame {fr:04d}")
            failed.append(fr)

    print(f"\nStage 1 done in {(time.time()-t0)/60:.1f} min.")
    if failed:
        print(f"{len(failed)} frame(s) failed: {' '.join(map(str, failed))}")
