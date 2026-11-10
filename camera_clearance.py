"""
camera_clearance.py  –  how close does dense material come to the camera?

For a few candidate camera directions, all at the distances of the Stage 2
camera PATH, reports the closest approach of dense (body) material to the
camera over the run.  It reads RUN_DIR/particles/body_*.ply, so a PARTICLES-mode
Stage 1 run is needed:

    python3 camera_clearance.py            # every 50th frame
    python3 camera_clearance.py 400 800 10 # frames 400-800, every 10th

Each candidate is a CAMERA_VIEW for Stage 2, (distance scale, turn about the
z-axis in degrees, elevation in degrees or None = as in PATH), applied to PATH
exactly as Stage 2 does; copy the one you choose into combined_02_build.py.
"""
import sys, re, os
import numpy as np

RUN_DIR = "/cosma5/data/dp004/vreke/swift/rssse/pE65_Th35_b60v10_3e6"

# Copy PATH from combined_02_build.py; SCALE is the distance scale of the views tried
SCALE = 0.24
PATH = [(1, (3.4, -26.9, 4.5)), (150, (2.4, -19.2, 3.2)), (250, (2.4, -19.2, 3.2)),
        (600, (3.2, -22.4, 4.8)), (750, (3.6, -24.0, 5.6)), (1000, (5.5, -35.2, 8.8)),
        (1200, (6.6, -42.0, 10.5)), (2162, (7.6, -48.0, 12.0))]

# (turn, elevation) of each CAMERA_VIEW tried, all at distance scale SCALE
CANDIDATES = [(0, None), (90, None), (180, None), (270, None),
              (0, 45), (0, 70), (180, 20), (0, 89)]
CANDIDATES = [(f"({SCALE:g}, {t}, {e})", t, e) for t, e in CANDIDATES]


def camera(frame, turn, elev):
    """Camera position for snapshot frame (Blender frame = frame + 1), linear between PATH."""
    fs = [f for f, _ in PATH]
    o = np.array([np.interp(frame + 1, fs, [p[k] for _, p in PATH]) for k in range(3)])
    o *= SCALE
    r, a = np.linalg.norm(o), np.arctan2(o[1], o[0]) + np.radians(turn)
    e = np.arcsin(o[2] / r) if elev is None else np.radians(elev)
    return r * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def read_xyz(path):
    d = open(path, "rb").read(); i = d.index(b"end_header\n") + 11; h = d[:i].decode()
    n = int(re.search(r"element vertex (\d+)", h).group(1))
    names = re.findall(r"property float (\w+)", h)
    v = np.frombuffer(d[i:i + n * 4 * len(names)], dtype=[(k, "<f4") for k in names])
    return np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)


lo, hi, step = (map(int, sys.argv[1:4]) if len(sys.argv) == 4 else (0, 2160, 50))
body  = lambda f: os.path.join(RUN_DIR, "particles", f"body_{f:04d}.ply")
frames = [f for f in range(lo, hi + 1, step) if os.path.exists(body(f))]
if not frames:
    raise SystemExit(f"No particle files found as {body(lo)} ...: check RUN_DIR, "
                     "or run Stage 1 in PARTICLES mode first.")
worst = {c[0]: (np.inf, None, None) for c in CANDIDATES}
print(f"{'frame':>5}  " + "  ".join(f"{c[0]:>18}" for c in CANDIDATES))
for f in frames:
    xyz, row = read_xyz(body(f)), []
    for label, turn, elev in CANDIDATES:
        c = camera(f, turn, elev)
        dist = np.sqrt(((xyz - c) ** 2).sum(1))
        near = np.percentile(dist, 0.01)          # ignore a handful of lone particles
        ratio = near / np.linalg.norm(c)          # vs. the camera's distance from the centre
        row.append(f"{near:7.2f} ({ratio:4.2f})")
        if ratio < worst[label][0]:
            worst[label] = (ratio, near, f)
    print(f"{f:5d}  " + "  ".join(f"{s:>18}" for s in row), flush=True)

print("\nClosest approach of dense material, as a fraction of the camera's distance from the centre")
print("(1 = no closer than the centre of mass; small = something passes close to the viewer):")
for label, (ratio, near, f) in sorted(worst.items(), key=lambda kv: -kv[1][0]):
    print(f"  CAMERA_VIEW = {label:<18} {ratio:4.2f}  ({near:.2f} R_E, frame {f})")
