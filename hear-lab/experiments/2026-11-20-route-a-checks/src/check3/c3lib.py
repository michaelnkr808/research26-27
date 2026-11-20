"""Pure-numpy helpers for check 3 (target visible at the start), route A checks.

Everything here runs without torch, gsplat or the simulator, so the projection,
visibility rule and episode selection can be unit-tested on a laptop
(test_c3lib.py). The simulator-facing code is in run_check3.py.

Conventions, read from the pinned sources (GN-Bench-Tools 0ce8367, GN0 c927781):

* Camera. GN_Bench_simulator.py '_render_observations' passes the sensor's
  HFOV value as the VERTICAL field of view (fovy = radians(HFOV)) and
  derives the horizontal one as fovx = 2 atan(tan(fovy / 2) * W / H).
  render/gsplat_renderer.py '_build_pinhole_camera' then builds
  fx = W / (2 tan(fovx / 2)), fy = H / (2 tan(fovy / 2)), cx = W / 2,
  cy = H / 2. 'camera_intrinsics' reproduces exactly that, so with the
  InteriorGS config (480 x 360, HFOV 70) the rendered image spans 70 degrees
  vertically and about 86 degrees horizontally.
* View matrix. '_build_look_at_matrix(eye, target, up=(0, 0, 1))': camera z
  points forward, x = forward cross up (image right), y = forward cross x
  (image down), the OpenCV convention gsplat expects. 'look_at_view' is a copy.
* Frames. The agent pose (GN-Matrix start_world_*) is in GN-Matrix world
  coordinates; the simulator maps it to splat (GS) coordinates with
  'transform_from_world_to_GS' and the per-episode affine fitted from the
  episode's own raster path ('derive_affine_transform'). InteriorGS
  labels.json boxes are in the GS frame (checked locally on scene
  0001_839920 by frame_check_local.py: furniture box footprints land on
  occupied pixels only under GN-Bench's own pixel-to-GS mapping).
* Depth. gsplat 'RGB+ED' returns expected depth along the camera z axis
  (z-depth, metres; recalled, measured on the machine); the simulator turns
  depth <= 1e-6 (nothing rendered) into float32 max and then clips to
  [MIN_DEPTH, MAX_DEPTH]. run_check3.py measures the convention on the
  machine with a synthetic plane before trusting it, and passes depth_kind
  ("z" or "ray") into 'evaluate_visibility'.
"""
from __future__ import annotations

import csv
import itertools
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

# gsplat near plane used by render_camera_gsplat (render/gsplat_renderer.py).
GSPLAT_NEAR_PLANE = 0.01
# Review frames are saved for this many times n_review_images_per_split
# candidates, so that enough usable ones remain after missing-id episodes.
REVIEW_OVERSAMPLE = 3

CSV_FIELDS = [
    "split", "row_index", "trajectory_id", "scene", "label_ins_id", "label_label",
    "status", "n_label_entries",
    "start_world_x", "start_world_y", "start_heading_deg",
    "goal_world_x", "goal_world_y",
    "straight_dist_m", "path_length_m",
    "within_radius_straight", "within_radius_path",
    "cam_to_box_centre_m", "goal_to_box_horiz_m",
    "n_box_points", "n_front", "n_in_image", "n_invalid_depth",
    "n_seen_primary", "share_seen_primary", "visible_primary",
    "n_seen_strict", "share_seen_strict", "visible_strict",
    "n_seen_loose", "share_seen_loose", "visible_loose",
    "depth_tol_primary_m", "depth_tol_strict_m", "depth_tol_loose_m",
    "depth_kind", "render_s",
]


# ----------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------
def load_config(path) -> dict:
    """Every check 3 parameter, read from the experiment's config.yaml."""
    import yaml  # only needed here

    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    c3 = cfg["check3"]
    tols = sorted(float(t) for t in c3["depth_tolerance_sensitivity_m"])
    if len(tols) != 2:
        raise ValueError("expected two sensitivity tolerances")
    return {
        "seed": int(cfg["seed"]),
        "pins": dict(cfg["pins"]),
        "splits": dict(c3["splits"]),
        "width": int(c3["camera"]["width"]),
        "height": int(c3["camera"]["height"]),
        "hfov_deg": float(c3["camera"]["hfov_deg"]),
        "camera_height_m": float(c3["camera"]["height_m"]),
        "spacing_m": float(c3["box_point_spacing_m"]),
        "tol_primary": float(c3["depth_tolerance_m"]),
        "tol_strict": tols[0],
        "tol_loose": tols[1],
        "success_radius_m": float(c3["success_radius_m"]),
        "n_review": int(c3["n_review_images_per_split"]),
    }


def tolerances(cfg: dict) -> Dict[str, float]:
    return {"primary": cfg["tol_primary"], "strict": cfg["tol_strict"], "loose": cfg["tol_loose"]}


# ----------------------------------------------------------------------------
# episode selection
# ----------------------------------------------------------------------------
def select_rows(n_rows: int, spec, seed: int) -> np.ndarray:
    """Row indices (0-based, into the split's parquet) used by check 3.

    spec "all": every row. An integer n: numpy default_rng(seed).choice(n_rows,
    n, replace=False), sorted ascending (PLAN: seed 20261120, without
    replacement, by row).
    """
    if isinstance(spec, str):
        if spec != "all":
            raise ValueError(f"unknown split spec {spec!r}")
        return np.arange(n_rows, dtype=np.int64)
    n = int(spec)
    if n > n_rows:
        raise ValueError(f"asked for {n} rows from a split of {n_rows}")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_rows, size=n, replace=False)).astype(np.int64)


def review_candidates(rows: Sequence[int], n_review: int, seed: int) -> List[int]:
    """Ordered review candidates for one split: a seeded permutation
    (default_rng(seed)) of the selected rows, first n_review * REVIEW_OVERSAMPLE
    kept. The summary uses the first n_review of these whose box was found."""
    rows = np.asarray(sorted(int(r) for r in rows), dtype=np.int64)
    perm = np.random.default_rng(seed).permutation(len(rows))
    return [int(rows[i]) for i in perm[: n_review * REVIEW_OVERSAMPLE]]


# ----------------------------------------------------------------------------
# camera
# ----------------------------------------------------------------------------
def camera_intrinsics(width: int, height: int, hfov_config_deg: float) -> dict:
    """Pinhole intrinsics exactly as GN-Bench renders a sensor whose config
    says WIDTH, HEIGHT, HFOV (HFOV is used as fovy; see module docstring)."""
    fovy = math.radians(hfov_config_deg)
    fovx = 2.0 * math.atan(math.tan(fovy * 0.5) * (width / height))
    fx = float(width) / (2.0 * math.tan(fovx * 0.5))
    fy = float(height) / (2.0 * math.tan(fovy * 0.5))
    return {
        "fx": fx, "fy": fy, "cx": width * 0.5, "cy": height * 0.5,
        "width": int(width), "height": int(height),
        "fovx_deg": math.degrees(fovx), "fovy_deg": math.degrees(fovy),
        "K": np.array([[fx, 0.0, width * 0.5], [0.0, fy, height * 0.5], [0.0, 0.0, 1.0]]),
    }


def look_at_view(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """GS-to-camera 4x4, a copy of gsplat_renderer._build_look_at_matrix
    (float64 here; the renderer uses float32)."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    z_axis = target - eye
    n = np.linalg.norm(z_axis)
    z_axis = np.array([0.0, 0.0, 1.0]) if n < 1e-8 else z_axis / n
    x_axis = np.cross(z_axis, up)
    nx = np.linalg.norm(x_axis)
    x_axis = np.array([1.0, 0.0, 0.0]) if nx < 1e-8 else x_axis / nx
    y_axis = np.cross(z_axis, x_axis)
    view = np.eye(4)
    view[0, :3] = x_axis
    view[1, :3] = y_axis
    view[2, :3] = z_axis
    view[:3, 3] = -view[:3, :3] @ eye
    return view


def heading_forward(heading_deg: float) -> np.ndarray:
    """World forward vector for a GN-Matrix heading; the simulator uses
    R.from_euler('z', heading, degrees=True).apply([1, 0, 0])."""
    h = math.radians(heading_deg)
    return np.array([math.cos(h), math.sin(h), 0.0])


# ----------------------------------------------------------------------------
# boxes
# ----------------------------------------------------------------------------
def box_corners(entry: dict) -> Optional[np.ndarray]:
    """(8, 3) corners of a labels.json entry, or None if it has no box."""
    bb = entry.get("bounding_box")
    if not bb:
        return None
    arr = np.array([[float(p["x"]), float(p["y"]), float(p["z"])] for p in bb])
    if arr.shape != (8, 3):
        raise ValueError(f"bounding_box with shape {arr.shape}")
    return arr


def box_frame(corners: np.ndarray, tol: float = 1e-4):
    """Origin o and three edge vectors (o + {0,1}a + {0,1}b + {0,1}c gives
    the 8 corners) of an oriented box given by its corners in any order.
    Edges may have zero length (flat boxes). Raises ValueError otherwise."""
    corners = np.asarray(corners, dtype=np.float64)
    tol = tol * max(1.0, float(np.abs(corners).max()))
    for oi in range(8):
        o = corners[oi]
        vecs = corners - o
        others = [i for i in range(8) if i != oi]
        for i, j, k in itertools.combinations(others, 3):
            a, b, c = vecs[i], vecs[j], vecs[k]
            gen = np.array([o + s0 * a + s1 * b + s2 * c
                            for s0 in (0, 1) for s1 in (0, 1) for s2 in (0, 1)])
            d = np.linalg.norm(gen[:, None, :] - corners[None, :, :], axis=2)
            if d.min(axis=1).max() > tol or d.min(axis=0).max() > tol:
                continue
            lens = [np.linalg.norm(v) for v in (a, b, c)]
            ok = True
            for (u, lu), (w, lw) in itertools.combinations(list(zip((a, b, c), lens)), 2):
                if lu > tol and lw > tol and abs(float(u @ w)) > 1e-3 * lu * lw:
                    ok = False
            if ok:
                return o, np.stack([a, b, c])
    raise ValueError("corners do not form a box")


def sample_box_surface(corners: np.ndarray, spacing: float) -> np.ndarray:
    """Points on the six faces of the box on a grid whose step along each edge
    is at most 'spacing' (edges and corners included, duplicates removed).
    Returns (N, 3)."""
    o, edges = box_frame(corners)
    lengths = np.linalg.norm(edges, axis=1)
    n_int = [max(1, int(math.ceil(L / spacing - 1e-9))) if L > 0 else 0 for L in lengths]
    pts = []
    for fixed in range(3):
        u_i, v_i = [i for i in range(3) if i != fixed]
        su = np.linspace(0.0, 1.0, n_int[u_i] + 1)
        sv = np.linspace(0.0, 1.0, n_int[v_i] + 1)
        uu, vv = np.meshgrid(su, sv, indexing="ij")
        base = o + uu.reshape(-1, 1) * edges[u_i] + vv.reshape(-1, 1) * edges[v_i]
        for side in ((0.0, 1.0) if lengths[fixed] > 0 else (0.0,)):
            pts.append(base + side * edges[fixed])
    pts = np.concatenate(pts, axis=0)
    _, keep = np.unique(np.round(pts, 6), axis=0, return_index=True)
    return pts[np.sort(keep)]


def point_box_distance(point, corners: np.ndarray) -> float:
    """Euclidean distance from a point to the (solid) oriented box; 0 inside."""
    o, edges = box_frame(corners)
    rel = np.asarray(point, dtype=np.float64) - o
    d2 = 0.0
    for e in edges:
        L = np.linalg.norm(e)
        if L <= 0:
            continue
        t = float(rel @ (e / L))
        d2 += (min(0.0, t) if t < 0 else max(0.0, t - L)) ** 2
    # component orthogonal to all non-zero edges (flat boxes)
    nz = [e / np.linalg.norm(e) for e in edges if np.linalg.norm(e) > 0]
    if len(nz) < 3:
        q, _ = np.linalg.qr(np.stack(nz, axis=1))
        perp = rel - q @ (q.T @ rel)
        d2 += float(perp @ perp)
    return float(math.sqrt(d2))


def points_in_box(points: np.ndarray, corners: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """Boolean mask of points inside the oriented box (used for the per-scene
    frame diagnostic in run_check3.py)."""
    o, edges = box_frame(corners)
    rel = np.asarray(points, dtype=np.float64) - o
    inside = np.ones(len(rel), dtype=bool)
    for e in edges:
        L = np.linalg.norm(e)
        if L <= 0:
            continue
        t = rel @ (e / L)
        inside &= (t >= -margin) & (t <= L + margin)
    return inside


# ----------------------------------------------------------------------------
# projection and the visibility rule
# ----------------------------------------------------------------------------
def project_points(points: np.ndarray, view: np.ndarray, intr: dict):
    """Camera-frame coordinates and pixel coordinates of GS-frame points.
    Returns cam (N, 3), u (N,), v (N,). The pixel (col, row) of a point is
    (floor(u), floor(v)); gsplat puts pixel centres at integer + 0.5."""
    pts = np.asarray(points, dtype=np.float64)
    cam = pts @ view[:3, :3].T + view[:3, 3]
    z = cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = intr["fx"] * cam[:, 0] / z + intr["cx"]
        v = intr["fy"] * cam[:, 1] / z + intr["cy"]
    return cam, u, v


def evaluate_visibility(points: np.ndarray, view: np.ndarray, intr: dict,
                        depth: np.ndarray, invalid: np.ndarray,
                        tols: Dict[str, float], depth_kind: str = "z",
                        near: float = GSPLAT_NEAR_PLANE) -> dict:
    """PLAN rule: a point is seen if it projects inside the image, lies in
    front of the camera, and its depth is at most the rendered depth at that
    pixel plus the tolerance. Pixels where nothing was rendered (invalid)
    never count as seen; the number of in-image points that hit such pixels
    is reported as n_invalid_depth.

    depth is (H, W) in metres along the camera z axis when depth_kind is "z",
    or along the viewing ray when it is "ray".
    """
    H, W = depth.shape
    if (H, W) != (intr["height"], intr["width"]):
        raise ValueError("depth image size does not match the intrinsics")
    cam, u, v = project_points(points, view, intr)
    front = cam[:, 2] > near
    in_img = front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    col = np.zeros(len(u), dtype=np.int64)
    row = np.zeros(len(u), dtype=np.int64)
    col[in_img] = np.floor(u[in_img]).astype(np.int64)
    row[in_img] = np.floor(v[in_img]).astype(np.int64)
    if depth_kind == "z":
        pd = cam[:, 2]
    elif depth_kind == "ray":
        pd = np.linalg.norm(cam, axis=1)
    else:
        raise ValueError(depth_kind)
    rendered = np.full(len(u), np.nan)
    rendered[in_img] = depth[row[in_img], col[in_img]]
    bad = np.zeros(len(u), dtype=bool)
    bad[in_img] = invalid[row[in_img], col[in_img]]
    n = int(len(points))
    out = {
        "n_box_points": n,
        "n_front": int(front.sum()),
        "n_in_image": int(in_img.sum()),
        "n_invalid_depth": int((in_img & bad).sum()),
        "u": u, "v": v, "in_image": in_img,
    }
    for name, tol in tols.items():
        seen = in_img & ~bad
        seen[seen] = pd[seen] <= rendered[seen] + tol
        out[f"seen_{name}"] = seen
        out[f"n_seen_{name}"] = int(seen.sum())
        out[f"share_seen_{name}"] = float(seen.sum()) / n if n else 0.0
        out[f"visible_{name}"] = bool(seen.any())
    return out


# ----------------------------------------------------------------------------
# distances
# ----------------------------------------------------------------------------
def polyline_length(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Length of the GN-Matrix raster_world path, same formula as
    GNBenchSim._compute_path_length."""
    p = np.stack([np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)], axis=1)
    if len(p) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def straight_distance(x0, y0, x1, y1) -> float:
    return float(math.hypot(float(x1) - float(x0), float(y1) - float(y0)))


# ----------------------------------------------------------------------------
# CSV helpers (resume-safe append)
# ----------------------------------------------------------------------------
def repair_csv_tail(csv_path) -> None:
    """Drop a partial last line (no trailing newline) left by a crash, so the
    next append starts on a fresh line."""
    p = Path(csv_path)
    if not p.exists() or p.stat().st_size == 0:
        return
    data = p.read_bytes()
    if data.endswith(b"\n"):
        return
    cut = data.rfind(b"\n")
    p.write_bytes(data[: cut + 1] if cut >= 0 else b"")


def read_rows(csv_path) -> List[dict]:
    """Complete rows of a results CSV (a torn last line is ignored)."""
    p = Path(csv_path)
    if not p.exists():
        return []
    out = []
    with open(p, "r", newline="", encoding="utf-8") as fh:
        for rec in csv.DictReader(fh):
            if rec.get("render_s") is None or not str(rec.get("row_index", "")).isdigit():
                continue
            out.append(rec)
    return out


def done_keys(csv_path) -> set:
    """(split, row_index) pairs already written to the CSV."""
    return {(r["split"], int(r["row_index"])) for r in read_rows(csv_path)}
