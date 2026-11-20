"""Verbatim copies of GN-Bench's coordinate code, for check 1 (see PLAN.md, check 1).

Source: GN-Bench-Tools at commit 0ce8367 (the GN0 submodule pinned in config.yaml, pins.gn_bench_tools),
file GN_Bench/sims/GN_Bench_simulator/GN_Bench_simulator.py.

Why copied and not imported: that module imports torch, cv2, gymnasium, scipy and the GN_Bench
package (with its gsplat renderer) at the top (lines 1 to 47), so importing it needs the full
CUDA rendering install. Check 1 runs before that install and the local unit test runs on a Mac.
The functions below are pure numpy/pathlib code; they are copied unchanged, with the line numbers
of the original given above each one. `transform_from_world_to_pixel` is a method of GNBenchSim in
the original; it is copied with `self.meta`, `self.map_width` and `self.map_height` turned into
arguments and no other change. `verify_against_gnbench()` compares these copies with the originals
when GN_Bench is importable (on the rented machine after the check 3 install).
"""
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

# GN_Bench_simulator.py line 553: `self.threshold = 200`; a pixel is passable when value >= threshold
# (reconstruct_passable_grid, lines 185 to 188).
PASSABLE_THRESHOLD = 200


# ---- GN_Bench_simulator.py lines 185 to 188 (verbatim)
def reconstruct_passable_grid(
    rows: List[List[int]], threshold: int
) -> List[List[bool]]:
    return [[value >= threshold for value in row] for row in rows]


# ---- GN_Bench_simulator.py lines 217 to 343 (verbatim: read_png_size, load_occupancy_metadata,
# ---- derive_affine_transform, transform_from_world_to_GS, transform_from_GS_to_world)
def read_png_size(path: Path) -> Tuple[int, int]:
    if not path.exists():
        return 0, 0  # Fallback
    with path.open("rb") as fh:
        fh.read(8)
        fh.read(4)
        fh.read(4)
        width = int.from_bytes(fh.read(4), "big")
        height = int.from_bytes(fh.read(4), "big")
    return width, height


def load_occupancy_metadata(scene_dir: Path) -> dict:
    occ_json = scene_dir / "occupancy.json"

    with occ_json.open("r", encoding="utf-8") as fh:
        occ = json.load(fh)

    scale = float(occ.get("scale"))

    occ_png = scene_dir / "occupancy.png"
    if occ_png.exists():
        width_px, height_px = read_png_size(occ_png)

    min_x, min_y, min_z = map(float, occ.get("min"))
    max_x, max_y, max_z = map(float, occ.get("max", (0.0, 0.0, 0.0)))

    gs_left = min_x
    gs_right = gs_left + width_px * scale
    gs_top = max_y
    gs_bottom = gs_top - height_px * scale

    lower_vals = occ.get("lower")
    upper_vals = occ.get("upper")
    lower = list(map(float, lower_vals or [min_x, min_y, min_z]))
    upper = list(map(float, upper_vals or [max_x, max_y, max_z]))

    return {
        "width": int(width_px),
        "height": int(height_px),
        "scale": scale,
        # GS coordinate bounds
        "left": gs_left,
        "right": gs_right,
        "top": gs_top,
        "bottom": gs_bottom,
        "lower_z": float(lower[2]),
        "upper_z": float(upper[2]),
        # pixel coordinate bounds
        "lower": lower,
        "upper": upper,
    }


def derive_affine_transform(points, pixels, meta):
    n = len(points)
    if n < 2:
        return 1.0, 0.0, 1.0, 0.0

    scale, left, top = float(meta["scale"]), float(meta["left"]), float(meta["top"])

    sum_x, sum_y = sum(p[0] for p in points), sum(p[1] for p in points)
    sum_x2, sum_y2 = sum(p[0] ** 2 for p in points), sum(p[1] ** 2 for p in points)

    sum_map_x, sum_map_y, sum_x_map_x, sum_y_map_y = 0.0, 0.0, 0.0, 0.0

    for pt, pix in zip(points, pixels):
        map_x = left + int(pix[0]) * scale
        map_y = top - int(pix[1]) * scale
        sum_map_x += map_x
        sum_map_y += map_y
        sum_x_map_x += pt[0] * map_x
        sum_y_map_y += pt[1] * map_y

    denom_x = n * sum_x2 - sum_x**2
    denom_y = n * sum_y2 - sum_y**2

    if abs(denom_x) < 1e-8 or abs(denom_y) < 1e-8:
        return 1.0, 0.0, 1.0, 0.0

    a_x = (n * sum_x_map_x - sum_x * sum_map_x) / denom_x
    b_x = (sum_map_x - a_x * sum_x) / n
    a_y = (n * sum_y_map_y - sum_y * sum_map_y) / denom_y
    b_y = (sum_map_y - a_y * sum_y) / n
    return a_x, b_x, a_y, b_y


def transform_from_world_to_GS(
    pt_world: np.ndarray, affine: tuple, meta: dict
) -> np.ndarray:
    a_x, b_x, a_y, b_y = affine
    x_aff = a_x * pt_world[0] + b_x
    y_aff = a_y * pt_world[1] + b_y

    center_x = 0.5 * (meta["left"] + meta["right"])
    center_y = 0.5 * (meta["top"] + meta["bottom"])
    x_final = center_x * 2.0 - x_aff
    y_final = center_y * 2.0 - y_aff

    return np.array([x_final, y_final, pt_world[2]], dtype=np.float32)


def transform_from_GS_to_world(
    pt_gs: np.ndarray, affine: tuple, meta: dict
) -> np.ndarray:
    a_x, b_x, a_y, b_y = affine
    x_gs = pt_gs[0]
    y_gs = pt_gs[1]

    center_x = 0.5 * (meta["left"] + meta["right"])
    center_y = 0.5 * (meta["top"] + meta["bottom"])
    x_aff = center_x * 2.0 - x_gs
    y_aff = center_y * 2.0 - y_gs

    if abs(a_x) < 1e-8:
        x_nav = x_aff
    else:
        x_nav = (x_aff - b_x) / a_x

    if abs(a_y) < 1e-8:
        y_nav = y_aff
    else:
        y_nav = (y_aff - b_y) / a_y

    z_nav = pt_gs[2] if pt_gs.shape[0] > 2 else 0.0

    return np.array([x_nav, y_nav, z_nav], dtype=np.float32)


# ---- GN_Bench_simulator.py lines 1413 to 1432, method GNBenchSim.transform_from_world_to_pixel.
# ---- Only change: self.meta / self.map_width / self.map_height are arguments.
def transform_from_world_to_pixel(
    meta: dict, map_width: int, map_height: int, position: np.ndarray
) -> Tuple[int, int]:
    wx, wy = position[0], position[1]

    span_x = meta["upper"][0] - meta["lower"][0]
    span_y = meta["upper"][1] - meta["lower"][1]

    scale_x = span_x / max(map_width, 1)
    scale_y = span_y / max(map_height, 1)

    px = (-wx - meta["lower"][0]) / scale_x - 0.5
    py = (meta["upper"][1] + wy) / scale_y - 0.5

    px_int = int(round(px))
    py_int = int(round(py))

    px_final = max(0, min(px_int, map_width - 1))
    py_final = max(0, min(py_int, map_height - 1))

    return px_final, py_final


# ---- Not copied from GN-Bench: the per-episode affine inputs, built the way
# ---- GNBenchSim._load_raster_points (GN_Bench_simulator.py, starting at line 792) builds them
# ---- from the GN-Matrix row (gn0/benchmark/dataset.py _build_path_info, GN0 c927781, starting at line 93).
def episode_raster_points(row) -> Tuple[list, list]:
    points = [
        np.array([float(x), float(y)], dtype=np.float32)
        for x, y in zip(row["path_raster_world_x"], row["path_raster_world_y"])
    ]
    pixels = [
        (int(px), int(py))
        for px, py in zip(row["path_raster_pixel_x"], row["path_raster_pixel_y"])
    ]
    return points, pixels


def episode_affine(row, meta: dict) -> tuple:
    points, pixels = episode_raster_points(row)
    return derive_affine_transform(points, pixels, meta)


def gs_points_to_world(points_gs: np.ndarray, affine: tuple, meta: dict) -> np.ndarray:
    """(N,3) InteriorGS-frame points to GN-Bench world frame, through transform_from_GS_to_world.

    The copied function is applied to the stacked (3, N) array, which it handles unchanged
    (every operation in it is elementwise; the z test reads shape[0] == 3).
    """
    pts = np.asarray(points_gs, dtype=np.float64)
    out = transform_from_GS_to_world(pts.T, affine, meta)
    return np.asarray(out, dtype=np.float64).T


def world_points_to_gs(points_world: np.ndarray, affine: tuple, meta: dict) -> np.ndarray:
    pts = np.asarray(points_world, dtype=np.float64)
    out = transform_from_world_to_GS(pts.T, affine, meta)
    return np.asarray(out, dtype=np.float64).T


def verify_against_gnbench(scene_dir: Path, row) -> dict:
    """Compare the copies with GN-Bench's own functions. Returns {"available": False} when
    GN_Bench cannot be imported (as on the Mac or before the check 3 install)."""
    try:
        from GN_Bench.sims.GN_Bench_simulator import GN_Bench_simulator as gbs  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - any import failure means "not available"
        return {"available": False, "reason": repr(exc)}
    meta_copy = load_occupancy_metadata(Path(scene_dir))
    meta_orig = gbs.load_occupancy_metadata(Path(scene_dir))
    pts, pix = episode_raster_points(row)
    aff_copy = derive_affine_transform(pts, pix, meta_copy)
    aff_orig = gbs.derive_affine_transform(pts, pix, meta_orig)
    probe = np.array([1.234, -2.345, 1.3], dtype=np.float64)
    gw_copy = transform_from_GS_to_world(probe, aff_copy, meta_copy)
    gw_orig = gbs.transform_from_GS_to_world(probe, aff_orig, meta_orig)
    return {
        "available": True,
        "meta_equal": meta_copy == meta_orig,
        "affine_equal": tuple(aff_copy) == tuple(aff_orig),
        "gs_to_world_equal": bool(np.array_equal(gw_copy, gw_orig)),
    }
