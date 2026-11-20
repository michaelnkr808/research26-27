"""Floor plan overlay and label_ins_id lookup for check 1 (PLAN.md, check 1, last two bullets).

For one InteriorGS scene: map the structure.json wall centre lines into GN-Bench's world frame
(gnbench_copy.transform_from_GS_to_world with an episode's affine) and then to occupancy.png pixels
with GN-Bench's own world-to-pixel mapping (gnbench_copy.transform_from_world_to_pixel, copied from
GN_Bench_simulator.py lines 1413 to 1432). Draw them on the occupancy image and compute the wall
share: the fraction of points sampled every check1.wall_sample_spacing_m along wall centre lines that
land within check1.wall_hit_tolerance_px pixels (Euclidean, in pixel units) of an occupied pixel.

"Occupied" is value 0, as the InteriorGS dataset card defines it (255 free, 0 occupied, 127 unknown).
The share is also written with GN-Bench's own notion (not passable: value < 200, threshold from
GN_Bench_simulator.py line 553), as a second number; the PLAN's pass bar is read on the first.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image, ImageDraw

from gnbench_copy import (
    PASSABLE_THRESHOLD,
    episode_affine,
    gs_points_to_world,
    load_occupancy_metadata,
    transform_from_world_to_pixel,
)

OCCUPIED_VALUE = 0  # InteriorGS dataset card: 0 = occupied
DRAW_SCALE = 4  # overlay image is the occupancy map enlarged 4x (nearest neighbour) for viewing


def scene_rows(df, scene: str):
    """GN-Matrix rows of one scene, in a fixed order (trajectory_id, then row order)."""
    sub = df[df["scene"] == scene].copy()
    sub["_row"] = np.arange(len(sub))
    return sub.sort_values(["trajectory_id", "_row"], kind="mergesort")


def wall_sample_points(structure: dict, spacing: float) -> np.ndarray:
    pts = []
    for w in structure.get("walls", []):
        p0, p1 = (np.asarray(p, dtype=np.float64) for p in w["location"])
        L = float(np.linalg.norm(p1 - p0))
        n = max(int(math.ceil(L / spacing)), 1)
        t = np.linspace(0.0, 1.0, n + 1)
        pts.append(p0[None, :] + t[:, None] * (p1 - p0)[None, :])
    if not pts:
        return np.zeros((0, 2))
    return np.vstack(pts)


def _disk_offsets(r: float):
    R = int(math.floor(r))
    return [(dx, dy) for dx in range(-R, R + 1) for dy in range(-R, R + 1) if dx * dx + dy * dy <= r * r]


def _share(mask: np.ndarray, px: np.ndarray, py: np.ndarray, tol: float) -> np.ndarray:
    H, W = mask.shape
    hit = np.zeros(len(px), dtype=bool)
    for dx, dy in _disk_offsets(tol):
        x, y = px + dx, py + dy
        ok = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        hit[ok] |= mask[y[ok], x[ok]]
    return hit


def gs_xy_to_pixels(xy_gs: np.ndarray, affine, meta, W: int, H: int) -> np.ndarray:
    pts = np.c_[xy_gs, np.zeros(len(xy_gs))]
    world = gs_points_to_world(pts, affine, meta)
    return np.array([transform_from_world_to_pixel(meta, W, H, p) for p in world], dtype=np.int64).reshape(-1, 2)


def run_overlay(scene_dir: Path, scene_rows_df, cfg: dict, out_dir: Path,
                affine_row=None, draw: bool = True) -> Dict:
    """scene_rows_df: this scene's GN-Matrix rows (scene_rows()). affine_row: the row whose affine is
    used (default: the first row in that order)."""
    scene_dir, out_dir = Path(scene_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scene = scene_dir.name
    meta = load_occupancy_metadata(scene_dir)
    img = np.array(Image.open(scene_dir / "occupancy.png"))
    if img.ndim != 2:
        raise ValueError(f"{scene}: occupancy.png is not single-channel ({img.shape})")
    H, W = img.shape
    if (W, H) != (meta["width"], meta["height"]):
        raise ValueError(f"{scene}: PNG size {W}x{H} != metadata {meta['width']}x{meta['height']}")
    structure = json.loads((scene_dir / "structure.json").read_text())
    labels = json.loads((scene_dir / "labels.json").read_text())

    rows = list(scene_rows_df.to_dict("records"))
    if not rows:
        raise ValueError(f"{scene}: no GN-Matrix rows")
    arow = affine_row if affine_row is not None else rows[0]
    affine = episode_affine(arow, meta)

    # spread of the per-episode affine across this scene's episodes: where does the scene centre land
    probe = np.array([[0.5 * (meta["left"] + meta["right"]), 0.5 * (meta["top"] + meta["bottom"]), 0.0]])
    centres = np.array([gs_points_to_world(probe, episode_affine(r, meta), meta)[0, :2] for r in rows])
    ref = gs_points_to_world(probe, affine, meta)[0, :2]
    spread = float(np.max(np.linalg.norm(centres - ref, axis=1)))

    spacing = float(cfg["wall_sample_spacing_m"])
    tol = float(cfg["wall_hit_tolerance_px"])
    pts = wall_sample_points(structure, spacing)
    pix = gs_xy_to_pixels(pts, affine, meta, W, H)
    px, py = pix[:, 0], pix[:, 1]
    hit_occ = _share(img == OCCUPIED_VALUE, px, py, tol)
    hit_np = _share(img < PASSABLE_THRESHOLD, px, py, tol)
    share = float(hit_occ.mean()) if len(hit_occ) else float("nan")
    share_np = float(hit_np.mean()) if len(hit_np) else float("nan")

    # label_ins_id lookup
    by_id: Dict[str, dict] = {}
    for e in labels:
        by_id.setdefault(str(e.get("ins_id")), e)
    lookup_rows = []
    n_miss = n_nobox = n_label_mismatch = 0
    for r in rows:
        iid = str(r["label_ins_id"])
        e = by_id.get(iid)
        rec = {"trajectory_id": r["trajectory_id"], "label_ins_id": iid, "label_label": r["label_label"],
               "found": e is not None, "has_bounding_box": bool(e and "bounding_box" in e),
               "labels_json_label": (e or {}).get("label"), "goal_to_box_centre_m": ""}
        if e is None:
            n_miss += 1
        else:
            if "bounding_box" not in e:
                n_nobox += 1
            else:
                c = np.array([[p["x"], p["y"], p["z"]] for p in e["bounding_box"]]).mean(axis=0)
                cw = gs_points_to_world(c[None, :], episode_affine(r, meta), meta)[0]
                rec["goal_to_box_centre_m"] = round(float(np.hypot(cw[0] - r["goal_world_x"], cw[1] - r["goal_world_y"])), 3)
            if str(e.get("label")) != str(r["label_label"]):
                n_label_mismatch += 1
        lookup_rows.append(rec)
    with open(out_dir / f"{scene}_label_lookup.csv", "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(lookup_rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(lookup_rows)
    dists = [x["goal_to_box_centre_m"] for x in lookup_rows if x["goal_to_box_centre_m"] != ""]

    if draw:
        S = DRAW_SCALE
        im = Image.fromarray(img).convert("RGB").resize((W * S, H * S), Image.NEAREST)
        dr = ImageDraw.Draw(im)
        c = lambda p: (p[0] * S + S // 2, p[1] * S + S // 2)  # noqa: E731
        for w in structure.get("walls", []):
            ends = gs_xy_to_pixels(np.asarray(w["location"], dtype=np.float64), affine, meta, W, H)
            dr.line([c(ends[0]), c(ends[1])], fill=(230, 0, 0), width=2)
        for h in structure.get("holes", []):
            prof = np.asarray(h["profile"], dtype=np.float64)[:, :2]
            ends = gs_xy_to_pixels(prof, affine, meta, W, H)
            col = (0, 90, 255) if str(h.get("type", "")).upper() == "DOOR" else (0, 200, 220)
            dr.line([c(ends[ends[:, 0].argmin()]), c(ends[ends[:, 0].argmax()])] if np.ptp(ends[:, 0]) >= np.ptp(ends[:, 1])
                    else [c(ends[ends[:, 1].argmin()]), c(ends[ends[:, 1].argmax()])], fill=col, width=3)
        for (x, y), ok in zip(pix, hit_occ):
            if not ok:
                cx, cy = c((x, y))
                dr.rectangle([cx - 2, cy - 2, cx + 2, cy + 2], outline=(255, 0, 255))
        im.save(out_dir / f"{scene}_overlay.png")

    summary = {
        "scene": scene, "image_wh": [W, H], "meta": meta, "affine": list(map(float, affine)),
        "affine_from_trajectory_id": int(arow["trajectory_id"]),
        "affine_spread_m": spread,
        "n_wall_points": int(len(pts)), "wall_hit_tolerance_px": tol, "wall_share_occupied": share,
        "wall_share_not_passable": share_np, "wall_share_pass_bar": float(cfg["wall_share_pass"]),
        "wall_share_pass": bool(share >= float(cfg["wall_share_pass"])),
        "n_episodes": len(rows), "label_ins_id_misses": n_miss, "label_found_without_box": n_nobox,
        "label_text_mismatch": n_label_mismatch,
        "goal_to_box_centre_m_median": float(np.median(dists)) if dists else None,
        "goal_to_box_centre_m_max": float(np.max(dists)) if dists else None,
    }
    (out_dir / f"{scene}_overlay_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def choose_overlay_scenes(df, main_scene: str, n_total: int, seed: int) -> List[str]:
    """PLAN: the main scene plus (n_total - 1) others from the split, chosen from the sorted scene
    ids with numpy default_rng(seed), without replacement."""
    others = sorted(s for s in set(df["scene"].astype(str)) if s != main_scene)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(others), size=n_total - 1, replace=False)
    return [main_scene] + [others[i] for i in sorted(pick)]


def print_overlay_summary(s: Dict) -> None:
    print(f"[overlay] {s['scene']}: wall share (within {s['wall_hit_tolerance_px']:g} px of occupied=0) = "
          f"{s['wall_share_occupied']:.4f} over {s['n_wall_points']} points "
          f"(bar {s['wall_share_pass_bar']}: {'PASS' if s['wall_share_pass'] else 'FAIL'}); "
          f"not-passable share = {s['wall_share_not_passable']:.4f}; "
          f"label_ins_id misses = {s['label_ins_id_misses']} of {s['n_episodes']} episodes "
          f"(found without box {s['label_found_without_box']}, label text mismatch {s['label_text_mismatch']}); "
          f"affine spread across episodes = {s['affine_spread_m']:.3f} m")
