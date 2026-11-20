#!/usr/bin/env python
"""Local frame check for check 3 (runs on the Mac, no splat file needed).

Question: in which frame are InteriorGS labels.json boxes, relative to the
splat (GS) frame GN-Bench renders in? GN-Bench maps occupancy pixels to the
GS frame through world coordinates: world-to-pixel
(GNBenchSim.transform_from_world_to_pixel) followed by world-to-GS
(transform_from_world_to_GS with the episode affine), which works out to
GS x = right - px * scale, GS y = bottom + py * scale (meta from the
simulator's own load_occupancy_metadata). If labels.json is in the GS frame,
furniture box footprints should land on occupied (0) pixels of occupancy.png
under that mapping and not under its mirror images.

Also prints, for this scene's GN-Matrix episodes: missing label ids, and the
horizontal distance from the goal position (mapped to GS with the
simulator's own derive_affine_transform and transform_from_world_to_GS) to the
goal box.

The simulator functions are taken from the pinned GN_Bench_simulator.py
source with ast (the module itself imports torch and gsplat).

Usage:
    python3.11 -I frame_check_local.py <GN_Bench_simulator.py> <scene_dir> <split.parquet>
"""
from __future__ import annotations

import ast
import json
import sys
import typing
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load_sim_functions(src_path: Path) -> dict:
    want = {"read_png_size", "load_occupancy_metadata", "derive_affine_transform",
            "transform_from_world_to_GS"}
    mod = ast.parse(src_path.read_text())
    body = [n for n in mod.body if isinstance(n, ast.FunctionDef) and n.name in want]
    if {n.name for n in body} != want:
        raise SystemExit("could not find the simulator functions in the source")
    ns = {"np": np, "Path": Path, "json": json}
    ns.update({k: getattr(typing, k) for k in ("Tuple", "List", "Optional", "Dict", "Any", "Sequence", "Union")})
    exec(compile(ast.Module(body=body, type_ignores=[]), str(src_path), "exec"), ns)
    return ns


def main():
    sys.path.insert(0, str(HERE))
    import c3lib
    import pyarrow.parquet as pq
    from PIL import Image

    sim_src, scene_dir, parquet = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    f = load_sim_functions(sim_src)
    meta = f["load_occupancy_metadata"](scene_dir)
    occ = np.array(Image.open(scene_dir / "occupancy.png"))
    H, W = occ.shape
    s, left, right, top, bottom = meta["scale"], meta["left"], meta["right"], meta["top"], meta["bottom"]
    to_px = {
        "gn_bench_gs (x=right-px*s, y=bottom+py*s)": lambda x, y: ((right - x) / s, (y - bottom) / s),
        "mirror x (x=left+px*s, y=bottom+py*s)": lambda x, y: ((x - left) / s, (y - bottom) / s),
        "mirror y (x=right-px*s, y=top-py*s)": lambda x, y: ((right - x) / s, (top - y) / s),
        "mirror xy (x=left+px*s, y=top-py*s)": lambda x, y: ((x - left) / s, (top - y) / s),
    }
    labels = json.loads((scene_dir / "labels.json").read_text())
    skip = {"door", "window", "floor", "wall", "ceiling", "carpet", "rug"}
    print(f"scene {scene_dir.name}: occupancy {W}x{H}, labels {len(labels)}, "
          f"without box {sum(1 for e in labels if c3lib.box_corners(e) is None)}")
    for name, fn in to_px.items():
        vals = []
        for e in labels:
            c = c3lib.box_corners(e)
            if c is None or str(e["label"]).lower() in skip or c[:, 2].min() > 0.3 or c[:, 2].max() < 0.3:
                continue
            lo, hi = c.min(0), c.max(0)
            for x in np.linspace(lo[0], hi[0], 5)[1:-1]:
                for y in np.linspace(lo[1], hi[1], 5)[1:-1]:
                    px, py = fn(x, y)
                    px, py = int(px), int(py)
                    if 0 <= px < W and 0 <= py < H:
                        vals.append(occ[py, px])
        vals = np.array(vals)
        print(f"  {name}: {len(vals)} footprint samples, occupied {np.mean(vals == 0):.3f}, free {np.mean(vals == 255):.3f}")

    df = pq.read_table(parquet).to_pandas()
    df = df[df.scene == scene_dir.name]
    by_id = {}
    for e in labels:
        by_id.setdefault(str(e["ins_id"]), []).append(e)
    missing = [r for r in df.label_ins_id if str(r) not in by_id]
    dists = []
    for _, r in df.iterrows():
        es = by_id.get(str(r.label_ins_id))
        c = c3lib.box_corners(es[0]) if es else None
        if c is None:
            continue
        pts = [np.array([x, y], dtype=np.float32) for x, y in zip(r.path_raster_world_x, r.path_raster_world_y)]
        pix = [(int(x), int(y)) for x, y in zip(r.path_raster_pixel_x, r.path_raster_pixel_y)]
        aff = f["derive_affine_transform"](pts, pix, meta)
        g = np.asarray(f["transform_from_world_to_GS"](np.array([r.goal_world_x, r.goal_world_y, 0.0]), aff, meta),
                       dtype=np.float64)
        g[2] = c[:, 2].mean()
        dists.append(c3lib.point_box_distance(g, c))
    d = np.array(dists)
    print(f"  episodes in {parquet.name}: {len(df)}, missing label ids: {len(missing)}")
    if len(d):
        print(f"  goal position to goal box (horizontal, m): median {np.median(d):.2f}, "
              f"within 1 m {int((d <= 1).sum())}, within 3 m {int((d <= 3).sum())}, max {d.max():.2f}")


if __name__ == "__main__":
    main()
