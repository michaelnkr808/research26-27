"""Geometry exporter (written for route A check 1; moved to hear-lab/src on 2026-11-26): InteriorGS floor plan and boxes to a triangle mesh in GN-Bench's
world frame, with one RLR material category per face.

Frames (see RUN.md, "Frames"):
- InteriorGS frame: structure.json, labels.json and occupancy.json coordinates (right-handed, z up,
  metres; InteriorGS dataset card). GN-Bench calls this the "GS" frame.
- GN-Bench world frame: the frame of the GN-Matrix start pose, goal and path. GN-Bench maps it to the
  GS frame with a per-episode affine fit (gnbench_copy.derive_affine_transform) and
  transform_from_world_to_GS. The exporter builds the mesh in the GS frame and maps every vertex
  to world with GN-Bench's own transform_from_GS_to_world (copied in gnbench_copy.py), using the
  affine of the episode being simulated. z is unchanged by GN-Bench's transform.

What goes into the mesh:
- walls: each structure.json wall segment extruded to a solid of its thickness, from z=0 to its height,
  with every door and window hole on it cut through the full thickness (openings left open);
- floor (z=0) and ceiling (z = highest wall) for each structure.json room profile;
- one box per labels.json instance that has a bounding_box, except the structural labels in
  STRUCTURAL_LABELS (walls, floors, ceilings and rooms are already built from structure.json; door and
  window boxes would close the openings) and any instance passed in exclude_ins_ids.
Materials: the label's lowercase text is matched against the shipped mp3d_material_config.json with the
library's own rule (header RLRA_SetMaterialDatabaseJSON: the material with the most label substrings
contained in the category wins); no match gives the config's Default material.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Labels that are not furniture boxes (lowercase, exact match). See module docstring.
STRUCTURAL_LABELS = ("wall", "floor", "ceiling", "room", "door", "window")
# Categories used for the floor plan surfaces (matched with the same rule as every other label).
WALL_CATEGORY = "wall"
FLOOR_CATEGORY = "floor"
CEILING_CATEGORY = "ceiling"
# Boxes thinner than this along any edge are skipped and counted (degenerate triangles).
MIN_BOX_EDGE_M = 1e-4
EPS = 1e-9


# ----------------------------------------------------------------------------- materials
@dataclass
class MaterialMatch:
    source_label: str
    category_passed: str  # the string handed to RLRA_AddMeshIndices
    material_name: str
    n_matches: int
    tie: bool


class MaterialTable:
    """The library's matching rule over the shipped material JSON (header lines on
    RLRA_SetMaterialDatabaseJSON): lowercase category, count labels that are substrings of it,
    the material with the most matches wins. Unmatched categories get the material whose labels
    contain "default". Ties are resolved by file order and flagged; the category actually passed
    to the library is a canonical one that the same rule maps to the chosen material uniquely, so
    the library cannot break the tie differently."""

    def __init__(self, path: Path):
        self.path = Path(path)
        data = json.loads(self.path.read_text())
        self.materials = data["materials"]
        self.names = [m["name"] for m in self.materials]
        self.default_index = next(
            i for i, m in enumerate(self.materials) if "default" in [s.lower() for s in m["labels"]]
        )
        self.canonical: Dict[int, str] = {}
        for i, m in enumerate(self.materials):
            for lab in m["labels"]:
                counts = self._counts(lab)
                best = max(counts)
                if best > 0 and counts[i] == best and counts.count(best) == 1:
                    self.canonical[i] = lab.lower()
                    break
        self._cache: Dict[str, MaterialMatch] = {}

    def _counts(self, category: str) -> List[int]:
        c = category.lower()
        return [sum(1 for lab in m["labels"] if lab.lower() in c) for m in self.materials]

    def match(self, label: str) -> MaterialMatch:
        if label in self._cache:
            return self._cache[label]
        counts = self._counts(label)
        best = max(counts)
        if best == 0:
            idx, tie = self.default_index, False
        else:
            winners = [i for i, n in enumerate(counts) if n == best]
            idx, tie = winners[0], len(winners) > 1
        cat = self.canonical.get(idx)
        if cat is None:
            raise ValueError(f"no unique canonical category for material {self.names[idx]!r}")
        mm = MaterialMatch(label, cat, self.names[idx], best, tie)
        self._cache[label] = mm
        return mm


# ----------------------------------------------------------------------------- mesh builder
@dataclass
class Part:
    kind: str  # wall | floor | ceiling | box
    ident: str
    label: str
    face_start: int
    face_end: int
    closed_expected: bool
    info: dict = field(default_factory=dict)


@dataclass
class SceneMesh:
    vertices: np.ndarray  # (N,3) float64, world frame
    faces: np.ndarray  # (M,3) uint32
    face_category: List[str]  # category passed to the library, per face
    face_material: List[str]  # material name the rule picks, per face
    parts: List[Part]
    material_rows: List[MaterialMatch]
    report: dict

    def faces_by_category(self) -> Dict[str, np.ndarray]:
        out: Dict[str, List[int]] = {}
        for i, c in enumerate(self.face_category):
            out.setdefault(c, []).append(i)
        return {c: self.faces[np.array(ix)] for c, ix in out.items()}


class _Builder:
    def __init__(self):
        self.v: List[np.ndarray] = []
        self.f: List[Tuple[int, int, int]] = []
        self.cat: List[str] = []
        self.mat: List[str] = []

    def nface(self) -> int:
        return len(self.f)

    def _vid(self, p) -> int:
        self.v.append(np.asarray(p, dtype=np.float64))
        return len(self.v) - 1

    def tri(self, a, b, c, mm: MaterialMatch, outward: Optional[np.ndarray] = None):
        a, b, c = (np.asarray(x, dtype=np.float64) for x in (a, b, c))
        if outward is not None:
            n = np.cross(b - a, c - a)
            if float(np.dot(n, outward)) < 0:
                b, c = c, b
        ia, ib, ic = self._vid(a), self._vid(b), self._vid(c)
        self.f.append((ia, ib, ic))
        self.cat.append(mm.category_passed)
        self.mat.append(mm.material_name)

    def quad(self, a, b, c, d, mm: MaterialMatch, outward: np.ndarray):
        """Quad a-b-c-d in cyclic order, split along a-c."""
        self.tri(a, b, c, mm, outward)
        self.tri(a, c, d, mm, outward)


def _dedupe_vertices(verts: np.ndarray, faces: np.ndarray, decimals: int = 6):
    key = np.round(verts, decimals)
    uniq, inverse = np.unique(key, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    # keep the first original coordinate of each unique key
    first = np.full(len(uniq), -1, dtype=np.int64)
    for i, u in enumerate(inverse):
        if first[u] < 0:
            first[u] = i
    return verts[first], inverse[faces]


# ---- polygon triangulation (ear clipping) for room floors and ceilings
def _clean_polygon(poly: np.ndarray) -> np.ndarray:
    pts = [p for p in np.asarray(poly, dtype=np.float64)]
    out = []
    for p in pts:
        if not out or np.linalg.norm(p - out[-1]) > 1e-7:
            out.append(p)
    if len(out) > 1 and np.linalg.norm(out[0] - out[-1]) <= 1e-7:
        out.pop()
    changed = True
    while changed and len(out) > 3:
        changed = False
        for i in range(len(out)):
            a, b, c = out[i - 1], out[i], out[(i + 1) % len(out)]
            cr = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
            if abs(cr) < 1e-10:
                out.pop(i)
                changed = True
                break
    return np.array(out)


def _signed_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def _point_in_tri(p, a, b, c) -> bool:
    def s(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    d1, d2, d3 = s(p, a, b), s(p, b, c), s(p, c, a)
    neg = (d1 < -EPS) or (d2 < -EPS) or (d3 < -EPS)
    pos = (d1 > EPS) or (d2 > EPS) or (d3 > EPS)
    return not (neg and pos)


def triangulate_polygon(poly: np.ndarray) -> List[Tuple[int, int, int]]:
    """Ear clipping of a simple polygon (counter-clockwise after cleaning). Returns index triples
    into the cleaned polygon, each triangle counter-clockwise (normal +z)."""
    n = len(poly)
    idx = list(range(n))
    tris: List[Tuple[int, int, int]] = []
    guard = 0
    while len(idx) > 3 and guard < 10 * n * n:
        guard += 1
        clipped = False
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = poly[i0], poly[i1], poly[i2]
            cr = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if cr <= EPS:
                continue  # reflex or degenerate
            if any(_point_in_tri(poly[j], a, b, c) for j in idx if j not in (i0, i1, i2)):
                continue
            tris.append((i0, i1, i2))
            idx.pop(k)
            clipped = True
            break
        if not clipped:
            raise ValueError("ear clipping failed (polygon not simple?)")
    if len(idx) == 3:
        tris.append((idx[0], idx[1], idx[2]))
    return tris


# ---- walls with holes
def assign_holes_to_walls(walls: Sequence[dict], holes: Sequence[dict]):
    """Return (per-wall list of (u0,u1,z0,z1,type,hole_index), unassigned hole indices, notes)."""
    per_wall: List[list] = [[] for _ in walls]
    unassigned, notes = [], []
    for hi, h in enumerate(holes):
        prof = np.asarray(h["profile"], dtype=np.float64)
        if prof.shape[1] != 3:
            notes.append(f"hole {hi}: profile is not 3D")
            unassigned.append(hi)
            continue
        if len(prof) != 4:
            notes.append(f"hole {hi}: {len(prof)} profile points, cut as its bounding rectangle")
        c = prof[:, :2].mean(axis=0)
        best = None
        for wi, w in enumerate(walls):
            p0, p1 = (np.asarray(p, dtype=np.float64) for p in w["location"])
            L = float(np.linalg.norm(p1 - p0))
            if L < 1e-9:
                continue
            u = (p1 - p0) / L
            nrm = np.array([-u[1], u[0]])
            d = abs(float(np.dot(c - p0, nrm)))
            tol = 0.5 * float(w["thickness"]) + 0.5 * float(h.get("thickness", 0.0)) + 0.01
            s = (prof[:, :2] - p0) @ u
            u0, u1 = float(s.min()), float(s.max())
            if d <= tol and u1 > 0.0 and u0 < L:
                if best is None or d < best[0]:
                    best = (d, wi, max(u0, 0.0), min(u1, L))
        if best is None:
            unassigned.append(hi)
            continue
        _, wi, u0, u1 = best
        z0, z1 = float(prof[:, 2].min()), float(prof[:, 2].max())
        per_wall[wi].append((u0, u1, z0, z1, str(h.get("type", "")), hi))
    return per_wall, unassigned, notes


def _add_wall(b: _Builder, w: dict, holes: list, mm: MaterialMatch) -> dict:
    p0, p1 = (np.asarray(p, dtype=np.float64) for p in w["location"])
    L = float(np.linalg.norm(p1 - p0))
    t = float(w["thickness"])
    H = float(w["height"])
    u = (p1 - p0) / L
    nrm = np.array([-u[1], u[0]])

    def P(s, z, side):  # side in {-1,+1}: which face of the slab
        xy = p0 + u * s + nrm * (0.5 * t * side)
        return np.array([xy[0], xy[1], z])

    rects = []
    for (u0, u1, z0, z1, _typ, _hi) in holes:
        z0c, z1c = max(z0, 0.0), min(z1, H)
        if u1 - u0 > 1e-6 and z1c - z0c > 1e-6:
            rects.append((u0, u1, z0c, z1c))
    ub = sorted(set([0.0, L] + [r[0] for r in rects] + [r[1] for r in rects]))
    zb = sorted(set([0.0, H] + [r[2] for r in rects] + [r[3] for r in rects]))
    ub = [x for i, x in enumerate(ub) if i == 0 or x - ub[i - 1] > 1e-7]
    zb = [x for i, x in enumerate(zb) if i == 0 or x - zb[i - 1] > 1e-7]
    nu, nz = len(ub) - 1, len(zb) - 1
    solid = np.ones((nu, nz), dtype=bool)
    for i in range(nu):
        for j in range(nz):
            cu, cz = 0.5 * (ub[i] + ub[i + 1]), 0.5 * (zb[j] + zb[j + 1])
            for (u0, u1, z0, z1) in rects:
                if u0 < cu < u1 and z0 < cz < z1:
                    solid[i, j] = False
    n3 = np.array([nrm[0], nrm[1], 0.0])
    u3 = np.array([u[0], u[1], 0.0])
    z3 = np.array([0.0, 0.0, 1.0])
    for i in range(nu):
        for j in range(nz):
            if not solid[i, j]:
                continue
            ua, ubb, za, zbb = ub[i], ub[i + 1], zb[j], zb[j + 1]
            # front (+n) and back (-n)
            b.quad(P(ua, za, 1), P(ubb, za, 1), P(ubb, zbb, 1), P(ua, zbb, 1), mm, n3)
            b.quad(P(ua, za, -1), P(ubb, za, -1), P(ubb, zbb, -1), P(ua, zbb, -1), mm, -n3)
            # sides where the neighbour is outside the wall or a hole
            if i == 0 or not solid[i - 1, j]:
                b.quad(P(ua, za, -1), P(ua, za, 1), P(ua, zbb, 1), P(ua, zbb, -1), mm, -u3)
            if i == nu - 1 or not solid[i + 1, j]:
                b.quad(P(ubb, za, -1), P(ubb, za, 1), P(ubb, zbb, 1), P(ubb, zbb, -1), mm, u3)
            if j == 0 or not solid[i, j - 1]:
                b.quad(P(ua, za, -1), P(ubb, za, -1), P(ubb, za, 1), P(ua, za, 1), mm, -z3)
            if j == nz - 1 or not solid[i, j + 1]:
                b.quad(P(ua, zbb, -1), P(ubb, zbb, -1), P(ubb, zbb, 1), P(ua, zbb, 1), mm, z3)
    return {"length_m": L, "thickness_m": t, "height_m": H, "n_holes_cut": len(rects),
            "grid_cells": int(nu * nz), "solid_cells": int(solid.sum())}


# ---- boxes
def box_corners(entry: dict) -> np.ndarray:
    return np.array([[c["x"], c["y"], c["z"]] for c in entry["bounding_box"]], dtype=np.float64)


def _box_faces_from_corners(c: np.ndarray) -> Optional[List[Tuple[int, int, int, int]]]:
    """InteriorGS boxes list 4 bottom corners in cyclic order, then the 4 top corners in the same
    order. Returns the 6 quads, or None if the corners do not have that layout."""
    if c.shape != (8, 3):
        return None
    if not (np.allclose(c[:4, 2], c[0, 2], atol=1e-4) and np.allclose(c[4:, 2], c[4, 2], atol=1e-4)):
        return None
    if not np.allclose(c[:4, :2], c[4:, :2], atol=1e-4):
        return None
    q = c[:4, :2]
    crosses = []
    for k in range(4):
        a, bb, cc = q[k - 1], q[k], q[(k + 1) % 4]
        crosses.append((bb[0] - a[0]) * (cc[1] - bb[1]) - (bb[1] - a[1]) * (cc[0] - bb[0]))
    if not (all(x > 0 for x in crosses) or all(x < 0 for x in crosses)):
        return None
    faces = [(0, 1, 2, 3), (4, 5, 6, 7)]
    for k in range(4):
        faces.append((k, (k + 1) % 4, (k + 1) % 4 + 4, k + 4))
    return faces


def _add_box(b: _Builder, corners: np.ndarray, mm: MaterialMatch) -> dict:
    faces = _box_faces_from_corners(corners)
    used_aabb = False
    if faces is None:
        lo, hi = corners.min(axis=0), corners.max(axis=0)
        corners = np.array([[lo[0], lo[1], lo[2]], [lo[0], hi[1], lo[2]], [hi[0], hi[1], lo[2]],
                            [hi[0], lo[1], lo[2]], [lo[0], lo[1], hi[2]], [lo[0], hi[1], hi[2]],
                            [hi[0], hi[1], hi[2]], [hi[0], lo[1], hi[2]]])
        faces = _box_faces_from_corners(corners)
        used_aabb = True
    centre = corners.mean(axis=0)
    for quad in faces:
        pts = [corners[k] for k in quad]
        fc = np.mean(pts, axis=0)
        b.quad(*pts, mm, fc - centre)
    return {"used_axis_aligned_fallback": used_aabb}


# ---- whole scene
def load_scene_files(scene_dir: Path) -> Tuple[dict, list]:
    scene_dir = Path(scene_dir)
    structure = json.loads((scene_dir / "structure.json").read_text())
    labels = json.loads((scene_dir / "labels.json").read_text())
    return structure, labels


def build_scene_mesh(
    scene_dir: Path,
    material_config: Path,
    gs_to_world: Callable[[np.ndarray], np.ndarray],
    exclude_ins_ids: Sequence[str] = (),
) -> SceneMesh:
    structure, labels = load_scene_files(scene_dir)
    mt = MaterialTable(material_config)
    b = _Builder()
    parts: List[Part] = []
    walls = structure.get("walls", [])
    holes = structure.get("holes", [])
    rooms = structure.get("rooms", [])
    per_wall, unassigned, hole_notes = assign_holes_to_walls(walls, holes)

    mm_wall = mt.match(WALL_CATEGORY)
    for wi, w in enumerate(walls):
        s = b.nface()
        info = _add_wall(b, w, per_wall[wi], mm_wall)
        parts.append(Part("wall", str(wi), WALL_CATEGORY, s, b.nface(), True, info))

    ceiling_z = max((float(w["height"]) for w in walls), default=0.0)
    mm_floor, mm_ceil = mt.match(FLOOR_CATEGORY), mt.match(CEILING_CATEGORY)
    for ri, r in enumerate(rooms):
        poly = _clean_polygon(np.asarray(r["profile"], dtype=np.float64)[:, :2])
        if _signed_area(poly) < 0:
            poly = poly[::-1]
        tris = triangulate_polygon(poly)
        up = np.array([0.0, 0.0, 1.0])
        s = b.nface()
        for (i, j, k) in tris:
            b.tri([*poly[i], 0.0], [*poly[j], 0.0], [*poly[k], 0.0], mm_floor, up)
        parts.append(Part("floor", str(ri), FLOOR_CATEGORY, s, b.nface(), False,
                          {"area_m2": _signed_area(poly), "room_type": r.get("room_type")}))
        s = b.nface()
        for (i, j, k) in tris:
            b.tri([*poly[i], ceiling_z], [*poly[j], ceiling_z], [*poly[k], ceiling_z], mm_ceil, -up)
        parts.append(Part("ceiling", str(ri), CEILING_CATEGORY, s, b.nface(), False,
                          {"z_m": ceiling_z}))

    excluded = {str(x) for x in exclude_ins_ids}
    skipped = {"structural": 0, "no_bounding_box": 0, "excluded_by_caller": 0, "degenerate": 0}
    n_fallback = 0
    for e in labels:
        lab = str(e.get("label", ""))
        if "bounding_box" not in e:
            skipped["no_bounding_box"] += 1
            continue
        if lab.strip().lower() in STRUCTURAL_LABELS:
            skipped["structural"] += 1
            continue
        if str(e.get("ins_id")) in excluded:
            skipped["excluded_by_caller"] += 1
            continue
        corners = box_corners(e)
        ext = corners.max(axis=0) - corners.min(axis=0)
        if float(ext.min()) < MIN_BOX_EDGE_M:
            skipped["degenerate"] += 1
            continue
        mm = mt.match(lab)
        s = b.nface()
        info = _add_box(b, corners, mm)
        n_fallback += int(info["used_axis_aligned_fallback"])
        parts.append(Part("box", str(e.get("ins_id")), lab, s, b.nface(), True, info))

    verts_gs = np.array(b.v, dtype=np.float64)
    faces = np.array(b.f, dtype=np.int64)
    verts_gs, faces = _dedupe_vertices(verts_gs, faces)
    verts_w = gs_to_world(verts_gs)
    # Keep outward winding if the GS->world map is a reflection (det of its linear part < 0).
    probe = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    pw = gs_to_world(probe)
    lin = np.stack([pw[1] - pw[0], pw[2] - pw[0], pw[3] - pw[0]], axis=1)
    det = float(np.linalg.det(lin))
    if det < 0:
        faces = faces[:, [0, 2, 1]]

    rows = sorted(mt._cache.values(), key=lambda m: m.source_label.lower())
    report = {
        "n_walls": len(walls), "n_rooms": len(rooms), "n_holes": len(holes),
        "holes_unassigned": unassigned, "hole_notes": hole_notes,
        "holes_per_type": _count([str(h.get("type", "")) for h in holes]),
        "ceiling_z_m": ceiling_z, "labels_total": len(labels), "labels_skipped": skipped,
        "boxes_axis_aligned_fallback": n_fallback,
        "gs_to_world_linear_part": lin.tolist(), "gs_to_world_det": det,
        "n_vertices": int(len(verts_w)), "n_faces": int(len(faces)),
        "structural_labels_skipped_as_boxes": list(STRUCTURAL_LABELS),
    }
    return SceneMesh(verts_w, faces.astype(np.uint32), b.cat, b.mat, parts, rows, report)


def _count(xs):
    out: Dict[str, int] = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return out


def part_is_closed(mesh: SceneMesh, part: Part) -> bool:
    """Every undirected edge of the part's triangles is used exactly twice."""
    f = mesh.faces[part.face_start:part.face_end].astype(np.int64)
    if len(f) == 0:
        return False
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    e.sort(axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    return bool(np.all(counts == 2))


def boxes_containing(point_gs, labels: list, skip_ins_ids=()) -> List[str]:
    """ins_ids of labels.json boxes (with a bounding_box, not structural) whose box contains the point
    (GS frame). Uses the box's own corner layout; boxes whose corners do not have that layout are
    tested as axis-aligned boxes."""
    p = np.asarray(point_gs, dtype=np.float64)
    hits = []
    skip = {str(x) for x in skip_ins_ids}
    for e in labels:
        if "bounding_box" not in e or str(e.get("ins_id")) in skip:
            continue
        if str(e.get("label", "")).strip().lower() in STRUCTURAL_LABELS:
            continue
        c = box_corners(e)
        faces = _box_faces_from_corners(c)
        if faces is None:
            if np.all(p >= c.min(0)) and np.all(p <= c.max(0)):
                hits.append(str(e.get("ins_id")))
            continue
        centre = c.mean(axis=0)
        inside = True
        for quad in faces:
            a, b_, c_ = c[quad[0]], c[quad[1]], c[quad[2]]
            n = np.cross(b_ - a, c_ - a)
            if np.dot(n, centre - a) > 0:
                n = -n
            if np.dot(n, p - a) > 1e-9:
                inside = False
                break
        if inside:
            hits.append(str(e.get("ins_id")))
    return hits


def write_material_map_csv(mesh: SceneMesh, path: Path) -> None:
    lines = ["source_label,category_passed_to_library,material_name,n_label_matches,tie"]
    for m in mesh.material_rows:
        lab = m.source_label.replace('"', "'")
        lines.append(f'"{lab}",{m.category_passed},"{m.material_name}",{m.n_matches},{m.tie}')
    Path(path).write_text("\n".join(lines) + "\n")
