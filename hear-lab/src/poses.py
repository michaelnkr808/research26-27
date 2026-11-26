"""Listener pose helpers (written for route A check 1; moved to hear-lab/src on 2026-11-26).

GN-Bench agent pose (GN_Bench_simulator.py, _load_initial_state from line 775 and _render_observations
from line 912, GN-Bench-Tools 0ce8367): position (start_world_x, start_world_y, 1.3) in the world frame,
rotation R.from_euler("z", heading_degrees), camera forward = rotation applied to local +x. So in the
world frame forward = (cos h, sin h, 0), up = +z, and (right-handed, z up, as the InteriorGS card says
for its frame) left = (-sin h, cos h, 0), right = (sin h, -cos h, 0).

RLR listener orientation (header, RLRA_SetListenerOrientationQuaternion): a quaternion [w, x, y, z]
rotating listener-local to world. The listener-local axes are set by the configuration fields
hrtfRight / hrtfUp / hrtfBack, library defaults (1,0,0), (0,1,0), (0,0,1). The quaternion below maps
those default local axes onto the agent's right, up and back, so the library's HRTF settings stay at
their defaults; listener_quat_wxyz() refuses other hrtf axes.
"""
from __future__ import annotations

import math

import numpy as np

DEFAULT_HRTF_AXES = {"hrtfRight": [1.0, 0.0, 0.0], "hrtfUp": [0.0, 1.0, 0.0], "hrtfBack": [0.0, 0.0, 1.0]}


def agent_axes(heading_deg: float):
    h = math.radians(heading_deg)
    fwd = np.array([math.cos(h), math.sin(h), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up)  # = (sin h, -cos h, 0)
    left = -right
    return fwd, left, right, up


def matrix_to_quat_wxyz(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def quat_wxyz_to_matrix(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def listener_quat_wxyz(heading_deg: float, config_used: dict | None = None) -> np.ndarray:
    if config_used is not None:
        for k, v in DEFAULT_HRTF_AXES.items():
            if not np.allclose(config_used[k], v):
                raise ValueError(f"{k} is {config_used[k]}, not the default {v}; quaternion assumes defaults")
    fwd, _left, right, up = agent_axes(heading_deg)
    back = -fwd
    m = np.stack([right, up, back], axis=1)  # columns: images of local x (right), y (up), z (back)
    return matrix_to_quat_wxyz(m)


def axes_from_forward(forward) -> tuple:
    """Unit forward, left, right and up for a listener whose forward is the horizontal part of the given
    vector, in a right-handed z-up frame (the same rule as agent_axes: right = forward x up, left = -right).
    Used where the forward vector comes from a frame map (for example GN-Bench's world-to-GS transform of
    pos and pos + forward) instead of a heading angle."""
    f = np.asarray(forward, dtype=np.float64).reshape(3).copy()
    f[2] = 0.0
    n = float(np.linalg.norm(f))
    if n < 1e-9:
        raise ValueError("forward vector has no horizontal part")
    fwd = f / n
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up)
    return fwd, -right, right, up


def listener_quat_from_forward(forward, config_used: dict | None = None) -> np.ndarray:
    """RLR listener quaternion [w, x, y, z] for a horizontal forward vector: maps the library's default local
    axes (right +x, up +y, back +z) onto right, up (+z) and back (-forward). Equal to listener_quat_wxyz(h)
    when forward = (cos h, sin h, 0)."""
    if config_used is not None:
        for k, v in DEFAULT_HRTF_AXES.items():
            if not np.allclose(config_used[k], v):
                raise ValueError(f"{k} is {config_used[k]}, not the default {v}; quaternion assumes defaults")
    fwd, _left, right, up = axes_from_forward(forward)
    m = np.stack([right, up, -fwd], axis=1)
    return matrix_to_quat_wxyz(m)
