"""ctypes binding to libRLRAudioPropagation.so, written from the C header of the pinned copy
(rlr-audio-propagation 4fd446b, RLRAudioPropagationPkg/headers/RLRAudioPropagation.h).

Covers only what check 1 needs: context with an explicit configuration, material database, one
triangle-mesh object with a material category per face group, sources, mono and binaural listeners,
simulate, read impulse responses, write WAV and metrics, and ray queries for the line-of-sight test.

Struct layouts follow the header exactly: every struct is inside "#pragma pack(push,1)", so every
ctypes Structure here has _pack_ = 1. C enums are passed as 32-bit ints (RLRA_Error,
RLRA_ChannelLayoutType, RLRA_RayHit). RLR_Bool is uint8_t. size_t is ctypes.c_size_t.
Header line numbers are given next to each declaration.
"""
from __future__ import annotations

import ctypes
import time
from ctypes import POINTER, c_char_p, c_float, c_int, c_size_t, c_uint8, c_uint32, c_void_p
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

# ---- header lines 46 to 58: RLRA_Error
RLRA_ERRORS = {
    0: "RLRA_Success", 2000: "RLRA_Error_Unknown", 2001: "RLRA_Error_InvalidParam",
    2002: "RLRA_Error_BadSampleRate", 2003: "RLRA_Error_MissingDLL", 2004: "RLRA_Error_BadAlignment",
    2005: "RLRA_Error_Uninitialized", 2018: "RLRA_Error_BadAlloc", 2019: "RLRA_Error_UnsupportedFeature",
    2099: "RLRA_Error_InternalEnd",
}
# ---- header lines 61 to 75: RLRA_ChannelLayoutType
RLRA_ChannelLayoutType_Mono = 1
RLRA_ChannelLayoutType_Binaural = 3


class RLRA_ChannelLayout(ctypes.Structure):  # header lines 78 to 91, pack 1
    _pack_ = 1
    _fields_ = [("channelCount", c_size_t), ("type", c_int)]


class RLRA_ContextConfiguration(ctypes.Structure):  # header lines 101 to 164, pack 1
    _pack_ = 1
    _fields_ = [
        ("thisSize", c_size_t),
        ("frequencyBands", c_size_t),
        ("directSHOrder", c_size_t),
        ("indirectSHOrder", c_size_t),
        ("directRayCount", c_size_t),
        ("indirectRayCount", c_size_t),
        ("indirectRayDepth", c_size_t),
        ("sourceRayCount", c_size_t),
        ("sourceRayDepth", c_size_t),
        ("maxDiffractionOrder", c_size_t),
        ("threadCount", c_size_t),
        ("sampleRate", c_float),
        ("maxIRLength", c_float),
        ("unitScale", c_float),
        ("globalVolume", c_float),
        ("hrtfRight", c_float * 3),
        ("hrtfUp", c_float * 3),
        ("hrtfBack", c_float * 3),
        ("direct", c_uint8),
        ("indirect", c_uint8),
        ("diffraction", c_uint8),
        ("transmission", c_uint8),
        ("meshSimplification", c_uint8),
        ("temporalCoherence", c_uint8),
    ]


class RLRA_Ray(ctypes.Structure):  # header lines 201 to 231, pack 1
    _pack_ = 1
    _fields_ = [
        ("origin", c_float * 3),
        ("direction", c_float * 3),
        ("tMin", c_float),
        ("tMax", c_float),
        ("hit", c_int),
        ("normal", c_float * 3),
    ]


RLRA_Context = c_void_p  # header: typedef struct RLRA_Context_* RLRA_Context
EXPECTED_SIZES = {"RLRA_ChannelLayout": 12, "RLRA_ContextConfiguration": 146, "RLRA_Ray": 48}

_BOOL_FIELDS = {"direct", "indirect", "diffraction", "transmission", "meshSimplification", "temporalCoherence"}
_VEC_FIELDS = {"hrtfRight", "hrtfUp", "hrtfBack"}


class RLRAError(RuntimeError):
    pass


def struct_sizes() -> Dict[str, int]:
    return {
        "RLRA_ChannelLayout": ctypes.sizeof(RLRA_ChannelLayout),
        "RLRA_ContextConfiguration": ctypes.sizeof(RLRA_ContextConfiguration),
        "RLRA_Ray": ctypes.sizeof(RLRA_Ray),
    }


def config_to_dict(cfg: RLRA_ContextConfiguration) -> Dict:
    out = {}
    for name, _ in cfg._fields_:
        v = getattr(cfg, name)
        if name in _VEC_FIELDS:
            v = [float(x) for x in v]
        elif name in _BOOL_FIELDS:
            v = bool(v)
        elif isinstance(v, float):
            v = float(v)
        else:
            v = int(v)
        out[name] = v
    return out


def load_library(path: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(path))
    E = c_int  # RLRA_Error
    sigs = {
        # name: (restype, argtypes)                                      header declaration
        "RLRA_ContextConfigurationDefault": (E, [POINTER(RLRA_ContextConfiguration)]),
        "RLRA_CreateContext": (E, [POINTER(RLRA_Context), POINTER(RLRA_ContextConfiguration)]),
        "RLRA_DestroyContext": (E, [RLRA_Context]),
        "RLRA_AddSource": (E, [RLRA_Context]),
        "RLRA_ClearSources": (E, [RLRA_Context]),
        "RLRA_GetSourceCount": (c_size_t, [RLRA_Context]),
        "RLRA_SetSourcePosition": (E, [RLRA_Context, c_size_t, POINTER(c_float)]),
        "RLRA_AddListener": (E, [RLRA_Context, POINTER(RLRA_ChannelLayout)]),
        "RLRA_ClearListeners": (E, [RLRA_Context]),
        "RLRA_GetListenerCount": (c_size_t, [RLRA_Context]),
        "RLRA_SetListenerPosition": (E, [RLRA_Context, c_size_t, POINTER(c_float)]),
        "RLRA_SetListenerHRTF": (E, [RLRA_Context, c_size_t, c_char_p]),  # header line 342
        "RLRA_SetListenerOrientationQuaternion": (E, [RLRA_Context, c_size_t, POINTER(c_float)]),
        "RLRA_AddObject": (E, [RLRA_Context]),
        "RLRA_GetObjectCount": (c_size_t, [RLRA_Context]),
        "RLRA_SetObjectPosition": (E, [RLRA_Context, c_size_t, POINTER(c_float)]),
        "RLRA_SetObjectOrientationQuaternion": (E, [RLRA_Context, c_size_t, POINTER(c_float)]),
        "RLRA_AddMeshVertices": (E, [RLRA_Context, POINTER(c_float), c_size_t]),
        "RLRA_AddMeshIndices": (E, [RLRA_Context, POINTER(c_uint32), c_size_t, c_size_t, c_char_p]),
        "RLRA_FinalizeObjectMesh": (E, [RLRA_Context, c_size_t]),
        "RLRA_SetMaterialDatabaseJSON": (E, [RLRA_Context, c_char_p]),
        "RLRA_WriteSceneMeshOBJ": (E, [RLRA_Context, c_char_p]),
        "RLRA_Simulate": (E, [RLRA_Context]),
        "RLRA_GetIRCount": (c_size_t, [RLRA_Context]),
        "RLRA_GetIRChannelCount": (c_size_t, [RLRA_Context, c_size_t, c_size_t]),
        "RLRA_GetIRSampleCount": (c_size_t, [RLRA_Context, c_size_t, c_size_t]),
        "RLRA_GetIRChannel": (POINTER(c_float), [RLRA_Context, c_size_t, c_size_t, c_size_t]),
        "RLRA_WriteIRWave": (E, [RLRA_Context, c_size_t, c_size_t, c_char_p]),
        "RLRA_WriteIRMetrics": (E, [RLRA_Context, c_size_t, c_size_t, c_char_p]),
        "RLRA_GetIndirectRayEfficiency": (c_float, [RLRA_Context]),
        "RLRA_TraceRayAnyHit": (E, [RLRA_Context, POINTER(RLRA_Ray)]),
        "RLRA_TraceRayFirstHit": (E, [RLRA_Context, POINTER(RLRA_Ray)]),
    }
    for name, (res, args) in sigs.items():
        fn = getattr(lib, name)
        fn.restype = res
        fn.argtypes = args
    return lib


def _f3(v) -> ctypes.Array:
    a = (c_float * 3)(*[float(x) for x in v])
    return a


def _check(code: int, what: str) -> None:
    if code != 0:
        raise RLRAError(f"{what} returned {code} ({RLRA_ERRORS.get(code, 'unlisted code')})")


class RLRContext:
    """One RLR context. overrides: dict of RLRA_ContextConfiguration field -> value, applied on top
    of RLRA_ContextConfigurationDefault(). Every field set is recorded in self.set_fields, and the
    complete configuration (defaults plus overrides) in self.config_used."""

    def __init__(self, lib: ctypes.CDLL, overrides: Dict, log=print):
        self.lib = lib
        self.log = log
        cfg = RLRA_ContextConfiguration()
        cfg.thisSize = ctypes.sizeof(RLRA_ContextConfiguration)
        _check(lib.RLRA_ContextConfigurationDefault(ctypes.byref(cfg)), "RLRA_ContextConfigurationDefault")
        self.config_defaults = config_to_dict(cfg)
        names = {n for n, _ in RLRA_ContextConfiguration._fields_}
        self.set_fields = {}
        for k, v in overrides.items():
            if k not in names or k == "thisSize":
                raise KeyError(f"unknown or reserved configuration field {k!r}")
            if k in _VEC_FIELDS:
                setattr(cfg, k, _f3(v))
            elif k in _BOOL_FIELDS:
                setattr(cfg, k, 1 if bool(v) else 0)
            else:
                setattr(cfg, k, v)
            self.set_fields[k] = v
            log(f"[rlr] set {k} = {v!r} (library default {self.config_defaults[k]!r})")
        self.config_used = config_to_dict(cfg)
        log("[rlr] configuration passed to RLRA_CreateContext: " + ", ".join(f"{k}={v}" for k, v in self.config_used.items()))
        self.ctx = RLRA_Context()
        _check(lib.RLRA_CreateContext(ctypes.byref(self.ctx), ctypes.byref(cfg)), "RLRA_CreateContext")

    # -- lifetime
    def close(self):
        if self.ctx:
            _check(self.lib.RLRA_DestroyContext(self.ctx), "RLRA_DestroyContext")
            self.ctx = RLRA_Context()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- scene
    def set_material_database(self, path: Path):
        _check(self.lib.RLRA_SetMaterialDatabaseJSON(self.ctx, str(path).encode()), "RLRA_SetMaterialDatabaseJSON")

    def add_mesh_object(self, vertices: np.ndarray, faces: np.ndarray, face_category: Sequence[str]) -> int:
        """One object at the origin with identity orientation; faces grouped by category, each group
        passed with RLRA_AddMeshIndices(verticesPerFace=3)."""
        _check(self.lib.RLRA_AddObject(self.ctx), "RLRA_AddObject")
        idx = int(self.lib.RLRA_GetObjectCount(self.ctx)) - 1
        _check(self.lib.RLRA_SetObjectPosition(self.ctx, idx, _f3([0, 0, 0])), "RLRA_SetObjectPosition")
        q = (c_float * 4)(1.0, 0.0, 0.0, 0.0)
        _check(self.lib.RLRA_SetObjectOrientationQuaternion(self.ctx, idx, q), "RLRA_SetObjectOrientationQuaternion")
        v = np.ascontiguousarray(vertices, dtype=np.float32).reshape(-1)
        _check(self.lib.RLRA_AddMeshVertices(self.ctx, v.ctypes.data_as(POINTER(c_float)), len(v) // 3),
               "RLRA_AddMeshVertices")
        cats: Dict[str, List[int]] = {}
        for i, c in enumerate(face_category):
            cats.setdefault(c, []).append(i)
        self._keep = [v]
        for c, ix in cats.items():
            ind = np.ascontiguousarray(np.asarray(faces)[ix], dtype=np.uint32).reshape(-1)
            self._keep.append(ind)
            _check(self.lib.RLRA_AddMeshIndices(self.ctx, ind.ctypes.data_as(POINTER(c_uint32)), len(ind), 3,
                                                c.encode()), f"RLRA_AddMeshIndices({c})")
        _check(self.lib.RLRA_FinalizeObjectMesh(self.ctx, idx), "RLRA_FinalizeObjectMesh")
        return idx

    def write_scene_obj(self, path: Path):
        _check(self.lib.RLRA_WriteSceneMeshOBJ(self.ctx, str(path).encode()), "RLRA_WriteSceneMeshOBJ")

    def add_source(self, position) -> int:
        _check(self.lib.RLRA_AddSource(self.ctx), "RLRA_AddSource")
        idx = int(self.lib.RLRA_GetSourceCount(self.ctx)) - 1
        _check(self.lib.RLRA_SetSourcePosition(self.ctx, idx, _f3(position)), "RLRA_SetSourcePosition")
        return idx

    def clear_sources(self):
        _check(self.lib.RLRA_ClearSources(self.ctx), "RLRA_ClearSources")

    def add_listener(self, layout: str, position, quat_wxyz) -> int:
        if layout == "mono":
            cl = RLRA_ChannelLayout(1, RLRA_ChannelLayoutType_Mono)
        elif layout == "binaural":
            cl = RLRA_ChannelLayout(2, RLRA_ChannelLayoutType_Binaural)
        else:
            raise ValueError(layout)
        _check(self.lib.RLRA_AddListener(self.ctx, ctypes.byref(cl)), "RLRA_AddListener")
        idx = int(self.lib.RLRA_GetListenerCount(self.ctx)) - 1
        _check(self.lib.RLRA_SetListenerPosition(self.ctx, idx, _f3(position)), "RLRA_SetListenerPosition")
        q = (c_float * 4)(*[float(x) for x in quat_wxyz])
        _check(self.lib.RLRA_SetListenerOrientationQuaternion(self.ctx, idx, q),
               "RLRA_SetListenerOrientationQuaternion")
        return idx

    def set_listener_hrtf(self, listener: int, path) -> None:
        """RLRA_SetListenerHRTF (header line 342): call after add_listener; a ".sofa" path selects an AES SOFA
        file. Not called anywhere unless a caller asks (None or "" would switch back to the default HRTF and is
        refused here so a default-HRTF run never touches this function)."""
        if path is None or str(path) == "":
            raise ValueError("set_listener_hrtf needs a path; do not call it for the default HRTF")
        self.log(f"[rlr] RLRA_SetListenerHRTF(listener {listener}, {str(path)!r})")
        _check(self.lib.RLRA_SetListenerHRTF(self.ctx, listener, str(path).encode()), "RLRA_SetListenerHRTF")

    def clear_listeners(self):
        _check(self.lib.RLRA_ClearListeners(self.ctx), "RLRA_ClearListeners")

    # -- simulation and output
    def simulate(self) -> float:
        t0 = time.perf_counter()
        _check(self.lib.RLRA_Simulate(self.ctx), "RLRA_Simulate")
        return time.perf_counter() - t0

    def get_ir(self, listener: int, source: int) -> np.ndarray:
        nch = int(self.lib.RLRA_GetIRChannelCount(self.ctx, listener, source))
        ns = int(self.lib.RLRA_GetIRSampleCount(self.ctx, listener, source))
        out = np.zeros((nch, ns), dtype=np.float32)
        for ch in range(nch):
            ptr = self.lib.RLRA_GetIRChannel(self.ctx, listener, source, ch)
            if not ptr:
                raise RLRAError(f"RLRA_GetIRChannel returned NULL (listener {listener}, source {source}, ch {ch})")
            if ns:
                out[ch] = np.ctypeslib.as_array(ptr, shape=(ns,))
        return out

    def ir_count(self) -> int:
        return int(self.lib.RLRA_GetIRCount(self.ctx))

    def write_ir_wave(self, listener: int, source: int, path: Path):
        _check(self.lib.RLRA_WriteIRWave(self.ctx, listener, source, str(path).encode()), "RLRA_WriteIRWave")

    def write_ir_metrics(self, listener: int, source: int, path: Path):
        _check(self.lib.RLRA_WriteIRMetrics(self.ctx, listener, source, str(path).encode()), "RLRA_WriteIRMetrics")

    def indirect_ray_efficiency(self) -> float:
        return float(self.lib.RLRA_GetIndirectRayEfficiency(self.ctx))

    def segment_blocked(self, a, b, end_margin_m: float = 0.0) -> bool:
        """True if any geometry lies on the straight segment a->b (header: call after RLRA_Simulate).
        Direction is b-a, so tMin/tMax are fractions of the segment; end_margin_m trims both ends."""
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        d = b - a
        L = float(np.linalg.norm(d))
        if L <= 2 * end_margin_m:
            return False
        r = RLRA_Ray()
        r.origin = _f3(a)
        r.direction = _f3(d)
        r.tMin = end_margin_m / L
        r.tMax = 1.0 - end_margin_m / L
        _check(self.lib.RLRA_TraceRayAnyHit(self.ctx, ctypes.byref(r)), "RLRA_TraceRayAnyHit")
        return int(r.hit) == 1
