#!/usr/bin/env python3
"""
AxCore Aperiodic Manifold experiment (Einstein/Spectre-inspired).

This script is an end-to-end stress test that uses most exported AxCore DLL APIs.
It ingests deterministic, non-repeating aperiodic patch signals and evaluates:

- Grounding/ambiguity behavior in manifold scans
- Working/episodic memory behavior
- Candidate routing and geometric gap deduction
- Consolidation + post-sleep recall behavior
- Sequence bundle similarity
- Manifold projection for compact inspection

Note:
The signal generator is "Spectre-inspired" (deterministic aperiodic geometry proxy),
not an exact polygonal Spectre substitution engine.
"""

from __future__ import annotations

import argparse
import ctypes as ct
import math
import statistics
import sys
from array import array
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


AXCORE_MAX_KEY_LENGTH = 63
AXCORE_MAX_DATASET_TYPE_LENGTH = 31
AXCORE_MAX_DATASET_ID_LENGTH = 63
AXCORE_MAX_LABEL_LENGTH = 31
AXCORE_MAX_STRATEGY_LENGTH = 31
AXCORE_MAX_SOURCE_LENGTH = 15

AX_STATUS_OK = 0
STATUS_NAMES = {
    0: "AX_STATUS_OK",
    1: "AX_STATUS_INVALID_ARGUMENT",
    2: "AX_STATUS_DIMENSION_MISMATCH",
    3: "AX_STATUS_OUT_OF_MEMORY",
    4: "AX_STATUS_BUFFER_TOO_SMALL",
    5: "AX_STATUS_NOT_FOUND",
    6: "AX_STATUS_STATE_ERROR",
    7: "AX_STATUS_LIMIT_EXCEEDED",
}


class AxVersion(ct.Structure):
    _fields_ = [
        ("major", ct.c_uint32),
        ("minor", ct.c_uint32),
        ("patch", ct.c_uint32),
        ("abi", ct.c_uint32),
    ]


class AxCoreCreateInfo(ct.Structure):
    _fields_ = [
        ("hdc_dim", ct.c_uint32),
        ("working_memory_capacity", ct.c_uint32),
        ("episodic_recent_limit", ct.c_uint32),
        ("episodic_max_levels", ct.c_uint32),
        ("arena_bytes", ct.c_uint32),
    ]


class AxSignalProfile(ct.Structure):
    _fields_ = [
        ("length", ct.c_uint32),
        ("mean", ct.c_float),
        ("standard_deviation", ct.c_float),
        ("skewness", ct.c_float),
        ("sparsity", ct.c_float),
        ("entropy", ct.c_float),
        ("unique_ratio", ct.c_float),
        ("range", ct.c_float),
        ("system1_similarity_threshold", ct.c_float),
        ("critic_acceptance_threshold", ct.c_float),
        ("deep_think_cost_bias", ct.c_float),
        ("label", ct.c_char * (AXCORE_MAX_LABEL_LENGTH + 1)),
    ]


class AxRecallResult(ct.Structure):
    _fields_ = [
        ("found", ct.c_uint32),
        ("similarity", ct.c_float),
        ("stored_step", ct.c_uint64),
        ("age_steps", ct.c_uint64),
        ("level", ct.c_uint32),
        ("span", ct.c_uint32),
        ("source", ct.c_char * (AXCORE_MAX_SOURCE_LENGTH + 1)),
    ]


class AxHeuristicConfig(ct.Structure):
    _fields_ = [
        ("system1_base", ct.c_float),
        ("system1_entropy_weight", ct.c_float),
        ("system1_sparsity_weight", ct.c_float),
        ("system1_min", ct.c_float),
        ("system1_max", ct.c_float),
        ("critic_base", ct.c_float),
        ("critic_entropy_weight", ct.c_float),
        ("critic_skewness_weight", ct.c_float),
        ("critic_min", ct.c_float),
        ("critic_max", ct.c_float),
        ("deep_think_cost_base", ct.c_float),
        ("deep_think_entropy_weight", ct.c_float),
        ("deep_think_sparsity_weight", ct.c_float),
        ("deep_think_cost_min", ct.c_float),
        ("deep_think_cost_max", ct.c_float),
        ("consolidation_min_fitness", ct.c_float),
        ("consolidation_max_normalized_burn", ct.c_float),
        ("consolidation_top_limit", ct.c_uint32),
        ("fatigue_remaining_ratio", ct.c_float),
        ("zombie_activation_ratio", ct.c_float),
        ("zombie_critic_threshold", ct.c_float),
    ]


class AxManifoldScanResult(ct.Structure):
    _fields_ = [
        ("candidate_count", ct.c_uint32),
        ("best_similarity", ct.c_float),
        ("second_best_similarity", ct.c_float),
        ("mean_similarity", ct.c_float),
        ("similarity_stddev", ct.c_float),
        ("ambiguity_gap", ct.c_float),
        ("grounding_score", ct.c_float),
        ("void_score", ct.c_float),
        ("entropy", ct.c_float),
        ("label", ct.c_char * (AXCORE_MAX_LABEL_LENGTH + 1)),
    ]


class AxSystemMetabolism(ct.Structure):
    _fields_ = [
        ("max_capacity", ct.c_float),
        ("current_energy_budget", ct.c_float),
        ("fatigue_threshold", ct.c_float),
        ("zombie_activation_threshold", ct.c_float),
        ("fatigue_remaining_ratio", ct.c_float),
        ("zombie_activation_ratio", ct.c_float),
        ("zombie_critic_threshold", ct.c_float),
        ("zombie_mode_active", ct.c_uint32),
    ]


class AxCacheMatch(ct.Structure):
    _fields_ = [
        ("found", ct.c_uint32),
        ("similarity", ct.c_float),
        ("fitness", ct.c_float),
        ("decay_score", ct.c_float),
        ("last_metabolic_burn", ct.c_float),
        ("average_metabolic_burn", ct.c_float),
        ("burn_samples", ct.c_uint32),
        ("hits", ct.c_uint32),
        ("last_touch", ct.c_uint64),
        ("is_anomaly", ct.c_uint32),
        ("key", ct.c_char * (AXCORE_MAX_KEY_LENGTH + 1)),
        ("dataset_type", ct.c_char * (AXCORE_MAX_DATASET_TYPE_LENGTH + 1)),
        ("dataset_id", ct.c_char * (AXCORE_MAX_DATASET_ID_LENGTH + 1)),
    ]


class AxTensorOpCandidate(ct.Structure):
    _fields_ = [
        ("fitness", ct.c_float),
        ("similarity", ct.c_float),
        ("cost", ct.c_float),
        ("strategy", ct.c_char * (AXCORE_MAX_STRATEGY_LENGTH + 1)),
    ]


def _decode_c_text(char_array: ct.Array[ct.c_char]) -> str:
    return bytes(char_array).split(b"\0", 1)[0].decode("utf-8", errors="ignore")


def _status_name(status: int) -> str:
    return STATUS_NAMES.get(status, f"AX_STATUS_{status}")


def _check(status: int, fn_name: str) -> None:
    if status != AX_STATUS_OK:
        raise RuntimeError(f"{fn_name} failed with {_status_name(status)} ({status})")


def _as_float_array(values: Sequence[float]) -> ct.Array[ct.c_float]:
    return (ct.c_float * len(values))(*values)


def _as_uint32_array(values: Sequence[int]) -> ct.Array[ct.c_uint32]:
    return (ct.c_uint32 * len(values))(*values)


def _sanitize(value: float) -> float:
    return value if math.isfinite(value) else 0.0


def _normalize_l2(values: array) -> array:
    norm_sq = 0.0
    for value in values:
        norm_sq += float(value) * float(value)
    norm = math.sqrt(norm_sq)
    if not (norm > 1.0e-8):
        return array("f", [0.0] * len(values))
    inv = 1.0 / norm
    return array("f", (float(v) * inv for v in values))


def _cosine_similarity(lhs: Sequence[float], rhs: Sequence[float]) -> float:
    if len(lhs) != len(rhs) or not lhs:
        return 0.0
    dot = 0.0
    lhs_sq = 0.0
    rhs_sq = 0.0
    for l, r in zip(lhs, rhs):
        lv = float(l)
        rv = float(r)
        dot += lv * rv
        lhs_sq += lv * lv
        rhs_sq += rv * rv
    denom = math.sqrt(lhs_sq) * math.sqrt(rhs_sq)
    if not (denom > 1.0e-12):
        return 0.0
    value = dot / denom
    if value < -1.0:
        return -1.0
    if value > 1.0:
        return 1.0
    return value


def _perturb_signal(values: Sequence[float], amplitude: float, step: int) -> array:
    if amplitude <= 0.0:
        return array("f", (_sanitize(float(v)) for v in values))
    phase = float(step + 1) * 0.073
    out = array("f")
    for i, value in enumerate(values):
        base = _sanitize(float(value))
        delta = (
            (amplitude * math.sin((float(i) + 1.0) * 0.037 + phase))
            + (amplitude * 0.5 * math.cos((float(i) + 1.0) * 0.011 - (phase * 0.5)))
        )
        out.append(_sanitize(base + delta))
    return out


def _parse_float_csv(values_csv: str | None) -> list[float]:
    if values_csv is None:
        return []
    out: list[float] = []
    for raw in values_csv.split(","):
        token = raw.strip()
        if token == "":
            continue
        out.append(float(token))
    return out


def prepare_hypervector(values: Sequence[float], target_dim: int) -> array:
    if len(values) <= 0 or target_dim <= 0:
        raise ValueError("prepare_hypervector requires non-empty values and target_dim > 0")

    if len(values) == target_dim:
        return _normalize_l2(array("f", (_sanitize(v) for v in values)))

    output = array("f", [0.0] * target_dim)
    for i, value in enumerate(values):
        v = _sanitize(float(value))
        s1 = (i * 1_315_423) % target_dim
        s2 = (i * 2_654_435) % target_dim
        s3 = (i * 805_459) % target_dim
        output[s1] += v
        output[s2] -= v * 0.5
        output[s3] += v * 0.5
    return _normalize_l2(output)


@dataclass
class ScalarStats:
    values: list[float] = field(default_factory=list)

    def add(self, value: float) -> None:
        if math.isfinite(value):
            self.values.append(float(value))

    @property
    def count(self) -> int:
        return len(self.values)

    @property
    def mean(self) -> float:
        return float(statistics.fmean(self.values)) if self.values else 0.0

    @property
    def minimum(self) -> float:
        return min(self.values) if self.values else 0.0

    @property
    def maximum(self) -> float:
        return max(self.values) if self.values else 0.0


class AperiodicPatchGenerator:
    """
    Deterministic aperiodic patch generator.

    It combines irrationally-related geometric phases and local motif hashing to
    produce non-repeating but rule-governed states.
    """

    def __init__(self, patch_side: int, stride: float, seed: int):
        if patch_side <= 1:
            raise ValueError("patch_side must be > 1")
        if stride <= 0.0:
            raise ValueError("stride must be > 0")
        self.patch_side = patch_side
        self.half = patch_side // 2
        self.stride = float(stride)
        self.seed = int(seed)
        self.channels = 3

        self._tau = (1.0 + math.sqrt(5.0)) * 0.5
        self._sigma = math.sqrt(2.0)
        self._golden_angle = math.pi * (3.0 - math.sqrt(5.0))
        self._phase = (self.seed * 0.000123) % (2.0 * math.pi)

    @property
    def raw_dim(self) -> int:
        return self.patch_side * self.patch_side * self.channels

    def _base_field(self, x: int, y: int) -> float:
        fx = float(x)
        fy = float(y)
        p = self._phase
        a = math.cos((fx + self._tau * fy) * 0.411 + p)
        b = math.cos((self._sigma * fx - fy) * 0.287 - (p * 0.7))
        c = math.sin((fx - self._tau * fy) * 0.173 + (p * 1.3))
        d = math.sin((0.131 * fx) + (0.199 * fy) + (p * 0.5))
        return (0.43 * a) + (0.31 * b) + (0.18 * c) + (0.08 * d)

    def _motif(self, x: int, y: int) -> float:
        # Cheap deterministic local motif tag in [-1, 1].
        h = (x * 0x9E3779B1) ^ (y * 0x85EBCA77) ^ self.seed
        h ^= (h >> 13)
        h *= 0xC2B2AE3D
        h ^= (h >> 16)
        bucket = int(h & 31)
        return (bucket / 15.5) - 1.0

    def center_for_step(self, step: int, phase_offset: int = 0) -> tuple[int, int]:
        idx = max(0, int(step) + int(phase_offset))
        angle = float(idx) * self._golden_angle
        radius = self.stride * math.sqrt(float(idx) + 1.0)
        cx = int(round(radius * math.cos(angle)))
        cy = int(round(radius * math.sin(angle)))
        return cx, cy

    def sample_patch(self, center_x: int, center_y: int) -> array:
        out = array("f")
        start_x = center_x - self.half
        start_y = center_y - self.half
        for oy in range(self.patch_side):
            y = start_y + oy
            for ox in range(self.patch_side):
                x = start_x + ox
                f0 = self._base_field(x, y)
                occ = 1.0 if f0 >= 0.0 else -1.0

                gx = self._base_field(x + 1, y) - self._base_field(x - 1, y)
                gy = self._base_field(x, y + 1) - self._base_field(x, y - 1)
                angle = math.atan2(gy, gx) / math.pi  # [-1, 1]

                motif = self._motif(x, y)
                out.extend((occ, angle, motif))
        return out

    def sample_step(self, step: int, phase_offset: int = 0) -> tuple[array, tuple[int, int]]:
        cx, cy = self.center_for_step(step, phase_offset=phase_offset)
        return self.sample_patch(cx, cy), (cx, cy)


class AxCoreBindings:
    def __init__(self, dll_path: Path):
        self.dll_path = dll_path
        self.lib = ct.CDLL(str(dll_path))
        self._bind_signatures()

    def _bind_signatures(self) -> None:
        lib = self.lib
        lib.AxCore_GetVersion.argtypes = [ct.POINTER(AxVersion)]
        lib.AxCore_GetVersion.restype = ct.c_int

        lib.AxCore_GetDefaultCreateInfo.argtypes = [ct.POINTER(AxCoreCreateInfo)]
        lib.AxCore_GetDefaultCreateInfo.restype = None

        lib.AxCore_GetDefaultHeuristics.argtypes = [ct.POINTER(AxHeuristicConfig)]
        lib.AxCore_GetDefaultHeuristics.restype = None

        lib.AxCore_GetDefaultMetabolism.argtypes = [ct.POINTER(AxSystemMetabolism)]
        lib.AxCore_GetDefaultMetabolism.restype = None

        lib.AxCore_ComputeRequiredArenaBytes.argtypes = [ct.POINTER(AxCoreCreateInfo), ct.POINTER(ct.c_uint32)]
        lib.AxCore_ComputeRequiredArenaBytes.restype = ct.c_int

        lib.AxCore_Create.argtypes = [ct.POINTER(AxCoreCreateInfo), ct.POINTER(ct.c_void_p)]
        lib.AxCore_Create.restype = ct.c_int

        lib.AxCore_Destroy.argtypes = [ct.c_void_p]
        lib.AxCore_Destroy.restype = None

        lib.AxCore_Reset.argtypes = [ct.c_void_p]
        lib.AxCore_Reset.restype = None

        lib.AxCore_SetHeuristics.argtypes = [ct.c_void_p, ct.POINTER(AxHeuristicConfig)]
        lib.AxCore_SetHeuristics.restype = ct.c_int

        lib.AxCore_GetMetabolism.argtypes = [ct.c_void_p, ct.POINTER(AxSystemMetabolism)]
        lib.AxCore_GetMetabolism.restype = ct.c_int

        lib.AxCore_ConsumeMetabolism.argtypes = [ct.c_void_p, ct.c_float]
        lib.AxCore_ConsumeMetabolism.restype = ct.c_int

        lib.AxCore_RechargeMetabolism.argtypes = [ct.c_void_p, ct.c_float]
        lib.AxCore_RechargeMetabolism.restype = ct.c_int

        lib.AxCore_StoreEpisode.argtypes = [ct.c_void_p, ct.POINTER(ct.c_float), ct.c_uint32]
        lib.AxCore_StoreEpisode.restype = ct.c_int

        lib.AxCore_RecallSimilar.argtypes = [
            ct.c_void_p,
            ct.POINTER(ct.c_float),
            ct.c_uint32,
            ct.POINTER(ct.c_float),
            ct.c_uint32,
            ct.POINTER(AxRecallResult),
        ]
        lib.AxCore_RecallSimilar.restype = ct.c_int

        lib.AxCore_RecallStepsAgo.argtypes = [
            ct.c_void_p,
            ct.c_uint64,
            ct.POINTER(ct.c_float),
            ct.c_uint32,
            ct.POINTER(AxRecallResult),
        ]
        lib.AxCore_RecallStepsAgo.restype = ct.c_int

        lib.AxCore_Consolidate.argtypes = [ct.c_void_p]
        lib.AxCore_Consolidate.restype = ct.c_int

        lib.AxCore_PromoteWorkingMemory.argtypes = [
            ct.c_void_p,
            ct.c_char_p,
            ct.c_char_p,
            ct.c_char_p,
            ct.POINTER(ct.c_float),
            ct.c_uint32,
            ct.c_float,
            ct.c_float,
        ]
        lib.AxCore_PromoteWorkingMemory.restype = ct.c_int

        lib.AxCore_QueryWorkingMemory.argtypes = [
            ct.c_void_p,
            ct.POINTER(ct.c_float),
            ct.c_uint32,
            ct.c_float,
            ct.POINTER(ct.c_float),
            ct.c_uint32,
            ct.POINTER(AxCacheMatch),
        ]
        lib.AxCore_QueryWorkingMemory.restype = ct.c_int

        lib.AxCore_ApplyWorkingMemoryDecay.argtypes = [ct.c_void_p, ct.c_float, ct.c_float]
        lib.AxCore_ApplyWorkingMemoryDecay.restype = None

        lib.AxCore_AnalyzeSignal.argtypes = [ct.c_void_p, ct.POINTER(ct.c_float), ct.c_uint32, ct.POINTER(AxSignalProfile)]
        lib.AxCore_AnalyzeSignal.restype = ct.c_int

        lib.AxCore_ScanManifoldEntropy.argtypes = [
            ct.c_void_p,
            ct.POINTER(ct.c_float),
            ct.c_uint32,
            ct.POINTER(ct.c_float),
            ct.c_uint32,
            ct.c_uint32,
            ct.POINTER(AxManifoldScanResult),
        ]
        lib.AxCore_ScanManifoldEntropy.restype = ct.c_int

        lib.AxCore_ProjectManifold.argtypes = [
            ct.POINTER(ct.c_float),
            ct.c_uint32,
            ct.c_uint32,
            ct.c_uint32,
            ct.c_uint32,
            ct.c_uint32,
            ct.POINTER(ct.c_float),
            ct.c_uint32,
        ]
        lib.AxCore_ProjectManifold.restype = ct.c_int

        lib.AxCore_RouteCandidate.argtypes = [
            ct.c_void_p,
            ct.POINTER(ct.c_float),
            ct.c_uint32,
            ct.c_uint32,
            ct.POINTER(ct.c_float),
            ct.c_uint32,
            ct.POINTER(AxTensorOpCandidate),
            ct.POINTER(AxSignalProfile),
        ]
        lib.AxCore_RouteCandidate.restype = ct.c_int

        lib.AxCore_DeduceGeometricGap.argtypes = [
            ct.c_void_p,
            ct.POINTER(ct.c_float),
            ct.c_uint32,
            ct.POINTER(ct.c_float),
            ct.c_uint32,
            ct.POINTER(ct.c_float),
            ct.c_uint32,
        ]
        lib.AxCore_DeduceGeometricGap.restype = ct.c_int

        lib.AxCore_BatchSequenceSimilarity.argtypes = [
            ct.POINTER(ct.c_float),
            ct.POINTER(ct.c_uint32),
            ct.c_uint32,
            ct.POINTER(ct.c_float),
            ct.c_uint32,
        ]
        lib.AxCore_BatchSequenceSimilarity.restype = ct.c_float

    def get_version(self) -> AxVersion:
        out = AxVersion()
        status = self.lib.AxCore_GetVersion(ct.byref(out))
        _check(status, "AxCore_GetVersion")
        return out

    def get_default_create_info(self) -> AxCoreCreateInfo:
        info = AxCoreCreateInfo()
        self.lib.AxCore_GetDefaultCreateInfo(ct.byref(info))
        return info

    def get_default_heuristics(self) -> AxHeuristicConfig:
        cfg = AxHeuristicConfig()
        self.lib.AxCore_GetDefaultHeuristics(ct.byref(cfg))
        return cfg

    def get_default_metabolism(self) -> AxSystemMetabolism:
        state = AxSystemMetabolism()
        self.lib.AxCore_GetDefaultMetabolism(ct.byref(state))
        return state

    def compute_required_arena_bytes(self, info: AxCoreCreateInfo) -> int:
        out = ct.c_uint32(0)
        status = self.lib.AxCore_ComputeRequiredArenaBytes(ct.byref(info), ct.byref(out))
        _check(status, "AxCore_ComputeRequiredArenaBytes")
        return int(out.value)

    def create(self, info: AxCoreCreateInfo) -> ct.c_void_p:
        out_handle = ct.c_void_p()
        status = self.lib.AxCore_Create(ct.byref(info), ct.byref(out_handle))
        _check(status, "AxCore_Create")
        if not out_handle:
            raise RuntimeError("AxCore_Create returned a null handle")
        return out_handle

    def destroy(self, handle: ct.c_void_p) -> None:
        if handle:
            self.lib.AxCore_Destroy(handle)

    def reset(self, handle: ct.c_void_p) -> None:
        self.lib.AxCore_Reset(handle)

    def set_heuristics(self, handle: ct.c_void_p, config: AxHeuristicConfig) -> None:
        status = self.lib.AxCore_SetHeuristics(handle, ct.byref(config))
        _check(status, "AxCore_SetHeuristics")

    def get_metabolism(self, handle: ct.c_void_p) -> AxSystemMetabolism:
        out = AxSystemMetabolism()
        status = self.lib.AxCore_GetMetabolism(handle, ct.byref(out))
        _check(status, "AxCore_GetMetabolism")
        return out

    def consume_metabolism(self, handle: ct.c_void_p, amount: float) -> None:
        status = self.lib.AxCore_ConsumeMetabolism(handle, ct.c_float(amount))
        _check(status, "AxCore_ConsumeMetabolism")

    def recharge_metabolism(self, handle: ct.c_void_p, amount: float) -> None:
        status = self.lib.AxCore_RechargeMetabolism(handle, ct.c_float(amount))
        _check(status, "AxCore_RechargeMetabolism")

    def store_episode(self, handle: ct.c_void_p, values: Sequence[float]) -> None:
        arr = _as_float_array(values)
        status = self.lib.AxCore_StoreEpisode(handle, arr, len(values))
        _check(status, "AxCore_StoreEpisode")

    def recall_similar(self, handle: ct.c_void_p, values: Sequence[float], out_dim: int) -> tuple[list[float] | None, AxRecallResult]:
        query = _as_float_array(values)
        out_vals = (ct.c_float * out_dim)()
        result = AxRecallResult()
        status = self.lib.AxCore_RecallSimilar(handle, query, len(values), out_vals, out_dim, ct.byref(result))
        _check(status, "AxCore_RecallSimilar")
        if result.found == 0:
            return None, result
        return [out_vals[i] for i in range(out_dim)], result

    def recall_steps_ago(self, handle: ct.c_void_p, steps_ago: int, out_dim: int) -> tuple[list[float] | None, AxRecallResult]:
        out_vals = (ct.c_float * out_dim)()
        result = AxRecallResult()
        status = self.lib.AxCore_RecallStepsAgo(handle, int(steps_ago), out_vals, out_dim, ct.byref(result))
        _check(status, "AxCore_RecallStepsAgo")
        if result.found == 0:
            return None, result
        return [out_vals[i] for i in range(out_dim)], result

    def consolidate(self, handle: ct.c_void_p) -> None:
        status = self.lib.AxCore_Consolidate(handle)
        _check(status, "AxCore_Consolidate")

    def promote_working_memory(
        self,
        handle: ct.c_void_p,
        key: str,
        dataset_type: str,
        dataset_id: str,
        values: Sequence[float],
        fitness: float,
        normalized_metabolic_burn: float,
    ) -> None:
        arr = _as_float_array(values)
        key_b = key.encode("utf-8")
        dt_b = dataset_type.encode("utf-8")
        did_b = dataset_id.encode("utf-8")
        status = self.lib.AxCore_PromoteWorkingMemory(
            handle,
            key_b,
            dt_b,
            did_b,
            arr,
            len(values),
            ct.c_float(fitness),
            ct.c_float(normalized_metabolic_burn),
        )
        _check(status, "AxCore_PromoteWorkingMemory")

    def query_working_memory(
        self,
        handle: ct.c_void_p,
        values: Sequence[float],
        threshold: float,
        out_dim: int = 0,
    ) -> tuple[AxCacheMatch, list[float] | None]:
        arr = _as_float_array(values)
        match = AxCacheMatch()
        out_ptr = None
        out_count = 0
        out_values = None
        if out_dim > 0:
            buffer = (ct.c_float * out_dim)()
            out_ptr = buffer
            out_count = out_dim
        status = self.lib.AxCore_QueryWorkingMemory(
            handle,
            arr,
            len(values),
            ct.c_float(threshold),
            out_ptr,
            out_count,
            ct.byref(match),
        )
        _check(status, "AxCore_QueryWorkingMemory")
        if out_dim > 0 and match.found != 0:
            out_values = [buffer[i] for i in range(out_dim)]
        return match, out_values

    def apply_working_memory_decay(self, handle: ct.c_void_p, factor: float, floor_value: float) -> None:
        self.lib.AxCore_ApplyWorkingMemoryDecay(handle, ct.c_float(factor), ct.c_float(floor_value))

    def analyze_signal(self, handle: ct.c_void_p, values: Sequence[float]) -> AxSignalProfile:
        arr = _as_float_array(values)
        out = AxSignalProfile()
        status = self.lib.AxCore_AnalyzeSignal(handle, arr, len(values), ct.byref(out))
        _check(status, "AxCore_AnalyzeSignal")
        return out

    def scan_manifold_entropy(
        self,
        handle: ct.c_void_p,
        query_values: Sequence[float],
        candidates: Iterable[Sequence[float]],
    ) -> AxManifoldScanResult:
        candidate_list = list(candidates)
        if not candidate_list:
            raise ValueError("scan_manifold_entropy requires at least one candidate vector")

        dim = len(query_values)
        flat = array("f")
        for candidate in candidate_list:
            if len(candidate) != dim:
                raise ValueError("All candidates must have the same dimension as query_values")
            flat.extend(candidate)

        query_arr = _as_float_array(query_values)
        candidate_arr = _as_float_array(flat)
        out = AxManifoldScanResult()
        status = self.lib.AxCore_ScanManifoldEntropy(
            handle,
            query_arr,
            dim,
            candidate_arr,
            len(candidate_list),
            dim,
            ct.byref(out),
        )
        _check(status, "AxCore_ScanManifoldEntropy")
        return out

    def project_manifold(
        self,
        vectors: Sequence[Sequence[float]],
        input_dim: int,
        output_dim: int,
        normalize_rows: bool,
        projection_seed: int,
    ) -> list[list[float]]:
        if not vectors:
            return []
        sample_count = len(vectors)
        flat = array("f")
        for row in vectors:
            if len(row) != input_dim:
                raise ValueError("Each vector must match input_dim")
            flat.extend(row)

        out_count = sample_count * output_dim
        out_flat = (ct.c_float * out_count)()
        in_arr = _as_float_array(flat)
        status = self.lib.AxCore_ProjectManifold(
            in_arr,
            sample_count,
            input_dim,
            output_dim,
            1 if normalize_rows else 0,
            projection_seed,
            out_flat,
            out_count,
        )
        _check(status, "AxCore_ProjectManifold")
        rows: list[list[float]] = []
        for i in range(sample_count):
            base = i * output_dim
            rows.append([float(out_flat[base + d]) for d in range(output_dim)])
        return rows

    def route_candidate(
        self,
        handle: ct.c_void_p,
        values: Sequence[float],
        iteration: int,
        out_dim: int,
        capture_output: bool = False,
    ) -> tuple[list[float] | None, AxTensorOpCandidate, AxSignalProfile]:
        arr = _as_float_array(values)
        out_values = (ct.c_float * out_dim)()
        candidate = AxTensorOpCandidate()
        profile = AxSignalProfile()
        status = self.lib.AxCore_RouteCandidate(
            handle,
            arr,
            len(values),
            int(iteration),
            out_values,
            out_dim,
            ct.byref(candidate),
            ct.byref(profile),
        )
        _check(status, "AxCore_RouteCandidate")
        if capture_output:
            return [out_values[i] for i in range(out_dim)], candidate, profile
        return None, candidate, profile

    def deduce_geometric_gap(
        self,
        handle: ct.c_void_p,
        current_state: Sequence[float],
        required_next_state: Sequence[float],
        out_dim: int,
        capture_output: bool = False,
        norm_cap: int = 1024,
    ) -> tuple[list[float] | None, float]:
        current_arr = _as_float_array(current_state)
        next_arr = _as_float_array(required_next_state)
        out_values = (ct.c_float * out_dim)()
        status = self.lib.AxCore_DeduceGeometricGap(
            handle,
            current_arr,
            len(current_state),
            next_arr,
            len(required_next_state),
            out_values,
            out_dim,
        )
        _check(status, "AxCore_DeduceGeometricGap")
        cap = out_dim if norm_cap <= 0 else min(out_dim, norm_cap)
        norm_sq = 0.0
        for i in range(cap):
            v = float(out_values[i])
            norm_sq += v * v
        if capture_output:
            return [out_values[i] for i in range(out_dim)], math.sqrt(norm_sq)
        return None, math.sqrt(norm_sq)

    def batch_sequence_similarity(
        self,
        vector_stack_flat: Sequence[float],
        sequence_indices: Sequence[int],
        target_vector: Sequence[float],
        dim: int,
    ) -> float:
        stack_arr = _as_float_array(vector_stack_flat)
        idx_arr = _as_uint32_array(sequence_indices)
        target_arr = _as_float_array(target_vector)
        return float(self.lib.AxCore_BatchSequenceSimilarity(stack_arr, idx_arr, len(sequence_indices), target_arr, dim))


def _shared_library_names() -> list[str]:
    if sys.platform.startswith("win"):
        return ["axcore.dll"]
    if sys.platform == "darwin":
        return ["libaxcore.dylib"]
    return ["libaxcore.so"]


def resolve_dll_path(explicit_path: str | None) -> Path:
    names = _shared_library_names()
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
        if path.is_file():
            return path
        if path.is_dir():
            for name in names:
                candidate = path / name
                if candidate.exists():
                    return candidate
        raise FileNotFoundError(f"AxCore shared library not found at: {path}")

    base = Path(__file__).resolve().parent
    roots = [
        base / "bin",
        base / "build",
        base / "build" / "Release",
        base,
        Path.cwd() / "bin",
        Path.cwd() / "build",
        Path.cwd() / "build" / "Release",
        Path.cwd(),
    ]
    candidates: list[Path] = []
    for root in roots:
        for name in names:
            candidates.append(root / name)
    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = "\n".join(f"  - {path}" for path in candidates)
    names_list = ", ".join(names)
    raise FileNotFoundError(f"Could not locate AxCore shared library ({names_list}). Searched:\n{searched}")


def _sample_for_projection(reference_bank: deque[array], sample_count: int) -> list[array]:
    if not reference_bank:
        return []
    data = list(reference_bank)
    if len(data) <= sample_count:
        return data
    stride = max(1, len(data) // sample_count)
    picked: list[array] = []
    i = 0
    while i < len(data) and len(picked) < sample_count:
        picked.append(data[i])
        i += stride
    return picked


def run_experiment(args: argparse.Namespace) -> None:
    dll_path = resolve_dll_path(args.dll)
    api = AxCoreBindings(dll_path)

    version = api.get_version()
    defaults = api.get_default_create_info()
    baseline_metabolism = api.get_default_metabolism()

    config = AxCoreCreateInfo(
        hdc_dim=args.hdc_dim,
        working_memory_capacity=args.working_memory_capacity,
        episodic_recent_limit=args.episodic_recent_limit,
        episodic_max_levels=args.episodic_max_levels,
        arena_bytes=args.arena_bytes,
    )
    if config.arena_bytes == 0:
        config.arena_bytes = defaults.arena_bytes
    required_arena = api.compute_required_arena_bytes(config)

    heuristics = api.get_default_heuristics()
    heuristics.consolidation_min_fitness = float(args.consolidation_min_fitness)
    heuristics.consolidation_top_limit = int(max(1, args.consolidation_top_limit))

    generator = AperiodicPatchGenerator(
        patch_side=args.patch_side,
        stride=args.patch_stride,
        seed=args.seed,
    )
    holdout_threshold_sweep = _parse_float_csv(args.holdout_threshold_sweep)
    recall_perturb_sweep = _parse_float_csv(args.recall_perturb_sweep)

    print(f"AxCore Aperiodic Manifold Experiment | DLL: {dll_path}")
    print(
        f"Version={version.major}.{version.minor}.{version.patch} (ABI {version.abi}) | "
        f"HDC={config.hdc_dim} | Raw={generator.raw_dim} | ArenaRequired={required_arena}"
    )
    print(
        f"WM={config.working_memory_capacity} | Recent={config.episodic_recent_limit} | Levels={config.episodic_max_levels} | "
        f"BaselineEnergy={baseline_metabolism.current_energy_budget:.1f}"
    )

    handle = api.create(config)
    api.set_heuristics(handle, heuristics)

    train_grounding = ScalarStats()
    train_ambiguity = ScalarStats()
    train_route_fitness = ScalarStats()
    train_route_cost = ScalarStats()
    train_seq_similarity = ScalarStats()
    train_recall_perturbed_similarity = ScalarStats()
    train_steps_ago_similarity = ScalarStats()
    train_steps_ago_returned_age_similarity = ScalarStats()
    train_gap_norm = ScalarStats()
    train_entropy = ScalarStats()
    train_wm_hits = 0
    train_wm_queries = 0
    train_steps_ago_eligible = 0
    train_steps_ago_found = 0
    train_steps_ago_exact_age_matches = 0
    train_steps_ago_age_mismatches = 0
    train_steps_ago_skipped_history = 0
    executed_train_steps = 0

    hold_grounding = ScalarStats()
    hold_ambiguity = ScalarStats()
    hold_wm_similarity = ScalarStats()
    hold_route_fitness = ScalarStats()
    hold_wm_hits = 0
    hold_wm_queries = 0

    candidate_history: deque[array] = deque(maxlen=args.candidate_history)
    reference_bank: deque[array] = deque(maxlen=args.reference_bank)
    full_hvec_history: list[array] = []
    full_raw_history: list[array] = []
    holdout_raw_signals: list[array] = []

    representation_discriminator_done = False
    representation_discriminator_found = 0
    representation_discriminator_age = -1
    representation_cos_to_hdc = 0.0
    representation_cos_to_raw_padded = 0.0
    representation_cos_prefix_raw = 0.0

    try:
        print("\n--- Training Phase (Aperiodic ingest) ---")
        for step in range(args.train_steps):
            executed_train_steps += 1
            raw_signal, (cx, cy) = generator.sample_step(step)
            next_signal, _ = generator.sample_step(step + 1)
            hvec = prepare_hypervector(raw_signal, args.hdc_dim)

            profile = api.analyze_signal(handle, raw_signal)
            train_entropy.add(float(profile.entropy))

            match, _match_values = api.query_working_memory(
                handle,
                raw_signal,
                threshold=args.working_memory_threshold,
                out_dim=args.hdc_dim if args.capture_match_vectors else 0,
            )
            train_wm_queries += 1
            if match.found != 0:
                train_wm_hits += 1

            _routed, candidate, _route_profile = api.route_candidate(
                handle,
                raw_signal,
                iteration=step,
                out_dim=args.hdc_dim,
                capture_output=False,
            )
            train_route_fitness.add(float(candidate.fitness))
            train_route_cost.add(float(candidate.cost))

            _gap, gap_norm = api.deduce_geometric_gap(
                handle,
                raw_signal,
                next_signal,
                out_dim=args.hdc_dim,
                capture_output=False,
                norm_cap=1024,
            )
            train_gap_norm.add(gap_norm)

            if (step % args.promote_every) == 0:
                key = f"spectre_seed_{step:05d}_{cx}_{cy}"
                fitness = max(0.05, min(1.0, (float(candidate.fitness) * 0.7) + ((1.0 - float(profile.entropy)) * 0.3)))
                burn = max(0.0, min(1.0, float(candidate.cost) / 20.0))
                api.promote_working_memory(
                    handle,
                    key=key,
                    dataset_type="aperiodic_proxy",
                    dataset_id=f"patch_{args.patch_side}x{args.patch_side}",
                    values=raw_signal,
                    fitness=fitness,
                    normalized_metabolic_burn=burn,
                )

            api.store_episode(handle, raw_signal)
            full_raw_history.append(raw_signal)
            full_hvec_history.append(hvec)

            if (not representation_discriminator_done) and full_hvec_history:
                discriminator_vals, discriminator_meta = api.recall_steps_ago(handle, steps_ago=0, out_dim=args.hdc_dim)
                representation_discriminator_done = True
                if discriminator_meta.found != 0 and discriminator_vals is not None:
                    representation_discriminator_found = 1
                    representation_discriminator_age = int(discriminator_meta.age_steps)
                    representation_cos_to_hdc = _cosine_similarity(discriminator_vals, full_hvec_history[-1])

                    raw_padded = array("f", [0.0] * args.hdc_dim)
                    copy_len = min(len(raw_signal), args.hdc_dim)
                    for idx in range(copy_len):
                        raw_padded[idx] = _sanitize(float(raw_signal[idx]))
                    raw_padded = _normalize_l2(raw_padded)
                    representation_cos_to_raw_padded = _cosine_similarity(discriminator_vals, raw_padded)

                    rec_prefix = _normalize_l2(array("f", discriminator_vals[: len(raw_signal)]))
                    raw_norm = _normalize_l2(array("f", (_sanitize(float(v)) for v in raw_signal)))
                    representation_cos_prefix_raw = _cosine_similarity(rec_prefix, raw_norm)

            if len(candidate_history) >= args.min_candidates_for_scan and ((step + 1) % args.scan_every) == 0:
                recent = list(candidate_history)[-args.scan_candidate_limit :]
                scan = api.scan_manifold_entropy(handle, hvec, recent)
                train_grounding.add(float(scan.grounding_score))
                train_ambiguity.add(float(scan.ambiguity_gap))

            if args.sequence_window > 1 and len(reference_bank) >= args.sequence_window and ((step + 1) % args.sequence_interval) == 0:
                window = list(reference_bank)[-args.sequence_window :]
                stack = array("f")
                for row in window:
                    stack.extend(row)
                indices = list(range(args.sequence_window))
                seq_sim = api.batch_sequence_similarity(stack, indices, hvec, dim=args.hdc_dim)
                train_seq_similarity.add(seq_sim)

            if ((step + 1) % args.recall_interval) == 0:
                perturbed = _perturb_signal(raw_signal, amplitude=args.recall_perturb_amplitude, step=step)
                _rec_vals, rec = api.recall_similar(handle, perturbed, out_dim=args.hdc_dim)
                if rec.found != 0:
                    train_recall_perturbed_similarity.add(float(rec.similarity))

                rec_steps_vals, rec_steps_meta = api.recall_steps_ago(handle, steps_ago=args.recall_steps_ago, out_dim=args.hdc_dim)
                requested_age = int(args.recall_steps_ago)
                if requested_age < len(full_hvec_history):
                    train_steps_ago_eligible += 1
                    if rec_steps_meta.found != 0 and rec_steps_vals is not None:
                        train_steps_ago_found += 1
                        returned_age = int(rec_steps_meta.age_steps)
                        if returned_age < len(full_hvec_history):
                            expected_returned = full_hvec_history[-(returned_age + 1)]
                            returned_score = _cosine_similarity(rec_steps_vals, expected_returned)
                            train_steps_ago_returned_age_similarity.add(returned_score)
                        if returned_age == requested_age:
                            expected_requested = full_hvec_history[-(requested_age + 1)]
                            requested_score = _cosine_similarity(rec_steps_vals, expected_requested)
                            train_steps_ago_similarity.add(requested_score)
                            train_steps_ago_exact_age_matches += 1
                        else:
                            train_steps_ago_age_mismatches += 1
                else:
                    train_steps_ago_skipped_history += 1

            reference_bank.append(hvec)
            candidate_history.append(hvec)

            api.apply_working_memory_decay(handle, args.decay_factor, args.decay_floor)
            api.consume_metabolism(handle, float(profile.entropy) * args.entropy_burn_scale)

            metabolism = api.get_metabolism(handle)
            if (step % args.log_every) == 0:
                print(
                    f"[train] step={step:04d} center=({cx:5d},{cy:5d}) "
                    f"entropy={float(profile.entropy):.3f} fit={float(candidate.fitness):.3f} "
                    f"cost={float(candidate.cost):5.2f} strategy={_decode_c_text(candidate.strategy):<16} "
                    f"energy={float(metabolism.current_energy_budget):7.2f}"
                )

            if float(metabolism.current_energy_budget) < args.collapse_energy_floor:
                print("[train] metabolic collapse floor reached; ending training early.")
                break

        print("\n--- Recall Perturbation Sweep ---")
        probe_count = min(len(full_raw_history), max(1, int(args.recall_sweep_probes)))
        recall_probe_set = full_raw_history[-probe_count:] if probe_count > 0 else []
        if recall_probe_set and recall_perturb_sweep:
            for amplitude in recall_perturb_sweep:
                stats = ScalarStats()
                found_count = 0
                for probe_index, probe_raw in enumerate(recall_probe_set):
                    perturbed_probe = _perturb_signal(probe_raw, amplitude=amplitude, step=probe_index)
                    _vals, rec = api.recall_similar(handle, perturbed_probe, out_dim=args.hdc_dim)
                    if rec.found != 0:
                        found_count += 1
                        stats.add(float(rec.similarity))
                found_rate = found_count / float(len(recall_probe_set))
                print(
                    f"amp={amplitude:.3f} found_rate={found_rate:.3f} "
                    f"sim_mean={stats.mean:.3f} sim_min={stats.minimum:.3f} sim_max={stats.maximum:.3f}"
                )
        else:
            print("Recall sweep skipped (no probes or empty --recall-perturb-sweep).")

        print("\n--- Consolidation ---")
        api.consolidate(handle)
        api.recharge_metabolism(handle, -1.0)
        post_sleep = api.get_metabolism(handle)
        print(f"Consolidated. Energy reset to {float(post_sleep.current_energy_budget):.1f}.")

        print("\n--- Holdout Phase (Unseen far-field patches) ---")
        reference_for_holdout = list(reference_bank)[-args.scan_candidate_limit :]
        holdout_threshold = (
            args.working_memory_threshold
            if args.holdout_working_memory_threshold is None
            else args.holdout_working_memory_threshold
        )
        print(f"Holdout WM threshold: {holdout_threshold:.3f}")
        for i in range(args.holdout_steps):
            raw_signal, (cx, cy) = generator.sample_step(i, phase_offset=args.holdout_offset)
            hvec = prepare_hypervector(raw_signal, args.hdc_dim)
            holdout_raw_signals.append(raw_signal)

            if reference_for_holdout:
                scan = api.scan_manifold_entropy(handle, hvec, reference_for_holdout)
                hold_grounding.add(float(scan.grounding_score))
                hold_ambiguity.add(float(scan.ambiguity_gap))

            match, _ = api.query_working_memory(handle, raw_signal, threshold=holdout_threshold, out_dim=0)
            hold_wm_queries += 1
            if match.found != 0:
                hold_wm_hits += 1
                hold_wm_similarity.add(float(match.similarity))

            _routed, candidate, _ = api.route_candidate(
                handle,
                raw_signal,
                iteration=args.train_steps + i,
                out_dim=args.hdc_dim,
                capture_output=False,
            )
            hold_route_fitness.add(float(candidate.fitness))

            if (i % args.log_every) == 0:
                print(
                    f"[hold ] step={i:04d} center=({cx:5d},{cy:5d}) "
                    f"wm_found={int(match.found)} wm_sim={float(match.similarity):.3f} "
                    f"fit={float(candidate.fitness):.3f} strategy={_decode_c_text(candidate.strategy):<16}"
                )

        print("\n--- Holdout WM Threshold Sweep ---")
        if holdout_raw_signals and holdout_threshold_sweep:
            for threshold in holdout_threshold_sweep:
                sim_stats = ScalarStats()
                hits = 0
                for raw_signal in holdout_raw_signals:
                    sweep_match, _ = api.query_working_memory(handle, raw_signal, threshold=threshold, out_dim=0)
                    if sweep_match.found != 0:
                        hits += 1
                        sim_stats.add(float(sweep_match.similarity))
                hit_rate = hits / float(len(holdout_raw_signals))
                print(
                    f"threshold={threshold:.3f} hit_rate={hit_rate:.3f} "
                    f"sim_mean={sim_stats.mean:.3f} sim_min={sim_stats.minimum:.3f} sim_max={sim_stats.maximum:.3f}"
                )
        else:
            print("Holdout threshold sweep skipped (no holdout samples or empty --holdout-threshold-sweep).")

        print("\n--- Projection + Sequence Readout ---")
        projection_inputs = _sample_for_projection(reference_bank, args.project_samples)
        projection_rows = api.project_manifold(
            vectors=projection_inputs,
            input_dim=args.hdc_dim,
            output_dim=args.project_dim,
            normalize_rows=True,
            projection_seed=args.projection_seed,
        )
        if projection_rows:
            dims = list(zip(*projection_rows))
            ranges = [(min(col), max(col)) for col in dims]
            print(f"Projected {len(projection_rows)} vectors -> {args.project_dim}D ranges: {ranges}")
        else:
            print("Projection skipped (no vectors).")

        coherent_grounding = hold_grounding.mean >= (train_grounding.mean * args.grounding_retention_ratio if train_grounding.count else 0.0)
        wide_ambiguity = hold_ambiguity.mean >= args.min_expected_ambiguity

        print("\n=== Experiment Summary ===")
        hold_wm_hit_rate = (hold_wm_hits / hold_wm_queries) if hold_wm_queries else 0.0
        print(
            f"Train: executed_steps={executed_train_steps}/{args.train_steps} scans={train_grounding.count} grounding={train_grounding.mean:.3f} "
            f"ambiguity={train_ambiguity.mean:.3f} entropy={train_entropy.mean:.3f}"
        )
        print(
            f"Train: route_fitness={train_route_fitness.mean:.3f} route_cost={train_route_cost.mean:.3f} "
            f"sequence_similarity={train_seq_similarity.mean:.3f}"
        )
        wm_hit_rate = (train_wm_hits / train_wm_queries) if train_wm_queries else 0.0
        steps_found_ratio = (train_steps_ago_found / train_steps_ago_eligible) if train_steps_ago_eligible else 0.0
        steps_exact_ratio = (train_steps_ago_exact_age_matches / train_steps_ago_eligible) if train_steps_ago_eligible else 0.0
        print(
            f"Train: wm_hit_rate={wm_hit_rate:.3f} recall_perturbed={train_recall_perturbed_similarity.mean:.3f} "
            f"steps_ago_similarity_exact={train_steps_ago_similarity.mean:.3f} "
            f"steps_ago_similarity_returned={train_steps_ago_returned_age_similarity.mean:.3f} "
            f"gap_probe_norm={train_gap_norm.mean:.3f}"
        )
        print(
            f"Train: steps_ago_found_ratio={steps_found_ratio:.3f} "
            f"steps_ago_exact_age_ratio={steps_exact_ratio:.3f} "
            f"steps_ago_age_mismatches={train_steps_ago_age_mismatches} "
            f"steps_ago_skipped_history={train_steps_ago_skipped_history}"
        )
        print(
            f"Holdout: scans={hold_grounding.count} grounding={hold_grounding.mean:.3f} "
            f"ambiguity={hold_ambiguity.mean:.3f} wm_hit_rate={hold_wm_hit_rate:.3f} "
            f"wm_similarity={hold_wm_similarity.mean:.3f} "
            f"route_fitness={hold_route_fitness.mean:.3f}"
        )
        print(
            f"Representation discriminator: found={representation_discriminator_found} "
            f"returned_age={representation_discriminator_age} dims(raw={generator.raw_dim}, hdc={args.hdc_dim}) "
            f"cos(return,hdc_expected)={representation_cos_to_hdc:.3f} "
            f"cos(return,raw_padded_hdc)={representation_cos_to_raw_padded:.3f} "
            f"cos(return_prefix,raw)={representation_cos_prefix_raw:.3f}"
        )
        print(
            f"Hypothesis: coherent_grounding={int(coherent_grounding)} "
            f"(retention_ratio_target={args.grounding_retention_ratio:.2f}), "
            f"wide_ambiguity={int(wide_ambiguity)} (min={args.min_expected_ambiguity:.3f})"
        )

        api.reset(handle)
        reset_state = api.get_metabolism(handle)
        print(f"Reset complete. Energy={float(reset_state.current_energy_budget):.1f}.")
    finally:
        api.destroy(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AxCore Einstein/Spectre-inspired aperiodic manifold experiment.")
    parser.add_argument("--dll", type=str, default=None, help="Path to axcore shared library file or folder containing it.")

    parser.add_argument("--hdc-dim", type=int, default=65536)
    parser.add_argument("--working-memory-capacity", type=int, default=96)
    parser.add_argument("--episodic-recent-limit", type=int, default=96)
    parser.add_argument("--episodic-max-levels", type=int, default=24)
    parser.add_argument("--arena-bytes", type=int, default=0)

    parser.add_argument("--patch-side", type=int, default=8, help="Patch width/height. Raw dim = patch_side * patch_side * 3.")
    parser.add_argument("--patch-stride", type=float, default=3.5, help="Spiral stride used to sample aperiodic space.")
    parser.add_argument("--seed", type=int, default=20260310)

    parser.add_argument("--train-steps", type=int, default=48)
    parser.add_argument("--holdout-steps", type=int, default=20)
    parser.add_argument("--holdout-offset", type=int, default=2500)

    parser.add_argument("--candidate-history", type=int, default=48)
    parser.add_argument("--reference-bank", type=int, default=96)
    parser.add_argument("--scan-candidate-limit", type=int, default=16)
    parser.add_argument("--min-candidates-for-scan", type=int, default=8)
    parser.add_argument("--scan-every", type=int, default=2)

    parser.add_argument("--promote-every", type=int, default=2)
    parser.add_argument("--working-memory-threshold", type=float, default=0.20)
    parser.add_argument(
        "--holdout-working-memory-threshold",
        type=float,
        default=None,
        help="Holdout WM query threshold. Defaults to --working-memory-threshold when omitted.",
    )
    parser.add_argument(
        "--holdout-threshold-sweep",
        type=str,
        default="0.20,0.35,0.50,0.65",
        help="Comma-separated thresholds for holdout WM sweep.",
    )
    parser.add_argument("--capture-match-vectors", action="store_true")

    parser.add_argument("--recall-interval", type=int, default=8)
    parser.add_argument("--recall-steps-ago", type=int, default=12)
    parser.add_argument(
        "--recall-perturb-amplitude",
        type=float,
        default=0.035,
        help="Amplitude for deterministic perturbation used in recall_similar probes.",
    )
    parser.add_argument(
        "--recall-perturb-sweep",
        type=str,
        default="0.01,0.03,0.05,0.10,0.20",
        help="Comma-separated perturb amplitudes for recall degradation sweep.",
    )
    parser.add_argument(
        "--recall-sweep-probes",
        type=int,
        default=8,
        help="How many latest training signals to use for perturbation sweep.",
    )
    parser.add_argument("--sequence-window", type=int, default=8)
    parser.add_argument("--sequence-interval", type=int, default=8)

    parser.add_argument("--decay-factor", type=float, default=0.997)
    parser.add_argument("--decay-floor", type=float, default=0.10)
    parser.add_argument("--entropy-burn-scale", type=float, default=8.0)
    parser.add_argument("--collapse-energy-floor", type=float, default=120.0)

    parser.add_argument("--consolidation-min-fitness", type=float, default=0.20)
    parser.add_argument("--consolidation-top-limit", type=int, default=48)

    parser.add_argument("--project-samples", type=int, default=24)
    parser.add_argument("--project-dim", type=int, default=3)
    parser.add_argument("--projection-seed", type=int, default=2026)

    parser.add_argument("--grounding-retention-ratio", type=float, default=0.70)
    parser.add_argument("--min-expected-ambiguity", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
