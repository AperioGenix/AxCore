#!/usr/bin/env python3
"""
AxCore World-Model release demo.

Demonstrates:
- Metabolic gating (entropy-driven energy burn + zombie mode)
- Manifold grounding scans (grounded/diffuse/void style labels)
- Episodic trace storage
- Sleep-cycle consolidation into working memory
- Holographic query against consolidated memory
"""

from __future__ import annotations

import argparse
import ctypes as ct
import math
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


AXCORE_MAX_KEY_LENGTH = 63
AXCORE_MAX_DATASET_TYPE_LENGTH = 31
AXCORE_MAX_DATASET_ID_LENGTH = 63
AXCORE_MAX_LABEL_LENGTH = 31

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
        ("source", ct.c_char * 16),
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


def _decode_c_text(char_array: ct.Array[ct.c_char]) -> str:
    return bytes(char_array).split(b"\0", 1)[0].decode("utf-8", errors="ignore")


def _status_name(status: int) -> str:
    return STATUS_NAMES.get(status, f"AX_STATUS_{status}")


def _check(status: int, fn_name: str) -> None:
    if status != AX_STATUS_OK:
        raise RuntimeError(f"{fn_name} failed with {_status_name(status)} ({status})")


def _as_float_array(values: Sequence[float]) -> ct.Array[ct.c_float]:
    return (ct.c_float * len(values))(*values)


def _sanitize(value: float) -> float:
    return value if math.isfinite(value) else 0.0


def _normalize_l2(values: list[float]) -> list[float]:
    norm_sq = 0.0
    for value in values:
        norm_sq += value * value
    norm = math.sqrt(norm_sq)
    if not (norm > 1.0e-8):
        return [0.0] * len(values)
    inv = 1.0 / norm
    return [value * inv for value in values]


def prepare_hypervector(values: Sequence[float], target_dim: int) -> list[float]:
    if len(values) <= 0 or target_dim <= 0:
        raise ValueError("prepare_hypervector requires non-empty values and target_dim > 0")

    if len(values) == target_dim:
        return _normalize_l2([_sanitize(value) for value in values])

    output = [0.0] * target_dim
    for i, value in enumerate(values):
        v = _sanitize(value)
        s1 = (i * 1_315_423) % target_dim
        s2 = (i * 2_654_435) % target_dim
        s3 = (i * 805_459) % target_dim
        output[s1] += v
        output[s2] -= v * 0.5
        output[s3] += v * 0.5
    return _normalize_l2(output)


@dataclass
class EnvironmentGenerator:
    raw_dim: int
    seed: int

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        self._sine_phase = 0.0
        self._logistic_x = 0.417
        self._gravity_t = 0.0

    def generate(self, mode: str, frame: int) -> list[float]:
        if mode == "Monolith (Sine)":
            return self._sine(frame)
        if mode == "Butterfly (Chaos)":
            return self._logistic(frame)
        return self._gravity(frame)

    def _sine(self, frame: int) -> list[float]:
        self._sine_phase += 0.075
        drift = 0.35 * math.sin(frame * 0.011)
        out: list[float] = []
        for i in range(self.raw_dim):
            x = (2.0 * math.pi * i) / float(self.raw_dim)
            carrier = math.sin((3.0 * x) + self._sine_phase)
            harmonic = 0.45 * math.sin((7.0 * x) + (0.5 * self._sine_phase) + drift)
            envelope = 0.9 + (0.1 * math.sin((0.02 * frame) + x))
            noise = 0.03 * (self.rng.random() - 0.5)
            out.append((carrier + harmonic) * envelope + noise)
        return out

    def _logistic(self, frame: int) -> list[float]:
        r = 3.86 + (0.13 * math.sin(frame * 0.03))
        x = self._logistic_x
        out: list[float] = []
        for i in range(self.raw_dim):
            x = r * x * (1.0 - x)
            jitter = 0.02 * math.sin((frame + i) * 0.11)
            out.append((2.0 * x) - 1.0 + jitter)
        self._logistic_x = x
        return out

    def _gravity(self, frame: int) -> list[float]:
        self._gravity_t += 0.085
        t = self._gravity_t
        radius = 1.2 + (0.4 * math.sin(0.37 * t))
        theta = t + (0.25 * math.sin(0.19 * t))
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        well = 1.0 / max(0.2, radius * radius)

        out: list[float] = []
        for i in range(self.raw_dim):
            angle = (2.0 * math.pi * i) / float(self.raw_dim)
            v = 0.55 * math.sin(angle + theta)
            v += 0.25 * math.cos((2.0 * angle) - theta)
            v += 0.20 * math.sin((3.0 * angle) + (2.5 * x))
            v *= 0.8 + (0.4 * well)
            v += 0.10 * math.sin((5.0 * angle) + (3.0 * y))
            v += 0.03 * (self.rng.random() - 0.5)
            out.append(v)
        return out


class AxCoreBindings:
    def __init__(self, dll_path: Path):
        self.dll_path = dll_path
        self.lib = ct.CDLL(str(dll_path))
        self._bind_signatures()

    def _bind_signatures(self) -> None:
        lib = self.lib
        lib.AxCore_Create.argtypes = [ct.POINTER(AxCoreCreateInfo), ct.POINTER(ct.c_void_p)]
        lib.AxCore_Create.restype = ct.c_int

        lib.AxCore_Destroy.argtypes = [ct.c_void_p]
        lib.AxCore_Destroy.restype = None

        lib.AxCore_GetMetabolism.argtypes = [ct.c_void_p, ct.POINTER(AxSystemMetabolism)]
        lib.AxCore_GetMetabolism.restype = ct.c_int

        lib.AxCore_ConsumeMetabolism.argtypes = [ct.c_void_p, ct.c_float]
        lib.AxCore_ConsumeMetabolism.restype = ct.c_int

        lib.AxCore_StoreEpisode.argtypes = [ct.c_void_p, ct.POINTER(ct.c_float), ct.c_uint32]
        lib.AxCore_StoreEpisode.restype = ct.c_int

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

        lib.AxCore_GetDefaultHeuristics.argtypes = [ct.POINTER(AxHeuristicConfig)]
        lib.AxCore_GetDefaultHeuristics.restype = None

        lib.AxCore_SetHeuristics.argtypes = [ct.c_void_p, ct.POINTER(AxHeuristicConfig)]
        lib.AxCore_SetHeuristics.restype = ct.c_int

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

    def analyze_signal(self, handle: ct.c_void_p, values: Sequence[float]) -> AxSignalProfile:
        profile = AxSignalProfile()
        arr = _as_float_array(values)
        status = self.lib.AxCore_AnalyzeSignal(handle, arr, len(values), ct.byref(profile))
        _check(status, "AxCore_AnalyzeSignal")
        return profile

    def consume_metabolism(self, handle: ct.c_void_p, amount: float) -> None:
        status = self.lib.AxCore_ConsumeMetabolism(handle, ct.c_float(amount))
        _check(status, "AxCore_ConsumeMetabolism")

    def get_metabolism(self, handle: ct.c_void_p) -> AxSystemMetabolism:
        metabolism = AxSystemMetabolism()
        status = self.lib.AxCore_GetMetabolism(handle, ct.byref(metabolism))
        _check(status, "AxCore_GetMetabolism")
        return metabolism

    def store_episode(self, handle: ct.c_void_p, values: Sequence[float]) -> None:
        arr = _as_float_array(values)
        status = self.lib.AxCore_StoreEpisode(handle, arr, len(values))
        _check(status, "AxCore_StoreEpisode")

    def recall_steps_ago(self, handle: ct.c_void_p, steps_ago: int, out_dim: int) -> tuple[list[float] | None, AxRecallResult]:
        values = (ct.c_float * out_dim)()
        result = AxRecallResult()
        status = self.lib.AxCore_RecallStepsAgo(handle, int(steps_ago), values, out_dim, ct.byref(result))
        _check(status, "AxCore_RecallStepsAgo")
        if result.found == 0:
            return None, result
        return [values[i] for i in range(out_dim)], result

    def consolidate(self, handle: ct.c_void_p) -> None:
        status = self.lib.AxCore_Consolidate(handle)
        _check(status, "AxCore_Consolidate")

    def set_demo_heuristics(self, handle: ct.c_void_p, consolidation_min_fitness: float, consolidation_top_limit: int) -> None:
        heuristics = AxHeuristicConfig()
        self.lib.AxCore_GetDefaultHeuristics(ct.byref(heuristics))
        heuristics.consolidation_min_fitness = float(consolidation_min_fitness)
        heuristics.consolidation_top_limit = max(1, int(consolidation_top_limit))
        status = self.lib.AxCore_SetHeuristics(handle, ct.byref(heuristics))
        _check(status, "AxCore_SetHeuristics")

    def apply_working_memory_decay(self, handle: ct.c_void_p, factor: float, floor_value: float) -> None:
        self.lib.AxCore_ApplyWorkingMemoryDecay(handle, ct.c_float(factor), ct.c_float(floor_value))

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
        flat: list[float] = []
        for candidate in candidate_list:
            if len(candidate) != dim:
                raise ValueError("All candidates must have the same dimension as query_values")
            flat.extend(candidate)

        query_arr = _as_float_array(query_values)
        candidate_arr = _as_float_array(flat)
        result = AxManifoldScanResult()
        status = self.lib.AxCore_ScanManifoldEntropy(
            handle,
            query_arr,
            dim,
            candidate_arr,
            len(candidate_list),
            dim,
            ct.byref(result),
        )
        _check(status, "AxCore_ScanManifoldEntropy")
        return result

    def query_working_memory(self, handle: ct.c_void_p, values: Sequence[float], threshold: float) -> AxCacheMatch:
        query_arr = _as_float_array(values)
        match = AxCacheMatch()
        status = self.lib.AxCore_QueryWorkingMemory(
            handle,
            query_arr,
            len(values),
            ct.c_float(threshold),
            None,
            0,
            ct.byref(match),
        )
        _check(status, "AxCore_QueryWorkingMemory")
        return match


def resolve_dll_path(explicit_path: str | None) -> Path:
    if explicit_path:
        dll = Path(explicit_path).expanduser().resolve()
        if dll.exists():
            return dll
        raise FileNotFoundError(f"AxCore DLL not found: {dll}")

    base = Path(__file__).resolve().parent
    candidates = [
        base / "build" / "axcore.dll",
        Path.cwd() / "build" / "axcore.dll",
        base / "axcore.dll",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate axcore.dll. Expected one of:\n"
        + "\n".join(f"  - {path}" for path in candidates)
    )


def run_axcore_world_model(args: argparse.Namespace) -> None:
    dll_path = resolve_dll_path(args.dll)
    api = AxCoreBindings(dll_path)

    config = AxCoreCreateInfo(
        hdc_dim=args.hdc_dim,
        working_memory_capacity=args.working_memory_capacity,
        episodic_recent_limit=args.episodic_recent_limit,
        episodic_max_levels=args.episodic_max_levels,
        arena_bytes=0,
    )
    handle = api.create(config)
    api.set_demo_heuristics(
        handle,
        consolidation_min_fitness=args.consolidation_min_fitness,
        consolidation_top_limit=args.consolidation_top_limit,
    )
    generator = EnvironmentGenerator(raw_dim=args.raw_dim, seed=args.seed)
    candidate_history: deque[list[float]] = deque(maxlen=args.candidate_history)

    environments = ["Monolith (Sine)", "Butterfly (Chaos)", "Gravity Well"]

    print(f"AxCore World-Model Demo | DLL: {dll_path}")
    print(
        f"HDC={args.hdc_dim} | Raw={args.raw_dim} | WM={args.working_memory_capacity} | "
        f"Frames/Env={args.frames_per_environment} | ConsolidationMinFitness={args.consolidation_min_fitness:.2f}"
    )

    try:
        for world_mode in environments:
            print(f"\n--- Entering Environment: {world_mode} ---")
            for frame in range(args.frames_per_environment):
                raw_signal = generator.generate(world_mode, frame)
                state_vector = prepare_hypervector(raw_signal, args.hdc_dim)

                profile = api.analyze_signal(handle, raw_signal)
                api.consume_metabolism(handle, float(profile.entropy) * args.metabolic_burn_scale)

                scan = None
                if len(candidate_history) >= args.min_candidates_for_scan:
                    recent = list(candidate_history)[-args.scan_candidate_limit :]
                    scan = api.scan_manifold_entropy(handle, state_vector, recent)

                match = api.query_working_memory(handle, raw_signal, args.working_memory_threshold)
                api.store_episode(handle, raw_signal)
                candidate_history.append(state_vector)

                metabolism = api.get_metabolism(handle)
                flow_state = "ZOMBIE" if metabolism.zombie_mode_active else "FLOW"
                grounding = scan.grounding_score if scan else 0.0
                scan_label = _decode_c_text(scan.label) if scan else "bootstrap"
                profile_label = _decode_c_text(profile.label)
                wm_hit = "-"
                if match.found:
                    wm_hit = f"{match.similarity:.3f}:{_decode_c_text(match.key)}"

                print(
                    f"[{flow_state}] frame={frame:03d} "
                    f"energy={metabolism.current_energy_budget:7.2f} "
                    f"entropy={profile.entropy:.3f} "
                    f"grounding={grounding:.3f} "
                    f"scan={scan_label:<10} "
                    f"signal={profile_label:<12} "
                    f"wm={wm_hit}"
                )

                if metabolism.current_energy_budget < args.collapse_energy_floor:
                    print("--- METABOLIC COLLAPSE: Triggering Consolidation (Sleep) ---")
                    break

            replay_probes: list[tuple[int, list[float], AxRecallResult]] = []
            for probe_steps in (1, 2, 4, 8, 16, 32, 64):
                replay_probe, replay_context = api.recall_steps_ago(handle, steps_ago=probe_steps, out_dim=args.hdc_dim)
                if replay_probe is not None:
                    replay_probes.append((probe_steps, replay_probe, replay_context))

            api.consolidate(handle)
            api.apply_working_memory_decay(handle, args.decay_factor, args.decay_floor)
            post_sleep = api.get_metabolism(handle)
            print(
                f"Consolidation complete. Energy reset to {post_sleep.current_energy_budget:.1f}. "
                f"Zombie={int(post_sleep.zombie_mode_active)}"
            )
            if replay_probes:
                best = None
                for probe_steps, replay_probe, replay_context in replay_probes:
                    replay = api.query_working_memory(handle, replay_probe, -1.0)
                    if replay.found and (best is None or replay.similarity > best[0]):
                        best = (replay.similarity, probe_steps, replay_context, replay)

                if best is not None:
                    similarity, probe_steps, replay_context, replay = best
                    print(
                        f"Sleep replay (steps_ago={probe_steps}, {_decode_c_text(replay_context.source)} L{replay_context.level}): "
                        f"{similarity:.3f} -> {_decode_c_text(replay.key)}"
                    )
                else:
                    print("Sleep replay: no promoted working-memory entries")
    finally:
        api.destroy(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AxCore World-Model release demo.")
    parser.add_argument("--dll", type=str, default=None, help="Path to axcore.dll")
    parser.add_argument("--hdc-dim", type=int, default=1024)
    parser.add_argument("--raw-dim", type=int, default=256)
    parser.add_argument("--working-memory-capacity", type=int, default=128)
    parser.add_argument("--episodic-recent-limit", type=int, default=256)
    parser.add_argument("--episodic-max-levels", type=int, default=32)
    parser.add_argument("--frames-per-environment", type=int, default=100)
    parser.add_argument("--candidate-history", type=int, default=96)
    parser.add_argument("--scan-candidate-limit", type=int, default=64)
    parser.add_argument("--min-candidates-for-scan", type=int, default=4)
    parser.add_argument("--metabolic-burn-scale", type=float, default=10.0)
    parser.add_argument("--collapse-energy-floor", type=float, default=200.0)
    parser.add_argument("--working-memory-threshold", type=float, default=0.20)
    parser.add_argument("--consolidation-min-fitness", type=float, default=0.20)
    parser.add_argument("--consolidation-top-limit", type=int, default=32)
    parser.add_argument("--decay-factor", type=float, default=0.995)
    parser.add_argument("--decay-floor", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


if __name__ == "__main__":
    run_axcore_world_model(parse_args())
