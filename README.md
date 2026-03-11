# AxCore

AxCore is a zero-dependency C++20 HDC runtime/library for tensor analysis, holographic memory, and geometric routing. It is a native shared library API (not an operating system kernel).

## Core Architecture

The system maintains a strict boundary between substrate state (Soma) and user program logic.

* **Connectome:** Routes candidates based on fitness, similarity, entropy profile, and metabolic state.
* **Metabolism:** Tracks energy budget and gates behavior with `FLOW`/`ZOMBIE` mode transitions.
* **Epigenetic Heuristics:** Profiles entropy, sparsity, skewness, and dynamic thresholds.
* **Working Memory:** LRU-like cache for fast retrieval and constraint metadata.
* **Episodic Memory:** Multi-level holographic trace (recent + log-trace summaries).
* **Consolidation:** Sleep-like promotion from episodic traces into working memory.

## Technical Specifications

* **Language:** C++20
* **Dependencies:** None
* **Build System:** CMake 3.24+
* **Default HDC dimension:** `1024`
* **Default working memory capacity:** `128`
* **Default episodic recent limit:** `256`
* **Default episodic max levels:** `32`

## Repository Interface Map

* [`include/axcore/axcore.h`](include/axcore/axcore.h): High-level engine lifecycle + memory + manifold API (primary DLL interface).
* [`include/axcore/math.h`](include/axcore/math.h): Shape and tensor math API (exported).
* [`include/axcore/types.h`](include/axcore/types.h): Status enums, constants, and all shared structs.
* [`include/axcore/export.h`](include/axcore/export.h): ABI/export macros (`AXCORE_API`, `AXCORE_CALL`).
* [`include/axcore/memory.h`](include/axcore/memory.h): Low-level arena/episodic/working-memory routines.
* [`include/axcore/connectome.h`](include/axcore/connectome.h): Low-level heuristic/metabolic/connectome routines.

Practical linking note:

* For shared-library consumers, prefer `axcore.h` (and `math.h`) as the stable exported entry points.
* `memory.h` and `connectome.h` are low-level module interfaces and are most useful when integrating at source/module level.

## Build

### Windows (recommended)

Run:

```cmd
build.bat
```

This script:

* Locates Visual C++ build tools via `vswhere`.
* Configures with NMake (`Release`).
* Builds `axcore.dll` in `build/`.

### Manual CMake

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Status Codes

`AxStatus` (`include/axcore/types.h`):

* `AX_STATUS_OK`
* `AX_STATUS_INVALID_ARGUMENT`
* `AX_STATUS_DIMENSION_MISMATCH`
* `AX_STATUS_OUT_OF_MEMORY`
* `AX_STATUS_BUFFER_TOO_SMALL`
* `AX_STATUS_NOT_FOUND`
* `AX_STATUS_STATE_ERROR`
* `AX_STATUS_LIMIT_EXCEEDED`

## Core Data Structures

Key structs used across APIs:

* `AxCoreCreateInfo`: `hdc_dim`, `working_memory_capacity`, `episodic_recent_limit`, `episodic_max_levels`, `arena_bytes`.
* `AxSystemMetabolism`: energy capacity/budget, fatigue threshold, zombie threshold, zombie flag.
* `AxHeuristicConfig`: signal threshold tuning + consolidation gates + metabolic ratios.
* `AxSignalProfile`: length, mean/std/skew/sparsity/entropy/range + dynamic thresholds + label.
* `AxManifoldScanResult`: best/second-best similarity, ambiguity, grounding, void score, entropy, label.
* `AxRecallResult`: episodic retrieval metadata (`found`, `similarity`, age, level/span/source).
* `AxCacheMatch`: working-memory hit metadata (`found`, `similarity`, decay/burn stats, key/type/id).
* `AxTensorOpCandidate`: route candidate fitness/similarity/cost/strategy.
* `AxShape`, `AxTensorView`, `AxConstTensorView`: tensor metadata wrappers.
* `AxMetabolicCriticConfig`, `AxGeneRuntimeState`, `AxSceneTickReport`: scene-level metabolic critic API.

## High-Level API Reference (`axcore.h`)

### Defaults and Lifecycle

* `AxCore_GetVersion(AxVersion* out_version)`: Returns runtime version (`major.minor.patch` + ABI).
* `AxCore_GetDefaultCreateInfo(AxCoreCreateInfo* out_info)`: Fills default create parameters.
* `AxCore_GetDefaultHeuristics(AxHeuristicConfig* out_config)`: Fills default heuristic config.
* `AxCore_GetDefaultMetabolism(AxSystemMetabolism* out_state)`: Fills default metabolic state.
* `AxCore_ComputeRequiredArenaBytes(const AxCoreCreateInfo* create_info, uint32_t* out_bytes)`: Computes minimum arena bytes for configuration.
* `AxCore_Create(const AxCoreCreateInfo* create_info, AxCoreHandle** out_handle)`: Allocates/initializes engine.
* `AxCore_Destroy(AxCoreHandle* handle)`: Releases engine and arena memory.
* `AxCore_Reset(AxCoreHandle* handle)`: Clears episodic + working memory and fully recharges metabolism.

### Heuristics and Metabolism

* `AxCore_SetHeuristics(AxCoreHandle* handle, const AxHeuristicConfig* config)`: Replaces active heuristic config.
* `AxCore_GetMetabolism(const AxCoreHandle* handle, AxSystemMetabolism* out_state)`: Reads current metabolic state.
* `AxCore_ConsumeMetabolism(AxCoreHandle* handle, float amount)`: Decreases energy budget (`amount` must be non-negative).
* `AxCore_RechargeMetabolism(AxCoreHandle* handle, float amount)`: Recharges energy; `amount <= 0` means full recharge.

### Episodic Memory

* `AxCore_StoreEpisode(AxCoreHandle* handle, const float* values, uint32_t value_count)`: Projects input to `hdc_dim` and stores in episodic memory.
* `AxCore_RecallSimilar(const AxCoreHandle* handle, const float* values, uint32_t value_count, float* out_values, uint32_t out_value_count, AxRecallResult* out_result)`: Nearest recall against recent+trace memory.
* `AxCore_RecallStepsAgo(const AxCoreHandle* handle, uint64_t steps_ago, float* out_values, uint32_t out_value_count, AxRecallResult* out_result)`: Temporal recall by age.
* `AxCore_Consolidate(AxCoreHandle* handle)`: Promotes high-fitness episodic traces to working memory, clears episodic memory, recharges metabolism.

### Working Memory

* `AxCore_PromoteWorkingMemory(AxCoreHandle* handle, const char* key, const char* dataset_type, const char* dataset_id, const float* values, uint32_t value_count, float fitness, float normalized_metabolic_burn)`: Upserts a normalized working-memory entry.
* `AxCore_QueryWorkingMemory(AxCoreHandle* handle, const float* values, uint32_t value_count, float threshold, float* out_values, uint32_t out_value_count, AxCacheMatch* out_match)`: Cosine query with decay weighting and metadata.
* `AxCore_ApplyWorkingMemoryDecay(AxCoreHandle* handle, float factor, float floor_value)`: Applies global decay to working-memory scores.

### Signal and Manifold

* `AxCore_AnalyzeSignal(const AxCoreHandle* handle, const float* values, uint32_t value_count, AxSignalProfile* out_profile)`: Computes entropy/sparsity/skew profile and thresholds.
* `AxCore_ScanManifoldEntropy(const AxCoreHandle* handle, const float* query_values, uint32_t query_value_count, const float* candidate_stack_flat, uint32_t candidate_count, uint32_t dim, AxManifoldScanResult* out_result)`: Computes ambiguity, grounding, and manifold scan label.
* `AxCore_ProjectManifold(const float* vector_stack_flat, uint32_t sample_count, uint32_t input_dim, uint32_t output_dim, uint32_t normalize_rows, uint32_t projection_seed, float* out_projected_flat, uint32_t out_value_count)`: Identity copy or hashed projection for manifold visualization/processing.

### Connectome and Geometry

* `AxCore_RouteCandidate(AxCoreHandle* handle, const float* values, uint32_t value_count, uint32_t iteration, float* out_values, uint32_t out_value_count, AxTensorOpCandidate* out_candidate, AxSignalProfile* out_profile)`: Computes next candidate route and consumes metabolic cost.
* `AxCore_DeduceGeometricGap(AxCoreHandle* handle, const float* current_state, uint32_t current_count, const float* required_next_state, uint32_t required_count, float* out_values, uint32_t out_value_count)`: Produces normalized geometric delta.
* `AxCore_BatchSequenceSimilarity(const float* vector_stack_flat, const uint32_t* sequence_indices, uint32_t index_count, const float* target_vector, uint32_t dim)`: Bundles indexed sequence vectors and returns cosine similarity.

## Tensor Math API (`math.h`)

* `AxShape_Make(const uint32_t* dims, uint32_t ndim, AxShape* out_shape)`
* `AxShape_Make1D(uint32_t total, AxShape* out_shape)`
* `AxShape_Equals(const AxShape* lhs, const AxShape* rhs)`
* `AxTensor_Copy(AxConstTensorView input, AxTensorView output)`
* `AxTensor_NormalizeL2(AxConstTensorView input, AxTensorView output)`
* `AxTensor_Subtract(AxConstTensorView lhs, AxConstTensorView rhs, AxTensorView output)`
* `AxTensor_Bundle(AxConstTensorView lhs, AxConstTensorView rhs, uint32_t normalize, AxTensorView output)`
* `AxTensor_BundleOptional(AxConstTensorView lhs, AxConstTensorView rhs, const uint32_t* normalize, AxTensorView output)`: `normalize == nullptr` defaults to normalize-on.
* `AxTensor_Permute(AxConstTensorView input, int32_t steps, AxTensorView output)`
* `AxTensor_CosineSimilarity(AxConstTensorView lhs, AxConstTensorView rhs, float* out_similarity)`

## Low-Level Memory API (`memory.h`)

* `AxArena_Init(AxLinearArena* arena, void* backing_memory, uint32_t capacity)`
* `AxArena_Reset(AxLinearArena* arena)`
* `AxArena_Alloc(AxLinearArena* arena, uint32_t bytes, uint32_t alignment)`
* `AxEpisodic_Init(AxEpisodicMemory* memory, AxLinearArena* arena, uint32_t dim, uint32_t max_levels, uint32_t recent_limit)`
* `AxEpisodic_Clear(AxEpisodicMemory* memory)`
* `AxEpisodic_Store(AxEpisodicMemory* memory, AxConstTensorView thought)`
* `AxEpisodic_RecallSimilar(const AxEpisodicMemory* memory, AxConstTensorView query, float* out_values, uint32_t out_value_count, AxRecallResult* out_result)`
* `AxEpisodic_RecallStepsAgo(const AxEpisodicMemory* memory, uint64_t steps_ago, float* out_values, uint32_t out_value_count, AxRecallResult* out_result)`
* `AxWorkingMemory_Init(AxWorkingMemoryCache* cache, AxLinearArena* arena, uint32_t dim, uint32_t capacity)`
* `AxWorkingMemory_Clear(AxWorkingMemoryCache* cache)`
* `AxWorkingMemory_Promote(AxWorkingMemoryCache* cache, const char* key, const char* dataset_type, const char* dataset_id, AxConstTensorView value, float fitness, float normalized_metabolic_burn)`
* `AxWorkingMemory_FlagAnomaly(AxWorkingMemoryCache* cache, const char* key, AxConstTensorView deduced_constraint)`
* `AxWorkingMemory_ClearAnomalies(AxWorkingMemoryCache* cache)`
* `AxWorkingMemory_ApplyTimeDecay(AxWorkingMemoryCache* cache, float factor, float floor_value)`
* `AxWorkingMemory_CosineHit(AxWorkingMemoryCache* cache, AxConstTensorView query, float threshold, float* out_values, uint32_t out_value_count, AxCacheMatch* out_match)`

## Low-Level Connectome API (`connectome.h`)

* `AxHeuristicConfig_Default(AxHeuristicConfig* out_config)`
* `AxSystemMetabolism_Default(AxSystemMetabolism* out_state)`
* `AxMetabolicCriticConfig_Default(AxMetabolicCriticConfig* out_config)`
* `AxSystemMetabolism_ConfigureRelative(AxSystemMetabolism* state, float max_capacity, float fatigue_remaining_ratio, float zombie_activation_ratio, float zombie_critic_threshold)`
* `AxSystemMetabolism_Consume(AxSystemMetabolism* state, float amount)`
* `AxSystemMetabolism_Recharge(AxSystemMetabolism* state, float amount)`
* `AxSystemMetabolism_TriggerZombieMode(AxSystemMetabolism* state)`
* `AxSystemMetabolism_EnergyPercent(const AxSystemMetabolism* state)`
* `AxSystemMetabolism_CriticThreshold(const AxSystemMetabolism* state)`
* `AxSystemMetabolism_CanDeepThink(const AxSystemMetabolism* state)`
* `AxSignalProfile_Analyze(AxConstTensorView input, const AxHeuristicConfig* config, AxSignalProfile* out_profile)`
* `AxConnectome_DeduceGeometricGap(AxConstTensorView current_state, AxConstTensorView required_next_state, AxTensorView output)`
* `AxConnectome_RouteCandidate(AxConstTensorView target, const AxSignalProfile* profile, const AxWorkingMemoryCache* cache, const AxSystemMetabolism* metabolism, uint32_t iteration, AxTensorView output, AxTensorView scratch, AxTensorOpCandidate* out_candidate)`
* `AxConnectome_CalculateThermodynamicCost(const AxTensorOpCandidate* candidate, const AxSignalProfile* profile)`
* `AxConnectome_PassesCriticThreshold(const AxTensorOpCandidate* candidate, const AxSignalProfile* profile, const AxSystemMetabolism* metabolism)`
* `AxMetabolicCritic_Tick(AxGeneRuntimeState* genes, uint32_t gene_count, const char* focused_gene_id, const AxMetabolicCriticConfig* config, float* io_noise_floor, AxSceneTickReport* out_report)`

## Runtime Behavior Notes

Implementation details that affect integration:

* Inputs are sanitized (`NaN`/`Inf` become `0`) before most math paths.
* Several high-level APIs auto-project to `hdc_dim` when `value_count != hdc_dim`:
  `AxCore_StoreEpisode`, `AxCore_RecallSimilar`, `AxCore_PromoteWorkingMemory`, `AxCore_QueryWorkingMemory`, `AxCore_RouteCandidate`, and `AxCore_DeduceGeometricGap`.
* `AxCore_AnalyzeSignal` does **not** auto-project; it profiles the vector exactly as given.
* `AxCore_ScanManifoldEntropy` requires exact dimensions:
  `query_value_count == dim` and candidate flat array length = `candidate_count * dim`.
* Optional output buffers are supported in recall/query APIs by passing `out_values = nullptr`; metadata outputs remain required where specified.
* `AxCore_RechargeMetabolism(handle, amount <= 0)` performs a full recharge.
* `AxCore_Consolidate` also performs a full recharge and clears episodic memory.
* Consolidation defaults are strict (`consolidation_min_fitness = 0.95`); demos often lower this (for example `0.20`) to force visible promotions.
* `AxCore_ProjectManifold` returns `AX_STATUS_BUFFER_TOO_SMALL` if output buffer is shorter than `sample_count * output_dim`.
* `AxCore_BatchSequenceSimilarity` returns `0.0f` on invalid input instead of an `AxStatus`.

## Minimal C Usage Pattern

```c
#include "axcore/axcore.h"

AxCoreHandle* handle = NULL;
AxCoreCreateInfo info;
AxCore_GetDefaultCreateInfo(&info);
info.hdc_dim = 1024;

if (AxCore_Create(&info, &handle) != AX_STATUS_OK) {
    return 1;
}

float signal[256] = {0};
AxSignalProfile profile;
AxCore_AnalyzeSignal(handle, signal, 256, &profile);
AxCore_StoreEpisode(handle, signal, 256);
AxCore_Consolidate(handle);
AxCore_Destroy(handle);
```

## World-Model Demo

### What The Demo Is

`world_model_demo.py` demonstrates AxCore as an autonomous observer across three synthetic environments:

* `Monolith (Sine)`
* `Butterfly (Chaos)`
* `Gravity Well`

It performs signal ingestion, metabolic burn, manifold scan, episodic storage, sleep consolidation, and post-sleep replay.

### How To Run

```powershell
python .\world_model_demo.py
```

Useful options:

* `--frames-per-environment`
* `--collapse-energy-floor`
* `--metabolic-burn-scale`
* `--consolidation-min-fitness`
* `--working-memory-threshold`

### How To Read Demo Output

* `[FLOW|ZOMBIE]`: metabolic mode.
* `energy`: remaining budget.
* `entropy`: current signal entropy estimate.
* `grounding`: manifold grounding score.
* `scan`: manifold label (`bootstrap`, `void`, `diffuse`, `ambiguous`, `grounded`).
* `signal`: profile label (`balanced`, `skewed`, `high_entropy`, `sparse`).
* `wm`: working-memory hit summary (`-` when none).
* `METABOLIC COLLAPSE`: energy fell below collapse floor.
* `Consolidation complete`: sleep cycle finished.
* `Sleep replay ... -> consolidated_trace_*`: replay successfully matched consolidated memory.

## Aperiodic Manifold Experiment

### What The Experiment Is

`aperiodic_manifold_experiment.py` stress-tests AxCore with a deterministic, non-repeating geometry stream (Einstein/Spectre-inspired proxy).

It is designed to probe:

* **Coherent grounding** on unseen holdout patches.
* **Ambiguity separation** in non-periodic structure.
* **Memory/routing behavior** under aperiodic ingest.
* **Consolidation + post-sleep behavior** on the same substrate.

Unlike the world demo, this script intentionally exercises a broad slice of the high-level DLL API in one run (routing, manifold scan, episodic + working memory, geometric gap, projection, sequence similarity, and metabolism).

### How To Run

```powershell
python .\aperiodic_manifold_experiment.py
```

For the full stress target mentioned in notes:

```powershell
python .\aperiodic_manifold_experiment.py --hdc-dim 65536
```

Useful options:

* `--train-steps`
* `--holdout-steps`
* `--holdout-offset`
* `--patch-side`
* `--patch-stride`
* `--scan-candidate-limit`
* `--working-memory-threshold`
* `--consolidation-min-fitness`
* `--entropy-burn-scale`
* `--collapse-energy-floor`

### How To Read Experiment Output

* `[train]`: in-distribution ingest phase.
* `[hold]`: unseen far-field holdout phase.
* `grounding`: manifold grounding score (higher suggests stronger law-consistent structure match).
* `ambiguity`: best-vs-second-best separation in scan results.
* `wm_hit_rate`: fraction of training queries that hit working memory.
* `route_fitness` / `route_cost`: connectome route quality vs metabolic burn pressure.
* `sequence_similarity`: bundled sequence coherence against current target.
* `Hypothesis: coherent_grounding=... wide_ambiguity=...`: pass/fail flags against configured thresholds.
* `Projected ... ranges`: compact manifold projection sanity check.

## Mathematical Substrate

AxCore uses holographic vector symbolic architecture (VSA):

* **Bundling:** Merge patterns into shared representational memory.
* **Permutation:** Encode sequence/time by index shifts.
* **Cosine Similarity:** Measure angular relation in manifold space.

## Use Cases

* **Signal Intelligence:** Categorize audio, grid, and molecular streams as tensors.
* **Holographic Search:** Recall patterns from episodic traces without classic DB indexes.
* **Geometric Computation:** Solve structural tasks via nearest-neighbor manifold interpolation.
* **Predictive Maintenance:** Track anomaly drift through grounding and entropy signals.
* **Metabolic AI:** Run efficient, low-overhead cognitive loops on commodity hardware.
