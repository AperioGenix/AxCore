#include "axcore_internal.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace
{
uint32_t AlignUp(uint32_t value, uint32_t alignment)
{
    const uint32_t safe_alignment = alignment == 0u ? 1u : alignment;
    const uint32_t mask = safe_alignment - 1u;
    return (value + mask) & ~mask;
}

uint32_t AddAligned(uint32_t total, uint32_t bytes, uint32_t alignment)
{
    const uint32_t aligned = AlignUp(total, alignment);
    return aligned + bytes;
}

AxCoreCreateInfo NormalizeCreateInfo(const AxCoreCreateInfo* create_info)
{
    AxCoreCreateInfo normalized{};
    normalized.hdc_dim = axcore::kDefaultHdcDim;
    normalized.working_memory_capacity = axcore::kDefaultWorkingMemoryCapacity;
    normalized.episodic_recent_limit = axcore::kDefaultRecentLimit;
    normalized.episodic_max_levels = axcore::kDefaultMaxLevels;
    normalized.arena_bytes = 0u;

    if (create_info != nullptr)
    {
        if (create_info->hdc_dim != 0u)
        {
            normalized.hdc_dim = create_info->hdc_dim;
        }
        if (create_info->working_memory_capacity != 0u)
        {
            normalized.working_memory_capacity = create_info->working_memory_capacity;
        }
        if (create_info->episodic_recent_limit != 0u)
        {
            normalized.episodic_recent_limit = create_info->episodic_recent_limit;
        }
        if (create_info->episodic_max_levels != 0u)
        {
            normalized.episodic_max_levels = create_info->episodic_max_levels;
        }
        normalized.arena_bytes = create_info->arena_bytes;
    }

    return normalized;
}
} // namespace

namespace
{
float ComputeBlockPeakMagnitude(const AxTraceBlock& block, uint32_t dim)
{
    float peak = 0.0f;
    for (uint32_t i = 0u; i < dim; ++i)
    {
        float value = axcore::Sanitize(block.summary[i]);
        if (value < 0.0f)
        {
            value = -value;
        }
        if (value > peak)
        {
            peak = value;
        }
    }
    return peak;
}

float ComputeBlockFitness(const AxEpisodicMemory& episodic, const AxTraceBlock& block)
{
    if (episodic.total_stored == 0u)
    {
        return 0.0f;
    }

    const float span_ratio = static_cast<float>(block.span) / static_cast<float>(episodic.total_stored);
    const float peak_similarity = ComputeBlockPeakMagnitude(block, episodic.dim);
    return axcore::Clamp(span_ratio > peak_similarity ? span_ratio : peak_similarity, 0.0f, 1.0f);
}

bool WasLevelSelected(const uint32_t* selected_levels, uint32_t selected_count, uint32_t level)
{
    for (uint32_t i = 0u; i < selected_count; ++i)
    {
        if (selected_levels[i] == level)
        {
            return true;
        }
    }
    return false;
}

uint64_t Mix64(uint64_t value)
{
    value += 0x9E3779B97F4A7C15ull;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ull;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBull;
    return value ^ (value >> 31);
}

void ProjectRowHashed(
    const float* input_row,
    uint32_t input_dim,
    uint32_t output_dim,
    uint32_t projection_seed,
    float* out_row)
{
    static constexpr uint64_t kTapOffsets[3] = {
        0xA24BAED4963EE407ull,
        0x9FB21C651E98DF25ull,
        0xC13FA9A902A6328Full,
    };
    static constexpr float kTapWeights[3] = {1.0f, 0.5f, 0.5f};

    for (uint32_t i = 0u; i < input_dim; ++i)
    {
        const float value = axcore::Sanitize(input_row[i]);
        if (value == 0.0f)
        {
            continue;
        }

        const uint64_t base = (static_cast<uint64_t>(projection_seed) << 32) ^ static_cast<uint64_t>(i);
        for (uint32_t tap = 0u; tap < 3u; ++tap)
        {
            const uint64_t h = Mix64(base ^ kTapOffsets[tap]);
            const uint32_t slot = static_cast<uint32_t>(h % static_cast<uint64_t>(output_dim));
            const float sign = ((h >> 63) == 0u) ? 1.0f : -1.0f;
            out_row[slot] += sign * value * kTapWeights[tap];
        }
    }
}
} // namespace

uint32_t axcore::ComputeRequiredArenaBytes(const AxCoreCreateInfo* create_info)
{
    const AxCoreCreateInfo normalized = NormalizeCreateInfo(create_info);
    uint32_t total = 0u;
    const uint32_t dim = normalized.hdc_dim;
    const uint32_t cache_capacity = normalized.working_memory_capacity;
    const uint32_t recent_limit = normalized.episodic_recent_limit;
    const uint32_t max_levels = normalized.episodic_max_levels;

    total = AddAligned(total, sizeof(AxTraceBlock) * max_levels, alignof(AxTraceBlock));
    total = AddAligned(total, sizeof(AxRecentTrace) * recent_limit, alignof(AxRecentTrace));
    total = AddAligned(total, sizeof(float) * max_levels * dim, alignof(float));
    total = AddAligned(total, sizeof(float) * recent_limit * dim, alignof(float));
    total = AddAligned(total, sizeof(float) * dim, alignof(float));
    total = AddAligned(total, sizeof(float) * dim, alignof(float));

    total = AddAligned(total, sizeof(AxWorkingMemoryEntry) * cache_capacity, alignof(AxWorkingMemoryEntry));
    total = AddAligned(total, sizeof(uint32_t) * cache_capacity, alignof(uint32_t));
    total = AddAligned(total, sizeof(float) * dim, alignof(float));
    total = AddAligned(total, sizeof(float) * cache_capacity * dim, alignof(float));
    total = AddAligned(total, sizeof(float) * cache_capacity * dim, alignof(float));

    total = AddAligned(total, sizeof(float) * dim, alignof(float));
    total = AddAligned(total, sizeof(float) * dim, alignof(float));

    total = AddAligned(total, 256u, 16u);
    return total;
}

AxStatus axcore::InitializeEngine(Engine* engine, const AxCoreCreateInfo* create_info)
{
    if (engine == nullptr)
    {
        return AX_STATUS_INVALID_ARGUMENT;
    }

    const AxCoreCreateInfo normalized = NormalizeCreateInfo(create_info);
    const uint32_t required_bytes = ComputeRequiredArenaBytes(&normalized);
    const uint32_t arena_bytes = normalized.arena_bytes == 0u ? required_bytes : normalized.arena_bytes;
    if (arena_bytes < required_bytes)
    {
        return AX_STATUS_BUFFER_TOO_SMALL;
    }

    ZeroBytes(engine, sizeof(Engine));
    engine->hdc_dim = normalized.hdc_dim;
    engine->arena_bytes = arena_bytes;
    engine->arena_memory = std::malloc(arena_bytes);
    if (engine->arena_memory == nullptr)
    {
        return AX_STATUS_OUT_OF_MEMORY;
    }

    ZeroBytes(engine->arena_memory, arena_bytes);
    AxArena_Init(&engine->arena, engine->arena_memory, arena_bytes);
    AxHeuristicConfig_Default(&engine->heuristics);
    AxSystemMetabolism_Default(&engine->metabolism);

    AxStatus status = AxEpisodic_Init(
        &engine->episodic,
        &engine->arena,
        engine->hdc_dim,
        normalized.episodic_max_levels,
        normalized.episodic_recent_limit);
    if (status != AX_STATUS_OK)
    {
        ShutdownEngine(engine);
        return status;
    }

    status = AxWorkingMemory_Init(
        &engine->working_memory,
        &engine->arena,
        engine->hdc_dim,
        normalized.working_memory_capacity);
    if (status != AX_STATUS_OK)
    {
        ShutdownEngine(engine);
        return status;
    }

    engine->route_target = static_cast<float*>(AxArena_Alloc(&engine->arena, sizeof(float) * engine->hdc_dim, alignof(float)));
    engine->route_scratch = static_cast<float*>(AxArena_Alloc(&engine->arena, sizeof(float) * engine->hdc_dim, alignof(float)));
    if (engine->route_target == nullptr || engine->route_scratch == nullptr)
    {
        ShutdownEngine(engine);
        return AX_STATUS_OUT_OF_MEMORY;
    }

    return AX_STATUS_OK;
}

void axcore::ShutdownEngine(Engine* engine)
{
    if (engine == nullptr)
    {
        return;
    }

    if (engine->arena_memory != nullptr)
    {
        std::free(engine->arena_memory);
    }
    ZeroBytes(engine, sizeof(Engine));
}

void axcore::ResetEngine(Engine* engine)
{
    if (engine == nullptr)
    {
        return;
    }

    AxEpisodic_Clear(&engine->episodic);
    AxWorkingMemory_Clear(&engine->working_memory);
    AxSystemMetabolism_Recharge(&engine->metabolism, -1.0f);
    ZeroVector(engine->route_target, engine->hdc_dim);
    ZeroVector(engine->route_scratch, engine->hdc_dim);
}

AxStatus AXCORE_CALL AxCore_GetVersion(AxVersion* out_version)
{
    if (out_version == nullptr)
    {
        return AX_STATUS_INVALID_ARGUMENT;
    }

    *out_version = axcore::kVersion;
    return AX_STATUS_OK;
}

void AXCORE_CALL AxCore_GetDefaultCreateInfo(AxCoreCreateInfo* out_info)
{
    if (out_info == nullptr)
    {
        return;
    }

    *out_info = NormalizeCreateInfo(nullptr);
}

void AXCORE_CALL AxCore_GetDefaultHeuristics(AxHeuristicConfig* out_config)
{
    AxHeuristicConfig_Default(out_config);
}

void AXCORE_CALL AxCore_GetDefaultMetabolism(AxSystemMetabolism* out_state)
{
    AxSystemMetabolism_Default(out_state);
}

AxStatus AXCORE_CALL AxCore_ComputeRequiredArenaBytes(const AxCoreCreateInfo* create_info, uint32_t* out_bytes)
{
    if (out_bytes == nullptr)
    {
        return AX_STATUS_INVALID_ARGUMENT;
    }

    *out_bytes = axcore::ComputeRequiredArenaBytes(create_info);
    return AX_STATUS_OK;
}

AxStatus AXCORE_CALL AxCore_Create(const AxCoreCreateInfo* create_info, AxCoreHandle** out_handle)
{
    if (out_handle == nullptr)
    {
        return AX_STATUS_INVALID_ARGUMENT;
    }

    *out_handle = nullptr;
    AxCoreHandle* handle = static_cast<AxCoreHandle*>(std::malloc(sizeof(AxCoreHandle)));
    if (handle == nullptr)
    {
        return AX_STATUS_OUT_OF_MEMORY;
    }

    axcore::ZeroBytes(handle, sizeof(AxCoreHandle));
    const AxStatus status = axcore::InitializeEngine(&handle->engine, create_info);
    if (status != AX_STATUS_OK)
    {
        std::free(handle);
        return status;
    }

    *out_handle = handle;
    return AX_STATUS_OK;
}

void AXCORE_CALL AxCore_Destroy(AxCoreHandle* handle)
{
    if (handle == nullptr)
    {
        return;
    }

    axcore::ShutdownEngine(&handle->engine);
    std::free(handle);
}

void AXCORE_CALL AxCore_Reset(AxCoreHandle* handle)
{
    if (handle == nullptr)
    {
        return;
    }

    axcore::ResetEngine(&handle->engine);
}

AxStatus AXCORE_CALL AxCore_SetHeuristics(AxCoreHandle* handle, const AxHeuristicConfig* config)
{
    if (handle == nullptr || config == nullptr)
    {
        return AX_STATUS_INVALID_ARGUMENT;
    }

    handle->engine.heuristics = *config;
    return AX_STATUS_OK;
}

AxStatus AXCORE_CALL AxCore_GetMetabolism(const AxCoreHandle* handle, AxSystemMetabolism* out_state)
{
    if (handle == nullptr || out_state == nullptr)
    {
        return AX_STATUS_INVALID_ARGUMENT;
    }

    *out_state = handle->engine.metabolism;
    return AX_STATUS_OK;
}

AxStatus AXCORE_CALL AxCore_ConsumeMetabolism(AxCoreHandle* handle, float amount)
{
    if (handle == nullptr || amount < 0.0f)
    {
        return AX_STATUS_INVALID_ARGUMENT;
    }

    AxSystemMetabolism_Consume(&handle->engine.metabolism, amount);
    return AX_STATUS_OK;
}

AxStatus AXCORE_CALL AxCore_RechargeMetabolism(AxCoreHandle* handle, float amount)
{
    if (handle == nullptr)
    {
        return AX_STATUS_INVALID_ARGUMENT;
    }

    AxSystemMetabolism_Recharge(&handle->engine.metabolism, amount);
    return AX_STATUS_OK;
}

AxStatus AXCORE_CALL AxCore_StoreEpisode(AxCoreHandle* handle, const float* values, uint32_t value_count)
{
    if (handle == nullptr)
    {
        return AX_STATUS_INVALID_ARGUMENT;
    }

    const AxStatus prepare_status =
        axcore::PrepareHypervector(values, value_count, handle->engine.hdc_dim, handle->engine.route_target);
    if (prepare_status != AX_STATUS_OK)
    {
        return prepare_status;
    }

    return AxEpisodic_Store(&handle->engine.episodic, axcore::MakeConstView(handle->engine.route_target, handle->engine.hdc_dim));
}

AxStatus AXCORE_CALL AxCore_RecallSimilar(
    const AxCoreHandle* handle,
    const float* values,
    uint32_t value_count,
    float* out_values,
    uint32_t out_value_count,
    AxRecallResult* out_result)
{
    if (handle == nullptr)
    {
        return AX_STATUS_INVALID_ARGUMENT;
    }

    const AxStatus prepare_status =
        axcore::PrepareHypervector(values, value_count, handle->engine.hdc_dim, handle->engine.route_target);
    if (prepare_status != AX_STATUS_OK)
    {
        return prepare_status;
    }

    return AxEpisodic_RecallSimilar(
        &handle->engine.episodic,
        axcore::MakeConstView(handle->engine.route_target, handle->engine.hdc_dim),
        out_values,
        out_value_count,
        out_result);
}

AxStatus AXCORE_CALL AxCore_RecallStepsAgo(
    const AxCoreHandle* handle,
    uint64_t steps_ago,
    float* out_values,
    uint32_t out_value_count,
    AxRecallResult* out_result)
{
    if (handle == nullptr)
    {
        return AX_STATUS_INVALID_ARGUMENT;
    }

    return AxEpisodic_RecallStepsAgo(&handle->engine.episodic, steps_ago, out_values, out_value_count, out_result);
}

AxStatus AXCORE_CALL AxCore_Consolidate(AxCoreHandle* handle)
{
    if (handle == nullptr)
    {
        return AX_STATUS_INVALID_ARGUMENT;
    }

    AxEpisodicMemory& episodic = handle->engine.episodic;
    if (episodic.total_stored == 0u)
    {
        AxSystemMetabolism_Recharge(&handle->engine.metabolism, -1.0f);
        return AX_STATUS_OK;
    }

    const float min_fitness = axcore::Clamp(handle->engine.heuristics.consolidation_min_fitness, 0.0f, 1.0f);
    uint32_t top_limit = handle->engine.heuristics.consolidation_top_limit;
    if (top_limit == 0u)
    {
        top_limit = 64u;
    }
    if (top_limit > 64u)
    {
        top_limit = 64u;
    }

    uint32_t promoted_count = 0u;
    uint32_t selected_levels[64]{};
    while (promoted_count < top_limit)
    {
        int32_t best_level = -1;
        float best_fitness = -1.0f;
        uint32_t best_span = 0u;

        for (uint32_t level = 0u; level < episodic.max_levels; ++level)
        {
            const AxTraceBlock& block = episodic.levels[level];
            if (block.valid == 0u || block.summary == nullptr)
            {
                continue;
            }
            if (WasLevelSelected(selected_levels, promoted_count, level))
            {
                continue;
            }

            const float fitness = ComputeBlockFitness(episodic, block);
            if (fitness < min_fitness)
            {
                continue;
            }

            if (best_level < 0 || fitness > best_fitness || (fitness == best_fitness && block.span > best_span))
            {
                best_level = static_cast<int32_t>(level);
                best_fitness = fitness;
                best_span = block.span;
            }
        }

        if (best_level < 0)
        {
            break;
        }

        const AxTraceBlock& block = episodic.levels[static_cast<uint32_t>(best_level)];
        char key[AXCORE_MAX_KEY_LENGTH + 1];
        std::snprintf(key, sizeof(key), "consolidated_trace_L%u_S%u", static_cast<uint32_t>(best_level), block.span);
        key[AXCORE_MAX_KEY_LENGTH] = '\0';

        const AxStatus status = AxWorkingMemory_Promote(
            &handle->engine.working_memory,
            key,
            "sleep_consolidation",
            "episodic_logtrace",
            axcore::MakeConstView(block.summary, episodic.dim),
            best_fitness,
            0.0f);
        if (status != AX_STATUS_OK)
        {
            return status;
        }

        selected_levels[promoted_count] = static_cast<uint32_t>(best_level);
        promoted_count += 1u;
    }

    AxEpisodic_Clear(&episodic);
    AxSystemMetabolism_Recharge(&handle->engine.metabolism, -1.0f);
    return AX_STATUS_OK;
}

AxStatus AXCORE_CALL AxCore_PromoteWorkingMemory(
    AxCoreHandle* handle,
    const char* key,
    const char* dataset_type,
    const char* dataset_id,
    const float* values,
    uint32_t value_count,
    float fitness,
    float normalized_metabolic_burn)
{
    if (handle == nullptr)
    {
        return AX_STATUS_INVALID_ARGUMENT;
    }

    const AxStatus prepare_status =
        axcore::PrepareHypervector(values, value_count, handle->engine.hdc_dim, handle->engine.route_target);
    if (prepare_status != AX_STATUS_OK)
    {
        return prepare_status;
    }

    return AxWorkingMemory_Promote(
        &handle->engine.working_memory,
        key,
        dataset_type,
        dataset_id,
        axcore::MakeConstView(handle->engine.route_target, handle->engine.hdc_dim),
        fitness,
        normalized_metabolic_burn);
}

AxStatus AXCORE_CALL AxCore_QueryWorkingMemory(
    AxCoreHandle* handle,
    const float* values,
    uint32_t value_count,
    float threshold,
    float* out_values,
    uint32_t out_value_count,
    AxCacheMatch* out_match)
{
    if (handle == nullptr)
    {
        return AX_STATUS_INVALID_ARGUMENT;
    }

    const AxStatus prepare_status =
        axcore::PrepareHypervector(values, value_count, handle->engine.hdc_dim, handle->engine.route_target);
    if (prepare_status != AX_STATUS_OK)
    {
        return prepare_status;
    }

    return AxWorkingMemory_CosineHit(
        &handle->engine.working_memory,
        axcore::MakeConstView(handle->engine.route_target, handle->engine.hdc_dim),
        threshold,
        out_values,
        out_value_count,
        out_match);
}

void AXCORE_CALL AxCore_ApplyWorkingMemoryDecay(AxCoreHandle* handle, float factor, float floor_value)
{
    if (handle == nullptr)
    {
        return;
    }

    AxWorkingMemory_ApplyTimeDecay(&handle->engine.working_memory, factor, floor_value);
}

AxStatus AXCORE_CALL AxCore_AnalyzeSignal(
    const AxCoreHandle* handle,
    const float* values,
    uint32_t value_count,
    AxSignalProfile* out_profile)
{
    if (values == nullptr || value_count == 0u)
    {
        return AX_STATUS_INVALID_ARGUMENT;
    }

    AxHeuristicConfig defaults{};
    const AxHeuristicConfig* config = nullptr;
    if (handle != nullptr)
    {
        config = &handle->engine.heuristics;
    }
    else
    {
        AxHeuristicConfig_Default(&defaults);
        config = &defaults;
    }

    return AxSignalProfile_Analyze(axcore::MakeConstView(values, value_count), config, out_profile);
}

AxStatus AXCORE_CALL AxCore_ScanManifoldEntropy(
    const AxCoreHandle* handle,
    const float* query_values,
    uint32_t query_value_count,
    const float* candidate_stack_flat,
    uint32_t candidate_count,
    uint32_t dim,
    AxManifoldScanResult* out_result)
{
    if (query_values == nullptr || candidate_stack_flat == nullptr || out_result == nullptr)
    {
        return AX_STATUS_INVALID_ARGUMENT;
    }
    if (candidate_count == 0u || dim == 0u || query_value_count != dim)
    {
        return AX_STATUS_INVALID_ARGUMENT;
    }

    axcore::ClearManifoldScanResult(out_result);

    double sum = 0.0;
    double sum_sq = 0.0;
    float best = -1.0f;
    float second = -1.0f;

    thread_local std::vector<float> weights;
    weights.resize(candidate_count);
    double weight_sum = 0.0;

    for (uint32_t i = 0u; i < candidate_count; ++i)
    {
        const float* candidate = candidate_stack_flat + (static_cast<std::size_t>(i) * dim);
        const float similarity = axcore::CosineRaw(query_values, candidate, dim);
        sum += static_cast<double>(similarity);
        sum_sq += static_cast<double>(similarity) * static_cast<double>(similarity);

        if (similarity > best)
        {
            second = best;
            best = similarity;
        }
        else if (similarity > second)
        {
            second = similarity;
        }

        const double shifted = std::exp(static_cast<double>(similarity) * 8.0);
        weights[i] = static_cast<float>(shifted);
        weight_sum += shifted;
    }

    const double count = static_cast<double>(candidate_count);
    const double mean = sum / count;
    double variance = (sum_sq / count) - (mean * mean);
    if (variance < 0.0)
    {
        variance = 0.0;
    }
    const double stddev = std::sqrt(variance);

    double entropy = 0.0;
    if (weight_sum > 0.0 && candidate_count > 1u)
    {
        for (uint32_t i = 0u; i < candidate_count; ++i)
        {
            const double p = static_cast<double>(weights[i]) / weight_sum;
            if (p > 1.0e-12)
            {
                entropy -= p * std::log(p);
            }
        }
        entropy /= std::log(static_cast<double>(candidate_count));
    }

    const float ambiguity_gap = best - second;
    const float grounding_score = axcore::Clamp01(((best - static_cast<float>(mean)) * 1.5f) + (ambiguity_gap * 2.0f));
    const float void_score = axcore::Clamp01(((0.20f - best) / 0.20f) + (static_cast<float>(entropy) * 0.25f));

    const char* label = "diffuse";
    if (best < 0.15f)
    {
        label = "void";
    }
    else if (ambiguity_gap < 0.05f)
    {
        label = "ambiguous";
    }
    else if (best >= 0.30f)
    {
        label = "grounded";
    }

    if (handle != nullptr && handle->engine.metabolism.zombie_mode_active != 0u && best < 0.25f)
    {
        label = "void";
    }

    out_result->candidate_count = candidate_count;
    out_result->best_similarity = best;
    out_result->second_best_similarity = second;
    out_result->mean_similarity = static_cast<float>(mean);
    out_result->similarity_stddev = static_cast<float>(stddev);
    out_result->ambiguity_gap = ambiguity_gap;
    out_result->grounding_score = grounding_score;
    out_result->void_score = void_score;
    out_result->entropy = static_cast<float>(entropy);
    axcore::CopyString(out_result->label, sizeof(out_result->label), label);
    return AX_STATUS_OK;
}

AxStatus AXCORE_CALL AxCore_ProjectManifold(
    const float* vector_stack_flat,
    uint32_t sample_count,
    uint32_t input_dim,
    uint32_t output_dim,
    uint32_t normalize_rows,
    uint32_t projection_seed,
    float* out_projected_flat,
    uint32_t out_value_count)
{
    if (vector_stack_flat == nullptr || out_projected_flat == nullptr)
    {
        return AX_STATUS_INVALID_ARGUMENT;
    }
    if (sample_count == 0u || input_dim == 0u || output_dim == 0u)
    {
        return AX_STATUS_INVALID_ARGUMENT;
    }

    const uint64_t required_values_64 = static_cast<uint64_t>(sample_count) * static_cast<uint64_t>(output_dim);
    if (required_values_64 > 0xFFFFFFFFull)
    {
        return AX_STATUS_LIMIT_EXCEEDED;
    }
    const uint32_t required_values = static_cast<uint32_t>(required_values_64);
    if (out_value_count < required_values)
    {
        return AX_STATUS_BUFFER_TOO_SMALL;
    }

    for (uint32_t sample_idx = 0u; sample_idx < sample_count; ++sample_idx)
    {
        const float* input_row = vector_stack_flat + (static_cast<std::size_t>(sample_idx) * input_dim);
        float* out_row = out_projected_flat + (static_cast<std::size_t>(sample_idx) * output_dim);
        axcore::ZeroVector(out_row, output_dim);

        if (input_dim == output_dim)
        {
            for (uint32_t d = 0u; d < input_dim; ++d)
            {
                out_row[d] = axcore::Sanitize(input_row[d]);
            }
        }
        else
        {
            ProjectRowHashed(input_row, input_dim, output_dim, projection_seed, out_row);
        }

        if (normalize_rows != 0u)
        {
            axcore::NormalizeRawInPlace(out_row, output_dim);
        }
    }

    return AX_STATUS_OK;
}

AxStatus AXCORE_CALL AxCore_RouteCandidate(
    AxCoreHandle* handle,
    const float* values,
    uint32_t value_count,
    uint32_t iteration,
    float* out_values,
    uint32_t out_value_count,
    AxTensorOpCandidate* out_candidate,
    AxSignalProfile* out_profile)
{
    if (handle == nullptr || out_candidate == nullptr || out_values == nullptr)
    {
        return AX_STATUS_INVALID_ARGUMENT;
    }
    if (out_value_count < handle->engine.hdc_dim)
    {
        return AX_STATUS_BUFFER_TOO_SMALL;
    }

    AxSignalProfile local_profile{};
    AxSignalProfile* profile = out_profile != nullptr ? out_profile : &local_profile;
    AxStatus status = AxCore_AnalyzeSignal(handle, values, value_count, profile);
    if (status != AX_STATUS_OK)
    {
        return status;
    }

    status = axcore::PrepareHypervector(values, value_count, handle->engine.hdc_dim, handle->engine.route_target);
    if (status != AX_STATUS_OK)
    {
        return status;
    }

    status = AxConnectome_RouteCandidate(
        axcore::MakeConstView(handle->engine.route_target, handle->engine.hdc_dim),
        profile,
        &handle->engine.working_memory,
        &handle->engine.metabolism,
        iteration,
        axcore::MakeView(out_values, handle->engine.hdc_dim),
        axcore::MakeView(handle->engine.route_scratch, handle->engine.hdc_dim),
        out_candidate);
    if (status != AX_STATUS_OK)
    {
        return status;
    }

    AxSystemMetabolism_Consume(&handle->engine.metabolism, out_candidate->cost);
    return AX_STATUS_OK;
}

AxStatus AXCORE_CALL AxCore_DeduceGeometricGap(
    AxCoreHandle* handle,
    const float* current_state,
    uint32_t current_count,
    const float* required_next_state,
    uint32_t required_count,
    float* out_values,
    uint32_t out_value_count)
{
    if (handle == nullptr || out_values == nullptr)
    {
        return AX_STATUS_INVALID_ARGUMENT;
    }
    if (out_value_count < handle->engine.hdc_dim)
    {
        return AX_STATUS_BUFFER_TOO_SMALL;
    }

    AxStatus status = axcore::PrepareHypervector(current_state, current_count, handle->engine.hdc_dim, handle->engine.route_target);
    if (status != AX_STATUS_OK)
    {
        return status;
    }

    status = axcore::PrepareHypervector(required_next_state, required_count, handle->engine.hdc_dim, handle->engine.route_scratch);
    if (status != AX_STATUS_OK)
    {
        return status;
    }

    return AxConnectome_DeduceGeometricGap(
        axcore::MakeConstView(handle->engine.route_target, handle->engine.hdc_dim),
        axcore::MakeConstView(handle->engine.route_scratch, handle->engine.hdc_dim),
        axcore::MakeView(out_values, handle->engine.hdc_dim));
}

float AXCORE_CALL AxCore_BatchSequenceSimilarity(
    const float* vector_stack_flat,
    const uint32_t* sequence_indices,
    uint32_t index_count,
    const float* target_vector,
    uint32_t dim)
{
    if (vector_stack_flat == nullptr || sequence_indices == nullptr || target_vector == nullptr)
    {
        return 0.0f;
    }
    if (index_count == 0u || dim == 0u)
    {
        return 0.0f;
    }

    thread_local std::vector<float> bundle_buffer;
    bundle_buffer.resize(dim);
    axcore::ZeroVector(bundle_buffer.data(), dim);

    for (uint32_t i = 0u; i < index_count; ++i)
    {
        const float* vector = vector_stack_flat + (static_cast<std::size_t>(sequence_indices[i]) * dim);
        for (uint32_t d = 0u; d < dim; ++d)
        {
            bundle_buffer[d] += axcore::Sanitize(vector[d]);
        }
    }

    if (!axcore::NormalizeRawInPlace(bundle_buffer.data(), dim))
    {
        return 0.0f;
    }

    return axcore::CosineRaw(bundle_buffer.data(), target_vector, dim);
}
