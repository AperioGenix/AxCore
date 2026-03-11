#pragma once

#include "axcore/axcore.h"

#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace axcore
{
inline constexpr AxVersion kVersion = {0u, 2u, 0u, 1u};
inline constexpr uint32_t kDefaultHdcDim = 1024u;
inline constexpr uint32_t kDefaultWorkingMemoryCapacity = 128u;
inline constexpr uint32_t kDefaultRecentLimit = 256u;
inline constexpr uint32_t kDefaultMaxLevels = 32u;

struct Engine
{
    AxHeuristicConfig heuristics;
    AxSystemMetabolism metabolism;
    AxLinearArena arena;
    void* arena_memory;
    uint32_t arena_bytes;
    uint32_t hdc_dim;
    AxEpisodicMemory episodic;
    AxWorkingMemoryCache working_memory;
    float* route_target;
    float* route_scratch;
};

inline float Clamp(float value, float min_value, float max_value)
{
    if (value < min_value)
    {
        return min_value;
    }
    if (value > max_value)
    {
        return max_value;
    }
    return value;
}

inline float Clamp01(float value)
{
    return Clamp(value, 0.0f, 1.0f);
}

inline bool IsFinite(float value)
{
    return std::isfinite(static_cast<double>(value)) != 0;
}

inline float Sanitize(float value)
{
    return IsFinite(value) ? value : 0.0f;
}

inline void ZeroBytes(void* data, std::size_t bytes)
{
    if (data != nullptr && bytes > 0u)
    {
        std::memset(data, 0, bytes);
    }
}

inline void ZeroVector(float* data, uint32_t count)
{
    ZeroBytes(data, static_cast<std::size_t>(count) * sizeof(float));
}

inline void CopyString(char* dst, std::size_t dst_size, const char* src)
{
    if (dst == nullptr || dst_size == 0u)
    {
        return;
    }

    if (src == nullptr)
    {
        dst[0] = '\0';
        return;
    }

    std::size_t index = 0u;
    for (; index + 1u < dst_size && src[index] != '\0'; ++index)
    {
        dst[index] = src[index];
    }
    dst[index] = '\0';
}

inline void CopyTrimmed(char* dst, std::size_t dst_size, const char* src, bool lowercase)
{
    if (dst == nullptr || dst_size == 0u)
    {
        return;
    }

    dst[0] = '\0';
    if (src == nullptr)
    {
        return;
    }

    const char* start = src;
    while (*start != '\0' && std::isspace(static_cast<unsigned char>(*start)) != 0)
    {
        ++start;
    }

    const char* end = start + std::strlen(start);
    while (end > start && std::isspace(static_cast<unsigned char>(end[-1])) != 0)
    {
        --end;
    }

    std::size_t length = static_cast<std::size_t>(end - start);
    if (length + 1u > dst_size)
    {
        length = dst_size - 1u;
    }

    for (std::size_t i = 0u; i < length; ++i)
    {
        unsigned char ch = static_cast<unsigned char>(start[i]);
        dst[i] = lowercase ? static_cast<char>(std::tolower(ch)) : static_cast<char>(ch);
    }
    dst[length] = '\0';
}

inline void ClearShape(AxShape* shape)
{
    if (shape == nullptr)
    {
        return;
    }

    shape->ndim = 0u;
    shape->total = 0u;
    for (uint32_t i = 0u; i < AXCORE_MAX_DIMS; ++i)
    {
        shape->dims[i] = 0u;
    }
}

inline AxConstTensorView MakeConstView(const float* data, uint32_t total)
{
    AxConstTensorView view{};
    view.data = data;
    view.shape.ndim = total == 0u ? 0u : 1u;
    view.shape.total = total;
    for (uint32_t i = 0u; i < AXCORE_MAX_DIMS; ++i)
    {
        view.shape.dims[i] = 0u;
    }
    if (total != 0u)
    {
        view.shape.dims[0] = total;
    }
    return view;
}

inline AxTensorView MakeView(float* data, uint32_t total)
{
    AxTensorView view{};
    view.data = data;
    view.shape.ndim = total == 0u ? 0u : 1u;
    view.shape.total = total;
    for (uint32_t i = 0u; i < AXCORE_MAX_DIMS; ++i)
    {
        view.shape.dims[i] = 0u;
    }
    if (total != 0u)
    {
        view.shape.dims[0] = total;
    }
    return view;
}

inline uint64_t AbsDiff(uint64_t a, uint64_t b)
{
    return a >= b ? (a - b) : (b - a);
}

inline void ClearRecallResult(AxRecallResult* result)
{
    if (result == nullptr)
    {
        return;
    }

    result->found = 0u;
    result->similarity = -1.0f;
    result->stored_step = 0u;
    result->age_steps = 0u;
    result->level = 0u;
    result->span = 0u;
    result->source[0] = '\0';
}

inline void ClearCacheMatch(AxCacheMatch* match)
{
    if (match == nullptr)
    {
        return;
    }

    match->found = 0u;
    match->similarity = -1.0f;
    match->fitness = 0.0f;
    match->decay_score = 0.0f;
    match->last_metabolic_burn = -1.0f;
    match->average_metabolic_burn = -1.0f;
    match->burn_samples = 0u;
    match->hits = 0u;
    match->last_touch = 0u;
    match->is_anomaly = 0u;
    match->key[0] = '\0';
    match->dataset_type[0] = '\0';
    match->dataset_id[0] = '\0';
}

inline void ClearCandidate(AxTensorOpCandidate* candidate)
{
    if (candidate == nullptr)
    {
        return;
    }

    candidate->fitness = 0.0f;
    candidate->similarity = -1.0f;
    candidate->cost = 0.0f;
    candidate->strategy[0] = '\0';
}

inline void ClearProfile(AxSignalProfile* profile)
{
    if (profile == nullptr)
    {
        return;
    }

    profile->length = 0u;
    profile->mean = 0.0f;
    profile->standard_deviation = 0.0f;
    profile->skewness = 0.0f;
    profile->sparsity = 0.0f;
    profile->entropy = 0.0f;
    profile->unique_ratio = 0.0f;
    profile->range = 0.0f;
    profile->system1_similarity_threshold = 0.0f;
    profile->critic_acceptance_threshold = 0.0f;
    profile->deep_think_cost_bias = 0.0f;
    profile->label[0] = '\0';
}

inline void ClearManifoldScanResult(AxManifoldScanResult* result)
{
    if (result == nullptr)
    {
        return;
    }

    result->candidate_count = 0u;
    result->best_similarity = -1.0f;
    result->second_best_similarity = -1.0f;
    result->mean_similarity = 0.0f;
    result->similarity_stddev = 0.0f;
    result->ambiguity_gap = 0.0f;
    result->grounding_score = 0.0f;
    result->void_score = 1.0f;
    result->entropy = 0.0f;
    result->label[0] = '\0';
}

int32_t RoundBucket(float value);
bool NormalizeRawInPlace(float* values, uint32_t dim);
float CosineRaw(const float* lhs, const float* rhs, uint32_t dim);
AxStatus FoldNumericSignal(const float* values, uint32_t count, uint32_t target_dim, float* out_values);
AxStatus PrepareHypervector(const float* values, uint32_t count, uint32_t target_dim, float* out_values);
uint32_t ComputeRequiredArenaBytes(const AxCoreCreateInfo* create_info);
AxStatus InitializeEngine(Engine* engine, const AxCoreCreateInfo* create_info);
void ShutdownEngine(Engine* engine);
void ResetEngine(Engine* engine);
} // namespace axcore

struct AxCoreHandle
{
    axcore::Engine engine;
};
