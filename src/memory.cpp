#include "axcore_internal.h"

#include <utility>

namespace
{
void CopyMatch(AxCacheMatch* out_match, const AxWorkingMemoryEntry* entry, float similarity)
{
    if (out_match == nullptr || entry == nullptr)
    {
        return;
    }

    out_match->found = 1u;
    out_match->similarity = similarity;
    out_match->fitness = entry->fitness;
    out_match->decay_score = entry->decay_score;
    out_match->last_metabolic_burn = entry->last_metabolic_burn;
    out_match->average_metabolic_burn = entry->average_metabolic_burn;
    out_match->burn_samples = entry->burn_samples;
    out_match->hits = entry->hits;
    out_match->last_touch = entry->last_touch;
    out_match->is_anomaly = entry->is_anomaly;
    axcore::CopyString(out_match->key, sizeof(out_match->key), entry->key);
    axcore::CopyString(out_match->dataset_type, sizeof(out_match->dataset_type), entry->dataset_type);
    axcore::CopyString(out_match->dataset_id, sizeof(out_match->dataset_id), entry->dataset_id);
}

void ClearEntry(AxWorkingMemoryEntry* entry, uint32_t dim)
{
    if (entry == nullptr)
    {
        return;
    }

    entry->in_use = 0u;
    entry->key[0] = '\0';
    entry->dataset_type[0] = '\0';
    entry->dataset_id[0] = '\0';
    entry->fitness = 0.0f;
    entry->decay_score = 1.0f;
    entry->last_metabolic_burn = -1.0f;
    entry->average_metabolic_burn = -1.0f;
    entry->burn_samples = 0u;
    entry->hits = 0u;
    entry->last_touch = 0u;
    entry->is_anomaly = 0u;
    if (entry->value != nullptr)
    {
        axcore::ZeroVector(entry->value, dim);
    }
    if (entry->deduced_constraint != nullptr)
    {
        axcore::ZeroVector(entry->deduced_constraint, dim);
    }
}

int32_t FindEntryByKey(const AxWorkingMemoryCache* cache, const char* key)
{
    if (cache == nullptr || key == nullptr)
    {
        return -1;
    }

    for (uint32_t i = 0u; i < cache->capacity; ++i)
    {
        const AxWorkingMemoryEntry& entry = cache->entries[i];
        if (entry.in_use != 0u && std::strcmp(entry.key, key) == 0)
        {
            return static_cast<int32_t>(i);
        }
    }
    return -1;
}

int32_t FindFreeEntry(const AxWorkingMemoryCache* cache)
{
    if (cache == nullptr)
    {
        return -1;
    }

    for (uint32_t i = 0u; i < cache->capacity; ++i)
    {
        if (cache->entries[i].in_use == 0u)
        {
            return static_cast<int32_t>(i);
        }
    }
    return -1;
}

void TouchLru(AxWorkingMemoryCache* cache, uint32_t index)
{
    if (cache == nullptr || cache->count == 0u)
    {
        return;
    }

    uint32_t found = cache->count;
    for (uint32_t i = 0u; i < cache->count; ++i)
    {
        if (cache->lru_order[i] == index)
        {
            found = i;
            break;
        }
    }

    if (found == cache->count)
    {
        cache->lru_order[cache->count - 1u] = index;
        return;
    }

    for (uint32_t i = found; i + 1u < cache->count; ++i)
    {
        cache->lru_order[i] = cache->lru_order[i + 1u];
    }
    cache->lru_order[cache->count - 1u] = index;
}

void UpdateBurn(AxWorkingMemoryEntry* entry, float normalized_metabolic_burn)
{
    if (entry == nullptr || normalized_metabolic_burn < 0.0f)
    {
        return;
    }

    const float burn = axcore::Clamp01(normalized_metabolic_burn);
    entry->last_metabolic_burn = burn;
    if (entry->burn_samples == 0u)
    {
        entry->average_metabolic_burn = burn;
        entry->burn_samples = 1u;
        return;
    }

    const float weighted = entry->average_metabolic_burn * static_cast<float>(entry->burn_samples);
    entry->burn_samples += 1u;
    entry->average_metabolic_burn = (weighted + burn) / static_cast<float>(entry->burn_samples);
}

void CopyRecall(AxRecallResult* out_result, float similarity, uint64_t stored_step, uint64_t age_steps, uint32_t level, uint32_t span, const char* source)
{
    if (out_result == nullptr)
    {
        return;
    }

    out_result->found = 1u;
    out_result->similarity = similarity;
    out_result->stored_step = stored_step;
    out_result->age_steps = age_steps;
    out_result->level = level;
    out_result->span = span;
    axcore::CopyString(out_result->source, sizeof(out_result->source), source);
}

void PushRecent(AxEpisodicMemory* memory, const float* values)
{
    if (memory == nullptr || values == nullptr || memory->recent_limit == 0u)
    {
        return;
    }

    uint32_t slot = 0u;
    if (memory->recent_count < memory->recent_limit)
    {
        slot = (memory->recent_head + memory->recent_count) % memory->recent_limit;
        memory->recent_count += 1u;
    }
    else
    {
        slot = memory->recent_head;
        memory->recent_head = (memory->recent_head + 1u) % memory->recent_limit;
    }

    AxRecentTrace& trace = memory->recent[slot];
    trace.valid = 1u;
    trace.step = memory->step;
    for (uint32_t i = 0u; i < memory->dim; ++i)
    {
        trace.value[i] = values[i];
    }
}

void WeightedMerge(const float* older, uint32_t older_span, const float* newer, uint32_t newer_span, uint32_t dim, float* out_values)
{
    const double older_weight = older_span == 0u ? 1.0 : static_cast<double>(older_span);
    const double newer_weight = newer_span == 0u ? 1.0 : static_cast<double>(newer_span);

    for (uint32_t i = 0u; i < dim; ++i)
    {
        const double merged =
            static_cast<double>(older[i]) * older_weight +
            static_cast<double>(newer[i]) * newer_weight;
        out_values[i] = static_cast<float>(merged);
    }

    AxTensor_NormalizeL2(axcore::MakeConstView(out_values, dim), axcore::MakeView(out_values, dim));
}
} // namespace

void AxArena_Init(AxLinearArena* arena, void* backing_memory, uint32_t capacity)
{
    if (arena == nullptr)
    {
        return;
    }

    arena->base = static_cast<uint8_t*>(backing_memory);
    arena->capacity = capacity;
    arena->head = 0u;
}

void AxArena_Reset(AxLinearArena* arena)
{
    if (arena == nullptr)
    {
        return;
    }

    arena->head = 0u;
    if (arena->base != nullptr && arena->capacity > 0u)
    {
        axcore::ZeroBytes(arena->base, arena->capacity);
    }
}

void* AxArena_Alloc(AxLinearArena* arena, uint32_t bytes, uint32_t alignment)
{
    if (arena == nullptr || arena->base == nullptr || bytes == 0u)
    {
        return nullptr;
    }

    const uint32_t safe_alignment = alignment == 0u ? 1u : alignment;
    const uint32_t mask = safe_alignment - 1u;
    const uint32_t head = (arena->head + mask) & ~mask;
    if (head > arena->capacity || bytes > arena->capacity - head)
    {
        return nullptr;
    }

    void* ptr = arena->base + head;
    arena->head = head + bytes;
    axcore::ZeroBytes(ptr, bytes);
    return ptr;
}

AxStatus AxEpisodic_Init(AxEpisodicMemory* memory, AxLinearArena* arena, uint32_t dim, uint32_t max_levels, uint32_t recent_limit)
{
    if (memory == nullptr || arena == nullptr || dim == 0u || max_levels == 0u || recent_limit == 0u)
    {
        return AX_STATUS_INVALID_ARGUMENT;
    }

    memory->levels = static_cast<AxTraceBlock*>(AxArena_Alloc(arena, sizeof(AxTraceBlock) * max_levels, alignof(AxTraceBlock)));
    memory->recent = static_cast<AxRecentTrace*>(AxArena_Alloc(arena, sizeof(AxRecentTrace) * recent_limit, alignof(AxRecentTrace)));
    float* level_values = static_cast<float*>(AxArena_Alloc(arena, sizeof(float) * max_levels * dim, alignof(float)));
    float* recent_values = static_cast<float*>(AxArena_Alloc(arena, sizeof(float) * recent_limit * dim, alignof(float)));
    memory->scratch_a = static_cast<float*>(AxArena_Alloc(arena, sizeof(float) * dim, alignof(float)));
    memory->scratch_b = static_cast<float*>(AxArena_Alloc(arena, sizeof(float) * dim, alignof(float)));

    if (memory->levels == nullptr || memory->recent == nullptr || level_values == nullptr || recent_values == nullptr ||
        memory->scratch_a == nullptr || memory->scratch_b == nullptr)
    {
        return AX_STATUS_OUT_OF_MEMORY;
    }

    memory->max_levels = max_levels;
    memory->recent_limit = recent_limit;
    memory->recent_head = 0u;
    memory->recent_count = 0u;
    memory->dim = dim;
    memory->step = 0u;
    memory->total_stored = 0u;

    for (uint32_t i = 0u; i < max_levels; ++i)
    {
        memory->levels[i].valid = 0u;
        memory->levels[i].summary = level_values + (static_cast<std::size_t>(i) * dim);
        memory->levels[i].start_step = 0u;
        memory->levels[i].end_step = 0u;
        memory->levels[i].span = 0u;
    }

    for (uint32_t i = 0u; i < recent_limit; ++i)
    {
        memory->recent[i].valid = 0u;
        memory->recent[i].step = 0u;
        memory->recent[i].value = recent_values + (static_cast<std::size_t>(i) * dim);
    }

    return AX_STATUS_OK;
}

void AxEpisodic_Clear(AxEpisodicMemory* memory)
{
    if (memory == nullptr)
    {
        return;
    }

    memory->recent_head = 0u;
    memory->recent_count = 0u;
    memory->step = 0u;
    memory->total_stored = 0u;

    for (uint32_t i = 0u; i < memory->max_levels; ++i)
    {
        memory->levels[i].valid = 0u;
        memory->levels[i].start_step = 0u;
        memory->levels[i].end_step = 0u;
        memory->levels[i].span = 0u;
        axcore::ZeroVector(memory->levels[i].summary, memory->dim);
    }

    for (uint32_t i = 0u; i < memory->recent_limit; ++i)
    {
        memory->recent[i].valid = 0u;
        memory->recent[i].step = 0u;
        axcore::ZeroVector(memory->recent[i].value, memory->dim);
    }

    axcore::ZeroVector(memory->scratch_a, memory->dim);
    axcore::ZeroVector(memory->scratch_b, memory->dim);
}

AxStatus AxEpisodic_Store(AxEpisodicMemory* memory, AxConstTensorView thought)
{
    if (memory == nullptr || thought.data == nullptr || thought.shape.total != memory->dim)
    {
        return AX_STATUS_DIMENSION_MISMATCH;
    }

    AxStatus status = AxTensor_NormalizeL2(thought, axcore::MakeView(memory->scratch_a, memory->dim));
    if (status != AX_STATUS_OK)
    {
        return status;
    }

    memory->step += 1u;
    memory->total_stored += 1u;
    PushRecent(memory, memory->scratch_a);

    float* carry = memory->scratch_a;
    float* temp = memory->scratch_b;
    uint64_t carry_start = memory->step;
    uint64_t carry_end = memory->step;
    uint32_t carry_span = 1u;
    uint32_t placed = 0u;

    for (uint32_t level = 0u; level < memory->max_levels; ++level)
    {
        AxTraceBlock& slot = memory->levels[level];
        if (slot.valid == 0u)
        {
            for (uint32_t i = 0u; i < memory->dim; ++i)
            {
                slot.summary[i] = carry[i];
            }
            slot.valid = 1u;
            slot.start_step = carry_start;
            slot.end_step = carry_end;
            slot.span = carry_span;
            placed = 1u;
            break;
        }

        WeightedMerge(slot.summary, slot.span, carry, carry_span, memory->dim, temp);
        carry_start = slot.start_step < carry_start ? slot.start_step : carry_start;
        carry_end = slot.end_step > carry_end ? slot.end_step : carry_end;
        carry_span += slot.span;

        slot.valid = 0u;
        slot.start_step = 0u;
        slot.end_step = 0u;
        slot.span = 0u;
        std::swap(carry, temp);
    }

    if (placed == 0u)
    {
        AxTraceBlock& top = memory->levels[memory->max_levels - 1u];
        for (uint32_t i = 0u; i < memory->dim; ++i)
        {
            top.summary[i] = carry[i];
        }
        top.valid = 1u;
        top.start_step = carry_start;
        top.end_step = carry_end;
        top.span = carry_span;
    }

    return AX_STATUS_OK;
}

AxStatus AxEpisodic_RecallSimilar(const AxEpisodicMemory* memory, AxConstTensorView query, float* out_values, uint32_t out_value_count, AxRecallResult* out_result)
{
    if (memory == nullptr || out_result == nullptr)
    {
        return AX_STATUS_INVALID_ARGUMENT;
    }

    axcore::ClearRecallResult(out_result);
    if (memory->total_stored == 0u)
    {
        return AX_STATUS_OK;
    }
    if (query.data == nullptr || query.shape.total != memory->dim || memory->scratch_a == nullptr)
    {
        return AX_STATUS_DIMENSION_MISMATCH;
    }

    AxTensor_NormalizeL2(query, axcore::MakeView(memory->scratch_a, memory->dim));

    float best_score = -2.0f;
    const float* best_values = nullptr;
    uint64_t best_step = 0u;
    uint32_t best_level = 0u;
    uint32_t best_span = 0u;
    const char* best_source = "";

    for (uint32_t i = 0u; i < memory->recent_count; ++i)
    {
        const uint32_t slot = (memory->recent_head + i) % memory->recent_limit;
        const AxRecentTrace& trace = memory->recent[slot];
        if (trace.valid == 0u)
        {
            continue;
        }

        float score = -1.0f;
        AxTensor_CosineSimilarity(axcore::MakeConstView(memory->scratch_a, memory->dim), axcore::MakeConstView(trace.value, memory->dim), &score);
        if (score > best_score)
        {
            best_score = score;
            best_values = trace.value;
            best_step = trace.step;
            best_level = 0u;
            best_span = 1u;
            best_source = "recent";
        }
    }

    for (uint32_t level = 0u; level < memory->max_levels; ++level)
    {
        const AxTraceBlock& block = memory->levels[level];
        if (block.valid == 0u)
        {
            continue;
        }

        float score = -1.0f;
        AxTensor_CosineSimilarity(axcore::MakeConstView(memory->scratch_a, memory->dim), axcore::MakeConstView(block.summary, memory->dim), &score);
        if (score > best_score)
        {
            best_score = score;
            best_values = block.summary;
            best_step = block.end_step;
            best_level = level;
            best_span = block.span;
            best_source = "logtrace";
        }
    }

    if (best_values == nullptr)
    {
        return AX_STATUS_OK;
    }

    if (out_values != nullptr)
    {
        if (out_value_count < memory->dim)
        {
            return AX_STATUS_BUFFER_TOO_SMALL;
        }
        for (uint32_t i = 0u; i < memory->dim; ++i)
        {
            out_values[i] = best_values[i];
        }
    }

    CopyRecall(out_result, best_score, best_step, memory->step - best_step, best_level, best_span, best_source);
    return AX_STATUS_OK;
}

AxStatus AxEpisodic_RecallStepsAgo(const AxEpisodicMemory* memory, uint64_t steps_ago, float* out_values, uint32_t out_value_count, AxRecallResult* out_result)
{
    if (memory == nullptr || out_result == nullptr)
    {
        return AX_STATUS_INVALID_ARGUMENT;
    }

    axcore::ClearRecallResult(out_result);
    if (memory->total_stored == 0u)
    {
        return AX_STATUS_OK;
    }

    const uint64_t target_step = memory->step > steps_ago ? (memory->step - steps_ago) : 1u;
    const float* best_values = nullptr;
    uint64_t best_step = 0u;
    uint64_t best_distance = UINT64_MAX;
    uint32_t best_level = 0u;
    uint32_t best_span = 0u;
    const char* best_source = "";

    for (uint32_t i = 0u; i < memory->recent_count; ++i)
    {
        const uint32_t slot = (memory->recent_head + i) % memory->recent_limit;
        const AxRecentTrace& trace = memory->recent[slot];
        if (trace.valid == 0u)
        {
            continue;
        }

        const uint64_t distance = axcore::AbsDiff(trace.step, target_step);
        if (distance < best_distance)
        {
            best_distance = distance;
            best_values = trace.value;
            best_step = trace.step;
            best_level = 0u;
            best_span = 1u;
            best_source = "recent";
        }
    }

    for (uint32_t level = 0u; level < memory->max_levels; ++level)
    {
        const AxTraceBlock& block = memory->levels[level];
        if (block.valid == 0u)
        {
            continue;
        }

        uint64_t representative = 0u;
        if (target_step >= block.start_step && target_step <= block.end_step)
        {
            representative = target_step;
        }
        else
        {
            representative = block.start_step + ((block.end_step - block.start_step) / 2u);
        }

        const uint64_t distance = axcore::AbsDiff(representative, target_step);
        if (best_values == nullptr || distance < best_distance || (distance == best_distance && block.span < best_span))
        {
            best_distance = distance;
            best_values = block.summary;
            best_step = representative;
            best_level = level;
            best_span = block.span;
            best_source = "logtrace";
        }
    }

    if (best_values == nullptr)
    {
        return AX_STATUS_OK;
    }

    if (out_values != nullptr)
    {
        if (out_value_count < memory->dim)
        {
            return AX_STATUS_BUFFER_TOO_SMALL;
        }
        for (uint32_t i = 0u; i < memory->dim; ++i)
        {
            out_values[i] = best_values[i];
        }
    }

    CopyRecall(out_result, 0.0f, best_step, memory->step - best_step, best_level, best_span, best_source);
    return AX_STATUS_OK;
}

AxStatus AxWorkingMemory_Init(AxWorkingMemoryCache* cache, AxLinearArena* arena, uint32_t dim, uint32_t capacity)
{
    if (cache == nullptr || arena == nullptr || dim == 0u || capacity == 0u)
    {
        return AX_STATUS_INVALID_ARGUMENT;
    }

    cache->entries = static_cast<AxWorkingMemoryEntry*>(AxArena_Alloc(arena, sizeof(AxWorkingMemoryEntry) * capacity, alignof(AxWorkingMemoryEntry)));
    cache->lru_order = static_cast<uint32_t*>(AxArena_Alloc(arena, sizeof(uint32_t) * capacity, alignof(uint32_t)));
    cache->query_scratch = static_cast<float*>(AxArena_Alloc(arena, sizeof(float) * dim, alignof(float)));
    float* value_slab = static_cast<float*>(AxArena_Alloc(arena, sizeof(float) * capacity * dim, alignof(float)));
    float* deduced_slab = static_cast<float*>(AxArena_Alloc(arena, sizeof(float) * capacity * dim, alignof(float)));

    if (cache->entries == nullptr || cache->lru_order == nullptr || cache->query_scratch == nullptr || value_slab == nullptr || deduced_slab == nullptr)
    {
        return AX_STATUS_OUT_OF_MEMORY;
    }

    cache->capacity = capacity;
    cache->dim = dim;
    cache->count = 0u;
    cache->touch_clock = 0u;

    for (uint32_t i = 0u; i < capacity; ++i)
    {
        cache->entries[i].value = value_slab + (static_cast<std::size_t>(i) * dim);
        cache->entries[i].deduced_constraint = deduced_slab + (static_cast<std::size_t>(i) * dim);
        ClearEntry(&cache->entries[i], dim);
        cache->lru_order[i] = 0u;
    }

    return AX_STATUS_OK;
}

void AxWorkingMemory_Clear(AxWorkingMemoryCache* cache)
{
    if (cache == nullptr)
    {
        return;
    }

    cache->count = 0u;
    cache->touch_clock = 0u;
    for (uint32_t i = 0u; i < cache->capacity; ++i)
    {
        ClearEntry(&cache->entries[i], cache->dim);
        cache->lru_order[i] = 0u;
    }
    axcore::ZeroVector(cache->query_scratch, cache->dim);
}

AxStatus AxWorkingMemory_Promote(
    AxWorkingMemoryCache* cache,
    const char* key,
    const char* dataset_type,
    const char* dataset_id,
    AxConstTensorView value,
    float fitness,
    float normalized_metabolic_burn)
{
    if (cache == nullptr || value.data == nullptr || value.shape.total != cache->dim)
    {
        return AX_STATUS_DIMENSION_MISMATCH;
    }

    char normalized_key[AXCORE_MAX_KEY_LENGTH + 1];
    axcore::CopyTrimmed(normalized_key, sizeof(normalized_key), key, false);
    if (normalized_key[0] == '\0')
    {
        return AX_STATUS_INVALID_ARGUMENT;
    }

    AxStatus status = AxTensor_NormalizeL2(value, axcore::MakeView(cache->query_scratch, cache->dim));
    if (status != AX_STATUS_OK)
    {
        return status;
    }

    const int32_t existing_index = FindEntryByKey(cache, normalized_key);
    uint32_t index = 0u;
    if (existing_index >= 0)
    {
        index = static_cast<uint32_t>(existing_index);
    }
    else if (cache->count >= cache->capacity)
    {
        index = cache->lru_order[0];
        for (uint32_t i = 0u; i + 1u < cache->count; ++i)
        {
            cache->lru_order[i] = cache->lru_order[i + 1u];
        }
    }
    else
    {
        const int32_t free_index = FindFreeEntry(cache);
        if (free_index < 0)
        {
            return AX_STATUS_STATE_ERROR;
        }
        index = static_cast<uint32_t>(free_index);
        cache->count += 1u;
    }

    AxWorkingMemoryEntry& entry = cache->entries[index];
    entry.in_use = 1u;
    axcore::CopyString(entry.key, sizeof(entry.key), normalized_key);
    axcore::CopyTrimmed(entry.dataset_type, sizeof(entry.dataset_type), dataset_type, false);
    axcore::CopyTrimmed(entry.dataset_id, sizeof(entry.dataset_id), dataset_id, false);
    for (uint32_t i = 0u; i < cache->dim; ++i)
    {
        entry.value[i] = cache->query_scratch[i];
    }
    entry.fitness = axcore::Clamp01(fitness);
    entry.decay_score = existing_index >= 0 ? axcore::Clamp(entry.decay_score + 0.05f, 0.0f, 1.0f) : 1.0f;
    entry.is_anomaly = 0u;
    axcore::ZeroVector(entry.deduced_constraint, cache->dim);
    UpdateBurn(&entry, normalized_metabolic_burn);
    cache->touch_clock += 1u;
    entry.last_touch = cache->touch_clock;

    if (existing_index >= 0)
    {
        TouchLru(cache, index);
    }
    else
    {
        cache->lru_order[cache->count - 1u] = index;
    }

    return AX_STATUS_OK;
}

AxStatus AxWorkingMemory_FlagAnomaly(AxWorkingMemoryCache* cache, const char* key, AxConstTensorView deduced_constraint)
{
    if (cache == nullptr || deduced_constraint.data == nullptr || deduced_constraint.shape.total != cache->dim)
    {
        return AX_STATUS_DIMENSION_MISMATCH;
    }

    char normalized_key[AXCORE_MAX_KEY_LENGTH + 1];
    axcore::CopyTrimmed(normalized_key, sizeof(normalized_key), key, false);
    const int32_t index = FindEntryByKey(cache, normalized_key);
    if (index < 0)
    {
        return AX_STATUS_NOT_FOUND;
    }

    AxWorkingMemoryEntry& entry = cache->entries[static_cast<uint32_t>(index)];
    entry.is_anomaly = 1u;
    for (uint32_t i = 0u; i < cache->dim; ++i)
    {
        entry.deduced_constraint[i] = axcore::Sanitize(deduced_constraint.data[i]);
    }
    return AX_STATUS_OK;
}

void AxWorkingMemory_ClearAnomalies(AxWorkingMemoryCache* cache)
{
    if (cache == nullptr)
    {
        return;
    }

    for (uint32_t i = 0u; i < cache->capacity; ++i)
    {
        AxWorkingMemoryEntry& entry = cache->entries[i];
        entry.is_anomaly = 0u;
        axcore::ZeroVector(entry.deduced_constraint, cache->dim);
    }
}

void AxWorkingMemory_ApplyTimeDecay(AxWorkingMemoryCache* cache, float factor, float floor_value)
{
    if (cache == nullptr)
    {
        return;
    }

    const float safe_factor = factor <= 0.0f ? 0.97f : axcore::Clamp(factor, 0.0f, 1.0f);
    const float safe_floor = axcore::Clamp(floor_value, 0.0f, 1.0f);
    for (uint32_t i = 0u; i < cache->capacity; ++i)
    {
        AxWorkingMemoryEntry& entry = cache->entries[i];
        if (entry.in_use == 0u)
        {
            continue;
        }
        entry.decay_score = entry.decay_score * safe_factor;
        if (entry.decay_score < safe_floor)
        {
            entry.decay_score = safe_floor;
        }
    }
}

AxStatus AxWorkingMemory_CosineHit(
    AxWorkingMemoryCache* cache,
    AxConstTensorView query,
    float threshold,
    float* out_values,
    uint32_t out_value_count,
    AxCacheMatch* out_match)
{
    if (cache == nullptr || out_match == nullptr)
    {
        return AX_STATUS_INVALID_ARGUMENT;
    }

    axcore::ClearCacheMatch(out_match);
    if (cache->count == 0u)
    {
        return AX_STATUS_OK;
    }
    if (query.data == nullptr || query.shape.total != cache->dim)
    {
        return AX_STATUS_DIMENSION_MISMATCH;
    }

    AxStatus status = AxTensor_NormalizeL2(query, axcore::MakeView(cache->query_scratch, cache->dim));
    if (status != AX_STATUS_OK)
    {
        return status;
    }

    float best_score = -1.0f;
    int32_t best_index = -1;
    for (uint32_t i = 0u; i < cache->capacity; ++i)
    {
        AxWorkingMemoryEntry& entry = cache->entries[i];
        if (entry.in_use == 0u)
        {
            continue;
        }

        float similarity = 0.0f;
        AxTensor_CosineSimilarity(
            axcore::MakeConstView(cache->query_scratch, cache->dim),
            axcore::MakeConstView(entry.value, cache->dim),
            &similarity);
        similarity *= entry.decay_score;
        if (similarity > best_score)
        {
            best_score = similarity;
            best_index = static_cast<int32_t>(i);
        }
    }

    if (best_index < 0)
    {
        return AX_STATUS_OK;
    }

    const float safe_threshold = axcore::Clamp(threshold, -1.0f, 1.0f);
    if (best_score < safe_threshold)
    {
        return AX_STATUS_OK;
    }

    AxWorkingMemoryEntry& hit = cache->entries[static_cast<uint32_t>(best_index)];
    hit.hits += 1u;
    hit.decay_score = axcore::Clamp(hit.decay_score + 0.02f, 0.0f, 1.0f);
    cache->touch_clock += 1u;
    hit.last_touch = cache->touch_clock;
    TouchLru(cache, static_cast<uint32_t>(best_index));

    if (out_values != nullptr)
    {
        if (out_value_count < cache->dim)
        {
            return AX_STATUS_BUFFER_TOO_SMALL;
        }
        for (uint32_t i = 0u; i < cache->dim; ++i)
        {
            out_values[i] = hit.value[i];
        }
    }

    CopyMatch(out_match, &hit, best_score);
    return AX_STATUS_OK;
}
