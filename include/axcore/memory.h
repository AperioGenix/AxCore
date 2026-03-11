#pragma once

#include "axcore/math.h"

typedef struct AxLinearArena
{
    uint8_t* base;
    uint32_t capacity;
    uint32_t head;
} AxLinearArena;

typedef struct AxTraceBlock
{
    uint32_t valid;
    float* summary;
    uint64_t start_step;
    uint64_t end_step;
    uint32_t span;
} AxTraceBlock;

typedef struct AxRecentTrace
{
    float* value;
    uint64_t step;
    uint32_t valid;
} AxRecentTrace;

typedef struct AxEpisodicMemory
{
    AxTraceBlock* levels;
    AxRecentTrace* recent;
    float* scratch_a;
    float* scratch_b;
    uint32_t max_levels;
    uint32_t recent_limit;
    uint32_t recent_head;
    uint32_t recent_count;
    uint32_t dim;
    uint64_t step;
    uint64_t total_stored;
} AxEpisodicMemory;

typedef struct AxWorkingMemoryEntry
{
    uint32_t in_use;
    char key[AXCORE_MAX_KEY_LENGTH + 1];
    char dataset_type[AXCORE_MAX_DATASET_TYPE_LENGTH + 1];
    char dataset_id[AXCORE_MAX_DATASET_ID_LENGTH + 1];
    float* value;
    float fitness;
    float decay_score;
    float last_metabolic_burn;
    float average_metabolic_burn;
    uint32_t burn_samples;
    uint32_t hits;
    uint64_t last_touch;
    uint32_t is_anomaly;
    float* deduced_constraint;
} AxWorkingMemoryEntry;

typedef struct AxWorkingMemoryCache
{
    AxWorkingMemoryEntry* entries;
    uint32_t* lru_order;
    float* query_scratch;
    uint32_t capacity;
    uint32_t dim;
    uint32_t count;
    uint64_t touch_clock;
} AxWorkingMemoryCache;

#if defined(__cplusplus)
extern "C"
{
#endif

void AxArena_Init(AxLinearArena* arena, void* backing_memory, uint32_t capacity);
void AxArena_Reset(AxLinearArena* arena);
void* AxArena_Alloc(AxLinearArena* arena, uint32_t bytes, uint32_t alignment);

AxStatus AxEpisodic_Init(AxEpisodicMemory* memory, AxLinearArena* arena, uint32_t dim, uint32_t max_levels, uint32_t recent_limit);
void AxEpisodic_Clear(AxEpisodicMemory* memory);
AxStatus AxEpisodic_Store(AxEpisodicMemory* memory, AxConstTensorView thought);
AxStatus AxEpisodic_RecallSimilar(const AxEpisodicMemory* memory, AxConstTensorView query, float* out_values, uint32_t out_value_count, AxRecallResult* out_result);
AxStatus AxEpisodic_RecallStepsAgo(const AxEpisodicMemory* memory, uint64_t steps_ago, float* out_values, uint32_t out_value_count, AxRecallResult* out_result);

AxStatus AxWorkingMemory_Init(AxWorkingMemoryCache* cache, AxLinearArena* arena, uint32_t dim, uint32_t capacity);
void AxWorkingMemory_Clear(AxWorkingMemoryCache* cache);
AxStatus AxWorkingMemory_Promote(
    AxWorkingMemoryCache* cache,
    const char* key,
    const char* dataset_type,
    const char* dataset_id,
    AxConstTensorView value,
    float fitness,
    float normalized_metabolic_burn);
AxStatus AxWorkingMemory_FlagAnomaly(AxWorkingMemoryCache* cache, const char* key, AxConstTensorView deduced_constraint);
void AxWorkingMemory_ClearAnomalies(AxWorkingMemoryCache* cache);
void AxWorkingMemory_ApplyTimeDecay(AxWorkingMemoryCache* cache, float factor, float floor_value);
AxStatus AxWorkingMemory_CosineHit(
    AxWorkingMemoryCache* cache,
    AxConstTensorView query,
    float threshold,
    float* out_values,
    uint32_t out_value_count,
    AxCacheMatch* out_match);

#if defined(__cplusplus)
}
#endif
