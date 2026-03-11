#pragma once

#include "axcore/connectome.h"
#include "axcore/export.h"

typedef struct AxCoreHandle AxCoreHandle;

typedef struct AxCoreCreateInfo
{
    uint32_t hdc_dim;
    uint32_t working_memory_capacity;
    uint32_t episodic_recent_limit;
    uint32_t episodic_max_levels;
    uint32_t arena_bytes;
} AxCoreCreateInfo;

#if defined(__cplusplus)
extern "C"
{
#endif

AXCORE_EXTERN_C AXCORE_API AxStatus AXCORE_CALL AxCore_GetVersion(AxVersion* out_version);
AXCORE_EXTERN_C AXCORE_API void AXCORE_CALL AxCore_GetDefaultCreateInfo(AxCoreCreateInfo* out_info);
AXCORE_EXTERN_C AXCORE_API void AXCORE_CALL AxCore_GetDefaultHeuristics(AxHeuristicConfig* out_config);
AXCORE_EXTERN_C AXCORE_API void AXCORE_CALL AxCore_GetDefaultMetabolism(AxSystemMetabolism* out_state);
AXCORE_EXTERN_C AXCORE_API AxStatus AXCORE_CALL AxCore_ComputeRequiredArenaBytes(const AxCoreCreateInfo* create_info, uint32_t* out_bytes);

AXCORE_EXTERN_C AXCORE_API AxStatus AXCORE_CALL AxCore_Create(const AxCoreCreateInfo* create_info, AxCoreHandle** out_handle);
AXCORE_EXTERN_C AXCORE_API void AXCORE_CALL AxCore_Destroy(AxCoreHandle* handle);
AXCORE_EXTERN_C AXCORE_API void AXCORE_CALL AxCore_Reset(AxCoreHandle* handle);

AXCORE_EXTERN_C AXCORE_API AxStatus AXCORE_CALL AxCore_SetHeuristics(AxCoreHandle* handle, const AxHeuristicConfig* config);
AXCORE_EXTERN_C AXCORE_API AxStatus AXCORE_CALL AxCore_GetMetabolism(const AxCoreHandle* handle, AxSystemMetabolism* out_state);
AXCORE_EXTERN_C AXCORE_API AxStatus AXCORE_CALL AxCore_ConsumeMetabolism(AxCoreHandle* handle, float amount);
AXCORE_EXTERN_C AXCORE_API AxStatus AXCORE_CALL AxCore_RechargeMetabolism(AxCoreHandle* handle, float amount);

AXCORE_EXTERN_C AXCORE_API AxStatus AXCORE_CALL AxCore_StoreEpisode(AxCoreHandle* handle, const float* values, uint32_t value_count);
AXCORE_EXTERN_C AXCORE_API AxStatus AXCORE_CALL AxCore_RecallSimilar(
    const AxCoreHandle* handle,
    const float* values,
    uint32_t value_count,
    float* out_values,
    uint32_t out_value_count,
    AxRecallResult* out_result);
AXCORE_EXTERN_C AXCORE_API AxStatus AXCORE_CALL AxCore_RecallStepsAgo(
    const AxCoreHandle* handle,
    uint64_t steps_ago,
    float* out_values,
    uint32_t out_value_count,
    AxRecallResult* out_result);
AXCORE_EXTERN_C AXCORE_API AxStatus AXCORE_CALL AxCore_Consolidate(AxCoreHandle* handle);

AXCORE_EXTERN_C AXCORE_API AxStatus AXCORE_CALL AxCore_PromoteWorkingMemory(
    AxCoreHandle* handle,
    const char* key,
    const char* dataset_type,
    const char* dataset_id,
    const float* values,
    uint32_t value_count,
    float fitness,
    float normalized_metabolic_burn);
AXCORE_EXTERN_C AXCORE_API AxStatus AXCORE_CALL AxCore_QueryWorkingMemory(
    AxCoreHandle* handle,
    const float* values,
    uint32_t value_count,
    float threshold,
    float* out_values,
    uint32_t out_value_count,
    AxCacheMatch* out_match);
AXCORE_EXTERN_C AXCORE_API void AXCORE_CALL AxCore_ApplyWorkingMemoryDecay(AxCoreHandle* handle, float factor, float floor_value);

AXCORE_EXTERN_C AXCORE_API AxStatus AXCORE_CALL AxCore_AnalyzeSignal(
    const AxCoreHandle* handle,
    const float* values,
    uint32_t value_count,
    AxSignalProfile* out_profile);
AXCORE_EXTERN_C AXCORE_API AxStatus AXCORE_CALL AxCore_ScanManifoldEntropy(const AxCoreHandle* handle, const float* query_values, uint32_t query_value_count, const float* candidate_stack_flat, uint32_t candidate_count, uint32_t dim, AxManifoldScanResult* out_result);
AXCORE_EXTERN_C AXCORE_API AxStatus AXCORE_CALL AxCore_ProjectManifold(
    const float* vector_stack_flat,
    uint32_t sample_count,
    uint32_t input_dim,
    uint32_t output_dim,
    uint32_t normalize_rows,
    uint32_t projection_seed,
    float* out_projected_flat,
    uint32_t out_value_count);
AXCORE_EXTERN_C AXCORE_API AxStatus AXCORE_CALL AxCore_RouteCandidate(
    AxCoreHandle* handle,
    const float* values,
    uint32_t value_count,
    uint32_t iteration,
    float* out_values,
    uint32_t out_value_count,
    AxTensorOpCandidate* out_candidate,
    AxSignalProfile* out_profile);
AXCORE_EXTERN_C AXCORE_API AxStatus AXCORE_CALL AxCore_DeduceGeometricGap(
    AxCoreHandle* handle,
    const float* current_state,
    uint32_t current_count,
    const float* required_next_state,
    uint32_t required_count,
    float* out_values,
    uint32_t out_value_count);
AXCORE_EXTERN_C AXCORE_API float AXCORE_CALL AxCore_BatchSequenceSimilarity(
    const float* vector_stack_flat,
    const uint32_t* sequence_indices,
    uint32_t index_count,
    const float* target_vector,
    uint32_t dim);

#if defined(__cplusplus)
}
#endif
