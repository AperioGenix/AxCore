#pragma once

#include <stdint.h>

enum
{
    AXCORE_MAX_DIMS = 8,
    AXCORE_MAX_KEY_LENGTH = 63,
    AXCORE_MAX_DATASET_TYPE_LENGTH = 31,
    AXCORE_MAX_DATASET_ID_LENGTH = 63,
    AXCORE_MAX_STRATEGY_LENGTH = 31,
    AXCORE_MAX_LABEL_LENGTH = 31,
    AXCORE_MAX_SOURCE_LENGTH = 15,
    AXCORE_MAX_GENE_ID_LENGTH = 31
};

typedef enum AxStatus
{
    AX_STATUS_OK = 0,
    AX_STATUS_INVALID_ARGUMENT = 1,
    AX_STATUS_DIMENSION_MISMATCH = 2,
    AX_STATUS_OUT_OF_MEMORY = 3,
    AX_STATUS_BUFFER_TOO_SMALL = 4,
    AX_STATUS_NOT_FOUND = 5,
    AX_STATUS_STATE_ERROR = 6,
    AX_STATUS_LIMIT_EXCEEDED = 7
} AxStatus;

typedef struct AxVersion
{
    uint32_t major;
    uint32_t minor;
    uint32_t patch;
    uint32_t abi;
} AxVersion;

typedef struct AxShape
{
    uint32_t ndim;
    uint32_t dims[AXCORE_MAX_DIMS];
    uint32_t total;
} AxShape;

typedef struct AxTensorView
{
    float* data;
    AxShape shape;
} AxTensorView;

typedef struct AxConstTensorView
{
    const float* data;
    AxShape shape;
} AxConstTensorView;

typedef struct AxRecallResult
{
    uint32_t found;
    float similarity;
    uint64_t stored_step;
    uint64_t age_steps;
    uint32_t level;
    uint32_t span;
    char source[AXCORE_MAX_SOURCE_LENGTH + 1];
} AxRecallResult;

typedef struct AxCacheMatch
{
    uint32_t found;
    float similarity;
    float fitness;
    float decay_score;
    float last_metabolic_burn;
    float average_metabolic_burn;
    uint32_t burn_samples;
    uint32_t hits;
    uint64_t last_touch;
    uint32_t is_anomaly;
    char key[AXCORE_MAX_KEY_LENGTH + 1];
    char dataset_type[AXCORE_MAX_DATASET_TYPE_LENGTH + 1];
    char dataset_id[AXCORE_MAX_DATASET_ID_LENGTH + 1];
} AxCacheMatch;

typedef struct AxTensorOpCandidate
{
    float fitness;
    float similarity;
    float cost;
    char strategy[AXCORE_MAX_STRATEGY_LENGTH + 1];
} AxTensorOpCandidate;

typedef struct AxHeuristicConfig
{
    float system1_base;
    float system1_entropy_weight;
    float system1_sparsity_weight;
    float system1_min;
    float system1_max;

    float critic_base;
    float critic_entropy_weight;
    float critic_skewness_weight;
    float critic_min;
    float critic_max;

    float deep_think_cost_base;
    float deep_think_entropy_weight;
    float deep_think_sparsity_weight;
    float deep_think_cost_min;
    float deep_think_cost_max;

    float consolidation_min_fitness;
    float consolidation_max_normalized_burn;
    uint32_t consolidation_top_limit;

    float fatigue_remaining_ratio;
    float zombie_activation_ratio;
    float zombie_critic_threshold;
} AxHeuristicConfig;

typedef struct AxSignalProfile
{
    uint32_t length;
    float mean;
    float standard_deviation;
    float skewness;
    float sparsity;
    float entropy;
    float unique_ratio;
    float range;
    float system1_similarity_threshold;
    float critic_acceptance_threshold;
    float deep_think_cost_bias;
    char label[AXCORE_MAX_LABEL_LENGTH + 1];
} AxSignalProfile;

typedef struct AxManifoldScanResult
{
    uint32_t candidate_count;
    float best_similarity;
    float second_best_similarity;
    float mean_similarity;
    float similarity_stddev;
    float ambiguity_gap;
    float grounding_score;
    float void_score;
    float entropy;
    char label[AXCORE_MAX_LABEL_LENGTH + 1];
} AxManifoldScanResult;

typedef struct AxSystemMetabolism
{
    float max_capacity;
    float current_energy_budget;
    float fatigue_threshold;
    float zombie_activation_threshold;
    float fatigue_remaining_ratio;
    float zombie_activation_ratio;
    float zombie_critic_threshold;
    uint32_t zombie_mode_active;
} AxSystemMetabolism;

typedef struct AxMetabolicCriticConfig
{
    float flux_budget;
    float max_noise_floor;
    float noise_gain;
    float noise_rise_rate;
    float noise_decay_rate;
    float focused_floor_weight;
    float background_attenuation_scale;
    float overload_penalty_scale;
} AxMetabolicCriticConfig;

typedef struct AxGeneRuntimeState
{
    char gene_id[AXCORE_MAX_GENE_ID_LENGTH + 1];
    float weight;
    float min_weight;
    float cost_estimate;
    uint32_t latent_tick_count;
    uint32_t is_autophagy_candidate;
} AxGeneRuntimeState;

typedef struct AxSceneTickReport
{
    uint32_t active_genes;
    char focused_gene_id[AXCORE_MAX_GENE_ID_LENGTH + 1];
    float budget_flux;
    float total_flux_before;
    float total_flux_after;
    float overload_ratio;
    float noise_floor;
    uint32_t latent_genes;
    uint32_t autophagy_candidates;
} AxSceneTickReport;
