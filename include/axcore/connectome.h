#pragma once

#include "axcore/memory.h"

#if defined(__cplusplus)
extern "C"
{
#endif

void AxHeuristicConfig_Default(AxHeuristicConfig* out_config);
void AxSystemMetabolism_Default(AxSystemMetabolism* out_state);
void AxMetabolicCriticConfig_Default(AxMetabolicCriticConfig* out_config);

void AxSystemMetabolism_ConfigureRelative(
    AxSystemMetabolism* state,
    float max_capacity,
    float fatigue_remaining_ratio,
    float zombie_activation_ratio,
    float zombie_critic_threshold);
void AxSystemMetabolism_Consume(AxSystemMetabolism* state, float amount);
void AxSystemMetabolism_Recharge(AxSystemMetabolism* state, float amount);
void AxSystemMetabolism_TriggerZombieMode(AxSystemMetabolism* state);
float AxSystemMetabolism_EnergyPercent(const AxSystemMetabolism* state);
float AxSystemMetabolism_CriticThreshold(const AxSystemMetabolism* state);
uint32_t AxSystemMetabolism_CanDeepThink(const AxSystemMetabolism* state);

AxStatus AxSignalProfile_Analyze(AxConstTensorView input, const AxHeuristicConfig* config, AxSignalProfile* out_profile);
AxStatus AxConnectome_DeduceGeometricGap(AxConstTensorView current_state, AxConstTensorView required_next_state, AxTensorView output);
AxStatus AxConnectome_RouteCandidate(
    AxConstTensorView target,
    const AxSignalProfile* profile,
    const AxWorkingMemoryCache* cache,
    const AxSystemMetabolism* metabolism,
    uint32_t iteration,
    AxTensorView output,
    AxTensorView scratch,
    AxTensorOpCandidate* out_candidate);
float AxConnectome_CalculateThermodynamicCost(const AxTensorOpCandidate* candidate, const AxSignalProfile* profile);
uint32_t AxConnectome_PassesCriticThreshold(
    const AxTensorOpCandidate* candidate,
    const AxSignalProfile* profile,
    const AxSystemMetabolism* metabolism);
AxStatus AxMetabolicCritic_Tick(
    AxGeneRuntimeState* genes,
    uint32_t gene_count,
    const char* focused_gene_id,
    const AxMetabolicCriticConfig* config,
    float* io_noise_floor,
    AxSceneTickReport* out_report);

#if defined(__cplusplus)
}
#endif
