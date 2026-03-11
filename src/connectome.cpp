#include "axcore_internal.h"

#include <cmath>

void AxHeuristicConfig_Default(AxHeuristicConfig* out_config)
{
    if (out_config == nullptr)
    {
        return;
    }

    out_config->system1_base = 0.90f;
    out_config->system1_entropy_weight = 0.10f;
    out_config->system1_sparsity_weight = 0.05f;
    out_config->system1_min = 0.85f;
    out_config->system1_max = 0.98f;

    out_config->critic_base = 0.85f;
    out_config->critic_entropy_weight = 0.05f;
    out_config->critic_skewness_weight = 0.02f;
    out_config->critic_min = 0.80f;
    out_config->critic_max = 0.95f;

    out_config->deep_think_cost_base = 0.80f;
    out_config->deep_think_entropy_weight = 0.90f;
    out_config->deep_think_sparsity_weight = 0.50f;
    out_config->deep_think_cost_min = 0.70f;
    out_config->deep_think_cost_max = 2.30f;

    out_config->consolidation_min_fitness = 0.95f;
    out_config->consolidation_max_normalized_burn = 0.20f;
    out_config->consolidation_top_limit = 64u;

    out_config->fatigue_remaining_ratio = 0.28f;
    out_config->zombie_activation_ratio = 0.20f;
    out_config->zombie_critic_threshold = 0.95f;
}

void AxSystemMetabolism_Default(AxSystemMetabolism* out_state)
{
    if (out_state == nullptr)
    {
        return;
    }

    out_state->max_capacity = 1000.0f;
    out_state->current_energy_budget = 1000.0f;
    out_state->fatigue_threshold = 280.0f;
    out_state->zombie_activation_threshold = 200.0f;
    out_state->fatigue_remaining_ratio = 0.28f;
    out_state->zombie_activation_ratio = 0.20f;
    out_state->zombie_critic_threshold = 0.95f;
    out_state->zombie_mode_active = 0u;
}

void AxMetabolicCriticConfig_Default(AxMetabolicCriticConfig* out_config)
{
    if (out_config == nullptr)
    {
        return;
    }

    out_config->flux_budget = 6.0f;
    out_config->max_noise_floor = 0.92f;
    out_config->noise_gain = 0.70f;
    out_config->noise_rise_rate = 0.35f;
    out_config->noise_decay_rate = 0.08f;
    out_config->focused_floor_weight = 0.75f;
    out_config->background_attenuation_scale = 1.0f;
    out_config->overload_penalty_scale = 0.75f;
}

void AxSystemMetabolism_ConfigureRelative(
    AxSystemMetabolism* state,
    float max_capacity,
    float fatigue_remaining_ratio,
    float zombie_activation_ratio,
    float zombie_critic_threshold)
{
    if (state == nullptr)
    {
        return;
    }

    const float safe_max = max_capacity <= 0.0f ? 1000.0f : max_capacity;
    const float safe_fatigue_ratio = axcore::Clamp(fatigue_remaining_ratio, 0.01f, 0.95f);
    const float safe_zombie_ratio = axcore::Clamp(zombie_activation_ratio, 0.01f, safe_fatigue_ratio);

    state->max_capacity = safe_max;
    state->fatigue_remaining_ratio = safe_fatigue_ratio;
    state->zombie_activation_ratio = safe_zombie_ratio;
    state->zombie_critic_threshold = axcore::Clamp01(zombie_critic_threshold <= 0.0f ? 0.95f : zombie_critic_threshold);
    state->fatigue_threshold = state->max_capacity * state->fatigue_remaining_ratio;
    state->zombie_activation_threshold = state->max_capacity * state->zombie_activation_ratio;
    state->current_energy_budget = state->max_capacity;
    state->zombie_mode_active = 0u;
}

void AxSystemMetabolism_Consume(AxSystemMetabolism* state, float amount)
{
    if (state == nullptr || amount <= 0.0f)
    {
        return;
    }

    state->current_energy_budget -= amount;
    if (state->current_energy_budget < 0.0f)
    {
        state->current_energy_budget = 0.0f;
    }
    if (state->current_energy_budget <= state->zombie_activation_threshold)
    {
        state->zombie_mode_active = 1u;
    }
}

void AxSystemMetabolism_Recharge(AxSystemMetabolism* state, float amount)
{
    if (state == nullptr)
    {
        return;
    }

    if (amount <= 0.0f)
    {
        state->current_energy_budget = state->max_capacity;
    }
    else
    {
        state->current_energy_budget += amount;
        if (state->current_energy_budget > state->max_capacity)
        {
            state->current_energy_budget = state->max_capacity;
        }
    }
    state->zombie_mode_active = 0u;
}

void AxSystemMetabolism_TriggerZombieMode(AxSystemMetabolism* state)
{
    if (state != nullptr)
    {
        state->zombie_mode_active = 1u;
    }
}

float AxSystemMetabolism_EnergyPercent(const AxSystemMetabolism* state)
{
    if (state == nullptr || state->max_capacity <= 0.0f)
    {
        return 0.0f;
    }
    return state->current_energy_budget / state->max_capacity;
}

float AxSystemMetabolism_CriticThreshold(const AxSystemMetabolism* state)
{
    if (state == nullptr)
    {
        return 0.5f;
    }
    return state->zombie_mode_active != 0u ? state->zombie_critic_threshold : 0.50f;
}

uint32_t AxSystemMetabolism_CanDeepThink(const AxSystemMetabolism* state)
{
    if (state == nullptr)
    {
        return 0u;
    }
    return state->current_energy_budget > state->fatigue_threshold && state->zombie_mode_active == 0u ? 1u : 0u;
}

AxStatus AxSignalProfile_Analyze(AxConstTensorView input, const AxHeuristicConfig* config, AxSignalProfile* out_profile)
{
    if (out_profile == nullptr)
    {
        return AX_STATUS_INVALID_ARGUMENT;
    }

    axcore::ClearProfile(out_profile);

    AxHeuristicConfig defaults{};
    if (config == nullptr)
    {
        AxHeuristicConfig_Default(&defaults);
        config = &defaults;
    }

    if (input.shape.total == 0u)
    {
        out_profile->length = 1u;
        out_profile->mean = 0.0f;
        out_profile->standard_deviation = 0.0f;
        out_profile->skewness = 0.0f;
        out_profile->sparsity = 1.0f;
        out_profile->entropy = 0.0f;
        out_profile->unique_ratio = 1.0f;
        out_profile->range = 0.0f;
    }
    else
    {
        constexpr uint32_t kBucketCapacity = 256u;
        int32_t bucket_keys[kBucketCapacity];
        uint32_t bucket_counts[kBucketCapacity];
        uint32_t bucket_used = 0u;
        uint32_t overflow_count = 0u;
        for (uint32_t i = 0u; i < kBucketCapacity; ++i)
        {
            bucket_keys[i] = 0;
            bucket_counts[i] = 0u;
        }

        out_profile->length = input.shape.total;
        float min_value = axcore::Sanitize(input.data[0]);
        float max_value = min_value;
        double sum = 0.0;
        uint32_t zeros = 0u;

        for (uint32_t i = 0u; i < input.shape.total; ++i)
        {
            const float value = axcore::Sanitize(input.data[i]);
            if (value < min_value)
            {
                min_value = value;
            }
            if (value > max_value)
            {
                max_value = value;
            }
            if (std::fabs(static_cast<double>(value)) < 1.0e-6)
            {
                zeros += 1u;
            }

            sum += value;
            const int32_t bucket = axcore::RoundBucket(value);
            uint32_t inserted = 0u;
            for (uint32_t j = 0u; j < bucket_used; ++j)
            {
                if (bucket_keys[j] == bucket)
                {
                    bucket_counts[j] += 1u;
                    inserted = 1u;
                    break;
                }
            }

            if (inserted == 0u)
            {
                if (bucket_used < kBucketCapacity)
                {
                    bucket_keys[bucket_used] = bucket;
                    bucket_counts[bucket_used] = 1u;
                    bucket_used += 1u;
                }
                else
                {
                    overflow_count += 1u;
                }
            }
        }

        const double mean = sum / static_cast<double>(input.shape.total);
        double variance = 0.0;
        double skew = 0.0;
        for (uint32_t i = 0u; i < input.shape.total; ++i)
        {
            const double diff = static_cast<double>(axcore::Sanitize(input.data[i])) - mean;
            variance += diff * diff;
        }
        variance /= static_cast<double>(input.shape.total);
        const double standard_deviation = std::sqrt(variance);
        if (standard_deviation > 1.0e-9)
        {
            for (uint32_t i = 0u; i < input.shape.total; ++i)
            {
                const double diff = static_cast<double>(axcore::Sanitize(input.data[i])) - mean;
                skew += diff * diff * diff;
            }
            skew /= static_cast<double>(input.shape.total);
            skew /= (standard_deviation * standard_deviation * standard_deviation);
        }

        double entropy = 0.0;
        for (uint32_t i = 0u; i < bucket_used; ++i)
        {
            const double p = static_cast<double>(bucket_counts[i]) / static_cast<double>(input.shape.total);
            if (p > 0.0)
            {
                entropy -= p * (std::log(p) / std::log(2.0));
            }
        }
        if (overflow_count > 0u)
        {
            const double p = static_cast<double>(overflow_count) / static_cast<double>(input.shape.total);
            entropy -= p * (std::log(p) / std::log(2.0));
        }

        const uint32_t unique_buckets = bucket_used + (overflow_count > 0u ? 1u : 0u);
        const double max_entropy = std::log(static_cast<double>(unique_buckets == 0u ? 1u : unique_buckets)) / std::log(2.0);
        const double entropy_norm = max_entropy > 0.0 ? entropy / max_entropy : 0.0;

        out_profile->mean = static_cast<float>(mean);
        out_profile->standard_deviation = static_cast<float>(standard_deviation);
        out_profile->skewness = static_cast<float>(skew);
        out_profile->sparsity = static_cast<float>(zeros) / static_cast<float>(input.shape.total);
        out_profile->entropy = static_cast<float>(entropy_norm);
        out_profile->unique_ratio = static_cast<float>(unique_buckets) / static_cast<float>(input.shape.total);
        out_profile->range = max_value - min_value;
    }

    out_profile->system1_similarity_threshold = axcore::Clamp(
        config->system1_base - (out_profile->entropy * config->system1_entropy_weight) +
            (out_profile->sparsity * config->system1_sparsity_weight),
        config->system1_min,
        config->system1_max);

    out_profile->critic_acceptance_threshold = axcore::Clamp(
        config->critic_base + (out_profile->entropy * config->critic_entropy_weight) +
            (std::fabs(out_profile->skewness) * config->critic_skewness_weight),
        config->critic_min,
        config->critic_max);

    out_profile->deep_think_cost_bias = axcore::Clamp(
        config->deep_think_cost_base + (out_profile->entropy * config->deep_think_entropy_weight) +
            (out_profile->sparsity * config->deep_think_sparsity_weight),
        config->deep_think_cost_min,
        config->deep_think_cost_max);

    if (out_profile->sparsity > 0.75f)
    {
        axcore::CopyString(out_profile->label, sizeof(out_profile->label), "sparse");
    }
    else if (out_profile->entropy > 0.70f)
    {
        axcore::CopyString(out_profile->label, sizeof(out_profile->label), "high_entropy");
    }
    else if (std::fabs(out_profile->skewness) > 1.0f)
    {
        axcore::CopyString(out_profile->label, sizeof(out_profile->label), "skewed");
    }
    else
    {
        axcore::CopyString(out_profile->label, sizeof(out_profile->label), "balanced");
    }

    return AX_STATUS_OK;
}

AxStatus AxConnectome_DeduceGeometricGap(AxConstTensorView current_state, AxConstTensorView required_next_state, AxTensorView output)
{
    if (current_state.data == nullptr || required_next_state.data == nullptr || output.data == nullptr)
    {
        return AX_STATUS_INVALID_ARGUMENT;
    }
    if (current_state.shape.total != required_next_state.shape.total || current_state.shape.total != output.shape.total)
    {
        return AX_STATUS_DIMENSION_MISMATCH;
    }

    const AxStatus subtract_status = AxTensor_Subtract(required_next_state, current_state, output);
    if (subtract_status != AX_STATUS_OK)
    {
        return subtract_status;
    }
    return AxTensor_NormalizeL2(axcore::MakeConstView(output.data, output.shape.total), output);
}

float AxConnectome_CalculateThermodynamicCost(const AxTensorOpCandidate* candidate, const AxSignalProfile* profile)
{
    if (candidate == nullptr || profile == nullptr)
    {
        return 0.0f;
    }

    const float base_cost = 4.0f + (12.0f * profile->deep_think_cost_bias);
    const float critic_penalty = (1.0f - axcore::Clamp(candidate->fitness, 0.0f, 1.0f)) * 8.0f;
    const float strategy_bias = std::strcmp(candidate->strategy, "cache_bundle") == 0 ? 0.85f : 1.0f;
    const float total_cost = (base_cost + critic_penalty) * strategy_bias;
    return total_cost < 0.5f ? 0.5f : total_cost;
}

uint32_t AxConnectome_PassesCriticThreshold(
    const AxTensorOpCandidate* candidate,
    const AxSignalProfile* profile,
    const AxSystemMetabolism* metabolism)
{
    if (candidate == nullptr || profile == nullptr)
    {
        return 0u;
    }

    const float base_threshold = axcore::Clamp(profile->critic_acceptance_threshold, 0.0f, 1.0f);
    const float active_threshold =
        metabolism != nullptr && metabolism->zombie_mode_active != 0u ? metabolism->zombie_critic_threshold : base_threshold;
    return candidate->fitness >= active_threshold ? 1u : 0u;
}

AxStatus AxConnectome_RouteCandidate(
    AxConstTensorView target,
    const AxSignalProfile* profile,
    const AxWorkingMemoryCache* cache,
    const AxSystemMetabolism* metabolism,
    uint32_t iteration,
    AxTensorView output,
    AxTensorView scratch,
    AxTensorOpCandidate* out_candidate)
{
    if (target.data == nullptr || profile == nullptr || output.data == nullptr || scratch.data == nullptr || out_candidate == nullptr)
    {
        return AX_STATUS_INVALID_ARGUMENT;
    }
    if (target.shape.total == 0u || target.shape.total != output.shape.total || target.shape.total != scratch.shape.total)
    {
        return AX_STATUS_DIMENSION_MISMATCH;
    }

    axcore::ClearCandidate(out_candidate);
    const AxStatus normalize_status = AxTensor_NormalizeL2(target, scratch);
    if (normalize_status != AX_STATUS_OK)
    {
        return normalize_status;
    }

    const AxWorkingMemoryEntry* best_memory = nullptr;
    float best_similarity = -1.0f;

    if (cache != nullptr)
    {
        for (uint32_t i = 0u; i < cache->capacity; ++i)
        {
            const AxWorkingMemoryEntry& entry = cache->entries[i];
            if (entry.in_use == 0u)
            {
                continue;
            }

            float similarity = -1.0f;
            AxTensor_CosineSimilarity(
                axcore::MakeConstView(scratch.data, scratch.shape.total),
                axcore::MakeConstView(entry.value, cache->dim),
                &similarity);
            if (similarity > best_similarity)
            {
                best_similarity = similarity;
                best_memory = &entry;
            }
        }
    }

    if (best_memory != nullptr)
    {
        const bool zombie_mode = metabolism != nullptr && metabolism->zombie_mode_active != 0u;
        const float memory_floor = zombie_mode ? 0.55f : 0.20f;
        const float memory_ceiling = zombie_mode ? 0.92f : 0.80f;
        const float zombie_bias = zombie_mode ? 0.12f : 0.0f;
        const float memory_weight = axcore::Clamp(0.30f + ((1.0f - profile->entropy) * 0.50f) + zombie_bias, memory_floor, memory_ceiling);
        const float target_weight = 1.0f - memory_weight;
        for (uint32_t i = 0u; i < output.shape.total; ++i)
        {
            output.data[i] = (scratch.data[i] * target_weight) + (best_memory->value[i] * memory_weight);
        }
        AxTensor_NormalizeL2(axcore::MakeConstView(output.data, output.shape.total), output);

        out_candidate->similarity = best_similarity;
        out_candidate->fitness = axcore::Clamp(
            ((best_similarity + 1.0f) * 0.5f) +
                ((1.0f - profile->entropy) * 0.15f) +
                (zombie_mode ? 0.05f : 0.0f),
            0.0f,
            1.0f);
        axcore::CopyString(out_candidate->strategy, sizeof(out_candidate->strategy), "cache_bundle");
    }
    else
    {
        const uint32_t shift = output.shape.total > 1u ? ((iteration % (output.shape.total - 1u)) + 1u) : 0u;
        const AxStatus permute_status = AxTensor_Permute(axcore::MakeConstView(scratch.data, scratch.shape.total), static_cast<int32_t>(shift), output);
        if (permute_status != AX_STATUS_OK)
        {
            return permute_status;
        }

        float field_to_sink = 0.0f;
        AxTensor_CosineSimilarity(
            axcore::MakeConstView(scratch.data, scratch.shape.total),
            axcore::MakeConstView(target.data, target.shape.total),
            &field_to_sink);

        const AxStatus bundle_status =
            AxTensor_Bundle(axcore::MakeConstView(scratch.data, scratch.shape.total), axcore::MakeConstView(output.data, output.shape.total), 1u, output);
        if (bundle_status != AX_STATUS_OK)
        {
            return bundle_status;
        }

        float similarity = 0.0f;
        AxTensor_CosineSimilarity(axcore::MakeConstView(scratch.data, scratch.shape.total), axcore::MakeConstView(output.data, output.shape.total), &similarity);
        out_candidate->similarity = similarity;
        out_candidate->fitness = axcore::Clamp(
            ((similarity + 1.0f) * 0.5f) - (profile->entropy * 0.15f) + ((1.0f - profile->sparsity) * 0.08f),
            0.0f,
            1.0f);
        for (uint32_t i = 0u; i < output.shape.total; ++i)
        {
            output.data[i] += scratch.data[i] * 0.12f;
        }
        if (field_to_sink < 0.10f)
        {
            for (uint32_t i = 0u; i < output.shape.total; ++i)
            {
                output.data[i] += target.data[i] * 0.18f;
            }
        }
        const AxStatus renormalize_status = AxTensor_NormalizeL2(axcore::MakeConstView(output.data, output.shape.total), output);
        if (renormalize_status != AX_STATUS_OK)
        {
            return renormalize_status;
        }
        axcore::CopyString(out_candidate->strategy, sizeof(out_candidate->strategy), "self_permute");
    }

    if (profile->entropy > 0.85f && out_candidate->similarity < 0.20f && iteration > 32u)
    {
        axcore::CopyString(out_candidate->strategy, sizeof(out_candidate->strategy), "discovery_induction");
        out_candidate->fitness = profile->critic_acceptance_threshold + 0.05f;
    }

    out_candidate->cost = AxConnectome_CalculateThermodynamicCost(out_candidate, profile);
    return AX_STATUS_OK;
}

AxStatus AxMetabolicCritic_Tick(
    AxGeneRuntimeState* genes,
    uint32_t gene_count,
    const char* focused_gene_id,
    const AxMetabolicCriticConfig* config,
    float* io_noise_floor,
    AxSceneTickReport* out_report)
{
    if (io_noise_floor == nullptr || out_report == nullptr)
    {
        return AX_STATUS_INVALID_ARGUMENT;
    }

    AxMetabolicCriticConfig defaults{};
    if (config == nullptr)
    {
        AxMetabolicCriticConfig_Default(&defaults);
        config = &defaults;
    }

    out_report->active_genes = gene_count;
    axcore::CopyTrimmed(out_report->focused_gene_id, sizeof(out_report->focused_gene_id), focused_gene_id, false);
    out_report->budget_flux = config->flux_budget > 0.1f ? config->flux_budget : 0.1f;
    out_report->total_flux_before = 0.0f;
    out_report->total_flux_after = 0.0f;
    out_report->overload_ratio = 0.0f;
    out_report->latent_genes = 0u;
    out_report->autophagy_candidates = 0u;

    if (genes == nullptr || gene_count == 0u)
    {
        *io_noise_floor = *io_noise_floor + ((0.0f - *io_noise_floor) * config->noise_decay_rate);
        out_report->noise_floor = *io_noise_floor;
        return AX_STATUS_OK;
    }

    for (uint32_t i = 0u; i < gene_count; ++i)
    {
        const float weight = axcore::Clamp(genes[i].weight, 0.0f, 1.5f);
        const float cost = genes[i].cost_estimate < 0.01f ? 0.01f : genes[i].cost_estimate;
        out_report->total_flux_before += weight * cost;
    }

    if (out_report->total_flux_before > out_report->budget_flux)
    {
        out_report->overload_ratio = (out_report->total_flux_before - out_report->budget_flux) / out_report->budget_flux;
    }

    const float target_noise_floor =
        axcore::Clamp(out_report->overload_ratio * config->noise_gain, 0.0f, config->max_noise_floor);
    const float adaptation_rate = target_noise_floor > *io_noise_floor
                                      ? axcore::Clamp(config->noise_rise_rate, 0.01f, 1.0f)
                                      : axcore::Clamp(config->noise_decay_rate, 0.01f, 1.0f);
    *io_noise_floor = *io_noise_floor + ((target_noise_floor - *io_noise_floor) * adaptation_rate);

    for (uint32_t i = 0u; i < gene_count; ++i)
    {
        AxGeneRuntimeState& gene = genes[i];
        const bool focused = std::strcmp(gene.gene_id, out_report->focused_gene_id) == 0;
        float weight = axcore::Clamp(gene.weight, 0.0f, 1.5f);
        const float cost = gene.cost_estimate < 0.01f ? 0.01f : gene.cost_estimate;

        if (focused)
        {
            weight = weight > config->focused_floor_weight ? weight : config->focused_floor_weight;
        }
        else
        {
            float attenuation = 1.0f - (*io_noise_floor * axcore::Clamp(config->background_attenuation_scale, 0.0f, 1.0f));
            attenuation = axcore::Clamp(attenuation, 0.02f, 1.0f);
            weight *= attenuation;

            if (out_report->overload_ratio > 0.0f)
            {
                float overload_penalty = 1.0f - (out_report->overload_ratio * axcore::Clamp(config->overload_penalty_scale, 0.0f, 1.0f));
                overload_penalty = axcore::Clamp(overload_penalty, 0.10f, 1.0f);
                weight *= overload_penalty;
            }
        }

        gene.weight = axcore::Clamp(weight, 0.0f, 1.5f);
        if (gene.weight < (gene.min_weight > 0.0001f ? gene.min_weight : 0.0001f))
        {
            gene.latent_tick_count += 1u;
            out_report->latent_genes += 1u;
        }
        else
        {
            gene.latent_tick_count = 0u;
        }

        if (gene.is_autophagy_candidate != 0u)
        {
            out_report->autophagy_candidates += 1u;
        }

        out_report->total_flux_after += gene.weight * cost;
    }

    out_report->noise_floor = *io_noise_floor;
    return AX_STATUS_OK;
}
