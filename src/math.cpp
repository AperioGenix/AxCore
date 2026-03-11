#include "axcore_internal.h"

namespace
{
uint32_t ShapesCompatible(AxConstTensorView input, AxTensorView output)
{
    return input.shape.total == output.shape.total;
}
} // namespace

int32_t axcore::RoundBucket(float value)
{
    return value >= 0.0f ? static_cast<int32_t>(value + 0.5f) : static_cast<int32_t>(value - 0.5f);
}

bool axcore::NormalizeRawInPlace(float* values, uint32_t dim)
{
    if (values == nullptr || dim == 0u)
    {
        return false;
    }

    double sum_sq = 0.0;
    for (uint32_t i = 0u; i < dim; ++i)
    {
        const float value = Sanitize(values[i]);
        sum_sq += static_cast<double>(value) * static_cast<double>(value);
    }

    const double norm = std::sqrt(sum_sq);
    if (!(norm > 1.0e-8))
    {
        ZeroVector(values, dim);
        return false;
    }

    const float inverse = static_cast<float>(1.0 / norm);
    for (uint32_t i = 0u; i < dim; ++i)
    {
        values[i] = Sanitize(values[i]) * inverse;
    }
    return true;
}

float axcore::CosineRaw(const float* lhs, const float* rhs, uint32_t dim)
{
    if (lhs == nullptr || rhs == nullptr || dim == 0u)
    {
        return 0.0f;
    }

    double dot = 0.0;
    double lhs_sq = 0.0;
    double rhs_sq = 0.0;
    for (uint32_t i = 0u; i < dim; ++i)
    {
        const double left = static_cast<double>(Sanitize(lhs[i]));
        const double right = static_cast<double>(Sanitize(rhs[i]));
        dot += left * right;
        lhs_sq += left * left;
        rhs_sq += right * right;
    }

    const double denominator = std::sqrt(lhs_sq) * std::sqrt(rhs_sq);
    if (!(denominator > 1.0e-8))
    {
        return 0.0f;
    }

    return Clamp(static_cast<float>(dot / denominator), -1.0f, 1.0f);
}

AxStatus AxShape_Make(const uint32_t* dims, uint32_t ndim, AxShape* out_shape)
{
    if (out_shape == nullptr)
    {
        return AX_STATUS_INVALID_ARGUMENT;
    }

    axcore::ClearShape(out_shape);
    if (ndim > AXCORE_MAX_DIMS)
    {
        return AX_STATUS_LIMIT_EXCEEDED;
    }
    if (ndim > 0u && dims == nullptr)
    {
        return AX_STATUS_INVALID_ARGUMENT;
    }
    if (ndim == 0u)
    {
        return AX_STATUS_OK;
    }

    uint64_t total = 1u;
    out_shape->ndim = ndim;
    for (uint32_t i = 0u; i < ndim; ++i)
    {
        out_shape->dims[i] = dims[i];
        total *= static_cast<uint64_t>(dims[i]);
        if (total > 0xFFFFFFFFull)
        {
            axcore::ClearShape(out_shape);
            return AX_STATUS_LIMIT_EXCEEDED;
        }
    }
    out_shape->total = static_cast<uint32_t>(total);
    return AX_STATUS_OK;
}

AxStatus AxShape_Make1D(uint32_t total, AxShape* out_shape)
{
    if (out_shape == nullptr)
    {
        return AX_STATUS_INVALID_ARGUMENT;
    }

    axcore::ClearShape(out_shape);
    if (total == 0u)
    {
        return AX_STATUS_OK;
    }

    out_shape->ndim = 1u;
    out_shape->dims[0] = total;
    out_shape->total = total;
    return AX_STATUS_OK;
}

uint32_t AxShape_Equals(const AxShape* lhs, const AxShape* rhs)
{
    if (lhs == nullptr || rhs == nullptr)
    {
        return 0u;
    }
    if (lhs->ndim != rhs->ndim || lhs->total != rhs->total)
    {
        return 0u;
    }
    for (uint32_t i = 0u; i < AXCORE_MAX_DIMS; ++i)
    {
        if (lhs->dims[i] != rhs->dims[i])
        {
            return 0u;
        }
    }
    return 1u;
}

AxStatus AxTensor_Copy(AxConstTensorView input, AxTensorView output)
{
    if (input.shape.total == 0u)
    {
        return AX_STATUS_OK;
    }
    if (input.data == nullptr || output.data == nullptr)
    {
        return AX_STATUS_INVALID_ARGUMENT;
    }
    if (ShapesCompatible(input, output) == 0u)
    {
        return AX_STATUS_DIMENSION_MISMATCH;
    }

    for (uint32_t i = 0u; i < input.shape.total; ++i)
    {
        output.data[i] = axcore::Sanitize(input.data[i]);
    }
    return AX_STATUS_OK;
}

AxStatus AxTensor_NormalizeL2(AxConstTensorView input, AxTensorView output)
{
    if (input.shape.total == 0u)
    {
        return AX_STATUS_OK;
    }
    if (input.data == nullptr || output.data == nullptr)
    {
        return AX_STATUS_INVALID_ARGUMENT;
    }
    if (ShapesCompatible(input, output) == 0u)
    {
        return AX_STATUS_DIMENSION_MISMATCH;
    }

    if (input.data != output.data)
    {
        for (uint32_t i = 0u; i < input.shape.total; ++i)
        {
            output.data[i] = axcore::Sanitize(input.data[i]);
        }
    }
    else
    {
        for (uint32_t i = 0u; i < input.shape.total; ++i)
        {
            output.data[i] = axcore::Sanitize(output.data[i]);
        }
    }
    axcore::NormalizeRawInPlace(output.data, output.shape.total);
    return AX_STATUS_OK;
}

AxStatus AxTensor_Subtract(AxConstTensorView lhs, AxConstTensorView rhs, AxTensorView output)
{
    if (lhs.data == nullptr || rhs.data == nullptr || output.data == nullptr)
    {
        return AX_STATUS_INVALID_ARGUMENT;
    }
    if (lhs.shape.total != rhs.shape.total || lhs.shape.total != output.shape.total)
    {
        return AX_STATUS_DIMENSION_MISMATCH;
    }

    for (uint32_t i = 0u; i < lhs.shape.total; ++i)
    {
        output.data[i] = axcore::Sanitize(lhs.data[i]) - axcore::Sanitize(rhs.data[i]);
    }
    return AX_STATUS_OK;
}

AxStatus AxTensor_Bundle(AxConstTensorView lhs, AxConstTensorView rhs, uint32_t normalize, AxTensorView output)
{
    if (lhs.data == nullptr || rhs.data == nullptr || output.data == nullptr)
    {
        return AX_STATUS_INVALID_ARGUMENT;
    }
    if (lhs.shape.total != rhs.shape.total || lhs.shape.total != output.shape.total)
    {
        return AX_STATUS_DIMENSION_MISMATCH;
    }

    for (uint32_t i = 0u; i < lhs.shape.total; ++i)
    {
        output.data[i] = axcore::Sanitize(lhs.data[i]) + axcore::Sanitize(rhs.data[i]);
    }

    if (normalize != 0u)
    {
        return AxTensor_NormalizeL2(axcore::MakeConstView(output.data, output.shape.total), output);
    }
    return AX_STATUS_OK;
}

AxStatus AxTensor_Permute(AxConstTensorView input, int32_t steps, AxTensorView output)
{
    if (input.shape.total == 0u)
    {
        return AX_STATUS_OK;
    }
    if (input.data == nullptr || output.data == nullptr)
    {
        return AX_STATUS_INVALID_ARGUMENT;
    }
    if (ShapesCompatible(input, output) == 0u)
    {
        return AX_STATUS_DIMENSION_MISMATCH;
    }

    const uint32_t count = input.shape.total;
    const int32_t mod = steps % static_cast<int32_t>(count);
    const uint32_t shift = mod < 0 ? static_cast<uint32_t>(mod + static_cast<int32_t>(count)) : static_cast<uint32_t>(mod);

    if (shift == 0u)
    {
        return AxTensor_Copy(input, output);
    }

    for (uint32_t i = 0u; i < count; ++i)
    {
        const uint32_t dst = (i + shift) % count;
        output.data[dst] = axcore::Sanitize(input.data[i]);
    }
    return AX_STATUS_OK;
}

AxStatus AxTensor_CosineSimilarity(AxConstTensorView lhs, AxConstTensorView rhs, float* out_similarity)
{
    if (out_similarity == nullptr)
    {
        return AX_STATUS_INVALID_ARGUMENT;
    }

    *out_similarity = 0.0f;
    if (lhs.data == nullptr || rhs.data == nullptr)
    {
        return AX_STATUS_INVALID_ARGUMENT;
    }
    if (lhs.shape.total != rhs.shape.total)
    {
        return AX_STATUS_DIMENSION_MISMATCH;
    }
    if (lhs.shape.total == 0u)
    {
        return AX_STATUS_OK;
    }

    *out_similarity = axcore::CosineRaw(lhs.data, rhs.data, lhs.shape.total);
    return AX_STATUS_OK;
}

AxStatus axcore::FoldNumericSignal(const float* values, uint32_t count, uint32_t target_dim, float* out_values)
{
    if (values == nullptr || out_values == nullptr || count == 0u || target_dim == 0u)
    {
        return AX_STATUS_INVALID_ARGUMENT;
    }

    ZeroVector(out_values, target_dim);
    for (uint32_t i = 0u; i < count; ++i)
    {
        const float value = Sanitize(values[i]);
        const uint64_t p1 = static_cast<uint64_t>(i) * 1315423ull;
        const uint64_t p2 = static_cast<uint64_t>(i) * 2654435ull;
        const uint64_t p3 = static_cast<uint64_t>(i) * 805459ull;

        const uint32_t s1 = static_cast<uint32_t>(p1 % target_dim);
        const uint32_t s2 = static_cast<uint32_t>(p2 % target_dim);
        const uint32_t s3 = static_cast<uint32_t>(p3 % target_dim);

        out_values[s1] += value;
        out_values[s2] -= value * 0.5f;
        out_values[s3] += value * 0.5f;
    }

    return AxTensor_NormalizeL2(MakeConstView(out_values, target_dim), MakeView(out_values, target_dim));
}

AxStatus axcore::PrepareHypervector(const float* values, uint32_t count, uint32_t target_dim, float* out_values)
{
    if (values == nullptr || out_values == nullptr || count == 0u || target_dim == 0u)
    {
        return AX_STATUS_INVALID_ARGUMENT;
    }

    if (count == target_dim)
    {
        for (uint32_t i = 0u; i < target_dim; ++i)
        {
            out_values[i] = Sanitize(values[i]);
        }
        return AxTensor_NormalizeL2(MakeConstView(out_values, target_dim), MakeView(out_values, target_dim));
    }

    return FoldNumericSignal(values, count, target_dim, out_values);
}
