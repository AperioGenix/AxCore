#pragma once

#include "axcore/export.h"
#include "axcore/types.h"

#if defined(__cplusplus)
extern "C"
{
#endif

AXCORE_API AxStatus AXCORE_CALL AxShape_Make(const uint32_t* dims, uint32_t ndim, AxShape* out_shape);
AXCORE_API AxStatus AXCORE_CALL AxShape_Make1D(uint32_t total, AxShape* out_shape);
AXCORE_API uint32_t AXCORE_CALL AxShape_Equals(const AxShape* lhs, const AxShape* rhs);

AXCORE_API AxStatus AXCORE_CALL AxTensor_Copy(AxConstTensorView input, AxTensorView output);
AXCORE_API AxStatus AXCORE_CALL AxTensor_NormalizeL2(AxConstTensorView input, AxTensorView output);
AXCORE_API AxStatus AXCORE_CALL AxTensor_Subtract(AxConstTensorView lhs, AxConstTensorView rhs, AxTensorView output);
AXCORE_API AxStatus AXCORE_CALL AxTensor_Bundle(AxConstTensorView lhs, AxConstTensorView rhs, uint32_t normalize, AxTensorView output);
AXCORE_API AxStatus AXCORE_CALL AxTensor_Permute(AxConstTensorView input, int32_t steps, AxTensorView output);
AXCORE_API AxStatus AXCORE_CALL AxTensor_CosineSimilarity(AxConstTensorView lhs, AxConstTensorView rhs, float* out_similarity);

#if defined(__cplusplus)
}
#endif
