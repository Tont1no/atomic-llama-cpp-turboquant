#pragma once

#include "common.cuh"

// Strict native ModelOpt scalar W8A8 path. Returns false only when the op is
// not an F8_E4M3 scaled matmul; malformed/unsupported F8 ops fail closed.
bool ggml_cuda_mul_mat_f8_e4m3(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * weight,
        const ggml_tensor * activation,
        ggml_tensor * dst);

void ggml_cuda_fp8_cache_clear(ggml_backend_cuda_context * ctx);
