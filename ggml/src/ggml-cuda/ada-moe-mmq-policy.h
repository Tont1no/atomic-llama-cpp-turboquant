#pragma once

#include <cstring>

enum class ggml_cuda_ada_moe_mmq_decision {
    disabled,
    enabled,
    invalid_value,
    unavailable_in_build,
    unsupported_compute_capability,
};

struct ggml_cuda_ada_moe_mmq_policy_result {
    ggml_cuda_ada_moe_mmq_decision decision;
    int min_batch;
};

// Pure host policy for the experimental SM89 MUL_MAT_ID dispatch boundary.
// The production default remains MMVQ. An operator may opt into MMQ beginning
// at batch 2..8 only when the experiment was explicitly compiled into the
// binary. Invalid or unsupported enable requests must never silently fall back.
static inline ggml_cuda_ada_moe_mmq_policy_result ggml_cuda_ada_moe_mmq_policy(
        const char * value, const bool compiled, const int compute_capability) {
    if (value == nullptr || std::strcmp(value, "0") == 0) {
        return {ggml_cuda_ada_moe_mmq_decision::disabled, 0};
    }
    if (value[0] < '2' || value[0] > '8' || value[1] != '\0') {
        return {ggml_cuda_ada_moe_mmq_decision::invalid_value, 0};
    }
    if (!compiled) {
        return {ggml_cuda_ada_moe_mmq_decision::unavailable_in_build, 0};
    }
    if (compute_capability != 890) {
        return {ggml_cuda_ada_moe_mmq_decision::unsupported_compute_capability, 0};
    }
    return {ggml_cuda_ada_moe_mmq_decision::enabled, value[0] - '0'};
}
