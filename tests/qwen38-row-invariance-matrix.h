#pragma once

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace qwen38_row_invariance {

enum class operation : uint8_t {
    fp8, nvfp4_raw, nvfp4_ffn, nvfp4_head,
    nvfp4_ffn_row_invariant, nvfp4_down_row_invariant, gdn,
    rms_norm, ssm_conv, l2_norm, gated_norm, bf16_projection, bf16_projection_candidate,
    fattn_vec_cols2_pb1,
};

struct matrix_case {
    std::string_view id;
    operation op;
    int64_t k;
    int64_t n;
    bool quick;
};

inline constexpr int HEAD_MIN_M = 2;
inline constexpr int HEAD_MAX_M = 16;

struct head_route_observation {
    uint64_t m1;
    uint64_t scaled_before;
    uint64_t scaled_after;
    uint64_t generic_m8;
};

inline bool head_route_observation_valid(const head_route_observation & observation) {
    return observation.m1 > 0 && observation.scaled_after > observation.scaled_before &&
           observation.generic_m8 == 0;
}

inline constexpr std::array<matrix_case, 10> FULL_MATRIX = {{
    { "fp8-k5120-n1024",  operation::fp8,       5120,  1024,  true  },
    { "fp8-k5120-n6144",  operation::fp8,       5120,  6144,  false },
    { "fp8-k6144-n5120",  operation::fp8,       6144,  5120,  false },
    { "fp8-k5120-n10240", operation::fp8,       5120, 10240,  false },
    { "fp8-k5120-n12288", operation::fp8,       5120, 12288,  false },
    { "nvfp4-k5120-n128", operation::nvfp4_raw, 5120,   128,  true  },
    { "nvfp4-k17408-n128",operation::nvfp4_raw,17408,   128,  true  },
    { "nvfp4-qwen-ffn",   operation::nvfp4_ffn, 5120,   128,  true  },
    { "gdn-k1-vs-k8",     operation::gdn,         128,    32,  true  },
    { "gdn-seq1-vs-batch8",operation::gdn,         128,    32,  true  },
}};

inline constexpr matrix_case HEAD_CANDIDATE = {
    "nvfp4-qwen35-lm-head-row-invariant", operation::nvfp4_head, 5120, 128, true,
};

inline constexpr std::array<matrix_case, 2> PROJECTION_CANDIDATE = {{
    { "nvfp4-qwen35-ffn-gate-up-swiglu-row-invariant",
      operation::nvfp4_ffn_row_invariant, 5120, 17408, true },
    { "nvfp4-qwen35-ffn-down-row-invariant",
      operation::nvfp4_down_row_invariant, 17408, 5120, true },
}};

inline constexpr matrix_case RMS_NORM_CASE = {
    "qwen35-rms-norm-e5120", operation::rms_norm, 5120, 1, true,
};

inline constexpr matrix_case SSM_CONV_CASE = {
    "qwen35-ssm-conv-d4-c10240", operation::ssm_conv, 4, 10240, true,
};

inline constexpr std::array<matrix_case, 2> L2_NORM_CASES = {{
    { "qwen35-l2-norm-q-s128-h16", operation::l2_norm, 128, 16, true },
    { "qwen35-l2-norm-k-s128-h16", operation::l2_norm, 128, 16, true },
}};

inline constexpr matrix_case GATED_NORM_CASE = {
    "qwen35-post-gdn-gated-norm-s128-h48", operation::gated_norm, 128, 48, true,
};

inline constexpr std::array<matrix_case, 2> BF16_PROJECTION_CASES = {{
    { "qwen35-bf16-beta-k5120-n48", operation::bf16_projection, 5120, 48, true },
    { "qwen35-bf16-alpha-k5120-n48", operation::bf16_projection, 5120, 48, true },
}};

inline constexpr std::array<matrix_case, 2> BF16_PROJECTION_CANDIDATE_CASES = {{
    { "qwen35-bf16-beta-k5120-n48-row-invariant", operation::bf16_projection_candidate, 5120, 48, true },
    { "qwen35-bf16-alpha-k5120-n48-row-invariant", operation::bf16_projection_candidate, 5120, 48, true },
}};

inline constexpr matrix_case FATTN_VEC_COLS2_PB1_CASE = {
    "qwen35-fattn-d256-q8-gqa6-vec-cols2-pb1",
    operation::fattn_vec_cols2_pb1, 256, 1024, true,
};

inline std::vector<matrix_case> select_matrix(bool full) {
    std::vector<matrix_case> result;
    for (const auto & item : FULL_MATRIX) {
        if (full || item.quick) {
            result.push_back(item);
        }
    }
    if (result.size() != (full ? 10u : 6u)) {
        throw std::logic_error("Qwen3.8 row-invariance matrix cardinality drift");
    }
    return result;
}

inline std::vector<matrix_case> select_operation(operation op) {
    std::vector<matrix_case> result;
    for (const auto & item : FULL_MATRIX) {
        if (item.op == op) {
            result.push_back(item);
        }
    }
    const size_t expected = op == operation::fp8 ? 5u : op == operation::gdn ? 2u : 0u;
    if (expected == 0 || result.size() != expected) {
        throw std::logic_error("Qwen3.8 row-invariance operator matrix cardinality drift");
    }
    return result;
}

} // namespace qwen38_row_invariance
