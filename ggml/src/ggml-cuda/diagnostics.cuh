#pragma once

#include <cstddef>
#include <cstdint>

// Internal, process-local route evidence for guarded CUDA diagnostics.
enum ggml_cuda_diagnostic_route : uint32_t {
    GGML_CUDA_DIAGNOSTIC_ROUTE_FP8_E4M3 = 0,
    GGML_CUDA_DIAGNOSTIC_ROUTE_FP8_E4M3_BATCH,
    GGML_CUDA_DIAGNOSTIC_ROUTE_NVFP4_MMVQ_M1,
    GGML_CUDA_DIAGNOSTIC_ROUTE_NVFP4_MMVQ_M8,
    GGML_CUDA_DIAGNOSTIC_ROUTE_NVFP4_M1_FUSED_FFN,
    GGML_CUDA_DIAGNOSTIC_ROUTE_NVFP4_ROW_INVARIANT_HEAD_SCALED,
    GGML_CUDA_DIAGNOSTIC_ROUTE_NVFP4_ROW_INVARIANT_FFN_FUSED,
    GGML_CUDA_DIAGNOSTIC_ROUTE_NVFP4_ROW_INVARIANT_FFN_DOWN_SCALED,
    GGML_CUDA_DIAGNOSTIC_ROUTE_GATED_DELTA_NET,
    GGML_CUDA_DIAGNOSTIC_ROUTE_GATED_DELTA_NET_FUSED_CACHE,
    GGML_CUDA_DIAGNOSTIC_ROUTE_RMS_NORM,
    GGML_CUDA_DIAGNOSTIC_ROUTE_RMS_NORM_MUL,
    GGML_CUDA_DIAGNOSTIC_ROUTE_SSM_CONV,
    GGML_CUDA_DIAGNOSTIC_ROUTE_SSM_CONV_SILU,
    GGML_CUDA_DIAGNOSTIC_ROUTE_L2_NORM,
    GGML_CUDA_DIAGNOSTIC_ROUTE_SILU,
    GGML_CUDA_DIAGNOSTIC_ROUTE_SILU_MUL,
    GGML_CUDA_DIAGNOSTIC_ROUTE_BF16_MMVF,
    GGML_CUDA_DIAGNOSTIC_ROUTE_BF16_CUBLAS,
    GGML_CUDA_DIAGNOSTIC_ROUTE_BF16_ROW_INVARIANT_BETA,
    GGML_CUDA_DIAGNOSTIC_ROUTE_BF16_ROW_INVARIANT_ALPHA,
    GGML_CUDA_DIAGNOSTIC_ROUTE_FATTN_VEC,
    GGML_CUDA_DIAGNOSTIC_ROUTE_FATTN_MMA_F16,
    GGML_CUDA_DIAGNOSTIC_ROUTE_FATTN_TILE,
    GGML_CUDA_DIAGNOSTIC_ROUTE_SIGMOID,
    GGML_CUDA_DIAGNOSTIC_ROUTE_SIGMOID_MUL,
    GGML_CUDA_DIAGNOSTIC_ROUTE_SET_ROWS,
    GGML_CUDA_DIAGNOSTIC_ROUTE_ROPE_VIEW_SET_ROWS,
    GGML_CUDA_DIAGNOSTIC_ROUTE_FATTN_QWEN35_D256_Q8_GQA6_VEC,
    GGML_CUDA_DIAGNOSTIC_ROUTE_FATTN_QWEN35_D256_Q8_GQA6_HINT,
    GGML_CUDA_DIAGNOSTIC_ROUTE_FATTN_QWEN35_D256_Q8_GQA6_VEC_COLS2_PB1,
    GGML_CUDA_DIAGNOSTIC_ROUTE_COUNT,
};

struct ggml_cuda_diagnostic_route_counters {
    uint32_t abi_version;
    uint32_t route_count;
    uint64_t count[GGML_CUDA_DIAGNOSTIC_ROUTE_COUNT];
};

struct ggml_cuda_diagnostic_fattn_candidate_observation {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t evaluated;
    uint32_t src4_null;
    uint64_t fail_mask;
    int32_t device_cc;
    int32_t compiled_arch;
    int32_t op;
    int32_t hint;
    int32_t q_type;
    int32_t k_type;
    int32_t v_type;
    int32_t mask_type;
    int32_t dst_type;
    int32_t prec;
    uint32_t scale_bits;
    uint32_t max_bias_bits;
    uint32_t softcap_bits;
    uint32_t mask_contiguous;
    uint32_t dst_contiguous;
    int64_t ne[5][4];
    uint64_t nb[5][4];
};

struct ggml_cuda_diagnostic_fattn_vec_launch_observation {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t evaluated;
    uint32_t forced_parallel_blocks;
    int32_t q_cols;
    int32_t ncols;
    int32_t ntiles_x;
    int32_t ntiles_kv;
    int32_t parallel_blocks;
};

void ggml_cuda_diagnostic_route_hit(ggml_cuda_diagnostic_route route);
void ggml_cuda_diagnostic_route_reset();
bool ggml_cuda_diagnostic_route_snapshot_v2(
        ggml_cuda_diagnostic_route_counters * out, size_t out_size);
void ggml_cuda_diagnostic_fattn_candidate_observe(
        const ggml_cuda_diagnostic_fattn_candidate_observation & observation);
bool ggml_cuda_diagnostic_fattn_candidate_snapshot_v1(
        ggml_cuda_diagnostic_fattn_candidate_observation * out, size_t out_size);
void ggml_cuda_diagnostic_fattn_vec_launch_observe(
        const ggml_cuda_diagnostic_fattn_vec_launch_observation & observation);
bool ggml_cuda_diagnostic_fattn_vec_launch_snapshot_v1(
        ggml_cuda_diagnostic_fattn_vec_launch_observation * out, size_t out_size);

#ifdef GGML_CUDA_DIAGNOSTIC_ROUTES
#define GGML_CUDA_DIAGNOSTIC_ROUTE_HIT(route) ggml_cuda_diagnostic_route_hit(route)
#define GGML_CUDA_DIAGNOSTIC_FATTN_CANDIDATE_OBSERVE(observation) \
    ggml_cuda_diagnostic_fattn_candidate_observe(observation)
#define GGML_CUDA_DIAGNOSTIC_FATTN_VEC_LAUNCH_OBSERVE(observation) \
    ggml_cuda_diagnostic_fattn_vec_launch_observe(observation)
#else
#define GGML_CUDA_DIAGNOSTIC_ROUTE_HIT(route) do { } while (0)
#define GGML_CUDA_DIAGNOSTIC_FATTN_CANDIDATE_OBSERVE(observation) do { } while (0)
#define GGML_CUDA_DIAGNOSTIC_FATTN_VEC_LAUNCH_OBSERVE(observation) do { } while (0)
#endif
