#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define GGML_TURBO4_NCOLS2_HOST_DEVICE __host__ __device__
#else
#define GGML_TURBO4_NCOLS2_HOST_DEVICE
#endif

enum class ggml_turbo4_sym_lut_ncols2_decision {
    disabled,
    enabled,
    invalid_value,
    unavailable_in_build,
    unsupported_compute_capability,
};

// Keep the two-column experiment independent from the qualified one-column
// path. An explicit enable request must never degrade to the baseline silently.
static inline ggml_turbo4_sym_lut_ncols2_decision ggml_turbo4_sym_lut_ncols2_policy(
        const char * value, const bool compiled, const int compute_capability) {
    if (value == nullptr || std::strcmp(value, "0") == 0) {
        return ggml_turbo4_sym_lut_ncols2_decision::disabled;
    }
    if (std::strcmp(value, "1") != 0) {
        return ggml_turbo4_sym_lut_ncols2_decision::invalid_value;
    }
    if (!compiled) {
        return ggml_turbo4_sym_lut_ncols2_decision::unavailable_in_build;
    }
    if (compute_capability != 890) {
        return ggml_turbo4_sym_lut_ncols2_decision::unsupported_compute_capability;
    }
    return ggml_turbo4_sym_lut_ncols2_decision::enabled;
}

static constexpr bool ggml_turbo4_sym_lut_ncols2_device_supported(const int compute_capability) {
    return compute_capability == 890;
}

// Returns -1 when every visible device is supported, otherwise the first
// unsupported index. An empty set is not a valid enabled runtime.
static inline int ggml_turbo4_sym_lut_ncols2_first_unsupported_device(
        const int * compute_capabilities, const size_t device_count) {
    if (device_count == 0) {
        return 0;
    }
    for (size_t i = 0; i < device_count; ++i) {
        if (!ggml_turbo4_sym_lut_ncols2_device_supported(compute_capabilities[i])) {
            return int(i);
        }
    }
    return -1;
}

static constexpr int GGML_TURBO4_SYM_LUT_NCOLS2_COLUMNS = 2;
static constexpr int GGML_TURBO4_SYM_LUT_NCOLS2_MAX_QUERY_COLUMNS = 8;
static constexpr int GGML_TURBO4_SYM_LUT_MAGNITUDES = 8;
static constexpr int GGML_TURBO4_SYM_LUT_PADDING = 2;

GGML_TURBO4_NCOLS2_HOST_DEVICE
static constexpr bool ggml_turbo4_sym_lut_ncols2_head_size_supported(const int head_size) {
    return head_size == 128 || head_size == 256 || head_size == 512;
}

// Selector contract for the CUDA VEC path. Keeping it independent of ggml
// types makes the default-off routing policy testable without a CUDA compiler.
// The experiment owns only multi-column Turbo4 K shapes on exact SM89; N=1
// remains on the existing baseline selector and N>8 remains on MMA/TILE.
static constexpr bool ggml_turbo4_sym_lut_ncols2_vec_candidate(
        const bool experiment_enabled,
        const int compute_capability,
        const bool key_is_turbo4,
        const bool value_type_supported,
        const int head_size,
        const int64_t query_columns,
        const bool vector_kernel_shape_supported) {
    return experiment_enabled &&
        ggml_turbo4_sym_lut_ncols2_device_supported(compute_capability) &&
        key_is_turbo4 &&
        value_type_supported &&
        ggml_turbo4_sym_lut_ncols2_head_size_supported(head_size) &&
        query_columns >= GGML_TURBO4_SYM_LUT_NCOLS2_COLUMNS &&
        query_columns <= GGML_TURBO4_SYM_LUT_NCOLS2_MAX_QUERY_COLUMNS &&
        vector_kernel_shape_supported;
}

GGML_TURBO4_NCOLS2_HOST_DEVICE
static constexpr size_t ggml_turbo4_sym_lut_ncols2_shared_bytes(const int head_size) {
    return size_t(GGML_TURBO4_SYM_LUT_NCOLS2_COLUMNS) * size_t(head_size) *
        size_t(GGML_TURBO4_SYM_LUT_MAGNITUDES + GGML_TURBO4_SYM_LUT_PADDING) * sizeof(uint16_t);
}

// CUDA cannot opt a kernel with more than 48 KiB of statically declared shared
// memory into Ada's larger per-block budget. D=512 therefore places its KQ
// combine buffer in the same dynamic workspace as the LUT. value_cols_per_iter
// is 8 for Turbo4 V, 4 for F16 V, and 1 for Q8_0 V.
GGML_TURBO4_NCOLS2_HOST_DEVICE
static constexpr size_t ggml_turbo4_sym_lut_ncols2_d512_kq_bytes(
        const int head_size, const int value_cols_per_iter) {
    return head_size == 512 ?
        size_t(4) * size_t(value_cols_per_iter) * size_t(head_size) * sizeof(uint32_t) :
        0;
}

GGML_TURBO4_NCOLS2_HOST_DEVICE
static constexpr size_t ggml_turbo4_sym_lut_ncols2_dynamic_shared_bytes(
        const int head_size, const int value_cols_per_iter) {
    return ggml_turbo4_sym_lut_ncols2_shared_bytes(head_size) +
        ggml_turbo4_sym_lut_ncols2_d512_kq_bytes(head_size, value_cols_per_iter);
}

static constexpr size_t GGML_TURBO4_SYM_LUT_NCOLS2_SM89_OPTIN_SHARED_LIMIT = 99u * 1024u;
// KQ_max/KQ_sum consume 512 bytes in the D=512 ncols2 specialization. The
// one-element fallback KQ/LUT arrays and compiler alignment stay well within
// this conservative additional bound.
static constexpr size_t GGML_TURBO4_SYM_LUT_NCOLS2_STATIC_OVERHEAD_BOUND = 1024u;

#undef GGML_TURBO4_NCOLS2_HOST_DEVICE
