#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

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
static constexpr int GGML_TURBO4_SYM_LUT_MAGNITUDES = 8;
static constexpr int GGML_TURBO4_SYM_LUT_PADDING = 2;

static constexpr bool ggml_turbo4_sym_lut_ncols2_head_size_supported(const int head_size) {
    return head_size == 128 || head_size == 256;
}

static constexpr size_t ggml_turbo4_sym_lut_ncols2_shared_bytes(const int head_size) {
    return size_t(GGML_TURBO4_SYM_LUT_NCOLS2_COLUMNS) * size_t(head_size) *
        size_t(GGML_TURBO4_SYM_LUT_MAGNITUDES + GGML_TURBO4_SYM_LUT_PADDING) * sizeof(uint16_t);
}
