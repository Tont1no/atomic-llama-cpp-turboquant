#pragma once

#include <cstring>

enum class ggml_turbo4_sym_lut_decision {
    disabled,
    enabled,
    invalid_value,
    unavailable_in_build,
    unsupported_compute_capability,
};

// Pure host policy used by the CUDA launch boundary and model-free tests.
// Environment parsing is intentionally strict: only unset, "0", and "1" are
// accepted, and an enable request never silently falls back.
static inline ggml_turbo4_sym_lut_decision ggml_turbo4_sym_lut_policy(
        const char * value, const bool compiled, const int compute_capability) {
    if (value == nullptr || std::strcmp(value, "0") == 0) {
        return ggml_turbo4_sym_lut_decision::disabled;
    }
    if (std::strcmp(value, "1") != 0) {
        return ggml_turbo4_sym_lut_decision::invalid_value;
    }
    if (!compiled) {
        return ggml_turbo4_sym_lut_decision::unavailable_in_build;
    }
    if (compute_capability != 890) {
        return ggml_turbo4_sym_lut_decision::unsupported_compute_capability;
    }
    return ggml_turbo4_sym_lut_decision::enabled;
}
