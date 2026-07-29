#pragma once

#include "ggml.h"

#include <cstdint>
#include <cstdlib>

// Keep experimental TurboQuant KV safety decisions pure and model-local.
// These helpers are shared by the cache constructor and model-free tests.
inline bool llama_kv_type_is_turbo(ggml_type type) {
    return type == GGML_TYPE_TURBO2_0 ||
           type == GGML_TYPE_TURBO3_0 ||
           type == GGML_TYPE_TURBO4_0;
}

inline bool llama_kv_turbo_k_can_shift(ggml_type type_k) {
    // A correct position shift requires inverse WHT, the RoPE delta, WHT and
    // requantization. Skipping that transformation corrupts shifted keys.
    return !llama_kv_type_is_turbo(type_k);
}

inline int llama_kv_turbo_adaptive_mode(
        ggml_type type_v,
        uint32_t n_layer,
        const char * env_value) {
    if (env_value != nullptr) {
        return std::atoi(env_value);
    }

    // Preserve the existing policy, but derive it for every cache instance
    // instead of pinning the first model's choice process-wide.
    return type_v == GGML_TYPE_TURBO2_0 && n_layer >= 8 ? 7 : 0;
}

inline bool llama_kv_turbo_innerq_requested(const char * env_value) {
    if (env_value == nullptr || env_value[0] == '\0') {
        return false;
    }

    char * end = nullptr;
    const long target_tokens = std::strtol(env_value, &end, 10);
    return end != env_value && target_tokens > 0;
}
