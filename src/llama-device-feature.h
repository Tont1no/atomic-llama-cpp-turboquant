#pragma once

#include <algorithm>
#include <cstddef>

inline size_t llama_device_feature_capacity(size_t n_batch, size_t enabled, size_t row_floats) {
    constexpr size_t budget = 60ULL << 20;
    if (!enabled || !row_floats || row_floats > budget / sizeof(float)) {
        return 0;
    }
    return std::min(n_batch, budget / enabled / (row_floats * sizeof(float)));
}

inline bool llama_device_feature_fits(size_t capacity, size_t token_offset, size_t n_tokens) {
    return n_tokens > 0 && token_offset <= capacity && n_tokens <= capacity - token_offset;
}
