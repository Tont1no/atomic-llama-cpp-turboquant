#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace qwen38_recurrent_parity {

inline bool batch_row_requires_logits(size_t row, size_t n_tokens, bool all_logits) {
    if (n_tokens == 0 || row >= n_tokens) {
        throw std::out_of_range("logits row is outside the batch");
    }
    return all_logits || row + 1 == n_tokens;
}

struct top2_result {
    int32_t top1 = -1;
    int32_t top2 = -1;
    float logit1 = -std::numeric_limits<float>::infinity();
    float logit2 = -std::numeric_limits<float>::infinity();

    float margin() const {
        return logit1 - logit2;
    }
};

inline top2_result select_top2(const float * logits, size_t n_vocab) {
    if (logits == nullptr || n_vocab < 2) {
        throw std::invalid_argument("top-2 selection requires at least two logits");
    }

    top2_result result;
    for (size_t i = 0; i < n_vocab; ++i) {
        const float value = logits[i];
        if (!std::isfinite(value)) {
            throw std::runtime_error("non-finite logit at token " + std::to_string(i));
        }
        if (value > result.logit1) {
            result.top2   = result.top1;
            result.logit2 = result.logit1;
            result.top1   = (int32_t) i;
            result.logit1 = value;
        } else if (value > result.logit2) {
            result.top2   = (int32_t) i;
            result.logit2 = value;
        }
    }

    if (result.top1 < 0 || result.top2 < 0) {
        throw std::runtime_error("top-2 selection found fewer than two finite logits");
    }
    return result;
}

inline uint64_t fnv1a64(const uint8_t * data, size_t size) {
    if (data == nullptr && size != 0) {
        throw std::invalid_argument("checksum data is null");
    }
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < size; ++i) {
        hash ^= data[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

inline uint32_t float_bits(float value) {
    uint32_t result;
    static_assert(sizeof(result) == sizeof(value), "unexpected float width");
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

inline std::string json_escape(const std::string & value) {
    static constexpr char hex[] = "0123456789abcdef";
    std::string result;
    result.reserve(value.size());
    for (unsigned char c : value) {
        switch (c) {
            case '"': result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\b': result += "\\b";  break;
            case '\f': result += "\\f";  break;
            case '\n': result += "\\n";  break;
            case '\r': result += "\\r";  break;
            case '\t': result += "\\t";  break;
            default:
                if (c < 0x20) {
                    result += "\\u00";
                    result += hex[c >> 4];
                    result += hex[c & 0x0f];
                } else {
                    result += (char) c;
                }
                break;
        }
    }
    return result;
}

inline std::vector<int32_t> parse_token_csv(const std::string & text) {
    std::vector<int32_t> result;
    size_t begin = 0;
    while (begin < text.size()) {
        const size_t end = text.find(',', begin);
        const std::string field = text.substr(begin, end == std::string::npos ? end : end - begin);
        if (field.empty()) {
            throw std::invalid_argument("empty token in reference CSV");
        }
        size_t parsed = 0;
        const long long value = std::stoll(field, &parsed, 10);
        if (parsed != field.size() || value < 0 || value > std::numeric_limits<int32_t>::max()) {
            throw std::invalid_argument("invalid token in reference CSV: " + field);
        }
        result.push_back((int32_t) value);
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1;
    }
    if (result.empty()) {
        throw std::invalid_argument("reference CSV is empty");
    }
    return result;
}

struct rollback_case {
    uint32_t rollback = 0;
    int32_t committed_token_index = -1;
    int32_t continuation_token_index = -1;
    int32_t continuation_prediction_index = -1;
};

struct boundary_comparison {
    int32_t first_mismatch_prediction = -1;
    uint32_t mismatch_count = 0;
    uint64_t scalar_window_hash = 0;
    uint64_t batch_window_hash = 0;
};

struct boundary_row_layout {
    size_t features = 0;
    size_t rows = 0;
    size_t row_bytes = 0;
    size_t total_bytes = 0;
};

// Comparable Qwen3.5 activation boundaries are canonical matrices:
// [n_embd, token_rows, 1, 1]. The recurrent GDN core is not such a boundary;
// its [S_v, H_v, token_rows, seqs] layout must never be flattened/remapped into
// one, even when scalar singleton axes make it superficially matrix-like.
inline bool canonical_f32_boundary_layout(
        const std::array<int64_t, 4> & ne,
        size_t expected_features,
        size_t expected_rows,
        boundary_row_layout & result) {
    result = {};
    if (expected_features == 0 || expected_rows == 0 || ne[0] <= 0 ||
            (uint64_t) ne[0] != expected_features ||
            expected_features > std::numeric_limits<size_t>::max() / sizeof(float)) {
        return false;
    }
    if (ne[1] <= 0 || (uint64_t) ne[1] != expected_rows || ne[2] != 1 || ne[3] != 1) {
        return false;
    }
    const size_t rows = (size_t) ne[1];
    const size_t row_bytes = expected_features * sizeof(float);
    if (rows > std::numeric_limits<size_t>::max() / row_bytes) {
        return false;
    }
    result = { expected_features, rows, row_bytes, rows * row_bytes };
    return true;
}

inline bool internal_f32_token_layout(
        const std::array<int64_t, 4> & ne,
        uint32_t feature_rank,
        const std::array<int64_t, 2> & expected_feature_shape,
        size_t expected_rows,
        boundary_row_layout & result) {
    result = {};
    if (feature_rank < 1 || feature_rank > 2 || expected_rows == 0) {
        return false;
    }
    size_t features = 1;
    for (uint32_t axis = 0; axis < feature_rank; ++axis) {
        if (expected_feature_shape[axis] <= 0 || ne[axis] != expected_feature_shape[axis] ||
                (uint64_t) ne[axis] > std::numeric_limits<size_t>::max() / features) {
            return false;
        }
        features *= (size_t) ne[axis];
    }
    if (ne[feature_rank] <= 0 || (uint64_t) ne[feature_rank] != expected_rows) {
        return false;
    }
    for (uint32_t axis = feature_rank + 1; axis < ne.size(); ++axis) {
        if (ne[axis] != 1) {
            return false;
        }
    }
    if (features > std::numeric_limits<size_t>::max() / sizeof(float) ||
            expected_rows > std::numeric_limits<size_t>::max() / (features * sizeof(float))) {
        return false;
    }
    const size_t row_bytes = features * sizeof(float);
    result = { features, expected_rows, row_bytes, expected_rows * row_bytes };
    return true;
}

inline uint64_t hash_u64_sequence(const std::vector<uint64_t> & values) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (uint64_t value : values) {
        for (uint32_t shift = 0; shift < 64; shift += 8) {
            hash ^= (uint8_t) (value >> shift);
            hash *= UINT64_C(1099511628211);
        }
    }
    return hash;
}

inline boundary_comparison compare_boundary_rows(
        const std::vector<uint64_t> & scalar_rows,
        const std::vector<uint64_t> & batch_rows,
        int32_t prediction_start) {
    if (prediction_start < 0 || scalar_rows.empty() || scalar_rows.size() != batch_rows.size()) {
        throw std::invalid_argument("boundary comparison requires equal non-empty row sets and a valid prediction start");
    }
    boundary_comparison result;
    result.scalar_window_hash = hash_u64_sequence(scalar_rows);
    result.batch_window_hash = hash_u64_sequence(batch_rows);
    for (size_t row = 0; row < scalar_rows.size(); ++row) {
        if (scalar_rows[row] != batch_rows[row]) {
            if (result.first_mismatch_prediction < 0) {
                result.first_mismatch_prediction = prediction_start + (int32_t) row;
            }
            ++result.mismatch_count;
        }
    }
    return result;
}

inline std::vector<rollback_case> make_rollback_cases(int32_t prediction_start, uint32_t width) {
    if (prediction_start <= 0 || width < 2) {
        throw std::invalid_argument("rollback plan requires a positive prediction start and width >= 2");
    }

    // A width-W teacher-forced verify batch predicts [prediction_start, ...,
    // prediction_start+W-1] from input tokens [prediction_start-1, ...,
    // prediction_start+W-2]. Rollback r commits through input W-1-r.
    std::vector<rollback_case> result;
    result.reserve(width - 1);
    for (uint32_t rollback = 1; rollback < width; ++rollback) {
        const int32_t committed = prediction_start + (int32_t) width - 2 - (int32_t) rollback;
        result.push_back({
            rollback,
            committed,
            committed + 1,
            committed + 2,
        });
    }
    return result;
}

} // namespace qwen38_recurrent_parity
