#include "llama.h"
#include "llama-pyramidkv-c1.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

// The schedule, recent-token protection, zero-padded average pooling, and
// deterministic position tie-break mirror pyramidkv_reference.h.  Selection
// is intentionally repeated independently for every KV head.

namespace {

constexpr std::size_t kMaxTokens = 1u << 20;
constexpr std::size_t kMaxObservation = 512;
constexpr std::size_t kMaxCapacity = 1u << 16;
constexpr std::size_t kMaxRecent = 1u << 14;
constexpr std::size_t kMaxHeads = 1024;
constexpr std::size_t kMinObserverBytes = 1u << 20;
constexpr std::size_t kMaxObserverBytes = 1ull << 36;
constexpr std::size_t kMaxObserverChunk = 512;
constexpr std::size_t kMinTransitionBytes = 64u*1024u*1024u;
constexpr std::size_t kMaxTransitionBytes = 1ull << 40;
constexpr std::size_t kMaxHotCapacity = 1u << 16;
constexpr std::size_t kMaxListCapacity = 1u << 22;

std::size_t checked_mul(std::size_t left, std::size_t right, bool & ok) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        ok = false;
        return 0;
    }
    return left * right;
}

void pool_average(const std::vector<float> & scores, std::size_t offset,
        std::size_t count, std::size_t kernel, std::vector<float> & output) {
    output.resize(count);
    const std::size_t radius = kernel / 2;

    for (std::size_t index = 0; index < count; ++index) {
        const std::size_t first = index > radius ? index - radius : 0;
        const std::size_t last = std::min(count - 1, index + radius);
        double sum = 0.0;
        for (std::size_t neighbour = first; neighbour <= last; ++neighbour) {
            sum += scores[offset + neighbour];
        }
        output[index] = static_cast<float>(sum / static_cast<double>(kernel));
    }
}

} // namespace

namespace {

bool copy_c1_value(const char * name, uint64_t value, uint64_t minimum,
        uint64_t maximum, std::size_t & output, std::string & error) {
    if (value < minimum || value > maximum ||
            value > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max())) {
        std::ostringstream message;
        message << name << " must be an integer in [" << minimum << ", "
                << maximum << "]";
        error = message.str();
        return false;
    }
    output = static_cast<std::size_t>(value);
    return true;
}

} // namespace

bool llama_pyramidkv_c1_make_config(
        const llama_pyramidkv_c1_params & params,
        std::size_t layer_count,
        std::size_t continuation_headroom,
        llama_pyramidkv_c1_config & output,
        std::string & error) {
    output = {};
    if (!params.enabled) {
        return true;
    }
    if (layer_count == 0 || layer_count > kMaxHeads) {
        error = "invalid attention layer count";
        return false;
    }
    if (continuation_headroom == 0) {
        error = "PyramidKV C1 requires a non-zero effective ubatch headroom";
        return false;
    }

    output.enabled = true;
    output.layer_count = layer_count;
    if (!copy_c1_value("pyramidkv_c1.max_capacity_prompt", params.max_capacity_prompt,
            2, kMaxCapacity, output.max_capacity_prompt, error) ||
        !copy_c1_value("pyramidkv_c1.beta", params.beta, 1, 1000,
            output.beta, error) ||
        !copy_c1_value("pyramidkv_c1.observation_window", params.observation_window,
            1, kMaxObservation, output.observation_window, error) ||
        !copy_c1_value("pyramidkv_c1.recent_window", params.recent_window,
            1, kMaxRecent, output.recent_window, error) ||
        !copy_c1_value("pyramidkv_c1.pooling_kernel", params.pooling_kernel,
            1, 63, output.pooling_kernel, error) ||
        !copy_c1_value("pyramidkv_c1.observer_max_bytes", params.observer_max_bytes,
            kMinObserverBytes, kMaxObserverBytes, output.observer_max_bytes, error) ||
        !copy_c1_value("pyramidkv_c1.observer_chunk", params.observer_chunk,
            1, kMaxObserverChunk, output.observer_chunk, error) ||
        !copy_c1_value("pyramidkv_c1.transition_max_bytes", params.transition_max_bytes,
            kMinTransitionBytes, kMaxTransitionBytes, output.transition_max_bytes, error) ||
        !copy_c1_value("pyramidkv_c1.hot_capacity", params.hot_capacity,
            2, kMaxHotCapacity, output.hot_capacity, error)) {
        return false;
    }
    output.continuation_headroom = continuation_headroom;

    if ((output.pooling_kernel & 1u) == 0 ||
        output.max_capacity_prompt <= output.recent_window) {
        error = "PyramidKV C1 requires odd pooling and capacity > recent";
        return false;
    }
    if (output.recent_window > std::numeric_limits<std::size_t>::max() -
            output.continuation_headroom) {
        error = "PyramidKV C1 recent window plus effective ubatch headroom overflows";
        return false;
    }
    const std::size_t minimum_hot_capacity =
        output.recent_window + output.continuation_headroom;
    if (output.hot_capacity < minimum_hot_capacity) {
        error = "PyramidKV C1 hot_capacity must cover recent_window plus effective ubatch headroom";
        return false;
    }
    output.paged = params.paged;
    if (output.paged) {
        if (!copy_c1_value("pyramidkv_c1.list_capacity", params.list_capacity,
                1, kMaxListCapacity, output.list_capacity, error)) {
            return false;
        }
        if (output.list_capacity < output.max_capacity_prompt + output.continuation_headroom) {
            error = "PyramidKV C1 list_capacity must cover max_capacity_prompt plus effective ubatch headroom";
            return false;
        }
        if (!copy_c1_value("pyramidkv_c1.paged_union_factor", params.paged_union_factor,
                1, 1024, output.paged_union_factor, error)) {
            return false;
        }
    }
    return true;
}

bool llama_pyramidkv_c1_select(
        const llama_pyramidkv_c1_score & score,
        const llama_pyramidkv_c1_config & config,
        llama_pyramidkv_c1_layer_selection & output,
        std::string & error) {
    if (score.il < 0 || config.layer_index >= config.layer_count) {
        error = "score layer is outside the attention layer range";
        return false;
    }
    if (score.query_heads == 0 || score.query_heads > kMaxHeads ||
        score.kv_heads == 0 || score.kv_heads > kMaxHeads ||
        score.query_heads % score.kv_heads != 0 || score.query_tokens == 0 ||
        score.key_tokens == 0 || score.key_tokens > kMaxTokens ||
        score.logical_key_tokens == 0 || score.logical_key_tokens > kMaxTokens ||
        score.key_stride < score.key_tokens ||
        score.query_positions.size() != score.query_tokens ||
        score.key_positions_per_head.size() != score.kv_heads ||
        score.key_cells_per_head.size() != score.kv_heads ||
        (!score.key_score_slots_per_head.empty() &&
         score.key_score_slots_per_head.size() != score.kv_heads)) {
        error = "invalid PyramidKV score geometry";
        return false;
    }

    if (score.observation_window > kMaxObservation ||
        score.query_tokens > kMaxObservation * 4) {
        error = "score tensor exceeds the C1 bounded observation limit";
        return false;
    }

    bool size_ok = true;
    const std::size_t head_score_count = checked_mul(
        score.kv_heads, score.key_stride, size_ok);
    if (!size_ok || score.head_scores.size() != head_score_count) {
        error = "reduced score tensor has an unexpected contiguous size";
        return false;
    }

    for (std::size_t i = 0; i < score.query_positions.size(); ++i) {
        if (score.query_positions[i] < 0 ||
            (i != 0 && score.query_positions[i] <= score.query_positions[i - 1])) {
            error = "query positions are not strictly increasing";
            return false;
        }
    }
    for (std::size_t head = 0; head < score.kv_heads; ++head) {
        const auto & positions = score.key_positions_per_head[head];
        const auto & cells = score.key_cells_per_head[head];
        const auto & slots = score.key_score_slots_per_head.empty() ? cells :
            score.key_score_slots_per_head[head];
        if (positions.empty() || positions.size() != cells.size() ||
                positions.size() > score.key_stride || slots.size() != cells.size()) {
            error = "a KV head has no bounded key position list";
            return false;
        }
        for (std::size_t i = 0; i < positions.size(); ++i) {
            if (positions[i] < 0 || cells[i] >= score.logical_key_tokens ||
                    slots[i] >= score.key_stride ||
                    (i != 0 && positions[i] <= positions[i - 1])) {
                error = "a KV head has non-monotonic original positions";
                return false;
            }
            const std::size_t score_index = slots[i] + score.key_stride * head;
            if (score_index >= score.head_scores.size() ||
                    !std::isfinite(score.head_scores[score_index])) {
                error = "score tensor contains a non-finite value";
                return false;
            }
        }
        auto unique_cells = cells;
        std::sort(unique_cells.begin(), unique_cells.end());
        if (std::adjacent_find(unique_cells.begin(), unique_cells.end()) != unique_cells.end()) {
            error = "a KV head contains duplicate logical cells";
            return false;
        }
        if (!score.key_score_slots_per_head.empty()) {
            auto unique_slots = slots;
            std::sort(unique_slots.begin(), unique_slots.end());
            if (std::adjacent_find(unique_slots.begin(), unique_slots.end()) != unique_slots.end()) {
                error = "a KV head aliases multiple logical cells to one score slot";
                return false;
            }
        }
    }

    output = {};
    output.il = score.il;
    output.source_tokens = score.logical_key_tokens;
    output.kv_heads = score.kv_heads;
    output.continuation_headroom = std::max<std::size_t>(1, config.continuation_headroom);
    output.heads.resize(score.kv_heads);

    for (std::size_t head = 0; head < score.kv_heads; ++head) {
        const auto & cells = score.key_cells_per_head[head];
        const auto & positions = score.key_positions_per_head[head];
        const auto & slots = score.key_score_slots_per_head.empty() ? cells :
            score.key_score_slots_per_head[head];
        const std::size_t valid_tokens = cells.size();
        const std::size_t recent_begin = valid_tokens > config.recent_window
            ? valid_tokens - config.recent_window : 0;
        const std::size_t past_count = recent_begin;

        // Retained rows are not the full context length. Reusing their count
        // would keep every layer in the uniform transition budget forever.
        const std::size_t schedule_tokens = std::max(valid_tokens,
            static_cast<std::size_t>(score.query_positions.back()) + 1);
        std::size_t older_to_keep = past_count;
        if (schedule_tokens > config.max_capacity_prompt) {
            const std::size_t base = config.max_capacity_prompt - config.recent_window;
            bool schedule_ok = true;
            const std::size_t double_base = checked_mul(base, 2, schedule_ok);
            if (!schedule_ok) {
                error = "PyramidKV capacity schedule overflow";
                return false;
            }
            std::size_t min_tokens = base / config.beta;
            std::size_t max_tokens = double_base - min_tokens;
            const std::size_t available_older = schedule_tokens - config.recent_window;
            if (max_tokens >= available_older) {
                max_tokens = available_older;
                min_tokens = double_base < max_tokens ? 0 : double_base - max_tokens;
            }
            if (min_tokens > max_tokens) {
                error = "PyramidKV layer schedule is invalid";
                return false;
            }
            const std::size_t step = config.layer_count == 1 ? 0 :
                (max_tokens - min_tokens) / (config.layer_count - 1);
            const bool transition = schedule_tokens < double_base;
            std::size_t decrement = 0;
            if (!transition) {
                decrement = checked_mul(config.layer_index, step, schedule_ok);
                if (!schedule_ok || decrement > max_tokens) {
                    error = "PyramidKV layer schedule overflow";
                    return false;
                }
            }
            older_to_keep = std::min(transition ? base : max_tokens - decrement, past_count);
        }

        std::vector<float> scores(valid_tokens, 0.0f);
        for (std::size_t key = 0; key < valid_tokens; ++key) {
            scores[key] = score.head_scores[slots[key] + score.key_stride * head] /
                static_cast<float>(score.query_tokens);
            if (!std::isfinite(scores[key])) {
                error = "score tensor contains a non-finite value";
                return false;
            }
        }

        std::vector<float> pooled;
        if (past_count != 0) {
            pool_average(scores, 0, past_count, config.pooling_kernel, pooled);
        }
        std::vector<std::size_t> candidates(past_count);
        for (std::size_t index = 0; index < past_count; ++index) {
            candidates[index] = index;
        }
        std::sort(candidates.begin(), candidates.end(),
            [&pooled, &positions](std::size_t left, std::size_t right) {
                if (pooled[left] != pooled[right]) {
                    return pooled[left] > pooled[right];
                }
                return positions[left] < positions[right];
            });

        auto & selected = output.heads[head];
        selected.keep_cells.reserve(older_to_keep + valid_tokens - past_count);
        std::vector<std::size_t> kept_indices;
        kept_indices.reserve(older_to_keep + valid_tokens - past_count);
        for (std::size_t i = 0; i < older_to_keep; ++i) {
            kept_indices.push_back(candidates[i]);
        }
        for (std::size_t index = past_count; index < valid_tokens; ++index) {
            kept_indices.push_back(index);
        }
        std::sort(kept_indices.begin(), kept_indices.end(),
            [&cells](size_t a, size_t b) { return cells[a] < cells[b]; });
        for (const size_t index : kept_indices) {
            selected.keep_cells.push_back(cells[index]);
            selected.keep_positions.push_back(positions[index]);
        }
        for (std::size_t index = recent_begin; index < valid_tokens; ++index) {
            selected.protected_recent_cells.push_back(cells[index]);
            selected.protected_recent_positions.push_back(positions[index]);
        }
        output.would_compact |= selected.keep_cells.size() < valid_tokens;
        if (selected.keep_cells.empty()) {
            error = "PyramidKV selection retained no rows for a KV head";
            return false;
        }
    }

    return true;
}
