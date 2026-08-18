#pragma once

#include "llama-batch.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

// Host-side description of the compact DSpark noise rows carried by one
// logical batch/ubatch. Rows stay in caller/source order. indptr therefore
// maps sequence run s to [indptr[s], indptr[s + 1]) without a gather or a
// permutation in the model graph.
struct llama_dspark_packed_layout {
    struct group {
        uint32_t row_begin = 0;
        uint32_t width     = 0;
        uint32_t n_seqs    = 0;
    };

    bool valid = false;

    std::vector<llama_seq_id> seq_ids;
    std::vector<uint32_t>     indptr;

    // Consecutive source-order runs with the same width can retain the old
    // vectorized Markov-head implementation. A width change starts a group.
    std::vector<group> groups;

    uint32_t max_width = 0;
    std::string reason;
};

inline bool llama_dspark_parse_gamma(const std::string & value, uint32_t & result) {
    try {
        size_t parsed = 0;
        const unsigned long long gamma = std::stoull(value, &parsed, 10);
        if (parsed != value.size() || gamma == 0 ||
                gamma > (unsigned long long) std::numeric_limits<int32_t>::max()) {
            return false;
        }
        result = (uint32_t) gamma;
        return true;
    } catch (const std::exception &) {
        return false;
    }
}

// Valid synthetic shapes used to reserve DSpark graph/Fit buffers. Shape zero
// maximizes dense rows; shape one (when distinct) maximizes adjacent width
// transitions while filling the same total-row tier whenever sequence slots
// permit it.
inline std::vector<std::vector<uint32_t>> llama_dspark_reserve_widths(
        uint32_t gamma,
        uint32_t n_ubatch,
        uint32_t n_seq_max) {
    std::vector<std::vector<uint32_t>> result;
    if (gamma == 0 || gamma > n_ubatch || n_seq_max == 0) {
        return result;
    }

    const uint32_t n_dense = std::min(n_seq_max, n_ubatch / gamma);
    result.emplace_back(n_dense, gamma);

    if (gamma == 1) {
        return result;
    }

    const uint32_t target_rows = (uint32_t) std::min<uint64_t>(
            n_ubatch, (uint64_t) gamma * n_seq_max);

    struct search_state {
        uint32_t pos;
        uint32_t remaining;
        uint32_t last;

        bool operator==(const search_state & other) const {
            return pos == other.pos && remaining == other.remaining && last == other.last;
        }
    };
    struct search_hash {
        size_t operator()(const search_state & state) const {
            size_t h = std::hash<uint32_t>{}(state.pos);
            h ^= std::hash<uint32_t>{}(state.remaining) + 0x9e3779b9u + (h << 6) + (h >> 2);
            h ^= std::hash<uint32_t>{}(state.last)      + 0x9e3779b9u + (h << 6) + (h >> 2);
            return h;
        }
    };

    // Exact max-transition composition. Bounds use the minimum 1/2 zig-zag
    // and maximum gamma/(gamma-1) zig-zag; DFS only explores values whose tail
    // can still reach the exact selected row tier (full budget when feasible,
    // otherwise the slot-limited maximum for that group count).
    for (uint32_t n_groups = std::min(n_seq_max, target_rows); n_groups >= 2; --n_groups) {
        const uint64_t min_sum = (uint64_t) n_groups + n_groups / 2;
        const uint64_t max_sum = (uint64_t) n_groups * gamma - n_groups / 2;
        const uint32_t mixed_target = (uint32_t) std::min<uint64_t>(target_rows, max_sum);
        if (mixed_target < min_sum) {
            continue;
        }

        std::vector<uint32_t> mixed;
        mixed.reserve(n_groups);
        std::unordered_set<search_state, search_hash> failed;

        const auto min_tail = [&](uint32_t count, uint32_t previous) -> uint64_t {
            return (uint64_t) count + (previous == 1 ? (count + 1) / 2 : count / 2);
        };
        const auto max_tail = [&](uint32_t count, uint32_t previous) -> uint64_t {
            return (uint64_t) count * gamma - (previous == gamma ? (count + 1) / 2 : count / 2);
        };

        std::function<bool(uint32_t, uint32_t, uint32_t)> build =
                [&](uint32_t pos, uint32_t remaining, uint32_t last) -> bool {
            if (pos == n_groups) {
                return remaining == 0;
            }
            const search_state state { pos, remaining, last };
            if (failed.count(state)) {
                return false;
            }

            const uint32_t tail_count = n_groups - pos - 1;
            const uint32_t max_width = std::min(gamma, remaining);
            for (uint32_t width = max_width; width >= 1; --width) {
                if (width == last || width > remaining) {
                    continue;
                }
                const uint32_t tail_rows = remaining - width;
                if (tail_rows < min_tail(tail_count, width) ||
                        tail_rows > max_tail(tail_count, width)) {
                    continue;
                }
                mixed.push_back(width);
                if (build(pos + 1, tail_rows, width)) {
                    return true;
                }
                mixed.pop_back();
            }

            failed.insert(state);
            return false;
        };

        if (build(0, mixed_target, 0) && mixed != result.front()) {
            result.push_back(std::move(mixed));
            break;
        }
    }

    return result;
}

inline llama_dspark_packed_layout llama_dspark_packed_layout_from_rows(
        uint32_t              n_tokens,
        const int32_t       * n_seq_id,
        llama_seq_id * const* seq_id,
        uint32_t              trained_gamma) {
    llama_dspark_packed_layout result;

    if (n_tokens == 0) {
        result.reason = "empty DSpark row set";
        return result;
    }
    if (trained_gamma == 0) {
        result.reason = "trained DSpark gamma must be positive";
        return result;
    }
    if (!n_seq_id || !seq_id) {
        result.reason = "missing DSpark sequence metadata";
        return result;
    }

    std::unordered_set<llama_seq_id> completed;
    result.indptr.push_back(0);

    llama_seq_id current = -1;
    uint32_t run_begin = 0;

    auto finish_run = [&](uint32_t row_end) -> bool {
        const uint32_t width = row_end - run_begin;
        if (width == 0 || width > trained_gamma) {
            result.reason = "DSpark proposal width is outside trained gamma";
            return false;
        }

        result.seq_ids.push_back(current);
        result.indptr.push_back(row_end);
        result.max_width = std::max(result.max_width, width);

        if (!result.groups.empty() && result.groups.back().width == width) {
            result.groups.back().n_seqs++;
        } else {
            result.groups.push_back({ run_begin, width, 1 });
        }
        return true;
    };

    for (uint32_t row = 0; row < n_tokens; ++row) {
        if (n_seq_id[row] != 1 || seq_id[row] == nullptr || seq_id[row][0] < 0) {
            result.reason = "DSpark rows must belong to exactly one valid sequence";
            return result;
        }

        const llama_seq_id next = seq_id[row][0];
        if (row == 0) {
            current = next;
            continue;
        }
        if (next == current) {
            continue;
        }

        if (!finish_run(row)) {
            return result;
        }
        completed.insert(current);
        if (completed.count(next) != 0) {
            result.reason = "DSpark sequence rows are not contiguous";
            return result;
        }

        current   = next;
        run_begin = row;
    }

    if (!finish_run(n_tokens)) {
        return result;
    }

    result.valid = true;
    return result;
}

inline llama_dspark_packed_layout llama_dspark_packed_layout_from_batch(
        const llama_batch & batch,
        uint32_t            trained_gamma) {
    if (batch.n_tokens < 0) {
        llama_dspark_packed_layout result;
        result.reason = "negative DSpark row count";
        return result;
    }
    return llama_dspark_packed_layout_from_rows(
            (uint32_t) batch.n_tokens, batch.n_seq_id, batch.seq_id, trained_gamma);
}

inline llama_dspark_packed_layout llama_dspark_packed_layout_from_ubatch(
        const llama_ubatch & ubatch,
        uint32_t             trained_gamma) {
    auto result = llama_dspark_packed_layout_from_rows(
            ubatch.n_tokens, ubatch.n_seq_id, ubatch.seq_id, trained_gamma);
    if (!result.valid) {
        return result;
    }

    if (result.seq_ids.size() != ubatch.n_seqs_unq) {
        result.valid  = false;
        result.reason = "DSpark unique-sequence count does not match packed runs";
        return result;
    }
    return result;
}
