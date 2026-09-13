#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

// CPU-only, lazy reorder for buffers with one row per original batch input.
// This permutation is independent of the subset selected for logits output.
class llama_row_reorder {
public:
    struct buffer {
        float * data;
        size_t size;  // capacity in floats
        size_t width; // floats per row
    };

    void clear() {
        n_rows = 0;
        swaps.clear();
    }

    bool pending() const {
        return n_rows != 0;
    }

    bool identity() const {
        return swaps.empty();
    }

    bool prepare(const std::vector<int32_t> & input_ids, size_t expected_rows) {
        clear();
        if (input_ids.size() != expected_rows) {
            return false;
        }

        std::vector<bool> seen(expected_rows, false);
        for (const int32_t id : input_ids) {
            if (id < 0 || (size_t) id >= expected_rows || seen[id]) {
                return false;
            }
            seen[id] = true;
        }

        // Each swap puts at least one row in its final position: O(N) swaps.
        std::vector<int32_t> order(input_ids);
        for (size_t i = 0; i < order.size(); ++i) {
            while ((size_t) order[i] != i) {
                const size_t j = (size_t) order[i];
                std::swap(order[i], order[j]);
                swaps.emplace_back(i, j);
            }
        }
        n_rows = expected_rows;
        return true;
    }

    // Validate every buffer before touching any row. Consume the permutation
    // only after all buffers have been reordered, so repeated getters are safe.
    bool apply(const std::vector<buffer> & buffers) {
        if (n_rows == 0) {
            return true;
        }
        for (const auto & buf : buffers) {
            if (!buf.data || buf.width == 0 || n_rows > buf.size / buf.width) {
                return false;
            }
        }
        for (const auto & swap : swaps) {
            for (const auto & buf : buffers) {
                float * row0 = buf.data + swap.first  * buf.width;
                float * row1 = buf.data + swap.second * buf.width;
                std::swap_ranges(row0, row0 + buf.width, row1);
            }
        }
        clear();
        return true;
    }

private:
    size_t n_rows = 0;
    std::vector<std::pair<size_t, size_t>> swaps;
};
