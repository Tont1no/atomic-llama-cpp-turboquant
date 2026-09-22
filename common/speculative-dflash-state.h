#pragma once

#include <algorithm>
#include <cstdint>

// Opt-in only for a validated all-STANDARD-SWA drafter. Its ISWA base cache has
// no tensors, but still accumulates cell metadata unless old positions are
// removed. Use the first incoming position, never the last: rejected verify
// rows may roll back. is_masked_swa(STANDARD) masks p0 when p1 - p0 >= window.
template <typename RemovePrefix>
inline bool common_speculative_dflash_prune_history(
        int32_t seq_id, uint32_t n_seq, uint32_t window,
        const int32_t * positions, int32_t count, RemovePrefix remove_prefix) {
    if (window == 0) {
        return true;
    }
    if (seq_id < 0 || static_cast<uint32_t>(seq_id) >= n_seq || !positions || count <= 0) {
        return false;
    }
    const int32_t first = *std::min_element(positions, positions + count);
    if (first < 0) {
        return false;
    }
    const int64_t keep_from = static_cast<int64_t>(first) - window + 1;
    // seq_rm's upper bound is exclusive. Never remove current/future positions,
    // another sequence's cells, or positions still inside this sequence's SWA.
    return keep_from <= 0 || remove_prefix(seq_id, 0, static_cast<int32_t>(keep_from));
}
