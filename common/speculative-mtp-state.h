#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

// begin() is called before prefill by native integrations, but after prefill
// by llama-server. Preserve the latter's freshly produced carryover.
inline bool common_speculative_mtp_reset_at_begin(bool empty_prompt, int64_t draft_pos_max) {
    return empty_prompt || draft_pos_max < 0;
}

// A fresh sequence has no previous hidden row to pair with its first token.
// Reset only this sequence's carryover; other active sequences remain intact.
inline void common_speculative_mtp_reset_carry(
        std::vector<float> & pending,
        std::vector<float> & verified,
        int32_t & verified_rows,
        std::vector<float> * chain = nullptr) {
    std::fill(pending.begin(), pending.end(), 0.0f);
    verified.clear();
    verified_rows = 0;
    if (chain) {
        chain->clear();
    }
}
