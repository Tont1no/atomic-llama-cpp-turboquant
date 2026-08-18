#pragma once

#include "llama.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

// One measured server-step cost. context_tokens is the largest active
// generation context in the tick. total_verify_rows includes the mandatory
// anchor row for every active generation slot.
struct common_speculative_sps_entry {
    int32_t context_tokens    = 0;
    int32_t active_slots      = 0;
    int32_t total_verify_rows = 0;
    double  cost_us           = 0.0;
};

struct common_speculative_sps_profile {
    std::vector<common_speculative_sps_entry> entries;

    // Conservative ceiling lookup. Returns no value when the measured table
    // does not cover the requested operating point.
    std::optional<double> lookup_cost_us(
            int32_t context_tokens,
            int32_t active_slots,
            int32_t total_verify_rows) const;
};

// Both functions validate the complete document and throw std::runtime_error
// on malformed, duplicate, non-finite, or non-monotonic input.
common_speculative_sps_profile common_speculative_sps_profile_parse(
        const std::string & contents,
        const std::string & source = "<memory>");

common_speculative_sps_profile common_speculative_sps_profile_load(const std::string & path);

struct common_speculative_sps_slot {
    llama_seq_id seq_id = -1;

    // Prefix survival P(accepted length > position). Values must be finite,
    // within [0, 1], and monotonically non-increasing.
    std::vector<float> survival;

    // Zero is always a valid target-only choice. Positive prefixes must be at
    // least min_prefix and at most max_prefix.
    int32_t min_prefix = 0;
    int32_t max_prefix = 0;
};

struct common_speculative_sps_plan {
    bool valid = false;

    std::vector<int32_t> prefixes;

    int32_t total_verify_rows = 0;
    double expected_useful_tokens = 0.0;
    double predicted_cost_us      = 0.0;
    double expected_tokens_per_us = 0.0;

    std::string reason;
};

// Select one prefix per slot. base_verify_rows contains all unavoidable rows
// for the tick (one anchor per active slot plus any fixed replay rows).
// base_useful_tokens is the corresponding expected output count. The planner
// maximizes expected useful tokens / measured SPS cost globally, with stable
// tie-breaking toward more useful tokens and then fewer verify rows.
common_speculative_sps_plan common_speculative_sps_plan_prefixes(
        const common_speculative_sps_profile & profile,
        int32_t context_tokens,
        int32_t active_slots,
        int32_t base_verify_rows,
        double base_useful_tokens,
        const std::vector<common_speculative_sps_slot> & slots);
