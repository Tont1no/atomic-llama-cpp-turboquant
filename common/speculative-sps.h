#pragma once

#include "llama.h"

#include <cstdint>
#include <map>
#include <memory>
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
    int32_t schema_version = 1;
    // Schema v2 profiles have an exact-active-slot domain whose row axis is
    // bounded independently for each active-slot count. Zero for v1.
    int32_t max_draft_tokens_per_slot = 0;
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

enum common_speculative_sps_execution_kind {
    COMMON_SPECULATIVE_SPS_EXECUTION_UNKNOWN,
    COMMON_SPECULATIVE_SPS_EXECUTION_DIRECT_WARMUP,
    COMMON_SPECULATIVE_SPS_EXECUTION_GRAPH_CAPTURE,
    COMMON_SPECULATIVE_SPS_EXECUTION_GRAPH_REPLAY,
    COMMON_SPECULATIVE_SPS_EXECUTION_GRAPHS_DISABLED,
};

enum common_speculative_sps_record_result {
    COMMON_SPECULATIVE_SPS_RECORD_RETAINED,
    COMMON_SPECULATIVE_SPS_RECORD_COORDINATE_READY,
    COMMON_SPECULATIVE_SPS_RECORD_ALL_READY,
    COMMON_SPECULATIVE_SPS_RECORD_SKIPPED_WARMUP,
    COMMON_SPECULATIVE_SPS_RECORD_SKIPPED_CAPTURE,
    COMMON_SPECULATIVE_SPS_RECORD_SKIPPED_INELIGIBLE,
    COMMON_SPECULATIVE_SPS_RECORD_SKIPPED_OUTSIDE_GRID,
    COMMON_SPECULATIVE_SPS_RECORD_SKIPPED_FULL,
};

struct common_speculative_sps_profile_recorder_config {
    // Immutable runtime fingerprint (models, build, backend, GPU and relevant
    // server args). Resume is rejected when this differs.
    std::string source_identity;

    // Actual contexts are assigned to the smallest configured ceiling that is
    // greater than or equal to them.
    std::vector<int32_t> context_buckets;

    // Exact-active-slot row axes. Each axis is independent in schema v2 and
    // must stay within [active, active * (1 + max_draft_tokens_per_slot)].
    std::map<int32_t, std::vector<int32_t>> verify_rows_by_active;

    int32_t  max_draft_tokens_per_slot = 0;
    uint32_t retained_samples_per_coordinate = 1;

    // Otherwise eligible non-capture executions excluded at every coordinate.
    // Explicit capture/direct-warmup kinds are always excluded and do not
    // consume this stable-execution allowance.
    uint32_t warmup_samples_per_coordinate = 0;

    // Quantile used for each coordinate before applying a conservative
    // monotonic upper envelope. Must be within (0, 1].
    double aggregate_quantile = 0.95;

    // Optional audit sidecar from an earlier process. Loading is strict and
    // fail-closed: schema, full configuration, counters, coordinates, and
    // retained samples must all match and validate.
    std::string resume_sidecar_path;
};

struct common_speculative_sps_profile_sample {
    int32_t actual_context_tokens = 0;
    int32_t active_slots          = 0;
    int32_t total_verify_rows     = 0;
    double  cost_us               = 0.0;

    bool decode_succeeded = false;
    bool synchronized     = false;
    bool pure_generation  = false;
    bool whole_batch      = false;

    common_speculative_sps_execution_kind execution_kind = COMMON_SPECULATIVE_SPS_EXECUTION_UNKNOWN;

    // Executed draft prefix per active slot, in stable server slot order.
    std::vector<int32_t> prefixes;
};

// Deterministic, backend-independent recorder core. The server supplies an
// already synchronized elapsed cost, so tests can use fake timestamps without
// a model, GPU, or injectable global clock.
class common_speculative_sps_profile_recorder {
public:
    explicit common_speculative_sps_profile_recorder(
            common_speculative_sps_profile_recorder_config config);
    ~common_speculative_sps_profile_recorder();

    common_speculative_sps_profile_recorder(common_speculative_sps_profile_recorder &&) noexcept;
    common_speculative_sps_profile_recorder & operator=(common_speculative_sps_profile_recorder &&) noexcept;
    common_speculative_sps_profile_recorder(const common_speculative_sps_profile_recorder &) = delete;
    common_speculative_sps_profile_recorder & operator=(const common_speculative_sps_profile_recorder &) = delete;

    common_speculative_sps_record_result observe(
            const common_speculative_sps_profile_sample & sample);

    bool ready() const;
    size_t expected_coordinates() const;
    size_t ready_coordinates() const;
    uint64_t retained_samples() const;
    uint64_t skipped_warmup() const;
    uint64_t skipped_capture() const;
    uint64_t skipped_ineligible() const;
    uint64_t skipped_outside_grid() const;
    uint64_t skipped_full() const;

    // Throws until every configured coordinate has enough retained samples.
    common_speculative_sps_profile profile() const;

    // The sidecar is always replaced atomically. The planner-compatible
    // primary profile is replaced only after all coordinates are sample-ready.
    void write_atomic(
            const std::string & profile_path,
            const std::string & sidecar_path) const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

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
