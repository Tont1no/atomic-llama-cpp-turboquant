#ifndef LLAMA_SPECULATIVE_RUNTIME_POLICY_H
#define LLAMA_SPECULATIVE_RUNTIME_POLICY_H

#include <cstdint>
#include <array>
#include <cstddef>
#include <vector>

namespace common_spec_runtime {

enum class action {
    target,
    lookup,
    model,
};

enum class model_drafter {
    none,
    mtp,
    dflash,
    dflash2,
    dspark,
};

enum class model_state {
    unavailable,
    probing,
    active,
    cooldown,
};

// Runtime load is grouped into stable powers-of-two classes.  The class is
// metadata only; no request text or prompt-derived feature is retained.
enum class load_bucket : uint8_t {
    c1 = 0,
    c2,
    c4,
    c8_plus,
};

enum class reason {
    target_default,
    target_bootstrap,
    target_refresh,
    invalid_config,
    invalid_context,
    invalid_observation,
    model_probe,
    model_profitable,
    load_cap,
    soft_vram_cap,
    hard_vram_guard,
    lookup_fallback,
    same_request_lockout,
    cooldown_active,
    model_state_stale,
    pending_feedback,
};

struct config {
    model_drafter drafter = model_drafter::none;
    std::vector<int32_t> depth_arms = { 0 };

    bool lookup_enabled = false;
    int32_t lookup_depth = 0;
    bool bootstrap_target_baseline = false;

    // Entry n contains the maximum model depth at total load n + 1. The last
    // entry is reused at higher load. A value between arms rounds down.
    std::vector<int32_t> load_depth_caps = { 0 };

    uint64_t hard_vram_free_bytes = 0;
    uint64_t soft_vram_free_bytes = 0;
    int32_t soft_vram_max_depth = 1;
    int64_t max_vram_sample_age_us = 10 * 1000 * 1000;

    double promote_gain = 0.05;
    double demote_loss = 0.0;
    double ewma_alpha = 0.25;
    // Force one fresh target-only group after this many clean non-target
    // observations in the same bucket. Zero disables periodic refresh.
    uint64_t target_remeasure_interval = 32;
    int32_t promote_samples = 6;
    int32_t demote_samples = 2;
    int32_t probe_depth = 1;
    uint64_t cooldown_requests = 2;
    uint64_t min_dwell_requests = 8;
};

struct observation {
    uint64_t decision_id = 0;
    uint64_t request_epoch = 0;
    uint64_t round_epoch = 0;
    action observed_action = action::target;
    model_drafter observed_drafter = model_drafter::none;
    int32_t observed_depth = 0;

    int64_t elapsed_us = 0;
    int32_t useful_tokens = 0;
    double target_tokens_per_us = 0.0;

    bool synchronized = false;
    bool complete = false;
};

struct runtime_context {
    uint64_t request_epoch = 0;
    int32_t active_slots = 0;
    int32_t queued_slots = 0;
    int32_t context_tokens = 0;
    // Optional whole-group cap derived before issuing a shared decision.
    // Negative means that no additional cap is applied.
    int32_t max_model_depth = -1;
    // Periodic baseline refresh is deferred until the server can time one
    // complete, non-overlapping group. Initial bootstrap remains fail-closed.
    // The server sets this only at a clean group-wide request boundary.
    // Refreshing mid-request would desynchronize the model drafter for the
    // remainder of every participating request.
    bool allow_target_remeasure = true;

    uint64_t vram_free_bytes = 0;
    uint64_t vram_total_bytes = 0;
    int64_t vram_sample_age_us = 0;

    // Model drafting can only resume when the draft state matches the target.
    // The current server guarantees that at a new request boundary, not after
    // model maintenance was disabled in the middle of a request.
    bool model_state_synchronized = false;
};

struct decision {
    uint64_t decision_id = 0;
    action selected_action = action::target;
    model_drafter drafter = model_drafter::none;
    int32_t depth = 0;
    reason why = reason::target_default;
    model_state state = model_state::unavailable;
    uint64_t next_eligible_request_epoch = 0;
    double model_speedup_ewma = 0.0;
    bool keep_model_state_synchronized = false;
    load_bucket load_class = load_bucket::c1;
};

class policy {
public:
    // One instance owns one sequential server-wide decision stream.  A
    // decision applies to the complete participating slot group and must be
    // followed by exactly one aggregate observation or discard.
    explicit policy(config cfg);

    static bool valid_config(const config & cfg);
    static bool valid_observation(const observation & obs, const config & cfg);

    bool observe(const observation & obs);
    bool discard(uint64_t decision_id, uint64_t request_epoch);
    decision decide(const runtime_context & ctx);
    void reset();

    const config & get_config() const;
    model_state get_model_state() const;
    model_state get_model_state(load_bucket bucket) const;
    double get_model_speedup_ewma() const;
    double get_model_speedup_ewma(load_bucket bucket) const;
    int32_t get_active_depth() const;
    uint64_t get_next_eligible_request_epoch() const;
    double get_target_tokens_per_us() const;
    double get_target_tokens_per_us(load_bucket bucket) const;
    uint64_t get_model_observations(load_bucket bucket) const;
    int32_t get_bucket_load_cap(load_bucket bucket) const;

    static load_bucket classify_load(int64_t total_load);
    static bool valid_produced_depth(action selected_action, int32_t produced_depth, int32_t offered_depth);

private:
    int32_t clamp_to_arm(int32_t cap) const;
    int32_t load_cap(int64_t total_load) const;
    int32_t bucket_load_cap(load_bucket bucket) const;
    int32_t probe_arm(load_bucket bucket) const;
    int32_t next_arm(int32_t depth) const;
    int32_t previous_arm(int32_t depth) const;
    struct bucket_state {
        model_state state = model_state::unavailable;
        uint64_t last_transition_request_epoch = 0;
        uint64_t next_eligible_request_epoch = 0;
        bool have_transition_anchor = false;
        int32_t consecutive_wins = 0;
        int32_t consecutive_losses = 0;
        bool have_speedup = false;
        double model_speedup_ewma = 0.0;
        double target_tokens_per_us = 0.0;
        uint64_t model_observations = 0;
        uint64_t observations_since_target = 0;
        int32_t qualified_depth = 0;
        int32_t probe_target_depth = 0;
    };

    static size_t bucket_index(load_bucket bucket);
    bucket_state & state_for(load_bucket bucket);
    const bucket_state & state_for(load_bucket bucket) const;
    decision issue(decision value, uint64_t request_epoch);
    decision fallback(reason why, uint64_t request_epoch, load_bucket bucket,
            bool stop_model_maintenance = false);
    void demote(load_bucket bucket, uint64_t request_epoch, reason why);
    void advance_request(load_bucket bucket, uint64_t request_epoch);

    config cfg_;
    bool config_valid_ = false;
    std::array<bucket_state, 4> bucket_states_ {};
    load_bucket selected_bucket_ = load_bucket::c1;

    uint64_t last_request_epoch_ = 0;
    uint64_t last_round_epoch_ = 0;
    bool have_epoch_ = false;
    bool locked_request_ = false;
    uint64_t locked_request_epoch_ = 0;

    int32_t active_depth_ = 0;
    uint64_t next_decision_id_ = 1;
    bool pending_decision_ = false;
    decision pending_;
    uint64_t pending_request_epoch_ = 0;
    uint64_t discarded_decision_id_ = 0;
    uint64_t discarded_request_epoch_ = 0;
    reason last_fault_reason_ = reason::target_default;
};

const char * to_string(action value);
const char * to_string(model_drafter value);
const char * to_string(model_state value);
const char * to_string(reason value);
const char * to_string(load_bucket value);

} // namespace common_spec_runtime

#endif // LLAMA_SPECULATIVE_RUNTIME_POLICY_H
