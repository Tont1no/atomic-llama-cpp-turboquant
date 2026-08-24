#include "speculative-runtime-policy.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace common_spec_runtime {

namespace {

bool valid_action(action value) {
    return value == action::target || value == action::lookup || value == action::model;
}

bool valid_drafter(model_drafter value) {
    return value == model_drafter::none || value == model_drafter::mtp ||
           value == model_drafter::dflash || value == model_drafter::dflash2 ||
           value == model_drafter::dspark;
}

uint64_t saturating_add(uint64_t left, uint64_t right) {
    if (right > std::numeric_limits<uint64_t>::max() - left) {
        return std::numeric_limits<uint64_t>::max();
    }
    return left + right;
}

} // namespace

policy::policy(config cfg) : cfg_(std::move(cfg)) {
    reset();
}

bool policy::valid_config(const config & cfg) {
    if (!valid_drafter(cfg.drafter) || cfg.depth_arms.empty() || cfg.depth_arms.front() != 0) {
        return false;
    }
    for (size_t i = 1; i < cfg.depth_arms.size(); ++i) {
        if (cfg.depth_arms[i] <= cfg.depth_arms[i - 1]) {
            return false;
        }
    }
    if (cfg.drafter == model_drafter::none &&
            (cfg.depth_arms.size() != 1 || cfg.depth_arms.front() != 0)) {
        return false;
    }
    if (cfg.drafter != model_drafter::none &&
            (cfg.depth_arms.size() < 2 || cfg.depth_arms.back() <= 0)) {
        return false;
    }
    if (cfg.load_depth_caps.empty()) {
        return false;
    }
    for (int32_t cap : cfg.load_depth_caps) {
        if (cap < 0) {
            return false;
        }
    }
    if ((cfg.lookup_enabled && cfg.lookup_depth <= 0) || cfg.soft_vram_max_depth < 0 ||
            cfg.probe_depth < 0 || cfg.promote_samples <= 0 || cfg.demote_samples <= 0) {
        return false;
    }
    if (!std::isfinite(cfg.promote_gain) || cfg.promote_gain < 0.0 ||
            !std::isfinite(cfg.demote_loss) || cfg.demote_loss < 0.0 || cfg.demote_loss >= 1.0 ||
            !std::isfinite(cfg.ewma_alpha) || cfg.ewma_alpha <= 0.0 || cfg.ewma_alpha > 1.0) {
        return false;
    }
    if (cfg.hard_vram_free_bytes > 0 && cfg.soft_vram_free_bytes > 0 &&
            cfg.hard_vram_free_bytes > cfg.soft_vram_free_bytes) {
        return false;
    }
    if ((cfg.hard_vram_free_bytes > 0 || cfg.soft_vram_free_bytes > 0) &&
            cfg.max_vram_sample_age_us <= 0) {
        return false;
    }
    return true;
}

bool policy::valid_observation(const observation & obs, const config & cfg) {
    if (!valid_action(obs.observed_action) || !valid_drafter(obs.observed_drafter) ||
            obs.decision_id == 0 || obs.round_epoch == 0 || !obs.synchronized || !obs.complete ||
            obs.elapsed_us <= 0 || obs.useful_tokens <= 0 ||
            !std::isfinite(obs.target_tokens_per_us) || obs.target_tokens_per_us < 0.0) {
        return false;
    }
    if (obs.observed_action == action::model) {
        if (cfg.drafter == model_drafter::none || obs.observed_drafter != cfg.drafter ||
                obs.observed_depth <= 0 ||
                std::find(cfg.depth_arms.begin(), cfg.depth_arms.end(), obs.observed_depth) == cfg.depth_arms.end()) {
            return false;
        }
    } else if (obs.observed_action == action::lookup) {
        if (obs.observed_drafter != model_drafter::none || !cfg.lookup_enabled ||
                obs.observed_depth != cfg.lookup_depth) {
            return false;
        }
    } else if (obs.observed_drafter != model_drafter::none || obs.observed_depth != 0) {
        return false;
    }
    return true;
}

void policy::reset() {
    config_valid_ = valid_config(cfg_);
    selected_bucket_ = load_bucket::c1;
    for (size_t i = 0; i < bucket_states_.size(); ++i) {
        auto & bucket = bucket_states_[i];
        bucket = {};
        bucket.state = config_valid_ && cfg_.drafter != model_drafter::none
                ? model_state::probing : model_state::unavailable;
        bucket.probe_target_depth = config_valid_
                ? probe_arm(static_cast<load_bucket>(i)) : 0;
    }
    last_request_epoch_ = 0;
    last_round_epoch_ = 0;
    have_epoch_ = false;
    locked_request_ = false;
    locked_request_epoch_ = 0;
    active_depth_ = 0;
    next_decision_id_ = 1;
    pending_decision_ = false;
    pending_ = {};
    pending_request_epoch_ = 0;
    discarded_decision_id_ = 0;
    discarded_request_epoch_ = 0;
    last_fault_reason_ = reason::target_default;
}

const config & policy::get_config() const {
    return cfg_;
}

model_state policy::get_model_state() const {
    return get_model_state(selected_bucket_);
}

model_state policy::get_model_state(load_bucket bucket) const {
    return state_for(bucket).state;
}

double policy::get_model_speedup_ewma() const {
    return get_model_speedup_ewma(selected_bucket_);
}

double policy::get_model_speedup_ewma(load_bucket bucket) const {
    return state_for(bucket).model_speedup_ewma;
}

int32_t policy::get_active_depth() const {
    return active_depth_;
}

uint64_t policy::get_next_eligible_request_epoch() const {
    return state_for(selected_bucket_).next_eligible_request_epoch;
}

double policy::get_target_tokens_per_us() const {
    return get_target_tokens_per_us(selected_bucket_);
}

double policy::get_target_tokens_per_us(load_bucket bucket) const {
    return state_for(bucket).target_tokens_per_us;
}

uint64_t policy::get_model_observations(load_bucket bucket) const {
    return state_for(bucket).model_observations;
}

int32_t policy::get_bucket_load_cap(load_bucket bucket) const {
    return bucket_load_cap(bucket);
}

load_bucket policy::classify_load(int64_t total_load) {
    if (total_load <= 1) {
        return load_bucket::c1;
    }
    if (total_load <= 2) {
        return load_bucket::c2;
    }
    if (total_load <= 4) {
        return load_bucket::c4;
    }
    return load_bucket::c8_plus;
}

bool policy::valid_produced_depth(action selected_action, int32_t produced_depth, int32_t offered_depth) {
    if (selected_action == action::target) {
        return produced_depth == 0 && offered_depth == 0;
    }
    return offered_depth > 0 && produced_depth > 0 && produced_depth <= offered_depth;
}

size_t policy::bucket_index(load_bucket bucket) {
    return static_cast<size_t>(bucket);
}

policy::bucket_state & policy::state_for(load_bucket bucket) {
    return bucket_states_[bucket_index(bucket)];
}

const policy::bucket_state & policy::state_for(load_bucket bucket) const {
    return bucket_states_[bucket_index(bucket)];
}

int32_t policy::clamp_to_arm(int32_t cap) const {
    int32_t result = 0;
    for (int32_t arm : cfg_.depth_arms) {
        if (arm > cap) {
            break;
        }
        result = arm;
    }
    return result;
}

int32_t policy::load_cap(int64_t total_load) const {
    if (total_load <= 0 || cfg_.load_depth_caps.empty()) {
        return 0;
    }
    const uint64_t index = std::min<uint64_t>(
            static_cast<uint64_t>(total_load - 1), cfg_.load_depth_caps.size() - 1);
    return cfg_.load_depth_caps[static_cast<size_t>(index)];
}

int32_t policy::bucket_load_cap(load_bucket bucket) const {
    int64_t first_load = 1;
    int64_t last_load = 1;
    switch (bucket) {
        case load_bucket::c1:
            break;
        case load_bucket::c2:
            first_load = last_load = 2;
            break;
        case load_bucket::c4:
            first_load = 3;
            last_load = 4;
            break;
        case load_bucket::c8_plus:
            first_load = 5;
            last_load = std::max<int64_t>(5, (int64_t) cfg_.load_depth_caps.size());
            break;
    }
    int32_t cap = std::numeric_limits<int32_t>::max();
    for (int64_t load = first_load; load <= last_load; ++load) {
        cap = std::min(cap, load_cap(load));
    }
    return cap == std::numeric_limits<int32_t>::max() ? 0 : cap;
}

int32_t policy::probe_arm(load_bucket bucket) const {
    if (cfg_.drafter == model_drafter::none || cfg_.depth_arms.size() < 2) {
        return 0;
    }
    const int32_t requested = cfg_.probe_depth > 0 ? cfg_.probe_depth : cfg_.depth_arms[1];
    return clamp_to_arm(std::min(requested, bucket_load_cap(bucket)));
}

int32_t policy::next_arm(int32_t depth) const {
    for (int32_t arm : cfg_.depth_arms) {
        if (arm > depth) {
            return arm;
        }
    }
    return depth;
}

int32_t policy::previous_arm(int32_t depth) const {
    int32_t previous = 0;
    for (int32_t arm : cfg_.depth_arms) {
        if (arm >= depth) {
            break;
        }
        previous = arm;
    }
    return previous;
}

decision policy::issue(decision value, uint64_t request_epoch) {
    value.decision_id = next_decision_id_;
    next_decision_id_ = next_decision_id_ == std::numeric_limits<uint64_t>::max() ? 1 : next_decision_id_ + 1;
    pending_decision_ = true;
    pending_ = value;
    pending_request_epoch_ = request_epoch;
    return value;
}

decision policy::fallback(reason why, uint64_t request_epoch, load_bucket bucket, bool stop_model_maintenance) {
    selected_bucket_ = bucket;
    const auto & runtime = state_for(bucket);
    active_depth_ = 0;
    if (stop_model_maintenance && cfg_.drafter != model_drafter::none) {
        locked_request_ = true;
        locked_request_epoch_ = request_epoch;
    }
    decision result;
    result.why = why;
    result.state = runtime.state;
    result.next_eligible_request_epoch = runtime.next_eligible_request_epoch;
    result.model_speedup_ewma = runtime.model_speedup_ewma;
    result.load_class = bucket;
    result.keep_model_state_synchronized = cfg_.drafter != model_drafter::none && !stop_model_maintenance;
    if (cfg_.lookup_enabled && why != reason::invalid_config && why != reason::invalid_context &&
            why != reason::hard_vram_guard && why != reason::target_bootstrap &&
            why != reason::target_refresh) {
        result.selected_action = action::lookup;
        result.depth = cfg_.lookup_depth;
    }
    return issue(result, request_epoch);
}

void policy::demote(load_bucket bucket, uint64_t request_epoch, reason why) {
    auto & runtime = state_for(bucket);
    runtime.state = model_state::cooldown;
    locked_request_ = true;
    locked_request_epoch_ = request_epoch;
    runtime.last_transition_request_epoch = request_epoch;
    runtime.next_eligible_request_epoch = saturating_add(
            request_epoch, std::max(cfg_.cooldown_requests, cfg_.min_dwell_requests));
    runtime.consecutive_wins = 0;
    runtime.consecutive_losses = 0;
    active_depth_ = 0;
    pending_decision_ = false;
    last_fault_reason_ = why;
}

void policy::advance_request(load_bucket bucket, uint64_t request_epoch) {
    auto & runtime = state_for(bucket);
    if (!have_epoch_ || request_epoch > last_request_epoch_) {
        locked_request_ = false;
    }
    if (!runtime.have_transition_anchor) {
        runtime.last_transition_request_epoch = request_epoch;
        runtime.have_transition_anchor = true;
    }
    if (runtime.state == model_state::cooldown && request_epoch >= runtime.next_eligible_request_epoch) {
        const uint64_t dwell_until = saturating_add(runtime.last_transition_request_epoch, cfg_.min_dwell_requests);
        if (request_epoch >= dwell_until) {
            runtime.state = model_state::probing;
            runtime.last_transition_request_epoch = request_epoch;
            runtime.consecutive_wins = 0;
            runtime.consecutive_losses = 0;
            runtime.qualified_depth = 0;
            runtime.probe_target_depth = probe_arm(bucket);
            runtime.have_speedup = false;
            runtime.model_speedup_ewma = 0.0;
            last_fault_reason_ = reason::target_default;
        }
    }
}

bool policy::observe(const observation & obs) {
    if (!config_valid_) {
        return false;
    }
    if (obs.decision_id == discarded_decision_id_ &&
            obs.request_epoch == discarded_request_epoch_) {
        discarded_decision_id_ = 0;
        discarded_request_epoch_ = 0;
        return false;
    }
    // A late or duplicate observation without an attributable pending
    // decision must not cool down the default-constructed C1 bucket.
    if (!pending_decision_) {
        return false;
    }
    const load_bucket pending_bucket = pending_.load_class;
    if (have_epoch_ && (obs.request_epoch < last_request_epoch_ ||
            (obs.request_epoch == last_request_epoch_ && obs.round_epoch <= last_round_epoch_))) {
        pending_decision_ = false;
        demote(pending_bucket, last_request_epoch_, reason::invalid_observation);
        return false;
    }
    if (!valid_observation(obs, cfg_)) {
        pending_decision_ = false;
        demote(pending_bucket, have_epoch_ ? last_request_epoch_ : obs.request_epoch,
                reason::invalid_observation);
        return false;
    }
    if (obs.decision_id == 0 || obs.decision_id != pending_.decision_id ||
            obs.request_epoch != pending_request_epoch_ || obs.observed_action != pending_.selected_action ||
            obs.observed_drafter != pending_.drafter || obs.observed_depth != pending_.depth) {
        pending_decision_ = false;
        demote(pending_bucket, have_epoch_ ? last_request_epoch_ : obs.request_epoch,
                reason::invalid_observation);
        return false;
    }
    selected_bucket_ = pending_.load_class;
    auto & runtime = state_for(selected_bucket_);
    pending_decision_ = false;
    last_request_epoch_ = obs.request_epoch;
    last_round_epoch_ = obs.round_epoch;
    have_epoch_ = true;
    const double measured_rate = static_cast<double>(obs.useful_tokens) /
            static_cast<double>(obs.elapsed_us);
    if (!std::isfinite(measured_rate) || measured_rate <= 0.0) {
        demote(selected_bucket_, obs.request_epoch, reason::invalid_observation);
        return false;
    }
    if (obs.observed_action == action::target) {
        runtime.target_tokens_per_us = runtime.target_tokens_per_us > 0.0
                ? cfg_.ewma_alpha * measured_rate + (1.0 - cfg_.ewma_alpha) * runtime.target_tokens_per_us
                : measured_rate;
        runtime.observations_since_target = 0;
        return true;
    }
    if (obs.observed_action != action::model) {
        return true;
    }
    if (runtime.observations_since_target < std::numeric_limits<uint64_t>::max()) {
        runtime.observations_since_target++;
    }
    const int32_t evaluated_depth = runtime.state == model_state::probing
            ? runtime.probe_target_depth : runtime.qualified_depth;
    if (obs.observed_depth != evaluated_depth) {
        runtime.consecutive_wins = 0;
        runtime.consecutive_losses = 0;
        runtime.have_speedup = false;
        runtime.model_speedup_ewma = 0.0;
        return true;
    }
    if (runtime.model_observations < std::numeric_limits<uint64_t>::max()) {
        runtime.model_observations++;
    }

    const double baseline_rate = obs.target_tokens_per_us > 0.0
            ? obs.target_tokens_per_us : runtime.target_tokens_per_us;
    if (!std::isfinite(baseline_rate) || baseline_rate <= 0.0) {
        demote(selected_bucket_, obs.request_epoch, reason::invalid_observation);
        return false;
    }
    const double speedup = measured_rate / baseline_rate;
    if (!std::isfinite(speedup) || speedup <= 0.0) {
        demote(selected_bucket_, obs.request_epoch, reason::invalid_observation);
        return false;
    }
    if (!runtime.have_speedup) {
        runtime.model_speedup_ewma = speedup;
        runtime.have_speedup = true;
    } else {
        runtime.model_speedup_ewma = cfg_.ewma_alpha * speedup +
                (1.0 - cfg_.ewma_alpha) * runtime.model_speedup_ewma;
    }

    if (runtime.model_speedup_ewma >= 1.0 + cfg_.promote_gain) {
        if (runtime.consecutive_wins < std::numeric_limits<int32_t>::max()) {
            runtime.consecutive_wins++;
        }
        runtime.consecutive_losses = 0;
        if (runtime.state == model_state::probing && obs.observed_depth == runtime.probe_target_depth &&
                runtime.consecutive_wins >= cfg_.promote_samples) {
            const uint64_t dwell_until = saturating_add(
                    runtime.last_transition_request_epoch, cfg_.min_dwell_requests);
            if (obs.request_epoch >= dwell_until) {
                runtime.qualified_depth = runtime.probe_target_depth;
                runtime.state = model_state::active;
                runtime.last_transition_request_epoch = obs.request_epoch;
                runtime.consecutive_wins = 0;
                runtime.have_speedup = false;
                runtime.model_speedup_ewma = 0.0;
            }
        } else if (runtime.state == model_state::active && obs.observed_depth == runtime.qualified_depth &&
                runtime.consecutive_wins >= cfg_.promote_samples) {
            const int32_t deeper = next_arm(runtime.qualified_depth);
            const int32_t bucket_cap = clamp_to_arm(bucket_load_cap(selected_bucket_));
            const uint64_t dwell_until = saturating_add(
                    runtime.last_transition_request_epoch, cfg_.min_dwell_requests);
            if (deeper > runtime.qualified_depth && deeper <= bucket_cap &&
                    obs.request_epoch >= dwell_until) {
                runtime.probe_target_depth = deeper;
                runtime.state = model_state::probing;
                runtime.last_transition_request_epoch = obs.request_epoch;
                runtime.consecutive_wins = 0;
                runtime.have_speedup = false;
                runtime.model_speedup_ewma = 0.0;
            }
        }
    } else if (runtime.model_speedup_ewma < 1.0 - cfg_.demote_loss) {
        if (runtime.consecutive_losses < std::numeric_limits<int32_t>::max()) {
            runtime.consecutive_losses++;
        }
        runtime.consecutive_wins = 0;
        if (runtime.consecutive_losses >= cfg_.demote_samples) {
            if (runtime.state == model_state::probing && runtime.qualified_depth > 0) {
                runtime.state = model_state::active;
                runtime.probe_target_depth = runtime.qualified_depth;
                runtime.last_transition_request_epoch = obs.request_epoch;
                runtime.consecutive_losses = 0;
                runtime.have_speedup = false;
                runtime.model_speedup_ewma = 0.0;
            } else if (runtime.state == model_state::active && previous_arm(runtime.qualified_depth) > 0) {
                runtime.qualified_depth = previous_arm(runtime.qualified_depth);
                runtime.last_transition_request_epoch = obs.request_epoch;
                runtime.consecutive_losses = 0;
                runtime.have_speedup = false;
                runtime.model_speedup_ewma = 0.0;
            } else {
                demote(selected_bucket_, obs.request_epoch, reason::same_request_lockout);
            }
        }
    } else {
        runtime.consecutive_wins = 0;
        runtime.consecutive_losses = 0;
    }
    return true;
}

bool policy::discard(uint64_t decision_id, uint64_t request_epoch) {
    if (!pending_decision_ || decision_id == 0 || decision_id != pending_.decision_id ||
            request_epoch != pending_request_epoch_) {
        return false;
    }
    pending_decision_ = false;
    discarded_decision_id_ = decision_id;
    discarded_request_epoch_ = request_epoch;
    pending_ = {};
    pending_request_epoch_ = 0;
    return true;
}

decision policy::decide(const runtime_context & ctx) {
    const int64_t total_load = static_cast<int64_t>(ctx.active_slots) +
            static_cast<int64_t>(ctx.queued_slots);
    const load_bucket bucket = classify_load(total_load);
    selected_bucket_ = bucket;
    if (!config_valid_) {
        return fallback(reason::invalid_config, ctx.request_epoch, bucket, true);
    }
    if (pending_decision_) {
        if (ctx.request_epoch == pending_request_epoch_) {
            decision result = pending_;
            result.why = reason::pending_feedback;
            return result;
        }
        discarded_decision_id_ = pending_.decision_id;
        discarded_request_epoch_ = pending_request_epoch_;
        pending_decision_ = false;
        pending_ = {};
        pending_request_epoch_ = 0;
    }
    if (ctx.active_slots < 0 || ctx.queued_slots < 0 || ctx.context_tokens < 0 ||
            ctx.max_model_depth < -1 ||
            (have_epoch_ && ctx.request_epoch < last_request_epoch_)) {
        return fallback(reason::invalid_context, ctx.request_epoch, bucket, true);
    }

    advance_request(bucket, ctx.request_epoch);
    auto & runtime = state_for(bucket);
    if (!have_epoch_ || ctx.request_epoch > last_request_epoch_) {
        last_request_epoch_ = ctx.request_epoch;
        last_round_epoch_ = 0;
        have_epoch_ = true;
    }

    const bool vram_guarded = cfg_.hard_vram_free_bytes > 0 || cfg_.soft_vram_free_bytes > 0;
    if (vram_guarded && (ctx.vram_total_bytes == 0 || ctx.vram_free_bytes > ctx.vram_total_bytes ||
            ctx.vram_sample_age_us < 0 || ctx.vram_sample_age_us > cfg_.max_vram_sample_age_us)) {
        return fallback(reason::invalid_context, ctx.request_epoch, bucket, true);
    }
    if (cfg_.hard_vram_free_bytes > 0 && ctx.vram_free_bytes < cfg_.hard_vram_free_bytes) {
        return fallback(reason::hard_vram_guard, ctx.request_epoch, bucket, true);
    }
    if (cfg_.drafter == model_drafter::none) {
        return fallback(reason::target_default, ctx.request_epoch, bucket);
    }
    if (cfg_.bootstrap_target_baseline && runtime.target_tokens_per_us <= 0.0) {
        return fallback(reason::target_bootstrap, ctx.request_epoch, bucket, true);
    }
    if (ctx.allow_target_remeasure && cfg_.bootstrap_target_baseline && cfg_.target_remeasure_interval > 0 &&
            runtime.observations_since_target >= cfg_.target_remeasure_interval) {
        return fallback(reason::target_refresh, ctx.request_epoch, bucket, true);
    }
    if (!ctx.model_state_synchronized) {
        return fallback(reason::model_state_stale, ctx.request_epoch, bucket, true);
    }
    if (locked_request_ && ctx.request_epoch == locked_request_epoch_) {
        return fallback(reason::same_request_lockout, ctx.request_epoch, bucket, true);
    }
    if (runtime.state == model_state::cooldown) {
        return fallback(reason::cooldown_active, ctx.request_epoch, bucket, true);
    }

    const int32_t conservative_load_cap = bucket_load_cap(bucket);
    int32_t cap = std::min(conservative_load_cap, runtime.qualified_depth);
    reason why = reason::model_profitable;
    if (runtime.state == model_state::probing) {
        cap = std::min(conservative_load_cap, runtime.probe_target_depth);
        why = reason::model_probe;
    } else if (cap < cfg_.depth_arms.back()) {
        why = reason::load_cap;
    }
    if (cfg_.soft_vram_free_bytes > 0 && ctx.vram_free_bytes < cfg_.soft_vram_free_bytes) {
        cap = std::min(cap, cfg_.soft_vram_max_depth);
        why = reason::soft_vram_cap;
    }
    if (ctx.max_model_depth >= 0 && cap > ctx.max_model_depth) {
        cap = ctx.max_model_depth;
        why = reason::load_cap;
    }

    const int32_t depth = clamp_to_arm(cap);
    if (depth <= 0) {
        active_depth_ = 0;
        // A zero load cap is a throughput decision, not merely a routing
        // preference. Stop draft maintenance for the remainder of this
        // request; otherwise the draft model still consumes a decode on every
        // target token and can make target-only C>1 substantially slower.
        return fallback(why, ctx.request_epoch, bucket, true);
    }

    active_depth_ = depth;
    decision result;
    result.selected_action = action::model;
    result.drafter = cfg_.drafter;
    result.depth = depth;
    result.why = why;
    result.state = runtime.state;
    result.next_eligible_request_epoch = runtime.next_eligible_request_epoch;
    result.model_speedup_ewma = runtime.model_speedup_ewma;
    result.keep_model_state_synchronized = true;
    result.load_class = bucket;
    return issue(result, ctx.request_epoch);
}

const char * to_string(action value) {
    switch (value) {
        case action::target: return "target";
        case action::lookup: return "lookup";
        case action::model:  return "model";
    }
    return "unknown";
}

const char * to_string(model_drafter value) {
    switch (value) {
        case model_drafter::none:    return "none";
        case model_drafter::mtp:     return "mtp";
        case model_drafter::dflash:  return "dflash";
        case model_drafter::dflash2: return "dflash2";
        case model_drafter::dspark:  return "dspark";
    }
    return "unknown";
}

const char * to_string(model_state value) {
    switch (value) {
        case model_state::unavailable: return "unavailable";
        case model_state::probing:     return "probing";
        case model_state::active:      return "active";
        case model_state::cooldown:    return "cooldown";
    }
    return "unknown";
}

const char * to_string(reason value) {
    switch (value) {
        case reason::target_default:       return "target_default";
        case reason::target_bootstrap:     return "target_bootstrap";
        case reason::target_refresh:       return "target_refresh";
        case reason::invalid_config:       return "invalid_config";
        case reason::invalid_context:      return "invalid_context";
        case reason::invalid_observation:  return "invalid_observation";
        case reason::model_probe:          return "model_probe";
        case reason::model_profitable:     return "model_profitable";
        case reason::load_cap:             return "load_cap";
        case reason::soft_vram_cap:        return "soft_vram_cap";
        case reason::hard_vram_guard:      return "hard_vram_guard";
        case reason::lookup_fallback:      return "lookup_fallback";
        case reason::same_request_lockout: return "same_request_lockout";
        case reason::cooldown_active:      return "cooldown_active";
        case reason::model_state_stale:    return "model_state_stale";
        case reason::pending_feedback:     return "pending_feedback";
    }
    return "unknown";
}

const char * to_string(load_bucket value) {
    switch (value) {
        case load_bucket::c1:      return "C1";
        case load_bucket::c2:      return "C2";
        case load_bucket::c4:      return "C4";
        case load_bucket::c8_plus: return "C8+";
    }
    return "unknown";
}

} // namespace common_spec_runtime
