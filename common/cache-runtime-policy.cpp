#include "cache-runtime-policy.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

bool valid_ratio(double value) {
    return std::isfinite(value) && value >= 0.0 && value <= 1.0;
}

bool valid_nonnegative(double value) {
    return std::isfinite(value) && value >= 0.0;
}

bool valid_config(const common_cache_runtime_config & config) {
    if (config.quota_min_bytes >= config.quota_max_bytes || config.quota_step_bytes == 0 || config.quota_step_bytes > config.quota_max_bytes - config.quota_min_bytes) {
        return false;
    }
    if (config.ttl_min_seconds > config.ttl_default_seconds || config.ttl_default_seconds > config.ttl_max_seconds || config.ttl_step_seconds == 0 || config.ttl_step_seconds > config.ttl_max_seconds - config.ttl_min_seconds) {
        return false;
    }
    if (!valid_ratio(config.reuse_promote_ratio) || !valid_ratio(config.reuse_demote_ratio) || config.reuse_demote_ratio >= config.reuse_promote_ratio) {
        return false;
    }
    if (!valid_ratio(config.hit_promote_ratio) || !valid_ratio(config.hit_demote_ratio) || config.hit_demote_ratio >= config.hit_promote_ratio) {
        return false;
    }
    if (!valid_nonnegative(config.saved_prefill_promote_us) || !valid_nonnegative(config.saved_prefill_demote_us) || config.saved_prefill_demote_us >= config.saved_prefill_promote_us) {
        return false;
    }
    if (!valid_nonnegative(config.lookup_winner_promote_us) || !valid_nonnegative(config.lookup_winner_demote_us) || config.lookup_winner_demote_us >= config.lookup_winner_promote_us) {
        return false;
    }
    if (!valid_ratio(config.occupancy_promote_max) || !valid_ratio(config.occupancy_demote_min) || config.occupancy_promote_max >= config.occupancy_demote_min) {
        return false;
    }
    if (!valid_ratio(config.decode_promote_max) || !valid_ratio(config.decode_demote_min) || config.decode_promote_max >= config.decode_demote_min) {
        return false;
    }
    if (config.ram_headroom_demote_bytes >= config.ram_headroom_promote_bytes || config.vram_headroom_demote_bytes >= config.vram_headroom_promote_bytes) {
        return false;
    }
    if (config.load_promote_max >= config.load_demote_min || config.promote_samples == 0 || config.demote_samples == 0 || config.min_dwell_ticks == 0 || config.telemetry_max_age_ms == 0) {
        return false;
    }
    return true;
}

common_cache_runtime_result fail_closed(
        const common_cache_runtime_config & config,
        common_cache_runtime_reason reason) {
    common_cache_runtime_result result;
    result.state = {};
    result.output.quota_bytes = 0;
    result.output.ttl_seconds = valid_config(config) ? config.ttl_min_seconds : 0;
    result.output.reason = reason;
    return result;
}

uint32_t increment_saturating(uint32_t value) {
    return value == std::numeric_limits<uint32_t>::max() ? value : value + 1;
}

uint64_t increase_bounded(uint64_t value, uint64_t step, uint64_t maximum) {
    return value >= maximum - std::min(step, maximum) ? maximum : value + step;
}

uint64_t decrease_bounded(uint64_t value, uint64_t step, uint64_t minimum) {
    return value <= minimum + std::min(step, std::numeric_limits<uint64_t>::max() - minimum) ? minimum : value - step;
}

uint32_t increase_bounded(uint32_t value, uint32_t step, uint32_t maximum) {
    return value >= maximum - std::min(step, maximum) ? maximum : value + step;
}

uint32_t decrease_bounded(uint32_t value, uint32_t step, uint32_t minimum) {
    return value <= minimum + std::min(step, std::numeric_limits<uint32_t>::max() - minimum) ? minimum : value - step;
}

uint64_t request_load(const common_cache_runtime_input & input) {
    return static_cast<uint64_t>(input.active_requests) + static_cast<uint64_t>(input.queued_requests);
}

} // namespace

common_cache_runtime_result common_cache_runtime_decide(
        const common_cache_runtime_config & config,
        const common_cache_runtime_state & state,
        const common_cache_runtime_input & input) {
    if (!valid_config(config)) {
        return fail_closed(config, COMMON_CACHE_RUNTIME_REASON_INVALID_CONFIG);
    }
    if (!input.scope_isolated) {
        return fail_closed(config, COMMON_CACHE_RUNTIME_REASON_SCOPE_ISOLATION);
    }
    if (!input.telemetry_valid || input.telemetry_age_ms > config.telemetry_max_age_ms) {
        return fail_closed(config, COMMON_CACHE_RUNTIME_REASON_STALE_TELEMETRY);
    }
    if (!valid_ratio(input.reuse_ratio) || !valid_ratio(input.hit_ratio) || !valid_nonnegative(input.saved_prefill_us) || !valid_nonnegative(input.lookup_winner_benefit_us) || !valid_ratio(input.occupancy_ratio) || !valid_ratio(input.decode_pressure)) {
        return fail_closed(config, COMMON_CACHE_RUNTIME_REASON_INVALID_INPUT);
    }
    if (input.current_quota_bytes < config.quota_min_bytes || input.current_quota_bytes > config.quota_max_bytes || input.current_ttl_seconds < config.ttl_min_seconds || input.current_ttl_seconds > config.ttl_max_seconds) {
        return fail_closed(config, COMMON_CACHE_RUNTIME_REASON_INVALID_INPUT);
    }
    if (state.initialized && (input.tick <= state.last_observed_tick || state.last_change_tick > state.last_observed_tick || state.target_quota_bytes < config.quota_min_bytes || state.target_quota_bytes > config.quota_max_bytes || state.target_ttl_seconds < config.ttl_min_seconds || state.target_ttl_seconds > config.ttl_max_seconds)) {
        return fail_closed(config, COMMON_CACHE_RUNTIME_REASON_INVALID_INPUT);
    }

    common_cache_runtime_result result;
    result.state = state.initialized ? state : common_cache_runtime_state{};
    result.output.reason = COMMON_CACHE_RUNTIME_REASON_HOLD;

    if (!result.state.initialized) {
        result.state.initialized = true;
        // A constructed server_prompt_cache is already explicitly enabled by
        // configuration. Preserve that state for neutral startup samples;
        // pressure or stale telemetry can still close the gate immediately.
        result.state.retention_enabled = true;
        result.state.last_change_tick = input.tick;
        result.state.target_quota_bytes = input.current_quota_bytes;
        result.state.target_ttl_seconds = input.current_ttl_seconds;
    }
    result.state.last_observed_tick = input.tick;

    const uint64_t load = request_load(input);
    const bool pressure_high = input.occupancy_ratio >= config.occupancy_demote_min || input.decode_pressure >= config.decode_demote_min || input.ram_headroom_bytes < config.ram_headroom_demote_bytes || input.vram_headroom_bytes < config.vram_headroom_demote_bytes || load >= config.load_demote_min;
    const bool pressure_low = input.occupancy_ratio <= config.occupancy_promote_max && input.decode_pressure <= config.decode_promote_max && input.ram_headroom_bytes >= config.ram_headroom_promote_bytes && input.vram_headroom_bytes >= config.vram_headroom_promote_bytes && load <= config.load_promote_max;
    const bool benefit_high = input.reuse_ratio >= config.reuse_promote_ratio || input.hit_ratio >= config.hit_promote_ratio || input.saved_prefill_us >= config.saved_prefill_promote_us || input.lookup_winner_benefit_us >= config.lookup_winner_promote_us;
    const bool benefit_low = input.reuse_ratio <= config.reuse_demote_ratio && input.hit_ratio <= config.hit_demote_ratio && input.saved_prefill_us <= config.saved_prefill_demote_us && input.lookup_winner_benefit_us <= config.lookup_winner_demote_us;

    if (pressure_high || benefit_low) {
        result.state.promote_streak = 0;
        result.state.demote_streak = increment_saturating(result.state.demote_streak);
    } else if (pressure_low && benefit_high) {
        result.state.demote_streak = 0;
        result.state.promote_streak = increment_saturating(result.state.promote_streak);
    } else {
        result.state.promote_streak = 0;
        result.state.demote_streak = 0;
        result.output.reason = COMMON_CACHE_RUNTIME_REASON_HYSTERESIS;
    }

    const bool dwell_complete = input.tick >= result.state.last_change_tick && input.tick - result.state.last_change_tick >= config.min_dwell_ticks;
    if (result.state.demote_streak >= config.demote_samples) {
        if (dwell_complete) {
            result.state.target_quota_bytes = decrease_bounded(result.state.target_quota_bytes, config.quota_step_bytes, config.quota_min_bytes);
            result.state.target_ttl_seconds = decrease_bounded(result.state.target_ttl_seconds, config.ttl_step_seconds, config.ttl_min_seconds);
            result.state.retention_enabled = false;
            result.state.demote_streak = 0;
            result.state.last_change_tick = input.tick;
            result.output.reason = pressure_high ? COMMON_CACHE_RUNTIME_REASON_DEMOTE_PRESSURE : COMMON_CACHE_RUNTIME_REASON_DEMOTE_LOW_BENEFIT;
        } else {
            result.output.reason = COMMON_CACHE_RUNTIME_REASON_WAIT_DWELL;
        }
    } else if (result.state.promote_streak >= config.promote_samples) {
        if (dwell_complete) {
            result.state.target_quota_bytes = increase_bounded(result.state.target_quota_bytes, config.quota_step_bytes, config.quota_max_bytes);
            result.state.target_ttl_seconds = increase_bounded(result.state.target_ttl_seconds, config.ttl_step_seconds, config.ttl_max_seconds);
            result.state.retention_enabled = true;
            result.state.promote_streak = 0;
            result.state.last_change_tick = input.tick;
            result.output.reason = COMMON_CACHE_RUNTIME_REASON_PROMOTE;
        } else {
            result.output.reason = COMMON_CACHE_RUNTIME_REASON_WAIT_DWELL;
        }
    } else if (result.output.reason == COMMON_CACHE_RUNTIME_REASON_HOLD) {
        result.output.reason = COMMON_CACHE_RUNTIME_REASON_WAIT_SAMPLES;
    }

    result.output.quota_bytes = result.state.target_quota_bytes;
    result.output.ttl_seconds = result.state.target_ttl_seconds;
    result.output.allow_new_retention = result.state.retention_enabled && !pressure_high;
    result.output.allow_geometry_rebuild =
            (result.state.target_quota_bytes != input.current_quota_bytes ||
             result.state.target_ttl_seconds != input.current_ttl_seconds) &&
            input.idle && input.active_requests == 0 && input.queued_requests == 0 &&
            !input.speculative_active;
    return result;
}
