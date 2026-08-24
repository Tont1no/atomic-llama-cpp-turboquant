#include "cache-runtime-policy.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <limits>

static common_cache_runtime_config test_config() {
    common_cache_runtime_config config;
    config.quota_min_bytes = 100;
    config.quota_max_bytes = 500;
    config.quota_step_bytes = 100;
    config.ttl_min_seconds = 30;
    config.ttl_default_seconds = 120;
    config.ttl_max_seconds = 300;
    config.ttl_step_seconds = 30;
    config.reuse_promote_ratio = 0.60;
    config.reuse_demote_ratio = 0.10;
    config.hit_promote_ratio = 0.60;
    config.hit_demote_ratio = 0.10;
    config.saved_prefill_promote_us = 100.0;
    config.saved_prefill_demote_us = 10.0;
    config.lookup_winner_promote_us = 50.0;
    config.lookup_winner_demote_us = 5.0;
    config.occupancy_promote_max = 0.30;
    config.occupancy_demote_min = 0.70;
    config.decode_promote_max = 0.30;
    config.decode_demote_min = 0.70;
    config.ram_headroom_demote_bytes = 100;
    config.ram_headroom_promote_bytes = 500;
    config.vram_headroom_demote_bytes = 100;
    config.vram_headroom_promote_bytes = 500;
    config.load_promote_max = 1;
    config.load_demote_min = 3;
    config.promote_samples = 2;
    config.demote_samples = 1;
    config.min_dwell_ticks = 2;
    config.telemetry_max_age_ms = 10;
    return config;
}

static common_cache_runtime_input good_input(uint64_t tick) {
    common_cache_runtime_input input;
    input.tick = tick;
    input.telemetry_valid = true;
    input.scope_isolated = true;
    input.reuse_ratio = 0.80;
    input.hit_ratio = 0.80;
    input.saved_prefill_us = 200.0;
    input.lookup_winner_benefit_us = 100.0;
    input.occupancy_ratio = 0.10;
    input.decode_pressure = 0.10;
    input.idle = true;
    input.ram_headroom_bytes = 1000;
    input.vram_headroom_bytes = 1000;
    input.current_quota_bytes = 200;
    input.current_ttl_seconds = 120;
    return input;
}

static common_cache_runtime_state promoted_state() {
    common_cache_runtime_state state;
    state.initialized = true;
    state.retention_enabled = true;
    state.last_change_tick = 0;
    state.last_observed_tick = 0;
    state.target_quota_bytes = 300;
    state.target_ttl_seconds = 120;
    return state;
}

static void test_stale_and_isolation_fail_closed() {
    const auto config = test_config();
    common_cache_runtime_state state;

    auto input = good_input(1);
    input.telemetry_age_ms = 11;
    auto result = common_cache_runtime_decide(config, state, input);
    assert(result.output.reason == COMMON_CACHE_RUNTIME_REASON_STALE_TELEMETRY);
    assert(result.output.quota_bytes == 0);
    assert(result.output.ttl_seconds == config.ttl_min_seconds);
    assert(!result.output.allow_new_retention);
    assert(!result.output.allow_geometry_rebuild);
    assert(!result.state.initialized);

    input = good_input(1);
    input.scope_isolated = false;
    result = common_cache_runtime_decide(config, state, input);
    assert(result.output.reason == COMMON_CACHE_RUNTIME_REASON_SCOPE_ISOLATION);
    assert(result.output.quota_bytes == 0);
    assert(!result.output.allow_new_retention);
}

static void test_pressure_demotion() {
    const auto config = test_config();
    auto state = promoted_state();
    auto input = good_input(2);
    input.current_quota_bytes = 300;
    input.occupancy_ratio = 0.90;

    const auto result = common_cache_runtime_decide(config, state, input);
    assert(result.output.reason == COMMON_CACHE_RUNTIME_REASON_DEMOTE_PRESSURE);
    assert(result.output.quota_bytes == 200);
    assert(result.output.ttl_seconds == 90);
    assert(!result.output.allow_new_retention);
}

static void test_neutral_cold_start_preserves_configured_retention() {
    const auto config = test_config();
    auto input = good_input(1);
    input.reuse_ratio = 0.40;
    input.hit_ratio = 0.40;
    input.saved_prefill_us = 50.0;
    input.lookup_winner_benefit_us = 20.0;
    input.occupancy_ratio = 0.50;
    input.decode_pressure = 0.50;

    const auto result = common_cache_runtime_decide(config, {}, input);
    assert(result.state.initialized);
    assert(result.state.retention_enabled);
    assert(result.output.allow_new_retention);
    assert(result.output.quota_bytes == input.current_quota_bytes);
    assert(result.output.ttl_seconds == input.current_ttl_seconds);
}

static void test_stable_promotion_after_samples_and_dwell() {
    const auto config = test_config();
    common_cache_runtime_state state;

    auto result = common_cache_runtime_decide(config, state, good_input(1));
    assert(result.output.quota_bytes == 200);
    assert(result.output.reason == COMMON_CACHE_RUNTIME_REASON_WAIT_SAMPLES);

    result = common_cache_runtime_decide(config, result.state, good_input(2));
    assert(result.output.quota_bytes == 200);
    assert(result.output.reason == COMMON_CACHE_RUNTIME_REASON_WAIT_DWELL);

    result = common_cache_runtime_decide(config, result.state, good_input(3));
    assert(result.output.reason == COMMON_CACHE_RUNTIME_REASON_PROMOTE);
    assert(result.output.quota_bytes == 300);
    assert(result.output.ttl_seconds == 150);
    assert(result.output.allow_new_retention);
    assert(result.output.allow_geometry_rebuild);
}

static void test_hysteresis_does_not_flap() {
    const auto config = test_config();
    auto state = promoted_state();
    auto input = good_input(2);
    input.current_quota_bytes = 300;
    input.reuse_ratio = 0.40;
    input.hit_ratio = 0.40;
    input.saved_prefill_us = 50.0;
    input.lookup_winner_benefit_us = 20.0;
    input.occupancy_ratio = 0.50;
    input.decode_pressure = 0.50;

    const auto result = common_cache_runtime_decide(config, state, input);
    assert(result.output.reason == COMMON_CACHE_RUNTIME_REASON_HYSTERESIS);
    assert(result.output.quota_bytes == 300);
    assert(result.output.ttl_seconds == 120);
    assert(result.output.allow_new_retention);
}

static void test_geometry_rebuild_guards() {
    const auto config = test_config();
    const auto state = promoted_state();
    auto input = good_input(1);

    auto result = common_cache_runtime_decide(config, state, input);
    assert(result.output.allow_geometry_rebuild);

    input.idle = false;
    result = common_cache_runtime_decide(config, state, input);
    assert(!result.output.allow_geometry_rebuild);

    input = good_input(1);
    input.active_requests = 1;
    result = common_cache_runtime_decide(config, state, input);
    assert(!result.output.allow_geometry_rebuild);

    input = good_input(1);
    input.queued_requests = 1;
    result = common_cache_runtime_decide(config, state, input);
    assert(!result.output.allow_geometry_rebuild);

    input = good_input(1);
    input.speculative_active = true;
    result = common_cache_runtime_decide(config, state, input);
    assert(!result.output.allow_geometry_rebuild);
}

static void test_step_bounds() {
    auto config = test_config();
    config.promote_samples = 1;

    auto state = promoted_state();
    state.target_quota_bytes = 450;
    state.target_ttl_seconds = 290;
    auto input = good_input(2);
    input.current_quota_bytes = 450;
    input.current_ttl_seconds = 290;
    auto result = common_cache_runtime_decide(config, state, input);
    assert(result.output.quota_bytes == 500);
    assert(result.output.ttl_seconds == 300);

    state.target_quota_bytes = 150;
    state.target_ttl_seconds = 40;
    input = good_input(2);
    input.current_quota_bytes = 150;
    input.current_ttl_seconds = 40;
    input.occupancy_ratio = 0.90;
    result = common_cache_runtime_decide(config, state, input);
    assert(result.output.quota_bytes == 100);
    assert(result.output.ttl_seconds == 30);
}

static void test_ttl_changes_at_quota_boundary() {
    auto config = test_config();
    config.promote_samples = 1;
    auto state = promoted_state();
    state.target_quota_bytes = config.quota_max_bytes;
    state.target_ttl_seconds = 120;

    auto input = good_input(2);
    input.current_quota_bytes = config.quota_max_bytes;
    input.current_ttl_seconds = 120;
    const auto result = common_cache_runtime_decide(config, state, input);
    assert(result.output.quota_bytes == config.quota_max_bytes);
    assert(result.output.ttl_seconds == 150);
    assert(result.output.allow_geometry_rebuild);
}

static void test_invalid_values() {
    const auto config = test_config();
    common_cache_runtime_state state;
    auto input = good_input(1);
    input.reuse_ratio = std::numeric_limits<double>::quiet_NaN();
    auto result = common_cache_runtime_decide(config, state, input);
    assert(result.output.reason == COMMON_CACHE_RUNTIME_REASON_INVALID_INPUT);
    assert(result.output.quota_bytes == 0);

    input = good_input(1);
    input.occupancy_ratio = 1.1;
    result = common_cache_runtime_decide(config, state, input);
    assert(result.output.reason == COMMON_CACHE_RUNTIME_REASON_INVALID_INPUT);

    input = good_input(1);
    input.current_quota_bytes = 99;
    result = common_cache_runtime_decide(config, state, input);
    assert(result.output.reason == COMMON_CACHE_RUNTIME_REASON_INVALID_INPUT);

    auto invalid_config = config;
    invalid_config.quota_step_bytes = 0;
    input = good_input(1);
    result = common_cache_runtime_decide(invalid_config, state, input);
    assert(result.output.reason == COMMON_CACHE_RUNTIME_REASON_INVALID_CONFIG);
    assert(result.output.ttl_seconds == 0);
}

int main() {
    test_stale_and_isolation_fail_closed();
    test_pressure_demotion();
    test_neutral_cold_start_preserves_configured_retention();
    test_stable_promotion_after_samples_and_dwell();
    test_hysteresis_does_not_flap();
    test_geometry_rebuild_guards();
    test_step_bounds();
    test_ttl_changes_at_quota_boundary();
    test_invalid_values();
    return 0;
}
