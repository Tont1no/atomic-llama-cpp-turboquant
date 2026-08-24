#pragma once

#include <cstdint>

enum common_cache_runtime_reason {
    COMMON_CACHE_RUNTIME_REASON_HOLD,
    COMMON_CACHE_RUNTIME_REASON_PROMOTE,
    COMMON_CACHE_RUNTIME_REASON_DEMOTE_PRESSURE,
    COMMON_CACHE_RUNTIME_REASON_DEMOTE_LOW_BENEFIT,
    COMMON_CACHE_RUNTIME_REASON_WAIT_SAMPLES,
    COMMON_CACHE_RUNTIME_REASON_WAIT_DWELL,
    COMMON_CACHE_RUNTIME_REASON_HYSTERESIS,
    COMMON_CACHE_RUNTIME_REASON_STALE_TELEMETRY,
    COMMON_CACHE_RUNTIME_REASON_SCOPE_ISOLATION,
    COMMON_CACHE_RUNTIME_REASON_INVALID_INPUT,
    COMMON_CACHE_RUNTIME_REASON_INVALID_CONFIG,
};

struct common_cache_runtime_config {
    uint64_t quota_min_bytes  = 64ULL * 1024 * 1024;
    uint64_t quota_max_bytes  = 8ULL * 1024 * 1024 * 1024;
    uint64_t quota_step_bytes = 256ULL * 1024 * 1024;

    uint32_t ttl_min_seconds     = 30;
    uint32_t ttl_max_seconds     = 600;
    uint32_t ttl_step_seconds    = 30;
    // Retention expires 120 seconds after a session closes by default.
    uint32_t ttl_default_seconds = 120;

    double reuse_promote_ratio = 0.30;
    double reuse_demote_ratio  = 0.05;
    double hit_promote_ratio   = 0.30;
    double hit_demote_ratio    = 0.05;

    double saved_prefill_promote_us       = 5000.0;
    double saved_prefill_demote_us        = 100.0;
    double lookup_winner_promote_us       = 1000.0;
    double lookup_winner_demote_us        = 50.0;

    double occupancy_promote_max = 0.60;
    double occupancy_demote_min  = 0.85;
    double decode_promote_max    = 0.50;
    double decode_demote_min     = 0.80;

    uint64_t ram_headroom_demote_bytes  = 512ULL * 1024 * 1024;
    uint64_t ram_headroom_promote_bytes = 2ULL * 1024 * 1024 * 1024;
    uint64_t vram_headroom_demote_bytes  = 512ULL * 1024 * 1024;
    uint64_t vram_headroom_promote_bytes = 2ULL * 1024 * 1024 * 1024;

    uint64_t load_promote_max = 1;
    uint64_t load_demote_min  = 4;

    uint32_t promote_samples = 3;
    uint32_t demote_samples  = 2;
    uint64_t min_dwell_ticks = 3;
    uint64_t telemetry_max_age_ms = 5000;
};

struct common_cache_runtime_input {
    uint64_t tick = 0;

    bool telemetry_valid = false;
    uint64_t telemetry_age_ms = 0;
    bool scope_isolated = false;

    double reuse_ratio = 0.0;
    double hit_ratio = 0.0;
    double saved_prefill_us = 0.0;
    double lookup_winner_benefit_us = 0.0;
    double occupancy_ratio = 0.0;
    double decode_pressure = 0.0;

    uint32_t active_requests = 0;
    uint32_t queued_requests = 0;
    bool idle = false;
    bool speculative_active = false;

    uint64_t ram_headroom_bytes = 0;
    uint64_t vram_headroom_bytes = 0;
    uint64_t current_quota_bytes = 0;
    uint32_t current_ttl_seconds = 0;
};

struct common_cache_runtime_state {
    bool initialized = false;
    bool retention_enabled = false;

    uint32_t promote_streak = 0;
    uint32_t demote_streak = 0;
    uint64_t last_change_tick = 0;
    uint64_t last_observed_tick = 0;

    uint64_t target_quota_bytes = 0;
    uint32_t target_ttl_seconds = 0;
};

struct common_cache_runtime_output {
    uint64_t quota_bytes = 0;
    uint32_t ttl_seconds = 0;
    bool allow_new_retention = false;
    bool allow_geometry_rebuild = false;
    common_cache_runtime_reason reason = COMMON_CACHE_RUNTIME_REASON_HOLD;
};

struct common_cache_runtime_result {
    common_cache_runtime_state state;
    common_cache_runtime_output output;
};

// Call this from a cache maintenance loop, not from the decode or speculative hot loop.
common_cache_runtime_result common_cache_runtime_decide(
        const common_cache_runtime_config & config,
        const common_cache_runtime_state & state,
        const common_cache_runtime_input & input);
