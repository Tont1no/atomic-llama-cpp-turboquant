#include "speculative-runtime-policy.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cmath>

using namespace common_spec_runtime;

static config make_config() {
    config cfg;
    cfg.drafter = model_drafter::dspark;
    cfg.depth_arms = { 0, 1, 2, 3, 5, 7 };
    cfg.lookup_enabled = true;
    cfg.lookup_depth = 4;
    cfg.load_depth_caps = { 7, 3, 2, 1, 1, 1, 1, 0 };
    cfg.hard_vram_free_bytes = 2'000;
    cfg.soft_vram_free_bytes = 4'000;
    cfg.soft_vram_max_depth = 1;
    cfg.promote_samples = 3;
    cfg.demote_samples = 2;
    cfg.cooldown_requests = 2;
    cfg.min_dwell_requests = 3;
    cfg.probe_depth = 1;
    cfg.ewma_alpha = 1.0;
    return cfg;
}

static runtime_context context(uint64_t request, int active = 1, int queued = 0) {
    runtime_context ctx;
    ctx.request_epoch = request;
    ctx.active_slots = active;
    ctx.queued_slots = queued;
    ctx.context_tokens = 1024;
    ctx.vram_free_bytes = 8'000;
    ctx.vram_total_bytes = 10'000;
    ctx.model_state_synchronized = true;
    return ctx;
}

static observation model_sample(uint64_t request, uint64_t round, double speedup) {
    observation obs;
    obs.request_epoch = request;
    obs.round_epoch = round;
    obs.observed_action = action::model;
    obs.observed_drafter = model_drafter::dspark;
    obs.observed_depth = 1;
    obs.elapsed_us = 1'000;
    obs.useful_tokens = static_cast<int32_t>(speedup * 100.0);
    obs.target_tokens_per_us = 0.1;
    obs.synchronized = true;
    obs.complete = true;
    return obs;
}

static bool observe_model(policy & runtime, uint64_t request, uint64_t round, double speedup,
        int active = 1, int queued = 0) {
    const auto selected = runtime.decide(context(request, active, queued));
    if (selected.selected_action != action::model) {
        return false;
    }
    auto obs = model_sample(request, round, speedup);
    obs.decision_id = selected.decision_id;
    obs.observed_depth = selected.depth;
    return runtime.observe(obs);
}

static bool observe_target(policy & runtime, uint64_t request, uint64_t round,
        int active, int queued = 0, int32_t useful_tokens = 100, int64_t elapsed_us = 1'000) {
    const auto selected = runtime.decide(context(request, active, queued));
    if (selected.selected_action != action::target) {
        return false;
    }
    observation obs;
    obs.decision_id = selected.decision_id;
    obs.request_epoch = request;
    obs.round_epoch = round;
    obs.observed_action = action::target;
    obs.elapsed_us = elapsed_us;
    obs.useful_tokens = useful_tokens;
    obs.synchronized = true;
    obs.complete = true;
    return runtime.observe(obs);
}

static bool observe_aggregate_model(policy & runtime, uint64_t group_epoch,
        int active, int32_t useful_tokens, int64_t elapsed_us, double baseline_tokens_per_us) {
    const auto selected = runtime.decide(context(group_epoch, active));
    if (selected.selected_action != action::model) {
        return false;
    }
    observation obs;
    obs.decision_id = selected.decision_id;
    obs.request_epoch = group_epoch;
    obs.round_epoch = group_epoch;
    obs.observed_action = action::model;
    obs.observed_drafter = model_drafter::dspark;
    obs.observed_depth = selected.depth;
    obs.elapsed_us = elapsed_us;
    obs.useful_tokens = useful_tokens;
    obs.target_tokens_per_us = baseline_tokens_per_us;
    obs.synchronized = true;
    obs.complete = true;
    return runtime.observe(obs);
}

static void test_config_validation() {
    config cfg = make_config();
    assert(policy::valid_config(cfg));

    cfg.depth_arms = { 0, 3, 2 };
    assert(!policy::valid_config(cfg));

    cfg = make_config();
    cfg.hard_vram_free_bytes = 5'000;
    assert(!policy::valid_config(cfg));

    cfg = make_config();
    cfg.drafter = model_drafter::none;
    assert(!policy::valid_config(cfg));

    cfg = make_config();
    cfg.depth_arms.clear();
    policy invalid(std::move(cfg));
    assert(invalid.decide(context(1)).why == reason::invalid_config);
}

static void test_probe_promote_and_load_cap() {
    policy runtime(make_config());
    assert(runtime.decide(context(3)).depth == 1);

    assert(observe_model(runtime, 3, 1, 1.20));
    assert(observe_model(runtime, 4, 1, 1.20));
    assert(observe_model(runtime, 6, 1, 1.20));
    assert(runtime.get_model_state() == model_state::active);

    auto low = runtime.decide(context(6, 1));
    assert(low.selected_action == action::model && low.depth == 1);
    assert(runtime.get_active_depth() == 1);
    assert(runtime.discard(low.decision_id, 6));

    auto high = runtime.decide(context(6, 4));
    assert(high.selected_action == action::model && high.depth == 1);
    assert(high.why == reason::model_probe);
    assert(runtime.discard(high.decision_id, 6));

    auto saturated = runtime.decide(context(6, 8));
    assert(saturated.selected_action == action::lookup);
    assert(!saturated.keep_model_state_synchronized);
    assert(runtime.get_active_depth() == 0);
}

static void test_demotion_latches_current_request() {
    policy runtime(make_config());
    assert(observe_model(runtime, 3, 1, 0.80));
    assert(observe_model(runtime, 3, 2, 0.80));
    assert(runtime.get_model_state() == model_state::cooldown);

    auto same = runtime.decide(context(3));
    assert(same.selected_action == action::lookup);
    assert(same.why == reason::same_request_lockout);
    assert(runtime.discard(same.decision_id, 3));

    auto early = runtime.decide(context(5));
    assert(early.selected_action == action::lookup);
    assert(runtime.get_model_state() == model_state::cooldown);
    assert(runtime.discard(early.decision_id, 5));

    auto probe = runtime.decide(context(6));
    assert(probe.selected_action == action::model);
    assert(probe.depth == 1);
    assert(probe.why == reason::model_probe);
}

static void test_deadband_does_not_flap() {
    policy runtime(make_config());
    assert(observe_model(runtime, 3, 1, 1.20));
    assert(observe_model(runtime, 4, 1, 1.01));
    assert(observe_model(runtime, 5, 1, 1.20));
    assert(runtime.get_model_state() == model_state::probing);
}

static void test_vram_guards() {
    policy runtime(make_config());
    auto soft_ctx = context(3);
    soft_ctx.vram_free_bytes = 3'000;
    auto soft = runtime.decide(soft_ctx);
    assert(soft.selected_action == action::model && soft.depth == 1);
    assert(soft.why == reason::soft_vram_cap);
    assert(runtime.discard(soft.decision_id, 3));

    auto hard_ctx = context(3);
    hard_ctx.vram_free_bytes = 1'000;
    auto hard = runtime.decide(hard_ctx);
    assert(hard.selected_action == action::target && hard.depth == 0);
    assert(hard.why == reason::hard_vram_guard);
    assert(runtime.discard(hard.decision_id, 3));

    auto stale = context(3);
    stale.vram_sample_age_us = 20 * 1000 * 1000;
    assert(runtime.decide(stale).why == reason::invalid_context);
}

static void test_invalid_measurement_fails_closed() {
    policy runtime(make_config());
    auto bad = model_sample(3, 1, 1.20);
    const auto selected = runtime.decide(context(3));
    bad.decision_id = selected.decision_id;
    bad.observed_depth = selected.depth;
    bad.synchronized = false;
    assert(!runtime.observe(bad));
    assert(runtime.get_model_state() == model_state::cooldown);
    assert(runtime.decide(context(3)).selected_action == action::lookup);

    auto duplicate = model_sample(3, 1, 1.20);
    duplicate.decision_id = selected.decision_id;
    duplicate.observed_depth = selected.depth;
    assert(!runtime.observe(duplicate));
}

static void test_stale_model_state_is_target_only() {
    policy runtime(make_config());
    auto ctx = context(3);
    ctx.model_state_synchronized = false;
    const auto result = runtime.decide(ctx);
    assert(result.selected_action == action::lookup);
    assert(result.why == reason::model_state_stale);
}

static void test_lookup_feedback_and_causal_binding() {
    policy runtime(make_config());
    const auto selected = runtime.decide(context(3, 8));
    assert(selected.selected_action == action::lookup);
    assert(!selected.keep_model_state_synchronized);

    observation lookup;
    lookup.decision_id = selected.decision_id;
    lookup.request_epoch = 3;
    lookup.round_epoch = 1;
    lookup.observed_action = action::lookup;
    lookup.observed_depth = selected.depth;
    lookup.elapsed_us = 100;
    lookup.useful_tokens = 2;
    lookup.target_tokens_per_us = 0.01;
    lookup.synchronized = true;
    lookup.complete = true;
    assert(runtime.observe(lookup));
    assert(runtime.get_model_state() == model_state::probing);

    const auto lookup_again = runtime.decide(context(3, 8));
    auto stale_model = model_sample(3, 2, 1.20);
    stale_model.decision_id = lookup_again.decision_id;
    stale_model.observed_depth = 1;
    assert(!runtime.observe(stale_model));
    assert(runtime.get_model_state() == model_state::cooldown);
}

static void test_depths_are_qualified_stepwise() {
    auto cfg = make_config();
    cfg.promote_samples = 2;
    cfg.min_dwell_requests = 1;
    policy runtime(cfg);

    assert(observe_model(runtime, 1, 1, 1.20));
    assert(observe_model(runtime, 2, 1, 1.20));
    assert(runtime.get_model_state() == model_state::active);
    const auto qualified = runtime.decide(context(2));
    assert(qualified.depth == 1);
    assert(runtime.discard(qualified.decision_id, 2));

    assert(observe_model(runtime, 3, 1, 1.20));
    assert(observe_model(runtime, 4, 1, 1.20));
    assert(runtime.get_model_state() == model_state::probing);
    const auto deeper_probe = runtime.decide(context(4));
    assert(deeper_probe.depth == 2);
    assert(deeper_probe.depth < cfg.depth_arms.back());
}

static void test_hard_guard_latches_model_until_next_request() {
    policy runtime(make_config());
    auto pressure = context(3);
    pressure.vram_free_bytes = 1'000;
    const auto stopped = runtime.decide(pressure);
    assert(stopped.selected_action == action::target);
    assert(!stopped.keep_model_state_synchronized);
    assert(runtime.discard(stopped.decision_id, 3));

    const auto recovered_same_request = runtime.decide(context(3));
    assert(recovered_same_request.selected_action == action::lookup);
    assert(recovered_same_request.why == reason::same_request_lockout);
    assert(!recovered_same_request.keep_model_state_synchronized);
    assert(runtime.discard(recovered_same_request.decision_id, 3));

    const auto next_request = runtime.decide(context(4));
    assert(next_request.selected_action == action::model);
    assert(next_request.depth == 1);
}

static void test_capped_depth_does_not_qualify_deeper_arm() {
    auto cfg = make_config();
    cfg.promote_samples = 2;
    cfg.min_dwell_requests = 1;
    policy runtime(cfg);

    assert(observe_model(runtime, 1, 1, 1.20));
    assert(observe_model(runtime, 2, 1, 1.20));
    assert(observe_model(runtime, 3, 1, 1.20));
    assert(observe_model(runtime, 4, 1, 1.20));
    assert(runtime.get_model_state() == model_state::probing);

    // The depth-2 probe is capped to depth 1 at load 4. Those wins must not
    // pre-qualify depth 2.
    assert(observe_model(runtime, 5, 1, 1.20, 4));
    assert(observe_model(runtime, 6, 1, 1.20, 4));
    assert(observe_model(runtime, 7, 1, 1.20));
    assert(runtime.get_model_state() == model_state::probing);
    assert(observe_model(runtime, 8, 1, 1.20));
    assert(runtime.get_model_state() == model_state::active);
}

static void test_pending_decision_is_not_overwritten_and_can_be_discarded() {
    policy runtime(make_config());
    const decision first = runtime.decide(context(1));
    const decision repeated = runtime.decide(context(1));
    assert(repeated.decision_id == first.decision_id);
    assert(repeated.why == reason::pending_feedback);
    assert(runtime.discard(first.decision_id, 1));
    assert(!runtime.discard(first.decision_id, 1));
    const decision next = runtime.decide(context(1));
    assert(next.decision_id != first.decision_id);
    observation late = model_sample(1, 1, 1.20);
    late.decision_id = first.decision_id;
    late.observed_depth = first.depth;
    assert(!runtime.observe(late));
    assert(runtime.get_model_state() == model_state::probing);
    assert(runtime.discard(next.decision_id, 1));
}

static void test_cold_start_target_bootstrap_precedes_model_probe() {
    auto cfg = make_config();
    cfg.bootstrap_target_baseline = true;
    policy runtime(cfg);
    const decision target = runtime.decide(context(1));
    assert(target.selected_action == action::target);
    assert(target.why == reason::target_bootstrap);
    assert(!target.keep_model_state_synchronized);

    observation obs;
    obs.decision_id = target.decision_id;
    obs.request_epoch = 1;
    obs.round_epoch = 1;
    obs.observed_action = action::target;
    obs.elapsed_us = 1000;
    obs.useful_tokens = 2;
    obs.synchronized = true;
    obs.complete = true;
    assert(runtime.observe(obs));
    const decision locked = runtime.decide(context(1));
    assert(locked.selected_action == action::lookup);
    assert(locked.why == reason::same_request_lockout);
    assert(runtime.discard(locked.decision_id, 1));
    const decision model = runtime.decide(context(2));
    assert(model.selected_action == action::model);
}

static void test_target_observation_bootstraps_internal_baseline() {
    auto cfg = make_config();
    cfg.drafter = model_drafter::none;
    cfg.depth_arms = { 0 };
    cfg.load_depth_caps = { 0 };
    cfg.lookup_enabled = false;
    cfg.lookup_depth = 0;
    policy runtime(cfg);
    const decision selected = runtime.decide(context(1));
    observation obs;
    obs.decision_id = selected.decision_id;
    obs.request_epoch = 1;
    obs.round_epoch = 1;
    obs.observed_action = action::target;
    obs.elapsed_us = 1000;
    obs.useful_tokens = 2;
    obs.synchronized = true;
    obs.complete = true;
    assert(runtime.observe(obs));
    assert(std::abs(runtime.get_target_tokens_per_us() - 0.002) < 1e-12);
}

static void test_load_buckets_qualify_independently() {
    auto cfg = make_config();
    cfg.promote_samples = 2;
    cfg.min_dwell_requests = 1;
    cfg.load_depth_caps = { 3, 3, 3, 3, 3, 3, 3, 3 };
    policy runtime(cfg);

    assert(policy::classify_load(1) == load_bucket::c1);
    assert(policy::classify_load(2) == load_bucket::c2);
    assert(policy::classify_load(3) == load_bucket::c4);
    assert(policy::classify_load(4) == load_bucket::c4);
    assert(policy::classify_load(5) == load_bucket::c8_plus);
    assert(policy::classify_load(64) == load_bucket::c8_plus);

    assert(observe_model(runtime, 1, 1, 1.20, 1));
    assert(observe_model(runtime, 2, 1, 1.20, 1));
    assert(runtime.get_model_state(load_bucket::c1) == model_state::active);
    assert(runtime.get_model_state(load_bucket::c2) == model_state::probing);

    assert(observe_model(runtime, 3, 1, 1.20, 2));
    assert(runtime.get_model_speedup_ewma(load_bucket::c2) > 1.19);
    assert(runtime.get_model_speedup_ewma(load_bucket::c1) == 0.0);
    assert(observe_model(runtime, 4, 1, 1.20, 2));
    assert(runtime.get_model_state(load_bucket::c2) == model_state::active);
    assert(runtime.get_model_state(load_bucket::c4) == model_state::probing);

    assert(observe_model(runtime, 5, 1, 0.80, 4));
    assert(observe_model(runtime, 6, 1, 0.80, 4));
    assert(runtime.get_model_state(load_bucket::c4) == model_state::cooldown);
    assert(runtime.get_model_state(load_bucket::c1) == model_state::active);
    assert(runtime.get_model_state(load_bucket::c2) == model_state::active);
}

static void test_target_bootstrap_is_load_bucket_local() {
    auto cfg = make_config();
    cfg.bootstrap_target_baseline = true;
    policy runtime(cfg);

    assert(observe_target(runtime, 1, 1, 1));
    assert(runtime.get_target_tokens_per_us(load_bucket::c1) > 0.0);
    assert(runtime.get_target_tokens_per_us(load_bucket::c2) == 0.0);

    const auto c2_target = runtime.decide(context(2, 2));
    assert(c2_target.selected_action == action::target);
    assert(c2_target.why == reason::target_bootstrap);
    assert(c2_target.load_class == load_bucket::c2);
    assert(runtime.discard(c2_target.decision_id, 2));

    const auto c1_probe = runtime.decide(context(3, 1));
    assert(c1_probe.selected_action == action::model);
    assert(c1_probe.load_class == load_bucket::c1);
    assert(runtime.discard(c1_probe.decision_id, 3));
}

static void test_one_aggregate_observation_per_c4_batch() {
    auto cfg = make_config();
    cfg.promote_samples = 2;
    cfg.min_dwell_requests = 1;
    cfg.load_depth_caps = { 3, 3, 3, 3 };
    policy runtime(cfg);

    // Four slots accepted unequal counts [1, 2, 4, 7].  The server-facing
    // contract submits their sum once, not four overlapping observations.
    const int32_t accepted_per_slot[] = { 1, 2, 4, 7 };
    int32_t useful_total = 0;
    for (int32_t value : accepted_per_slot) {
        useful_total += value;
    }
    assert(observe_aggregate_model(runtime, 1, 4, useful_total, 1'000, 0.010));
    assert(runtime.get_model_observations(load_bucket::c4) == 1);
    assert(std::abs(runtime.get_model_speedup_ewma(load_bucket::c4) - 1.4) < 1e-12);
    assert(runtime.get_model_state(load_bucket::c4) == model_state::probing);

    assert(observe_aggregate_model(runtime, 2, 4, useful_total, 1'000, 0.010));
    assert(runtime.get_model_observations(load_bucket::c4) == 2);
    assert(runtime.get_model_state(load_bucket::c4) == model_state::active);
}

static void test_group_epoch_ignores_shuffled_slot_reuse() {
    auto cfg = make_config();
    cfg.promote_samples = 2;
    cfg.min_dwell_requests = 1;
    cfg.load_depth_caps = { 3, 3, 3, 3 };
    policy runtime(cfg);

    // Slot-local request epochs are intentionally unrelated.  They are not
    // inputs to the server-wide controller; only group epochs 1 and 2 are.
    const uint64_t slot_request_epochs[] = { 100, 1, 57, 3 };
    assert(slot_request_epochs[0] != slot_request_epochs[1]);
    assert(observe_aggregate_model(runtime, 1, 4, 16, 1'000, 0.010));
    assert(observe_aggregate_model(runtime, 2, 4, 16, 1'000, 0.010));
    assert(runtime.get_model_state(load_bucket::c4) == model_state::active);
    assert(runtime.get_model_observations(load_bucket::c4) == 2);
}

static void test_discarded_partial_group_does_not_starve_next_clean_batch() {
    auto cfg = make_config();
    cfg.promote_samples = 1;
    cfg.min_dwell_requests = 1;
    cfg.load_depth_caps = { 3, 3, 3, 3 };
    policy runtime(cfg);

    const auto partial = runtime.decide(context(1, 4));
    assert(partial.selected_action == action::model);
    assert(runtime.discard(partial.decision_id, 1));
    assert(runtime.get_model_observations(load_bucket::c4) == 0);

    const auto clean = runtime.decide(context(2, 4));
    assert(clean.selected_action == action::model);
    assert(clean.decision_id != partial.decision_id);
    observation obs = model_sample(2, 2, 1.20);
    obs.decision_id = clean.decision_id;
    obs.observed_depth = clean.depth;
    assert(runtime.observe(obs));
    assert(runtime.get_model_observations(load_bucket::c4) == 1);
    assert(runtime.get_model_state(load_bucket::c4) == model_state::active);
}

static void test_group_depth_cap_is_hard() {
    auto cfg = make_config();
    cfg.probe_depth = 3;
    cfg.load_depth_caps = { 7, 7, 7, 7 };
    policy runtime(cfg);
    auto ctx = context(1, 4);
    ctx.max_model_depth = 1;
    const auto selected = runtime.decide(ctx);
    assert(selected.selected_action == action::model);
    assert(selected.depth == 1);
    assert(selected.why == reason::load_cap);
}

static void test_conservative_bucket_cap_survives_alternating_c3_c4() {
    auto cfg = make_config();
    cfg.promote_samples = 2;
    cfg.min_dwell_requests = 1;
    cfg.probe_depth = 3;
    // Exact caps differ at load 3 and 4.  Their shared C4 learner must use
    // min(2, 1) == 1 for both observations.
    cfg.load_depth_caps = { 7, 3, 2, 1, 3, 2, 1, 2 };
    policy runtime(cfg);
    assert(runtime.get_bucket_load_cap(load_bucket::c1) == 7);
    assert(runtime.get_bucket_load_cap(load_bucket::c2) == 3);
    assert(runtime.get_bucket_load_cap(load_bucket::c4) == 1);
    assert(runtime.get_bucket_load_cap(load_bucket::c8_plus) == 1);

    const auto c3_first = runtime.decide(context(1, 3));
    assert(c3_first.selected_action == action::model && c3_first.depth == 1);
    auto first = model_sample(1, 1, 1.20);
    first.decision_id = c3_first.decision_id;
    first.observed_depth = c3_first.depth;
    assert(runtime.observe(first));

    const auto c4_second = runtime.decide(context(2, 4));
    assert(c4_second.selected_action == action::model && c4_second.depth == 1);
    auto second = model_sample(2, 2, 1.20);
    second.decision_id = c4_second.decision_id;
    second.observed_depth = c4_second.depth;
    assert(runtime.observe(second));
    assert(runtime.get_model_state(load_bucket::c4) == model_state::active);
    assert(runtime.get_model_observations(load_bucket::c4) == 2);

    // Continued alternation cannot push the shared learner into an
    // unobservable depth-2 probe or reset its accumulated evidence.
    assert(observe_model(runtime, 3, 3, 1.20, 3));
    assert(observe_model(runtime, 4, 4, 1.20, 4));
    assert(runtime.get_model_state(load_bucket::c4) == model_state::active);
    const auto still_capped = runtime.decide(context(5, 3));
    assert(still_capped.depth == 1);
    assert(runtime.discard(still_capped.decision_id, 5));

    // C8+ is likewise conservative over every configured tail entry.
    const auto c8_tail = runtime.decide(context(5, 5));
    assert(c8_tail.selected_action == action::model && c8_tail.depth == 1);
    assert(runtime.discard(c8_tail.decision_id, 5));
}

static void test_periodic_target_refresh_and_bucket_isolation() {
    auto cfg = make_config();
    cfg.bootstrap_target_baseline = true;
    cfg.target_remeasure_interval = 2;
    cfg.promote_samples = 8;
    cfg.min_dwell_requests = 1;
    policy runtime(cfg);

    assert(observe_target(runtime, 1, 1, 1, 0, 100, 1'000));
    assert(std::abs(runtime.get_target_tokens_per_us(load_bucket::c1) - 0.1) < 1e-12);
    assert(runtime.get_target_tokens_per_us(load_bucket::c2) == 0.0);
    assert(observe_model(runtime, 2, 2, 1.20, 1));
    assert(observe_model(runtime, 3, 3, 1.20, 1));

    auto mixed = context(4, 1);
    mixed.allow_target_remeasure = false;
    const auto deferred = runtime.decide(mixed);
    assert(deferred.selected_action == action::model);
    assert(runtime.discard(deferred.decision_id, 4));

    const auto refresh = runtime.decide(context(5, 1));
    assert(refresh.selected_action == action::target);
    assert(refresh.why == reason::target_refresh);
    observation refreshed;
    refreshed.decision_id = refresh.decision_id;
    refreshed.request_epoch = 5;
    refreshed.round_epoch = 5;
    refreshed.observed_action = action::target;
    refreshed.elapsed_us = 1'000;
    refreshed.useful_tokens = 50;
    refreshed.synchronized = true;
    refreshed.complete = true;
    assert(runtime.observe(refreshed));
    assert(std::abs(runtime.get_target_tokens_per_us(load_bucket::c1) - 0.05) < 1e-12);
    assert(runtime.get_target_tokens_per_us(load_bucket::c2) == 0.0);

    const auto c2 = runtime.decide(context(6, 2));
    assert(c2.selected_action == action::target);
    assert(c2.why == reason::target_bootstrap);
    assert(c2.load_class == load_bucket::c2);
    assert(runtime.discard(c2.decision_id, 6));
}

static void test_short_produced_drafts_are_valid_samples() {
    assert(policy::valid_produced_depth(action::model, 1, 7));
    assert(policy::valid_produced_depth(action::model, 6, 7));
    assert(policy::valid_produced_depth(action::lookup, 2, 4));
    assert(!policy::valid_produced_depth(action::model, 0, 7));
    assert(!policy::valid_produced_depth(action::model, 8, 7));
    assert(!policy::valid_produced_depth(action::lookup, 0, 4));
    assert(policy::valid_produced_depth(action::target, 0, 0));
    assert(!policy::valid_produced_depth(action::target, 1, 0));
}

static void test_lookup_during_refresh_lockout_does_not_retrigger_refresh() {
    auto cfg = make_config();
    cfg.bootstrap_target_baseline = true;
    cfg.target_remeasure_interval = 1;
    cfg.promote_samples = 8;
    cfg.min_dwell_requests = 1;
    policy runtime(cfg);

    assert(observe_target(runtime, 1, 1, 1));
    assert(observe_model(runtime, 2, 2, 1.20, 1));

    const auto refresh = runtime.decide(context(3, 1));
    assert(refresh.selected_action == action::target && refresh.why == reason::target_refresh);
    observation target;
    target.decision_id = refresh.decision_id;
    target.request_epoch = 3;
    target.round_epoch = 3;
    target.observed_action = action::target;
    target.elapsed_us = 1'000;
    target.useful_tokens = 100;
    target.synchronized = true;
    target.complete = true;
    assert(runtime.observe(target));

    const auto locked_lookup = runtime.decide(context(3, 1));
    assert(locked_lookup.selected_action == action::lookup);
    assert(locked_lookup.why == reason::same_request_lockout);
    observation lookup;
    lookup.decision_id = locked_lookup.decision_id;
    lookup.request_epoch = 3;
    lookup.round_epoch = 4;
    lookup.observed_action = action::lookup;
    lookup.observed_depth = locked_lookup.depth;
    lookup.elapsed_us = 1'000;
    lookup.useful_tokens = 1;
    lookup.synchronized = true;
    lookup.complete = true;
    assert(runtime.observe(lookup));

    const auto next_request = runtime.decide(context(4, 1));
    assert(next_request.selected_action == action::model);
    assert(next_request.why != reason::target_refresh);
    assert(runtime.discard(next_request.decision_id, 4));
}

static void test_unattributed_late_observation_does_not_demote_c1() {
    auto cfg = make_config();
    cfg.load_depth_caps = { 3, 3, 3, 3 };
    policy runtime(cfg);

    const auto c4 = runtime.decide(context(1, 4));
    assert(c4.load_class == load_bucket::c4);
    assert(runtime.discard(c4.decision_id, 1));

    auto late = model_sample(2, 2, 1.20);
    late.decision_id = c4.decision_id + 100;
    late.observed_depth = 1;
    assert(!runtime.observe(late));
    assert(runtime.get_model_state(load_bucket::c1) == model_state::probing);
    assert(runtime.get_model_state(load_bucket::c4) == model_state::probing);
}

int main() {
    test_config_validation();
    test_probe_promote_and_load_cap();
    test_demotion_latches_current_request();
    test_deadband_does_not_flap();
    test_vram_guards();
    test_invalid_measurement_fails_closed();
    test_stale_model_state_is_target_only();
    test_lookup_feedback_and_causal_binding();
    test_depths_are_qualified_stepwise();
    test_hard_guard_latches_model_until_next_request();
    test_capped_depth_does_not_qualify_deeper_arm();
    test_pending_decision_is_not_overwritten_and_can_be_discarded();
    test_cold_start_target_bootstrap_precedes_model_probe();
    test_target_observation_bootstraps_internal_baseline();
    test_load_buckets_qualify_independently();
    test_target_bootstrap_is_load_bucket_local();
    test_one_aggregate_observation_per_c4_batch();
    test_group_epoch_ignores_shuffled_slot_reuse();
    test_discarded_partial_group_does_not_starve_next_clean_batch();
    test_group_depth_cap_is_hard();
    test_conservative_bucket_cap_survives_alternating_c3_c4();
    test_periodic_target_refresh_and_bucket_isolation();
    test_short_produced_drafts_are_valid_samples();
    test_lookup_during_refresh_lockout_does_not_retrigger_refresh();
    test_unattributed_late_observation_does_not_demote_c1();
    return 0;
}
