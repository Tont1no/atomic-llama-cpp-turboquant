#include "ggml.h"
#include "speculative.h"
#include "suffix-draft.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

static common_suffix_draft_config test_config() {
    common_suffix_draft_config config;
    config.match_min = 3;
    config.match_max = 8;
    config.draft_min = 2;
    config.draft_max = 6;
    config.max_candidates = 16;
    config.min_confidence = 0.5f;
    config.acceptance_alpha = 1.0f;
    return config;
}

static void test_repeated_edit_and_variable_suffix() {
    common_suffix_drafter drafter(test_config());

    llama_tokens short_prompt = { 10, 11, 12, 13, 14, 15, 16, 99, 10, 11 };
    drafter.begin(short_prompt);
    const llama_tokens short_draft = drafter.draft(short_prompt, 12);
    assert((short_draft == llama_tokens { 13, 14, 15, 16 }));
    assert(drafter.metrics().last_match == 3);

    llama_tokens long_prompt = { 90, 91, 10, 11, 12, 13, 14, 15, 16, 77, 90, 91, 10, 11 };
    drafter.begin(long_prompt);
    const llama_tokens long_draft = drafter.draft(long_prompt, 12);
    assert(long_draft.size() == 5);
    assert((llama_tokens(long_draft.begin(), long_draft.begin() + 4) == llama_tokens { 13, 14, 15, 16 }));
    assert(drafter.metrics().last_match == 5);
    assert(drafter.metrics().hits == 2);
}

static void test_weak_match_returns_empty_for_model_fallback() {
    common_suffix_drafter drafter(test_config());
    llama_tokens prompt = { 1, 2, 3, 4, 8, 1, 2, 3, 5, 9, 1, 2 };
    drafter.begin(prompt);

    assert(drafter.draft(prompt, 3).empty());
    assert(drafter.metrics().weak_misses == 1);
    assert(common_speculative_route_allows_source(
            COMMON_SPECULATIVE_ROUTE_LOOKUP_THEN_MODEL,
            COMMON_SPECULATIVE_TYPE_SUFFIX_SESSION));
    assert(common_speculative_route_allows_source(
            COMMON_SPECULATIVE_ROUTE_LOOKUP_THEN_MODEL,
            COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH));
    assert(!common_speculative_route_allows_source(
            COMMON_SPECULATIVE_ROUTE_MODEL_ONLY,
            COMMON_SPECULATIVE_TYPE_SUFFIX_SESSION));
}

static void test_runtime_chain_continues_after_suffix_miss() {
    common_params_speculative params;
    params.types = {
        COMMON_SPECULATIVE_TYPE_SUFFIX_SESSION,
        COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE,
    };
    params.suffix_session.match_min = 3;
    params.suffix_session.match_max = 8;
    params.suffix_session.draft_min = 2;
    params.suffix_session.draft_max = 6;
    params.suffix_session.min_confidence = 0.5f;
    params.ngram_simple.size_n = 3;
    params.ngram_simple.size_m = 3;

    common_speculative_ptr spec(common_speculative_init(params, 1));
    assert(spec);

    llama_tokens prompt = { 1, 2, 3, 4, 8, 1, 2, 3, 5, 9, 1, 2 };
    llama_tokens result;
    common_speculative_begin(spec.get(), 0, prompt);
    auto & draft = common_speculative_get_draft_params(spec.get(), 0);
    draft.drafting = true;
    draft.n_max = 6;
    draft.id_last = 3;
    draft.prompt = &prompt;
    draft.result = &result;
    common_speculative_draft(spec.get());

    assert(!result.empty());
    assert(common_speculative_get_last_type(spec.get(), 0) == COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE);
}

static void test_adaptive_length() {
    common_suffix_drafter drafter(test_config());
    llama_tokens prompt = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 99, 1, 2, 3, 4, 5 };

    drafter.begin(prompt);
    const llama_tokens full = drafter.draft(prompt, 6);
    assert(full.size() == 6);
    drafter.accept(0);

    const llama_tokens reduced = drafter.draft(prompt, 6);
    assert(reduced.size() == 2);
    drafter.accept(2);

    assert(drafter.draft(prompt, 6).size() == 6);
}

static void test_equal_votes_use_stable_token_order() {
    auto config = test_config();
    config.min_confidence = 0.2f;
    common_suffix_drafter drafter(config);
    llama_tokens prompt = { 1, 2, 3, 9, 1, 2, 3, 8, 1, 2 };

    drafter.begin(prompt);
    const llama_tokens result = drafter.draft(prompt, 3);
    assert(!result.empty());
    assert(result.front() == 8);
}

static void test_scope_reset_and_sequence_isolation() {
    common_suffix_drafter first(test_config());
    common_suffix_drafter second(test_config());
    llama_tokens repeated = { 1, 2, 3, 4, 5, 6, 9, 1, 2 };
    llama_tokens isolated = { 20, 21, 22, 23 };

    first.begin(repeated);
    second.begin(isolated);
    assert(!first.draft(repeated, 3).empty());
    assert(second.draft(isolated, 24).empty());
    assert(second.metrics().hits == 0);

    first.reset();
    assert(first.history_size() == 0);
    first.begin(isolated);
    assert(first.draft(isolated, 24).empty());
    assert(first.metrics().hits == 0);
    assert(first.metrics().lookups == 1);
}

static void test_edited_request_keeps_previous_session_version() {
    common_suffix_drafter drafter(test_config());
    llama_tokens old_version = { 50, 1, 2, 3, 4, 5 };
    llama_tokens edited = { 60, 1, 2 };

    drafter.begin(old_version);
    drafter.begin(edited);
    const llama_tokens result = drafter.draft(edited, 3);
    assert((result == llama_tokens { 4, 5 }));
}

static void test_session_history_is_bounded() {
    auto config = test_config();
    config.history_max = 8;
    common_suffix_drafter drafter(config);

    drafter.begin({ 1, 2, 3, 4, 5, 6 });
    drafter.begin({ 7, 8, 9, 10, 11, 12 });
    assert(drafter.history_size() <= 6 + config.history_max);
    drafter.begin({ 13, 14, 15, 16, 17, 18 });
    assert(drafter.history_size() == 6);

    llama_tokens long_prompt = { 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 };
    drafter.begin(long_prompt);
    assert(drafter.history_size() <= long_prompt.size() + config.history_max);
}

static void test_opt_in_name() {
    const auto types = common_speculative_types_from_names({ "suffix-session", "draft-dflash" });
    assert((types == std::vector<common_speculative_type> {
            COMMON_SPECULATIVE_TYPE_SUFFIX_SESSION,
            COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH,
    }));
    assert(common_speculative_type_to_str(COMMON_SPECULATIVE_TYPE_SUFFIX_SESSION) == "suffix-session");
}

int main() {
    ggml_time_init();
    test_repeated_edit_and_variable_suffix();
    test_weak_match_returns_empty_for_model_fallback();
    test_runtime_chain_continues_after_suffix_miss();
    test_adaptive_length();
    test_equal_votes_use_stable_token_order();
    test_scope_reset_and_sequence_isolation();
    test_edited_request_keeps_previous_session_version();
    test_session_history_is_bounded();
    test_opt_in_name();
    return 0;
}
