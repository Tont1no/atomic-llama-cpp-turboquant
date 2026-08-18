#include "common.h"
#include "speculative.h"

#include <cassert>
#include <cmath>
#include <limits>
#include <vector>

static void test_load_caps() {
    common_params_speculative_draft params;
    params.n_max = 7;
    assert(!params.adaptive);

    const std::vector<float> ema;
    const std::vector<uint64_t> offered;

    assert(common_speculative_adaptive_n_max(params, 1, 7, ema, offered) == 7);
    assert(common_speculative_adaptive_n_max(params, 2, 7, ema, offered) == 3);
    assert(common_speculative_adaptive_n_max(params, 3, 7, ema, offered) == 2);
    assert(common_speculative_adaptive_n_max(params, 4, 7, ema, offered) == 1);
    assert(common_speculative_adaptive_n_max(params, 8, 7, ema, offered) == 0);
    assert(common_speculative_adaptive_n_max(params, 9, 7, ema, offered) == 0);

    // A context/request cap between supported arms rounds down without exceeding it.
    assert(common_speculative_adaptive_n_max(params, 1, 6, ema, offered) == 3);
    assert(common_speculative_adaptive_n_max(params, 1, 2, ema, offered) == 2);

    params.n_min = 3;
    assert(common_speculative_adaptive_n_max(params, 4, 7, ema, offered) == 0);
}

static void test_per_sequence_limit() {
    assert(common_speculative_draft_n_max_for_seq(7, -1) == 7);
    assert(common_speculative_draft_n_max_for_seq(7,  0) == 0);
    assert(common_speculative_draft_n_max_for_seq(7,  2) == 2);
    assert(common_speculative_draft_n_max_for_seq(7, 12) == 7);
}

static void test_acceptance_ema() {
    common_params_speculative_draft params;
    params.n_max = 7;
    params.adaptive_ema_alpha = 0.5f;
    params.adaptive_ema_threshold = 0.55f;
    params.adaptive_ema_warmup = 8;

    std::vector<float> ema;
    std::vector<uint64_t> offered(7, 0);

    // Unknown positions do not constrain the load cap during warm-up.
    assert(common_speculative_adaptive_n_max(params, 1, 7, ema, offered) == 7);

    // A stable accepted prefix of two tokens reduces the safe arm from seven to two.
    for (int i = 0; i < 8; ++i) {
        common_speculative_adaptive_acceptance_update(ema, 7, 2, params.adaptive_ema_alpha);
        for (auto & n : offered) {
            n++;
        }
    }
    assert(ema.size() == 7);
    assert(ema[0] == 1.0f && ema[1] == 1.0f && ema[2] == 0.0f);
    assert(common_speculative_adaptive_n_max(params, 1, 7, ema, offered) == 2);

    // EMA adaptation is optional; alpha zero restores the load-only decision.
    params.adaptive_ema_alpha = 0.0f;
    assert(common_speculative_adaptive_n_max(params, 1, 7, ema, offered) == 7);
}

static void test_fail_closed() {
    common_params_speculative_draft params;
    const std::vector<float> ema;
    const std::vector<uint64_t> offered;

    params.adaptive_load_caps.clear();
    assert(common_speculative_adaptive_n_max(params, 1, 7, ema, offered) == 0);

    params.adaptive_load_caps = { 4 }; // not a supported arm
    assert(common_speculative_adaptive_n_max(params, 1, 7, ema, offered) == 0);

    params.adaptive_load_caps = { 7 };
    params.adaptive_ema_alpha = std::numeric_limits<float>::quiet_NaN();
    assert(common_speculative_adaptive_n_max(params, 1, 7, ema, offered) == 0);

    params.adaptive_ema_alpha = 0.5f;
    params.adaptive_ema_warmup = 0;
    assert(common_speculative_adaptive_n_max(params, 1, 7, ema, offered) == 0);
}

int main() {
    test_per_sequence_limit();
    test_load_caps();
    test_acceptance_ema();
    test_fail_closed();
    return 0;
}
