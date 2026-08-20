#include "llama-memory-recurrent.h"
#include "common.h"
#include "speculative.h"

#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <vector>

struct batch_fixture {
    llama_batch batch = {};

    std::vector<int32_t> n_seq_id;
    std::vector<llama_seq_id *> seq_id;
    std::vector<llama_seq_id> seq_id_data;
    std::vector<size_t> offsets;
    std::vector<uint32_t> rs_depth;

    batch_fixture(
            std::initializer_list<std::initializer_list<llama_seq_id>> rows,
            std::initializer_list<uint32_t> depths) : rs_depth(depths) {
        assert(rows.size() == depths.size());
        n_seq_id.reserve(rows.size());
        offsets.reserve(rows.size());
        for (const auto & row : rows) {
            offsets.push_back(seq_id_data.size());
            n_seq_id.push_back((int32_t) row.size());
            seq_id_data.insert(seq_id_data.end(), row.begin(), row.end());
        }

        seq_id.reserve(rows.size());
        for (size_t offset : offsets) {
            seq_id.push_back(seq_id_data.data() + offset);
        }

        batch.n_tokens = (int32_t) rows.size();
        batch.n_seq_id = n_seq_id.data();
        batch.seq_id   = seq_id.data();
        batch.rs_depth = rs_depth.data();
    }
};

static void expect_depth(
        std::initializer_list<std::initializer_list<llama_seq_id>> rows,
        std::initializer_list<uint32_t> depths,
        uint32_t configured,
        uint32_t n_seq_max,
        uint32_t expected) {
    const batch_fixture fixture(rows, depths);
    bool fallback = true;
    const uint32_t actual = llama_recurrent_batch_active_rs_depth(
            fixture.batch, configured, n_seq_max, &fallback);
    assert(!fallback);
    assert(actual == expected);
}

static void expect_packed_dspark_depth(
        const std::vector<uint32_t> & caps,
        uint32_t configured,
        uint32_t expected) {
    std::vector<int32_t> n_seq_id;
    std::vector<llama_seq_id> seq_id_data;
    std::vector<llama_seq_id *> seq_id;
    std::vector<uint32_t> depths;

    size_t n_rows = 0;
    for (const uint32_t cap : caps) {
        n_rows += 1 + cap; // sampled anchor plus the selected DSpark prefix
    }
    n_seq_id.reserve(n_rows);
    seq_id_data.reserve(n_rows);
    depths.reserve(n_rows);

    for (size_t seq = 0; seq < caps.size(); ++seq) {
        for (uint32_t row = 0; row < 1 + caps[seq]; ++row) {
            n_seq_id.push_back(1);
            seq_id_data.push_back((llama_seq_id) seq);
            depths.push_back(caps[seq]);
        }
    }
    seq_id.reserve(n_rows);
    for (size_t row = 0; row < n_rows; ++row) {
        seq_id.push_back(seq_id_data.data() + row);
    }

    llama_batch batch = {};
    batch.n_tokens = (int32_t) n_rows;
    batch.n_seq_id = n_seq_id.data();
    batch.seq_id   = seq_id.data();
    batch.rs_depth = depths.data();

    bool fallback = true;
    const uint32_t actual = llama_recurrent_batch_active_rs_depth(
            batch, configured, (uint32_t) caps.size(), &fallback);
    assert(!fallback);
    assert(actual == expected);
}

int main() {
    // Once explicitly enabled, dynamic recurrent snapshots are an execution-
    // layout property of the pure packed DSpark path, independent of whether
    // its current cap comes from fixed drafting, SPS planning, shadow planning,
    // or the recorder. DFlash retains its adaptive-only opt-in and mixed chains
    // fail closed.
    {
        common_params dspark;
        dspark.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK };
        dspark.speculative.draft.n_max = 7;
        assert(common_speculative_is_only_dspark(dspark.speculative.types));
        assert(!common_speculative_uses_dynamic_rs(dspark.speculative));
        assert(!common_context_params_to_llama(dspark).rs_seq_dynamic);
        dspark.speculative.draft.dynamic_rs = true;
        assert(common_speculative_uses_dynamic_rs(dspark.speculative));
        auto cparams = common_context_params_to_llama(dspark);
        assert(cparams.rs_seq_dynamic);
        assert(cparams.n_rs_seq == 7);

        dspark.speculative.draft.dynamic_rs = false;
        assert(!common_speculative_uses_dynamic_rs(dspark.speculative));
        assert(!common_context_params_to_llama(dspark).rs_seq_dynamic);
        dspark.speculative.draft.dynamic_rs = true;

        dspark.speculative.draft.sps_profile = "sps-v2.json";
        assert(common_speculative_uses_dynamic_rs(dspark.speculative));
        dspark.speculative.draft.sps_shadow = true;
        assert(common_speculative_uses_dynamic_rs(dspark.speculative));
        dspark.speculative.draft.sps_profile.clear();
        dspark.speculative.draft.sps_shadow = false;
        dspark.speculative.draft.sps_record = "samples.json";
        dspark.speculative.draft.sps_force_verify_rows = 0;
        assert(common_speculative_uses_dynamic_rs(dspark.speculative));

        dspark.speculative.draft.adaptive = true;
        assert(common_speculative_sps_conflicts_with_adaptive(dspark.speculative));
        assert(!common_speculative_uses_dynamic_rs(dspark.speculative));
        dspark.speculative.draft.sps_record.clear();
        assert(!common_speculative_sps_conflicts_with_adaptive(dspark.speculative));
        assert(common_speculative_uses_dynamic_rs(dspark.speculative));

        common_params dflash;
        dflash.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH };
        assert(!common_speculative_uses_dynamic_rs(dflash.speculative));
        dflash.speculative.draft.adaptive = true;
        assert(!dflash.speculative.draft.dynamic_rs);
        assert(common_speculative_uses_dynamic_rs(dflash.speculative));

        common_params mixed;
        mixed.speculative.types = {
            COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK,
            COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE,
        };
        mixed.speculative.draft.adaptive = true;
        assert(!common_speculative_is_only_dspark(mixed.speculative.types));
        assert(!common_speculative_uses_dynamic_rs(mixed.speculative));

        common_params duplicate;
        duplicate.speculative.types = {
            COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK,
            COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK,
        };
        assert(!common_speculative_is_only_dspark(duplicate.speculative.types));
        assert(!common_speculative_uses_dynamic_rs(duplicate.speculative));
    }

    // cap=0 target-only iteration: one row per active sequence executes the
    // exact zero-snapshot graph even though the configured ceiling is seven.
    expect_depth({ { 0 }, { 1 }, { 2 }, { 3 } }, { 0, 0, 0, 0 }, 7, 4, 0);

    expect_depth({ { 0 } },                      { 0 },          7, 1, 0);
    expect_depth({ { 0 }, { 0 } },               { 1, 1 },       7, 1, 1);
    expect_depth({ { 0 }, { 0 }, { 0 } },        { 2, 2, 2 },    7, 1, 2);
    expect_depth({ { 0 }, { 0 }, { 0 }, { 0 } }, { 3, 3, 3, 3 }, 7, 1, 3);

    // SPS/packed server metadata matrix: C1/C2/C4/C8 at every qualified cap.
    // Physical storage follows the global maximum cap, while each source run
    // retains its own exact depth tag.
    for (const uint32_t slots : { 1u, 2u, 4u, 8u }) {
        for (const uint32_t cap : { 0u, 1u, 2u, 3u, 7u }) {
            expect_packed_dspark_depth(std::vector<uint32_t>(slots, cap), 7, cap);
        }
    }
    expect_packed_dspark_depth({ 0, 7 }, 7, 7);
    expect_packed_dspark_depth({ 0, 1, 2, 3 }, 7, 3);
    expect_packed_dspark_depth({ 7, 0, 3, 1, 2, 7, 0, 1 }, 7, 7);

    // A dynamic rollback must never select a stale or no-longer-resident
    // plane. Fixed contexts retain their historical configured-depth rule.
    assert( llama_recurrent_rollback_is_valid(true,  3, 7, 3, 3));
    assert(!llama_recurrent_rollback_is_valid(true,  3, 7, 2, 3));
    assert(!llama_recurrent_rollback_is_valid(true,  3, 7, 3, 2));
    assert(!llama_recurrent_rollback_is_valid(true,  1, 7, 0, 7));
    assert(!llama_recurrent_rollback_is_valid(true,  0, 7, 7, 7));
    assert(!llama_recurrent_rollback_is_valid(true,  8, 7, 7, 7));
    assert( llama_recurrent_rollback_is_valid(false, 3, 7, 0, 0));

    // Ragged verification: seq 0 has one row, seq 1 has two, seq 2 has four.
    expect_depth(
            { { 0 }, { 1 }, { 1 }, { 2 }, { 2 }, { 2 }, { 2 } },
            { 0, 1, 1, 3, 3, 3, 3 }, 7, 3, 3);

    // Prompt row count is unrelated to rollback depth. Eight ordinary prompt
    // rows stay at depth zero, while a mixed prompt+verify batch selects only
    // the explicitly tagged verification span.
    expect_depth(
            { { 0 }, { 0 }, { 0 }, { 0 }, { 0 }, { 0 }, { 0 }, { 0 } },
            { 0, 0, 0, 0, 0, 0, 0, 0 }, 7, 2, 0);
    expect_depth(
            { { 0 }, { 0 }, { 0 }, { 0 }, { 1 }, { 1 }, { 1 }, { 1 } },
            { 0, 0, 0, 0, 3, 3, 3, 3 }, 7, 2, 3);

    // Coupled sequence sets count one row for every participating sequence.
    expect_depth({ { 0, 1 }, { 0, 1 }, { 0, 1 } }, { 2, 2, 2 }, 7, 2, 2);

    // Malformed metadata must never lower the configured safety bound.
    {
        batch_fixture invalid({ { 0, 0 } }, { 0 });
        bool fallback = false;
        assert(llama_recurrent_batch_active_rs_depth(invalid.batch, 7, 1, &fallback) == 7);
        assert(fallback);
    }
    {
        batch_fixture invalid({ { 2 } }, { 0 });
        bool fallback = false;
        assert(llama_recurrent_batch_active_rs_depth(invalid.batch, 7, 2, &fallback) == 7);
        assert(fallback);
    }
    {
        llama_batch invalid = {};
        invalid.n_tokens = 1;
        bool fallback = false;
        assert(llama_recurrent_batch_active_rs_depth(invalid, 7, 1, &fallback) == 7);
        assert(fallback);
    }
    {
        batch_fixture external({ { 0 }, { 0 } }, { 0, 0 });
        external.batch.rs_depth = nullptr;
        bool fallback = false;
        assert(llama_recurrent_batch_active_rs_depth(external.batch, 7, 1, &fallback) == 7);
        assert(fallback);
    }
    {
        batch_fixture invalid({ { 0 } }, { 8 });
        bool fallback = false;
        assert(llama_recurrent_batch_active_rs_depth(invalid.batch, 7, 1, &fallback) == 7);
        assert(fallback);
    }

    return 0;
}
