#include "llama-memory-recurrent.h"

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

    batch_fixture(std::initializer_list<std::initializer_list<llama_seq_id>> rows) {
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
    }
};

static void expect_depth(const batch_fixture & fixture, uint32_t configured, uint32_t n_seq_max, uint32_t expected) {
    bool fallback = true;
    const uint32_t actual = llama_recurrent_batch_active_rs_depth(
            fixture.batch, configured, n_seq_max, &fallback);
    assert(!fallback);
    assert(actual == expected);
}

int main() {
    // cap=0 target-only iteration: one row per active sequence executes the
    // exact zero-snapshot graph even though seven planes remain allocated.
    expect_depth({ { 0 }, { 1 }, { 2 }, { 3 } }, 7, 4, 0);

    expect_depth({ { 0 } },                         7, 1, 0);
    expect_depth({ { 0 }, { 0 } },                  7, 1, 1);
    expect_depth({ { 0 }, { 0 }, { 0 } },           7, 1, 2);
    expect_depth({ { 0 }, { 0 }, { 0 }, { 0 } },    7, 1, 3);

    // Ragged verification: seq 0 has one row, seq 1 has two, seq 2 has four.
    expect_depth({ { 0 }, { 1 }, { 1 }, { 2 }, { 2 }, { 2 }, { 2 } }, 7, 3, 3);
    expect_depth({ { 0 }, { 1 }, { 1 }, { 2 }, { 2 }, { 2 }, { 2 } }, 2, 3, 2);

    // Coupled sequence sets count one row for every participating sequence.
    expect_depth({ { 0, 1 }, { 0, 1 }, { 0, 1 } }, 7, 2, 2);

    // Malformed metadata must never lower the configured safety bound.
    {
        batch_fixture invalid = { { 0, 0 } };
        bool fallback = false;
        assert(llama_recurrent_batch_active_rs_depth(invalid.batch, 7, 1, &fallback) == 7);
        assert(fallback);
    }
    {
        batch_fixture invalid = { { 2 } };
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

    return 0;
}
