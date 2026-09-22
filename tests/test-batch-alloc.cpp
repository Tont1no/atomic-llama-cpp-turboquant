#include "testing.h"

#include "llama.h"

#include "../src/llama-batch.h"
#include "../src/llama-memory.h"
#include "../src/llama-vocab.h"
#include "../src/llama-row-reorder.h"

#include <cstdlib>
#include <initializer_list>
#include <map>
#include <string>
#include <utility>
#include <vector>

// mock memory that only provides per-sequence position ranges
struct mock_memory : public llama_memory_i {
    std::map<llama_seq_id, std::pair<llama_pos, llama_pos>> ranges; // seq_id -> [pos_min, pos_max]

    llama_memory_context_ptr init_batch(llama_batch_allocr &, uint32_t, bool) override {  GGML_ASSERT(false && "not implemented"); }
    llama_memory_context_ptr init_full() override {  GGML_ASSERT(false && "not implemented"); }
    llama_memory_context_ptr init_update(llama_context *, bool) override { GGML_ASSERT(false && "not implemented"); }

    bool get_can_shift() const override { GGML_ASSERT(false && "not implemented"); }

    void clear(bool) override { GGML_ASSERT(false && "not implemented"); }

    bool seq_rm  (llama_seq_id, llama_pos, llama_pos) override { GGML_ASSERT(false && "not implemented"); }
    void seq_cp  (llama_seq_id, llama_seq_id, llama_pos, llama_pos) override { GGML_ASSERT(false && "not implemented"); }
    void seq_keep(llama_seq_id) override { GGML_ASSERT(false && "not implemented"); }
    void seq_add (llama_seq_id, llama_pos, llama_pos, llama_pos) override { GGML_ASSERT(false && "not implemented"); }
    void seq_div (llama_seq_id, llama_pos, llama_pos, int) override { GGML_ASSERT(false && "not implemented");  }

    llama_pos seq_pos_min(llama_seq_id seq_id) const override {
        auto it = ranges.find(seq_id);
        return it == ranges.end() ? -1 : it->second.first;
    }

    llama_pos seq_pos_max(llama_seq_id seq_id) const override {
        auto it = ranges.find(seq_id);
        return it == ranges.end() ? -1 : it->second.second;
    }

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override { return {}; }

    void state_write(llama_io_write_i &, llama_seq_id, llama_state_seq_flags) const override { GGML_ASSERT(false && "not implemented"); }
    void state_read (llama_io_read_i &,  llama_seq_id, llama_state_seq_flags) override { GGML_ASSERT(false && "not implemented"); }
};

// builds embedding batches - an empty llama_vocab rejects all token ids, so
// the tests use embeddings everywhere except the token validation tests
struct batch_builder {
    uint32_t n_embd;

    std::vector<float>     embd;
    std::vector<llama_pos> pos;
    std::vector<int32_t>   n_seq_id;
    std::vector<int8_t>    logits;

    std::vector<std::vector<llama_seq_id>> seq;
    std::vector<llama_seq_id *>            seq_ptr;

    batch_builder(uint32_t n_embd = 2) : n_embd(n_embd) {}

    // embd values are 100*i + k so that ubatch contents can be traced back to batch indices
    void add(llama_pos p, std::initializer_list<llama_seq_id> seq_ids, bool output) {
        const int32_t i = (int32_t) seq.size();
        for (uint32_t k = 0; k < n_embd; ++k) {
            embd.push_back(100.0f*i + k);
        }
        pos.push_back(p);
        n_seq_id.push_back((int32_t) seq_ids.size());
        seq.emplace_back(seq_ids);
        logits.push_back(output ? 1 : 0);
    }

    llama_batch make(bool with_pos = true, bool with_seq = true, bool with_logits = true) {
        seq_ptr.clear();
        for (auto & s : seq) {
            seq_ptr.push_back(s.data());
        }
        seq_ptr.push_back(nullptr);

        llama_batch res = {};
        res.n_tokens = (int32_t) seq.size();
        res.embd     = embd.data();
        res.pos      = with_pos    ? pos.data()      : nullptr;
        res.n_seq_id = with_seq    ? n_seq_id.data() : nullptr;
        res.seq_id   = with_seq    ? seq_ptr.data()  : nullptr;
        res.logits   = with_logits ? logits.data()   : nullptr;

        return res;
    }
};

static void test_init(testing & t) {
    llama_vocab vocab;

    t.test("rejects_n_seq_max_too_large", [&](testing & t) {
        batch_builder bb;
        bb.add(0, {0}, true);

        llama_batch_allocr ba(1);
        t.assert_true(!ba.init(bb.make(), vocab, nullptr, bb.n_embd, LLAMA_MAX_SEQ + 1, false));
    });

    t.test("rejects_invalid_token", [&](testing & t) {
        llama_token tok = 0; // empty vocab -> every token id is out of range
        llama_batch batch = llama_batch_get_one(&tok, 1);

        llama_batch_allocr ba(1);
        t.assert_true("token id >= n_tokens", !ba.init(batch, vocab, nullptr, 0, 1, false));

        tok = -1;
        t.assert_true("negative token id", !ba.init(batch, vocab, nullptr, 0, 1, false));
    });

    t.test("rejects_invalid_seq_id", [&](testing & t) {
        llama_batch_allocr ba(1);

        {
            batch_builder bb;
            bb.add(0, {4}, true);
            t.assert_true("seq_id >= n_seq_max", !ba.init(bb.make(), vocab, nullptr, bb.n_embd, 4, false));
        }
        {
            batch_builder bb;
            bb.add(0, {-1}, true);
            t.assert_true("negative seq_id", !ba.init(bb.make(), vocab, nullptr, bb.n_embd, 4, false));
        }
    });

    t.test("autofill_defaults", [&](testing & t) {
        batch_builder bb;
        for (int i = 0; i < 4; ++i) {
            bb.add(0, {0}, false);
        }

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(bb.make(false, false, false), vocab, nullptr, bb.n_embd, 4, false));

        const llama_batch & batch = ba.get_batch();

        t.assert_equal(4u, ba.get_n_tokens());

        for (int i = 0; i < 4; ++i) {
            t.assert_equal("pos defaults to 0..n-1", i, batch.pos[i]);
            t.assert_equal("n_seq_id defaults to 1", 1, batch.n_seq_id[i]);
            t.assert_equal("seq_id defaults to 0",   0, batch.seq_id[i][0]);
        }

        t.assert_equal("only the last token is an output", 1u, ba.get_n_outputs());
        t.assert_equal(0, (int) batch.logits[0]);
        t.assert_equal(1, (int) batch.logits[3]);

        t.assert_equal(0, ba.seq_pos_min(0));
        t.assert_equal(3, ba.seq_pos_max(0));
        t.assert_equal(-1, ba.seq_pos_min(1));
    });

    t.test("output_all", [&](testing & t) {
        batch_builder bb;
        for (int i = 0; i < 4; ++i) {
            bb.add(i, {0}, false);
        }

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(bb.make(true, true, false), vocab, nullptr, bb.n_embd, 4, true));
        t.assert_equal(4u, ba.get_n_outputs());
    });

    t.test("explicit_logits", [&](testing & t) {
        batch_builder bb;
        bb.add(0, {0}, true);
        bb.add(1, {0}, false);
        bb.add(2, {0}, true);

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(bb.make(), vocab, nullptr, bb.n_embd, 4, false));
        t.assert_equal(2u, ba.get_n_outputs());

        llama_ubatch ub = ba.split_simple(10);
        t.assert_equal(3u, ub.n_tokens);
        t.assert_equal(1, (int) ub.output[0]);
        t.assert_equal(0, (int) ub.output[1]);
        t.assert_equal(1, (int) ub.output[2]);

        const auto & out_ids = ba.get_out_ids();
        t.assert_equal((size_t) 2, out_ids.size());
        t.assert_equal(0, out_ids[0]);
        t.assert_equal(2, out_ids[1]);
    });

    t.test("pos_from_memory", [&](testing & t) {
        mock_memory mem;
        mem.ranges[0] = {0, 9};

        batch_builder bb;
        for (int i = 0; i < 3; ++i) {
            bb.add(0, {0}, false);
        }

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(bb.make(false, true, false), vocab, &mem, bb.n_embd, 4, false));

        t.assert_equal("pos continues after memory", 10, ba.seq_pos_min(0));
        t.assert_equal(12, ba.seq_pos_max(0));
    });

    t.test("pos_continuity_with_memory", [&](testing & t) {
        mock_memory mem;
        mem.ranges[0] = {0, 9};

        llama_batch_allocr ba(1);

        {
            batch_builder bb;
            bb.add(10, {0}, false);
            bb.add(11, {0}, true);
            t.assert_true("pos_max + 1 is accepted", ba.init(bb.make(), vocab, &mem, bb.n_embd, 4, false));
        }
        {
            batch_builder bb;
            bb.add(11, {0}, false);
            bb.add(12, {0}, true);
            t.assert_true("gap after memory is rejected", !ba.init(bb.make(), vocab, &mem, bb.n_embd, 4, false));
        }
        {
            batch_builder bb;
            bb.add(9, {0}, false);
            bb.add(10, {0}, true);
            t.assert_true("overlap with memory is rejected", !ba.init(bb.make(), vocab, &mem, bb.n_embd, 4, false));
        }
    });

    t.test("rejects_non_continuous_positions", [&](testing & t) {
        batch_builder bb;
        bb.add(0, {0}, false);
        bb.add(1, {0}, false);
        bb.add(3, {0}, true);

        llama_batch_allocr ba(1);
        t.assert_true(!ba.init(bb.make(), vocab, nullptr, bb.n_embd, 4, false));
    });

    t.test("rejects_decreasing_positions", [&](testing & t) {
        batch_builder bb;
        const llama_pos    pos[7] = {4, 5, 0, 1, 6, 2, 3};
        const llama_seq_id seq[7] = {0, 0, 1, 1, 0, 1, 0};
        for (int i = 0; i < 7; ++i) {
            bb.add(pos[i], {seq[i]}, false);
        }
        // seq 0 sees positions 4,5,6,3 in batch order -> the trailing 3 decreases

        llama_batch_allocr ba(1);
        t.assert_true(!ba.init(bb.make(true, true, false), vocab, nullptr, bb.n_embd, 4, false));
    });

    t.test("allows_equal_positions_in_seq", [&](testing & t) {
        batch_builder bb;
        bb.add(0, {0}, false);
        bb.add(0, {0}, false);
        bb.add(1, {0}, true);

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(bb.make(true, true, false), vocab, nullptr, bb.n_embd, 4, false));
    });


    t.test("rejects_coupled_diverged_seqs", [&](testing & t) {
        batch_builder bb;
        bb.add(6, {0, 1}, true);

        llama_batch_allocr ba(1);

        mock_memory mem;
        mem.ranges[0] = {0, 5};
        mem.ranges[1] = {2, 5}; // same pos_max, different pos_min -> diverged
        t.assert_true(!ba.init(bb.make(), vocab, &mem, bb.n_embd, 4, false));

        mem.ranges[1] = {0, 5};
        t.assert_true(ba.init(bb.make(), vocab, &mem, bb.n_embd, 4, false));
    });
}

static void test_split(testing & t) {
    llama_vocab vocab;

    t.test("split_simple_chunks", [&](testing & t) {
        batch_builder bb;
        for (int i = 0; i < 5; ++i) {
            bb.add(i, {0}, i == 4);
        }

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(bb.make(), vocab, nullptr, bb.n_embd, 4, false));

        llama_ubatch ub = ba.split_simple(2);
        t.assert_equal(2u, ub.n_tokens);
        t.assert_true(!ub.equal_seqs());
        t.assert_equal(1u, ub.n_seqs_unq);
        t.assert_equal(0, ub.seq_id_unq[0]);
        t.assert_equal(0, ub.seq_idx[0]);
        for (int i = 0; i < 2; ++i) {
            t.assert_equal(i, ub.pos[i]);
            t.assert_equal(1, ub.n_seq_id[i]);
            t.assert_equal(0, ub.seq_id[i][0]);
            t.assert_equal(100.0f*i, ub.embd[i*bb.n_embd]);
            t.assert_equal(100.0f*i + 1, ub.embd[i*bb.n_embd + 1]);
        }

        ub = ba.split_simple(2);
        t.assert_equal(2u, ub.n_tokens);
        t.assert_equal(2, ub.pos[0]);
        t.assert_equal(3, ub.pos[1]);

        ub = ba.split_simple(2);
        t.assert_equal(1u, ub.n_tokens);
        t.assert_equal(4, ub.pos[0]);
        t.assert_equal(1, (int) ub.output[0]);

        t.assert_equal(5u, ba.get_n_used());

        ub = ba.split_simple(2);
        t.assert_equal("batch is consumed", 0u, ub.n_tokens);

        const auto & out_ids = ba.get_out_ids();
        t.assert_equal((size_t) 1, out_ids.size());
        t.assert_equal(4, out_ids[0]);
    });

    t.test("split_reset_allows_resplit", [&](testing & t) {
        batch_builder bb;
        for (int i = 0; i < 3; ++i) {
            bb.add(i, {0}, i == 2);
        }

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(bb.make(), vocab, nullptr, bb.n_embd, 4, false));

        while (ba.split_simple(1).n_tokens > 0) {
        }
        t.assert_equal(3u, ba.get_n_used());

        ba.split_reset();
        t.assert_equal(0u, ba.get_n_used());

        llama_ubatch ub = ba.split_simple(10);
        t.assert_equal(3u, ub.n_tokens);
    });

    t.test("split_equal_unequal_lengths", [&](testing & t) {
        batch_builder bb;
        for (int i = 0; i < 4; ++i) {
            bb.add(i, {0}, i == 3);
        }
        for (int i = 0; i < 2; ++i) {
            bb.add(i, {1}, i == 1);
        }

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(bb.make(), vocab, nullptr, bb.n_embd, 4, false));

        llama_ubatch ub = ba.split_equal(8, false, 0);
        t.assert_true(ub.equal_seqs());
        t.assert_equal("both seqs advance by the shorter length", 4u, ub.n_tokens);
        t.assert_equal(2u, ub.n_seq_tokens);
        t.assert_equal(2u, ub.n_seqs);
        t.assert_equal(2u, ub.n_seqs_unq);
        // tokens are grouped per sequence set: [s0 s0 s1 s1]
        t.assert_equal(0, ub.seq_id[0][0]);
        t.assert_equal(0, ub.seq_id[1][0]);
        t.assert_equal(1, ub.seq_id[2][0]);
        t.assert_equal(1, ub.seq_id[3][0]);
        t.assert_equal(0, ub.pos[0]);
        t.assert_equal(1, ub.pos[1]);
        t.assert_equal(0, ub.pos[2]);
        t.assert_equal(1, ub.pos[3]);

        ub = ba.split_equal(8, false, 0);
        t.assert_equal("only seq 0 remains", 2u, ub.n_tokens);
        t.assert_equal(1u, ub.n_seqs);
        t.assert_equal(2, ub.pos[0]);
        t.assert_equal(3, ub.pos[1]);

        ub = ba.split_equal(8, false, 0);
        t.assert_equal(0u, ub.n_tokens);

        t.assert_equal(6u, ba.get_n_used());
    });

    t.test("split_equal_coupled", [&](testing & t) {
        batch_builder bb;
        bb.add(0, {0, 1}, false);
        bb.add(1, {0, 1}, true);

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(bb.make(), vocab, nullptr, bb.n_embd, 4, false));

        llama_ubatch ub = ba.split_equal(4, true, 0);
        t.assert_equal("sequential split rejects coupled seqs", 0u, ub.n_tokens);

        ub = ba.split_equal(4, false, 0);
        t.assert_equal(2u, ub.n_tokens);
        t.assert_equal("one sequence set", 1u, ub.n_seqs);
        t.assert_equal("two unique seq ids", 2u, ub.n_seqs_unq);
        t.assert_equal(2, ub.n_seq_id[0]);
        t.assert_equal(0, ub.seq_idx[0]);
        t.assert_equal(1, ub.seq_idx[1]);
        t.assert_true("coupled sequences retain one index per input row", ba.get_input_ids() == std::vector<int32_t>({0, 1}));
    });

    t.test("split_equal_selected_decode_and_single_prefill", [&](testing & t) {
        batch_builder bb;
        const llama_seq_id seqs[] = {1, 2, 3, 0};
        const int lengths[] = {5, 2, 4, 2};
        for (size_t s = 0; s < 4; ++s) {
            for (int pos = 0; pos < lengths[s]; ++pos) {
                bb.add(pos, {seqs[s]}, pos + 1 == lengths[s]);
            }
        }

        llama_batch_allocr ba(1);
        if (!t.assert_true(ba.init(bb.make(), vocab, nullptr, bb.n_embd, 4, false))) {
            return;
        }
        const auto selected = [](llama_seq_id seq) { return seq == 0 || seq == 2; };
        auto split = [&]() {
            auto ub = ba.split_equal(4, false, 2, selected);
            if (ub.n_tokens == 0) {
                ub = ba.split_equal(4, false, 4, [&](llama_seq_id seq) { return !selected(seq); }, true);
            }
            return ub;
        };

        std::vector<float> hidden;
        const uint32_t sizes[] = {4, 1, 4, 4};
        for (size_t u = 0; u < 4; ++u) {
            const auto ub = split();
            if (!t.assert_equal(sizes[u], ub.n_tokens)) {
                return;
            }
            t.assert_true(ub.equal_seqs());
            t.assert_equal("decoders stay grouped; prefills stay separate", u == 0 ? 2u : 1u, ub.n_seqs_unq);
            for (uint32_t i = 0; i < ub.n_tokens; ++i) {
                t.assert_true("no mixed attention phases", selected(ub.seq_id[i][0]) == (u == 0));
            }
            if (u == 0) {
                t.assert_equal("both decoder rollback tails remain whole", 2u, ub.n_seq_tokens);
            }
            if (u == 2) {
                t.assert_equal("final observer window starts before the prompt end", 1, ub.pos[0]);
                t.assert_equal(4, ub.pos[3]);
                t.assert_equal(1, (int) ub.output[3]);
            }
            hidden.insert(hidden.end(), ub.embd, ub.embd + (size_t) ub.n_tokens * bb.n_embd);
        }
        t.assert_equal(0u, split().n_tokens);
        t.assert_equal(ba.get_n_tokens(), ba.get_n_used());
        t.assert_true(ba.get_input_ids() == std::vector<int32_t>({5, 6, 11, 12, 0, 1, 2, 3, 4, 7, 8, 9, 10}));
        t.assert_true("sparse logits follow extraction order", ba.get_out_ids() == std::vector<int32_t>({6, 12, 4, 10}));
        llama_row_reorder reorder;
        if (!t.assert_true(reorder.prepare(ba.get_input_ids(), ba.get_n_tokens()))) {
            return;
        }
        t.assert_true(reorder.apply({{hidden.data(), hidden.size(), bb.n_embd}}));
        t.assert_true("hidden rows recover the original batch order", hidden == bb.embd);

        batch_builder shared;
        shared.add(0, {0, 1}, true);
        t.assert_true(ba.init(shared.make(), vocab, nullptr, shared.n_embd, 4, false));
        t.assert_equal("a filter must select every owner of a shared row", 0u, ba.split_equal(4, false, 0, selected).n_tokens);
        t.assert_equal(0u, ba.get_n_used());
    });

    t.test("split_seq_per_sequence", [&](testing & t) {
        batch_builder bb;
        for (llama_seq_id s = 0; s < 3; ++s) {
            bb.add(0, {s}, false);
            bb.add(1, {s}, true);
        }

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(bb.make(), vocab, nullptr, bb.n_embd, 4, false));

        for (llama_seq_id s = 0; s < 3; ++s) {
            llama_ubatch ub = ba.split_seq(8);
            t.assert_equal(2u, ub.n_tokens);
            t.assert_equal(1u, ub.n_seqs);
            t.assert_equal(s, ub.seq_id[0][0]);
            t.assert_equal(s, ub.seq_id_unq[0]);
        }

        t.assert_equal(0u, ba.split_seq(8).n_tokens);
        t.assert_equal(6u, ba.get_n_used());
    });

    t.test("ubatch_reserve", [&](testing & t) {
        llama_batch_allocr ba(1);

        llama_ubatch ub = ba.ubatch_reserve(3, 2);
        t.assert_equal(6u, ub.n_tokens);
        t.assert_equal(3u, ub.n_seq_tokens);
        t.assert_equal(2u, ub.n_seqs);
        t.assert_equal(2u, ub.n_seqs_unq);
        t.assert_true(ub.equal_seqs());
        t.assert_equal(0, ub.seq_id_unq[0]);
        t.assert_equal(1, ub.seq_id_unq[1]);
        t.assert_true(ub.token != nullptr);
        t.assert_true(ub.embd == nullptr);
    });
}

static void test_keep_tail(testing & t) {
    llama_vocab vocab;

    // batch with n_tokens[s] tokens for each seq s, output on the last token of each seq
    auto make_batch = [](batch_builder & bb, std::initializer_list<int> n_tokens) {
        llama_seq_id s = 0;
        for (int n : n_tokens) {
            for (int i = 0; i < n; ++i) {
                bb.add(i, {s}, i == n - 1);
            }
            ++s;
        }
        return bb.make();
    };

    t.test("noop_when_seqs_complete", [&](testing & t) {
        batch_builder bb;

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(make_batch(bb, {2, 2}), vocab, nullptr, bb.n_embd, 4, false));

        llama_ubatch ub = ba.split_equal(4, false, 2);
        t.assert_equal("both seqs fit whole", 4u, ub.n_tokens);
        t.assert_equal(2u, ub.n_seqs);
        t.assert_equal(2u, ub.n_seq_tokens);

        t.assert_equal(0u, ba.split_equal(4, false, 2).n_tokens);
    });

    t.test("defers_seq_with_short_remainder", [&](testing & t) {
        batch_builder bb;

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(make_batch(bb, {2, 3}), vocab, nullptr, bb.n_embd, 4, false));

        // expansion stops at 2 tokens per seq: seq 0 completes, seq 1 would be left
        // with 1 < n_keep_tail remaining, so it is deferred entirely
        llama_ubatch ub = ba.split_equal(4, true, 2);
        t.assert_equal(2u, ub.n_tokens);
        t.assert_equal(1u, ub.n_seqs);
        t.assert_equal(0, ub.seq_id[0][0]);
        t.assert_equal(2u, ba.get_n_used());

        ub = ba.split_equal(4, true, 2);
        t.assert_equal("deferred seq comes back whole", 3u, ub.n_tokens);
        t.assert_equal(1u, ub.n_seqs);
        t.assert_equal(1, ub.seq_id[0][0]);
        for (int i = 0; i < 3; ++i) {
            t.assert_equal(i, ub.pos[i]);
        }

        t.assert_equal(5u, ba.get_n_used());
        t.assert_equal(0u, ba.split_equal(4, true, 2).n_tokens);
    });

    t.test("completes_first_seq_when_all_violate", [&](testing & t) {
        batch_builder bb;

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(make_batch(bb, {3, 3}), vocab, nullptr, bb.n_embd, 4, false));

        // expansion stops at 2 tokens per seq, leaving both with 1 < n_keep_tail remaining;
        // seq 0 still fits in n_ubatch, so it is extended to completion and emitted alone
        llama_ubatch ub = ba.split_equal(4, false, 2);
        t.assert_equal(3u, ub.n_tokens);
        t.assert_equal(1u, ub.n_seqs);
        t.assert_equal(3u, ub.n_seq_tokens);
        t.assert_equal(0, ub.seq_id[0][0]);
        for (int i = 0; i < 3; ++i) {
            t.assert_equal(i, ub.pos[i]);
        }
        t.assert_equal(3u, ba.get_n_used());

        ub = ba.split_equal(4, false, 2);
        t.assert_equal(3u, ub.n_tokens);
        t.assert_equal(1, ub.seq_id[0][0]);
        t.assert_equal(6u, ba.get_n_used());
    });

    t.test("truncates_to_preserve_tail", [&](testing & t) {
        batch_builder bb;

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(make_batch(bb, {5}), vocab, nullptr, bb.n_embd, 4, false));

        // 4 tokens would leave a remainder of 1, and the seq does not fit in n_ubatch,
        // so the ubatch is truncated until n_keep_tail tokens remain
        llama_ubatch ub = ba.split_equal(4, false, 2);
        t.assert_equal(3u, ub.n_tokens);
        t.assert_equal(1u, ub.n_seqs);
        t.assert_equal(2, ub.pos[2]);
        t.assert_equal(3u, ba.get_n_used());

        ub = ba.split_equal(4, false, 2);
        t.assert_equal("trailing tokens stay in one ubatch", 2u, ub.n_tokens);
        t.assert_equal(3, ub.pos[0]);
        t.assert_equal(4, ub.pos[1]);
        t.assert_equal(1, (int) ub.output[1]);

        t.assert_equal(5u, ba.get_n_used());
    });

    t.test("keeps_full_ubatch_with_sufficient_remainder", [&](testing & t) {
        batch_builder bb;

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(make_batch(bb, {6}), vocab, nullptr, bb.n_embd, 4, false));

        llama_ubatch ub = ba.split_equal(4, false, 2);
        t.assert_equal("remainder >= n_keep_tail, no truncation", 4u, ub.n_tokens);

        ub = ba.split_equal(4, false, 2);
        t.assert_equal(2u, ub.n_tokens);
        t.assert_equal(4, ub.pos[0]);
        t.assert_equal(5, ub.pos[1]);

        t.assert_equal(6u, ba.get_n_used());
    });

    t.test("multi_seq_prefix_kept", [&](testing & t) {
        batch_builder bb;

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(make_batch(bb, {3, 4}), vocab, nullptr, bb.n_embd, 6, false));

        // expansion stops at 3 tokens per seq: seq 0 completes, seq 1 has 1 < n_keep_tail
        // remaining and is deferred even though its tokens were already gathered
        llama_ubatch ub = ba.split_equal(6, true, 2);
        t.assert_equal(3u, ub.n_tokens);
        t.assert_equal(1u, ub.n_seqs);
        t.assert_equal(0, ub.seq_id[0][0]);
        t.assert_equal(3u, ba.get_n_used());

        ub = ba.split_equal(6, true, 2);
        t.assert_equal(4u, ub.n_tokens);
        t.assert_equal(1, ub.seq_id[0][0]);
        t.assert_equal(7u, ba.get_n_used());
    });
}

static void test_mrope(testing & t) {
    llama_vocab vocab;

    t.test("pos_layout_and_split", [&](testing & t) {
        const uint32_t n_pos = 4;
        const uint32_t n_embd = 2;

        batch_builder bb(n_embd);
        bb.add(10, {0}, false);
        bb.add(11, {0}, true);

        // M-RoPE positions for embeddings are laid out [n_pos][n_tokens]
        std::vector<llama_pos> pos = {
            10, 11, // temporal
             5,  6, // y
             7,  8, // x
             0,  0,
        };

        llama_batch batch = bb.make(false, true, true);
        batch.pos = pos.data();

        llama_batch_allocr ba(n_pos);
        t.assert_true(ba.init(batch, vocab, nullptr, n_embd, 4, false));

        llama_ubatch ub = ba.split_simple(2);
        t.assert_equal(2u, ub.n_tokens);
        t.assert_equal(n_pos, ub.n_pos);
        t.assert_true(ub.is_pos_2d());

        const llama_pos expected[8] = {10, 11, 5, 6, 7, 8, 0, 0};
        for (int i = 0; i < 8; ++i) {
            t.assert_equal(expected[i], ub.pos[i]);
        }
    });

    t.test("pos_jump_allowed", [&](testing & t) {
        const uint32_t n_pos = 4;
        const uint32_t n_embd = 2;

        mock_memory mem;
        mem.ranges[0] = {0, 9};

        llama_batch_allocr ba(n_pos);

        auto try_pos = [&](llama_pos p0) {
            batch_builder bb(n_embd);
            bb.add(p0, {0}, true);

            std::vector<llama_pos> pos = {p0, 1, 1, 0};

            llama_batch batch = bb.make(false, true, true);
            batch.pos = pos.data();

            return ba.init(batch, vocab, &mem, n_embd, 4, false);
        };

        t.assert_true("gap after memory is allowed",     try_pos(15));
        t.assert_true("overlap is allowed for embd",     try_pos(9));
        t.assert_true("pos behind memory is rejected",  !try_pos(8));
    });
}

static void test_input_row_order(testing & t) {
    llama_vocab vocab;

    // All four splitting routes must retain original indices, even when only
    // each sequence's final token requests logits and recurrent tails are kept.
    for (int mode = 0; mode < 4; ++mode) {
        t.test("split_mode_" + std::to_string(mode), [&](testing & t) {
            batch_builder bb;
            const int lengths[] = {110, 99, 73, 110};
            for (int seq = 0; seq < 4; ++seq) {
                for (int pos = 0; pos < lengths[seq]; ++pos) {
                    bb.add(pos, {seq}, pos + 1 == lengths[seq]);
                }
            }
            llama_batch_allocr ba(1);
            if (!t.assert_true(ba.init(bb.make(), vocab, nullptr, bb.n_embd, 4, false))) {
                return;
            }
            std::vector<float> extracted;
            std::vector<int32_t> observed_ids;
            while (true) {
                const auto ub = mode == 0 ? ba.split_simple(128) :
                                mode == 3 ? ba.split_seq(128) : ba.split_equal(128, mode == 2, 6);
                if (ub.n_tokens == 0) {
                    break;
                }
                extracted.insert(extracted.end(), ub.embd, ub.embd + (size_t) ub.n_tokens * bb.n_embd);
                for (uint32_t i = 0; i < ub.n_tokens; ++i) {
                    observed_ids.push_back((int32_t) (ub.embd[i * bb.n_embd] / 100.0f));
                }
            }
            t.assert_equal(ba.get_n_tokens(), ba.get_n_used());
            t.assert_true("input mapping covers actual ubatch extraction order", ba.get_input_ids() == observed_ids);
            t.assert_equal((size_t) 4, ba.get_out_ids().size());
            if (mode == 1 || mode == 2) {
                t.assert_true("hybrid splitting produced a non-identity permutation", extracted != bb.embd);
            }
            llama_row_reorder reorder;
            if (!t.assert_true(reorder.prepare(ba.get_input_ids(), ba.get_n_tokens()))) {
                return;
            }
            t.assert_true(reorder.apply({{extracted.data(), extracted.size(), bb.n_embd}}));
            t.assert_true("all hidden rows recover original batch order", extracted == bb.embd);

            // Reset/resplit models preparation retries and the next batch.
            ba.split_reset();
            t.assert_true(ba.get_input_ids().empty());
            t.assert_true(ba.get_out_ids().empty());
            const auto encoder_ub = ba.split_simple(ba.get_n_tokens());
            t.assert_equal(ba.get_n_tokens(), encoder_ub.n_tokens);
            t.assert_true(reorder.prepare(ba.get_input_ids(), ba.get_n_tokens()));
            t.assert_true(reorder.apply({{extracted.data(), extracted.size(), bb.n_embd}}));
            t.assert_true("single embedding-input encoder batch is identity", extracted == bb.embd);
        });
    }

    t.test("equal_c4_sparse_logits_are_already_sorted", [&](testing & t) {
        batch_builder bb;
        for (int seq = 0; seq < 4; ++seq) {
            for (int pos = 0; pos < 110; ++pos) {
                bb.add(pos, {seq}, pos == 109);
            }
        }
        llama_batch_allocr ba(1);
        if (!t.assert_true(ba.init(bb.make(), vocab, nullptr, bb.n_embd, 4, false))) {
            return;
        }
        std::vector<float> extracted;
        while (true) {
            const auto ub = ba.split_equal(128, true, 6);
            if (ub.n_tokens == 0) {
                break;
            }
            extracted.insert(extracted.end(), ub.embd, ub.embd + (size_t) ub.n_tokens * bb.n_embd);
        }
        t.assert_true("selected outputs alone cannot repair hidden rows", ba.get_out_ids() == std::vector<int32_t>({109, 219, 329, 439}));
        if (!t.assert_equal((size_t) 440, ba.get_input_ids().size())) {
            return;
        }
        t.assert_equal("raw row 32 belongs to the next sequence", 110, ba.get_input_ids()[32]);
        llama_row_reorder reorder;
        t.assert_true(reorder.prepare(ba.get_input_ids(), 440));
        t.assert_true(reorder.apply({{extracted.data(), extracted.size(), bb.n_embd}}));
        t.assert_true("hidden rows match the original batch", extracted == bb.embd);
    });

    t.test("independent_masked_unmasked_and_repeated_getters", [&](testing & t) {
        const std::vector<int32_t> ids = {2, 0, 4, 1, 3};
        std::vector<float> layer = {20, 21, 0, 1, 40, 41, 10, 11, 30, 31};
        std::vector<float> nextn = {2, 0, 4, 1, 3};
        std::vector<float> masked = {4, 1};
        llama_row_reorder input_order;
        t.assert_true(input_order.prepare(ids, 5));
        t.assert_true(input_order.apply({{layer.data(), layer.size(), 2}, {nextn.data(), nextn.size(), 1}}));
        t.assert_true(layer == std::vector<float>({0, 1, 10, 11, 20, 21, 30, 31, 40, 41}));
        t.assert_true(nextn == std::vector<float>({0, 1, 2, 3, 4}));
        t.assert_true("masked nextn does not receive input swaps", masked == std::vector<float>({4, 1}));

        llama_row_reorder selected_order;
        t.assert_true(selected_order.prepare({1, 0}, 2));
        t.assert_true(selected_order.apply({{masked.data(), masked.size(), 1}}));
        t.assert_true(masked == std::vector<float>({1, 4}));
        t.assert_true(!input_order.pending());
        t.assert_true(input_order.apply({{layer.data(), layer.size(), 2}, {nextn.data(), nextn.size(), 1}}));
        t.assert_true("another getter does not apply swaps twice", nextn == std::vector<float>({0, 1, 2, 3, 4}));
        t.assert_true(input_order.prepare(ids, 5));
        input_order.clear();
        t.assert_true(input_order.apply({{nextn.data(), nextn.size(), 1}}));
        t.assert_true("reset discards a previous pending permutation", nextn == std::vector<float>({0, 1, 2, 3, 4}));
    });

    t.test("invalid_mapping_and_buffer_fail_without_mutation", [&](testing & t) {
        llama_row_reorder reorder;
        t.assert_true(!reorder.prepare({0, 1}, 3));
        t.assert_true(!reorder.prepare({0, 0, 2}, 3));
        t.assert_true(!reorder.prepare({0, -1, 2}, 3));
        t.assert_true(!reorder.prepare({0, 1, 3}, 3));
        t.assert_true(!reorder.pending());
        t.assert_true(reorder.prepare({1, 0}, 2));
        std::vector<float> valid = {1, 0};
        std::vector<float> short_buffer = {1};
        t.assert_true(!reorder.apply({{valid.data(), valid.size(), 1}, {short_buffer.data(), short_buffer.size(), 1}}));
        t.assert_true("validation precedes every write", valid == std::vector<float>({1, 0}));
        t.assert_true(reorder.pending());
        t.assert_true(!reorder.apply({{nullptr, 2, 1}}));
        t.assert_true(!reorder.apply({{valid.data(), valid.size(), 0}}));
        t.assert_true(reorder.apply({{valid.data(), valid.size(), 1}}));
        t.assert_true(valid == std::vector<float>({0, 1}));
        t.assert_true(reorder.prepare({}, 0));
        t.assert_true(reorder.apply({}));
    });
}

int main(int argc, char ** argv) {
    testing t;

    const char * verbose = getenv("LLAMA_TEST_VERBOSE");
    if (verbose) {
        t.verbose = std::string(verbose) == "1";
    }
    if (!t.verbose) {
        llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    }

    if (argc > 1) {
        t.set_filter(argv[1]);
    }

    t.test("init",      test_init);
    t.test("split",     test_split);
    t.test("keep_tail", test_keep_tail);
    t.test("mrope",     test_mrope);
    t.test("input_rows", test_input_row_order);

    return t.summary();
}
