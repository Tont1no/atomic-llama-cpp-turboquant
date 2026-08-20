#include "speculative.h"
#include "../src/llama-dspark-packed.h"
#include "../src/llama-graph.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

#undef NDEBUG
#include <cassert>

struct packed_rows_fixture {
    std::vector<llama_seq_id> row_ids;
    std::vector<int32_t> n_seq_id;
    std::vector<llama_seq_id *> seq_id;

    explicit packed_rows_fixture(std::vector<llama_seq_id> ids) : row_ids(std::move(ids)) {
        n_seq_id.assign(row_ids.size(), 1);
        seq_id.reserve(row_ids.size());
        for (auto & id : row_ids) {
            seq_id.push_back(&id);
        }
    }

    llama_batch batch() {
        llama_batch result = {};
        result.n_tokens = (int32_t) row_ids.size();
        result.n_seq_id = n_seq_id.data();
        result.seq_id   = seq_id.data();
        return result;
    }

    llama_ubatch ubatch(uint32_t n_unique) {
        llama_ubatch result = {};
        result.n_tokens   = (uint32_t) row_ids.size();
        result.n_seqs_unq = n_unique;
        result.n_seq_id   = n_seq_id.data();
        result.seq_id     = seq_id.data();
        return result;
    }
};

static std::vector<llama_seq_id> rows_for_widths(const std::vector<int32_t> & widths) {
    std::vector<llama_seq_id> rows;
    for (size_t seq = 0; seq < widths.size(); ++seq) {
        rows.insert(rows.end(), (size_t) widths[seq], (llama_seq_id) seq);
    }
    return rows;
}

static void test_caps_and_indptr() {
    const auto caps = common_speculative_dspark_pack_rows(7, 1, 7, 64, { 0, 1, 2, 4, 7 });
    assert(caps.valid);
    assert(caps.widths == std::vector<int32_t>({ 0, 1, 2, 4, 7 }));
    assert(caps.indptr == std::vector<int32_t>({ 0, 0, 1, 3, 7, 14 }));
    assert(caps.total_rows == 14);

    const auto min_two = common_speculative_dspark_pack_rows(7, 2, 7, 64, { 0, 1, 2, 4, 7 });
    assert(min_two.valid);
    assert(min_two.widths == std::vector<int32_t>({ 0, 0, 2, 4, 7 }));
    assert(min_two.indptr == std::vector<int32_t>({ 0, 0, 0, 2, 6, 13 }));
}

static void test_n1_n4_n8_mixed_plans() {
    const auto n1 = common_speculative_dspark_pack_rows(7, 1, 7, 64, { 7 });
    assert(n1.valid && n1.total_rows == 7);

    const auto n4 = common_speculative_dspark_pack_rows(7, 1, 7, 64, { 7, 4, 2, 1 });
    assert(n4.valid && n4.total_rows == 14);
    assert(n4.indptr == std::vector<int32_t>({ 0, 7, 11, 13, 14 }));

    const auto n8 = common_speculative_dspark_pack_rows(7, 1, 7, 64, { 7, 0, 4, 1, 2, 7, 0, 4 });
    assert(n8.valid && n8.total_rows == 25);
    assert(n8.widths == std::vector<int32_t>({ 7, 0, 4, 1, 2, 7, 0, 4 }));
    assert(n8.indptr == std::vector<int32_t>({ 0, 7, 7, 11, 12, 14, 21, 21, 25 }));
}

static void test_source_order_and_equal_width_groups() {
    packed_rows_fixture equal(rows_for_widths({ 4, 4, 4, 4 }));
    const auto equal_layout = llama_dspark_packed_layout_from_ubatch(equal.ubatch(4), 7);
    assert(equal_layout.valid);
    assert(equal_layout.indptr == std::vector<uint32_t>({ 0, 4, 8, 12, 16 }));
    assert(equal_layout.groups.size() == 1);
    assert(equal_layout.groups[0].row_begin == 0);
    assert(equal_layout.groups[0].width == 4);
    assert(equal_layout.groups[0].n_seqs == 4);

    // Only adjacent equal widths group. This keeps every produced logit and
    // confidence row in the exact source sequence order.
    packed_rows_fixture mixed(rows_for_widths({ 7, 4, 4, 2, 7, 7, 1, 1 }));
    const auto mixed_layout = llama_dspark_packed_layout_from_ubatch(mixed.ubatch(8), 7);
    assert(mixed_layout.valid);
    assert(mixed_layout.seq_ids == std::vector<llama_seq_id>({ 0, 1, 2, 3, 4, 5, 6, 7 }));
    assert(mixed_layout.indptr == std::vector<uint32_t>({ 0, 7, 11, 15, 17, 24, 31, 32, 33 }));
    assert(mixed_layout.groups.size() == 5);
    assert(mixed_layout.groups[0].row_begin == 0  && mixed_layout.groups[0].width == 7 && mixed_layout.groups[0].n_seqs == 1);
    assert(mixed_layout.groups[1].row_begin == 7  && mixed_layout.groups[1].width == 4 && mixed_layout.groups[1].n_seqs == 2);
    assert(mixed_layout.groups[2].row_begin == 15 && mixed_layout.groups[2].width == 2 && mixed_layout.groups[2].n_seqs == 1);
    assert(mixed_layout.groups[3].row_begin == 17 && mixed_layout.groups[3].width == 7 && mixed_layout.groups[3].n_seqs == 2);
    assert(mixed_layout.groups[4].row_begin == 31 && mixed_layout.groups[4].width == 1 && mixed_layout.groups[4].n_seqs == 2);
}

static void test_malformed_fails_closed() {
    assert(!common_speculative_dspark_pack_rows(8, 1, 7, 64, { 7 }).valid);
    assert(!common_speculative_dspark_pack_rows(7, 1, 7, 10, { 7, 4 }).valid);
    assert(!common_speculative_dspark_pack_rows(7, 1, 7, 64, { -2 }).valid);
    assert(!common_speculative_dspark_pack_rows(7, 8, 7, 64, { 7 }).valid);

    packed_rows_fixture repeated({ 0, 0, 1, 1, 0 });
    assert(!llama_dspark_packed_layout_from_ubatch(repeated.ubatch(2), 7).valid);

    packed_rows_fixture too_wide(rows_for_widths({ 8 }));
    assert(!llama_dspark_packed_layout_from_ubatch(too_wide.ubatch(1), 7).valid);

    packed_rows_fixture wrong_unique(rows_for_widths({ 2, 2 }));
    assert(!llama_dspark_packed_layout_from_ubatch(wrong_unique.ubatch(1), 7).valid);

    packed_rows_fixture multi_owner({ 0, 0 });
    multi_owner.n_seq_id[1] = 2;
    assert(!llama_dspark_packed_layout_from_batch(multi_owner.batch(), 7).valid);

    llama_batch missing = {};
    missing.n_tokens = 1;
    assert(!llama_dspark_packed_layout_from_batch(missing, 7).valid);

    uint32_t gamma = 0;
    assert(llama_dspark_parse_gamma("7", gamma) && gamma == 7);
    assert(!llama_dspark_parse_gamma("0", gamma));
    assert(!llama_dspark_parse_gamma("7junk", gamma));
    assert(!llama_dspark_parse_gamma("7.5", gamma));
    assert(!llama_dspark_parse_gamma("3000000000", gamma));
}

static void test_graph_reuse_is_layout_exact() {
    packed_rows_fixture rows_a(rows_for_widths({ 4, 2 }));
    packed_rows_fixture rows_b(rows_for_widths({ 3, 3 }));

    llm_graph_params lhs = {};
    llm_graph_params rhs = {};
    lhs.ubatch = rows_a.ubatch(2);
    rhs.ubatch = rows_b.ubatch(2);
    lhs.dspark_packed_indptr = { 0, 4, 6 };
    rhs.dspark_packed_indptr = { 0, 3, 6 };

    assert(!lhs.allow_reuse(rhs));
    rhs.dspark_packed_indptr = lhs.dspark_packed_indptr;
    assert(lhs.allow_reuse(rhs));
}

static void test_reserve_shapes_cover_rows_and_groups() {
    const auto shapes = llama_dspark_reserve_widths(8, 64, 9);
    assert(shapes.size() == 2);
    assert(shapes[0] == std::vector<uint32_t>(8, 8));
    assert(shapes[1].size() == 9);
    assert(std::accumulate(shapes[1].begin(), shapes[1].end(), 0u) == 64);
    for (size_t i = 1; i < shapes[1].size(); ++i) {
        assert(shapes[1][i - 1] != shapes[1][i]);
    }

    const auto gamma_two = llama_dspark_reserve_widths(2, 4, 3);
    assert(gamma_two.size() == 2);
    assert(std::accumulate(gamma_two[1].begin(), gamma_two[1].end(), 0u) == 4);
    assert(gamma_two[1] == std::vector<uint32_t>({ 1, 2, 1 }));

    const auto tight_two = llama_dspark_reserve_widths(3, 3, 2);
    assert(tight_two.size() == 2);
    assert(tight_two[1].size() == 2);
    assert(std::accumulate(tight_two[1].begin(), tight_two[1].end(), 0u) == 3);
    assert(tight_two[1][0] != tight_two[1][1]);

    const auto tight_four = llama_dspark_reserve_widths(8, 16, 4);
    assert(tight_four.size() == 2);
    assert(tight_four[1].size() == 4);
    assert(std::accumulate(tight_four[1].begin(), tight_four[1].end(), 0u) == 16);
    for (size_t i = 1; i < tight_four[1].size(); ++i) {
        assert(tight_four[1][i - 1] != tight_four[1][i]);
    }

    const auto slot_limited = llama_dspark_reserve_widths(8, 512, 8);
    assert(slot_limited.size() == 2);
    assert(slot_limited[0] == std::vector<uint32_t>(8, 8));
    assert(slot_limited[1].size() == 8);
    assert(std::accumulate(slot_limited[1].begin(), slot_limited[1].end(), 0u) == 60);
    for (size_t i = 1; i < slot_limited[1].size(); ++i) {
        assert(slot_limited[1][i - 1] != slot_limited[1][i]);
    }

    const auto gamma_two_many = llama_dspark_reserve_widths(2, 512, 256);
    assert(gamma_two_many.size() == 2);
    assert(gamma_two_many[1].size() == 256);
    assert(std::accumulate(gamma_two_many[1].begin(), gamma_two_many[1].end(), 0u) == 384);

    const auto gamma_one = llama_dspark_reserve_widths(1, 4, 4);
    assert(gamma_one.size() == 1);
    assert(gamma_one[0] == std::vector<uint32_t>(4, 1));

    assert(llama_dspark_reserve_widths(0, 64, 8).empty());
    assert(llama_dspark_reserve_widths(9, 8, 8).empty());
}

static void test_reserve_outputs_follow_packed_sequence_limits() {
    // P8's worst-transition reserve topology has 52 packed rows, but the
    // sampling graph contract permits only two output rows for each of the
    // eight logical sequences. The old reserve selected all 52 rows, although
    // graph_max_nodes() used n_sampling_outputs_max=16 (plus its conservative
    // per-sampler baseline), exhausting graph capacity during sched_reserve().
    const auto p8_shapes = llama_dspark_reserve_widths(7, 128, 8);
    assert(p8_shapes.size() == 2);
    const auto & p8_widths = p8_shapes[1];
    assert(p8_widths == std::vector<uint32_t>({ 7, 6, 7, 6, 7, 6, 7, 6 }));
    assert(std::accumulate(p8_widths.begin(), p8_widths.end(), 0u) == 52);
    const std::vector<bool> all_p8_samplers(8, true);
    assert(llama_dspark_reserve_output_rows(p8_widths, 2048, 2, all_p8_samplers) == std::vector<uint32_t>({
            0, 1, 7, 8, 13, 14, 20, 21, 26, 27, 33, 34, 39, 40, 46, 47 }));

    assert(llama_dspark_reserve_output_rows(std::vector<uint32_t>(8, 1), 2048, 2, all_p8_samplers) ==
            std::vector<uint32_t>({ 0, 1, 2, 3, 4, 5, 6, 7 }));
    assert(llama_dspark_reserve_output_rows(p8_widths, 5, 2, all_p8_samplers) ==
            std::vector<uint32_t>({ 0, 1, 7, 8, 13 }));

    const std::vector<bool> partial_samplers = { false, true, false, true, false, false, false, false };
    assert(llama_dspark_reserve_output_rows(p8_widths, 5, 2, partial_samplers) ==
            std::vector<uint32_t>({ 0, 7, 8, 20, 21 }));
    const auto partial_outputs = llama_dspark_reserve_output_rows(
            p8_widths, 2048, 2, partial_samplers);
    assert(partial_outputs.size() == 44);
    assert(llama_dspark_reserve_output_rows(p8_widths, 2048, 2, std::vector<bool>(8, false)).size() == 52);

    const auto sparse_plans = llama_dspark_reserve_seq_id_plans(p8_widths, 8, 128, 2, { 100, 3 });
    assert(sparse_plans.size() == 2);
    std::vector<size_t> sparse_output_sizes;
    for (const auto & plan : sparse_plans) {
        assert(plan.size() == 8);
        assert(std::find(plan.begin(), plan.end(), 100) != plan.end());
        assert(std::find(plan.begin(), plan.end(), 3) != plan.end());
        std::vector<bool> sparse_mask;
        for (llama_seq_id seq_id : plan) {
            sparse_mask.push_back(seq_id == 100 || seq_id == 3);
        }
        sparse_output_sizes.push_back(
                llama_dspark_reserve_output_rows(p8_widths, 2048, 2, sparse_mask).size());
    }
    std::sort(sparse_output_sizes.begin(), sparse_output_sizes.end());
    assert(sparse_output_sizes == std::vector<size_t>({ 43, 44 }));

    assert(llama_dspark_reserve_output_rows({}, 2048, 2, {}).empty());
    assert(llama_dspark_reserve_output_rows(p8_widths, 0, 2, all_p8_samplers).empty());
    assert(llama_dspark_reserve_output_rows(p8_widths, 2048, 0, all_p8_samplers).empty());
    assert(llama_dspark_reserve_output_rows({ 7, 0, 6 }, 2048, 2, { true, true, true }).empty());
    assert(llama_dspark_reserve_output_rows(p8_widths, 2048, 2, { true }).empty());
    assert(llama_dspark_reserve_seq_id_plans(p8_widths, 7, 128, 2, { 3 }).empty());
    assert(llama_dspark_reserve_seq_id_plans(p8_widths, 8, 7, 2, { 3 }).empty());
}

int main() {
    test_caps_and_indptr();
    test_n1_n4_n8_mixed_plans();
    test_source_order_and_equal_width_groups();
    test_malformed_fails_closed();
    test_graph_reuse_is_layout_exact();
    test_reserve_shapes_cover_rows_and_groups();
    test_reserve_outputs_follow_packed_sequence_limits();
    return 0;
}
