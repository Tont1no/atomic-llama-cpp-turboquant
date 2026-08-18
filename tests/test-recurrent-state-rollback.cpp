#include "arg.h"
#include "common.h"
#include "llama.h"
#include "llama-ext.h"

#include <algorithm>
#include <array>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <vector>

static llama_context * make_ctx(
        const common_params & params,
        llama_model * model,
        uint32_t n_seq_max = 1,
        bool rs_seq_dynamic = false) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = n_seq_max;
    cparams.n_rs_seq  = 8;
    cparams.rs_seq_dynamic = rs_seq_dynamic;
    cparams.n_batch   = std::max(cparams.n_batch,  (uint32_t) (cparams.n_rs_seq + 1));
    cparams.n_ubatch  = std::max(cparams.n_ubatch, (uint32_t) (cparams.n_rs_seq + 1));
    return llama_init_from_model(model, cparams);
}

static bool decode_tokens(llama_context * ctx, const std::vector<llama_token> & tokens, uint32_t count);
static bool decode_one(llama_context * ctx, llama_token tok, llama_pos pos);

static size_t context_memory_bytes(const llama_context * ctx) {
    size_t total = 0;
    for (const auto & [_, mb] : llama_get_memory_breakdown(ctx)) {
        total += mb.context;
    }
    return total;
}

static std::vector<uint32_t> tag_recurrent_depth(llama_batch & batch, size_t n_rows, uint32_t depth) {
    std::vector<uint32_t> depths(n_rows, depth);
    batch.rs_depth = depths.data();
    return depths;
}

static bool decode_ragged_and_compare(
        const common_params & params,
        llama_model * model,
        int n_vocab) {
    llama_context * ctx_batched = make_ctx(params, model, 3, true);
    llama_context * ctx_ref[3] = {
        make_ctx(params, model, 1, true),
        make_ctx(params, model, 1, true),
        make_ctx(params, model, 1, true),
    };
    if (ctx_batched == nullptr || ctx_ref[0] == nullptr || ctx_ref[1] == nullptr || ctx_ref[2] == nullptr) {
        fprintf(stderr, "%s : failed to create ragged contexts\n", __func__);
        return false;
    }
    const size_t bytes_depth_zero = context_memory_bytes(ctx_batched);

    std::vector<std::vector<llama_token>> rows = {
        { 1 },
        { 2, 3 },
        { 4, 5, 6, 7 },
    };
    for (auto & seq : rows) {
        for (auto & token : seq) {
            token %= n_vocab;
        }
    }

    llama_batch batch = llama_batch_init(7, 0, 3);
    std::vector<uint32_t> batch_depths;
    for (llama_seq_id seq = 0; seq < 3; ++seq) {
        for (llama_pos pos = 0; pos < (llama_pos) rows[seq].size(); ++pos) {
            common_batch_add(batch, rows[seq][pos], pos, { seq }, pos + 1 == (llama_pos) rows[seq].size());
            batch_depths.push_back((uint32_t) rows[seq].size() - 1);
        }
        if (!decode_tokens(ctx_ref[seq], rows[seq], (uint32_t) rows[seq].size())) {
            fprintf(stderr, "%s : reference ragged decode failed for sequence %d\n", __func__, seq);
            llama_batch_free(batch);
            return false;
        }
    }
    batch.rs_depth = batch_depths.data();
    if (llama_decode(ctx_batched, batch) != 0) {
        fprintf(stderr, "%s : batched ragged decode failed\n", __func__);
        llama_batch_free(batch);
        return false;
    }
    llama_batch_free(batch);
    const size_t bytes_depth_three = context_memory_bytes(ctx_batched);
    const bool uses_compact_storage = bytes_depth_three > bytes_depth_zero;

    constexpr float eps = 1e-5f;
    const auto compare_outputs = [&](
            const char * phase,
            const std::array<int32_t, 3> & batched_idxs,
            const std::array<int32_t, 3> & ref_idxs) {
        for (int seq = 0; seq < 3; ++seq) {
            const float * actual   = llama_get_logits_ith(ctx_batched, batched_idxs[seq]);
            const float * expected = llama_get_logits_ith(ctx_ref[seq], ref_idxs[seq]);
            if (actual == nullptr || expected == nullptr) {
                fprintf(stderr, "%s : missing %s logits for sequence %d\n", __func__, phase, seq);
                return false;
            }
            for (int token = 0; token < n_vocab; ++token) {
                if (std::fabs(actual[token] - expected[token]) > eps) {
                    fprintf(stderr, "%s : %s mismatch for sequence %d token %d (%g != %g)\n",
                            __func__, phase, seq, token, (double) actual[token], (double) expected[token]);
                    return false;
                }
            }
        }
        return true;
    };
    if (!compare_outputs("ragged", { 0, 2, 6 }, { 0, 1, 3 })) {
        return false;
    }

    // Follow the 1/2/4-row ragged batch with one row per sequence. This forces
    // active depth 3 -> 0 immediately while physical shrink is deferred.
    llama_set_recurrent_load_hint(ctx_batched, 3);
    llama_batch one_each = llama_batch_init(3, 0, 3);
    auto one_each_depths = tag_recurrent_depth(one_each, 3, 0);
    for (llama_seq_id seq = 0; seq < 3; ++seq) {
        const llama_token token = (llama_token) ((8 + seq) % n_vocab);
        const llama_pos pos = (llama_pos) rows[seq].size();
        common_batch_add(one_each, token, pos, { seq }, true);
        if (!decode_one(ctx_ref[seq], token, pos)) {
            fprintf(stderr, "%s : reference depth-zero decode failed for sequence %d\n", __func__, seq);
            llama_batch_free(one_each);
            return false;
        }
    }
    if (llama_decode(ctx_batched, one_each) != 0) {
        fprintf(stderr, "%s : batched depth-zero decode failed\n", __func__);
        llama_batch_free(one_each);
        return false;
    }
    llama_batch_free(one_each);

    if (!compare_outputs("depth-zero-first", { 0, 1, 2 }, { 0, 0, 0 })) {
        return false;
    }
    const llama_recurrent_resize_stats pending_first = llama_get_recurrent_resize_stats(ctx_batched);
    if (uses_compact_storage && (context_memory_bytes(ctx_batched) != bytes_depth_three ||
            pending_first.resident_depth != 3 || pending_first.pending_depth != 0 ||
            pending_first.stable_ticks != 1)) {
        fprintf(stderr, "%s : first low-depth epoch did not defer shrink\n", __func__);
        return false;
    }

    // A second decode chunk in the same scheduling epoch must not satisfy the
    // hysteresis dwell by itself.
    llama_batch same_epoch = llama_batch_init(3, 0, 3);
    auto same_epoch_depths = tag_recurrent_depth(same_epoch, 3, 0);
    for (llama_seq_id seq = 0; seq < 3; ++seq) {
        const llama_token token = (llama_token) ((11 + seq) % n_vocab);
        const llama_pos pos = (llama_pos) rows[seq].size() + 1;
        common_batch_add(same_epoch, token, pos, { seq }, true);
        if (!decode_one(ctx_ref[seq], token, pos)) {
            llama_batch_free(same_epoch);
            return false;
        }
    }
    if (llama_decode(ctx_batched, same_epoch) != 0) {
        llama_batch_free(same_epoch);
        return false;
    }
    llama_batch_free(same_epoch);
    if (uses_compact_storage && context_memory_bytes(ctx_batched) != bytes_depth_three) {
        fprintf(stderr, "%s : repeated chunk in one epoch shrank storage early\n", __func__);
        return false;
    }

    // The next stable scheduling epoch performs the single coalesced shrink.
    llama_set_recurrent_load_hint(ctx_batched, 3);
    llama_batch next_epoch = llama_batch_init(3, 0, 3);
    auto next_epoch_depths = tag_recurrent_depth(next_epoch, 3, 0);
    for (llama_seq_id seq = 0; seq < 3; ++seq) {
        const llama_token token = (llama_token) ((14 + seq) % n_vocab);
        const llama_pos pos = (llama_pos) rows[seq].size() + 2;
        common_batch_add(next_epoch, token, pos, { seq }, true);
        if (!decode_one(ctx_ref[seq], token, pos)) {
            llama_batch_free(next_epoch);
            return false;
        }
    }
    if (llama_decode(ctx_batched, next_epoch) != 0) {
        llama_batch_free(next_epoch);
        return false;
    }
    llama_batch_free(next_epoch);

    const size_t bytes_depth_zero_again = context_memory_bytes(ctx_batched);
    if (uses_compact_storage && bytes_depth_zero_again != bytes_depth_zero) {
        fprintf(stderr, "%s : compact recurrent storage did not return to depth-zero size (%zu != %zu)\n",
                __func__, bytes_depth_zero_again, bytes_depth_zero);
        return false;
    }
    if (uses_compact_storage) {
        fprintf(stderr, "%s : compact recurrent context bytes depth0=%zu depth3=%zu depth0=%zu\n",
                __func__, bytes_depth_zero, bytes_depth_three, bytes_depth_zero_again);
    }

    bool ok = compare_outputs("depth-zero-final", { 0, 1, 2 }, { 0, 0, 0 });
    const llama_recurrent_resize_stats final_stats = llama_get_recurrent_resize_stats(ctx_batched);
    if (ok && uses_compact_storage && (final_stats.count != 2 || final_stats.resident_depth != 0 ||
            final_stats.pending_depth != UINT32_MAX)) {
        fprintf(stderr, "%s : unexpected resize telemetry count=%llu resident=%u pending=%u\n",
                __func__, (unsigned long long) final_stats.count,
                final_stats.resident_depth, final_stats.pending_depth);
        ok = false;
    }
    if (ok && llama_memory_seq_rm(llama_get_memory(ctx_batched), 2, 4, -1)) {
        fprintf(stderr, "%s : depth-zero graph incorrectly allowed rollback into a stale snapshot plane\n", __func__);
        ok = false;
    }
    llama_free(ctx_batched);
    for (auto * ctx : ctx_ref) {
        llama_free(ctx);
    }
    return ok;
}

static bool compare_last_logits(
        const char * phase,
        llama_context * batched,
        int32_t batched_idx,
        llama_context * reference,
        int n_vocab) {
    const float * actual   = llama_get_logits_ith(batched, batched_idx);
    const float * expected = llama_get_logits_ith(reference, 0);
    if (actual == nullptr || expected == nullptr) {
        fprintf(stderr, "%s : missing logits\n", phase);
        return false;
    }
    constexpr float eps = 1e-5f;
    for (int token = 0; token < n_vocab; ++token) {
        if (std::fabs(actual[token] - expected[token]) > eps) {
            fprintf(stderr, "%s : mismatch at token %d (%g != %g)\n",
                    phase, token, (double) actual[token], (double) expected[token]);
            return false;
        }
    }
    return true;
}

static bool decode_multi_rollback_shrink(
        const common_params & params,
        llama_model * model,
        int n_vocab) {
    llama_context * ctx = make_ctx(params, model, 2, true);
    llama_context * refs[2] = {
        make_ctx(params, model, 1, true),
        make_ctx(params, model, 1, true),
    };
    if (ctx == nullptr || refs[0] == nullptr || refs[1] == nullptr) {
        fprintf(stderr, "%s : failed to create contexts\n", __func__);
        return false;
    }
    const size_t bytes_depth_zero = context_memory_bytes(ctx);

    const uint32_t count = llama_n_rs_seq(ctx) + 1;
    constexpr uint32_t rollback = 3;
    if (count <= rollback) {
        return true;
    }
    std::vector<std::vector<llama_token>> tokens(2, std::vector<llama_token>(count));
    llama_batch prompt = llama_batch_init(2*count, 0, 2);
    auto prompt_depths = tag_recurrent_depth(prompt, 2*count, count - 1);
    for (llama_seq_id seq = 0; seq < 2; ++seq) {
        for (uint32_t pos = 0; pos < count; ++pos) {
            tokens[seq][pos] = (llama_token) ((1 + seq*count + pos) % n_vocab);
            common_batch_add(prompt, tokens[seq][pos], pos, { seq }, pos + 1 == count);
        }
        if (!decode_tokens(refs[seq], tokens[seq], count)) {
            fprintf(stderr, "%s : reference prompt failed for seq %d\n", __func__, seq);
            llama_batch_free(prompt);
            return false;
        }
    }
    if (llama_decode(ctx, prompt) != 0) {
        fprintf(stderr, "%s : batched prompt failed\n", __func__);
        llama_batch_free(prompt);
        return false;
    }
    llama_batch_free(prompt);

    const llama_pos rollback_pos = (llama_pos) count - rollback;
    for (llama_seq_id seq = 0; seq < 2; ++seq) {
        if (!llama_memory_seq_rm(llama_get_memory(ctx), seq, rollback_pos, -1) ||
                !llama_memory_seq_rm(llama_get_memory(refs[seq]), 0, rollback_pos, -1)) {
            fprintf(stderr, "%s : rollback failed for seq %d\n", __func__, seq);
            return false;
        }
    }

    // Both sequences have pending rollback planes when prepare_batch shrinks
    // the resident allocation. This catches temporary-view accumulation.
    llama_batch replay = llama_batch_init(2, 0, 2);
    auto replay_depths = tag_recurrent_depth(replay, 2, 0);
    for (llama_seq_id seq = 0; seq < 2; ++seq) {
        common_batch_add(replay, tokens[seq][rollback_pos], rollback_pos, { seq }, true);
        if (!decode_one(refs[seq], tokens[seq][rollback_pos], rollback_pos)) {
            fprintf(stderr, "%s : reference replay failed for seq %d\n", __func__, seq);
            llama_batch_free(replay);
            return false;
        }
    }
    if (llama_decode(ctx, replay) != 0) {
        fprintf(stderr, "%s : batched replay failed\n", __func__);
        llama_batch_free(replay);
        return false;
    }
    llama_batch_free(replay);

    if (!compare_last_logits("multi rollback seq0", ctx, 0, refs[0], n_vocab) ||
            !compare_last_logits("multi rollback seq1", ctx, 1, refs[1], n_vocab)) {
        return false;
    }
    if (context_memory_bytes(ctx) == bytes_depth_zero) {
        fprintf(stderr, "%s : first cap0 tick shrank before hysteresis dwell\n", __func__);
        return false;
    }

    llama_batch settle = llama_batch_init(2, 0, 2);
    auto settle_depths = tag_recurrent_depth(settle, 2, 0);
    for (llama_seq_id seq = 0; seq < 2; ++seq) {
        const llama_token token = (llama_token) ((tokens[seq][rollback_pos] + 5) % n_vocab);
        common_batch_add(settle, token, rollback_pos + 1, { seq }, true);
        if (!decode_one(refs[seq], token, rollback_pos + 1)) {
            llama_batch_free(settle);
            return false;
        }
    }
    if (llama_decode(ctx, settle) != 0) {
        llama_batch_free(settle);
        return false;
    }
    llama_batch_free(settle);

    const size_t bytes_depth_zero_again = context_memory_bytes(ctx);
    const bool ok = compare_last_logits("multi rollback seq0", ctx, 0, refs[0], n_vocab) &&
                    compare_last_logits("multi rollback seq1", ctx, 1, refs[1], n_vocab) &&
                    bytes_depth_zero_again == bytes_depth_zero;
    if (!ok && bytes_depth_zero_again != bytes_depth_zero) {
        fprintf(stderr, "%s : storage did not return to depth0 (%zu != %zu)\n",
                __func__, bytes_depth_zero_again, bytes_depth_zero);
    }
    llama_free(ctx);
    llama_free(refs[0]);
    llama_free(refs[1]);
    return ok;
}

static bool decode_shared_rollback_shrink(
        const common_params & params,
        llama_model * model,
        int n_vocab) {
    llama_context * ctx = make_ctx(params, model, 2, true);
    llama_context * ref_rollback = make_ctx(params, model, 1, true);
    llama_context * ref_sibling  = make_ctx(params, model, 1, true);
    if (ctx == nullptr || ref_rollback == nullptr || ref_sibling == nullptr) {
        fprintf(stderr, "%s : failed to create contexts\n", __func__);
        return false;
    }
    const size_t bytes_depth_zero = context_memory_bytes(ctx);
    const uint32_t count = llama_n_rs_seq(ctx) + 1;
    constexpr uint32_t rollback = 3;
    std::vector<llama_token> tokens(count);
    for (uint32_t pos = 0; pos < count; ++pos) {
        tokens[pos] = (llama_token) ((17 + pos) % n_vocab);
    }
    if (!decode_tokens(ctx, tokens, count) ||
            !decode_tokens(ref_rollback, tokens, count) ||
            !decode_tokens(ref_sibling, tokens, count)) {
        fprintf(stderr, "%s : prompt failed\n", __func__);
        return false;
    }
    llama_memory_seq_cp(llama_get_memory(ctx), 0, 1, 0, -1);

    const llama_pos rollback_pos = (llama_pos) count - rollback;
    if (!llama_memory_seq_rm(llama_get_memory(ctx), 0, rollback_pos, -1) ||
            !llama_memory_seq_rm(llama_get_memory(ref_rollback), 0, rollback_pos, -1)) {
        fprintf(stderr, "%s : rollback failed\n", __func__);
        return false;
    }

    // The first cap0 tick must retain the pending plane until find_slot has
    // detached seq0 from the tail shared with seq1.
    llama_batch first = llama_batch_init(2, 0, 2);
    auto first_depths = tag_recurrent_depth(first, 2, 0);
    const llama_token sibling_token = (llama_token) ((tokens.back() + 1) % n_vocab);
    common_batch_add(first, tokens[rollback_pos], rollback_pos, { 0 }, true);
    common_batch_add(first, sibling_token, count, { 1 }, true);
    if (!decode_one(ref_rollback, tokens[rollback_pos], rollback_pos) ||
            !decode_one(ref_sibling, sibling_token, count) ||
            llama_decode(ctx, first) != 0) {
        fprintf(stderr, "%s : first cap0 decode failed\n", __func__);
        llama_batch_free(first);
        return false;
    }
    llama_batch_free(first);
    const size_t bytes_pending_shared = context_memory_bytes(ctx);
    if (bytes_pending_shared <= bytes_depth_zero) {
        fprintf(stderr, "%s : shared rollback plane was not retained until tail detachment\n", __func__);
        return false;
    }
    if (!compare_last_logits("shared rollback seq0", ctx, 0, ref_rollback, n_vocab) ||
            !compare_last_logits("shared sibling seq1", ctx, 1, ref_sibling, n_vocab)) {
        return false;
    }

    // A second cap0 tick has no pending index and arms the lower shrink target
    // without changing either independently split state.
    llama_batch second = llama_batch_init(2, 0, 2);
    auto second_depths = tag_recurrent_depth(second, 2, 0);
    const llama_token rollback_next = (llama_token) ((tokens[rollback_pos] + 3) % n_vocab);
    const llama_token sibling_next  = (llama_token) ((sibling_token + 3) % n_vocab);
    common_batch_add(second, rollback_next, rollback_pos + 1, { 0 }, true);
    common_batch_add(second, sibling_next, count + 1, { 1 }, true);
    if (!decode_one(ref_rollback, rollback_next, rollback_pos + 1) ||
            !decode_one(ref_sibling, sibling_next, count + 1) ||
            llama_decode(ctx, second) != 0) {
        fprintf(stderr, "%s : second cap0 decode failed\n", __func__);
        llama_batch_free(second);
        return false;
    }
    llama_batch_free(second);
    if (!compare_last_logits("shared second seq0", ctx, 0, ref_rollback, n_vocab) ||
            !compare_last_logits("shared second seq1", ctx, 1, ref_sibling, n_vocab)) {
        return false;
    }
    if (context_memory_bytes(ctx) <= bytes_depth_zero) {
        fprintf(stderr, "%s : changed shrink target did not restart dwell\n", __func__);
        return false;
    }

    llama_batch third = llama_batch_init(2, 0, 2);
    auto third_depths = tag_recurrent_depth(third, 2, 0);
    const llama_token rollback_final = (llama_token) ((rollback_next + 3) % n_vocab);
    const llama_token sibling_final  = (llama_token) ((sibling_next + 3) % n_vocab);
    common_batch_add(third, rollback_final, rollback_pos + 2, { 0 }, true);
    common_batch_add(third, sibling_final, count + 2, { 1 }, true);
    if (!decode_one(ref_rollback, rollback_final, rollback_pos + 2) ||
            !decode_one(ref_sibling, sibling_final, count + 2) ||
            llama_decode(ctx, third) != 0) {
        fprintf(stderr, "%s : third cap0 decode failed\n", __func__);
        llama_batch_free(third);
        return false;
    }
    llama_batch_free(third);
    const size_t bytes_depth_zero_again = context_memory_bytes(ctx);
    const bool ok = compare_last_logits("shared shrink seq0", ctx, 0, ref_rollback, n_vocab) &&
                    compare_last_logits("shared shrink seq1", ctx, 1, ref_sibling, n_vocab) &&
                    bytes_depth_zero_again == bytes_depth_zero;
    if (!ok && bytes_depth_zero_again != bytes_depth_zero) {
        fprintf(stderr, "%s : storage did not return to depth0 (%zu != %zu)\n",
                __func__, bytes_depth_zero_again, bytes_depth_zero);
    }
    llama_free(ctx);
    llama_free(ref_rollback);
    llama_free(ref_sibling);
    return ok;
}

static bool decode_tokens(llama_context * ctx, const std::vector<llama_token> & tokens, uint32_t count) {
    llama_batch batch = llama_batch_init(count, 0, 1);
    auto depths = tag_recurrent_depth(batch, count, count - 1);
    for (uint32_t pos = 0; pos < count; ++pos) {
        common_batch_add(batch, tokens[pos], pos, { 0 }, pos + 1 == count);
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static bool decode_one(llama_context * ctx, llama_token tok, llama_pos pos) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    auto depths = tag_recurrent_depth(batch, 1, 0);
    common_batch_add(batch, tok, pos, { 0 }, true);
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.sampling.seed = 1234;
    params.n_predict = 1;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    ggml_backend_load_all();

    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model * model = llama_init->model();
    if (model == nullptr) {
        fprintf(stderr, "%s : failed to init model\n", __func__);
        return 1;
    }

    if (!llama_model_is_recurrent(model) && !llama_model_is_hybrid(model)) {
        fprintf(stderr, "%s : skipping for non-recurrent model\n", __func__);
        return 0;
    }

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    if (!decode_ragged_and_compare(params, model, n_vocab)) {
        return 1;
    }
    if (!decode_multi_rollback_shrink(params, model, n_vocab)) {
        return 1;
    }
    if (!decode_shared_rollback_shrink(params, model, n_vocab)) {
        return 1;
    }

    llama_context * ctx_src = make_ctx(params, model);
    llama_context * ctx_dst = make_ctx(params, model);
    if (ctx_src == nullptr || ctx_dst == nullptr) {
        fprintf(stderr, "%s : failed to init contexts\n", __func__);
        return 1;
    }

    if (llama_n_rs_seq(ctx_src) == 0) {
        fprintf(stderr, "%s : skipping because n_rs_seq is disabled\n", __func__);
        llama_free(ctx_src);
        llama_free(ctx_dst);
        return 0;
    }

    std::vector<llama_token> tokens;
    if (llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_NONE) {
        tokens = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    } else {
        tokens = common_tokenize(ctx_src, "The quick brown fox jumps over the lazy dog", true);
    }
    const uint32_t n_rs_seq = llama_n_rs_seq(ctx_src);
    constexpr uint32_t n_rollback = 3;
    if (n_rs_seq < n_rollback) {
        fprintf(stderr, "%s : skipping because n_rs_seq is too small\n", __func__);
        llama_free(ctx_src);
        llama_free(ctx_dst);
        return 0;
    }
    if (tokens.empty()) {
        fprintf(stderr, "%s : not enough prompt tokens\n", __func__);
        return 1;
    }
    tokens.resize(n_rs_seq + 1, tokens.back());

    const uint32_t  n_tokens     = tokens.size();
    const llama_pos rollback_pos = (llama_pos) n_tokens - n_rollback;

    // Decode the full prompt on the source, then roll back three positions.
    // Replaying them crosses DSV4's ratio-4 compressor boundary.
    // Rollback leaves the recurrent memory in a snapshot state (rs_idx != 0).
    if (!decode_tokens(ctx_src, tokens, n_tokens)) {
        fprintf(stderr, "%s : failed to decode prompt\n", __func__);
        return 1;
    }
    if (!llama_memory_seq_rm(llama_get_memory(ctx_src), 0, rollback_pos, -1)) {
        fprintf(stderr, "%s : rollback failed\n", __func__);
        return 1;
    }

    // Save the rolled-back state and restore it into a fresh context.
    common_prompt_checkpoint ckpt;
    ckpt.update_tgt(ctx_src, 0, 0);
    ckpt.load_tgt(ctx_dst, 0, 0);

    constexpr float eps = 1e-5f;
    std::vector<std::vector<float>> logits_src_replay(n_rollback);
    const auto replay_and_compare = [&](const char * mode) {
        for (uint32_t i = 0; i < n_rollback; ++i) {
            const llama_pos pos = rollback_pos + i;
            if (!decode_one(ctx_src, tokens[pos], pos) ||
                !decode_one(ctx_dst, tokens[pos], pos)) {
                fprintf(stderr, "%s : %s replay failed at position %d\n", __func__, mode, pos);
                return false;
            }

            const float * logits_src = llama_get_logits_ith(ctx_src, 0);
            const float * logits_dst = llama_get_logits_ith(ctx_dst, 0);
            if (logits_src == nullptr || logits_dst == nullptr) {
                fprintf(stderr, "%s : missing %s logits at position %d\n", __func__, mode, pos);
                return false;
            }

            logits_src_replay[i].assign(logits_src, logits_src + n_vocab);
            for (int token = 0; token < n_vocab; ++token) {
                if (std::fabs(logits_src[token] - logits_dst[token]) > eps) {
                    fprintf(stderr, "%s : %s logits mismatch at position %d, token %d (%g != %g)\n",
                            __func__, mode, pos, token, (double) logits_src[token], (double) logits_dst[token]);
                    return false;
                }
            }
        }
        return true;
    };
    if (!replay_and_compare("full")) {
        return 1;
    }

    if (!llama_memory_seq_rm(llama_get_memory(ctx_src), 0, rollback_pos, -1) ||
        !llama_memory_seq_rm(llama_get_memory(ctx_dst), 0, rollback_pos, -1)) {
        fprintf(stderr, "%s : partial rollback failed\n", __func__);
        return 1;
    }

    constexpr llama_state_seq_flags partial_flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    common_prompt_checkpoint ckpt_partial;
    ckpt_partial.update_tgt(ctx_src, 0, partial_flags);
    ckpt_partial.load_tgt(ctx_dst, 0, partial_flags);

    if (!replay_and_compare("partial")) {
        return 1;
    }

    // Repeat the load into a context that already has its own rollback state:
    // groups 1..n_rs_seq hold a different prompt's history, and rs_idx[0] is
    // non-zero at load time. The restore must wipe that state and still match.
    llama_context * ctx_dirty = make_ctx(params, model);
    if (ctx_dirty == nullptr) {
        fprintf(stderr, "%s : failed to init dirty ctx\n", __func__);
        return 1;
    }

    std::vector<llama_token> noise = tokens;
    for (auto & t : noise) {
        t = (t + 1) % n_vocab;
        if (t < 0) {
            t = 0;
        }
    }
    if (!decode_tokens(ctx_dirty, noise, n_tokens)) {
        fprintf(stderr, "%s : dirty prompt decode failed\n", __func__);
        return 1;
    }
    if (!llama_memory_seq_rm(llama_get_memory(ctx_dirty), 0, rollback_pos, -1)) {
        fprintf(stderr, "%s : dirty rollback failed\n", __func__);
        return 1;
    }

    ckpt.load_tgt(ctx_dirty, 0, 0);

    for (uint32_t i = 0; i < n_rollback; ++i) {
        const llama_pos pos = rollback_pos + i;
        if (!decode_one(ctx_dirty, tokens[pos], pos)) {
            fprintf(stderr, "%s : dirty replay failed at position %d\n", __func__, pos);
            return 1;
        }

        const float * logits_dirty = llama_get_logits_ith(ctx_dirty, 0);
        if (logits_dirty == nullptr) {
            fprintf(stderr, "%s : missing dirty logits at position %d\n", __func__, pos);
            return 1;
        }

        for (int token = 0; token < n_vocab; ++token) {
            if (std::fabs(logits_src_replay[i][token] - logits_dirty[token]) > eps) {
                fprintf(stderr, "%s : dirty-ctx logits mismatch at position %d, token %d (%g != %g)\n",
                        __func__, pos, token, (double) logits_src_replay[i][token], (double) logits_dirty[token]);
                return 1;
            }
        }
    }

    fprintf(stderr, "%s : recurrent rollback checkpoint restored successfully\n", __func__);
    llama_free(ctx_src);
    llama_free(ctx_dst);
    llama_free(ctx_dirty);
    return 0;
}
