#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#    include <shellapi.h>
#endif

constexpr uint32_t TEST_RS_SEQ_MIN = 3;
constexpr uint32_t TEST_RS_SEQ_MAX = (uint32_t) std::numeric_limits<int32_t>::max() - 1;

#ifdef _WIN32
struct test_utf8_argv {
    std::vector<std::string> buf;
    std::vector<char *> ptrs;
};

static test_utf8_argv make_test_utf8_argv() {
    test_utf8_argv out;
    int wargc = 0;
    LPWSTR * wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);
    if (wargv == nullptr) {
        return out;
    }

    out.buf.reserve(wargc);
    for (int i = 0; i < wargc; ++i) {
        const int n = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, wargv[i], -1, nullptr, 0, nullptr, nullptr);
        if (n <= 0) {
            out.buf.emplace_back();
            continue;
        }
        auto & s = out.buf.emplace_back();
        s.resize((size_t) n - 1);
        (void) WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, s.data(), n, nullptr, nullptr);
    }
    LocalFree(wargv);

    out.ptrs.reserve(out.buf.size());
    for (auto & s : out.buf) {
        out.ptrs.push_back(s.data());
    }
    return out;
}
#endif

static llama_context * make_ctx(const common_params & params, llama_model * model, uint32_t test_rs_seq) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = 1;
    cparams.n_rs_seq  = test_rs_seq;
    cparams.n_batch   = std::max(cparams.n_batch,  (uint32_t) (cparams.n_rs_seq + 1));
    cparams.n_ubatch  = std::max(cparams.n_ubatch, (uint32_t) (cparams.n_rs_seq + 1));
    return llama_init_from_model_with_recurrent_cache_type(
            model, cparams, common_params_get_recurrent_cache_type(params));
}

static void print_test_usage(int, char **) {
    fprintf(stderr, "\nrecurrent rollback test options:\n");
    fprintf(stderr, "  --full-restore-only  test the valid full-checkpoint paths and skip the second rollback\n");
    fprintf(stderr, "  --test-rs-seq N      recurrent snapshot count for test contexts (range: %u-%u, default: 8)\n",
            TEST_RS_SEQ_MIN, TEST_RS_SEQ_MAX);
}

static bool decode_tokens(llama_context * ctx, const std::vector<llama_token> & tokens, uint32_t count) {
    llama_batch batch = llama_batch_init(count, 0, 1);
    for (uint32_t pos = 0; pos < count; ++pos) {
        common_batch_add(batch, tokens[pos], pos, { 0 }, pos + 1 == count);
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static bool decode_one(llama_context * ctx, llama_token tok, llama_pos pos) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    common_batch_add(batch, tok, pos, { 0 }, true);
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

#ifdef _WIN32
    auto utf8_argv = make_test_utf8_argv();
    if ((int) utf8_argv.ptrs.size() == argc) {
        argv = utf8_argv.ptrs.data();
    }
#endif

    // The complete test also exercises a second rollback/replay cycle. Some
    // architectures currently fail that independent baseline behavior. Keep a
    // narrow mode for validating one rollback plus a full checkpoint roundtrip.
    bool full_restore_only = false;
    uint32_t test_rs_seq = 8;
    std::vector<char *> common_argv;
    common_argv.reserve(argc);
    common_argv.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--full-restore-only") == 0) {
            full_restore_only = true;
        } else if (strcmp(argv[i], "--test-rs-seq") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "error: --test-rs-seq requires a value\n");
                return 1;
            }

            const char * value_arg = argv[i];
            const bool all_digits = value_arg[0] != '\0' && std::all_of(
                    value_arg, value_arg + strlen(value_arg),
                    [](unsigned char c) { return std::isdigit(c) != 0; });
            char * end = nullptr;
            errno = 0;
            const unsigned long value = strtoul(value_arg, &end, 10);
            if (!all_digits || errno != 0 || end == value_arg || *end != '\0' ||
                value < TEST_RS_SEQ_MIN || value > TEST_RS_SEQ_MAX) {
                fprintf(stderr, "error: --test-rs-seq must be an integer in [3, %u]\n",
                        TEST_RS_SEQ_MAX);
                return 1;
            }
            test_rs_seq = (uint32_t) value;
        } else {
            common_argv.push_back(argv[i]);
        }
    }

    common_params params;
    params.sampling.seed = 1234;
    params.n_predict = 1;

    common_init();

    if (!common_params_parse(
                (int) common_argv.size(), common_argv.data(), params, LLAMA_EXAMPLE_COMMON, print_test_usage)) {
        return 1;
    }

    ggml_backend_load_all();

    common_init_result_ptr llama_init = common_init_from_params(params, true);
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

    llama_context * ctx_src = make_ctx(params, model, test_rs_seq);
    llama_context * ctx_dst = make_ctx(params, model, test_rs_seq);
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
    std::vector<std::vector<float>> logits_full_replay(n_rollback);
    const auto replay_and_compare = [&](const char * mode, bool capture_full) {
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

            if (capture_full) {
                logits_full_replay[i].assign(logits_src, logits_src + n_vocab);
            } else {
                for (int token = 0; token < n_vocab; ++token) {
                    if (std::fabs(logits_full_replay[i][token] - logits_src[token]) > eps) {
                        fprintf(stderr, "%s : %s source differs from full replay at position %d, token %d (%g != %g)\n",
                                __func__, mode, pos, token,
                                (double) logits_full_replay[i][token], (double) logits_src[token]);
                        return false;
                    }
                }
            }
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
    if (!replay_and_compare("full", true)) {
        return 1;
    }

    // The narrow GPU gate no longer needs the fresh destination. Release it
    // before allocating the dirty destination so at most two contexts coexist.
    if (full_restore_only) {
        llama_free(ctx_dst);
        ctx_dst = nullptr;
    }

    // Repeat the load into a context that already has its own rollback state:
    // groups 1..n_rs_seq hold a different prompt's history, and rs_idx[0] is
    // non-zero at load time. The restore must wipe that state and still match.
    llama_context * ctx_dirty = make_ctx(params, model, test_rs_seq);
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
            if (std::fabs(logits_full_replay[i][token] - logits_dirty[token]) > eps) {
                fprintf(stderr, "%s : dirty-ctx logits mismatch at position %d, token %d (%g != %g)\n",
                        __func__, pos, token, (double) logits_full_replay[i][token], (double) logits_dirty[token]);
                return 1;
            }
        }
    }

    llama_free(ctx_dirty);

    if (full_restore_only) {
        fprintf(stderr, "%s : recurrent full rollback checkpoint restored successfully\n", __func__);
        llama_free(ctx_src);
        return 0;
    }

    // A second rollback after replaying each token in its own decode call does
    // not have a valid multi-token snapshot history on all recurrent models.
    // Keep this coverage in the complete test, but do not use it for the narrow
    // full-checkpoint correctness gate above.
    if (!llama_memory_seq_rm(llama_get_memory(ctx_src), 0, rollback_pos, -1) ||
        !llama_memory_seq_rm(llama_get_memory(ctx_dst), 0, rollback_pos, -1)) {
        fprintf(stderr, "%s : partial rollback failed\n", __func__);
        return 1;
    }

    constexpr llama_state_seq_flags partial_flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    common_prompt_checkpoint ckpt_partial;
    ckpt_partial.update_tgt(ctx_src, 0, partial_flags);
    ckpt_partial.load_tgt(ctx_dst, 0, partial_flags);

    if (!replay_and_compare("partial", false)) {
        return 1;
    }

    fprintf(stderr, "%s : recurrent rollback checkpoint restored successfully\n", __func__);
    llama_free(ctx_src);
    llama_free(ctx_dst);
    return 0;
}
