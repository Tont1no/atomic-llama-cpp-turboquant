#include "arg.h"
#include "common.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"
#include "llama.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <clocale>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace {

struct sequence_result {
    llama_tokens draft;
    llama_tokens accepted;
    std::vector<uint32_t> draft_logits;
    std::vector<uint32_t> target_logits;
    std::vector<uint8_t> draft_state;
    std::vector<uint8_t> verified_state;
};

struct scenario_result {
    std::string name;
    std::vector<sequence_result> sequences;
};

struct run_result {
    std::vector<scenario_result> scenarios;
};

enum class test_arm : int {
    none,
    fallback,
    fused,
};

struct marker_counts {
    std::atomic<uint64_t> active  {0};
    std::atomic<uint64_t> fallback{0};
    std::atomic<uint64_t> raw_pass{0};
};

static std::atomic<test_arm> g_arm{test_arm::none};
static marker_counts g_markers[2];

static void test_log_callback(enum ggml_log_level level, const char * text, void *) {
    const test_arm arm = g_arm.load(std::memory_order_relaxed);
    if (arm != test_arm::none && text != nullptr) {
        marker_counts & counts = g_markers[arm == test_arm::fallback ? 0 : 1];
        if (std::strstr(text, "DFlash-K FWHT cache fusion active") != nullptr) {
            counts.active.fetch_add(1, std::memory_order_relaxed);
        }
        if (std::strstr(text, "DFlash-K model fusion fallback") != nullptr) {
            counts.fallback.fetch_add(1, std::memory_order_relaxed);
        }
        if (std::strstr(text, "DFlash-K raw cache validation PASS") != nullptr) {
            counts.raw_pass.fetch_add(1, std::memory_order_relaxed);
        }
    }
    common_log_default_callback(level, text, nullptr);
}

static void set_process_env(const char * name, const char * value) {
#ifdef _WIN32
    if (_putenv_s(name, value) != 0) {
        GGML_ABORT("failed to set environment variable %s", name);
    }
#else
    if (setenv(name, value, 1) != 0) {
        GGML_ABORT("failed to set environment variable %s", name);
    }
#endif
}

static std::vector<uint32_t> get_logits_bits(llama_context * ctx, int32_t output, int32_t n_vocab) {
    const float * logits = llama_get_logits_ith(ctx, output);
    GGML_ASSERT(logits != nullptr);
    std::vector<uint32_t> bits(n_vocab);
    std::memcpy(bits.data(), logits, bits.size()*sizeof(uint32_t));
    return bits;
}

static std::vector<uint8_t> get_sequence_state(llama_context * ctx, llama_seq_id seq_id) {
    std::vector<uint8_t> state(llama_state_seq_get_size(ctx, seq_id));
    const size_t copied = llama_state_seq_get_data(ctx, state.data(), state.size(), seq_id);
    GGML_ASSERT(copied == state.size());
    return state;
}

static llama_tokens make_prompt(
        llama_context * ctx, const std::string & seed_text, size_t desired_tokens) {
    llama_tokens seed = common_tokenize(ctx, seed_text, true, true);
    GGML_ASSERT(seed.size() >= 2 && desired_tokens >= 2);

    llama_tokens result;
    result.reserve(desired_tokens);
    result.push_back(seed.front());
    for (size_t i = 1; result.size() < desired_tokens; ++i) {
        result.push_back(seed[1 + (i - 1)%(seed.size() - 1)]);
    }
    return result;
}

static scenario_result run_scenario(
        const char * name,
        llama_model * model_tgt,
        llama_context * ctx_tgt,
        llama_context * ctx_dft,
        common_params_speculative & spec_params,
        const std::vector<size_t> & prompt_lengths) {
    const int32_t n_seq = (int32_t) prompt_lengths.size();
    GGML_ASSERT(n_seq > 0);

    // Keep the same non-monotonic group order in both arms so unified-cache
    // row allocation sees realistic inter-sequence permutation as well as
    // ragged lengths.
    std::vector<int32_t> batch_order(n_seq);
    for (int32_t i = 0; i < n_seq; ++i) {
        batch_order[i] = (3 + 5*i)%n_seq;
    }

    llama_memory_clear(llama_get_memory(ctx_tgt), true);
    llama_memory_clear(llama_get_memory(ctx_dft), true);

    // Construction enables target layer-input extraction. It must precede the
    // first target decode or DFlash cannot receive prompt features.
    common_speculative_ptr spec(common_speculative_init(spec_params, n_seq));
    GGML_ASSERT(spec != nullptr);

    std::vector<llama_tokens> prompts(n_seq);
    std::vector<llama_tokens> prefixes(n_seq);
    std::vector<llama_tokens> drafts(n_seq);
    const std::array<const char *, 8> seeds = {
        "Implement a deterministic bounded queue with cancellation.",
        "Explain why an index can wrap in a unified cache.",
        "Find the race in this concurrent request scheduler.",
        "Write a compact test for quantized cache injection.",
        "Review a CUDA kernel for out of bounds writes.",
        "Compare two speculative decoding acceptance traces.",
        "Design a reproducible ragged batch benchmark.",
        "Summarize the invariants of sliding window attention.",
    };

    size_t n_prompt_rows = 0;
    for (int32_t seq = 0; seq < n_seq; ++seq) {
        prompts[seq] = make_prompt(ctx_tgt, seeds[seq], prompt_lengths[seq]);
        prefixes[seq].assign(prompts[seq].begin(), prompts[seq].end() - 1);
        n_prompt_rows += prefixes[seq].size();
    }

    llama_batch prompt_batch = llama_batch_init((int32_t) n_prompt_rows, 0, n_seq);
    for (int32_t seq : batch_order) {
        for (size_t pos = 0; pos < prefixes[seq].size(); ++pos) {
            common_batch_add(prompt_batch, prefixes[seq][pos], (llama_pos) pos, {seq}, false);
        }
    }

    if (llama_decode(ctx_tgt, prompt_batch) != 0) {
        GGML_ABORT("target prompt decode failed for %s", name);
    }

    if (!common_speculative_process(spec.get(), prompt_batch)) {
        GGML_ABORT("DFlash prompt processing failed for %s", name);
    }
    llama_batch_free(prompt_batch);

    for (int32_t seq = 0; seq < n_seq; ++seq) {
        common_speculative_begin(spec.get(), seq, prefixes[seq]);
        common_speculative_get_draft_params(spec.get(), seq) = {
            /* .drafting = */ true,
            /* .n_max    = */ spec_params.draft.n_max,
            /* .n_past   = */ (llama_pos) prefixes[seq].size(),
            /* .id_last  = */ prompts[seq].back(),
            /* .prompt   = */ &prefixes[seq],
            /* .result   = */ &drafts[seq],
        };
    }
    common_speculative_draft(spec.get());

    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model_tgt));
    GGML_ASSERT(n_vocab == llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx_dft))));
    scenario_result result;
    result.name = name;
    result.sequences.resize(n_seq);

    int32_t draft_output = 0;
    for (int32_t seq = 0; seq < n_seq; ++seq) {
        result.sequences[seq].draft = drafts[seq];
        GGML_ASSERT(drafts[seq].size() == (size_t) spec_params.draft.n_max);
        // DFlash emits [last, mask_0, mask_1, ...] for every sequence.  The
        // predicted draft logits start at mask_0.
        ++draft_output;
        for (size_t i = 0; i < drafts[seq].size(); ++i) {
            const auto bits = get_logits_bits(ctx_dft, draft_output++, n_vocab);
            result.sequences[seq].draft_logits.insert(
                    result.sequences[seq].draft_logits.end(), bits.begin(), bits.end());
        }
        result.sequences[seq].draft_state = get_sequence_state(ctx_dft, seq);

        // Match the server/example lifecycle: the temporary DFlash noise block
        // is discarded before verified target features are injected.
        const llama_pos n_past = (llama_pos) prefixes[seq].size();
        GGML_ASSERT(llama_memory_seq_rm(llama_get_memory(ctx_dft), seq, n_past, -1));
    }

    size_t n_verify_rows = 0;
    for (const auto & draft : drafts) {
        n_verify_rows += 1 + draft.size();
    }
    llama_batch verify_batch = llama_batch_init((int32_t) n_verify_rows, 0, n_seq);
    std::vector<std::vector<int>> output_indices(n_seq);

    int32_t output = 0;
    for (int32_t seq : batch_order) {
        const llama_pos n_past = (llama_pos) prefixes[seq].size();
        common_batch_add(verify_batch, prompts[seq].back(), n_past, {seq}, true);
        output_indices[seq].push_back(output++);
        for (size_t i = 0; i < drafts[seq].size(); ++i) {
            common_batch_add(verify_batch, drafts[seq][i], n_past + (llama_pos) i + 1, {seq}, true);
            output_indices[seq].push_back(output++);
        }
    }

    if (llama_decode(ctx_tgt, verify_batch) != 0) {
        GGML_ABORT("target verification decode failed for %s", name);
    }
    if (!common_speculative_process(spec.get(), verify_batch)) {
        GGML_ABORT("DFlash verification processing failed for %s", name);
    }
    llama_batch_free(verify_batch);

    for (int32_t seq = 0; seq < n_seq; ++seq) {
        result.sequences[seq].verified_state = get_sequence_state(ctx_dft, seq);
    }

    common_params_sampling sampling;
    sampling.seed = 0xDFA57u;
    sampling.temp = 0.0f;
    sampling.top_k = 0;
    sampling.top_p = 1.0f;
    sampling.min_p = 0.0f;
    sampling.samplers = {COMMON_SAMPLER_TYPE_TEMPERATURE};

    for (int32_t seq = 0; seq < n_seq; ++seq) {
        for (int idx : output_indices[seq]) {
            const auto bits = get_logits_bits(ctx_tgt, idx, n_vocab);
            result.sequences[seq].target_logits.insert(
                    result.sequences[seq].target_logits.end(), bits.begin(), bits.end());
        }

        common_sampler_ptr sampler(common_sampler_init(model_tgt, sampling));
        GGML_ASSERT(sampler != nullptr);
        for (llama_token token : prompts[seq]) {
            common_sampler_accept(sampler.get(), token, false);
        }
        result.sequences[seq].accepted = common_sampler_sample_and_accept_n(
                sampler.get(), ctx_tgt, output_indices[seq], drafts[seq]);
        GGML_ASSERT(!result.sequences[seq].accepted.empty());
    }

    LOG_INF("dflash deterministic %s: sequences=%d prompt_rows=%zu verify_rows=%zu\n",
            name, n_seq, n_prompt_rows, n_verify_rows);
    return result;
}

static run_result run_once(const common_params & base_params, bool fusion) {
    set_process_env("GGML_CUDA_DFLASH_K_FUSION", fusion ? "1" : "0");
    set_process_env("GGML_CUDA_DFLASH_K_VALIDATE", fusion ? "1" : "0");

    common_params params = base_params;
    params.speculative.draft.ctx_tgt = nullptr;
    params.speculative.draft.ctx_dft = nullptr;

    common_init_result_ptr target = common_init_from_params(params);
    GGML_ASSERT(target != nullptr && target->model() != nullptr && target->context() != nullptr);

    common_params params_dft = common_base_params_to_speculative(params);
    common_speculative_init_result_ptr draft = common_speculative_init_from_params(
            params_dft, target->model(), target->context());
    GGML_ASSERT(draft != nullptr && draft->model() != nullptr && draft->context() != nullptr);

    params.speculative.draft.ctx_tgt = target->context();
    params.speculative.draft.ctx_dft = draft->context();

    run_result result;
    result.scenarios.push_back(run_scenario(
            "ragged-n4", target->model(), target->context(), draft->context(),
            params.speculative, {73, 72, 77, 68}));
    result.scenarios.push_back(run_scenario(
            "ragged-n8", target->model(), target->context(), draft->context(),
            params.speculative, {73, 72, 77, 68, 73, 76, 72, 73}));
    return result;
}

template<typename T>
static bool compare_vector(
        const char * scenario, int32_t seq, const char * field,
        const std::vector<T> & fallback, const std::vector<T> & fused) {
    if (fallback == fused) {
        return true;
    }
    size_t first = 0;
    while (first < fallback.size() && first < fused.size() && fallback[first] == fused[first]) {
        ++first;
    }
    LOG_ERR("DFlash deterministic mismatch: scenario=%s seq=%d field=%s first=%zu fallback_size=%zu fused_size=%zu\n",
            scenario, seq, field, first, fallback.size(), fused.size());
    return false;
}

static bool compare_runs(const run_result & fallback, const run_result & fused) {
    if (fallback.scenarios.size() != fused.scenarios.size()) {
        return false;
    }
    bool ok = true;
    for (size_t s = 0; s < fallback.scenarios.size(); ++s) {
        const auto & a = fallback.scenarios[s];
        const auto & b = fused.scenarios[s];
        if (a.name != b.name || a.sequences.size() != b.sequences.size()) {
            LOG_ERR("DFlash deterministic scenario shape mismatch at %zu\n", s);
            ok = false;
            continue;
        }
        for (size_t seq = 0; seq < a.sequences.size(); ++seq) {
            ok &= compare_vector(a.name.c_str(), (int32_t) seq, "draft_tokens",
                    a.sequences[seq].draft, b.sequences[seq].draft);
            ok &= compare_vector(a.name.c_str(), (int32_t) seq, "draft_logits",
                    a.sequences[seq].draft_logits, b.sequences[seq].draft_logits);
            ok &= compare_vector(a.name.c_str(), (int32_t) seq, "target_logits",
                    a.sequences[seq].target_logits, b.sequences[seq].target_logits);
            ok &= compare_vector(a.name.c_str(), (int32_t) seq, "draft_kv_state",
                    a.sequences[seq].draft_state, b.sequences[seq].draft_state);
            ok &= compare_vector(a.name.c_str(), (int32_t) seq, "verified_kv_state",
                    a.sequences[seq].verified_state, b.sequences[seq].verified_state);
            ok &= compare_vector(a.name.c_str(), (int32_t) seq, "accepted_prefix_and_final",
                    a.sequences[seq].accepted, b.sequences[seq].accepted);
        }
    }
    return ok;
}

} // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    common_init();

    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }
    if (params.model.path.empty() || params.speculative.draft.mparams.path.empty()) {
        LOG_ERR("test requires -m <target.gguf> -md <dflash.gguf> --spec-type draft-dflash\n");
        return 1;
    }

    params.n_parallel = 8;
    params.kv_unified = true;
    params.n_batch = std::max(params.n_batch, 2048);
    params.n_ubatch = std::max(params.n_ubatch, 512);
    params.n_ctx = std::max(params.n_ctx, 65536);
    params.n_outputs_max = common_speculative_get_output_limits(
            params.n_batch, params.n_parallel, 2).total;
    params.n_outputs_max_per_seq = common_speculative_get_output_limits(
            params.n_batch, params.n_parallel, 2).per_seq;
    params.speculative.types = {COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH};
    params.speculative.draft.n_max = 2;
    params.speculative.draft.n_min = 0;
    params.speculative.draft.p_min = 0.0f;
    params.speculative.draft.backend_sampling = false;
    params.speculative.draft.seed = 0xDFA57u;
    params.sampling.backend_sampling = false;
    params.sampling.seed = 0xDFA57u;
    params.sampling.temp = 0.0f;

    if (params.speculative.draft.cache_type_k != GGML_TYPE_Q4_0 &&
        params.speculative.draft.cache_type_k != GGML_TYPE_Q8_0) {
        LOG_ERR("test requires -ctkd q4_0 or -ctkd q8_0 so the exact FWHT cache writer is exercised\n");
        return 1;
    }

    // Raw validation synchronizes after every matching K write and therefore
    // deliberately excludes CUDA graph capture from this correctness test.
    set_process_env("GGML_CUDA_DISABLE_GRAPHS", "1");

    llama_backend_init();
    llama_numa_init(params.numa);
    llama_log_set(test_log_callback, nullptr);

    LOG_INF("DFlash deterministic regression: fallback run\n");
    g_arm.store(test_arm::fallback, std::memory_order_relaxed);
    // This is intentionally the first DFlash decode after spec construction.
    // It regresses first-use scheduler reservation before external feature
    // aliases are installed, as well as the OFF cache-writer path itself.
    const run_result fallback = run_once(params, false);
    LOG_INF("DFlash deterministic regression: fused+raw-validation run\n");
    g_arm.store(test_arm::fused, std::memory_order_relaxed);
    const run_result fused = run_once(params, true);
    g_arm.store(test_arm::none, std::memory_order_relaxed);

    const uint64_t off_active   = g_markers[0].active.load(std::memory_order_relaxed);
    const uint64_t off_fallback = g_markers[0].fallback.load(std::memory_order_relaxed);
    const uint64_t off_raw      = g_markers[0].raw_pass.load(std::memory_order_relaxed);
    const uint64_t on_active    = g_markers[1].active.load(std::memory_order_relaxed);
    const uint64_t on_fallback  = g_markers[1].fallback.load(std::memory_order_relaxed);
    const uint64_t on_raw       = g_markers[1].raw_pass.load(std::memory_order_relaxed);
    LOG_INF("DFlash deterministic markers: OFF active=%llu fallback=%llu raw=%llu; ON active=%llu fallback=%llu raw=%llu\n",
            (unsigned long long) off_active, (unsigned long long) off_fallback, (unsigned long long) off_raw,
            (unsigned long long) on_active, (unsigned long long) on_fallback, (unsigned long long) on_raw);

    const bool markers_ok = off_active == 0 && off_fallback > 0 && off_raw == 0 &&
                            on_active > 0 && on_fallback == 0 && on_raw > 0;
    const bool ok = markers_ok && compare_runs(fallback, fused);
    llama_backend_free();

    if (!ok) {
        LOG_ERR("DFlash deterministic regression FAILED\n");
        return 2;
    }
    LOG_INF("DFlash deterministic regression PASS: raw K validation plus draft/target logits, drafts, and acceptance match\n");
    return 0;
}
