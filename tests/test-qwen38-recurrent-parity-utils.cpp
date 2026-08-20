#include "qwen38-recurrent-parity-utils.h"
#include "llama.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace qwen38_recurrent_parity;

static void require(bool condition) {
    if (!condition) {
        std::abort();
    }
}

int main() {
    require(json_escape("quote=\" slash=\\ line=\n\x01") ==
            "quote=\\\" slash=\\\\ line=\\n\\u0001");

    for (size_t row = 0; row < 736; ++row) {
        require(batch_row_requires_logits(row, 736, false) == (row == 735));
        require(batch_row_requires_logits(row, 736, true));
    }
    bool invalid_row_rejected = false;
    try {
        (void) batch_row_requires_logits(736, 736, false);
    } catch (const std::out_of_range &) {
        invalid_row_rejected = true;
    }
    require(invalid_row_rejected);

    {
        const std::array<float, 5> logits = { 2.0f, 9.0f, 9.0f, -3.0f, 8.5f };
        const auto result = select_top2(logits.data(), logits.size());
        // Stable tie handling preserves the lower token ID, matching a forward scan.
        require(result.top1 == 1);
        require(result.top2 == 2);
        require(result.logit1 == 9.0f);
        require(result.logit2 == 9.0f);
        require(result.margin() == 0.0f);

        bool rejected = false;
        try {
            const std::array<float, 3> invalid = { 1.0f, NAN, 0.0f };
            (void) select_top2(invalid.data(), invalid.size());
        } catch (const std::runtime_error &) {
            rejected = true;
        }
        require(rejected);
    }

    {
        // The P1 server trace is greedy, but it is not raw argmax: the
        // request's ignore_eos=true inserts -inf EOG biases before temp=0.
        // Pin that production ordering so the reference oracle cannot regress
        // to the raw-logit comparison that caused attempt 4 to stop at token 37.
        std::array<llama_token_data, 3> candidates = {{
            { 0, 12.0f, 0.0f }, // synthetic raw EOG winner
            { 1, 11.0f, 0.0f }, // expected production token
            { 2,  3.0f, 0.0f },
        }};
        llama_token_data_array cur = { candidates.data(), candidates.size(), -1, false };
        const llama_logit_bias eog_bias = { 0, -INFINITY };

        llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(chain, llama_sampler_init_logit_bias(3, 1, &eog_bias));
        llama_sampler_chain_add(chain, llama_sampler_init_temp_ext(0.0f, 0.0f, 1.0f));
        llama_sampler_chain_add(chain, llama_sampler_init_dist(20260819));
        llama_sampler_apply(chain, &cur);

        require(cur.selected >= 0);
        require(cur.data[cur.selected].id == 1);
        require(cur.data[0].logit == -INFINITY);
        llama_sampler_free(chain);
    }

    {
        const auto tokens = parse_token_csv("271,2064,0,248077");
        require((tokens == std::vector<int32_t>{ 271, 2064, 0, 248077 }));

        bool rejected = false;
        try {
            (void) parse_token_csv("1,,2");
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        require(rejected);
    }

    {
        static const uint8_t bytes[] = { 'h', 'e', 'l', 'l', 'o' };
        require(fnv1a64(bytes, sizeof(bytes)) == UINT64_C(0xa430d84680aabd0b));
        require(float_bits(1.0f) == UINT32_C(0x3f800000));
    }

    {
        const std::vector<uint64_t> scalar = { 1, 2, 3, 4 };
        const auto equal = compare_boundary_rows(scalar, scalar, 176);
        require(equal.first_mismatch_prediction == -1);
        require(equal.mismatch_count == 0);
        require(equal.scalar_window_hash == equal.batch_window_hash);

        const std::vector<uint64_t> batch = { 1, 9, 3, 8 };
        const auto different = compare_boundary_rows(scalar, batch, 176);
        require(different.first_mismatch_prediction == 177);
        require(different.mismatch_count == 2);
        require(different.scalar_window_hash != different.batch_window_hash);

        bool rejected = false;
        try {
            (void) compare_boundary_rows({ 1 }, { 1, 2 }, 176);
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        require(rejected);
    }

    {
        boundary_row_layout layout;
        // Real comparable Qwen3.5 layer-0 shapes: scalar and W8 activation
        // matrices keep n_embd on ne[0] and token rows on ne[1].
        require(canonical_f32_boundary_layout({ 5120, 1, 1, 1 }, 5120, 1, layout));
        require(layout.features == 5120);
        require(layout.rows == 1);
        require(layout.row_bytes == 20480);
        require(layout.total_bytes == 20480);
        require(canonical_f32_boundary_layout({ 5120, 8, 1, 1 }, 5120, 8, layout));
        require(layout.rows == 8);
        require(layout.total_bytes == 163840);

        // The recurrent GDN core is also named attn_output, but has distinct
        // [S_v,H_v,tokens,seqs] shapes. Scalar singleton axes must not make it
        // eligible as the post-projection attention boundary.
        require(!canonical_f32_boundary_layout({ 128, 48, 1, 1 }, 5120, 1, layout));
        require(!canonical_f32_boundary_layout({ 128, 48, 8, 1 }, 5120, 8, layout));

        // Do not silently flatten/remap a token or sequence axis.
        require(!canonical_f32_boundary_layout({ 5120, 1, 8, 1 }, 5120, 8, layout));
        require(!canonical_f32_boundary_layout({ 5120, 8, 1, 2 }, 5120, 8, layout));
        require(!canonical_f32_boundary_layout({ 5120, 7, 1, 1 }, 5120, 8, layout));

        // Pinned Qwen3.8 layer-0 internal token-slice layouts.
        require(internal_f32_token_layout({ 5120, 1, 1, 1 }, 1, { 5120, 1 }, 1, layout));
        require(internal_f32_token_layout({ 10240, 8, 1, 1 }, 1, { 10240, 1 }, 8, layout));
        require(layout.row_bytes == 40960);
        require(internal_f32_token_layout({ 1, 48, 8, 1 }, 2, { 1, 48 }, 8, layout));
        require(layout.row_bytes == 192);
        require(internal_f32_token_layout({ 128, 16, 8, 1 }, 2, { 128, 16 }, 8, layout));
        require(layout.row_bytes == 8192);
        require(internal_f32_token_layout({ 128, 48, 8, 1 }, 2, { 128, 48 }, 8, layout));
        require(layout.row_bytes == 24576);

        require(!internal_f32_token_layout({ 10239, 8, 1, 1 }, 1, { 10240, 1 }, 8, layout));
        require(!internal_f32_token_layout({ 128, 48, 1, 8 }, 2, { 128, 48 }, 8, layout));
        require(!internal_f32_token_layout({ 128, 16, 8, 2 }, 2, { 128, 16 }, 8, layout));
        require(!internal_f32_token_layout({ 1, 48, 7, 1 }, 2, { 1, 48 }, 8, layout));

        // Pinned full-attention layer-3 layouts from the reviewed Qwen3.8
        // metadata: D=256, Hq=24, Hkv=4. These protect against accidentally
        // reusing the recurrent D=128/H16/H48 contract.
        require(internal_f32_token_layout({ 12288, 8, 1, 1 }, 1, { 12288, 1 }, 8, layout));
        require(layout.row_bytes == 49152);
        require(internal_f32_token_layout({ 256, 24, 8, 1 }, 2, { 256, 24 }, 8, layout));
        require(layout.row_bytes == 24576);
        require(internal_f32_token_layout({ 1024, 8, 1, 1 }, 1, { 1024, 1 }, 8, layout));
        require(internal_f32_token_layout({ 256, 4, 8, 1 }, 2, { 256, 4 }, 8, layout));
        require(layout.row_bytes == 4096);
        require(internal_f32_token_layout({ 6144, 8, 1, 1 }, 1, { 6144, 1 }, 8, layout));

        require(!internal_f32_token_layout({ 10240, 8, 1, 1 }, 1, { 12288, 1 }, 8, layout));
        require(!internal_f32_token_layout({ 128, 40, 8, 1 }, 2, { 256, 24 }, 8, layout));
        require(!internal_f32_token_layout({ 256, 8, 4, 1 }, 2, { 256, 4 }, 8, layout));
    }

    {
        const auto cases = make_rollback_cases(184, 8);
        require(cases.size() == 7);
        require(cases.front().rollback == 1);
        require(cases.front().committed_token_index == 189);
        require(cases.front().continuation_token_index == 190);
        require(cases.front().continuation_prediction_index == 191);
        require(cases.back().rollback == 7);
        require(cases.back().committed_token_index == 183);
        require(cases.back().continuation_token_index == 184);
        require(cases.back().continuation_prediction_index == 185);
    }

    return 0;
}
