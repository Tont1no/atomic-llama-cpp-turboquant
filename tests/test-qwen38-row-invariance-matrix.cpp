#include "qwen38-row-invariance-matrix.h"
#include "../src/llama-model-opt.h"
#include "../ggml/src/ggml-impl.h"

#include <cstdio>
#include <cstdlib>
#include <set>
#include <string>

using namespace qwen38_row_invariance;

static void require(bool value, const char * message) {
    if (!value) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}

int main() {
    ggml_tensor fattn_params{};
    ggml_set_op_params_f32(&fattn_params, 0, 0.0625f);
    ggml_set_op_params_f32(&fattn_params, 1, 0.0f);
    ggml_set_op_params_f32(&fattn_params, 2, 0.0f);
    ggml_set_op_params_i32(&fattn_params, 3, GGML_PREC_F32);
    ggml_set_op_params_i32(&fattn_params, 4, GGML_HINT_FLASH_ATTN_QWEN35_D256_Q8_GQA6_VEC);
    require(ggml_get_op_params_f32(&fattn_params, 0) == 0.0625f &&
            ggml_get_op_params_f32(&fattn_params, 1) == 0.0f &&
            ggml_get_op_params_f32(&fattn_params, 2) == 0.0f &&
            ggml_get_op_params_i32(&fattn_params, 3) == GGML_PREC_F32 &&
            ggml_get_op_params_i32(&fattn_params, 4) == GGML_HINT_FLASH_ATTN_QWEN35_D256_Q8_GQA6_VEC,
            "flash-attention scale/max-bias/softcap/precision/hint slots must not overlap");
    ggml_set_op_params_f32(&fattn_params, 1, 0.25f);
    require(ggml_get_op_params_f32(&fattn_params, 1) == 0.25f &&
            ggml_get_op_params_i32(&fattn_params, 4) == GGML_HINT_FLASH_ATTN_QWEN35_D256_Q8_GQA6_VEC,
            "writing max-bias must not overwrite the flash-attention hint slot");
    ggml_set_op_params_i32(&fattn_params, 4, GGML_HINT_NONE);
    require(ggml_get_op_params_f32(&fattn_params, 0) == 0.0625f &&
            ggml_get_op_params_f32(&fattn_params, 1) == 0.25f &&
            ggml_get_op_params_f32(&fattn_params, 2) == 0.0f,
            "writing the flash-attention hint must not overwrite float parameters");

    const auto quick = select_matrix(false);
    const auto full  = select_matrix(true);
    const auto fp8_only = select_operation(operation::fp8);
    const auto gdn_only = select_operation(operation::gdn);
    require(quick.size() == 6, "quick matrix must contain six cases");
    require(full.size() == 10, "full matrix must contain ten cases");
    require(fp8_only.size() == 5, "FP8-only matrix must contain five production shapes");
    require(gdn_only.size() == 2, "GDN-only matrix must contain both semantic cases");

    std::set<std::string> ids;
    size_t fp8 = 0, raw = 0, ffn = 0, gdn = 0;
    for (const auto & item : full) {
        require(ids.emplace(item.id).second, "case IDs must be unique");
        require(item.k > 0 && item.n > 0, "case dimensions must be positive");
        fp8 += item.op == operation::fp8;
        raw  += item.op == operation::nvfp4_raw;
        ffn  += item.op == operation::nvfp4_ffn;
        gdn  += item.op == operation::gdn;
    }
    require(fp8 == 5 && raw == 2 && ffn == 1 && gdn == 2,
            "full matrix must be 5 FP8 + 2 raw NVFP4 + 1 FFN + 2 GDN");

    fp8 = raw = ffn = gdn = 0;
    for (const auto & item : quick) {
        fp8 += item.op == operation::fp8;
        raw  += item.op == operation::nvfp4_raw;
        ffn  += item.op == operation::nvfp4_ffn;
        gdn  += item.op == operation::gdn;
    }
    require(fp8 == 1 && raw == 2 && ffn == 1 && gdn == 2,
            "quick matrix must be 1 FP8 + 2 raw NVFP4 + 1 FFN + 2 GDN");
    require(quick[0].k == 5120 && quick[0].n == 1024,
            "quick FP8 case must use the pinned production shape");
    for (const auto & item : fp8_only) {
        require(item.op == operation::fp8, "FP8-only matrix leaked another operator");
    }
    for (const auto & item : gdn_only) {
        require(item.op == operation::gdn, "GDN-only matrix leaked another operator");
    }
    require(HEAD_CANDIDATE.op == operation::nvfp4_head &&
            HEAD_CANDIDATE.k == 5120 && HEAD_CANDIDATE.n == 128,
            "Qwen3.5 LM-head candidate shape must remain pinned");
    require(HEAD_MIN_M == 2 && HEAD_MAX_M == 16,
            "Qwen3.5 LM-head accepted-width contract must remain M2 through M16");
    require(PROJECTION_CANDIDATE.size() == 2 &&
            PROJECTION_CANDIDATE[0].op == operation::nvfp4_ffn_row_invariant &&
            PROJECTION_CANDIDATE[0].k == 5120 && PROJECTION_CANDIDATE[0].n == 17408 &&
            PROJECTION_CANDIDATE[1].op == operation::nvfp4_down_row_invariant &&
            PROJECTION_CANDIDATE[1].k == 17408 && PROJECTION_CANDIDATE[1].n == 5120,
            "Qwen3.5 dense FFN candidate shapes must remain gate/up 5120x17408 and down 17408x5120");
    require(RMS_NORM_CASE.op == operation::rms_norm && RMS_NORM_CASE.k == 5120 && RMS_NORM_CASE.n == 1,
            "Qwen3.5 RMSNorm diagnostic must remain E5120");
    require(SSM_CONV_CASE.op == operation::ssm_conv && SSM_CONV_CASE.k == 4 && SSM_CONV_CASE.n == 10240,
            "Qwen3.5 SSM_CONV diagnostic must remain d_conv4/channels10240");
    require(L2_NORM_CASES.size() == 2 && L2_NORM_CASES[0].op == operation::l2_norm &&
            L2_NORM_CASES[1].op == operation::l2_norm && L2_NORM_CASES[0].k == 128 &&
            L2_NORM_CASES[0].n == 16 && L2_NORM_CASES[1].k == 128 && L2_NORM_CASES[1].n == 16,
            "Qwen3.5 q/k L2 diagnostics must remain S128/H16");
    require(GATED_NORM_CASE.op == operation::gated_norm &&
            GATED_NORM_CASE.k == 128 && GATED_NORM_CASE.n == 48,
            "Qwen3.5 post-GDN gated norm diagnostic must remain S128/H48");
    require(BF16_PROJECTION_CASES.size() == 2 &&
            BF16_PROJECTION_CASES[0].op == operation::bf16_projection &&
            BF16_PROJECTION_CASES[1].op == operation::bf16_projection &&
            BF16_PROJECTION_CASES[0].k == 5120 && BF16_PROJECTION_CASES[0].n == 48 &&
            BF16_PROJECTION_CASES[1].k == 5120 && BF16_PROJECTION_CASES[1].n == 48,
            "Qwen3.5 beta/alpha BF16 projections must remain K5120/N48");
    require(BF16_PROJECTION_CANDIDATE_CASES.size() == 2 &&
            BF16_PROJECTION_CANDIDATE_CASES[0].op == operation::bf16_projection_candidate &&
            BF16_PROJECTION_CANDIDATE_CASES[1].op == operation::bf16_projection_candidate &&
            BF16_PROJECTION_CANDIDATE_CASES[0].k == 5120 && BF16_PROJECTION_CANDIDATE_CASES[0].n == 48 &&
            BF16_PROJECTION_CANDIDATE_CASES[1].k == 5120 && BF16_PROJECTION_CANDIDATE_CASES[1].n == 48,
            "Qwen3.5 BF16 row-invariant beta/alpha candidate shapes must remain K5120/N48");
    require(FATTN_VEC_COLS2_PB1_CASE.op == operation::fattn_vec_cols2_pb1 &&
            FATTN_VEC_COLS2_PB1_CASE.k == 256 && FATTN_VEC_COLS2_PB1_CASE.n == 1024,
            "Qwen3.5 FA PB1 locator must remain D256/n_kv1024");
    require(llama_model_opt_mul_mat_hint_allowed(
                    GGML_HINT_MUL_MAT_ROW_INVARIANT_BF16_BETA, false, GGML_OP_MUL_MAT) &&
            llama_model_opt_mul_mat_hint_allowed(
                    GGML_HINT_MUL_MAT_ROW_INVARIANT_BF16_ALPHA, false, GGML_OP_MUL_MAT),
            "bare Qwen3.5 BF16 alpha/beta matmuls must accept their reviewed hints");
    require(!llama_model_opt_mul_mat_hint_allowed(
                    GGML_HINT_MUL_MAT_ROW_INVARIANT_BF16_BETA, true, GGML_OP_MUL_MAT) &&
            !llama_model_opt_mul_mat_hint_allowed(
                    GGML_HINT_MUL_MAT_ROW_INVARIANT_BF16_ALPHA, true, GGML_OP_MUL_MAT),
            "Qwen3.5 BF16 alpha/beta LoRA residuals must suppress model-opt hints");
    require(!llama_model_opt_mul_mat_hint_allowed(
                    GGML_HINT_MUL_MAT_ROW_INVARIANT_BF16_ALPHA, false, GGML_OP_ADD),
            "a transformed non-MUL_MAT graph must suppress model-opt hints");
    auto fattn_contract = [](ggml_op_hint hint, ggml_op op, ggml_type q_type,
            ggml_type k_type, ggml_type v_type, ggml_type mask_type, ggml_type dst_type,
            int64_t q_d, int64_t q_cols, int64_t q_heads, int64_t k_rows,
            int64_t k_heads, bool layout, bool causal, bool no_sinks, bool params,
            bool single_stream = true) {
        return llama_model_opt_qwen35_fattn_vec_hint_allowed(
                hint, op, q_type, k_type, v_type, mask_type, dst_type,
                q_d, q_cols, q_heads, 256, k_rows, k_heads,
                256, k_rows, k_heads, k_rows, q_cols, 256, q_heads, q_cols,
                layout, causal, no_sinks, params, single_stream);
    };
    require(fattn_contract(GGML_HINT_FLASH_ATTN_QWEN35_D256_Q8_GQA6_VEC,
                    GGML_OP_FLASH_ATTN_EXT, GGML_TYPE_F32, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0,
                    GGML_TYPE_F16, GGML_TYPE_F32, 256, 3, 24, 1024, 4, true, true, true, true) &&
            fattn_contract(GGML_HINT_FLASH_ATTN_QWEN35_D256_Q8_GQA6_VEC,
                    GGML_OP_FLASH_ATTN_EXT, GGML_TYPE_F32, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0,
                    GGML_TYPE_F16, GGML_TYPE_F32, 256, 8, 24, 1024, 4, true, true, true, true),
            "Qwen3.5 D256/Q8/GQA6 flash-attention hint must accept exactly query columns 3..8");
    for (int64_t q_cols : { 1, 2, 9 }) {
        require(!fattn_contract(GGML_HINT_FLASH_ATTN_QWEN35_D256_Q8_GQA6_VEC,
                        GGML_OP_FLASH_ATTN_EXT, GGML_TYPE_F32, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0,
                        GGML_TYPE_F16, GGML_TYPE_F32, 256, q_cols, 24, 1024, 4, true, true, true, true),
                "Qwen3.5 flash-attention candidate must reject M1, M2, and M9");
    }
    require(!fattn_contract(GGML_HINT_NONE, GGML_OP_FLASH_ATTN_EXT,
                    GGML_TYPE_F32, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, GGML_TYPE_F16, GGML_TYPE_F32,
                    256, 4, 24, 1024, 4, true, true, true, true) &&
            !fattn_contract(GGML_HINT_FLASH_ATTN_QWEN35_D256_Q8_GQA6_VEC, GGML_OP_FLASH_ATTN_EXT,
                    GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_F16, GGML_TYPE_F32,
                    256, 4, 24, 1024, 4, true, true, true, true) &&
            !fattn_contract(GGML_HINT_FLASH_ATTN_QWEN35_D256_Q8_GQA6_VEC, GGML_OP_FLASH_ATTN_EXT,
                    GGML_TYPE_F32, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, GGML_TYPE_F16, GGML_TYPE_F32,
                    256, 4, 24, 1024, 4, false, true, true, true) &&
            !fattn_contract(GGML_HINT_FLASH_ATTN_QWEN35_D256_Q8_GQA6_VEC, GGML_OP_FLASH_ATTN_EXT,
                    GGML_TYPE_F32, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, GGML_TYPE_F16, GGML_TYPE_F32,
                    256, 4, 24, 1024, 4, true, false, true, true) &&
            !fattn_contract(GGML_HINT_FLASH_ATTN_QWEN35_D256_Q8_GQA6_VEC, GGML_OP_FLASH_ATTN_EXT,
                    GGML_TYPE_F32, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, GGML_TYPE_F16, GGML_TYPE_F32,
                    128, 4, 24, 1024, 4, true, true, true, true),
            "Qwen3.5 flash-attention candidate must reject missing hints, wrong types, layouts, non-causal use, and dimensions");
    require(!fattn_contract(GGML_HINT_FLASH_ATTN_QWEN35_D256_Q8_GQA6_VEC, GGML_OP_FLASH_ATTN_EXT,
                    GGML_TYPE_F32, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, GGML_TYPE_F16, GGML_TYPE_F32,
                    256, 4, 24, 1024, 4, true, true, true, true, false),
            "Qwen3.5 flash-attention candidate must reject multi-stream Q/K/V/mask/output graphs");
    const size_t q8_row_d = ggml_row_size(GGML_TYPE_Q8_0, 256);
    const size_t q8_row_all_heads = ggml_row_size(GGML_TYPE_Q8_0, 256 * 4);
    require(llama_model_opt_qwen35_fattn_kv_cache_layout_allowed(
                    GGML_TYPE_Q8_0, 256, 1024, 4, 1,
                    ggml_type_size(GGML_TYPE_Q8_0), q8_row_all_heads, q8_row_d,
                    q8_row_all_heads * 2048),
            "Qwen3.5 flash-attention candidate must accept the production token-major Q8 KV cache view");
    require(!llama_model_opt_qwen35_fattn_kv_cache_layout_allowed(
                    GGML_TYPE_Q8_0, 256, 1024, 4, 1,
                    ggml_type_size(GGML_TYPE_Q8_0), q8_row_d, q8_row_d * 1024,
                    q8_row_all_heads * 2048) &&
            !llama_model_opt_qwen35_fattn_kv_cache_layout_allowed(
                    GGML_TYPE_Q8_0, 256, 1024, 4, 1,
                    ggml_type_size(GGML_TYPE_Q8_0), q8_row_all_heads, q8_row_d,
                    q8_row_all_heads * 512) &&
            !llama_model_opt_qwen35_fattn_kv_cache_layout_allowed(
                    GGML_TYPE_Q8_0, 256, 1024, 4, 2,
                    ggml_type_size(GGML_TYPE_Q8_0), q8_row_all_heads, q8_row_d,
                    q8_row_all_heads * 2048),
            "Qwen3.5 flash-attention candidate must reject logical-contiguous and overlapping KV views");
    require(head_route_observation_valid({ 1, 3, 4, 0 }),
            "LM-head route observation must accept M1 plus a tagged-width delta");
    require(!head_route_observation_valid({ 0, 3, 4, 0 }),
            "LM-head route observation must reject a missing unchanged-M1 route");
    require(!head_route_observation_valid({ 1, 3, 3, 0 }),
            "LM-head route observation must reject a missing tagged-width dispatch");
    require(!head_route_observation_valid({ 1, 3, 4, 1 }),
            "LM-head route observation must reject a generic batched route");

    std::puts("Qwen3.8 row-invariance matrix contract PASS (quick=6 full=10 fp8=5 gdn=2 recurrent_ops=5)");
    return 0;
}
