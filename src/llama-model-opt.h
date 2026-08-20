#pragma once

#include "ggml.h"

// A semantic graph hint is valid only while the MUL_MAT remains the complete
// operation. A LoRA residual changes that operation and must keep the graph on
// the established backend routes.
static inline bool llama_model_opt_mul_mat_hint_allowed(
        enum ggml_op_hint hint, bool has_lora, enum ggml_op op) {
    return hint != GGML_HINT_NONE && !has_lora && op == GGML_OP_MUL_MAT;
}

// The Qwen3.5 KV cache is physically token-major across all KV heads. After
// build_attn_mha permutes it to logical [D, n_kv, H, 1], the token stride is
// one complete D*H row and the head stride is one D row. Dimension 3 is a
// single stream here; require a non-overlapping cache-capacity stride without
// pretending the logical view itself is contiguous.
static inline bool llama_model_opt_qwen35_fattn_kv_cache_layout_allowed(
        enum ggml_type type,
        int64_t d,
        int64_t rows,
        int64_t heads,
        int64_t streams,
        size_t nb0,
        size_t nb1,
        size_t nb2,
        size_t nb3) {
    if (type != GGML_TYPE_Q8_0 || d != 256 || rows <= 0 || heads != 4 || streams != 1) {
        return false;
    }
    const size_t row_d = ggml_row_size(type, d);
    const size_t row_all_heads = ggml_row_size(type, d * heads);
    return nb0 == ggml_type_size(type) && nb1 == row_all_heads && nb2 == row_d &&
            nb3 >= nb1 * (size_t) rows && nb3 % nb1 == 0;
}

// Model-side half of the Qwen3.5 full-attention diagnostic contract. The
// graph builder supplies the actual cache/output types and reviewed layout;
// CUDA independently repeats this predicate and adds the device guard.
static inline bool llama_model_opt_qwen35_fattn_vec_hint_allowed(
        enum ggml_op_hint hint,
        enum ggml_op op,
        enum ggml_type q_type,
        enum ggml_type k_type,
        enum ggml_type v_type,
        enum ggml_type mask_type,
        enum ggml_type dst_type,
        int64_t q_d,
        int64_t q_cols,
        int64_t q_heads,
        int64_t k_d,
        int64_t k_rows,
        int64_t k_heads,
        int64_t v_d,
        int64_t v_rows,
        int64_t v_heads,
        int64_t mask_rows,
        int64_t mask_cols,
        int64_t dst_d,
        int64_t dst_heads,
        int64_t dst_cols,
        bool exact_layout,
        bool causal_mask,
        bool no_sinks,
        bool exact_params,
        bool single_stream) {
    return hint == GGML_HINT_FLASH_ATTN_QWEN35_D256_Q8_GQA6_VEC &&
            op == GGML_OP_FLASH_ATTN_EXT &&
            q_type == GGML_TYPE_F32 && k_type == GGML_TYPE_Q8_0 && v_type == GGML_TYPE_Q8_0 &&
            mask_type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F32 &&
            q_d == 256 && q_cols >= 3 && q_cols <= 8 && q_heads == 24 &&
            k_d == 256 && k_rows > 0 && k_rows % 256 == 0 && k_heads == 4 &&
            v_d == 256 && v_rows == k_rows && v_heads == 4 &&
            mask_rows == k_rows && mask_cols == q_cols &&
            dst_d == 256 && dst_heads == 24 && dst_cols == q_cols &&
            exact_layout && causal_mask && no_sinks && exact_params && single_stream;
}
