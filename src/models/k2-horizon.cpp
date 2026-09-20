#include "models.h"

void llama_model_k2_horizon::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_GROUPNORM_GROUPS, hparams.n_norm_groups, false);

    hparams.f_norm_group_eps = hparams.f_norm_rms_eps;
    if (hparams.n_norm_groups == 0) {
        hparams.n_norm_groups = 1;
    }

    // This port targets the dense IFM K2-Horizon-7B geometry only.
    if (hparams.n_layer() != 36 ||
            hparams.n_embd != 4096 ||
            hparams.n_head() != 32 ||
            hparams.n_head_kv() != 8 ||
            hparams.n_embd_head_k() != 128 ||
            hparams.n_embd_head_v() != 128 ||
            hparams.n_norm_groups != 4) {
        throw std::runtime_error("K2-Horizon port requires dense 7B geometry: 36 layers, 4096 hidden, 32/8 heads, D128, 4 norm groups");
    }

    if (hparams.n_expert != 0) {
        throw std::runtime_error("K2-Horizon-7B port does not support MoE tensors");
    }

    type = LLM_TYPE_7B;
}

void llama_model_k2_horizon::load_arch_tensors(llama_model_loader & ml) {
    GGML_UNUSED(ml);
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(
        tn(LLM_TENSOR_TOKEN_EMBD, "weight"),
        {n_embd, n_vocab},
        0
    );

    output_norm = create_tensor(
        tn(LLM_TENSOR_OUTPUT_NORM, "weight"),
        {n_embd},
        0
    );

    output = create_tensor(
        tn(LLM_TENSOR_OUTPUT, "weight"),
        {n_embd, n_vocab},
        TENSOR_NOT_REQUIRED
    );
    if (output == nullptr) {
        output = create_tensor(
            tn(LLM_TENSOR_TOKEN_EMBD, "weight"),
            {n_embd, n_vocab},
            TENSOR_DUPLICATED
        );
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(
            tn(LLM_TENSOR_ATTN_NORM, "weight", i),
            {n_embd},
            0
        );

        layer.wq = create_tensor(
            tn(LLM_TENSOR_ATTN_Q, "weight", i),
            {n_embd, n_embd_head_k * n_head},
            0
        );
        layer.attn_q_norm = create_tensor(
            tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i),
            {n_embd_head_k * n_head},
            TENSOR_NOT_REQUIRED
        );

        layer.wk = create_tensor(
            tn(LLM_TENSOR_ATTN_K, "weight", i),
            {n_embd, n_embd_k_gqa},
            0
        );
        layer.attn_k_norm = create_tensor(
            tn(LLM_TENSOR_ATTN_K_NORM, "weight", i),
            {n_embd_k_gqa},
            TENSOR_NOT_REQUIRED
        );

        layer.wv = create_tensor(
            tn(LLM_TENSOR_ATTN_V, "weight", i),
            {n_embd, n_embd_v_gqa},
            0
        );

        layer.wo = create_tensor(
            tn(LLM_TENSOR_ATTN_OUT, "weight", i),
            {n_embd_head_v * n_head, n_embd},
            0
        );

        layer.ffn_norm = create_tensor(
            tn(LLM_TENSOR_FFN_NORM, "weight", i),
            {n_embd},
            0
        );
        layer.ffn_up = create_tensor(
            tn(LLM_TENSOR_FFN_UP, "weight", i),
            {n_embd, n_ff},
            0
        );
        layer.ffn_gate = create_tensor(
            tn(LLM_TENSOR_FFN_GATE, "weight", i),
            {n_embd, n_ff},
            0
        );
        layer.ffn_down = create_tensor(
            tn(LLM_TENSOR_FFN_DOWN, "weight", i),
            {n_ff, n_embd},
            0
        );
    }
}

static ggml_tensor * k2_horizon_group_rms_norm(
    ggml_context * ctx,
    ggml_tensor * cur,
    ggml_tensor * weight,
    int64_t n_groups,
    float eps
) {
    GGML_ASSERT(n_groups > 0);
    GGML_ASSERT(cur->ne[0] % n_groups == 0);

    const int64_t n_embd = cur->ne[0];
    const int64_t n_tokens = cur->ne[1];
    cur = ggml_reshape_3d(
        ctx,
        cur,
        n_embd / n_groups,
        n_groups,
        n_tokens
    );
    cur = ggml_rms_norm(ctx, cur, eps);
    cur = ggml_reshape_2d(ctx, cur, n_embd, n_tokens);

    return weight != nullptr ? ggml_mul(ctx, cur, weight) : cur;
}

llama_model_k2_horizon::graph::graph(
    const llama_model & model,
    const llm_graph_params & params
) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * cur;
    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_pos = build_inp_pos();
    auto * inp_attn = build_attn_inp_kv();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = inpL;
        ggml_tensor * inpSA = inpL;

        cur = k2_horizon_group_rms_norm(
            ctx0,
            inpL,
            model.layers[il].attn_norm,
            hparams.n_norm_groups,
            hparams.f_norm_group_eps
        );
        cb(cur, "attn_norm", il);

        ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur, model.layers[il].wq_s);
        if (model.layers[il].attn_q_norm != nullptr) {
            Qcur = k2_horizon_group_rms_norm(
                ctx0,
                Qcur,
                model.layers[il].attn_q_norm,
                n_head,
                hparams.f_norm_group_eps
            );
        }

        ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur, model.layers[il].wk_s);
        if (model.layers[il].attn_k_norm != nullptr) {
            Kcur = k2_horizon_group_rms_norm(
                ctx0,
                Kcur,
                model.layers[il].attn_k_norm,
                n_head_kv,
                hparams.f_norm_group_eps
            );
        }

        ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur, model.layers[il].wv_s);
        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

        Qcur = ggml_rope_ext(
            ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
            freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
        );
        Kcur = ggml_rope_ext(
            ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
            freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
        );
        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);

        cur = build_attn(
            inp_attn,
            model.layers[il].wo,
            model.layers[il].wo_b,
            model.layers[il].wo_s,
            Qcur,
            Kcur,
            Vcur,
            nullptr,
            nullptr,
            nullptr,
            1.0f / sqrtf(static_cast<float>(n_embd_head)),
            il
        );

        if (il == n_layer - 1 && inp_out_ids != nullptr) {
            cur = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = k2_horizon_group_rms_norm(
            ctx0,
            ffn_inp,
            model.layers[il].ffn_norm,
            hparams.n_norm_groups,
            hparams.f_norm_group_eps
        );
        cb(cur, "ffn_norm", il);

        cur = build_ffn(
            cur,
            model.layers[il].ffn_up,
            nullptr,
            nullptr,
            model.layers[il].ffn_gate,
            nullptr,
            nullptr,
            model.layers[il].ffn_down,
            nullptr,
            nullptr,
            nullptr,
            LLM_FFN_SILU,
            LLM_FFN_PAR,
            il
        );
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);
        inpL = cur;
    }

    cur = k2_horizon_group_rms_norm(
        ctx0,
        inpL,
        model.output_norm,
        hparams.n_norm_groups,
        hparams.f_norm_group_eps
    );
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

std::unique_ptr<llm_graph_context> llama_model_k2_horizon::build_arch_graph(
    const llm_graph_params & params
) const {
    return std::make_unique<graph>(*this, params);
}
