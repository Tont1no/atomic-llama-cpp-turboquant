#include "gated_delta_net.cuh"
#include "ggml-cuda/common.cuh"

template <int S_v, bool KDA, bool keep_rs_t>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K,
                                     int           base1,
                                     const float * rp,
                                     const int32_t * rp_idx,
                                     int           R,
                                     int64_t       rp_w) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    // ReplaySSM: replay the accepted tokens kept from the last ubatch first
    // (state update only; their outputs were produced back then)
    if constexpr (!KDA) {
        if (rp != nullptr) {
            const int32_t src = rp_idx[3*sequence + 0];
            const int32_t cnt = rp_idx[3*sequence + 1];
            const int64_t kvw = (int64_t) S_v * H;
            for (int32_t rr = 0; rr < cnt; ++rr) {
                const float * row = rp + ((int64_t) src * R + rr) * rp_w;
                const float * k_t = row + iq1 * S_v;
                const float * v_t = row + kvw + h_idx * S_v;
                const float g_val = expf(row[2*kvw + h_idx]);
                const float beta_val = row[2*kvw + H + h_idx];
                float k_reg[rows_per_lane];
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    k_reg[r] = k_t[r * warp_size + lane];
                }
                float kv_shard = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    kv_shard += s_shard[r] * k_reg[r];
                }
                const float kv_col = warp_reduce_sum<warp_size>(kv_shard);
                const float delta_col = (v_t[col] - g_val * kv_col) * beta_val;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    s_shard[r] = g_val * s_shard[r] + k_reg[r] * delta_col;
                }
            }
        }
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            const float g_val = expf(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            // ReplaySSM (base1 > 0): slot 0 after the last token, slot 1 after token base1 - 1.
            const int target_slot = (int) n_tokens - 1 - t;
            const bool write0 = base1 > 0 ? t == n_tokens - 1 : (target_slot >= 0 && target_slot < K);
            const bool write1 = base1 > 0 && t == base1 - 1;
            if (write0 || write1) {
                float * curr_state = state + (base1 > 0 ? 0 : target_slot) * state_slot_stride;
                float * base_state = state + state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    if (write0) { curr_state[col * S_v + i] = s_shard[r]; }
                    if (write1) { base_state[col * S_v + i] = s_shard[r]; }
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * warp_size + lane;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, int base1,
        const float * rp, const int32_t * rp_idx, int R, int64_t rp_w, cudaStream_t stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, base1, rp, rp_idx, R, rp_w);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, base1, rp, rp_idx, R, rp_w);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, base1, rp, rp_idx, R, rp_w);
            break;
        }
        case 128: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, base1, rp, rp_idx, R, rp_w);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const int base1 = ggml_get_op_params_i32(dst, 1);
    const bool keep_rs = K > 1;
    // ReplaySSM replay rows (see ggml_gated_delta_net_replay)
    const ggml_tensor * rp_t = dst->src[6];
    const float *   rp_d     = rp_t ? (const float *) rp_t->data : nullptr;
    const int32_t * rp_idx_d = rp_t ? (const int32_t *) dst->src[7]->data : nullptr;
    const int       R        = rp_t ? ggml_get_op_params_i32(dst, 2) : 0;
    const int64_t   rp_w     = rp_t ? rp_t->ne[0] : 0;
    GGML_ASSERT(rp_t == nullptr || !kda);

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, base1, rp_d, rp_idx_d, R, rp_w, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, base1, rp_d, rp_idx_d, R, rp_w, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, base1, rp_d, rp_idx_d, R, rp_w, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, base1, rp_d, rp_idx_d, R, rp_w, stream);
        }
    }
}

// ReplaySSM: store the ubatch's last `tail` tokens (k, v, g, beta) as replay
// rows of each sequence's destination cell. Runs after the main kernel, which
// may still read the old rows of the same cells.
static __global__ void gated_delta_net_store_tail(
        const float * k, const float * v, const float * g, const float * b,
        float * rp, const int32_t * rp_idx, int R, int64_t rp_w,
        int64_t S_v, int64_t H, int64_t nk, int64_t n_tokens, int tail, int64_t rk3,
        int64_t sk1, int64_t sk2, int64_t sk3, int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3) {
    const int64_t s  = blockIdx.x / tail;
    const int64_t ti = blockIdx.x % tail;
    const int64_t t  = n_tokens - tail + ti;
    float * row = rp + ((int64_t) rp_idx[3*s + 2] * R + ti) * rp_w;
    const int64_t kvw = S_v * H;
    for (int64_t e = threadIdx.x; e < nk * S_v; e += blockDim.x) {
        const int64_t h = e / S_v, i = e % S_v;
        row[e] = k[(s / rk3) * sk3 + t * sk2 + h * sk1 + i];
    }
    for (int64_t e = threadIdx.x; e < kvw; e += blockDim.x) {
        const int64_t h = e / S_v, i = e % S_v;
        row[kvw + e] = v[s * sv3 + t * sv2 + h * sv1 + i];
    }
    for (int64_t h = threadIdx.x; h < H; h += blockDim.x) {
        row[2*kvw + h]     = g[s * sb3 + t * sb2 + h * sb1];
        row[2*kvw + H + h] = b[s * sb3 + t * sb2 + h * sb1];
    }
}

static void ggml_cuda_gated_delta_net_store_tail(ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    const ggml_tensor * rp_t = dst->src[6];
    if (rp_t == nullptr) {
        return;
    }
    const int tail = ggml_get_op_params_i32(dst, 3);
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * g = dst->src[3];
    const ggml_tensor * b = dst->src[4];
    const int64_t n_seqs = v->ne[3];
    if (tail <= 0 || n_seqs <= 0) {
        return;
    }
    GGML_ASSERT(g->nb[1] == b->nb[1] && g->nb[2] == b->nb[2] && g->nb[3] == b->nb[3]);
    const int64_t fs = sizeof(float);
    gated_delta_net_store_tail<<<(unsigned) (n_seqs * tail), 256, 0, ctx.stream()>>>(
        (const float *) k->data, (const float *) v->data, (const float *) g->data, (const float *) b->data,
        (float *) rp_t->data, (const int32_t *) dst->src[7]->data, ggml_get_op_params_i32(dst, 2), rp_t->ne[0],
        v->ne[0], v->ne[1], k->ne[1], v->ne[2], tail, n_seqs / k->ne[3],
        k->nb[1] / fs, k->nb[2] / fs, k->nb[3] / fs, v->nb[1] / fs, v->nb[2] / fs, v->nb[3] / fs,
        g->nb[1] / fs, g->nb[2] / fs, g->nb[3] / fs);
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
    ggml_cuda_gated_delta_net_store_tail(ctx, dst);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
    ggml_cuda_gated_delta_net_store_tail(ctx, dst);
}
