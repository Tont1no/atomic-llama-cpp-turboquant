#include "common.cuh"
#include "fattn-hybrid.cuh"
#include "turbo4-cuda.cuh"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <type_traits>

// One TQ4 element of a cold row. The centroid comes from the warp-resident
// table (see ggml_cuda_turbo4_centroid_shfl): every lane calls this with the
// same key row and a different element, so the shuffle is warp-uniform.
static __device__ __forceinline__ float ggml_cuda_hybrid_turbo4_value(
        const char * row, const int index, const float lane_centroid) {
    const int block_index = index / GGML_TURBO4_QK;
    const int offset = index % GGML_TURBO4_QK;
    const block_turbo4_0 * block = (const block_turbo4_0 *) row + block_index;
    const uint8_t packed = block->qs[offset >> 1];
    const uint8_t code = (packed >> ((offset & 1) * 4)) & 0x0f;
    return ggml_cuda_turbo4_centroid_shfl(code, lane_centroid) * __half2float(block->norm);
}

// Both packed formats share the same rotation, causal sources and reductions.
template<ggml_type type>
static __device__ __forceinline__ float ggml_cuda_hybrid_turbo_value(
        const char * row, const int index, const float lane_centroid) {
    if constexpr (type == GGML_TYPE_TURBO3_5) {
        return ggml_cuda_turbo35_dequant_value_lane(row, index, lane_centroid);
    } else {
        return ggml_cuda_hybrid_turbo4_value(row, index, lane_centroid);
    }
}

static __device__ __forceinline__ float ggml_cuda_hybrid_warp_sum(float value) {
    constexpr unsigned int warp_mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(warp_mask, value, offset);
    }
    return __shfl_sync(warp_mask, value, 0);
}

static __device__ __forceinline__ float ggml_cuda_hybrid_warp_max(float value) {
    constexpr unsigned int warp_mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(warp_mask, value, offset));
    }
    return __shfl_sync(warp_mask, value, 0);
}

static __device__ __forceinline__ bool ggml_cuda_hybrid_is_finite(float value) {
    return value == value && value != INFINITY && value != -INFINITY;
}

template<ggml_type type, bool split_k_path>
static __global__ void ggml_cuda_flash_attn_ext_hybrid_kernel(
        const float * q,
        const char * k_cold,
        const char * v_cold,
        const char * k_hot,
        const char * v_hot,
        const char * mask,
        const char * q_positions,
        const char * k_positions,
        const float * sinks,
        float * dst,
        const int64_t D,
        const int64_t NQ,
        const int64_t NQ_HEADS,
        const int64_t NSEQ,
        const int64_t NCOLD,
        const int64_t NHOT,
        const int64_t NKV_HEADS,
        const size_t q_nb1,
        const size_t q_nb2,
        const size_t q_nb3,
        const size_t dst_nb1,
        const size_t dst_nb2,
        const size_t dst_nb3,
        const size_t kc_nb1,
        const size_t kc_nb2,
        const size_t kc_nb3,
        const size_t vc_nb1,
        const size_t vc_nb2,
        const size_t vc_nb3,
        const size_t kh_nb1,
        const size_t kh_nb2,
        const size_t kh_nb3,
        const size_t vh_nb1,
        const size_t vh_nb2,
        const size_t vh_nb3,
        const size_t mask_nb0,
        const size_t mask_nb1,
        const size_t mask_nb2,
        const size_t mask_nb3,
        const int64_t mask_ne2,
        const int64_t mask_ne3,
        const size_t qpos_nb0,
        const size_t qpos_nb1,
        const size_t qpos_nb2,
        const int64_t qpos_ne1,
        const int64_t qpos_ne2,
        const size_t kpos_nb0,
        const size_t kpos_nb1,
        const size_t kpos_nb2,
        const int64_t kpos_ne2,
        const float scale,
        const float max_bias,
        const float logit_softcap,
        float * partial_o,
        float2 * partial_meta,
        const int split_k) {
    const int64_t blocks_per_row = split_k_path ? split_k : 1;
    const int64_t block = (int64_t) blockIdx.x;
    const int64_t row = block / blocks_per_row;
    const int64_t split = block % blocks_per_row;
    const int64_t rows = NQ*NQ_HEADS*NSEQ;
    if (row >= rows) {
        return;
    }
    const int lane = threadIdx.x & 31;
    constexpr unsigned int warp_mask = 0xffffffffu;
    const float lane_centroid = type == GGML_TYPE_TURBO3_5
        ? ggml_cuda_turbo35_centroid_lane() : ggml_cuda_turbo4_centroid_lane();

    const int64_t iq3 = row/(NQ_HEADS*NQ);
    const int64_t iq2 = (row - iq3*NQ_HEADS*NQ)/NQ;
    const int64_t iq1 = row - iq3*NQ_HEADS*NQ - iq2*NQ;
    const int64_t q_to_kv = NQ_HEADS/NKV_HEADS;
    const int64_t kv_head = iq2/q_to_kv;

    const uint32_t head = (uint32_t) iq2;
    const uint32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((float) NQ_HEADS));
    const float m0 = powf(2.0f, -max_bias/n_head_log2);
    const float m1 = powf(2.0f, -(max_bias/2.0f)/n_head_log2);
    const float slope = max_bias > 0.0f
        ? head < n_head_log2 ? powf(m0, head + 1) : powf(m1, 2*(head - n_head_log2) + 1)
        : 1.0f;

    const float * q_row = (const float *) ((const char *) q + iq1*q_nb1 + iq2*q_nb2 + iq3*q_nb3);
    // One warp owns one query row. Each lane carries D/32 output values;
    // this keeps the online softmax state shared by the warp while the dot
    // product and value accumulation are cooperative.
    float output[8] = {};
    float M = -INFINITY;
    float S = 0.0f;

    const half * mask_row = mask ? (const half *) (mask + iq1*mask_nb1 +
        (iq2 % mask_ne2)*mask_nb2 + (iq3 % mask_ne3)*mask_nb3) : nullptr;
    int32_t q_position = 0;
    if (q_positions) {
        const int64_t q_head_pos = qpos_ne1 == 1 ? 0 : iq2;
        const int64_t q_seq_pos = qpos_ne2 == 1 ? 0 : iq3;
        q_position = *(const int32_t *) (q_positions + iq1*qpos_nb0 +
            q_head_pos*qpos_nb1 + q_seq_pos*qpos_nb2);
    }

    const int64_t n_keys = NCOLD + NHOT;
    const int64_t key_begin = split_k_path ? (n_keys*split)/split_k : 0;
    const int64_t key_end   = split_k_path ? (n_keys*(split + 1))/split_k : n_keys;
    for (int64_t key_index = key_begin; key_index < key_end; ++key_index) {
        if (mask_row && __half2float(*(const half *) ((const char *) mask_row + key_index*mask_nb0)) == -INFINITY) {
            continue;
        }
        if (k_positions) {
            const int64_t key_seq_pos = kpos_ne2 == 1 ? 0 : iq3;
            const int32_t key_position = *(const int32_t *) (k_positions + key_index*kpos_nb0 +
                kv_head*kpos_nb1 + key_seq_pos*kpos_nb2);
            if (key_position < 0 || q_position < key_position) {
                continue;
            }
        }

        const bool hot = key_index >= NCOLD;
        const int64_t local_key = hot ? key_index - NCOLD : key_index;
        const char * k_row = hot
            ? k_hot + local_key*kh_nb1 + kv_head*kh_nb2 + iq3*kh_nb3
            : k_cold + local_key*kc_nb1 + kv_head*kc_nb2 + iq3*kc_nb3;
        const char * v_row = hot
            ? v_hot + local_key*vh_nb1 + kv_head*vh_nb2 + iq3*vh_nb3
            : v_cold + local_key*vc_nb1 + kv_head*vc_nb2 + iq3*vc_nb3;

        float dot = 0.0f;
        for (int64_t d = lane; d < D; d += 32) {
            const float key = hot
                ? __half2float(*(const half *) (k_row + d*sizeof(half)))
                : ggml_cuda_hybrid_turbo_value<type>(k_row, (int) d, lane_centroid);
            dot += q_row[d]*key;
        }
        dot = ggml_cuda_hybrid_warp_sum(dot);
        const float score_dot = dot;
        float score = score_dot;
        score *= scale;
        if (logit_softcap != 0.0f) {
            score = logit_softcap*tanhf(score);
        }
        if (mask_row) {
            score += slope*__half2float(*(const half *) ((const char *) mask_row + key_index*mask_nb0));
        }

        const float M_new = fmaxf(M, score);
        const float old_scale = S == 0.0f ? 0.0f : expf(M - M_new);
        const float weight = expf(score - M_new);
        for (int64_t d = lane; d < D; d += 32) {
            const float value = hot
                ? __half2float(*(const half *) (v_row + d*sizeof(half)))
                : ggml_cuda_hybrid_turbo_value<type>(v_row, (int) d, lane_centroid);
            output[d/32] = output[d/32]*old_scale + value*weight;
        }
        S = S*old_scale + weight;
        M = M_new;
    }

    if constexpr (!split_k_path) {
        if (sinks) {
            const float sink = sinks[iq2];
            const float M_new = fmaxf(M, sink);
            const float old_scale = S == 0.0f ? 0.0f : expf(M - M_new);
            const float weight = expf(sink - M_new);
            for (int64_t d = lane; d < D; d += 32) {
                output[d/32] *= old_scale;
            }
            S = S*old_scale + weight;
        }

        float * dst_row = (float *) ((char *) dst + iq2*dst_nb1 + iq1*dst_nb2 + iq3*dst_nb3);
        const float inv = S == 0.0f ? 0.0f : 1.0f/S;
        for (int64_t d = lane; d < D; d += 32) {
            dst_row[d] = output[d/32]*inv;
        }
    } else {
        float * partial_row = partial_o + (row*split_k + split)*D;
        for (int64_t d = lane; d < D; d += 32) {
            partial_row[d] = output[d/32];
        }
        if (lane == 0) {
            partial_meta[row*split_k + split] = make_float2(M, S);
        }
    }
}

template<int D>
static __global__ void ggml_cuda_flash_attn_ext_hybrid_merge_kernel(
        const float * partial_o,
        const float2 * partial_meta,
        const float * sinks,
        float * dst,
        const int64_t NQ,
        const int64_t NQ_HEADS,
        const int64_t NSEQ,
        const size_t dst_nb1,
        const size_t dst_nb2,
        const size_t dst_nb3,
        const int split_k) {
    const int64_t row = (int64_t) blockIdx.x;
    const int64_t rows = NQ*NQ_HEADS*NSEQ;
    if (row >= rows) {
        return;
    }

    const int lane = threadIdx.x & 31;
    const int64_t iq3 = row/(NQ_HEADS*NQ);
    const int64_t iq2 = (row - iq3*NQ_HEADS*NQ)/NQ;
    const int64_t iq1 = row - iq3*NQ_HEADS*NQ - iq2*NQ;
    const size_t meta_base = (size_t) row*split_k;
    const size_t value_base = meta_base*D;

    float local_max = -INFINITY;
    for (int i = lane; i < split_k; i += 32) {
        const float2 meta = partial_meta[meta_base + i];
        if (ggml_cuda_hybrid_is_finite(meta.x) && meta.y > 0.0f) {
            local_max = fmaxf(local_max, meta.x);
        }
    }
    if (sinks && lane == 0 && ggml_cuda_hybrid_is_finite(sinks[iq2])) {
        local_max = fmaxf(local_max, sinks[iq2]);
    }
    const float global_max = ggml_cuda_hybrid_warp_max(local_max);

    float local_denominator = 0.0f;
    if (ggml_cuda_hybrid_is_finite(global_max)) {
        for (int i = lane; i < split_k; i += 32) {
            const float2 meta = partial_meta[meta_base + i];
            if (ggml_cuda_hybrid_is_finite(meta.x) && meta.y > 0.0f) {
                local_denominator += expf(meta.x - global_max)*meta.y;
            }
        }
        if (sinks && lane == 0 && ggml_cuda_hybrid_is_finite(sinks[iq2])) {
            local_denominator += expf(sinks[iq2] - global_max);
        }
    }
    const float denominator = ggml_cuda_hybrid_warp_sum(local_denominator);

    float * dst_row = (float *) ((char *) dst + iq2*dst_nb1 + iq1*dst_nb2 + iq3*dst_nb3);
    for (int d = lane; d < D; d += 32) {
        float numerator = 0.0f;
        if (denominator > 0.0f && ggml_cuda_hybrid_is_finite(global_max)) {
            for (int i = 0; i < split_k; ++i) {
                const float2 meta = partial_meta[meta_base + i];
                if (ggml_cuda_hybrid_is_finite(meta.x) && meta.y > 0.0f) {
                    numerator += expf(meta.x - global_max)*partial_o[value_base + i*D + d];
                }
            }
        }
        dst_row[d] = denominator > 0.0f ? numerator/denominator : 0.0f;
    }
}

// One warp per (query row, key range). The kernel walks its keys serially
// with a dependent load chain per key, so latency hiding comes only from
// warps resident per SM: with 16 query heads and 32 splits a 5090 (170 SMs)
// held under two warps per SM and the 1024+96 key perf case took 128 us.
// Smaller key ranges and more splits put ten or more warps on every SM; the
// merge kernel reduces any split count. GGML_CUDA_HYBRID_KEYS_PER_SPLIT and
// GGML_CUDA_HYBRID_MAX_SPLIT override the defaults for A/B runs.
static int ggml_cuda_hybrid_split_setting(const char * name, const int fallback, const int lo, const int hi) {
    const char * text = std::getenv(name);
    if (!text || !*text) {
        return fallback;
    }
    char * end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    return end != text && *end == '\0' && parsed >= lo && parsed <= hi ? int(parsed) : fallback;
}

// n_rows query rows share the split budget: the partial output buffer is
// n_rows*split_k*D floats, capped at 64 MiB so a draft-verification or
// append ubatch with hundreds of rows does not exhaust the pool.
static int ggml_cuda_hybrid_split_count(const int64_t n_keys, const uint64_t n_rows, const int64_t D) {
    static const int target_keys_per_block = ggml_cuda_hybrid_split_setting("GGML_CUDA_HYBRID_KEYS_PER_SPLIT", 16, 1, 4096);
    static const int max_split_limit       = ggml_cuda_hybrid_split_setting("GGML_CUDA_HYBRID_MAX_SPLIT", 512, 1, 4096);
    constexpr int64_t min_keys_for_split = 32;
    constexpr uint64_t partial_budget_bytes = 64ull << 20;
    if (n_keys < min_keys_for_split || n_rows == 0 || D <= 0) {
        return 1;
    }

    const int device = ggml_cuda_get_device();
    int max_split_k = std::max(1, std::min(max_split_limit, ggml_cuda_info().devices[device].nsm*16));
    const uint64_t bytes_per_split = n_rows*(uint64_t) D*sizeof(float) + n_rows*sizeof(float2);
    if (bytes_per_split > 0) {
        max_split_k = (int) std::max<uint64_t>(1, std::min<uint64_t>(max_split_k, partial_budget_bytes/bytes_per_split));
    }
    const int requested = (int) ((n_keys + target_keys_per_block - 1)/target_keys_per_block);
    return std::max(1, std::min(max_split_k, requested));
}

template<ggml_type type>
static void ggml_cuda_flash_attn_ext_hybrid_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k_cold = dst->src[1];
    const ggml_tensor * v_cold = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    const ggml_tensor * k_hot = dst->src[5];
    const ggml_tensor * v_hot = dst->src[6];
    const ggml_tensor * q_positions = dst->src[7];
    const ggml_tensor * k_positions = dst->src[8];

    GGML_ASSERT(q && k_cold && v_cold && k_hot && v_hot);
    const int64_t n_keys = k_cold->ne[1] + k_hot->ne[1];
    GGML_ASSERT(q->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(k_cold->type == type && v_cold->type == type);
    GGML_ASSERT(k_hot->type == GGML_TYPE_F16 && v_hot->type == GGML_TYPE_F16);
    GGML_ASSERT(q->ne[0] == k_cold->ne[0] && (q->ne[0] == 128 || q->ne[0] == 256));
    GGML_ASSERT(k_cold->ne[0] == v_cold->ne[0] && k_hot->ne[0] == v_hot->ne[0]);
    GGML_ASSERT(k_cold->ne[0] == k_hot->ne[0]);
    GGML_ASSERT(k_cold->ne[1] == v_cold->ne[1] && k_hot->ne[1] == v_hot->ne[1]);
    GGML_ASSERT(k_cold->ne[1] > 0 && k_hot->ne[1] > 0);
    GGML_ASSERT(q->ne[2] > 0 && k_cold->ne[2] > 0 && q->ne[2] % k_cold->ne[2] == 0 && k_cold->ne[2] == k_hot->ne[2]);
    GGML_ASSERT(q->ne[3] == k_cold->ne[3] && q->ne[3] == k_hot->ne[3]);
    GGML_ASSERT(q->nb[0] == sizeof(float));
    GGML_ASSERT(dst->nb[0] == sizeof(float));
    GGML_ASSERT(k_hot->nb[0] == sizeof(half) && v_hot->nb[0] == sizeof(half));
    GGML_ASSERT(!sinks || (sinks->type == GGML_TYPE_F32 && ggml_is_contiguous(sinks) && sinks->ne[0] == q->ne[2]));
    GGML_ASSERT(!mask || mask->type == GGML_TYPE_F16);
    GGML_ASSERT((q_positions == nullptr) == (k_positions == nullptr));
    GGML_ASSERT(!q_positions || (q_positions->type == GGML_TYPE_I32 && k_positions->type == GGML_TYPE_I32));
    GGML_ASSERT(!q_positions || (ggml_is_contiguous(q_positions) && ggml_is_contiguous(k_positions)));
    GGML_ASSERT(!q_positions || (q_positions->ne[0] == q->ne[1] &&
        (q_positions->ne[1] == 1 || q_positions->ne[1] == q->ne[2]) &&
        (q_positions->ne[2] == 1 || q_positions->ne[2] == q->ne[3]) && q_positions->ne[3] == 1 &&
        k_positions->ne[0] == n_keys && k_positions->ne[1] == k_cold->ne[2] &&
        (k_positions->ne[2] == 1 || k_positions->ne[2] == q->ne[3]) && k_positions->ne[3] == 1));
    GGML_ASSERT(!mask || (mask->type == GGML_TYPE_F16 && ggml_is_contiguous(mask) &&
        mask->ne[0] == n_keys && mask->ne[1] == q->ne[1] && mask->ne[2] > 0 && mask->ne[3] > 0 &&
        q->ne[2] % mask->ne[2] == 0 && q->ne[3] % mask->ne[3] == 0));

    const uint64_t n_rows = (uint64_t) q->ne[1]*q->ne[2]*q->ne[3];
    GGML_ASSERT(n_rows > 0 && n_rows <= std::numeric_limits<unsigned int>::max());

    float scale = 1.0f;
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    // Every query row is one warp; a draft-verification ubatch (a handful
    // of rows) or an append ubatch needs the key split just as much as
    // single-token decode, otherwise each warp walks every key serially.
    const bool use_split_k = q->ne[0] == 128 || q->ne[0] == 256;
    const int split_k = use_split_k ? ggml_cuda_hybrid_split_count(n_keys, n_rows, q->ne[0]) : 1;

    if (split_k > 1) {
        const uint64_t partial_rows = n_rows*(uint64_t) split_k;
        GGML_ASSERT(partial_rows <= std::numeric_limits<unsigned int>::max());
        GGML_ASSERT(partial_rows <= std::numeric_limits<size_t>::max()/((size_t) q->ne[0]));

        ggml_cuda_pool & pool = ctx.pool();
        ggml_cuda_pool_alloc<float> partial_o_alloc(pool, (size_t) partial_rows*(size_t) q->ne[0]);
        ggml_cuda_pool_alloc<float2> partial_meta_alloc(pool, (size_t) partial_rows);
        ggml_cuda_flash_attn_ext_hybrid_kernel<type, true><<<(unsigned int) (n_rows*split_k), 32, 0, ctx.stream()>>>(
            (const float *) q->data,
            (const char *) k_cold->data,
            (const char *) v_cold->data,
            (const char *) k_hot->data,
            (const char *) v_hot->data,
            mask ? (const char *) mask->data : nullptr,
            q_positions ? (const char *) q_positions->data : nullptr,
            k_positions ? (const char *) k_positions->data : nullptr,
            sinks ? (const float *) sinks->data : nullptr,
            (float *) dst->data,
            q->ne[0], q->ne[1], q->ne[2], q->ne[3], k_cold->ne[1], k_hot->ne[1], k_cold->ne[2],
            q->nb[1], q->nb[2], q->nb[3],
            dst->nb[1], dst->nb[2], dst->nb[3],
            k_cold->nb[1], k_cold->nb[2], k_cold->nb[3],
            v_cold->nb[1], v_cold->nb[2], v_cold->nb[3],
            k_hot->nb[1], k_hot->nb[2], k_hot->nb[3],
            v_hot->nb[1], v_hot->nb[2], v_hot->nb[3],
            mask ? mask->nb[0] : 0, mask ? mask->nb[1] : 0, mask ? mask->nb[2] : 0, mask ? mask->nb[3] : 0,
            mask ? mask->ne[2] : 1, mask ? mask->ne[3] : 1,
            q_positions ? q_positions->nb[0] : 0, q_positions ? q_positions->nb[1] : 0, q_positions ? q_positions->nb[2] : 0,
            q_positions ? q_positions->ne[1] : 1, q_positions ? q_positions->ne[2] : 1,
            k_positions ? k_positions->nb[0] : 0, k_positions ? k_positions->nb[1] : 0, k_positions ? k_positions->nb[2] : 0,
            k_positions ? k_positions->ne[2] : 1,
            scale, max_bias, logit_softcap, partial_o_alloc.get(), partial_meta_alloc.get(), split_k);
        CUDA_CHECK(cudaGetLastError());

        const dim3 merge_blocks((unsigned int) n_rows, 1, 1);
        if (q->ne[0] == 128) {
            ggml_cuda_flash_attn_ext_hybrid_merge_kernel<128><<<merge_blocks, 32, 0, ctx.stream()>>>(
                partial_o_alloc.get(), partial_meta_alloc.get(), sinks ? (const float *) sinks->data : nullptr,
                (float *) dst->data, q->ne[1], q->ne[2], q->ne[3], dst->nb[1], dst->nb[2], dst->nb[3], split_k);
        } else {
            ggml_cuda_flash_attn_ext_hybrid_merge_kernel<256><<<merge_blocks, 32, 0, ctx.stream()>>>(
                partial_o_alloc.get(), partial_meta_alloc.get(), sinks ? (const float *) sinks->data : nullptr,
                (float *) dst->data, q->ne[1], q->ne[2], q->ne[3], dst->nb[1], dst->nb[2], dst->nb[3], split_k);
        }
        CUDA_CHECK(cudaGetLastError());
    } else {
        ggml_cuda_flash_attn_ext_hybrid_kernel<type, false><<<(unsigned int) n_rows, 32, 0, ctx.stream()>>>(
            (const float *) q->data,
            (const char *) k_cold->data,
            (const char *) v_cold->data,
            (const char *) k_hot->data,
            (const char *) v_hot->data,
            mask ? (const char *) mask->data : nullptr,
            q_positions ? (const char *) q_positions->data : nullptr,
            k_positions ? (const char *) k_positions->data : nullptr,
            sinks ? (const float *) sinks->data : nullptr,
            (float *) dst->data,
            q->ne[0], q->ne[1], q->ne[2], q->ne[3], k_cold->ne[1], k_hot->ne[1], k_cold->ne[2],
            q->nb[1], q->nb[2], q->nb[3],
            dst->nb[1], dst->nb[2], dst->nb[3],
            k_cold->nb[1], k_cold->nb[2], k_cold->nb[3],
            v_cold->nb[1], v_cold->nb[2], v_cold->nb[3],
            k_hot->nb[1], k_hot->nb[2], k_hot->nb[3],
            v_hot->nb[1], v_hot->nb[2], v_hot->nb[3],
            mask ? mask->nb[0] : 0, mask ? mask->nb[1] : 0, mask ? mask->nb[2] : 0, mask ? mask->nb[3] : 0,
            mask ? mask->ne[2] : 1, mask ? mask->ne[3] : 1,
            q_positions ? q_positions->nb[0] : 0, q_positions ? q_positions->nb[1] : 0, q_positions ? q_positions->nb[2] : 0,
            q_positions ? q_positions->ne[1] : 1, q_positions ? q_positions->ne[2] : 1,
            k_positions ? k_positions->nb[0] : 0, k_positions ? k_positions->nb[1] : 0, k_positions ? k_positions->nb[2] : 0,
            k_positions ? k_positions->ne[2] : 1,
            scale, max_bias, logit_softcap, nullptr, nullptr, 1);
        CUDA_CHECK(cudaGetLastError());
    }
}

// ---------------------------------------------------------------------------
// Paged hybrid attention (ggml_flash_attn_ext_hybrid_paged).
//
// Every query token attends the rows named by its own sequence's per-KV-head
// list: entry i is (arena row, position, hot row). The query's recent window
// selects F16; older positions always read the TurboQuant4 arena.
// Rows of other sequences never enter the walk, so a decode token
// of a compacted sequence touches ~2K rows while another sequence's 256K
// prompt sits in the same arena. One warp per (query row, key split); the
// merge kernel above reduces the splits.
// ---------------------------------------------------------------------------

template<ggml_type type, bool split_k_path>
static __global__ void ggml_cuda_flash_attn_ext_hybrid_paged_kernel(
        const float * q,
        const char * k_cold,
        const char * v_cold,
        const char * k_hot,
        const char * v_hot,
        const int32_t * q_meta,      // [2, NQ]: position, seq
        const int32_t * k_list,      // [3, max_len, NKV_HEADS, NSEQ]
        const int32_t * k_list_len,  // [NKV_HEADS, NSEQ]
        float * dst,
        const int64_t D,
        const int64_t NQ,
        const int64_t NQ_HEADS,
        const int64_t NKV_HEADS,
        const int64_t NSEQ,
        const int64_t max_len,
        const int64_t n_cold_rows,
        const int64_t n_hot_rows,
        const size_t q_nb1,
        const size_t q_nb2,
        const size_t dst_nb1,
        const size_t dst_nb2,
        const size_t kc_nb1,
        const size_t kc_nb2,
        const size_t vc_nb1,
        const size_t vc_nb2,
        const size_t kh_nb1,
        const size_t kh_nb2,
        const size_t vh_nb1,
        const size_t vh_nb2,
        const float scale,
        const float logit_softcap,
        const int32_t recent_window,
        float * partial_o,
        float2 * partial_meta,
        const int split_k) {
    const int64_t blocks_per_row = split_k_path ? split_k : 1;
    const int64_t block = (int64_t) blockIdx.x;
    const int64_t row = block / blocks_per_row;
    const int64_t split = block % blocks_per_row;
    const int64_t rows = NQ*NQ_HEADS;
    if (row >= rows) {
        return;
    }
    const int lane = threadIdx.x & 31;
    const float lane_centroid = type == GGML_TYPE_TURBO3_5
        ? ggml_cuda_turbo35_centroid_lane() : ggml_cuda_turbo4_centroid_lane();

    // row = iq2*NQ + iq1 (head-major, like the dense kernel with NSEQ == 1)
    const int64_t iq2 = row/NQ;
    const int64_t iq1 = row - iq2*NQ;
    const int64_t kv_head = iq2/(NQ_HEADS/NKV_HEADS);

    const int32_t q_position = q_meta[2*iq1 + 0];
    const int32_t q_seq      = q_meta[2*iq1 + 1];

    float output[8] = {};
    float M = -INFINITY;
    float S = 0.0f;

    const float * q_row = (const float *) ((const char *) q + iq1*q_nb1 + iq2*q_nb2);

    if (q_seq >= 0 && q_seq < NSEQ) {
        const int64_t list_len = min((int64_t) k_list_len[kv_head + NKV_HEADS*q_seq], max_len);
        const int32_t * list = k_list + 3*max_len*(kv_head + NKV_HEADS*q_seq);
        int64_t first = 0, last = list_len;
        while (first < last) {
            const int64_t middle = first + (last - first)/2;
            if (list[3*middle + 1] <= q_position) {
                first = middle + 1;
            } else {
                last = middle;
            }
        }
        // Future draft rows must not change an earlier query's reduction groups.
        const int64_t n_keys = first;
        const int64_t key_begin = split_k_path ? (n_keys*split)/split_k : 0;
        const int64_t key_end   = split_k_path ? (n_keys*(split + 1))/split_k : n_keys;
        for (int64_t key_index = key_begin; key_index < key_end; ++key_index) {
            const int32_t cold_row = list[3*key_index + 0];
            const int32_t key_position = list[3*key_index + 1];
            const int32_t hot_row = list[3*key_index + 2];
            if (cold_row < 0 || cold_row >= n_cold_rows || key_position < 0 || q_position < key_position) {
                continue;
            }
            // Recent keys read their F16 hot row. The ring protects the newest
            // cells, not positions, so an M-RoPE image (many cells on one
            // position) can have recent keys without one; those read their
            // TurboQuant arena copy, which every token has.
            const bool hot = (int64_t) q_position - key_position < recent_window &&
                hot_row >= 0 && hot_row < n_hot_rows;
            const int64_t local_key = hot ? hot_row : cold_row;
            const char * k_row = hot
                ? k_hot + local_key*kh_nb1 + kv_head*kh_nb2
                : k_cold + local_key*kc_nb1 + kv_head*kc_nb2;
            const char * v_row = hot
                ? v_hot + local_key*vh_nb1 + kv_head*vh_nb2
                : v_cold + local_key*vc_nb1 + kv_head*vc_nb2;

            float dot = 0.0f;
            for (int64_t d = lane; d < D; d += 32) {
                const float key = hot
                    ? __half2float(*(const half *) (k_row + d*sizeof(half)))
                    : ggml_cuda_hybrid_turbo_value<type>(k_row, (int) d, lane_centroid);
                dot += q_row[d]*key;
            }
            dot = ggml_cuda_hybrid_warp_sum(dot);
            float score = dot*scale;
            if (logit_softcap != 0.0f) {
                score = logit_softcap*tanhf(score);
            }

            const float M_new = fmaxf(M, score);
            const float old_scale = S == 0.0f ? 0.0f : expf(M - M_new);
            const float weight = expf(score - M_new);
            for (int64_t d = lane; d < D; d += 32) {
                const float value = hot
                    ? __half2float(*(const half *) (v_row + d*sizeof(half)))
                    : ggml_cuda_hybrid_turbo_value<type>(v_row, (int) d, lane_centroid);
                output[d/32] = __fmaf_rn(value, weight, __fmul_rn(output[d/32], old_scale));
            }
            S = S*old_scale + weight;
            M = M_new;
        }
    }

    if constexpr (!split_k_path) {
        float * dst_row = (float *) ((char *) dst + iq2*dst_nb1 + iq1*dst_nb2);
        const float inv = S == 0.0f ? 0.0f : 1.0f/S;
        for (int64_t d = lane; d < D; d += 32) {
            dst_row[d] = output[d/32]*inv;
        }
    } else {
        float * partial_row = partial_o + (row*split_k + split)*D;
        for (int64_t d = lane; d < D; d += 32) {
            partial_row[d] = output[d/32];
        }
        if (lane == 0) {
            partial_meta[row*split_k + split] = make_float2(M, S);
        }
    }
}

// One query position and all six Q heads of one KV head share each K/V decode.
// Per-head reductions keep the scalar F32 order and the existing split layout.
template<ggml_type type, bool split_k_path>
static __global__ void ggml_cuda_flash_attn_ext_hybrid_paged_gqa6_kernel(
        const float * q,
        const char * k_cold,
        const char * v_cold,
        const char * k_hot,
        const char * v_hot,
        const int32_t * q_meta,
        const int32_t * k_list,
        const int32_t * k_list_len,
        float * dst,
        const int64_t D,
        const int64_t NQ,
        const int64_t NQ_HEADS,
        const int64_t NKV_HEADS,
        const int64_t NSEQ,
        const int64_t max_len,
        const int64_t n_cold_rows,
        const int64_t n_hot_rows,
        const size_t q_nb1,
        const size_t q_nb2,
        const size_t dst_nb1,
        const size_t dst_nb2,
        const size_t kc_nb1,
        const size_t kc_nb2,
        const size_t vc_nb1,
        const size_t vc_nb2,
        const size_t kh_nb1,
        const size_t kh_nb2,
        const size_t vh_nb1,
        const size_t vh_nb2,
        const float scale,
        const float logit_softcap,
        const int32_t recent_window,
        float * partial_o,
        float2 * partial_meta,
        const int split_k) {
    constexpr int n_heads = 6;
    constexpr int n_values = 256/32;
    const int64_t blocks_per_group = split_k_path ? split_k : 1;
    const int64_t block = (int64_t) blockIdx.x;
    const int64_t group = block/blocks_per_group;
    const int64_t split = block%blocks_per_group;
    if (group >= NQ*(NQ_HEADS/n_heads)) {
        return;
    }
    const int lane = threadIdx.x & 31;
    const float lane_centroid = type == GGML_TYPE_TURBO3_5
        ? ggml_cuda_turbo35_centroid_lane() : ggml_cuda_turbo4_centroid_lane();
    const int64_t kv_head = group/NQ;
    const int64_t iq1 = group - kv_head*NQ;
    const int64_t head0 = kv_head*n_heads;
    const int32_t q_position = q_meta[2*iq1 + 0];
    const int32_t q_seq = q_meta[2*iq1 + 1];

    float output[n_heads][n_values] = {};
    float M[n_heads];
    float S[n_heads] = {};
#pragma unroll
    for (int h = 0; h < n_heads; ++h) {
        M[h] = -INFINITY;
    }

    if (q_seq >= 0 && q_seq < NSEQ) {
        const int64_t list_len = min((int64_t) k_list_len[kv_head + NKV_HEADS*q_seq], max_len);
        const int32_t * list = k_list + 3*max_len*(kv_head + NKV_HEADS*q_seq);
        int64_t first = 0, last = list_len;
        while (first < last) {
            const int64_t middle = first + (last - first)/2;
            if (list[3*middle + 1] <= q_position) {
                first = middle + 1;
            } else {
                last = middle;
            }
        }
        const int64_t n_keys = first;
        const int64_t key_begin = split_k_path ? (n_keys*split)/split_k : 0;
        const int64_t key_end = split_k_path ? (n_keys*(split + 1))/split_k : n_keys;
        for (int64_t key_index = key_begin; key_index < key_end; ++key_index) {
            const int32_t cold_row = list[3*key_index + 0];
            const int32_t key_position = list[3*key_index + 1];
            const int32_t hot_row = list[3*key_index + 2];
            if (cold_row < 0 || cold_row >= n_cold_rows || key_position < 0 || q_position < key_position) {
                continue;
            }
            // Recent keys read their F16 hot row. The ring protects the newest
            // cells, not positions, so an M-RoPE image (many cells on one
            // position) can have recent keys without one; those read their
            // TurboQuant arena copy, which every token has.
            const bool hot = (int64_t) q_position - key_position < recent_window &&
                hot_row >= 0 && hot_row < n_hot_rows;
            const int64_t local_key = hot ? hot_row : cold_row;
            const char * k_row = hot
                ? k_hot + local_key*kh_nb1 + kv_head*kh_nb2
                : k_cold + local_key*kc_nb1 + kv_head*kc_nb2;
            const char * v_row = hot
                ? v_hot + local_key*vh_nb1 + kv_head*vh_nb2
                : v_cold + local_key*vc_nb1 + kv_head*vc_nb2;

            float keys[n_values];
            float values[n_values];
#pragma unroll
            for (int i = 0; i < n_values; ++i) {
                const int d = lane + 32*i;
                keys[i] = hot
                    ? __half2float(*(const half *) (k_row + d*sizeof(half)))
                    : ggml_cuda_hybrid_turbo_value<type>(k_row, d, lane_centroid);
                values[i] = hot
                    ? __half2float(*(const half *) (v_row + d*sizeof(half)))
                    : ggml_cuda_hybrid_turbo_value<type>(v_row, d, lane_centroid);
            }
#pragma unroll
            for (int h = 0; h < n_heads; ++h) {
                const float * q_row = (const float *) ((const char *) q + iq1*q_nb1 + (head0 + h)*q_nb2);
                float dot = 0.0f;
#pragma unroll
                for (int i = 0; i < n_values; ++i) {
                    dot += q_row[lane + 32*i]*keys[i];
                }
                dot = ggml_cuda_hybrid_warp_sum(dot);
                float score = dot*scale;
                if (logit_softcap != 0.0f) {
                    score = logit_softcap*tanhf(score);
                }
                const float M_new = fmaxf(M[h], score);
                const float old_scale = S[h] == 0.0f ? 0.0f : expf(M[h] - M_new);
                const float weight = expf(score - M_new);
#pragma unroll
                for (int i = 0; i < n_values; ++i) {
                    output[h][i] = __fmaf_rn(values[i], weight, __fmul_rn(output[h][i], old_scale));
                }
                S[h] = S[h]*old_scale + weight;
                M[h] = M_new;
            }
        }
    }

#pragma unroll
    for (int h = 0; h < n_heads; ++h) {
        const int64_t iq2 = head0 + h;
        const int64_t row = iq2*NQ + iq1;
        if constexpr (!split_k_path) {
            float * dst_row = (float *) ((char *) dst + iq2*dst_nb1 + iq1*dst_nb2);
            const float inv = S[h] == 0.0f ? 0.0f : 1.0f/S[h];
#pragma unroll
            for (int i = 0; i < n_values; ++i) {
                dst_row[lane + 32*i] = output[h][i]*inv;
            }
        } else {
            float * partial_row = partial_o + (row*split_k + split)*D;
#pragma unroll
            for (int i = 0; i < n_values; ++i) {
                partial_row[lane + 32*i] = output[h][i];
            }
            if (lane == 0) {
                partial_meta[row*split_k + split] = make_float2(M[h], S[h]);
            }
        }
    }
}

template<ggml_type type>
static void ggml_cuda_flash_attn_ext_hybrid_paged_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k_cold = dst->src[1];
    const ggml_tensor * v_cold = dst->src[2];
    const ggml_tensor * k_hot = dst->src[5];
    const ggml_tensor * v_hot = dst->src[6];
    const ggml_tensor * q_meta = dst->src[7];
    const ggml_tensor * k_list = dst->src[8];
    const ggml_tensor * k_list_len = dst->src[9];

    GGML_ASSERT(q && k_cold && v_cold && k_hot && v_hot && q_meta && k_list && k_list_len);
    GGML_ASSERT(q->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(k_cold->type == type && v_cold->type == type);
    GGML_ASSERT(k_hot->type == GGML_TYPE_F16 && v_hot->type == GGML_TYPE_F16);
    GGML_ASSERT(q->ne[0] == k_cold->ne[0] && (q->ne[0] == 128 || q->ne[0] == 256));
    GGML_ASSERT(q->ne[3] == 1);
    GGML_ASSERT(q->nb[0] == sizeof(float) && dst->nb[0] == sizeof(float));
    GGML_ASSERT(k_hot->nb[0] == sizeof(half) && v_hot->nb[0] == sizeof(half));
    GGML_ASSERT(q_meta->type == GGML_TYPE_I32 && k_list->type == GGML_TYPE_I32 && k_list_len->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(q_meta) && ggml_is_contiguous(k_list) && ggml_is_contiguous(k_list_len));
    GGML_ASSERT(q_meta->ne[0] == 2 && q_meta->ne[1] == q->ne[1]);
    GGML_ASSERT(k_list->ne[0] == 3 && k_list->ne[2] == k_cold->ne[2]);
    GGML_ASSERT(k_list_len->ne[0] == k_cold->ne[2] && k_list_len->ne[1] == k_list->ne[3]);

    const uint64_t n_rows = (uint64_t) q->ne[1]*q->ne[2];
    GGML_ASSERT(n_rows > 0 && n_rows <= std::numeric_limits<unsigned int>::max());

    int32_t recent_window;
    memcpy(&recent_window, (const int32_t *) dst->op_params + 5, sizeof(recent_window));
    GGML_ASSERT(recent_window > 0);

    float scale = 1.0f;
    float logit_softcap = 0.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    // The split count follows the longest list any query in this batch can
    // walk; short lists simply leave splits empty (the merge handles S == 0).
    const int64_t max_len = k_list->ne[1];
    const int split_k = ggml_cuda_hybrid_split_count(max_len, n_rows, q->ne[0]);
    // On by default: Qwen3.5 27B paged decode with 4 users x MTP3 measured
    // 244 -> 269 tok/s and 8 users plain 199 -> 216 (64K prompts, RTX 5090),
    // bit-identical results. GGML_CUDA_PAGED_GQA6=0 restores the per-row path.
    static const bool gqa6_enabled = []() {
        const char * value = std::getenv("GGML_CUDA_PAGED_GQA6");
        return !(value && value[0] == '0' && value[1] == '\0');
    }();
    // A single query exposes too few groups on large GPUs; retain its baseline.
    const bool use_gqa6 = gqa6_enabled && q->ne[1] >= 8 &&
        q->ne[0] == 256 && q->ne[2] == 24 && k_cold->ne[2] == 4;

    const auto launch = [&](auto split_tag, float * partial_o, float2 * partial_meta) {
        constexpr bool split_path = decltype(split_tag)::value;
        if (use_gqa6) {
            const uint64_t n_groups = n_rows/6;
            const unsigned int blocks = (unsigned int) (split_path ? n_groups*split_k : n_groups);
            ggml_cuda_flash_attn_ext_hybrid_paged_gqa6_kernel<type, split_path><<<blocks, 32, 0, ctx.stream()>>>(
                (const float *) q->data,
                (const char *) k_cold->data, (const char *) v_cold->data,
                (const char *) k_hot->data, (const char *) v_hot->data,
                (const int32_t *) q_meta->data, (const int32_t *) k_list->data, (const int32_t *) k_list_len->data,
                (float *) dst->data,
                q->ne[0], q->ne[1], q->ne[2], k_cold->ne[2], k_list->ne[3], max_len,
                k_cold->ne[1], k_hot->ne[1],
                q->nb[1], q->nb[2], dst->nb[1], dst->nb[2],
                k_cold->nb[1], k_cold->nb[2], v_cold->nb[1], v_cold->nb[2],
                k_hot->nb[1], k_hot->nb[2], v_hot->nb[1], v_hot->nb[2],
                scale, logit_softcap, recent_window, partial_o, partial_meta, split_path ? split_k : 1);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        const unsigned int blocks = (unsigned int) (split_path ? n_rows*split_k : n_rows);
        ggml_cuda_flash_attn_ext_hybrid_paged_kernel<type, split_path><<<blocks, 32, 0, ctx.stream()>>>(
            (const float *) q->data,
            (const char *) k_cold->data, (const char *) v_cold->data,
            (const char *) k_hot->data, (const char *) v_hot->data,
            (const int32_t *) q_meta->data, (const int32_t *) k_list->data, (const int32_t *) k_list_len->data,
            (float *) dst->data,
            q->ne[0], q->ne[1], q->ne[2], k_cold->ne[2], k_list->ne[3], max_len,
            k_cold->ne[1], k_hot->ne[1],
            q->nb[1], q->nb[2], dst->nb[1], dst->nb[2],
            k_cold->nb[1], k_cold->nb[2], v_cold->nb[1], v_cold->nb[2],
            k_hot->nb[1], k_hot->nb[2], v_hot->nb[1], v_hot->nb[2],
            scale, logit_softcap, recent_window, partial_o, partial_meta, split_path ? split_k : 1);
        CUDA_CHECK(cudaGetLastError());
    };

    if (split_k > 1) {
        const uint64_t partial_rows = n_rows*(uint64_t) split_k;
        ggml_cuda_pool & pool = ctx.pool();
        ggml_cuda_pool_alloc<float> partial_o_alloc(pool, (size_t) partial_rows*(size_t) q->ne[0]);
        ggml_cuda_pool_alloc<float2> partial_meta_alloc(pool, (size_t) partial_rows);
        launch(std::true_type{}, partial_o_alloc.get(), partial_meta_alloc.get());
        const dim3 merge_blocks((unsigned int) n_rows, 1, 1);
        if (q->ne[0] == 128) {
            ggml_cuda_flash_attn_ext_hybrid_merge_kernel<128><<<merge_blocks, 32, 0, ctx.stream()>>>(
                partial_o_alloc.get(), partial_meta_alloc.get(), nullptr,
                (float *) dst->data, q->ne[1], q->ne[2], 1, dst->nb[1], dst->nb[2], dst->nb[3], split_k);
        } else {
            ggml_cuda_flash_attn_ext_hybrid_merge_kernel<256><<<merge_blocks, 32, 0, ctx.stream()>>>(
                partial_o_alloc.get(), partial_meta_alloc.get(), nullptr,
                (float *) dst->data, q->ne[1], q->ne[2], 1, dst->nb[1], dst->nb[2], dst->nb[3], split_k);
        }
        CUDA_CHECK(cudaGetLastError());
    } else {
        launch(std::false_type{}, nullptr, nullptr);
    }
}

void ggml_cuda_flash_attn_ext_hybrid(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    switch (dst->src[1]->type) {
        case GGML_TYPE_TURBO4_0:
            return ggml_cuda_flash_attn_ext_hybrid_impl<GGML_TYPE_TURBO4_0>(ctx, dst);
        case GGML_TYPE_TURBO3_5:
            return ggml_cuda_flash_attn_ext_hybrid_impl<GGML_TYPE_TURBO3_5>(ctx, dst);
        default:
            GGML_ABORT("unsupported hybrid TurboQuant cache type");
    }
}

void ggml_cuda_flash_attn_ext_hybrid_paged(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    switch (dst->src[1]->type) {
        case GGML_TYPE_TURBO4_0:
            return ggml_cuda_flash_attn_ext_hybrid_paged_impl<GGML_TYPE_TURBO4_0>(ctx, dst);
        case GGML_TYPE_TURBO3_5:
            return ggml_cuda_flash_attn_ext_hybrid_paged_impl<GGML_TYPE_TURBO3_5>(ctx, dst);
        default:
            GGML_ABORT("unsupported hybrid TurboQuant cache type");
    }
}
