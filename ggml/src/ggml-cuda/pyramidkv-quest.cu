#include "pyramidkv-quest.cuh"
#include "ggml-quest8.h"

#include <climits>

// PyramidKV Quest: per-page channel bounds of the unrotated keys, widened as
// tokens are written, and a per-decode-step page choice from them
// (ggml_pyramidkv_quest_update / ggml_pyramidkv_quest_select).

static constexpr int QUEST_MAX_SELECT = 256;
static constexpr int QUEST_EMIT_THREADS = 256;

static __device__ __forceinline__ void quest_atomic_min_half(half * addr, const float value) {
    unsigned short * p = (unsigned short *) addr;
    unsigned short old = *p;
    unsigned short assumed;
    do {
        assumed = old;
        if (!(value < __half2float(__ushort_as_half(assumed)))) {
            return;
        }
        old = atomicCAS(p, assumed, __half_as_ushort(__float2half(value)));
    } while (old != assumed);
}

static __device__ __forceinline__ void quest_atomic_max_half(half * addr, const float value) {
    unsigned short * p = (unsigned short *) addr;
    unsigned short old = *p;
    unsigned short assumed;
    do {
        assumed = old;
        if (!(value > __half2float(__ushort_as_half(assumed)))) {
            return;
        }
        old = atomicCAS(p, assumed, __half_as_ushort(__float2half(value)));
    } while (old != assumed);
}

// byte min/max through the aligned 32-bit word that holds the byte
static __device__ __forceinline__ void quest_atomic_min_u8(uint8_t * addr, const uint8_t value) {
    unsigned int * word = (unsigned int *) ((size_t) addr & ~(size_t) 3);
    const unsigned int shift = ((unsigned int) ((size_t) addr & 3))*8;
    unsigned int old = *word;
    unsigned int assumed;
    do {
        assumed = old;
        if (value >= ((assumed >> shift) & 0xffu)) {
            return;
        }
        old = atomicCAS(word, assumed, (assumed & ~(0xffu << shift)) | ((unsigned int) value << shift));
    } while (old != assumed);
}

static __device__ __forceinline__ void quest_atomic_max_u8(uint8_t * addr, const uint8_t value) {
    unsigned int * word = (unsigned int *) ((size_t) addr & ~(size_t) 3);
    const unsigned int shift = ((unsigned int) ((size_t) addr & 3))*8;
    unsigned int old = *word;
    unsigned int assumed;
    do {
        assumed = old;
        if (value <= ((assumed >> shift) & 0xffu)) {
            return;
        }
        old = atomicCAS(word, assumed, (assumed & ~(0xffu << shift)) | ((unsigned int) value << shift));
    } while (old != assumed);
}

template<bool i8>
static __global__ void quest_reset_kernel(
        char * bounds, int32_t * page_seqs, const int32_t * resets,
        const int64_t D, const int64_t n_kv, const int64_t n_pages,
        const size_t b_nb1, const size_t b_nb2, const bool write_meta) {
    const int64_t i = blockIdx.x;
    if (i >= resets[0]) {
        return;
    }
    const int32_t page = resets[1 + i];
    if (page < 0 || page >= n_pages) {
        return;
    }
    for (int64_t h = 0; h < n_kv; ++h) {
        char * row = bounds + h*b_nb1 + page*b_nb2;
        for (int64_t d = threadIdx.x; d < 2*D; d += blockDim.x) {
            if constexpr (i8) {
                ((uint8_t *) row)[d] = d < D ? 255 : 0;
            } else {
                ((half *) row)[d] = __float2half(d < D ? INFINITY : -INFINITY);
            }
        }
    }
    if (write_meta && threadIdx.x == 0) {
        page_seqs[page] = 0;
    }
}

template<bool i8>
static __global__ void quest_update_kernel(
        char * bounds, int32_t * page_seqs, int32_t * cell_meta,
        const char * k, const int32_t * writes,
        const int64_t D, const int64_t n_pages, const int64_t n_cells, const int page_size,
        const size_t b_nb1, const size_t b_nb2, const size_t k_nb1, const size_t k_nb2,
        const bool write_meta) {
    const int64_t t = blockIdx.x;
    const int64_t h = blockIdx.y;
    const int32_t cell = writes[3*t + 0];
    if (cell < 0 || cell >= n_cells) {
        return;
    }
    const int64_t page = cell/page_size;
    if (page >= n_pages) {
        return;
    }
    const float * key = (const float *) (k + h*k_nb1 + t*k_nb2);
    char * row = bounds + h*b_nb1 + page*b_nb2;
    for (int64_t d = threadIdx.x; d < D; d += blockDim.x) {
        const float value = key[d];
        if constexpr (i8) {
            quest_atomic_min_u8((uint8_t *) row + d, ggml_quest8_encode_down(value));
            quest_atomic_max_u8((uint8_t *) row + D + d, ggml_quest8_encode_up(value));
        } else {
            quest_atomic_min_half((half *) row + d, value);
            quest_atomic_max_half((half *) row + D + d, value);
        }
    }
    if (write_meta && h == 0 && threadIdx.x == 0) {
        const int32_t seq = writes[3*t + 2];
        cell_meta[2*cell + 0] = writes[3*t + 1];
        cell_meta[2*cell + 1] = seq;
        if (seq >= 0 && seq < 32) {
            atomicOr((unsigned int *) page_seqs + page, 1u << seq);
        }
    }
}

void ggml_cuda_op_pyramidkv_quest_update(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * bounds    = dst->src[0];
    ggml_tensor * page_seqs = dst->src[1];
    ggml_tensor * cell_meta = dst->src[2];
    const ggml_tensor * k      = dst->src[3];
    const ggml_tensor * writes = dst->src[4];
    const ggml_tensor * resets = dst->src[5];

    const int32_t page_size  = ggml_get_op_params_i32(dst, 0);
    const bool    write_meta = ggml_get_op_params_i32(dst, 1) != 0;
    const int64_t D       = k->ne[0];
    const int64_t n_kv    = k->ne[1];
    const int64_t n_tok   = k->ne[2];
    const int64_t n_pages = bounds->ne[2];
    const int64_t n_cells = cell_meta->ne[1];
    cudaStream_t stream = ctx.stream();

    const bool i8 = bounds->type == GGML_TYPE_I8;
    const int64_t n_resets = resets->ne[0] - 1;
    if (n_resets > 0) {
        auto * reset = i8 ? quest_reset_kernel<true> : quest_reset_kernel<false>;
        reset<<<(unsigned int) n_resets, 256, 0, stream>>>(
            (char *) bounds->data, (int32_t *) page_seqs->data, (const int32_t *) resets->data,
            D, n_kv, n_pages, bounds->nb[1], bounds->nb[2], write_meta);
        CUDA_CHECK(cudaGetLastError());
    }
    if (n_tok > 0) {
        const dim3 grid((unsigned int) n_tok, (unsigned int) n_kv, 1);
        auto * update = i8 ? quest_update_kernel<true> : quest_update_kernel<false>;
        update<<<grid, (unsigned int) std::min<int64_t>(D, 256), 0, stream>>>(
            (char *) bounds->data, (int32_t *) page_seqs->data, (int32_t *) cell_meta->data,
            (const char *) k->data, (const int32_t *) writes->data,
            D, n_pages, n_cells, page_size,
            bounds->nb[1], bounds->nb[2], k->nb[1], k->nb[2], write_meta);
        CUDA_CHECK(cudaGetLastError());
    }
}

static __device__ __forceinline__ float quest_warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    return value;
}

// One warp per page: the best bound over the slot's query tokens and the KV
// head's query heads. Pages the sequence never wrote score -inf.
template<int D, bool i8>
static __global__ void quest_score_kernel(
        float * scores, const char * q, const char * bounds, const int32_t * page_seqs,
        const int32_t * q_meta, const int32_t * seq_ids,
        const int64_t n_tokens, const int64_t n_q_heads, const int64_t n_kv, const int64_t n_pages,
        const size_t q_nb1, const size_t q_nb2, const size_t b_nb1, const size_t b_nb2) {
    constexpr int n_values = D/32;
    // decode table in shared memory: per-lane codes differ, so a constant
    // table would serialize
    __shared__ float lut[i8 ? 256 : 1];
    if constexpr (i8) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) {
            lut[i] = ggml_quest8_decode((uint8_t) i);
        }
        __syncthreads();
    }
    const int warp = threadIdx.x/32;
    const int lane = threadIdx.x%32;
    const int64_t page = (int64_t) blockIdx.x*(blockDim.x/32) + warp;
    const int64_t h = blockIdx.y;
    const int64_t s = blockIdx.z;
    if (page >= n_pages) {
        return;
    }
    float * out = scores + page + n_pages*(h + n_kv*s);
    const int32_t sid = seq_ids[s];
    if (sid < 0 || sid >= 32 || ((page_seqs[page] >> sid) & 1) == 0) {
        if (lane == 0) {
            *out = -INFINITY;
        }
        return;
    }
    const char * row = bounds + h*b_nb1 + page*b_nb2;
    float lo_v[n_values];
    float hi_v[n_values];
#pragma unroll
    for (int i = 0; i < n_values; ++i) {
        if constexpr (i8) {
            lo_v[i] = lut[((const uint8_t *) row)[lane + 32*i]];
            hi_v[i] = lut[((const uint8_t *) row)[D + lane + 32*i]];
        } else {
            lo_v[i] = __half2float(((const half *) row)[lane + 32*i]);
            hi_v[i] = __half2float(((const half *) row)[D + lane + 32*i]);
        }
    }
    const int64_t group = n_q_heads/n_kv;
    float best = -INFINITY;
    for (int64_t t = 0; t < n_tokens; ++t) {
        if (q_meta[2*t + 1] != s) {
            continue;
        }
        for (int64_t g = 0; g < group; ++g) {
            const float * q_row = (const float *) (q + t*q_nb2 + (h*group + g)*q_nb1);
            float sum = 0.0f;
#pragma unroll
            for (int i = 0; i < n_values; ++i) {
                const float qd = q_row[lane + 32*i];
                sum += qd >= 0.0f ? qd*hi_v[i] : qd*lo_v[i];
            }
            best = fmaxf(best, quest_warp_sum(sum));
        }
    }
    if (lane == 0) {
        *out = best;
    }
}

// One block per (KV head, sequence slot): the n_select best pages in page
// order, then their cells of the sequence that the base list does not hold.
static __global__ void quest_emit_kernel(
        int32_t * out, float * scores, uint32_t * bitmap,
        const int32_t * cell_meta, const int32_t * seq_ids,
        const int32_t * base, const int32_t * base_len,
        const int64_t n_kv, const int64_t n_pages, const int64_t n_cells, const int64_t max_len,
        const int64_t out_stride, const int n_select, const int page_size, const int64_t words) {
    const int64_t h = blockIdx.x;
    const int64_t s = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid%32;
    const int warp = tid/32;
    constexpr int n_warps = QUEST_EMIT_THREADS/32;

    uint32_t * bm = bitmap + words*(h + n_kv*s);
    int32_t * o = out + out_stride*(h + n_kv*s);
    float * sc = scores + n_pages*(h + n_kv*s);
    const int32_t sid = seq_ids[s];

    const int64_t len = min((int64_t) base_len[h + n_kv*s], max_len);
    const int32_t * bl = base + 3*max_len*(h + n_kv*s);
    for (int64_t i = tid; i < len; i += QUEST_EMIT_THREADS) {
        const int32_t cell = bl[3*i + 0];
        if (cell >= 0 && cell < n_cells) {
            atomicOr(bm + (cell >> 5), 1u << (cell & 31));
        }
    }

    __shared__ int sel[QUEST_MAX_SELECT];
    __shared__ int n_sel;
    __shared__ int done;
    __shared__ float red_v[n_warps];
    __shared__ int red_p[n_warps];
    if (tid == 0) {
        n_sel = 0;
        done = 0;
    }
    __syncthreads();

    for (int k = 0; k < n_select; ++k) {
        float bv = -INFINITY;
        int bp = INT_MAX;
        for (int64_t p = tid; p < n_pages; p += QUEST_EMIT_THREADS) {
            const float v = sc[p];
            if (v > bv || (v == bv && p < bp)) {
                bv = v;
                bp = (int) p;
            }
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float ov = __shfl_down_sync(0xffffffffu, bv, offset);
            const int op = __shfl_down_sync(0xffffffffu, bp, offset);
            if (ov > bv || (ov == bv && op < bp)) {
                bv = ov;
                bp = op;
            }
        }
        if (lane == 0) {
            red_v[warp] = bv;
            red_p[warp] = bp;
        }
        __syncthreads();
        if (tid == 0) {
            float v = red_v[0];
            int p = red_p[0];
            for (int w = 1; w < n_warps; ++w) {
                if (red_v[w] > v || (red_v[w] == v && red_p[w] < p)) {
                    v = red_v[w];
                    p = red_p[w];
                }
            }
            if (v > -INFINITY && p != INT_MAX) {
                sel[n_sel++] = p;
                sc[p] = -INFINITY;
            } else {
                done = 1;
            }
        }
        __syncthreads();
        if (done) {
            break;
        }
    }
    if (tid == 0) {
        for (int i = 1; i < n_sel; ++i) {
            const int v = sel[i];
            int j = i - 1;
            while (j >= 0 && sel[j] > v) {
                sel[j + 1] = sel[j];
                --j;
            }
            sel[j + 1] = v;
        }
    }
    __syncthreads();

    __shared__ int count;
    __shared__ int warp_counts[n_warps];
    if (tid == 0) {
        count = 0;
    }
    __syncthreads();
    for (int i = 0; i < n_sel; ++i) {
        const int64_t page = sel[i];
        for (int offset = 0; offset < page_size; offset += QUEST_EMIT_THREADS) {
            const int64_t cell = page*page_size + offset + tid;
            bool valid = offset + tid < page_size && cell < n_cells;
            int32_t pos = -1;
            if (valid) {
                pos = cell_meta[2*cell + 0];
                valid = cell_meta[2*cell + 1] == sid && pos >= 0 &&
                    ((bm[cell >> 5] >> (cell & 31)) & 1u) == 0;
            }
            const unsigned int ballot = __ballot_sync(0xffffffffu, valid);
            if (lane == 0) {
                warp_counts[warp] = __popc(ballot);
            }
            __syncthreads();
            int before = count;
            for (int w = 0; w < warp; ++w) {
                before += warp_counts[w];
            }
            if (valid) {
                const int idx = before + __popc(ballot & ((1u << lane) - 1u));
                o[1 + 3*idx + 0] = (int32_t) cell;
                o[1 + 3*idx + 1] = pos;
                o[1 + 3*idx + 2] = -1;
            }
            __syncthreads();
            if (tid == 0) {
                int total = 0;
                for (int w = 0; w < n_warps; ++w) {
                    total += warp_counts[w];
                }
                count += total;
            }
            __syncthreads();
        }
    }
    if (tid == 0) {
        o[0] = count;
    }
}

void ggml_cuda_op_pyramidkv_quest_select(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q         = dst->src[0];
    const ggml_tensor * bounds    = dst->src[1];
    const ggml_tensor * page_seqs = dst->src[2];
    const ggml_tensor * cell_meta = dst->src[3];
    const ggml_tensor * q_meta    = dst->src[4];
    const ggml_tensor * seq_ids   = dst->src[5];
    const ggml_tensor * base      = dst->src[6];
    const ggml_tensor * base_len  = dst->src[7];

    const int32_t n_select  = ggml_get_op_params_i32(dst, 0);
    const int32_t page_size = ggml_get_op_params_i32(dst, 1);
    GGML_ASSERT(n_select > 0 && n_select <= QUEST_MAX_SELECT);

    const int64_t D       = q->ne[0];
    const int64_t n_qh    = q->ne[1];
    const int64_t n_tok   = q->ne[2];
    const int64_t n_kv    = bounds->ne[1];
    const int64_t n_pages = bounds->ne[2];
    const int64_t n_cells = cell_meta->ne[1];
    const int64_t n_seqs  = dst->ne[2];
    const int64_t max_len = base->ne[1];
    const int64_t words   = (n_cells + 31)/32;
    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool & pool = ctx.pool();
    ggml_cuda_pool_alloc<float> scores(pool, (size_t) (n_pages*n_kv*n_seqs));
    ggml_cuda_pool_alloc<uint32_t> bitmap(pool, (size_t) (words*n_kv*n_seqs));
    CUDA_CHECK(cudaMemsetAsync(bitmap.get(), 0, (size_t) (words*n_kv*n_seqs)*sizeof(uint32_t), stream));

    constexpr int pages_per_block = 8;
    const dim3 score_grid((unsigned int) ((n_pages + pages_per_block - 1)/pages_per_block),
        (unsigned int) n_kv, (unsigned int) n_seqs);
    const bool i8 = bounds->type == GGML_TYPE_I8;
    if (D == 128) {
        auto * score = i8 ? quest_score_kernel<128, true> : quest_score_kernel<128, false>;
        score<<<score_grid, 32*pages_per_block, 0, stream>>>(
            scores.get(), (const char *) q->data, (const char *) bounds->data,
            (const int32_t *) page_seqs->data, (const int32_t *) q_meta->data, (const int32_t *) seq_ids->data,
            n_tok, n_qh, n_kv, n_pages, q->nb[1], q->nb[2], bounds->nb[1], bounds->nb[2]);
    } else {
        GGML_ASSERT(D == 256);
        auto * score = i8 ? quest_score_kernel<256, true> : quest_score_kernel<256, false>;
        score<<<score_grid, 32*pages_per_block, 0, stream>>>(
            scores.get(), (const char *) q->data, (const char *) bounds->data,
            (const int32_t *) page_seqs->data, (const int32_t *) q_meta->data, (const int32_t *) seq_ids->data,
            n_tok, n_qh, n_kv, n_pages, q->nb[1], q->nb[2], bounds->nb[1], bounds->nb[2]);
    }
    CUDA_CHECK(cudaGetLastError());

    const dim3 emit_grid((unsigned int) n_kv, (unsigned int) n_seqs, 1);
    quest_emit_kernel<<<emit_grid, QUEST_EMIT_THREADS, 0, stream>>>(
        (int32_t *) dst->data, scores.get(), bitmap.get(),
        (const int32_t *) cell_meta->data, (const int32_t *) seq_ids->data,
        (const int32_t *) base->data, (const int32_t *) base_len->data,
        n_kv, n_pages, n_cells, max_len, dst->ne[0], n_select, page_size, words);
    CUDA_CHECK(cudaGetLastError());
}
