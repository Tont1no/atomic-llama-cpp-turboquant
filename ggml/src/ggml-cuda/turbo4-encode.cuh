#pragma once

#include "turbo4-cuda.cuh"

#include <type_traits>

static __device__ __forceinline__ uint8_t ggml_cuda_turbo_nearest(
        const block_turbo4_0 *, const float value, const int) {
    return ggml_cuda_turbo4_nearest(value);
}

static __device__ __forceinline__ uint8_t ggml_cuda_turbo_nearest(
        const block_turbo3_5 *, const float value, const int index) {
    return ggml_cuda_turbo35_nearest(value, index);
}

static __device__ __forceinline__ float ggml_cuda_turbo_centroid(
        const block_turbo4_0 *, const uint8_t code, const int) {
    return ggml_cuda_turbo4_centroid(code);
}

static __device__ __forceinline__ float ggml_cuda_turbo_centroid(
        const block_turbo3_5 *, const uint8_t code, const int index) {
    return ggml_cuda_turbo35_centroid(code, index);
}

static __device__ __forceinline__ void ggml_cuda_turbo_pack(
        block_turbo4_0 * block, const uint8_t code, const int j) {
    const uint8_t other = __shfl_sync(0xffffffff, code, (j & (WARP_SIZE - 1)) ^ 1);
    if ((j & 1) == 0) {
        block->qs[j/2] = code | (other << 4);
    }
}

static __device__ __forceinline__ void ggml_cuda_turbo_pack(
        block_turbo3_5 * block, const uint8_t code, const int j) {
    // The 64-coordinate boundary is warp-aligned; each branch has full warps.
    if (j < 64) {
        const uint8_t other = __shfl_sync(0xffffffff, code, (j & (WARP_SIZE - 1)) ^ 1);
        if ((j & 1) == 0) {
            block->qs4[j/2] = code | (other << 4);
        }
    } else {
        const int lane = j & (WARP_SIZE - 1);
        uint8_t low = 0;
        uint8_t high = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint8_t other = __shfl_sync(0xffffffff, code, (lane & ~7) + i);
            high |= ((other >> 2) & 1) << i;
            if ((i >> 2) == ((lane & 7) >> 2)) {
                low |= (other & 3) << (2 * (i & 3));
            }
        }
        if ((j & 3) == 0) {
            block->qs3[(j - 64)/4] = low;
        }
        if ((j & 7) == 0) {
            block->qs3[16 + (j - 64)/8] = high;
        }
    }
}

// Shared encoder for cache SET_ROWS and generic F32-to-TurboQuant copies.
// All 128 threads must call this with the same shared input array.
template<typename block_t>
static __device__ __forceinline__ void ggml_cuda_turbo_encode_block(
        float * values, block_t * block, const int j) {
    __shared__ float warp_sums[GGML_TURBO4_QK/WARP_SIZE];
    __shared__ float norm_sq;
    float sum = values[j] * values[j];
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, offset);
    }
    if ((j & (WARP_SIZE - 1)) == 0) {
        warp_sums[j/WARP_SIZE] = sum;
    }
    __syncthreads();
    if (j == 0) {
        float total = 0.0f;
        for (int i = 0; i < GGML_TURBO4_QK/WARP_SIZE; ++i) {
            total += warp_sums[i];
        }
        norm_sq = total;
    }
    __syncthreads();

    // Fast sqrt/div can flip the sign of WHT cancellation residues at codebook zero.
    constexpr bool is_turbo35 = std::is_same<block_t, block_turbo3_5>::value;
    const float norm = is_turbo35 ? __fsqrt_rn(norm_sq) : sqrtf(norm_sq);
    const float inverse_norm = norm > 1e-10f ?
        (is_turbo35 ? __fdiv_rn(1.0f, norm) : 1.0f/norm) : 0.0f;
    values[j] *= inverse_norm;
    __syncthreads();
    ggml_cuda_turbo4_rotate_forward(values, j);

    // Encoder v2 keeps the decoder/codebook intact. Compare the legacy
    // thresholds with a wider distribution, including the stored half norm.
    __shared__ float candidate_norms[2];
    __shared__ float candidate_errors[2];
    uint8_t codes[2];
#pragma unroll
    for (int candidate = 0; candidate < 2; ++candidate) {
        const float scale = candidate == 0 ? 1.0f : 0.7071067811865475f;
        codes[candidate] = ggml_cuda_turbo_nearest(block, values[j] * scale, j);
        const float centroid = ggml_cuda_turbo_centroid(block, codes[candidate], j);
        float recon_sum = centroid * centroid;
        for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
            recon_sum += __shfl_xor_sync(0xffffffff, recon_sum, offset);
        }
        if ((j & (WARP_SIZE - 1)) == 0) {
            warp_sums[j/WARP_SIZE] = recon_sum;
        }
        __syncthreads();
        if (j == 0) {
            float total = 0.0f;
            for (int i = 0; i < GGML_TURBO4_QK/WARP_SIZE; ++i) {
                total += warp_sums[i];
            }
            const float recon_norm = sqrtf(total);
            candidate_norms[candidate] = __half2float(__float2half(
                recon_norm > 1e-10f ? norm/recon_norm : norm));
        }
        __syncthreads();

        const float reconstructed = __fmul_rn(
            __fmul_rn(centroid, candidate_norms[candidate]), inverse_norm);
        const float error = __fsub_rn(values[j], reconstructed);
        float error_sum = __fmul_rn(error, error);
        for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
            error_sum += __shfl_xor_sync(0xffffffff, error_sum, offset);
        }
        if ((j & (WARP_SIZE - 1)) == 0) {
            warp_sums[j/WARP_SIZE] = error_sum;
        }
        __syncthreads();
        if (j == 0) {
            float total = 0.0f;
            for (int i = 0; i < GGML_TURBO4_QK/WARP_SIZE; ++i) {
                total += warp_sums[i];
            }
            candidate_errors[candidate] = total;
        }
        __syncthreads();
    }

    const int selected = candidate_errors[1] < candidate_errors[0] ? 1 : 0;
    const uint8_t code = codes[selected];
    ggml_cuda_turbo_pack(block, code, j);
    if (j == 0) {
        block->norm = __float2half(candidate_norms[selected]);
        block->rnorm = __float2half(0.0f);
    }
}

static __device__ __forceinline__ void ggml_cuda_turbo4_encode_block(
        float * values, block_turbo4_0 * block, const int j) {
    ggml_cuda_turbo_encode_block(values, block, j);
}

static __device__ __forceinline__ void ggml_cuda_turbo35_encode_block(
        float * values, block_turbo3_5 * block, const int j) {
    ggml_cuda_turbo_encode_block(values, block, j);
}
