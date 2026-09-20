#pragma once

#include "common.cuh"
#include "../ggml-turbo4.h"

#include <cstdint>

// TQ4 CUDA code is adapted from the MIT-licensed source fork at commit
// 1d6c42eeef8c0d8ae3a451b4c3c1527144689135.

#ifndef QR_TURBO4
#define QR_TURBO4 1
#endif

// Keep the CUDA constants bound to the shared CPU definition. The local
// constant-memory copies avoid device code dereferencing host-only arrays.
static __constant__ const float ggml_cuda_turbo4_centroids[16] = GGML_TURBO4_CENTROIDS_INIT;
static __constant__ const float ggml_cuda_turbo4_midpoints[15] = GGML_TURBO4_MIDPOINTS_INIT;
static __constant__ const int8_t ggml_cuda_turbo4_signs1[GGML_TURBO4_QK] = GGML_TURBO4_SIGNS1_INIT;
static __constant__ const int8_t ggml_cuda_turbo4_signs2[GGML_TURBO4_QK] = GGML_TURBO4_SIGNS2_INIT;

static __device__ __forceinline__ float ggml_cuda_turbo4_centroid(const uint8_t index) {
    return ggml_cuda_turbo4_centroids[index & 0x0f];
}

static __device__ __forceinline__ uint8_t ggml_cuda_turbo4_nearest(const float value) {
    uint8_t index = 0;
#pragma unroll
    for (int i = 0; i < 15; ++i) {
        index += value >= ggml_cuda_turbo4_midpoints[i];
    }
    return index;
}

// Warp-resident centroid table. Indexing the __constant__ table with a
// different code per lane serialises the constant cache one address at a
// time (up to 16 replays per warp load); holding centroid[lane & 15] in a
// register and shuffling it by code is a single instruction. Every lane of
// the warp must call the shuffle (uniform control flow).
static __device__ __forceinline__ float ggml_cuda_turbo4_centroid_lane() {
    return ggml_cuda_turbo4_centroids[threadIdx.x & 0x0f];
}

static __device__ __forceinline__ float ggml_cuda_turbo4_centroid_shfl(const uint8_t code, const float lane_centroid) {
    return __shfl_sync(0xFFFFFFFF, lane_centroid, code & 0x0f, 32);
}

static __device__ __forceinline__ float ggml_cuda_turbo4_dequant_element(
        const block_turbo4_0 * block, const int index, const float norm) {
    const uint8_t packed = block->qs[index >> 1];
    const uint8_t code = (packed >> ((index & 1) * 4)) & 0x0f;
    return ggml_cuda_turbo4_centroid(code) * norm;
}

// Rotate one 128-value group in-place. Every caller must use a 128-thread
// block and call this function from every thread in the block.
static __device__ __forceinline__ void ggml_cuda_turbo4_rotate_forward(float * values, const int index) {
    values[index] *= ggml_cuda_turbo4_signs1[index];
    __syncthreads();

#pragma unroll
    for (int half_width = 1; half_width < GGML_TURBO4_QK; half_width <<= 1) {
        if ((index & (2 * half_width - 1)) < half_width) {
            const float a = values[index];
            const float b = values[index + half_width];
            values[index] = a + b;
            values[index + half_width] = a - b;
        }
        __syncthreads();
    }

    values[index] *= GGML_TURBO4_INV_SQRT128 * ggml_cuda_turbo4_signs2[index];
    __syncthreads();
}
