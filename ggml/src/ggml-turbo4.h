#pragma once

// Atomic source binding:
//   https://github.com/Tont1no/atomic-llama-cpp-turboquant.git
//   commit 1d6c42eeef8c0d8ae3a451b4c3c1527144689135
// The imported constants and sign tables are MIT licensed.
//
// MIT License
//
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <stdint.h>

// Shared TurboQuant4 KV-cache format constants. Keep these macros usable from C/CUDA.
// This rotated format is not a general weight quantizer.
#define GGML_TURBO4_QK             128
#define GGML_TURBO4_PACKED_BYTES   64
#define GGML_TURBO4_BLOCK_BYTES    68
#define GGML_TURBO4_CENTROID_COUNT 16
#define GGML_TURBO4_MIDPOINT_COUNT 15
#define GGML_TURBO4_INV_SQRT128    0.08838834764831845f
#define QK_TURBO4                 GGML_TURBO4_QK

#define GGML_TURBO4_CENTROID_0     (-0.173926f)
#define GGML_TURBO4_CENTROID_1     (-0.117195f)
#define GGML_TURBO4_CENTROID_2     (-0.089527f)
#define GGML_TURBO4_CENTROID_3     (-0.068756f)
#define GGML_TURBO4_CENTROID_4     (-0.051262f)
#define GGML_TURBO4_CENTROID_5     (-0.035597f)
#define GGML_TURBO4_CENTROID_6     (-0.020989f)
#define GGML_TURBO4_CENTROID_7     (-0.006938f)
#define GGML_TURBO4_CENTROID_8     ( 0.006938f)
#define GGML_TURBO4_CENTROID_9     ( 0.020989f)
#define GGML_TURBO4_CENTROID_10    ( 0.035597f)
#define GGML_TURBO4_CENTROID_11    ( 0.051262f)
#define GGML_TURBO4_CENTROID_12    ( 0.068756f)
#define GGML_TURBO4_CENTROID_13    ( 0.089527f)
#define GGML_TURBO4_CENTROID_14    ( 0.117195f)
#define GGML_TURBO4_CENTROID_15    ( 0.173926f)

#define GGML_TURBO4_CENTROIDS_INIT { \
    GGML_TURBO4_CENTROID_0, GGML_TURBO4_CENTROID_1, \
    GGML_TURBO4_CENTROID_2, GGML_TURBO4_CENTROID_3, \
    GGML_TURBO4_CENTROID_4, GGML_TURBO4_CENTROID_5, \
    GGML_TURBO4_CENTROID_6, GGML_TURBO4_CENTROID_7, \
    GGML_TURBO4_CENTROID_8, GGML_TURBO4_CENTROID_9, \
    GGML_TURBO4_CENTROID_10, GGML_TURBO4_CENTROID_11, \
    GGML_TURBO4_CENTROID_12, GGML_TURBO4_CENTROID_13, \
    GGML_TURBO4_CENTROID_14, GGML_TURBO4_CENTROID_15 \
}

#define GGML_TURBO4_MIDPOINT_0    (-0.145560f)
#define GGML_TURBO4_MIDPOINT_1    (-0.103361f)
#define GGML_TURBO4_MIDPOINT_2    (-0.079142f)
#define GGML_TURBO4_MIDPOINT_3    (-0.060009f)
#define GGML_TURBO4_MIDPOINT_4    (-0.043430f)
#define GGML_TURBO4_MIDPOINT_5    (-0.028293f)
#define GGML_TURBO4_MIDPOINT_6    (-0.013963f)
#define GGML_TURBO4_MIDPOINT_7    ( 0.000000f)
#define GGML_TURBO4_MIDPOINT_8    ( 0.013963f)
#define GGML_TURBO4_MIDPOINT_9    ( 0.028293f)
#define GGML_TURBO4_MIDPOINT_10   ( 0.043430f)
#define GGML_TURBO4_MIDPOINT_11   ( 0.060009f)
#define GGML_TURBO4_MIDPOINT_12   ( 0.079142f)
#define GGML_TURBO4_MIDPOINT_13   ( 0.103361f)
#define GGML_TURBO4_MIDPOINT_14   ( 0.145560f)

// CPU golden midpoint literals. CUDA should consume these definitions too.
#define GGML_TURBO4_MIDPOINTS_INIT { \
    GGML_TURBO4_MIDPOINT_0, GGML_TURBO4_MIDPOINT_1, \
    GGML_TURBO4_MIDPOINT_2, GGML_TURBO4_MIDPOINT_3, \
    GGML_TURBO4_MIDPOINT_4, GGML_TURBO4_MIDPOINT_5, \
    GGML_TURBO4_MIDPOINT_6, GGML_TURBO4_MIDPOINT_7, \
    GGML_TURBO4_MIDPOINT_8, GGML_TURBO4_MIDPOINT_9, \
    GGML_TURBO4_MIDPOINT_10, GGML_TURBO4_MIDPOINT_11, \
    GGML_TURBO4_MIDPOINT_12, GGML_TURBO4_MIDPOINT_13, \
    GGML_TURBO4_MIDPOINT_14 \
}

#define GGML_TURBO4_SIGNS1_INIT { \
    -1, 1, 1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, 1, 1, 1, \
    1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, 1, 1, -1, -1, -1, \
    -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1, \
    1, 1, 1, -1, -1, -1, -1, -1, 1, -1, 1, 1, 1, 1, -1, 1, \
    -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, 1, \
    1, -1, -1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, -1, \
    -1, 1, 1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, 1, -1, 1, \
    1, -1, 1, 1, -1, -1, -1, -1, -1, 1, 1, -1, 1, 1, -1, 1 \
}

#define GGML_TURBO4_SIGNS2_INIT { \
    1, 1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1, -1, -1, -1, \
    1, 1, -1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, 1, 1, \
    1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, -1, \
    1, -1, 1, 1, 1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, 1, \
    1, -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, -1, -1, 1, 1, \
    -1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, -1, -1, 1, -1, \
    1, -1, 1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, 1, \
    -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, -1 \
}

// Host matrix for ggml_mul_mat: contiguous dimension is input, next is output.
// Unlike a plain Hadamard matrix, the signed forward rotation is not symmetric.
static inline void ggml_turbo4_rotation_matrix(float * data, int inverse) {
    static const int8_t signs1[GGML_TURBO4_QK] = GGML_TURBO4_SIGNS1_INIT;
    static const int8_t signs2[GGML_TURBO4_QK] = GGML_TURBO4_SIGNS2_INIT;
    for (uint32_t row = 0; row < GGML_TURBO4_QK; ++row) {
        for (uint32_t col = 0; col < GGML_TURBO4_QK; ++col) {
            uint32_t bits = row & col;
            int parity = 0;
            while (bits != 0) {
                parity ^= bits & 1u;
                bits >>= 1;
            }
            const int8_t row_sign = inverse ? signs1[row] : signs2[row];
            const int8_t col_sign = inverse ? signs2[col] : signs1[col];
            data[row*GGML_TURBO4_QK + col] = GGML_TURBO4_INV_SQRT128 *
                (parity ? -1.0f : 1.0f) * row_sign * col_sign;
        }
    }
}
