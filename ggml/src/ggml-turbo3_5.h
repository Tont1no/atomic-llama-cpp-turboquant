#pragma once

// Experimental Ai-Loader KV format: 64 four-bit + 64 three-bit coordinates
// within the existing signed WHT128 basis. This is our own format definition,
// not a reproduction of a published TurboQuant 3.5-bit format. It is not a
// general weight quantizer. Both K and V use this same type.
//
// Three-bit centroids copied from the local MIT-licensed legacy implementation:
//   .codex-deploy/codacus/ggml/src/ggml-turbo-quant.c (CENTROIDS_3BIT)
//   SHA256 43a5686ec52c44213413f821972e76b160b58099bcddab8b66158ef7f9ac9a7d
// Four-bit centroids, WHT signs and rotation are defined by ggml-turbo4.h.
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

#include "ggml-turbo4.h"

#define GGML_TURBO3_5_QK              128
#define GGML_TURBO3_5_QS4_BYTES       32
#define GGML_TURBO3_5_QS3_LOW_BYTES   16
#define GGML_TURBO3_5_QS3_HIGH_BYTES  8
#define GGML_TURBO3_5_QS3_BYTES       24
#define GGML_TURBO3_5_PACKED_BYTES    56
#define GGML_TURBO3_5_BLOCK_BYTES     60
#define GGML_TURBO3_5_CENTROID3_COUNT 8
#define GGML_TURBO3_5_MIDPOINT3_COUNT 7
#define QK_TURBO3_5                  GGML_TURBO3_5_QK

// Payload order: coordinates 0..63 in qs4; coordinates 64..127 in qs3.
// qs3[0..15] stores four low-two-bit indices per byte, least significant first.
// qs3[16..23] stores eight high-one-bit indices per byte, least significant first.
// 56 payload bytes / 128 values = 3.5 bits. Including norm/rnorm: 3.75 bits.
// One common FP16 norm covers all 128 values; reserved rnorm is exactly zero.

#define GGML_TURBO3_5_CENTROIDS3_INIT { \
    -0.190207f, -0.118786f, -0.066822f, -0.021663f, \
     0.021663f,  0.066822f,  0.118786f,  0.190207f  \
}

// Exact format literals; equality selects the higher code, as in TQ4.
#define GGML_TURBO3_5_MIDPOINTS3_INIT { \
    -0.1544965f, -0.092804f, -0.0442425f, 0.0f, \
     0.0442425f,  0.092804f,  0.1544965f \
}
