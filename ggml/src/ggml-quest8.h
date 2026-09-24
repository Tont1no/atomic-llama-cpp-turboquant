#pragma once

// 8-bit PyramidKV Quest page bounds (ggml_pyramidkv_quest_update/_select with
// I8 bounds): an E4M3-like magnitude (4 exponent bits, bias 7, 3 mantissa
// bits, subnormals, no NaN code, largest value 480) stored as an ORDERED byte:
// 128 + m for x >= 0, 127 - m for x < 0. Byte order is number order, so the
// running page min/max stays a plain byte min/max. Lower bounds round toward
// -inf and upper bounds toward +inf, so the codes still bound the keys.

#include <math.h>
#include <stdint.h>

#if defined(__CUDACC__)
#define GGML_QUEST8_HD __host__ __device__
#else
#define GGML_QUEST8_HD
#endif

GGML_QUEST8_HD static inline float ggml_quest8_mag_value(int m) {
    const int e = m >> 3;
    const int f = m & 7;
    return e == 0 ? (float) f * 0.001953125f : ldexpf(1.0f + (float) f * 0.125f, e - 7);
}

// largest code whose value does not exceed a (a >= 0)
GGML_QUEST8_HD static inline int ggml_quest8_mag_floor(float a) {
    if (!(a > 0.0f)) {
        return 0;
    }
    if (a >= 480.0f) {
        return 127;
    }
    int ex = 0;
    const float fr = frexpf(a, &ex);   // a = fr * 2^ex, fr in [0.5, 1)
    const int e = ex + 6;              // normal: (1 + f/8) * 2^(e - 7)
    if (e <= 0) {
        const int f = (int) (a * 512.0f);
        return f > 7 ? 7 : f;
    }
    int f = (int) ((2.0f*fr - 1.0f) * 8.0f);
    f = f > 7 ? 7 : f;
    const int m = (e << 3) | f;
    return m > 127 ? 127 : m;
}

// smallest code whose value is not below a (saturates at 480)
GGML_QUEST8_HD static inline int ggml_quest8_mag_ceil(float a) {
    const int m = ggml_quest8_mag_floor(a);
    return ggml_quest8_mag_value(m) < a && m < 127 ? m + 1 : m;
}

GGML_QUEST8_HD static inline uint8_t ggml_quest8_encode_down(float x) {
    return x >= 0.0f ? (uint8_t) (128 + ggml_quest8_mag_floor(x)) : (uint8_t) (127 - ggml_quest8_mag_ceil(-x));
}

GGML_QUEST8_HD static inline uint8_t ggml_quest8_encode_up(float x) {
    return x >= 0.0f ? (uint8_t) (128 + ggml_quest8_mag_ceil(x)) : (uint8_t) (127 - ggml_quest8_mag_floor(-x));
}

GGML_QUEST8_HD static inline float ggml_quest8_decode(uint8_t o) {
    return o >= 128 ? ggml_quest8_mag_value(o - 128) : -ggml_quest8_mag_value(127 - o);
}
