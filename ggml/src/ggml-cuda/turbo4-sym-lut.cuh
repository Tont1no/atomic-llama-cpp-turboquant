#pragma once

#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define GGML_TURBO4_HOST_DEVICE __host__ __device__
#else
#define GGML_TURBO4_HOST_DEVICE
#endif

GGML_TURBO4_HOST_DEVICE static constexpr uint8_t
ggml_turbo4_sym_magnitude_index(const uint8_t packed_index) {
    return packed_index < 8 ? uint8_t(7 - packed_index) : uint8_t(packed_index - 8);
}

GGML_TURBO4_HOST_DEVICE static constexpr bool
ggml_turbo4_sym_is_negative(const uint8_t packed_index) {
    return packed_index < 8;
}

// Positive half of the sign-symmetric Turbo4 centroid table, ordered from the
// smallest to the largest magnitude. A constexpr selector is used instead of a
// namespace-scope array so the same source of truth is valid in host and device
// compilation and can be checked by a model-free host test.
GGML_TURBO4_HOST_DEVICE static constexpr float
ggml_turbo4_sym_magnitude(const uint8_t magnitude_index) {
    return magnitude_index == 0 ? 0.006938f :
           magnitude_index == 1 ? 0.020989f :
           magnitude_index == 2 ? 0.035597f :
           magnitude_index == 3 ? 0.051262f :
           magnitude_index == 4 ? 0.068756f :
           magnitude_index == 5 ? 0.089527f :
           magnitude_index == 6 ? 0.117195f :
                                  0.173926f;
}

GGML_TURBO4_HOST_DEVICE static constexpr float
ggml_turbo4_sym_centroid(const uint8_t packed_index) {
    const float magnitude = ggml_turbo4_sym_magnitude(ggml_turbo4_sym_magnitude_index(packed_index));
    return ggml_turbo4_sym_is_negative(packed_index) ? -magnitude : magnitude;
}

static_assert(ggml_turbo4_sym_magnitude_index(0)  == 7, "Turbo4 negative endpoint mapping changed");
static_assert(ggml_turbo4_sym_magnitude_index(7)  == 0, "Turbo4 negative near-zero mapping changed");
static_assert(ggml_turbo4_sym_magnitude_index(8)  == 0, "Turbo4 positive near-zero mapping changed");
static_assert(ggml_turbo4_sym_magnitude_index(15) == 7, "Turbo4 positive endpoint mapping changed");
static_assert(ggml_turbo4_sym_centroid(0)  == -0.173926f, "Turbo4 negative endpoint changed");
static_assert(ggml_turbo4_sym_centroid(7)  == -0.006938f, "Turbo4 negative near-zero changed");
static_assert(ggml_turbo4_sym_centroid(8)  ==  0.006938f, "Turbo4 positive near-zero changed");
static_assert(ggml_turbo4_sym_centroid(15) ==  0.173926f, "Turbo4 positive endpoint changed");

#undef GGML_TURBO4_HOST_DEVICE
