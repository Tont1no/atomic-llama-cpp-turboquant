#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#undef GGML_COMMON_DECL_CPP
#include "turbo4-sym-lut.cuh"
#include "turbo4-sym-lut-policy.h"

#include <cmath>
#include <cstdint>
#include <cstdio>

static int checks = 0;
static int failures = 0;

#define CHECK(condition)                                                             \
    do {                                                                             \
        ++checks;                                                                    \
        if (!(condition)) {                                                          \
            ++failures;                                                              \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        }                                                                            \
    } while (0)

static float decode_nibble(const uint8_t packed, const int lane) {
    const uint8_t index = uint8_t((packed >> (4 * lane)) & 0x0f);
    return ggml_turbo4_sym_centroid(index);
}

int main() {
    static constexpr float expected[16] = {
        -0.173926f, -0.117195f, -0.089527f, -0.068756f,
        -0.051262f, -0.035597f, -0.020989f, -0.006938f,
         0.006938f,  0.020989f,  0.035597f,  0.051262f,
         0.068756f,  0.089527f,  0.117195f,  0.173926f,
    };

    for (uint8_t index = 0; index < 16; ++index) {
        CHECK(ggml_turbo4_sym_centroid(index) == expected[index]);
        CHECK(ggml_turbo4_sym_centroid(index) == -ggml_turbo4_sym_centroid(uint8_t(15 - index)));
        CHECK(ggml_turbo4_sym_magnitude_index(index) < 8);
    }

    // Packed known vectors cover both nibbles, both signs, near-zero values,
    // and the two magnitude endpoints used by the CUDA hot loop.
    CHECK(decode_nibble(0xf0, 0) == -0.173926f);
    CHECK(decode_nibble(0xf0, 1) ==  0.173926f);
    CHECK(decode_nibble(0x87, 0) == -0.006938f);
    CHECK(decode_nibble(0x87, 1) ==  0.006938f);
    CHECK(decode_nibble(0x1e, 0) ==  0.117195f);
    CHECK(decode_nibble(0x1e, 1) == -0.117195f);

    // The largest specialization is D=256. Eight half magnitudes plus two
    // padding columns use 5 KiB. A 10-half row is five 32-bit banks wide and
    // therefore walks every bank before repeating (gcd(5, 32) == 1).
    CHECK(256 * (8 + 2) * 2 == 5120);
    CHECK(256 * (16 + 1) * 2 == 8704);
    bool banks_stride_9[32] = {};
    bool banks_stride_10[32] = {};
    for (int row = 0; row < 32; ++row) {
        banks_stride_9 [(row * 9  / 2) % 32] = true;
        banks_stride_10[(row * 10 / 2) % 32] = true;
    }
    int unique_banks_stride_9 = 0;
    int unique_banks_stride_10 = 0;
    for (int bank = 0; bank < 32; ++bank) {
        unique_banks_stride_9  += banks_stride_9[bank]  ? 1 : 0;
        unique_banks_stride_10 += banks_stride_10[bank] ? 1 : 0;
    }
    CHECK(unique_banks_stride_9 == 20);
    CHECK(unique_banks_stride_10 == 32);
    CHECK(QK_TURBO4 == 128);
    CHECK(64 % QK_TURBO4 != 0);  // D=64 cannot hold one valid Turbo4 block.
    CHECK(128 % QK_TURBO4 == 0);
    CHECK(256 % QK_TURBO4 == 0);

    using decision = ggml_turbo4_sym_lut_decision;
    CHECK(ggml_turbo4_sym_lut_policy(nullptr, false, 890) == decision::disabled);
    CHECK(ggml_turbo4_sym_lut_policy("0", true, 890) == decision::disabled);
    CHECK(ggml_turbo4_sym_lut_policy("", true, 890) == decision::invalid_value);
    CHECK(ggml_turbo4_sym_lut_policy("true", true, 890) == decision::invalid_value);
    CHECK(ggml_turbo4_sym_lut_policy("1", false, 890) == decision::unavailable_in_build);
    CHECK(ggml_turbo4_sym_lut_policy("1", true, 1200) == decision::unsupported_compute_capability);
    CHECK(ggml_turbo4_sym_lut_policy("1", true, 860) == decision::unsupported_compute_capability);
    CHECK(ggml_turbo4_sym_lut_policy("1", true, 890) == decision::enabled);

    if (failures != 0) {
        std::fprintf(stderr, "%d/%d checks failed\n", failures, checks);
        return 1;
    }
    std::printf("all %d Turbo4 symmetric-LUT mapping checks passed\n", checks);
    return 0;
}
