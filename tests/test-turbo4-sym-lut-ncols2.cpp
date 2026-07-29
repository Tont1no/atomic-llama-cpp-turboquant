#include "turbo4-sym-lut-ncols2-policy.h"

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

int main() {
    using decision = ggml_turbo4_sym_lut_ncols2_decision;

    CHECK(ggml_turbo4_sym_lut_ncols2_policy(nullptr, false, 890) == decision::disabled);
    CHECK(ggml_turbo4_sym_lut_ncols2_policy("0", true, 890) == decision::disabled);
    CHECK(ggml_turbo4_sym_lut_ncols2_policy("", true, 890) == decision::invalid_value);
    CHECK(ggml_turbo4_sym_lut_ncols2_policy("true", true, 890) == decision::invalid_value);
    CHECK(ggml_turbo4_sym_lut_ncols2_policy("1", false, 890) == decision::unavailable_in_build);
    CHECK(ggml_turbo4_sym_lut_ncols2_policy("1", true, 860) == decision::unsupported_compute_capability);
    CHECK(ggml_turbo4_sym_lut_ncols2_policy("1", true, 1200) == decision::unsupported_compute_capability);
    CHECK(ggml_turbo4_sym_lut_ncols2_policy("1", true, 890) == decision::enabled);

    CHECK( ggml_turbo4_sym_lut_ncols2_device_supported(890));
    CHECK(!ggml_turbo4_sym_lut_ncols2_device_supported(860));
    CHECK(!ggml_turbo4_sym_lut_ncols2_device_supported(1200));
    const int all_sm89[] = {890, 890};
    const int mixed_sm89_sm120[] = {890, 1200};
    const int mixed_sm86_sm89[] = {860, 890};
    CHECK(ggml_turbo4_sym_lut_ncols2_first_unsupported_device(nullptr, 0) == 0);
    CHECK(ggml_turbo4_sym_lut_ncols2_first_unsupported_device(all_sm89, 2) == -1);
    CHECK(ggml_turbo4_sym_lut_ncols2_first_unsupported_device(mixed_sm89_sm120, 2) == 1);
    CHECK(ggml_turbo4_sym_lut_ncols2_first_unsupported_device(mixed_sm86_sm89, 2) == 0);

    CHECK(GGML_TURBO4_SYM_LUT_NCOLS2_COLUMNS == 2);
    CHECK(GGML_TURBO4_SYM_LUT_MAGNITUDES == 8);
    CHECK(GGML_TURBO4_SYM_LUT_PADDING == 2);
    CHECK(!ggml_turbo4_sym_lut_ncols2_head_size_supported(64));
    CHECK( ggml_turbo4_sym_lut_ncols2_head_size_supported(128));
    CHECK( ggml_turbo4_sym_lut_ncols2_head_size_supported(256));
    CHECK(!ggml_turbo4_sym_lut_ncols2_head_size_supported(512));

    // Two independent 10-half rows per dimension preserve the bank-friendly
    // stride of the one-column experiment. D=256 is the largest specialization
    // and consumes exactly 10 KiB for the new LUT.
    CHECK(ggml_turbo4_sym_lut_ncols2_shared_bytes(128) ==  5120);
    CHECK(ggml_turbo4_sym_lut_ncols2_shared_bytes(256) == 10240);

    bool banks[32] = {};
    for (int row = 0; row < 32; ++row) {
        banks[(row * (GGML_TURBO4_SYM_LUT_MAGNITUDES + GGML_TURBO4_SYM_LUT_PADDING) / 2) % 32] = true;
    }
    int unique_banks = 0;
    for (bool visited : banks) {
        unique_banks += visited ? 1 : 0;
    }
    CHECK(unique_banks == 32);

    if (failures != 0) {
        std::fprintf(stderr, "%d/%d checks failed\n", failures, checks);
        return 1;
    }
    std::printf("all %d Turbo4 two-column symmetric-LUT contract checks passed\n", checks);
    return 0;
}
