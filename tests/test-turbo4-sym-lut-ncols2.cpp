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
    CHECK(!ggml_turbo4_sym_lut_ncols2_head_size_supported(384));
    CHECK( ggml_turbo4_sym_lut_ncols2_head_size_supported(512));
    CHECK(!ggml_turbo4_sym_lut_ncols2_head_size_supported(640));

    // Default OFF and all non-SM89/non-Turbo4/unsupported-shape cases retain
    // their existing selector. The experiment expands only N=2..8.
    CHECK(!ggml_turbo4_sym_lut_ncols2_vec_candidate(false, 890, true, true, 128, 2, true));
    CHECK(!ggml_turbo4_sym_lut_ncols2_vec_candidate(true,  860, true, true, 128, 2, true));
    CHECK(!ggml_turbo4_sym_lut_ncols2_vec_candidate(true, 1200, true, true, 128, 2, true));
    CHECK(!ggml_turbo4_sym_lut_ncols2_vec_candidate(true,  890, false, true, 128, 2, true));
    CHECK(!ggml_turbo4_sym_lut_ncols2_vec_candidate(true,  890, true, false, 128, 2, true));
    CHECK(!ggml_turbo4_sym_lut_ncols2_vec_candidate(true,  890, true, true,  64, 2, true));
    CHECK(!ggml_turbo4_sym_lut_ncols2_vec_candidate(true,  890, true, true, 192, 2, true));
    CHECK(!ggml_turbo4_sym_lut_ncols2_vec_candidate(true,  890, true, true, 384, 2, true));
    CHECK(!ggml_turbo4_sym_lut_ncols2_vec_candidate(true,  890, true, true, 640, 2, true));
    CHECK(!ggml_turbo4_sym_lut_ncols2_vec_candidate(true,  890, true, true, 128, 1, true));
    CHECK(!ggml_turbo4_sym_lut_ncols2_vec_candidate(true,  890, true, true, 128, 9, true));
    CHECK(!ggml_turbo4_sym_lut_ncols2_vec_candidate(true,  890, true, true, 128, INT64_MAX, true));
    CHECK(!ggml_turbo4_sym_lut_ncols2_vec_candidate(true,  890, true, true, 128, 2, false));
    for (int columns = 2; columns <= 8; ++columns) {
        CHECK(ggml_turbo4_sym_lut_ncols2_vec_candidate(
            true, 890, true, true, 128, columns, true));
        CHECK(ggml_turbo4_sym_lut_ncols2_vec_candidate(
            true, 890, true, true, 256, columns, true));
        CHECK(ggml_turbo4_sym_lut_ncols2_vec_candidate(
            true, 890, true, true, 512, columns, true));
    }

    // Two independent 10-half rows per dimension preserve the bank-friendly
    // stride of the one-column experiment. D=512 consumes exactly 20 KiB for
    // the dynamic LUT. D=512 moves the CUDA float KQ/combine buffer into the
    // same dynamic workspace because static CUDA shared memory cannot opt in
    // past 48 KiB. Turbo4/F16/Q8_0 V therefore request 84/52/28 KiB plus a
    // bounded sub-1-KiB static remainder.
    CHECK(ggml_turbo4_sym_lut_ncols2_shared_bytes(128) ==  5120);
    CHECK(ggml_turbo4_sym_lut_ncols2_shared_bytes(256) == 10240);
    CHECK(ggml_turbo4_sym_lut_ncols2_shared_bytes(512) == 20480);
    CHECK(ggml_turbo4_sym_lut_ncols2_d512_kq_bytes(512, 8) == 65536);
    CHECK(ggml_turbo4_sym_lut_ncols2_d512_kq_bytes(512, 4) == 32768);
    CHECK(ggml_turbo4_sym_lut_ncols2_d512_kq_bytes(512, 1) ==  8192);
    CHECK(ggml_turbo4_sym_lut_ncols2_dynamic_shared_bytes(512, 8) == 84u * 1024u);
    CHECK(ggml_turbo4_sym_lut_ncols2_dynamic_shared_bytes(512, 4) == 52u * 1024u);
    CHECK(ggml_turbo4_sym_lut_ncols2_dynamic_shared_bytes(512, 1) == 28u * 1024u);
    CHECK(
        ggml_turbo4_sym_lut_ncols2_dynamic_shared_bytes(512, 8) +
            GGML_TURBO4_SYM_LUT_NCOLS2_STATIC_OVERHEAD_BOUND <=
        GGML_TURBO4_SYM_LUT_NCOLS2_SM89_OPTIN_SHARED_LIMIT);

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
