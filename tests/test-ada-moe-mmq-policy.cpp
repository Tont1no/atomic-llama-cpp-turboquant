#include "ada-moe-mmq-policy.h"

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
    using decision = ggml_cuda_ada_moe_mmq_decision;

    CHECK(ggml_cuda_ada_moe_mmq_policy(nullptr, false, 890).decision == decision::disabled);
    CHECK(ggml_cuda_ada_moe_mmq_policy("0", true, 890).decision == decision::disabled);

    for (int batch = 2; batch <= 8; ++batch) {
        const char value[] = {char('0' + batch), '\0'};
        const auto result = ggml_cuda_ada_moe_mmq_policy(value, true, 890);
        CHECK(result.decision == decision::enabled);
        CHECK(result.min_batch == batch);
    }

    CHECK(ggml_cuda_ada_moe_mmq_policy("", true, 890).decision == decision::invalid_value);
    CHECK(ggml_cuda_ada_moe_mmq_policy("1", true, 890).decision == decision::invalid_value);
    CHECK(ggml_cuda_ada_moe_mmq_policy("9", true, 890).decision == decision::invalid_value);
    CHECK(ggml_cuda_ada_moe_mmq_policy("4x", true, 890).decision == decision::invalid_value);
    CHECK(ggml_cuda_ada_moe_mmq_policy("true", true, 890).decision == decision::invalid_value);
    CHECK(ggml_cuda_ada_moe_mmq_policy("4", false, 890).decision == decision::unavailable_in_build);
    CHECK(ggml_cuda_ada_moe_mmq_policy("4", true, 860).decision == decision::unsupported_compute_capability);
    CHECK(ggml_cuda_ada_moe_mmq_policy("4", true, 1200).decision == decision::unsupported_compute_capability);

    if (failures != 0) {
        std::fprintf(stderr, "%d/%d checks failed\n", failures, checks);
        return 1;
    }
    std::printf("all %d Ada MoE MMQ policy checks passed\n", checks);
    return 0;
}
