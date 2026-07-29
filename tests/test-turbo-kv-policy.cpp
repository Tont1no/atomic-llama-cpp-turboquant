#include "llama-kv-cache-turbo.h"

#include <cstdio>

static int checks = 0;
static int failures = 0;

#define CHECK(condition)                                                        \
    do {                                                                        \
        ++checks;                                                               \
        if (!(condition)) {                                                     \
            ++failures;                                                         \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        }                                                                       \
    } while (0)

int main() {
    CHECK(llama_kv_type_is_turbo(GGML_TYPE_TURBO2_0));
    CHECK(llama_kv_type_is_turbo(GGML_TYPE_TURBO3_0));
    CHECK(llama_kv_type_is_turbo(GGML_TYPE_TURBO4_0));
    CHECK(!llama_kv_type_is_turbo(GGML_TYPE_Q8_0));

    CHECK(!llama_kv_turbo_k_can_shift(GGML_TYPE_TURBO2_0));
    CHECK(!llama_kv_turbo_k_can_shift(GGML_TYPE_TURBO3_0));
    CHECK(!llama_kv_turbo_k_can_shift(GGML_TYPE_TURBO4_0));
    CHECK(llama_kv_turbo_k_can_shift(GGML_TYPE_Q8_0));

    // Sequentially use different model-local inputs. A static first-model
    // decision would make at least one of these assertions fail.
    CHECK(llama_kv_turbo_adaptive_mode(GGML_TYPE_TURBO2_0, 40, nullptr) == 7);
    CHECK(llama_kv_turbo_adaptive_mode(GGML_TYPE_TURBO4_0, 40, nullptr) == 0);
    CHECK(llama_kv_turbo_adaptive_mode(GGML_TYPE_TURBO2_0, 4, nullptr) == 0);
    CHECK(llama_kv_turbo_adaptive_mode(GGML_TYPE_TURBO4_0, 40, "5") == 5);
    CHECK(llama_kv_turbo_adaptive_mode(GGML_TYPE_TURBO2_0, 40, "0") == 0);

    CHECK(!llama_kv_turbo_innerq_requested(nullptr));
    CHECK(!llama_kv_turbo_innerq_requested(""));
    CHECK(!llama_kv_turbo_innerq_requested("0"));
    CHECK(!llama_kv_turbo_innerq_requested("invalid"));
    CHECK(llama_kv_turbo_innerq_requested("1"));
    CHECK(llama_kv_turbo_innerq_requested("128"));

    if (failures != 0) {
        std::fprintf(stderr, "%d/%d checks failed\n", failures, checks);
        return 1;
    }

    std::printf("all %d TurboQuant KV policy checks passed\n", checks);
    return 0;
}
