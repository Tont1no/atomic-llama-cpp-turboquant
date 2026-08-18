#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>

static std::vector<float> run_native_case(
        enum ggml_backend_dev_type backend_type,
        int64_t                    k,
        int64_t                    n,
        int64_t                    m,
        int                        replays) {
    ggml_backend_ptr backend(ggml_backend_init_by_type(backend_type, nullptr));
    if (!backend) {
        std::fprintf(stderr, "failed to initialize backend type %d\n", (int) backend_type);
        return {};
    }

    ggml_init_params params{
        ggml_tensor_overhead() * 8 + ggml_graph_overhead_custom(16, false),
        nullptr,
        true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        return {};
    }

    ggml_tensor * weight       = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F8_E4M3, k, n);
    ggml_tensor * input        = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, k, m);
    ggml_tensor * weight_scale = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
    ggml_tensor * input_scale  = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
    ggml_tensor * output       = ggml_mul_mat_f8_e4m3(ctx.get(), weight, input, weight_scale, input_scale);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (!buffer) {
        return {};
    }

    // All input values are exactly representable after division by input_scale,
    // so this isolates cuBLASLt layout, scalar-scale, and graph-replay parity.
    constexpr std::array<uint8_t, 10> codes = {
        0x00, 0x38, 0xB8, 0x40, 0xC0, 0x30, 0xB0, 0x28, 0xA8, 0x48,
    };
    constexpr std::array<float, 10> values = {
        0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 0.5f, -0.5f, 0.25f, -0.25f, 4.0f,
    };
    std::vector<uint8_t> w((size_t) (k * n));
    std::vector<float> x((size_t) (k * m));
    for (int64_t row = 0; row < n; ++row) {
        for (int64_t col = 0; col < k; ++col) {
            w[(size_t) (row * k + col)] = codes[(size_t) ((row * 3 + col * 7) % codes.size())];
        }
    }
    constexpr float ws = 0.25f;
    constexpr float is = 0.5f;
    for (int64_t col = 0; col < m; ++col) {
        for (int64_t row = 0; row < k; ++row) {
            x[(size_t) (col * k + row)] = is * values[(size_t) ((col * 5 + row * 3) % values.size())];
        }
    }
    ggml_backend_tensor_set(weight, w.data(), 0, w.size());
    ggml_backend_tensor_set(input, x.data(), 0, x.size() * sizeof(float));
    ggml_backend_tensor_set(weight_scale, &ws, 0, sizeof(ws));
    ggml_backend_tensor_set(input_scale, &is, 0, sizeof(is));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, output);
    for (int replay = 0; replay < replays; ++replay) {
        if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "FP8 graph compute failed for M=%lld replay=%d\n", (long long) m, replay);
            return {};
        }
    }

    std::vector<float> result((size_t) (n * m));
    ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));
    return result;
}

int main(int argc, char ** argv) {
#if defined(_WIN32)
    _putenv_s("GGML_FP8_E4M3_CPU_REFERENCE", "1");
#else
    setenv("GGML_FP8_E4M3_CPU_REFERENCE", "1", 1);
#endif
    ggml_backend_load_all();

    if (argc == 2 && std::string(argv[1]) == "--cuda") {
        constexpr int64_t k = 16;
        constexpr int64_t n = 16;
        for (const int64_t m : { 1, 4, 8, 32, 512 }) {
            const std::vector<float> expected = run_native_case(GGML_BACKEND_DEVICE_TYPE_CPU, k, n, m, 1);
            const std::vector<float> actual   = run_native_case(GGML_BACKEND_DEVICE_TYPE_GPU, k, n, m, 8);
            if (expected.empty() || actual.size() != expected.size()) {
                return 1;
            }
            for (size_t i = 0; i < actual.size(); ++i) {
                const float tolerance = 2e-3f * std::fmax(1.0f, std::fabs(expected[i]));
                if (!std::isfinite(actual[i]) || std::fabs(actual[i] - expected[i]) > tolerance) {
                    std::fprintf(stderr,
                                 "FP8 CUDA mismatch M=%lld at %zu: got %.9g expected %.9g tolerance %.9g\n",
                                 (long long) m, i, actual[i], expected[i], tolerance);
                    return 1;
                }
            }
            std::fprintf(stderr, "FP8 CUDA parity PASS M=%lld (%zu outputs, 8 graph replays)\n",
                         (long long) m, actual.size());
        }
        return 0;
    }
    ggml_backend_ptr backend(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    if (!backend) {
        std::fprintf(stderr, "failed to initialize CPU backend\n");
        return 1;
    }

    constexpr int64_t k = 4;
    constexpr int64_t n = 3;
    constexpr int64_t m = 4;
    ggml_init_params params{
        ggml_tensor_overhead() * 8 + ggml_graph_overhead_custom(16, false),
        nullptr,
        true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        return 1;
    }

    ggml_tensor * weight = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F8_E4M3, k, n);
    ggml_tensor * input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, k, m);
    ggml_tensor * weight_scale = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
    ggml_tensor * input_scale = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
    ggml_tensor * output = ggml_mul_mat_f8_e4m3(ctx.get(), weight, input, weight_scale, input_scale);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (!buffer) {
        return 1;
    }

    // E4M3 codes for rows [1,2,-1,.5], [-2,1,.25,-.5], [0,1,2,4].
    const std::array<uint8_t, k * n> w = {
        0x38, 0x40, 0xB8, 0x30,
        0xC0, 0x38, 0x28, 0xB0,
        0x00, 0x38, 0x40, 0x48,
    };
    // input_scale=.5 makes the statically quantized codes exact.
    const std::array<float, k * m> x = {
        0.5f, 1.0f, -0.5f, 0.25f,
        1.0f, -1.0f, 0.25f, 2.0f,
        // Exact E4M3 midpoints after division by input_scale. RNE must
        // choose 1.0 (even code 0x38) and 1.25 (even code 0x3A).
        0.53125f, 0.59375f, 0.0f, 0.0f,
        // Negative midpoint mirrors: choose -1.0 and -1.25.
        -0.53125f, -0.59375f, -0.0f, 0.0f,
    };
    const float ws = 0.25f;
    const float is = 0.5f;
    ggml_backend_tensor_set(weight, w.data(), 0, sizeof(w));
    ggml_backend_tensor_set(input, x.data(), 0, sizeof(x));
    ggml_backend_tensor_set(weight_scale, &ws, 0, sizeof(ws));
    ggml_backend_tensor_set(input_scale, &is, 0, sizeof(is));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, output);
    if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
        return 1;
    }

    std::array<float, n * m> actual{};
    ggml_backend_tensor_get(output, actual.data(), 0, sizeof(actual));
    const std::array<float, n * m> expected = {
        0.78125f, -0.0625f, 0.25f,
        -0.0625f, -0.984375f, 1.875f,
        0.4375f, -0.09375f, 0.15625f,
        -0.4375f, 0.09375f, -0.15625f,
    };
    for (size_t i = 0; i < actual.size(); ++i) {
        if (std::fabs(actual[i] - expected[i]) > 1e-6f) {
            std::fprintf(stderr, "FP8 reference mismatch at %zu: got %.9g expected %.9g\n",
                         i, actual[i], expected[i]);
            return 1;
        }
    }
    return 0;
}
