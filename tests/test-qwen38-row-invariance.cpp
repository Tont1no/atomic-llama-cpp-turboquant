#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "qwen38-row-invariance-matrix.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace qwen38_row_invariance;

namespace {

constexpr int REPLAYS = 8;
constexpr uint32_t ROUTE_ABI = 12;
enum route : size_t {
    FP8, FP8_BATCH, NVFP4_MMVQ_M1, NVFP4_MMVQ_M8, NVFP4_FUSED_FFN,
    NVFP4_ROW_INVARIANT_HEAD_SCALED,
    NVFP4_ROW_INVARIANT_FFN_FUSED,
    NVFP4_ROW_INVARIANT_FFN_DOWN_SCALED,
    GDN, GDN_FUSED_CACHE,
    RMS_NORM, RMS_NORM_MUL,
    SSM_CONV, SSM_CONV_SILU,
    L2_NORM, SILU, SILU_MUL, BF16_MMVF, BF16_CUBLAS,
    BF16_ROW_INVARIANT_BETA, BF16_ROW_INVARIANT_ALPHA,
    FATTN_VEC, FATTN_MMA_F16, FATTN_TILE, SIGMOID, SIGMOID_MUL,
    SET_ROWS, ROPE_VIEW_SET_ROWS, FATTN_QWEN35_D256_Q8_GQA6_VEC,
    FATTN_QWEN35_D256_Q8_GQA6_HINT,
    FATTN_QWEN35_D256_Q8_GQA6_VEC_COLS2_PB1,
    ROUTE_COUNT,
};
constexpr std::array<uint8_t, 12> FP8_CODES = {
    0x38, 0x40, 0xb8, 0x30, 0xc0, 0xb0, 0x28, 0x48, 0xa8, 0xc8, 0x20, 0xa0,
};

struct route_counts {
    uint32_t abi_version;
    uint32_t route_count;
    uint64_t count[ROUTE_COUNT];
};

struct fattn_vec_launch_observation {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t evaluated;
    uint32_t forced_parallel_blocks;
    int32_t q_cols;
    int32_t ncols;
    int32_t ntiles_x;
    int32_t ntiles_kv;
    int32_t parallel_blocks;
};

using route_reset_fn = void (*)();
using route_snapshot_fn = bool (*)(route_counts *, size_t);
using fattn_vec_launch_snapshot_fn = bool (*)(fattn_vec_launch_observation *, size_t);

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error(message);
}

uint64_t fnv1a(const std::vector<float> & values) {
    const auto * bytes = reinterpret_cast<const uint8_t *>(values.data());
    size_t size = values.size() * sizeof(float);
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < size; ++i) {
        hash = (hash ^ bytes[i]) * UINT64_C(1099511628211);
    }
    return hash;
}

void require_finite(const std::vector<float> & values, std::string_view id) {
    for (size_t i = 0; i < values.size(); ++i) {
        if (!std::isfinite(values[i])) {
            fail(std::string(id) + ": non-finite output at " + std::to_string(i));
        }
    }
}

void require_bits_equal(
        const std::vector<float> & scalar, const std::vector<float> & batch,
        std::string_view id, int replay) {
    if (scalar.size() != batch.size()) {
        fail(std::string(id) + ": output cardinality mismatch");
    }
    require_finite(scalar, id);
    require_finite(batch, id);
    for (size_t i = 0; i < scalar.size(); ++i) {
        uint32_t a, b;
        std::memcpy(&a, &scalar[i], sizeof(a));
        std::memcpy(&b, &batch[i], sizeof(b));
        if (a != b) {
            char detail[256];
            std::snprintf(detail, sizeof(detail),
                    "%.*s: bit mismatch replay=%d output=%zu scalar=0x%08x batch=0x%08x scalar=%.9g batch=%.9g",
                    (int) id.size(), id.data(), replay, i, a, b, scalar[i], batch[i]);
            fail(detail);
        }
    }
}

std::vector<float> make_input(int64_t k, int64_t m) {
    // Adjacent positive/negative terms deliberately nearly cancel. The small,
    // deterministic column perturbation makes row mixing and reduction-order
    // changes visible while keeping every input finite.
    std::vector<float> result((size_t) (k * m));
    for (int64_t col = 0; col < m; ++col) {
        for (int64_t row = 0; row < k; ++row) {
            const float magnitude = 0.25f * (1 + ((row / 2 + col * 3) % 7));
            const float sign = (row & 1) ? -1.0f : 1.0f;
            const float perturb = ((row + col * 11) % 13 == 0) ? 0.03125f : 0.0f;
            result[(size_t) col * k + row] = sign * (magnitude + perturb);
        }
    }
    return result;
}

bool nvfp4_weight_negative(int64_t col, int64_t row, int salt) {
    return (((col / 3) + row * 5 + salt) & 1) != 0;
}

void validate_cancellation_sign_contract() {
    for (int64_t output_row = 0; output_row < 8; ++output_row) {
        bool fp8_positive = false, fp8_negative = false;
        bool nv_positive = false, nv_negative = false;
        for (int64_t reduction = 0; reduction < 5120; ++reduction) {
            const bool input_negative = (reduction & 1) != 0;
            const uint8_t code = FP8_CODES[(size_t) ((reduction + output_row * 5) % FP8_CODES.size())];
            const bool fp8_product_negative = ((code & 0x80) != 0) != input_negative;
            const bool nv_product_negative = nvfp4_weight_negative(reduction, output_row, 3) != input_negative;
            fp8_positive |= !fp8_product_negative;
            fp8_negative |= fp8_product_negative;
            nv_positive |= !nv_product_negative;
            nv_negative |= nv_product_negative;
        }
        if (!fp8_positive || !fp8_negative || !nv_positive || !nv_negative) {
            fail("cancellation fixture does not contain both product signs");
        }
    }
}

std::vector<uint8_t> make_fp8_weight(int64_t k, int64_t n) {
    std::vector<uint8_t> result((size_t) (k * n));
    for (int64_t row = 0; row < n; ++row) {
        bool positive_product = false, negative_product = false;
        for (int64_t col = 0; col < k; ++col) {
            const uint8_t code = FP8_CODES[(size_t) ((col + row * 5) % FP8_CODES.size())];
            result[(size_t) row * k + col] = code;
            const bool product_negative = ((code & 0x80) != 0) != ((col & 1) != 0);
            positive_product |= !product_negative;
            negative_product |= product_negative;
        }
        if (!positive_product || !negative_product) fail("FP8 fixture lost cancellation product signs");
    }
    return result;
}

std::vector<uint8_t> make_nvfp4_weight(int64_t k, int64_t n, int salt) {
    std::vector<float> source((size_t) (k * n));
    for (int64_t row = 0; row < n; ++row) {
        bool positive_product = false, negative_product = false;
        for (int64_t col = 0; col < k; ++col) {
            const float sign = nvfp4_weight_negative(col, row, salt) ? -1.0f : 1.0f;
            source[(size_t) row * k + col] = sign * (0.125f + 0.0625f * ((col * 3 + row * 7 + salt) % 15));
            const bool product_negative = std::signbit(sign) != ((col & 1) != 0);
            positive_product |= !product_negative;
            negative_product |= product_negative;
        }
        if (!positive_product || !negative_product) fail("NVFP4 fixture lost cancellation product signs");
    }
    std::vector<uint8_t> quantized(ggml_row_size(GGML_TYPE_NVFP4, k) * (size_t) n);
    const size_t written = ggml_quantize_chunk(
            GGML_TYPE_NVFP4, source.data(), quantized.data(), 0, n, k, nullptr);
    if (written != quantized.size()) {
        fail("NVFP4 quantizer byte-count drift");
    }
    return quantized;
}

struct dense_runner {
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * output = nullptr;
    int64_t k = 0, n = 0, m = 0;

    dense_runner(ggml_backend_t backend, operation op, int64_t k_, int64_t n_, int64_t m_, bool tag_candidate = true)
        : k(k_), n(n_), m(m_) {
        ggml_init_params params{
            ggml_tensor_overhead() * 24 + ggml_graph_overhead_custom(32, false), nullptr, true,
        };
        ctx.reset(ggml_init(params));
        if (!ctx) fail("failed to create dense diagnostic context");

        input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, k, m);
        ggml_set_name(input, "cancellation_input");
        ggml_tensor * diagnostic_mm = nullptr;
        ggml_tensor * diagnostic_gate_mm = nullptr;
        ggml_tensor * diagnostic_scale = nullptr;

        if (op == operation::fp8) {
            ggml_tensor * weight = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F8_E4M3, k, n);
            ggml_tensor * ws = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
            ggml_tensor * xs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
            output = ggml_mul_mat_f8_e4m3(ctx.get(), weight, input, ws, xs);
            buffer.reset(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
            if (!buffer) fail("failed to allocate FP8 tensors");
            const auto w = make_fp8_weight(k, n);
            constexpr float scale = 0.5f;
            ggml_backend_tensor_set(weight, w.data(), 0, w.size());
            ggml_backend_tensor_set(ws, &scale, 0, sizeof(scale));
            ggml_backend_tensor_set(xs, &scale, 0, sizeof(scale));
        } else {
            ggml_tensor * up = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_NVFP4, k, n);
            ggml_tensor * up_mm = ggml_mul_mat(ctx.get(), up, input);
            diagnostic_mm = up_mm;
            if (op == operation::nvfp4_head && tag_candidate) {
                ggml_mul_mat_set_hint(up_mm, GGML_HINT_MUL_MAT_ROW_INVARIANT);
            }
            if ((op == operation::nvfp4_ffn || op == operation::nvfp4_ffn_row_invariant)) {
                ggml_tensor * gate = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_NVFP4, k, n);
                ggml_tensor * up_scale = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
                ggml_tensor * gate_scale = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
                ggml_tensor * gate_mm = ggml_mul_mat(ctx.get(), gate, input);
                diagnostic_gate_mm = gate_mm;
                if (op == operation::nvfp4_ffn_row_invariant && tag_candidate) {
                    ggml_mul_mat_set_hint(up_mm, GGML_HINT_MUL_MAT_ROW_INVARIANT_FFN_UP);
                    ggml_mul_mat_set_hint(gate_mm, GGML_HINT_MUL_MAT_ROW_INVARIANT_FFN_GATE);
                }
                up_mm = ggml_mul(ctx.get(), up_mm, up_scale);
                gate_mm = ggml_mul(ctx.get(), gate_mm, gate_scale);
                output = ggml_glu_split(ctx.get(), gate_mm, up_mm, GGML_GLU_OP_SWIGLU);
                buffer.reset(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
                if (!buffer) fail("failed to allocate NVFP4 FFN tensors");
                const auto gate_q = make_nvfp4_weight(k, n, 17);
                ggml_backend_tensor_set(gate, gate_q.data(), 0, gate_q.size());
                constexpr float up_s = 0.75f, gate_s = 1.25f;
                ggml_backend_tensor_set(up_scale, &up_s, 0, sizeof(up_s));
                ggml_backend_tensor_set(gate_scale, &gate_s, 0, sizeof(gate_s));
            } else if (op == operation::nvfp4_head || op == operation::nvfp4_down_row_invariant) {
                if (op == operation::nvfp4_down_row_invariant && tag_candidate) {
                    ggml_mul_mat_set_hint(up_mm, GGML_HINT_MUL_MAT_ROW_INVARIANT_FFN_DOWN);
                }
                ggml_tensor * output_scale = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
                diagnostic_scale = output_scale;
                output = ggml_mul(ctx.get(), up_mm, output_scale);
                buffer.reset(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
                if (!buffer) fail("failed to allocate NVFP4 LM-head tensors");
                constexpr float scale = 0.6875f;
                ggml_backend_tensor_set(output_scale, &scale, 0, sizeof(scale));
            } else {
                output = up_mm;
                buffer.reset(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
                if (!buffer) fail("failed to allocate NVFP4 tensors");
            }
            const auto up_q = make_nvfp4_weight(k, n, 3);
            ggml_backend_tensor_set(up, up_q.data(), 0, up_q.size());
        }
        ggml_set_name(output, "row_invariance_output");
        if (!ggml_backend_supports_op(backend, output)) {
            fail("CUDA backend rejected diagnostic output op");
        }
        graph = ggml_new_graph_custom(ctx.get(), 32, false);
        ggml_build_forward_expand(graph, output);
        if (op == operation::nvfp4_head || op == operation::nvfp4_down_row_invariant) {
            const int32_t tagged_hint = op == operation::nvfp4_head ?
                    GGML_HINT_MUL_MAT_ROW_INVARIANT : GGML_HINT_MUL_MAT_ROW_INVARIANT_FFN_DOWN;
            const int32_t expected_hint = tag_candidate ? tagged_hint : GGML_HINT_NONE;
            if (!diagnostic_mm || diagnostic_mm->op != GGML_OP_MUL_MAT ||
                    diagnostic_mm->op_params[1] != expected_hint ||
                    !diagnostic_scale || diagnostic_scale->type != GGML_TYPE_F32 ||
                    !ggml_is_scalar(diagnostic_scale) || !ggml_is_contiguous(diagnostic_scale) ||
                    output->op != GGML_OP_MUL ||
                    !((output->src[0] == diagnostic_mm && output->src[1] == diagnostic_scale) ||
                      (output->src[1] == diagnostic_mm && output->src[0] == diagnostic_scale)) ||
                    ggml_graph_n_nodes(graph) != 2 ||
                    ggml_graph_node(graph, 0) != diagnostic_mm || ggml_graph_node(graph, 1) != output) {
                fail("NVFP4 LM-head diagnostic graph lost its hint or scalar-scale fusion shape");
            }
        }
        if (op == operation::nvfp4_ffn_row_invariant) {
            const int32_t expected_up = tag_candidate ?
                    GGML_HINT_MUL_MAT_ROW_INVARIANT_FFN_UP : GGML_HINT_NONE;
            const int32_t expected_gate = tag_candidate ?
                    GGML_HINT_MUL_MAT_ROW_INVARIANT_FFN_GATE : GGML_HINT_NONE;
            if (!diagnostic_mm || !diagnostic_gate_mm ||
                    diagnostic_mm->op_params[1] != expected_up ||
                    diagnostic_gate_mm->op_params[1] != expected_gate ||
                    output->op != GGML_OP_GLU || ggml_get_glu_op(output) != GGML_GLU_OP_SWIGLU) {
                fail("NVFP4 Qwen3.5 FFN graph lost its gate/up role hints or SwiGLU shape");
            }
        }
    }

    std::vector<float> run(ggml_backend_t backend, const std::vector<float> & values) {
        if (values.size() != (size_t) (k * m)) fail("dense input cardinality mismatch");
        ggml_backend_tensor_set(input, values.data(), 0, values.size() * sizeof(float));
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            fail("dense CUDA graph compute failed");
        }
        std::vector<float> result((size_t) ggml_nelements(output));
        ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));
        require_finite(result, "dense");
        return result;
    }
};

std::vector<float> run_scalar_rows(
        dense_runner & scalar, ggml_backend_t backend, const std::vector<float> & all, int64_t k, int64_t n) {
    if (all.size() % (size_t) k != 0) fail("scalar dense input is not a whole number of rows");
    const int64_t m = (int64_t) (all.size() / (size_t) k);
    std::vector<float> joined((size_t) n * m);
    std::vector<float> row((size_t) k);
    for (int64_t col = 0; col < m; ++col) {
        std::copy_n(all.data() + (size_t) col * k, k, row.data());
        const auto out = scalar.run(backend, row);
        if (out.size() != (size_t) n) fail("scalar dense output cardinality mismatch");
        std::copy(out.begin(), out.end(), joined.begin() + (size_t) col * n);
    }
    return joined;
}

void run_dense_case_with_runners(
        ggml_backend_t backend, const matrix_case & item,
        dense_runner & scalar, dense_runner & batch);

void run_head_candidate(
        ggml_backend_t backend, const matrix_case & item,
        route_reset_fn reset, route_snapshot_fn snapshot) {
    // Keep every graph alive at once. CUDA graphs are keyed by the first node
    // pointer, so destroying/recreating a context can reuse a key whose kernel
    // is now a replay. Host route markers intentionally count dispatch, not
    // replay, and must not be reset underneath such a cached M1 graph.
    dense_runner untagged(backend, item.op, item.k, item.n, 8, false);
    dense_runner scalar(backend, item.op, item.k, item.n, 1);
    std::vector<std::unique_ptr<dense_runner>> batches;
    batches.reserve(HEAD_MAX_M - HEAD_MIN_M + 1);
    for (int m = HEAD_MIN_M; m <= HEAD_MAX_M; ++m) {
        batches.emplace_back(std::make_unique<dense_runner>(backend, item.op, item.k, item.n, m));
    }

    reset();
    (void) untagged.run(backend, make_input(item.k, 8));
    route_counts negative{};
    if (!snapshot(&negative, sizeof(negative)) ||
            negative.abi_version != ROUTE_ABI || negative.route_count != ROUTE_COUNT) {
        fail("CUDA route-marker ABI rejected LM-head negative control");
    }
    if (negative.count[NVFP4_ROW_INVARIANT_HEAD_SCALED] != 0) {
        fail("untagged NVFP4 matmul entered the row-invariant LM-head route");
    }
    std::printf("{\"type\":\"route_negative_control\",\"id\":\"%.*s\",\"row_invariant_head\":0}\n",
            (int) item.id.size(), item.id.data());

    reset();
    for (int m = HEAD_MIN_M; m <= HEAD_MAX_M; ++m) {
        const auto input = make_input(item.k, m);
        const auto one = run_scalar_rows(scalar, backend, input, item.k, item.n);
        route_counts before{};
        if (!snapshot(&before, sizeof(before)) || before.abi_version != ROUTE_ABI || before.route_count != ROUTE_COUNT) {
            fail("CUDA route-marker ABI rejected LM-head pre-width snapshot");
        }
        const auto many = batches[(size_t) (m - HEAD_MIN_M)]->run(backend, input);
        require_bits_equal(one, many, item.id, m);
        route_counts after{};
        if (!snapshot(&after, sizeof(after)) || after.abi_version != ROUTE_ABI || after.route_count != ROUTE_COUNT) {
            fail("CUDA route-marker ABI rejected LM-head width coverage");
        }
        const head_route_observation observation{
            after.count[NVFP4_MMVQ_M1],
            before.count[NVFP4_ROW_INVARIANT_HEAD_SCALED],
            after.count[NVFP4_ROW_INVARIANT_HEAD_SCALED],
            after.count[NVFP4_MMVQ_M8],
        };
        std::printf("{\"type\":\"width_route\",\"id\":\"%.*s\",\"m\":%d,"
                    "\"observed\":{\"m1\":%llu,\"tagged_scaled_before\":%llu,"
                    "\"tagged_scaled_after\":%llu,\"tagged_scaled_delta\":%llu,\"generic_m8\":%llu},"
                    "\"expected\":{\"m1_min\":1,\"tagged_scaled_delta_min\":1,\"generic_m8\":0}}\n",
                (int) item.id.size(), item.id.data(), m,
                (unsigned long long) observation.m1,
                (unsigned long long) observation.scaled_before,
                (unsigned long long) observation.scaled_after,
                (unsigned long long) (observation.scaled_after - observation.scaled_before),
                (unsigned long long) observation.generic_m8);
        std::fflush(stdout);
        if (!head_route_observation_valid(observation)) {
            char detail[320];
            std::snprintf(detail, sizeof(detail),
                    "LM-head width route mismatch m=%d observed_m1=%llu expected_m1_min=1 "
                    "observed_tagged_scaled_before=%llu observed_tagged_scaled_after=%llu expected_delta_min=1 "
                    "observed_generic_m8=%llu expected_generic_m8=0",
                    m, (unsigned long long) observation.m1,
                    (unsigned long long) observation.scaled_before,
                    (unsigned long long) observation.scaled_after,
                    (unsigned long long) observation.generic_m8);
            fail(detail);
        }
    }
    std::printf("{\"type\":\"width_coverage\",\"id\":\"%.*s\",\"status\":\"passed\",\"min_m\":2,\"max_m\":16,\"comparison\":\"bitwise_f32\",\"route_checked_each_width\":true}\n",
            (int) item.id.size(), item.id.data());

    // The same live graph owners must also serve the intensive replay case.
    // Destroying them here and allocating new M1/M8 graphs could recycle a
    // first-node CUDA-graph key and turn the first post-reset execution into a
    // replay with no new host route-marker hit.
    run_dense_case_with_runners(
            backend, item, scalar, *batches[(size_t) (8 - HEAD_MIN_M)]);
}

void run_projection_candidate(
        ggml_backend_t backend, const matrix_case & item,
        route_reset_fn reset, route_snapshot_fn snapshot) {
    const route selected_route = item.op == operation::nvfp4_ffn_row_invariant ?
            NVFP4_ROW_INVARIANT_FFN_FUSED : NVFP4_ROW_INVARIANT_FFN_DOWN_SCALED;
    const route scalar_route = item.op == operation::nvfp4_ffn_row_invariant ?
            NVFP4_FUSED_FFN : NVFP4_MMVQ_M1;
    dense_runner untagged(backend, item.op, item.k, item.n, 8, false);
    dense_runner scalar(backend, item.op, item.k, item.n, 1);
    std::vector<std::unique_ptr<dense_runner>> batches;
    batches.reserve(HEAD_MAX_M - HEAD_MIN_M + 1);
    for (int m = HEAD_MIN_M; m <= HEAD_MAX_M; ++m) {
        batches.emplace_back(std::make_unique<dense_runner>(backend, item.op, item.k, item.n, m));
    }

    reset();
    (void) untagged.run(backend, make_input(item.k, 8));
    route_counts negative{};
    if (!snapshot(&negative, sizeof(negative)) ||
            negative.abi_version != ROUTE_ABI || negative.route_count != ROUTE_COUNT ||
            negative.count[selected_route] != 0) {
        fail("CUDA route-marker ABI or Qwen3.5 projection negative control failed");
    }
    std::printf("{\"type\":\"route_negative_control\",\"id\":\"%.*s\",\"selected_route\":0}\n",
            (int) item.id.size(), item.id.data());

    reset();
    for (int m = HEAD_MIN_M; m <= HEAD_MAX_M; ++m) {
        const auto input = make_input(item.k, m);
        const auto one = run_scalar_rows(scalar, backend, input, item.k, item.n);
        route_counts before{};
        if (!snapshot(&before, sizeof(before)) || before.abi_version != ROUTE_ABI || before.route_count != ROUTE_COUNT) {
            fail("CUDA route-marker ABI rejected Qwen3.5 projection pre-width snapshot");
        }
        const auto many = batches[(size_t) (m - HEAD_MIN_M)]->run(backend, input);
        require_bits_equal(one, many, item.id, m);
        route_counts after{};
        if (!snapshot(&after, sizeof(after)) || after.abi_version != ROUTE_ABI || after.route_count != ROUTE_COUNT ||
                after.count[scalar_route] == 0 || after.count[selected_route] <= before.count[selected_route] ||
                after.count[NVFP4_MMVQ_M8] != 0) {
            fail("Qwen3.5 projection width route coverage failed");
        }
        std::printf("{\"type\":\"width_route\",\"id\":\"%.*s\",\"m\":%d,"
                    "\"scalar_route\":%llu,\"selected_before\":%llu,\"selected_after\":%llu,"
                    "\"selected_delta\":%llu,\"generic_m8\":%llu}\n",
                (int) item.id.size(), item.id.data(), m,
                (unsigned long long) after.count[scalar_route],
                (unsigned long long) before.count[selected_route],
                (unsigned long long) after.count[selected_route],
                (unsigned long long) (after.count[selected_route] - before.count[selected_route]),
                (unsigned long long) after.count[NVFP4_MMVQ_M8]);
    }
    std::printf("{\"type\":\"width_coverage\",\"id\":\"%.*s\",\"status\":\"passed\",\"min_m\":2,\"max_m\":16,\"comparison\":\"bitwise_f32\",\"route_checked_each_width\":true}\n",
            (int) item.id.size(), item.id.data());
    run_dense_case_with_runners(
            backend, item, scalar, *batches[(size_t) (8 - HEAD_MIN_M)]);
}

void run_dense_case_with_runners(
        ggml_backend_t backend, const matrix_case & item,
        dense_runner & scalar, dense_runner & batch) {
    const auto distinct = make_input(item.k, 8);
    std::vector<std::vector<float>> fixtures = { distinct };
    if (item.op == operation::nvfp4_head ||
            item.op == operation::nvfp4_ffn_row_invariant ||
            item.op == operation::nvfp4_down_row_invariant) {
        std::vector<float> duplicate(distinct.size());
        std::vector<float> permuted(distinct.size());
        constexpr std::array<int, 8> permutation = { 7, 0, 5, 2, 6, 1, 4, 3 };
        for (int col = 0; col < 8; ++col) {
            std::copy_n(distinct.data() + (size_t) (col % 2) * item.k, item.k,
                    duplicate.data() + (size_t) col * item.k);
            std::copy_n(distinct.data() + (size_t) permutation[(size_t) col] * item.k, item.k,
                    permuted.data() + (size_t) col * item.k);
        }
        fixtures.push_back(std::move(duplicate));
        fixtures.push_back(std::move(permuted));
    }

    std::vector<float> final_output;
    for (size_t fixture = 0; fixture < fixtures.size(); ++fixture) {
        std::vector<float> stable_scalar, stable_batch;
        for (int replay = 0; replay < REPLAYS; ++replay) {
            std::vector<float> one, eight;
            if ((replay & 1) == 0) {
                one = run_scalar_rows(scalar, backend, fixtures[fixture], item.k, item.n);
                eight = batch.run(backend, fixtures[fixture]);
            } else {
                eight = batch.run(backend, fixtures[fixture]);
                one = run_scalar_rows(scalar, backend, fixtures[fixture], item.k, item.n);
            }
            require_bits_equal(one, eight, item.id, replay);
            if (replay == 0) {
                stable_scalar = one;
                stable_batch = eight;
            } else {
                require_bits_equal(stable_scalar, one, item.id, replay);
                require_bits_equal(stable_batch, eight, item.id, replay);
            }
        }
        final_output.insert(final_output.end(), stable_batch.begin(), stable_batch.end());
    }
    std::printf("{\"type\":\"case\",\"id\":\"%.*s\",\"status\":\"passed\",\"replays\":8,\"order\":\"alternating\",\"fixtures\":%zu,\"output_scale\":%s,\"outputs\":%zu,\"fnv1a64\":\"%016llx\"}\n",
            (int) item.id.size(), item.id.data(), fixtures.size(),
            (item.op == operation::nvfp4_head || item.op == operation::nvfp4_ffn_row_invariant ||
             item.op == operation::nvfp4_down_row_invariant) ? "true" : "false", final_output.size(),
            (unsigned long long) fnv1a(final_output));
}

void run_dense_case(ggml_backend_t backend, const matrix_case & item) {
    dense_runner scalar(backend, item.op, item.k, item.n, 1);
    dense_runner batch(backend, item.op, item.k, item.n, 8);
    run_dense_case_with_runners(backend, item, scalar, batch);
}

void run_fp8_case(
        ggml_backend_t backend, const matrix_case & item,
        route_reset_fn reset, route_snapshot_fn snapshot) {
    constexpr int min_batch_m = 2;
    constexpr int max_batch_m = 8;
    dense_runner scalar(backend, item.op, item.k, item.n, 1);
    std::vector<std::unique_ptr<dense_runner>> batches;
    batches.reserve(max_batch_m - min_batch_m + 1);
    for (int m = min_batch_m; m <= max_batch_m; ++m) {
        batches.emplace_back(std::make_unique<dense_runner>(backend, item.op, item.k, item.n, m));
    }

    // Keep every width-specific graph alive. This prevents CUDA graph-key
    // address reuse from replaying a graph captured for a different M.
    reset();
    for (int m = min_batch_m; m <= max_batch_m; ++m) {
        const auto input = make_input(item.k, m);
        route_counts before{};
        if (!snapshot(&before, sizeof(before)) ||
                before.abi_version != ROUTE_ABI || before.route_count != ROUTE_COUNT) {
            fail("CUDA route-marker ABI rejected FP8 pre-width snapshot");
        }

        std::vector<float> stable_scalar, stable_batch;
        for (int replay = 0; replay < REPLAYS; ++replay) {
            std::vector<float> one, many;
            if ((replay & 1) == 0) {
                one = run_scalar_rows(scalar, backend, input, item.k, item.n);
                many = batches[(size_t) (m - min_batch_m)]->run(backend, input);
            } else {
                many = batches[(size_t) (m - min_batch_m)]->run(backend, input);
                one = run_scalar_rows(scalar, backend, input, item.k, item.n);
            }
            require_finite(one, item.id);
            require_finite(many, item.id);
            require_bits_equal(one, many, item.id, replay);
            if (replay == 0) {
                stable_scalar = one;
                stable_batch = many;
            } else {
                require_bits_equal(stable_scalar, one, item.id, replay);
                require_bits_equal(stable_batch, many, item.id, replay);
            }
        }

        route_counts after{};
        if (!snapshot(&after, sizeof(after)) ||
                after.abi_version != ROUTE_ABI || after.route_count != ROUTE_COUNT ||
                after.count[FP8_BATCH] <= before.count[FP8_BATCH]) {
            fail("FP8 width did not execute the mandatory native E4M3 route");
        }
        std::printf(
                "{\"type\":\"width_route\",\"id\":\"%.*s\",\"m\":%d,"
                "\"fp8_batch_before\":%llu,\"fp8_batch_after\":%llu,\"fp8_batch_delta\":%llu,"
                "\"finite\":true,\"comparison\":\"bitwise_f32\",\"replays\":8}\n",
                (int) item.id.size(), item.id.data(), m,
                (unsigned long long) before.count[FP8_BATCH],
                (unsigned long long) after.count[FP8_BATCH],
                (unsigned long long) (after.count[FP8_BATCH] - before.count[FP8_BATCH]));
    }
    std::printf(
            "{\"type\":\"width_coverage\",\"id\":\"%.*s\",\"status\":\"passed\","
            "\"min_m\":2,\"max_m\":8,\"widths\":7,\"replays\":8,"
            "\"finite\":true,\"comparison\":\"bitwise_f32\",\"route_checked_each_width\":true}\n",
            (int) item.id.size(), item.id.data());
    std::printf(
            "{\"type\":\"case\",\"id\":\"%.*s\",\"status\":\"passed\","
            "\"replays\":8,\"min_batch_m\":2,\"max_batch_m\":8,\"widths\":7,"
            "\"finite\":true,\"comparison\":\"bitwise_f32\"}\n",
            (int) item.id.size(), item.id.data());
}

std::vector<float> make_gdn_values(size_t count, int salt, float scale = 1.0f) {
    std::vector<float> result(count);
    for (size_t i = 0; i < count; ++i) {
        const float sign = ((i + (size_t) salt) & 1) ? -1.0f : 1.0f;
        result[i] = sign * scale * (0.03125f + 0.015625f * ((i * 7 + salt) % 11));
    }
    return result;
}

struct gdn_runner {
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * q = nullptr, * k = nullptr, * v = nullptr, * g = nullptr, * beta = nullptr, * state = nullptr;
    ggml_tensor * output = nullptr, * cache = nullptr, * graph_root = nullptr;
    int64_t s, h, tokens, snapshots, state_size, slot_stride;

    gdn_runner(ggml_backend_t backend, int64_t s_, int64_t h_, int64_t tokens_, int64_t snapshots_)
        : s(s_), h(h_), tokens(tokens_), snapshots(snapshots_), state_size(s * s * h), slot_stride(state_size + 64) {
        ggml_init_params params{ ggml_tensor_overhead() * 28 + ggml_graph_overhead_custom(32, false), nullptr, true };
        ctx.reset(ggml_init(params));
        if (!ctx) fail("failed to create GDN context");
        q = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, s, h, tokens);
        k = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, s, h, tokens);
        v = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, s, h, tokens);
        g = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 1, h, tokens);
        beta = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 1, h, tokens);
        state = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, s, s, h);
        output = ggml_gated_delta_net(ctx.get(), ggml_l2_norm(ctx.get(), q, 1e-6f),
                ggml_l2_norm(ctx.get(), k, 1e-6f), v, g, beta, state, snapshots);
        ggml_set_name(output, "gdn_row_invariance_output");

        // Match the production graph exactly: view the contiguous snapshot
        // tail, then scatter it into rollback-group cache slots with a padded
        // stride. The CUDA graph optimizer must replace this VIEW -> CPY with
        // the fused-cache GDN entry point.
        const int64_t n_written = std::min(tokens, snapshots);
        const size_t tail_offset = (size_t) s * h * tokens * sizeof(float);
        ggml_tensor * tail = ggml_view_3d(ctx.get(), output, state_size, 1, n_written,
                (size_t) state_size * sizeof(float), (size_t) state_size * sizeof(float), tail_offset);
        cache = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32,
                slot_stride * (snapshots - 1) + state_size);
        ggml_tensor * cache_view = ggml_view_3d(ctx.get(), cache, state_size, 1, n_written,
                (size_t) state_size * sizeof(float), (size_t) slot_stride * sizeof(float), 0);
        graph_root = ggml_cpy(ctx.get(), tail, cache_view);
        ggml_set_name(graph_root, "gdn_snapshot_cache_scatter");
        buffer.reset(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
        if (!buffer) fail("failed to allocate GDN tensors");
        if (!ggml_backend_supports_op(backend, output) || !ggml_backend_supports_op(backend, graph_root)) {
            fail("CUDA backend rejected GDN fused-cache diagnostic");
        }
        graph = ggml_new_graph_custom(ctx.get(), 32, false);
        ggml_build_forward_expand(graph, graph_root);
    }

    std::vector<float> run(ggml_backend_t backend,
            const float * qv, const float * kv, const float * vv,
            const float * gv, const float * bv, const std::vector<float> & sv) {
        const size_t vectors = (size_t) s * h * tokens;
        const size_t gates = (size_t) h * tokens;
        if (sv.size() != (size_t) s * s * h) fail("GDN state cardinality mismatch");
        ggml_backend_tensor_set(q, qv, 0, vectors * sizeof(float));
        ggml_backend_tensor_set(k, kv, 0, vectors * sizeof(float));
        ggml_backend_tensor_set(v, vv, 0, vectors * sizeof(float));
        ggml_backend_tensor_set(g, gv, 0, gates * sizeof(float));
        ggml_backend_tensor_set(beta, bv, 0, gates * sizeof(float));
        ggml_backend_tensor_set(state, sv.data(), 0, sv.size() * sizeof(float));
        std::vector<float> cache_zero((size_t) ggml_nelements(cache), 0.0f);
        ggml_backend_tensor_set(cache, cache_zero.data(), 0, cache_zero.size() * sizeof(float));
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) fail("GDN CUDA graph compute failed");
        const size_t attention_size = (size_t) s * h * tokens;
        std::vector<float> result(attention_size + (size_t) snapshots * state_size);
        ggml_backend_tensor_get(output, result.data(), 0, attention_size * sizeof(float));
        std::vector<float> cache_host((size_t) ggml_nelements(cache));
        ggml_backend_tensor_get(cache, cache_host.data(), 0, cache_host.size() * sizeof(float));
        for (int64_t slot = 0; slot < snapshots; ++slot) {
            std::copy_n(cache_host.data() + slot * slot_stride, state_size,
                    result.data() + attention_size + slot * state_size);
        }
        require_finite(result, "gdn");
        return result;
    }
};

struct gdn_fixture {
    std::vector<float> q, k, v, g, beta, state;
};

gdn_fixture make_gdn_fixture(int64_t s, int64_t h) {
    gdn_fixture f;
    f.q = make_gdn_values((size_t) s * h * 8, 1);
    f.k = make_gdn_values((size_t) s * h * 8, 4);
    f.v = make_gdn_values((size_t) s * h * 8, 7, 0.5f);
    f.g = make_gdn_values((size_t) h * 8, 2, 0.25f);
    for (float & value : f.g) value -= 2.0f;
    f.beta = make_gdn_values((size_t) h * 8, 9, 0.125f);
    for (float & value : f.beta) value = std::fabs(value);
    f.state = make_gdn_values((size_t) s * s * h, 5, 0.0625f);
    return f;
}

std::vector<float> run_gdn_sequential(
        gdn_runner & scalar, ggml_backend_t backend, const gdn_fixture & f,
        int64_t s, int64_t h, std::vector<std::vector<float>> & states) {
    const size_t vector_stride = (size_t) s * h;
    const size_t state_size = (size_t) s * s * h;
    std::vector<float> current = f.state;
    std::vector<float> attention(vector_stride * 8);
    states.clear();
    for (int token = 0; token < 8; ++token) {
        const auto out = scalar.run(backend,
                f.q.data() + token * vector_stride, f.k.data() + token * vector_stride,
                f.v.data() + token * vector_stride, f.g.data() + token * h,
                f.beta.data() + token * h, current);
        if (out.size() != vector_stride + state_size) fail("GDN scalar output layout drift");
        std::copy_n(out.data(), vector_stride, attention.data() + token * vector_stride);
        current.assign(out.begin() + vector_stride, out.end());
        states.push_back(current);
    }
    return attention;
}

void run_gdn_slot_case(ggml_backend_t backend, const matrix_case & item) {
    const int64_t s = item.k, h = item.n;
    gdn_runner k1(backend, s, h, 1, 1), k8(backend, s, h, 1, 8);
    const auto f = make_gdn_fixture(s, h);
    const size_t vec = (size_t) s * h, state_size = (size_t) s * s * h;
    std::vector<float> stable;
    for (int replay = 0; replay < REPLAYS; ++replay) {
        std::vector<float> a, b;
        auto run1 = [&] { return k1.run(backend, f.q.data(), f.k.data(), f.v.data(), f.g.data(), f.beta.data(), f.state); };
        auto run8 = [&] { return k8.run(backend, f.q.data(), f.k.data(), f.v.data(), f.g.data(), f.beta.data(), f.state); };
        if ((replay & 1) == 0) { a = run1(); b = run8(); } else { b = run8(); a = run1(); }
        std::vector<float> a_cmp(a.begin(), a.begin() + vec + state_size);
        std::vector<float> b_cmp(b.begin(), b.begin() + vec + state_size);
        require_bits_equal(a_cmp, b_cmp, item.id, replay);
        if (replay == 0) stable = b_cmp; else require_bits_equal(stable, b_cmp, item.id, replay);
    }
    std::printf("{\"type\":\"case\",\"id\":\"%.*s\",\"status\":\"passed\",\"replays\":8,\"slot\":0,\"finite\":true,\"comparison\":\"bitwise_f32\",\"fnv1a64\":\"%016llx\"}\n",
            (int) item.id.size(), item.id.data(), (unsigned long long) fnv1a(stable));
}

void run_gdn_sequence_case(ggml_backend_t backend, const matrix_case & item) {
    const int64_t s = item.k, h = item.n;
    gdn_runner scalar(backend, s, h, 1, 1), batch(backend, s, h, 8, 8);
    const auto f = make_gdn_fixture(s, h);
    const size_t attention_size = (size_t) s * h * 8, state_size = (size_t) s * s * h;
    std::vector<float> stable;
    for (int replay = 0; replay < REPLAYS; ++replay) {
        std::vector<float> seq_attention, batch_out;
        std::vector<std::vector<float>> states;
        auto run_seq = [&] { return run_gdn_sequential(scalar, backend, f, s, h, states); };
        auto run_batch = [&] { return batch.run(backend, f.q.data(), f.k.data(), f.v.data(), f.g.data(), f.beta.data(), f.state); };
        if ((replay & 1) == 0) { seq_attention = run_seq(); batch_out = run_batch(); }
        else { batch_out = run_batch(); seq_attention = run_seq(); }
        std::vector<float> batch_attention(batch_out.begin(), batch_out.begin() + attention_size);
        require_bits_equal(seq_attention, batch_attention, item.id, replay);
        for (int slot = 0; slot < 8; ++slot) {
            std::vector<float> got(batch_out.begin() + attention_size + slot * state_size,
                                   batch_out.begin() + attention_size + (slot + 1) * state_size);
            require_bits_equal(states[(size_t) (7 - slot)], got, item.id, replay);
        }
        if (replay == 0) stable = batch_out; else require_bits_equal(stable, batch_out, item.id, replay);
    }
    std::printf("{\"type\":\"case\",\"id\":\"%.*s\",\"status\":\"passed\",\"replays\":8,\"snapshot_order\":\"newest_first\",\"finite\":true,\"comparison\":\"bitwise_f32\",\"fnv1a64\":\"%016llx\"}\n",
            (int) item.id.size(), item.id.data(), (unsigned long long) fnv1a(stable));
}

std::vector<float> make_operator_values(size_t per_token, int tokens, int salt) {
    std::vector<float> result(per_token * (size_t) tokens);
    for (int token = 0; token < tokens; ++token) {
        for (size_t i = 0; i < per_token; ++i) {
            const float sign = ((i * 5 + (size_t) token * 3 + (size_t) salt) & 1) ? -1.0f : 1.0f;
            const float scale = ((i + (size_t) token * 11 + (size_t) salt) % 17 == 0) ? 4.0f : 0.25f;
            result[(size_t) token * per_token + i] =
                    sign * scale * (0.03125f + 0.0078125f * (float) ((i * 7 + token + salt) % 23));
        }
    }
    return result;
}

struct bf16_runner {
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * weight = nullptr;
    ggml_tensor * input_storage = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * transform_bias = nullptr;
    ggml_tensor * transform_scale = nullptr;
    ggml_tensor * projection_output = nullptr;
    ggml_tensor * output = nullptr;
    int64_t k;
    int64_t n;
    int64_t m;

    bool alpha_transform = false;
    bool beta_transform = false;
    bool strided_input = false;

    bf16_runner(ggml_backend_t backend, int64_t k_, int64_t n_, int64_t m_,
            ggml_op_hint hint = GGML_HINT_NONE, ggml_op_hint transform_role = GGML_HINT_NONE,
            bool strided_input_ = false)
        : k(k_), n(n_), m(m_), strided_input(strided_input_) {
        ggml_init_params params{
            ggml_tensor_overhead() * 24 + ggml_graph_overhead_custom(32, false), nullptr, true,
        };
        ctx.reset(ggml_init(params));
        if (!ctx) fail("failed to create BF16 projection diagnostic context");
        weight = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_BF16, k, n);
        if (strided_input) {
            input_storage = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, k + 1, m);
            input = ggml_view_2d(ctx.get(), input_storage, k, m, (size_t) (k + 1) * sizeof(float), 0);
        } else {
            input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, k, m);
            input_storage = input;
        }
        projection_output = ggml_mul_mat(ctx.get(), weight, input);
        output = projection_output;
        if (hint != GGML_HINT_NONE) {
            ggml_mul_mat_set_hint(output, hint);
        }
        alpha_transform = transform_role == GGML_HINT_MUL_MAT_ROW_INVARIANT_BF16_ALPHA;
        beta_transform = transform_role == GGML_HINT_MUL_MAT_ROW_INVARIANT_BF16_BETA;
        if (beta_transform) {
            // Match qwen35 exactly: beta is reshaped before the sigmoid callback.
            output = ggml_reshape_4d(ctx.get(), output, 1, n, m, 1);
            output = ggml_sigmoid(ctx.get(), output);
        } else if (alpha_transform) {
            transform_bias = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n);
            transform_scale = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n);
            // Preserve the production reshape boundary so the M=1 graph cannot
            // become an artificial adjacent MUL_MAT+ADD fusion.
            output = ggml_reshape_3d(ctx.get(), output, n, m, 1);
            output = ggml_mul(ctx.get(), ggml_softplus(ctx.get(), ggml_add(ctx.get(), output, transform_bias)), transform_scale);
            output = ggml_reshape_4d(ctx.get(), output, 1, n, m, 1);
        }
        ggml_set_name(output, "qwen35_bf16_projection_output");
        buffer.reset(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
        if (!buffer || !ggml_backend_supports_op(backend, output)) {
            fail("CUDA backend rejected BF16 projection diagnostic graph");
        }
        graph = ggml_new_graph_custom(ctx.get(), 32, false);
        ggml_build_forward_expand(graph, output);
    }

    std::vector<float> run(
            ggml_backend_t backend, const std::vector<ggml_bf16_t> & weights,
            const std::vector<float> & values) {
        prepare(weights, values);
        compute(backend);
        return read_output();
    }

    void prepare(const std::vector<ggml_bf16_t> & weights, const std::vector<float> & values) {
        if (weights.size() != (size_t) k * n || values.size() != (size_t) k * m) {
            fail("BF16 projection fixture cardinality mismatch");
        }
        ggml_backend_tensor_set(weight, weights.data(), 0, weights.size() * sizeof(ggml_bf16_t));
        if (strided_input) {
            std::vector<float> padded((size_t) (k + 1) * m, 0.0f);
            for (int64_t token = 0; token < m; ++token) {
                std::copy(values.begin() + (size_t) token * k,
                        values.begin() + (size_t) (token + 1) * k,
                        padded.begin() + (size_t) token * (k + 1));
            }
            ggml_backend_tensor_set(input_storage, padded.data(), 0, padded.size() * sizeof(float));
        } else {
            ggml_backend_tensor_set(input, values.data(), 0, values.size() * sizeof(float));
        }
        if (alpha_transform) {
            std::vector<float> bias((size_t) n), scale((size_t) n);
            for (int64_t i = 0; i < n; ++i) {
                bias[(size_t) i] = -0.125f + 0.0078125f * (float) (i % 17);
                scale[(size_t) i] = -(0.5f + 0.015625f * (float) (i % 11));
            }
            ggml_backend_tensor_set(transform_bias, bias.data(), 0, bias.size() * sizeof(float));
            ggml_backend_tensor_set(transform_scale, scale.data(), 0, scale.size() * sizeof(float));
        }
    }

    void compute(ggml_backend_t backend) {
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            fail("BF16 projection CUDA graph compute failed");
        }
    }

    std::vector<float> read_output() {
        std::vector<float> result((size_t) n * m);
        ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));
        require_finite(result, "bf16_projection");
        return result;
    }

    std::vector<float> read_projection_output() {
        std::vector<float> result((size_t) n * m);
        ggml_backend_tensor_get(projection_output, result.data(), 0, result.size() * sizeof(float));
        require_finite(result, "bf16_projection_raw");
        return result;
    }
};

struct bf16_observation {
    std::vector<float> raw_projection;
    std::vector<float> transformed;
};

bf16_observation run_bf16_observed(
        bf16_runner & runner, ggml_backend_t backend,
        const std::vector<ggml_bf16_t> & weights,
        const std::vector<float> & values) {
    runner.prepare(weights, values);
    runner.compute(backend);
    return { runner.read_projection_output(), runner.read_output() };
}

bf16_observation run_bf16_scalar_rows_observed(
        bf16_runner & scalar, ggml_backend_t backend,
        const std::vector<ggml_bf16_t> & weights,
        const std::vector<float> & values, int64_t k, int64_t n, int m) {
    bf16_observation result;
    result.raw_projection.resize((size_t) n * m);
    result.transformed.resize((size_t) n * m);
    for (int token = 0; token < m; ++token) {
        const std::vector<float> one(values.begin() + (size_t) token * k,
                                     values.begin() + (size_t) (token + 1) * k);
        const auto observation = run_bf16_observed(scalar, backend, weights, one);
        std::copy(observation.raw_projection.begin(), observation.raw_projection.end(),
                result.raw_projection.begin() + (size_t) token * n);
        std::copy(observation.transformed.begin(), observation.transformed.end(),
                result.transformed.begin() + (size_t) token * n);
    }
    return result;
}

std::vector<ggml_bf16_t> make_bf16_weights(int64_t k, int64_t n, int salt) {
    std::vector<float> source((size_t) k * n);
    for (int64_t row = 0; row < n; ++row) {
        for (int64_t col = 0; col < k; ++col) {
            const float sign = ((col * 7 + row * 3 + salt) & 1) ? -1.0f : 1.0f;
            source[(size_t) row * k + col] =
                    sign * (0.015625f + 0.00390625f * (float) ((col * 5 + row * 11 + salt) % 29));
        }
    }
    std::vector<ggml_bf16_t> result(source.size());
    ggml_fp32_to_bf16_row_ref(source.data(), result.data(), (int64_t) source.size());
    return result;
}

std::vector<float> make_bf16_projection_input(int64_t k, int m, int salt) {
    // Deliberately stay off the BF16 grid.  The production M=1 MMVF route
    // consumes these F32 values directly, while the M>1 cuBLAS route converts
    // the activation to BF16 first.  A BF16-exact fixture would erase the
    // route-dependent numeric behavior this diagnostic is meant to locate.
    std::vector<float> result((size_t) k * m);
    size_t changed_by_roundtrip = 0;
    for (int token = 0; token < m; ++token) {
        for (int64_t col = 0; col < k; ++col) {
            const float sign = ((col * 11 + token * 7 + salt) & 1) ? -1.0f : 1.0f;
            const float magnitude = 0.013271f +
                    0.000137f * (float) ((col * 17 + token * 29 + salt) % 251);
            const float value = sign * magnitude;
            result[(size_t) token * k + col] = value;
            if (ggml_bf16_to_fp32(ggml_fp32_to_bf16(value)) != value) {
                ++changed_by_roundtrip;
            }
        }
    }
    if (changed_by_roundtrip * 4 < result.size() * 3) {
        fail("BF16 projection input fixture is insufficiently sensitive to F32-to-BF16 conversion");
    }
    return result;
}

std::vector<float> run_bf16_scalar_rows(
        bf16_runner & scalar, ggml_backend_t backend,
        const std::vector<ggml_bf16_t> & weights,
        const std::vector<float> & values, int64_t k, int64_t n, int m) {
    std::vector<float> result((size_t) n * m);
    for (int token = 0; token < m; ++token) {
        const std::vector<float> one(values.begin() + (size_t) token * k,
                                     values.begin() + (size_t) (token + 1) * k);
        const auto output = scalar.run(backend, weights, one);
        std::copy(output.begin(), output.end(), result.begin() + (size_t) token * n);
    }
    return result;
}

void run_bf16_case(
        ggml_backend_t backend, const matrix_case & item,
        route_snapshot_fn snapshot) {
    const int salt = item.id.find("alpha") != std::string_view::npos ? 17 : 5;
    const auto weights = make_bf16_weights(item.k, item.n, salt);
    std::vector<std::unique_ptr<bf16_runner>> scalars;
    std::vector<std::unique_ptr<bf16_runner>> batches;
    for (int m = 2; m <= 8; ++m) {
        scalars.emplace_back(std::make_unique<bf16_runner>(backend, item.k, item.n, 1));
        batches.emplace_back(std::make_unique<bf16_runner>(backend, item.k, item.n, m));
    }
    for (int m = 2; m <= 8; ++m) {
        const auto values = make_bf16_projection_input(item.k, m, salt);
        route_counts before{};
        if (!snapshot(&before, sizeof(before)) || before.abi_version != ROUTE_ABI || before.route_count != ROUTE_COUNT) {
            fail("CUDA route-marker ABI rejected BF16 projection pre-width snapshot");
        }
        std::vector<float> stable_scalar, stable_batch;
        for (int replay = 0; replay < REPLAYS; ++replay) {
            std::vector<float> one, many;
            auto run_one = [&] { return run_bf16_scalar_rows(
                    *scalars[(size_t) (m - 2)], backend, weights, values, item.k, item.n, m); };
            auto run_many = [&] { return batches[(size_t) (m - 2)]->run(backend, weights, values); };
            if (replay == 0) {
                one = run_one();
                route_counts after_scalar{};
                if (!snapshot(&after_scalar, sizeof(after_scalar)) ||
                        after_scalar.abi_version != ROUTE_ABI || after_scalar.route_count != ROUTE_COUNT ||
                        after_scalar.count[BF16_MMVF] <= before.count[BF16_MMVF] ||
                        after_scalar.count[BF16_CUBLAS] != before.count[BF16_CUBLAS]) {
                    fail("BF16 scalar projection did not exclusively select MMVF");
                }
                many = run_many();
                route_counts after_batch{};
                if (!snapshot(&after_batch, sizeof(after_batch)) ||
                        after_batch.abi_version != ROUTE_ABI || after_batch.route_count != ROUTE_COUNT ||
                        after_batch.count[BF16_CUBLAS] <= after_scalar.count[BF16_CUBLAS] ||
                        after_batch.count[BF16_MMVF] != after_scalar.count[BF16_MMVF]) {
                    fail("BF16 batched projection did not exclusively select cuBLAS");
                }
                std::printf(
                        "{\"type\":\"bf16_route_probe\",\"id\":\"%.*s\",\"m\":%d,"
                        "\"scalar_mmvf_delta\":%llu,\"scalar_cublas_delta\":0,"
                        "\"batch_mmvf_delta\":0,\"batch_cublas_delta\":%llu,"
                        "\"input_bf16_roundtrip_sensitive\":true}\n",
                        (int) item.id.size(), item.id.data(), m,
                        (unsigned long long) (after_scalar.count[BF16_MMVF] - before.count[BF16_MMVF]),
                        (unsigned long long) (after_batch.count[BF16_CUBLAS] - after_scalar.count[BF16_CUBLAS]));
                std::fflush(stdout);
            } else if ((replay & 1) == 0) { one = run_one(); many = run_many(); }
            else { many = run_many(); one = run_one(); }
            require_bits_equal(one, many, item.id, replay);
            if (replay == 0) { stable_scalar = one; stable_batch = many; }
            else {
                require_bits_equal(stable_scalar, one, item.id, replay);
                require_bits_equal(stable_batch, many, item.id, replay);
            }
        }
        route_counts after{};
        if (!snapshot(&after, sizeof(after)) || after.abi_version != ROUTE_ABI || after.route_count != ROUTE_COUNT ||
                after.count[BF16_CUBLAS] <= before.count[BF16_CUBLAS] || after.count[BF16_MMVF] == 0) {
            fail("BF16 projection did not prove scalar MMVF and batched cuBLAS routes");
        }
        std::printf(
                "{\"type\":\"width_route\",\"id\":\"%.*s\",\"m\":%d,"
                "\"bf16_mmvf\":%llu,\"bf16_cublas_before\":%llu,"
                "\"bf16_cublas_after\":%llu,\"bf16_cublas_delta\":%llu,"
                "\"finite\":true,\"comparison\":\"bitwise_f32\",\"replays\":8}\n",
                (int) item.id.size(), item.id.data(), m,
                (unsigned long long) after.count[BF16_MMVF],
                (unsigned long long) before.count[BF16_CUBLAS],
                (unsigned long long) after.count[BF16_CUBLAS],
                (unsigned long long) (after.count[BF16_CUBLAS] - before.count[BF16_CUBLAS]));
    }
    std::printf(
            "{\"type\":\"width_coverage\",\"id\":\"%.*s\",\"status\":\"passed\","
            "\"min_m\":2,\"max_m\":8,\"widths\":7,\"replays\":8,"
            "\"scalar_route\":\"bf16_mmvf\",\"batch_route\":\"bf16_cublas\","
            "\"finite\":true,\"comparison\":\"bitwise_f32\",\"route_checked_each_width\":true}\n",
            (int) item.id.size(), item.id.data());
    std::printf(
             "{\"type\":\"case\",\"id\":\"%.*s\",\"status\":\"passed\","
             "\"semantic\":\"%s\",\"replays\":8,\"min_batch_m\":2,\"max_batch_m\":8,\"widths\":7,"
            "\"finite\":true,\"comparison\":\"bitwise_f32\"}\n",
            (int) item.id.size(), item.id.data(),
            item.id.find("alpha") != std::string_view::npos ? "alpha" : "beta");
}

double percentile_us(std::vector<double> values, double quantile) {
    if (values.empty()) fail("microbenchmark sample set is empty");
    std::sort(values.begin(), values.end());
    const size_t index = (size_t) std::ceil(quantile * (double) values.size()) - 1;
    return values[std::min(index, values.size() - 1)];
}

double timed_bf16_compute_us(bf16_runner & runner, ggml_backend_t backend) {
    ggml_backend_synchronize(backend);
    const auto begin = std::chrono::steady_clock::now();
    runner.compute(backend);
    ggml_backend_synchronize(backend);
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - begin).count();
}

void print_samples(const std::vector<double> & samples) {
    std::printf("[");
    for (size_t i = 0; i < samples.size(); ++i) {
        std::printf("%s%.3f", i ? "," : "", samples[i]);
    }
    std::printf("]");
}

void run_bf16_candidate_case(
        ggml_backend_t backend, const matrix_case & item,
        route_reset_fn reset, route_snapshot_fn snapshot) {
    const bool alpha = item.id.find("alpha") != std::string_view::npos;
    const int salt = alpha ? 17 : 5;
    const ggml_op_hint role = alpha ? GGML_HINT_MUL_MAT_ROW_INVARIANT_BF16_ALPHA :
            GGML_HINT_MUL_MAT_ROW_INVARIANT_BF16_BETA;
    const route candidate_route = alpha ? BF16_ROW_INVARIANT_ALPHA : BF16_ROW_INVARIANT_BETA;
    const auto weights = make_bf16_weights(item.k, item.n, salt);

    // Untagged exact-shape M8 must remain on the established cuBLAS path.
    bf16_runner negative(backend, item.k, item.n, 8, GGML_HINT_NONE, role);
    (void) negative.run(backend, weights, make_bf16_projection_input(item.k, 8, salt));
    route_counts after_untagged{};
    if (!snapshot(&after_untagged, sizeof(after_untagged)) ||
            after_untagged.abi_version != ROUTE_ABI || after_untagged.route_count != ROUTE_COUNT ||
            after_untagged.count[BF16_CUBLAS] == 0 ||
            after_untagged.count[BF16_ROW_INVARIANT_BETA] != 0 ||
            after_untagged.count[BF16_ROW_INVARIANT_ALPHA] != 0) {
        fail("untagged BF16 M8 negative control did not preserve cuBLAS");
    }
    // A tagged but non-contiguous activation is outside the candidate layout
    // contract and must also remain on cuBLAS.
    bf16_runner noncontiguous_negative(backend, item.k, item.n, 8, role, role, true);
    (void) noncontiguous_negative.run(
            backend, weights, make_bf16_projection_input(item.k, 8, salt + 31));
    route_counts after_noncontiguous{};
    if (!snapshot(&after_noncontiguous, sizeof(after_noncontiguous)) ||
            after_noncontiguous.abi_version != ROUTE_ABI || after_noncontiguous.route_count != ROUTE_COUNT ||
            after_noncontiguous.count[BF16_CUBLAS] <= after_untagged.count[BF16_CUBLAS] ||
            after_noncontiguous.count[BF16_ROW_INVARIANT_BETA] != 0 ||
            after_noncontiguous.count[BF16_ROW_INVARIANT_ALPHA] != 0) {
        fail("tagged non-contiguous BF16 M8 negative control did not preserve cuBLAS");
    }
    std::printf(
            "{\"type\":\"route_negative_control\",\"id\":\"%.*s\","
            "\"untagged_m8_cublas\":%llu,\"tagged_noncontiguous_m8_cublas\":%llu,"
            "\"candidate_beta\":0,\"candidate_alpha\":0}\n",
            (int) item.id.size(), item.id.data(),
            (unsigned long long) after_untagged.count[BF16_CUBLAS],
            (unsigned long long) (after_noncontiguous.count[BF16_CUBLAS] - after_untagged.count[BF16_CUBLAS]));
    reset();

    std::vector<std::unique_ptr<bf16_runner>> scalars;
    std::vector<std::unique_ptr<bf16_runner>> batches;
    for (int m = 2; m <= 8; ++m) {
        scalars.emplace_back(std::make_unique<bf16_runner>(backend, item.k, item.n, 1, role, role));
        batches.emplace_back(std::make_unique<bf16_runner>(backend, item.k, item.n, m, role, role));
    }
    for (int m = 2; m <= 8; ++m) {
        const auto values = make_bf16_projection_input(item.k, m, salt);
        route_counts before{};
        if (!snapshot(&before, sizeof(before)) || before.abi_version != ROUTE_ABI || before.route_count != ROUTE_COUNT) {
            fail("CUDA route-marker ABI rejected BF16 candidate pre-width snapshot");
        }
        bf16_observation stable_scalar, stable_batch;
        for (int replay = 0; replay < REPLAYS; ++replay) {
            bf16_observation one, many;
            auto run_one = [&] { return run_bf16_scalar_rows_observed(
                    *scalars[(size_t) (m - 2)], backend, weights, values, item.k, item.n, m); };
            auto run_many = [&] { return run_bf16_observed(
                    *batches[(size_t) (m - 2)], backend, weights, values); };
            if (replay == 0) {
                one = run_one();
                route_counts after_scalar{};
                if (!snapshot(&after_scalar, sizeof(after_scalar)) ||
                        after_scalar.abi_version != ROUTE_ABI || after_scalar.route_count != ROUTE_COUNT ||
                        after_scalar.count[BF16_MMVF] <= before.count[BF16_MMVF] ||
                        after_scalar.count[candidate_route] != before.count[candidate_route] ||
                        after_scalar.count[BF16_CUBLAS] != before.count[BF16_CUBLAS]) {
                    fail("tagged BF16 M1 did not preserve the ordinary MMVF route");
                }
                many = run_many();
                route_counts after_batch{};
                if (!snapshot(&after_batch, sizeof(after_batch)) ||
                        after_batch.abi_version != ROUTE_ABI || after_batch.route_count != ROUTE_COUNT ||
                        after_batch.count[candidate_route] <= after_scalar.count[candidate_route] ||
                        after_batch.count[BF16_MMVF] != after_scalar.count[BF16_MMVF] ||
                        after_batch.count[BF16_CUBLAS] != after_scalar.count[BF16_CUBLAS]) {
                    fail("tagged BF16 batch did not exclusively select row-invariant MMVF");
                }
                std::printf(
                        "{\"type\":\"bf16_candidate_route_probe\",\"id\":\"%.*s\",\"m\":%d,"
                        "\"scalar_mmvf_delta\":%llu,\"scalar_candidate_delta\":0,\"scalar_cublas_delta\":0,"
                        "\"batch_mmvf_delta\":0,\"batch_candidate_delta\":%llu,\"batch_cublas_delta\":0,"
                        "\"input_bf16_roundtrip_sensitive\":true}\n",
                        (int) item.id.size(), item.id.data(), m,
                        (unsigned long long) (after_scalar.count[BF16_MMVF] - before.count[BF16_MMVF]),
                        (unsigned long long) (after_batch.count[candidate_route] - after_scalar.count[candidate_route]));
                std::fflush(stdout);
            } else if ((replay & 1) == 0) { one = run_one(); many = run_many(); }
            else { many = run_many(); one = run_one(); }
            require_bits_equal(one.raw_projection, many.raw_projection, item.id, replay);
            require_bits_equal(one.transformed, many.transformed, item.id, replay);
            if (replay == 0) { stable_scalar = one; stable_batch = many; }
            else {
                require_bits_equal(stable_scalar.raw_projection, one.raw_projection, item.id, replay);
                require_bits_equal(stable_scalar.transformed, one.transformed, item.id, replay);
                require_bits_equal(stable_batch.raw_projection, many.raw_projection, item.id, replay);
                require_bits_equal(stable_batch.transformed, many.transformed, item.id, replay);
            }
        }
        std::printf(
                "{\"type\":\"width_route\",\"id\":\"%.*s\",\"m\":%d,"
                "\"replays\":8,\"finite\":true,\"comparison\":\"bitwise_f32\","
                "\"raw_projection_bitwise\":true,"
                "\"transformed_path\":\"%s\"}\n",
                (int) item.id.size(), item.id.data(), m, alpha ? "add_softplus_mul" : "sigmoid");
    }

    // Small, correctness-adjacent launch benchmark. Raw synchronized samples
    // are retained; this is not an end-to-end performance qualification.
    const auto bench_values = make_bf16_projection_input(item.k, 8, salt);
    bf16_runner baseline(backend, item.k, item.n, 8, GGML_HINT_NONE, role);
    bf16_runner candidate(backend, item.k, item.n, 8, role, role);
    baseline.prepare(weights, bench_values);
    candidate.prepare(weights, bench_values);
    for (int i = 0; i < 4; ++i) {
        if (i & 1) { candidate.compute(backend); baseline.compute(backend); }
        else { baseline.compute(backend); candidate.compute(backend); }
    }
    ggml_backend_synchronize(backend);
    std::vector<double> baseline_us, candidate_us;
    for (int i = 0; i < 16; ++i) {
        if (i & 1) {
            candidate_us.push_back(timed_bf16_compute_us(candidate, backend));
            baseline_us.push_back(timed_bf16_compute_us(baseline, backend));
        } else {
            baseline_us.push_back(timed_bf16_compute_us(baseline, backend));
            candidate_us.push_back(timed_bf16_compute_us(candidate, backend));
        }
    }
    const double baseline_median = percentile_us(baseline_us, 0.5);
    const double candidate_median = percentile_us(candidate_us, 0.5);
    std::printf(
            "{\"type\":\"microbenchmark\",\"id\":\"%.*s\",\"m\":8,"
            "\"warmups\":4,\"samples_per_arm\":16,\"order\":\"AB_BA\","
            "\"baseline\":\"untagged_cublas\",\"candidate\":\"tagged_row_invariant_mmvf\","
            "\"baseline_median_us\":%.3f,\"baseline_p95_us\":%.3f,"
            "\"candidate_median_us\":%.3f,\"candidate_p95_us\":%.3f,\"median_ratio\":%.6f,"
            "\"baseline_samples_us\":",
            (int) item.id.size(), item.id.data(), baseline_median, percentile_us(baseline_us, 0.95),
            candidate_median, percentile_us(candidate_us, 0.95), candidate_median / baseline_median);
    print_samples(baseline_us);
    std::printf(",\"candidate_samples_us\":");
    print_samples(candidate_us);
    std::printf("}\n");
    std::printf(
            "{\"type\":\"width_coverage\",\"id\":\"%.*s\",\"status\":\"passed\","
            "\"min_m\":2,\"max_m\":8,\"widths\":7,\"replays\":8,"
            "\"finite\":true,\"comparison\":\"bitwise_f32\",\"route_checked_each_width\":true}\n",
            (int) item.id.size(), item.id.data());
    std::printf(
            "{\"type\":\"case\",\"id\":\"%.*s\",\"status\":\"passed\","
            "\"semantic\":\"%s\",\"transformed_path\":\"%s\",\"replays\":8,"
            "\"min_batch_m\":2,\"max_batch_m\":8,\"widths\":7,"
            "\"finite\":true,\"comparison\":\"bitwise_f32\",\"raw_projection_bitwise\":true}\n",
            (int) item.id.size(), item.id.data(), alpha ? "alpha" : "beta",
            alpha ? "add_softplus_mul" : "sigmoid");
}

struct norm_runner {
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * input_storage = nullptr;
    ggml_tensor * scale = nullptr;
    ggml_tensor * gate = nullptr;
    ggml_tensor * output = nullptr;
    operation op;
    int64_t features;
    int64_t heads;
    int64_t tokens;
    int64_t input_values_per_token;

    norm_runner(ggml_backend_t backend, operation op_, int64_t features_, int64_t heads_, int64_t tokens_,
            int64_t l2_offset = 0)
        : op(op_), features(features_), heads(heads_), tokens(tokens_),
          input_values_per_token(features_ * heads_) {
        ggml_init_params params{
            ggml_tensor_overhead() * 16 + ggml_graph_overhead_custom(32, false), nullptr, true,
        };
        ctx.reset(ggml_init(params));
        if (!ctx) fail("failed to create normalization diagnostic context");
        if (op == operation::rms_norm) {
            input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, features, tokens);
            input_storage = input;
            scale = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, features);
            output = ggml_mul(ctx.get(), ggml_rms_norm(ctx.get(), input, 1e-6f), scale);
        } else if (op == operation::l2_norm) {
            input_values_per_token = 10240;
            input_storage = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, input_values_per_token, tokens);
            input = ggml_view_4d(ctx.get(), input_storage, features, heads, tokens, 1,
                    (size_t) features * sizeof(float),
                    (size_t) input_values_per_token * sizeof(float),
                    (size_t) input_values_per_token * tokens * sizeof(float),
                    (size_t) l2_offset * sizeof(float));
            output = ggml_l2_norm(ctx.get(), input, 1e-6f);
        } else if (op == operation::gated_norm) {
            input = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, features, heads, tokens, 1);
            input_storage = input;
            scale = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, features);
            gate = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, features, heads, tokens, 1);
            ggml_tensor * normalized = ggml_mul(ctx.get(), ggml_rms_norm(ctx.get(), input, 1e-6f), scale);
            output = ggml_mul(ctx.get(), normalized, ggml_silu(ctx.get(), gate));
        } else {
            fail("invalid normalization diagnostic operation");
        }
        ggml_set_name(output, "qwen35_row_invariance_norm_output");
        buffer.reset(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
        if (!buffer || !ggml_backend_supports_op(backend, output)) {
            fail("CUDA backend rejected normalization diagnostic graph");
        }
        graph = ggml_new_graph_custom(ctx.get(), 32, false);
        ggml_build_forward_expand(graph, output);
    }

    std::vector<float> run(
            ggml_backend_t backend,
            const std::vector<float> & values,
            const std::vector<float> & scales,
            const std::vector<float> & gates = {}) {
        const size_t count = (size_t) features * heads * tokens;
        const size_t input_count = (size_t) input_values_per_token * tokens;
        if (values.size() != input_count || (scale && scales.size() != (size_t) features) ||
                (gate && gates.size() != count)) {
            fail("normalization diagnostic fixture cardinality mismatch");
        }
        ggml_backend_tensor_set(input_storage, values.data(), 0, values.size() * sizeof(float));
        if (scale) ggml_backend_tensor_set(scale, scales.data(), 0, scales.size() * sizeof(float));
        if (gate) ggml_backend_tensor_set(gate, gates.data(), 0, gates.size() * sizeof(float));
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            fail("normalization CUDA graph compute failed");
        }
        std::vector<float> result(count);
        ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));
        require_finite(result, "normalization");
        return result;
    }
};

std::vector<float> run_norm_scalar_rows(
        norm_runner & scalar, ggml_backend_t backend,
        const std::vector<float> & values, const std::vector<float> & scales,
        const std::vector<float> & gates, size_t input_per_token, size_t output_per_token, int tokens) {
    std::vector<float> result(output_per_token * (size_t) tokens);
    for (int token = 0; token < tokens; ++token) {
        const std::vector<float> one(values.begin() + (size_t) token * input_per_token,
                                     values.begin() + (size_t) (token + 1) * input_per_token);
        const std::vector<float> one_gate = gates.empty() ? std::vector<float>{} :
                std::vector<float>(gates.begin() + (size_t) token * output_per_token,
                                   gates.begin() + (size_t) (token + 1) * output_per_token);
        const auto output = scalar.run(backend, one, scales, one_gate);
        std::copy(output.begin(), output.end(), result.begin() + (size_t) token * output_per_token);
    }
    return result;
}

void run_norm_case(
        ggml_backend_t backend, const matrix_case & item,
        route_snapshot_fn snapshot) {
    const int64_t features = item.k;
    const int64_t heads = item.n;
    const size_t per_token = (size_t) features * heads;
    const int64_t l2_offset = item.op == operation::l2_norm && item.id.find("-k-") != std::string_view::npos ? 2048 : 0;
    const size_t input_per_token = item.op == operation::l2_norm ? 10240 : per_token;
    std::vector<std::unique_ptr<norm_runner>> scalars;
    std::vector<std::unique_ptr<norm_runner>> batches;
    for (int m = 2; m <= 8; ++m) {
        scalars.emplace_back(std::make_unique<norm_runner>(backend, item.op, features, heads, 1, l2_offset));
        batches.emplace_back(std::make_unique<norm_runner>(backend, item.op, features, heads, m, l2_offset));
    }
    std::vector<float> scales((size_t) features);
    for (int64_t i = 0; i < features; ++i) {
        scales[(size_t) i] = 0.75f + 0.03125f * (float) ((i * 5 + 3) % 13);
    }

    for (int m = 2; m <= 8; ++m) {
        const auto values = make_operator_values(input_per_token, m, 3);
        const auto gates = item.op == operation::gated_norm ? make_operator_values(per_token, m, 9) :
                std::vector<float>{};
        route_counts before{};
        if (!snapshot(&before, sizeof(before)) || before.abi_version != ROUTE_ABI || before.route_count != ROUTE_COUNT) {
            fail("CUDA route-marker ABI rejected normalization pre-width snapshot");
        }
        const route selected = item.op == operation::rms_norm ? RMS_NORM_MUL :
                item.op == operation::l2_norm ? L2_NORM : SILU_MUL;
        const route secondary = item.op == operation::gated_norm ? RMS_NORM_MUL : ROUTE_COUNT;
        uint64_t scalar_selected_delta = 0, batch_selected_delta = 0;
        uint64_t scalar_secondary_delta = 0, batch_secondary_delta = 0;
        uint64_t scalar_fallback_delta = 0, batch_fallback_delta = 0;
        std::vector<float> stable_scalar, stable_batch;
        for (int replay = 0; replay < REPLAYS; ++replay) {
            std::vector<float> one, many;
            auto run_one = [&] { return run_norm_scalar_rows(
                    *scalars[(size_t) (m - 2)], backend, values, scales, gates, input_per_token, per_token, m); };
            auto run_many = [&] { return batches[(size_t) (m - 2)]->run(backend, values, scales, gates); };
            if (replay == 0) {
                one = run_one();
                route_counts after_scalar{};
                if (!snapshot(&after_scalar, sizeof(after_scalar)) ||
                        after_scalar.abi_version != ROUTE_ABI || after_scalar.route_count != ROUTE_COUNT ||
                        after_scalar.count[selected] <= before.count[selected]) {
                    fail("scalar normalization missed its mandatory production CUDA route");
                }
                scalar_selected_delta = after_scalar.count[selected] - before.count[selected];
                if (secondary != ROUTE_COUNT) {
                    if (after_scalar.count[secondary] <= before.count[secondary]) {
                        fail("scalar gated normalization missed its secondary fused CUDA route");
                    }
                    scalar_secondary_delta = after_scalar.count[secondary] - before.count[secondary];
                }
                scalar_fallback_delta = (after_scalar.count[RMS_NORM] - before.count[RMS_NORM]) +
                        (after_scalar.count[SILU] - before.count[SILU]);
                if (scalar_fallback_delta != 0) {
                    fail("scalar normalization selected a standalone fallback route");
                }

                many = run_many();
                route_counts after_batch{};
                if (!snapshot(&after_batch, sizeof(after_batch)) ||
                        after_batch.abi_version != ROUTE_ABI || after_batch.route_count != ROUTE_COUNT ||
                        after_batch.count[selected] <= after_scalar.count[selected]) {
                    fail("batched normalization missed its mandatory production CUDA route");
                }
                batch_selected_delta = after_batch.count[selected] - after_scalar.count[selected];
                if (secondary != ROUTE_COUNT) {
                    if (after_batch.count[secondary] <= after_scalar.count[secondary]) {
                        fail("batched gated normalization missed its secondary fused CUDA route");
                    }
                    batch_secondary_delta = after_batch.count[secondary] - after_scalar.count[secondary];
                }
                batch_fallback_delta = (after_batch.count[RMS_NORM] - after_scalar.count[RMS_NORM]) +
                        (after_batch.count[SILU] - after_scalar.count[SILU]);
                if (batch_fallback_delta != 0) {
                    fail("batched normalization selected a standalone fallback route");
                }
                std::printf(
                        "{\"type\":\"norm_route_probe\",\"id\":\"%.*s\",\"m\":%d,"
                        "\"scalar_selected_delta\":%llu,\"batch_selected_delta\":%llu,"
                        "\"scalar_secondary_delta\":%llu,\"batch_secondary_delta\":%llu,"
                        "\"scalar_fallback_delta\":%llu,\"batch_fallback_delta\":%llu}\n",
                        (int) item.id.size(), item.id.data(), m,
                        (unsigned long long) scalar_selected_delta, (unsigned long long) batch_selected_delta,
                        (unsigned long long) scalar_secondary_delta, (unsigned long long) batch_secondary_delta,
                        (unsigned long long) scalar_fallback_delta, (unsigned long long) batch_fallback_delta);
                std::fflush(stdout);
            } else if ((replay & 1) == 0) { one = run_one(); many = run_many(); }
            else { many = run_many(); one = run_one(); }
            require_bits_equal(one, many, item.id, replay);
            if (replay == 0) { stable_scalar = one; stable_batch = many; }
            else {
                require_bits_equal(stable_scalar, one, item.id, replay);
                require_bits_equal(stable_batch, many, item.id, replay);
            }
        }
        route_counts after{};
        if (!snapshot(&after, sizeof(after)) || after.abi_version != ROUTE_ABI || after.route_count != ROUTE_COUNT) {
            fail("CUDA route-marker ABI rejected normalization post-width snapshot");
        }
        if (after.count[selected] <= before.count[selected]) {
            fail("normalization width missed its mandatory production CUDA route");
        }
        if ((item.op == operation::rms_norm || item.op == operation::gated_norm) &&
                (after.count[RMS_NORM] != before.count[RMS_NORM] ||
                 after.count[RMS_NORM_MUL] <= before.count[RMS_NORM_MUL])) {
            fail("normalization width fell through from fused RMS_NORM->MUL");
        }
        if (item.op == operation::gated_norm &&
                (after.count[SILU] != before.count[SILU] || after.count[SILU_MUL] <= before.count[SILU_MUL])) {
            fail("gated normalization width fell through from fused SILU->MUL");
        }
        std::printf(
                "{\"type\":\"width_route\",\"id\":\"%.*s\",\"m\":%d,"
                "\"selected_before\":%llu,\"selected_after\":%llu,\"selected_delta\":%llu,"
                "\"scalar_selected_delta\":%llu,\"batch_selected_delta\":%llu,"
                "\"scalar_secondary_delta\":%llu,\"batch_secondary_delta\":%llu,"
                "\"scalar_fallback_delta\":%llu,\"batch_fallback_delta\":%llu,"
                "\"finite\":true,\"comparison\":\"bitwise_f32\",\"replays\":8}\n",
                (int) item.id.size(), item.id.data(), m,
                (unsigned long long) before.count[selected],
                (unsigned long long) after.count[selected],
                (unsigned long long) (after.count[selected] - before.count[selected]),
                (unsigned long long) scalar_selected_delta, (unsigned long long) batch_selected_delta,
                (unsigned long long) scalar_secondary_delta, (unsigned long long) batch_secondary_delta,
                (unsigned long long) scalar_fallback_delta, (unsigned long long) batch_fallback_delta);
    }
    std::printf(
            "{\"type\":\"width_coverage\",\"id\":\"%.*s\",\"status\":\"passed\","
            "\"min_m\":2,\"max_m\":8,\"widths\":7,\"replays\":8,"
            "\"finite\":true,\"comparison\":\"bitwise_f32\",\"route_checked_each_width\":true}\n",
            (int) item.id.size(), item.id.data());
    std::printf(
            "{\"type\":\"case\",\"id\":\"%.*s\",\"status\":\"passed\","
            "\"replays\":8,\"min_batch_m\":2,\"max_batch_m\":8,\"widths\":7,"
            "\"finite\":true,\"comparison\":\"bitwise_f32\"}\n",
            (int) item.id.size(), item.id.data());
}

struct ssm_conv_runner {
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * kernel = nullptr;
    ggml_tensor * output = nullptr;
    ggml_tensor * cache = nullptr;
    int64_t d_conv;
    int64_t channels;
    int64_t tokens;

    ssm_conv_runner(ggml_backend_t backend, int64_t d_conv_, int64_t channels_, int64_t tokens_)
        : d_conv(d_conv_), channels(channels_), tokens(tokens_) {
        ggml_init_params params{
            ggml_tensor_overhead() * 16 + ggml_graph_overhead_custom(32, false), nullptr, true,
        };
        ctx.reset(ggml_init(params));
        if (!ctx) fail("failed to create SSM_CONV diagnostic context");
        input = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, d_conv - 1 + tokens, channels, 1);
        kernel = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, d_conv, channels);
        output = ggml_silu(ctx.get(), ggml_ssm_conv(ctx.get(), input, kernel));
        const int64_t state_size = (d_conv - 1) * channels;
        cache = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, state_size * tokens);
        ggml_set_name(output, "qwen35_ssm_conv_silu_output");
        buffer.reset(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
        if (!buffer || !ggml_backend_supports_op(backend, output)) {
            fail("CUDA backend rejected SSM_CONV diagnostic graph");
        }
        graph = ggml_new_graph_custom(ctx.get(), 32, false);
        const size_t state_bytes = (size_t) state_size * sizeof(float);
        for (int64_t t = 1; t <= tokens; ++t) {
            const int64_t source_start = t;
            const int64_t destination_slot = tokens - t;
            ggml_tensor * tail = ggml_view_3d(ctx.get(), input, d_conv - 1, channels, 1,
                    input->nb[1], input->nb[2], (size_t) source_start * sizeof(float));
            ggml_tensor * slot = ggml_view_2d(ctx.get(), cache, state_size, 1,
                    state_bytes, (size_t) destination_slot * state_bytes);
            ggml_tensor * cache_write = ggml_cpy(ctx.get(), tail, slot);
            if (!ggml_backend_supports_op(backend, cache_write)) {
                fail("CUDA backend rejected SSM_CONV cache-slot copy");
            }
            ggml_build_forward_expand(graph, cache_write);
        }
        ggml_build_forward_expand(graph, output);
    }

    std::pair<std::vector<float>, std::vector<float>> run(
            ggml_backend_t backend, const std::vector<float> & input_values,
            const std::vector<float> & kernel_values) {
        if (input_values.size() != (size_t) (d_conv - 1 + tokens) * channels ||
                kernel_values.size() != (size_t) d_conv * channels) {
            fail("SSM_CONV diagnostic fixture cardinality mismatch");
        }
        ggml_backend_tensor_set(input, input_values.data(), 0, input_values.size() * sizeof(float));
        ggml_backend_tensor_set(kernel, kernel_values.data(), 0, kernel_values.size() * sizeof(float));
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            fail("SSM_CONV CUDA graph compute failed");
        }
        std::vector<float> outputs((size_t) channels * tokens);
        std::vector<float> evolved((size_t) channels * (d_conv - 1) * tokens);
        ggml_backend_tensor_get(output, outputs.data(), 0, outputs.size() * sizeof(float));
        ggml_backend_tensor_get(cache, evolved.data(), 0, evolved.size() * sizeof(float));
        require_finite(outputs, "ssm_conv_output");
        require_finite(evolved, "ssm_conv_cache");
        return { std::move(outputs), std::move(evolved) };
    }
};

std::vector<float> make_ssm_input(
        const std::vector<float> & history, const std::vector<float> & tokens,
        int64_t channels, int token_count) {
    constexpr int history_rows = 3;
    if (history.size() != (size_t) history_rows * channels ||
            tokens.size() != (size_t) token_count * channels) {
        fail("SSM_CONV input fixture cardinality mismatch");
    }
    std::vector<float> result((size_t) (history_rows + token_count) * channels);
    for (int64_t channel = 0; channel < channels; ++channel) {
        for (int row = 0; row < history_rows; ++row) {
            result[(size_t) channel * (history_rows + token_count) + row] =
                    history[(size_t) channel * history_rows + row];
        }
        for (int token = 0; token < token_count; ++token) {
            result[(size_t) channel * (history_rows + token_count) + history_rows + token] =
                    tokens[(size_t) token * channels + channel];
        }
    }
    return result;
}

std::pair<std::vector<float>, std::vector<float>> run_ssm_scalar_sequence(
        ssm_conv_runner & scalar, ggml_backend_t backend,
        const std::vector<float> & initial_history, const std::vector<float> & tokens,
        const std::vector<float> & kernel, int64_t channels) {
    std::vector<float> history = initial_history;
    std::vector<float> outputs((size_t) channels * 8);
    std::vector<std::vector<float>> states;
    states.reserve(8);
    for (int token = 0; token < 8; ++token) {
        const std::vector<float> one_token(tokens.begin() + (size_t) token * channels,
                                           tokens.begin() + (size_t) (token + 1) * channels);
        auto current = scalar.run(backend, make_ssm_input(history, one_token, channels, 1), kernel);
        std::copy(current.first.begin(), current.first.end(), outputs.begin() + (size_t) token * channels);
        history = std::move(current.second);
        states.push_back(history);
    }
    std::vector<float> newest_first(history.size() * 8);
    for (int slot = 0; slot < 8; ++slot) {
        std::copy(states[(size_t) (7 - slot)].begin(), states[(size_t) (7 - slot)].end(),
                newest_first.begin() + (size_t) slot * history.size());
    }
    return { std::move(outputs), std::move(newest_first) };
}

void run_ssm_conv_case(
        ggml_backend_t backend, const matrix_case & item,
        route_snapshot_fn snapshot) {
    const int64_t d_conv = item.k;
    const int64_t channels = item.n;
    ssm_conv_runner scalar(backend, d_conv, channels, 1);
    ssm_conv_runner batch(backend, d_conv, channels, 8);
    auto history_token_major = make_operator_values((size_t) channels, 3, 5);
    std::vector<float> history((size_t) channels * 3);
    for (int64_t channel = 0; channel < channels; ++channel) {
        for (int row = 0; row < 3; ++row) {
            history[(size_t) channel * 3 + row] = history_token_major[(size_t) row * channels + channel];
        }
    }
    const auto tokens = make_operator_values((size_t) channels, 8, 11);
    std::vector<float> kernel((size_t) d_conv * channels);
    for (int64_t channel = 0; channel < channels; ++channel) {
        for (int64_t tap = 0; tap < d_conv; ++tap) {
            const float sign = ((channel * 3 + tap) & 1) ? -1.0f : 1.0f;
            kernel[(size_t) channel * d_conv + tap] = sign * (0.125f + 0.03125f * (float) ((channel + tap * 7) % 9));
        }
    }

    route_counts before{};
    if (!snapshot(&before, sizeof(before)) || before.abi_version != ROUTE_ABI || before.route_count != ROUTE_COUNT) {
        fail("CUDA route-marker ABI rejected SSM_CONV pre-run snapshot");
    }
    std::vector<float> stable_output, stable_cache;
    route_counts scalar_route{}, batch_route{};
    for (int replay = 0; replay < REPLAYS; ++replay) {
        std::pair<std::vector<float>, std::vector<float>> seq, many;
        auto run_seq = [&] { return run_ssm_scalar_sequence(
                scalar, backend, history, tokens, kernel, channels); };
        auto run_many = [&] { return batch.run(
                backend, make_ssm_input(history, tokens, channels, 8), kernel); };
        if ((replay & 1) == 0) {
            seq = run_seq();
            if (replay == 0 && (!snapshot(&scalar_route, sizeof(scalar_route)) ||
                    scalar_route.count[SSM_CONV_SILU] <= before.count[SSM_CONV_SILU])) {
                fail("scalar sequential SSM_CONV missed the fused SILU route");
            }
            many = run_many();
            if (replay == 0 && (!snapshot(&batch_route, sizeof(batch_route)) ||
                    batch_route.count[SSM_CONV_SILU] <= scalar_route.count[SSM_CONV_SILU])) {
                fail("width-8 SSM_CONV missed the fused SILU route");
            }
        } else {
            many = run_many();
            seq = run_seq();
        }
        require_bits_equal(seq.first, many.first, item.id, replay);
        require_bits_equal(seq.second, many.second, item.id, replay);
        if (replay == 0) { stable_output = seq.first; stable_cache = seq.second; }
        else {
            require_bits_equal(stable_output, seq.first, item.id, replay);
            require_bits_equal(stable_cache, seq.second, item.id, replay);
        }
    }
    route_counts after{};
    if (!snapshot(&after, sizeof(after)) || after.abi_version != ROUTE_ABI || after.route_count != ROUTE_COUNT ||
            after.count[SSM_CONV] != before.count[SSM_CONV] ||
            after.count[SILU] != before.count[SILU]) {
        fail("SSM_CONV diagnostic fell through from the fused SSM_CONV->SILU route");
    }
    std::printf(
            "{\"type\":\"ssm_conv_evolution\",\"id\":\"%.*s\",\"d_conv\":4,"
            "\"channels\":10240,\"initial_cache_rows\":3,\"tokens\":8,"
            "\"scalar_steps\":8,\"batch_rows\":8,\"output_exact\":true,"
            "\"evolved_cache_exact\":true,\"cache_mapping\":\"eight_slots_newest_first\","
            "\"finite\":true,\"replays\":8,\"output_fnv1a64\":\"%016llx\","
            "\"cache_fnv1a64\":\"%016llx\"}\n",
            (int) item.id.size(), item.id.data(),
            (unsigned long long) fnv1a(stable_output), (unsigned long long) fnv1a(stable_cache));
    std::printf(
            "{\"type\":\"case\",\"id\":\"%.*s\",\"status\":\"passed\","
            "\"replays\":8,\"finite\":true,\"comparison\":\"bitwise_f32\"}\n",
            (int) item.id.size(), item.id.data());
}

constexpr int64_t FATTN_D = 256;
constexpr int64_t FATTN_N_KV = 1024;
constexpr int64_t FATTN_H_Q = 24;
constexpr int64_t FATTN_H_KV = 4;

std::vector<uint8_t> make_fattn_q8_cache(int salt) {
    const int64_t rows = FATTN_N_KV * FATTN_H_KV;
    std::vector<float> source((size_t) rows * FATTN_D);
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t col = 0; col < FATTN_D; ++col) {
            const float sign = ((row * 7 + col * 11 + salt) & 1) ? -1.0f : 1.0f;
            source[(size_t) row * FATTN_D + col] =
                    sign * (0.015625f + 0.00390625f * (float) ((row * 3 + col * 5 + salt) % 31));
        }
    }
    std::vector<uint8_t> result(ggml_row_size(GGML_TYPE_Q8_0, FATTN_D) * (size_t) rows);
    const size_t written = ggml_quantize_chunk(
            GGML_TYPE_Q8_0, source.data(), result.data(), 0, rows, FATTN_D, nullptr);
    if (written != result.size()) fail("Q8_0 attention cache quantizer byte-count drift");
    return result;
}

std::vector<float> make_fattn_q(int m) {
    std::vector<float> result((size_t) FATTN_D * FATTN_H_Q * m);
    for (int token = 0; token < m; ++token) {
        for (int64_t head = 0; head < FATTN_H_Q; ++head) {
            for (int64_t col = 0; col < FATTN_D; ++col) {
                const size_t index = ((size_t) token * FATTN_H_Q + head) * FATTN_D + col;
                const float sign = ((token * 13 + head * 7 + col * 3) & 1) ? -1.0f : 1.0f;
                result[index] = sign * (0.0078125f +
                        0.001953125f * (float) ((token * 17 + head * 5 + col * 11) % 37));
            }
        }
    }
    return result;
}

std::vector<ggml_fp16_t> make_fattn_mask(int m) {
    std::vector<ggml_fp16_t> result((size_t) FATTN_N_KV * m);
    for (int token = 0; token < m; ++token) {
        const int64_t cutoff = FATTN_N_KV - m + token;
        int64_t zeros = 0, negative_infinities = 0;
        for (int64_t key = 0; key < FATTN_N_KV; ++key) {
            const float value = key <= cutoff ? 0.0f : -std::numeric_limits<float>::infinity();
            result[(size_t) token * FATTN_N_KV + key] = ggml_fp32_to_fp16(value);
            const float roundtrip = ggml_fp16_to_fp32(result[(size_t) token * FATTN_N_KV + key]);
            zeros += roundtrip == 0.0f;
            negative_infinities += std::isinf(roundtrip) && std::signbit(roundtrip);
        }
        if (zeros != cutoff + 1 || negative_infinities != FATTN_N_KV - cutoff - 1) {
            fail("causal F16 attention mask contents drift");
        }
    }
    return result;
}

struct fattn_pb1_runner {
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * q_storage = nullptr;
    ggml_tensor * q = nullptr;
    ggml_tensor * k_storage = nullptr;
    ggml_tensor * k = nullptr;
    ggml_tensor * v_storage = nullptr;
    ggml_tensor * v = nullptr;
    ggml_tensor * mask = nullptr;
    ggml_tensor * output = nullptr;
    int m = 0;

    fattn_pb1_runner(
            ggml_backend_t backend, int m_, bool tagged,
            const std::vector<uint8_t> & k_values,
            const std::vector<uint8_t> & v_values) : m(m_) {
        ggml_init_params params{
            ggml_tensor_overhead() * 20 + ggml_graph_overhead_custom(32, false), nullptr, true,
        };
        ctx.reset(ggml_init(params));
        if (!ctx) fail("failed to create FATTN PB1 diagnostic context");
        q_storage = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, FATTN_D, FATTN_H_Q, m, 1);
        q = ggml_permute(ctx.get(), q_storage, 0, 2, 1, 3);
        k_storage = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_Q8_0, FATTN_D, FATTN_H_KV, FATTN_N_KV, 1);
        v_storage = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_Q8_0, FATTN_D, FATTN_H_KV, FATTN_N_KV, 1);
        k = ggml_permute(ctx.get(), k_storage, 0, 2, 1, 3);
        v = ggml_permute(ctx.get(), v_storage, 0, 2, 1, 3);
        mask = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, FATTN_N_KV, m, 1, 1);
        output = ggml_flash_attn_ext(ctx.get(), q, k, v, mask, 0.0625f, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(output, GGML_PREC_F32);
        if (tagged) {
            ggml_flash_attn_ext_set_hint(
                    output, GGML_HINT_FLASH_ATTN_QWEN35_D256_Q8_GQA6_VEC_PB1_DIAGNOSTIC);
        }

        const size_t q_token_bytes = sizeof(float) * FATTN_D * FATTN_H_Q;
        const size_t q8_row = ggml_row_size(GGML_TYPE_Q8_0, FATTN_D);
        const size_t q8_all_heads = q8_row * FATTN_H_KV;
        float scale = 0.0f, max_bias = 0.0f, softcap = 0.0f;
        std::memcpy(&scale, (const float *) output->op_params + 0, sizeof(scale));
        std::memcpy(&max_bias, (const float *) output->op_params + 1, sizeof(max_bias));
        std::memcpy(&softcap, (const float *) output->op_params + 2, sizeof(softcap));
        if (q->ne[0] != FATTN_D || q->ne[1] != m || q->ne[2] != FATTN_H_Q || q->ne[3] != 1 ||
                q->nb[0] != sizeof(float) || q->nb[1] != q_token_bytes ||
                q->nb[2] != sizeof(float) * FATTN_D || q->nb[3] != q_token_bytes * (size_t) m ||
                k->ne[0] != FATTN_D || k->ne[1] != FATTN_N_KV || k->ne[2] != FATTN_H_KV || k->ne[3] != 1 ||
                v->ne[0] != FATTN_D || v->ne[1] != FATTN_N_KV || v->ne[2] != FATTN_H_KV || v->ne[3] != 1 ||
                k->nb[0] != ggml_type_size(GGML_TYPE_Q8_0) || k->nb[1] != q8_all_heads || k->nb[2] != q8_row ||
                k->nb[3] != q8_all_heads * FATTN_N_KV ||
                std::memcmp(k->nb, v->nb, sizeof(k->nb)) != 0 ||
                mask->nb[0] != sizeof(ggml_fp16_t) || mask->nb[1] != sizeof(ggml_fp16_t) * FATTN_N_KV ||
                mask->nb[2] != sizeof(ggml_fp16_t) * FATTN_N_KV * (size_t) m || mask->nb[3] != mask->nb[2] ||
                output->ne[0] != FATTN_D || output->ne[1] != FATTN_H_Q || output->ne[2] != m || output->ne[3] != 1 ||
                output->nb[0] != sizeof(float) || output->nb[1] != sizeof(float) * FATTN_D ||
                output->nb[2] != q_token_bytes || output->nb[3] != q_token_bytes * (size_t) m ||
                output->src[4] != nullptr || ggml_flash_attn_ext_get_prec(output) != GGML_PREC_F32 ||
                scale != 0.0625f || max_bias != 0.0f || softcap != 0.0f ||
                ggml_flash_attn_ext_get_hint(output) != (tagged ?
                    GGML_HINT_FLASH_ATTN_QWEN35_D256_Q8_GQA6_VEC_PB1_DIAGNOSTIC : GGML_HINT_NONE)) {
            fail("FATTN PB1 production tuple shape/layout/op-parameter contract drift");
        }

        buffer.reset(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
        if (!buffer || !ggml_backend_supports_op(backend, output)) {
            fail("CUDA backend rejected FATTN PB1 diagnostic graph");
        }
        ggml_backend_tensor_set(k_storage, k_values.data(), 0, k_values.size());
        ggml_backend_tensor_set(v_storage, v_values.data(), 0, v_values.size());
        graph = ggml_new_graph_custom(ctx.get(), 32, false);
        ggml_build_forward_expand(graph, output);
    }

    std::vector<float> run(
            ggml_backend_t backend,
            const std::vector<float> & q_values,
            const std::vector<ggml_fp16_t> & mask_values) {
        if (q_values.size() != (size_t) FATTN_D * FATTN_H_Q * m ||
                mask_values.size() != (size_t) FATTN_N_KV * m) {
            fail("FATTN PB1 fixture cardinality mismatch");
        }
        ggml_backend_tensor_set(q_storage, q_values.data(), 0, q_values.size() * sizeof(float));
        ggml_backend_tensor_set(mask, mask_values.data(), 0, mask_values.size() * sizeof(ggml_fp16_t));
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            fail("FATTN PB1 CUDA graph compute failed");
        }
        ggml_backend_synchronize(backend);
        std::vector<float> result((size_t) FATTN_D * FATTN_H_Q * m);
        ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));
        require_finite(result, "fattn_vec_pb1");
        return result;
    }
};

void require_launch_observation(
        fattn_vec_launch_snapshot_fn snapshot, int q_cols, int ncols, int ntiles_x) {
    fattn_vec_launch_observation observed{};
    if (!snapshot(&observed, sizeof(observed)) || observed.abi_version != 1 ||
            observed.struct_size != sizeof(observed) || observed.evaluated != 1 ||
            observed.forced_parallel_blocks != 1 || observed.q_cols != q_cols ||
            observed.ncols != ncols || observed.ntiles_x != ntiles_x ||
            observed.ntiles_kv != 4 || observed.parallel_blocks != 1) {
        fail("FATTN PB1 launch geometry observation mismatch");
    }
}

void run_fattn_pb1_case(
        ggml_backend_t backend, const matrix_case & item,
        route_reset_fn reset, route_snapshot_fn snapshot,
        fattn_vec_launch_snapshot_fn launch_snapshot) {
    const auto k_values = make_fattn_q8_cache(5);
    const auto v_values = make_fattn_q8_cache(17);
    fattn_pb1_runner negative(backend, 3, false, k_values, v_values);
    std::array<std::vector<std::unique_ptr<fattn_pb1_runner>>, 6> scalar_runners;
    std::array<std::unique_ptr<fattn_pb1_runner>, 6> batch_runners;
    for (int m = 3; m <= 8; ++m) {
        auto & scalars = scalar_runners[(size_t) (m - 3)];
        scalars.reserve((size_t) m);
        for (int token = 0; token < m; ++token) {
            scalars.emplace_back(std::make_unique<fattn_pb1_runner>(backend, 1, true, k_values, v_values));
        }
        batch_runners[(size_t) (m - 3)] =
                std::make_unique<fattn_pb1_runner>(backend, m, true, k_values, v_values);
    }

    reset();
    (void) negative.run(backend, make_fattn_q(3), make_fattn_mask(3));
    route_counts negative_routes{};
    if (!snapshot(&negative_routes, sizeof(negative_routes)) ||
            negative_routes.abi_version != ROUTE_ABI || negative_routes.route_count != ROUTE_COUNT ||
            negative_routes.count[FATTN_QWEN35_D256_Q8_GQA6_VEC_COLS2_PB1] != 0 ||
            negative_routes.count[FATTN_VEC] != 0 || negative_routes.count[FATTN_MMA_F16] != 1 ||
            negative_routes.count[FATTN_TILE] != 0) {
        fail("untagged FATTN M3 negative control did not use only MMA_F16");
    }
    std::printf(
            "{\"type\":\"route_negative_control\",\"id\":\"%.*s\",\"m\":3,"
            "\"tagged\":false,\"pb1_candidate\":0,\"fattn_vec\":0,\"fattn_mma_f16\":1,"
            "\"fattn_tile\":0,\"content_free\":true}\n",
            (int) item.id.size(), item.id.data());
    std::fflush(stdout);

    for (int m = 3; m <= 8; ++m) {
        auto q_values = make_fattn_q(m);
        auto mask_values = make_fattn_mask(m);
        auto & scalars = scalar_runners[(size_t) (m - 3)];
        auto & batch = *batch_runners[(size_t) (m - 3)];
        std::vector<float> stable_scalar, stable_batch;
        reset();
        for (int replay = 0; replay < REPLAYS; ++replay) {
            std::vector<float> scalar_joined((size_t) FATTN_D * FATTN_H_Q * m);
            std::vector<float> batch_output;
            auto run_scalar = [&] {
                for (int token = 0; token < m; ++token) {
                    const std::vector<float> q_one(
                            q_values.begin() + (size_t) token * FATTN_D * FATTN_H_Q,
                            q_values.begin() + (size_t) (token + 1) * FATTN_D * FATTN_H_Q);
                    const std::vector<ggml_fp16_t> mask_one(
                            mask_values.begin() + (size_t) token * FATTN_N_KV,
                            mask_values.begin() + (size_t) (token + 1) * FATTN_N_KV);
                    const auto one = scalars[(size_t) token]->run(backend, q_one, mask_one);
                    require_launch_observation(launch_snapshot, 1, 1, 1);
                    std::copy(one.begin(), one.end(),
                            scalar_joined.begin() + (size_t) token * FATTN_D * FATTN_H_Q);
                }
            };
            auto run_batch = [&] {
                batch_output = batch.run(backend, q_values, mask_values);
                require_launch_observation(launch_snapshot, m, 2, (m + 1) / 2);
            };
            if ((replay & 1) == 0) { run_scalar(); run_batch(); }
            else { run_batch(); run_scalar(); }

            if (replay == 0) {
                route_counts routes{};
                if (!snapshot(&routes, sizeof(routes)) || routes.abi_version != ROUTE_ABI ||
                        routes.route_count != ROUTE_COUNT ||
                        routes.count[FATTN_QWEN35_D256_Q8_GQA6_VEC_COLS2_PB1] != (uint64_t) (m + 1) ||
                        routes.count[FATTN_VEC] != (uint64_t) (m + 1) ||
                        routes.count[FATTN_MMA_F16] != 0 || routes.count[FATTN_TILE] != 0) {
                    fail("FATTN PB1 route probe did not isolate every scalar row and batch graph");
                }
                for (size_t route_index = 0; route_index < ROUTE_COUNT; ++route_index) {
                    if (route_index != FATTN_QWEN35_D256_Q8_GQA6_VEC_COLS2_PB1 &&
                            route_index != FATTN_VEC && routes.count[route_index] != 0) {
                        fail("FATTN PB1 route probe was contaminated by an unrelated CUDA route");
                    }
                }
                std::printf(
                        "{\"type\":\"fattn_pb1_route_probe\",\"id\":\"%.*s\",\"m\":%d,"
                        "\"scalar_rows\":%d,\"scalar_q_cols\":1,\"scalar_ncols\":1,"
                        "\"scalar_ntiles_x\":1,\"scalar_ntiles_kv\":4,\"scalar_parallel_blocks\":1,"
                        "\"batch_q_cols\":%d,\"batch_ncols\":2,\"batch_ntiles_x\":%d,"
                        "\"batch_ntiles_kv\":4,\"batch_parallel_blocks\":1,"
                        "\"pb1_candidate\":%d,\"fattn_vec\":%d,\"fattn_mma_f16\":0,"
                        "\"fattn_tile\":0,\"causal_mask_attested\":true,\"content_free\":true}\n",
                        (int) item.id.size(), item.id.data(), m, m, m, (m + 1) / 2, m + 1, m + 1);
                std::fflush(stdout);
            }
            require_bits_equal(scalar_joined, batch_output, item.id, replay);
            if (replay == 0) {
                stable_scalar = scalar_joined;
                stable_batch = batch_output;
            } else {
                require_bits_equal(stable_scalar, scalar_joined, item.id, replay);
                require_bits_equal(stable_batch, batch_output, item.id, replay);
            }
        }
        std::printf(
                "{\"type\":\"width_route\",\"id\":\"%.*s\",\"m\":%d,"
                "\"replays\":8,\"finite\":true,\"comparison\":\"bitwise_f32\","
                "\"scalar_rows_independent\":true,\"causal_mask_attested\":true}\n",
                (int) item.id.size(), item.id.data(), m);
    }
    std::printf(
            "{\"type\":\"case\",\"id\":\"%.*s\",\"status\":\"passed\","
            "\"device_scope\":\"sm120\",\"d\":256,\"n_kv\":1024,\"h_q\":24,"
            "\"h_kv\":4,\"gqa\":6,\"q_type\":\"f32\",\"k_type\":\"q8_0\","
            "\"v_type\":\"q8_0\",\"mask_type\":\"f16\",\"causal_mask_attested\":true,"
            "\"scale\":0.0625,\"max_bias\":0.0,\"softcap\":0.0,\"prec\":\"f32\","
            "\"scalar_m\":1,\"min_batch_m\":3,\"max_batch_m\":8,\"widths\":6,"
            "\"replays\":8,\"ncols_scalar\":1,\"ncols_batch\":2,"
            "\"forced_parallel_blocks\":1,\"finite\":true,\"comparison\":\"bitwise_f32\"}\n",
            (int) item.id.size(), item.id.data());
}

void validate_routes(const matrix_case & item, const route_counts & counts) {
    if (counts.abi_version != ROUTE_ABI || counts.route_count != ROUTE_COUNT) fail("CUDA route-marker ABI mismatch");
    auto require_route = [&](route r, const char * name) {
        if (counts.count[r] == 0) fail(std::string(item.id) + ": mandatory CUDA route missing: " + name);
    };
    auto require_zero = [&](route r, const char * name) {
        if (counts.count[r] != 0) fail(std::string(item.id) + ": unrelated CUDA route was selected: " + name);
    };
    auto require_only = [&](std::initializer_list<route> selected) {
        for (size_t index = 0; index < ROUTE_COUNT; ++index) {
            const route current = (route) index;
            const bool expected = std::find(selected.begin(), selected.end(), current) != selected.end();
            if (expected && counts.count[index] == 0) {
                fail(std::string(item.id) + ": mandatory isolated CUDA route was not selected");
            }
            if (!expected && counts.count[index] != 0) {
                fail(std::string(item.id) + ": unrelated CUDA route contaminated isolated evidence");
            }
        }
    };
    if (item.op == operation::fp8) {
        require_route(FP8, "fp8_e4m3");
        require_route(FP8_BATCH, "fp8_e4m3_batch");
        if (counts.count[FP8] <= counts.count[FP8_BATCH]) {
            fail("FP8 diagnostic did not prove distinct scalar and batched native routes");
        }
        require_zero(NVFP4_MMVQ_M1, "nvfp4_mmvq_m1");
        require_zero(NVFP4_MMVQ_M8, "nvfp4_mmvq_m8");
        require_zero(NVFP4_FUSED_FFN, "nvfp4_fused_ffn");
        require_zero(NVFP4_ROW_INVARIANT_HEAD_SCALED, "nvfp4_row_invariant_head_scaled");
        require_zero(NVFP4_ROW_INVARIANT_FFN_FUSED, "nvfp4_row_invariant_ffn_fused");
        require_zero(NVFP4_ROW_INVARIANT_FFN_DOWN_SCALED, "nvfp4_row_invariant_ffn_down_scaled");
        require_zero(GDN, "gdn");
        require_zero(GDN_FUSED_CACHE, "gdn_fused_cache");
        require_zero(RMS_NORM, "rms_norm");
        require_zero(RMS_NORM_MUL, "rms_norm_mul");
        require_zero(SSM_CONV, "ssm_conv");
        require_zero(SSM_CONV_SILU, "ssm_conv_silu");
        require_zero(L2_NORM, "l2_norm");
        require_zero(SILU, "silu");
        require_zero(SILU_MUL, "silu_mul");
        require_zero(BF16_MMVF, "bf16_mmvf");
        require_zero(BF16_CUBLAS, "bf16_cublas");
    }
    if (item.op == operation::nvfp4_raw) {
        require_route(NVFP4_MMVQ_M1, "nvfp4_mmvq_m1");
        require_route(NVFP4_MMVQ_M8, "nvfp4_mmvq_m8");
    }
    if (item.op == operation::nvfp4_ffn) {
        require_route(NVFP4_FUSED_FFN, "nvfp4_m1_fused_ffn");
        require_route(NVFP4_MMVQ_M8, "nvfp4_m8_unfused_mmvq");
    }
    if (item.op == operation::nvfp4_head) {
        require_route(NVFP4_MMVQ_M1, "unchanged_nvfp4_m1");
        require_route(NVFP4_ROW_INVARIANT_HEAD_SCALED, "tagged_nvfp4_row_invariant_head_scaled");
        if (counts.count[NVFP4_MMVQ_M8] != 0) {
            fail("tagged NVFP4 LM-head M8 fell through to the generic M8 route");
        }
    }
    if (item.op == operation::nvfp4_ffn_row_invariant) {
        require_route(NVFP4_FUSED_FFN, "unchanged_nvfp4_m1_fused_ffn");
        require_route(NVFP4_ROW_INVARIANT_FFN_FUSED, "tagged_nvfp4_ffn_gate_up_swiglu");
        if (counts.count[NVFP4_MMVQ_M8] != 0) fail("tagged NVFP4 FFN fell through to generic M8 route");
    }
    if (item.op == operation::nvfp4_down_row_invariant) {
        require_route(NVFP4_MMVQ_M1, "unchanged_nvfp4_m1_down");
        require_route(NVFP4_ROW_INVARIANT_FFN_DOWN_SCALED, "tagged_nvfp4_ffn_down_scaled");
        if (counts.count[NVFP4_MMVQ_M8] != 0) fail("tagged NVFP4 FFN down fell through to generic M8 route");
    }
    if (item.op == operation::gdn) {
        require_route(GDN_FUSED_CACHE, "gated_delta_net_fused_cache");
        require_zero(FP8, "fp8_e4m3");
        require_zero(FP8_BATCH, "fp8_e4m3_batch");
        require_zero(NVFP4_MMVQ_M1, "nvfp4_mmvq_m1");
        require_zero(NVFP4_MMVQ_M8, "nvfp4_mmvq_m8");
        require_zero(NVFP4_FUSED_FFN, "nvfp4_fused_ffn");
        require_zero(NVFP4_ROW_INVARIANT_HEAD_SCALED, "nvfp4_row_invariant_head_scaled");
        require_zero(NVFP4_ROW_INVARIANT_FFN_FUSED, "nvfp4_row_invariant_ffn_fused");
        require_zero(NVFP4_ROW_INVARIANT_FFN_DOWN_SCALED, "nvfp4_row_invariant_ffn_down_scaled");
        require_zero(GDN, "gated_delta_net_unfused");
        require_zero(RMS_NORM, "rms_norm");
        require_zero(RMS_NORM_MUL, "rms_norm_mul");
        require_zero(SSM_CONV, "ssm_conv");
        require_zero(SSM_CONV_SILU, "ssm_conv_silu");
        require_zero(L2_NORM, "l2_norm");
        require_zero(SILU, "silu");
        require_zero(SILU_MUL, "silu_mul");
        require_zero(BF16_MMVF, "bf16_mmvf");
        require_zero(BF16_CUBLAS, "bf16_cublas");
    }
    if (item.op == operation::rms_norm) require_only({ RMS_NORM_MUL });
    if (item.op == operation::ssm_conv) require_only({ SSM_CONV_SILU });
    if (item.op == operation::l2_norm) require_only({ L2_NORM });
    if (item.op == operation::gated_norm) require_only({ RMS_NORM_MUL, SILU_MUL });
    if (item.op == operation::bf16_projection) require_only({ BF16_MMVF, BF16_CUBLAS });
    if (item.op == operation::bf16_projection_candidate) {
        require_only({ BF16_MMVF, BF16_CUBLAS,
                item.id.find("alpha") != std::string_view::npos ?
                    BF16_ROW_INVARIANT_ALPHA : BF16_ROW_INVARIANT_BETA });
    }
    std::printf("{\"type\":\"routes\",\"id\":\"%.*s\",\"fp8\":%llu,\"fp8_batch\":%llu,\"nvfp4_mmvq_m1\":%llu,\"nvfp4_mmvq_m8\":%llu,\"nvfp4_fused_ffn\":%llu,\"nvfp4_row_invariant_head_scaled\":%llu,\"nvfp4_row_invariant_ffn_fused\":%llu,\"nvfp4_row_invariant_ffn_down_scaled\":%llu,\"gdn\":%llu,\"gdn_fused_cache\":%llu,\"rms_norm\":%llu,\"rms_norm_mul\":%llu,\"ssm_conv\":%llu,\"ssm_conv_silu\":%llu,\"l2_norm\":%llu,\"silu\":%llu,\"silu_mul\":%llu,\"bf16_mmvf\":%llu,\"bf16_cublas\":%llu,\"bf16_row_invariant_beta\":%llu,\"bf16_row_invariant_alpha\":%llu}\n",
            (int) item.id.size(), item.id.data(),
            (unsigned long long) counts.count[FP8], (unsigned long long) counts.count[FP8_BATCH],
            (unsigned long long) counts.count[NVFP4_MMVQ_M1],
            (unsigned long long) counts.count[NVFP4_MMVQ_M8], (unsigned long long) counts.count[NVFP4_FUSED_FFN],
            (unsigned long long) counts.count[NVFP4_ROW_INVARIANT_HEAD_SCALED],
            (unsigned long long) counts.count[NVFP4_ROW_INVARIANT_FFN_FUSED],
            (unsigned long long) counts.count[NVFP4_ROW_INVARIANT_FFN_DOWN_SCALED],
            (unsigned long long) counts.count[GDN], (unsigned long long) counts.count[GDN_FUSED_CACHE],
            (unsigned long long) counts.count[RMS_NORM], (unsigned long long) counts.count[RMS_NORM_MUL],
            (unsigned long long) counts.count[SSM_CONV], (unsigned long long) counts.count[SSM_CONV_SILU],
            (unsigned long long) counts.count[L2_NORM], (unsigned long long) counts.count[SILU],
            (unsigned long long) counts.count[SILU_MUL],
            (unsigned long long) counts.count[BF16_MMVF], (unsigned long long) counts.count[BF16_CUBLAS],
            (unsigned long long) counts.count[BF16_ROW_INVARIANT_BETA],
            (unsigned long long) counts.count[BF16_ROW_INVARIANT_ALPHA]);
}

} // namespace

int main(int argc, char ** argv) {
    try {
        bool cuda = false, full = false, head_candidate = false, projection_candidate = false;
        bool fp8_only = false, gdn_only = false, rms_only = false, ssm_conv_only = false;
        bool l2_only = false, gated_norm_only = false, bf16_only = false, bf16_candidate = false;
        bool fattn_pb1 = false, list = false;
        int matrix_mode_arguments = 0;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--cuda") cuda = true;
            else if (arg == "--full") { full = true; ++matrix_mode_arguments; }
            else if (arg == "--quick") { ++matrix_mode_arguments; }
            else if (arg == "--head-candidate") { head_candidate = true; ++matrix_mode_arguments; }
            else if (arg == "--projection-candidate") { projection_candidate = true; ++matrix_mode_arguments; }
            else if (arg == "--fp8") { fp8_only = true; ++matrix_mode_arguments; }
            else if (arg == "--gdn") { gdn_only = true; ++matrix_mode_arguments; }
            else if (arg == "--rms") { rms_only = true; ++matrix_mode_arguments; }
            else if (arg == "--ssm-conv") { ssm_conv_only = true; ++matrix_mode_arguments; }
            else if (arg == "--l2") { l2_only = true; ++matrix_mode_arguments; }
            else if (arg == "--gated-norm") { gated_norm_only = true; ++matrix_mode_arguments; }
            else if (arg == "--bf16-projections") { bf16_only = true; ++matrix_mode_arguments; }
            else if (arg == "--bf16-candidate") { bf16_candidate = true; ++matrix_mode_arguments; }
            else if (arg == "--fattn-vec-pb1") { fattn_pb1 = true; ++matrix_mode_arguments; }
            else if (arg == "--list" || arg == "--validate-matrix") list = true;
            else fail("unknown argument: " + arg);
        }
        if (matrix_mode_arguments > 1) {
            fail("select exactly one matrix mode: --quick, --full, --head-candidate, --projection-candidate, --fp8, --gdn, --rms, --ssm-conv, --l2, --gated-norm, --bf16-projections, --bf16-candidate, or --fattn-vec-pb1");
        }
        if (cuda && list) {
            fail("--cuda cannot be combined with the CPU-only --list/--validate-matrix mode");
        }
        validate_cancellation_sign_contract();
        const std::vector<matrix_case> cases = fp8_only ? select_operation(operation::fp8) :
            gdn_only ? select_operation(operation::gdn) : head_candidate ?
            std::vector<matrix_case>{ HEAD_CANDIDATE } : projection_candidate ?
            std::vector<matrix_case>(PROJECTION_CANDIDATE.begin(), PROJECTION_CANDIDATE.end()) :
            rms_only ? std::vector<matrix_case>{ RMS_NORM_CASE } :
            ssm_conv_only ? std::vector<matrix_case>{ SSM_CONV_CASE } :
            l2_only ? std::vector<matrix_case>(L2_NORM_CASES.begin(), L2_NORM_CASES.end()) :
            gated_norm_only ? std::vector<matrix_case>{ GATED_NORM_CASE } :
            bf16_only ? std::vector<matrix_case>(BF16_PROJECTION_CASES.begin(), BF16_PROJECTION_CASES.end()) :
            bf16_candidate ? std::vector<matrix_case>(BF16_PROJECTION_CANDIDATE_CASES.begin(), BF16_PROJECTION_CANDIDATE_CASES.end()) :
            fattn_pb1 ? std::vector<matrix_case>{ FATTN_VEC_COLS2_PB1_CASE } :
            select_matrix(full);
        const char * mode = fp8_only ? "fp8" : gdn_only ? "gdn" : head_candidate ? "head-candidate" :
                projection_candidate ? "projection-candidate" : rms_only ? "rms" :
                ssm_conv_only ? "ssm-conv" : l2_only ? "l2" : gated_norm_only ? "gated-norm" :
                bf16_only ? "bf16-projections" : bf16_candidate ? "bf16-candidate" :
                fattn_pb1 ? "fattn-vec-pb1" : full ? "full" : "quick";
        if (!cuda) {
            if (!list) fail("CUDA execution is opt-in: pass --cuda under the exclusive guard, or use --validate-matrix");
            for (const auto & item : cases) {
                std::printf("{\"type\":\"matrix_case\",\"id\":\"%.*s\",\"k\":%lld,\"n\":%lld}\n",
                        (int) item.id.size(), item.id.data(), (long long) item.k, (long long) item.n);
            }
            std::printf("{\"type\":\"matrix\",\"mode\":\"%s\",\"cases\":%zu,\"replays\":8,\"comparison\":\"bitwise_f32\",\"cuda_initialized\":false}\n",
                    mode, cases.size());
            return 0;
        }
        auto env = [](const char * name) { const char * value = std::getenv(name); return value ? value : ""; };
        if (std::string(env("LLAMACPP_QWEN38_ROW_INVARIANCE_GUARD")) != "EXCLUSIVE_GPU" ||
                std::string(env("AI_LOADER_EXCLUSIVE_GPU_GUARD")) != "1" ||
                *env("AI_LOADER_EXCLUSIVE_GPU_UUID") == '\0' ||
                *env("AI_LOADER_EXCLUSIVE_GPU_LEASE_PATH") == '\0' ||
                *env("AI_LOADER_EXCLUSIVE_GPU_LEASE_NONCE") == '\0' ||
                *env("AI_LOADER_EXCLUSIVE_GPU_OWNER_PID") == '\0' ||
                *env("AI_LOADER_EXCLUSIVE_GPU_JOB_NAME") == '\0' ||
                std::string(env("CUDA_DEVICE_ORDER")) != "PCI_BUS_ID" ||
                std::string(env("CUDA_VISIBLE_DEVICES")) != env("AI_LOADER_EXCLUSIVE_GPU_UUID")) {
            fail("missing or inconsistent exclusive GPU lease/UUID/owner/job contract");
        }

        ggml_backend_load_all();
        ggml_backend_ptr backend(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr));
        if (!backend) fail("failed to initialize CUDA backend");
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend.get()));
        auto reset = (route_reset_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_diagnostic_route_reset");
        auto snapshot = (route_snapshot_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_diagnostic_route_snapshot_v2");
        if (!reset || !snapshot) fail("mandatory CUDA route-marker functions are unavailable");
        auto launch_snapshot = (fattn_vec_launch_snapshot_fn) ggml_backend_reg_get_proc_address(
                reg, "ggml_backend_cuda_diagnostic_fattn_vec_launch_snapshot_v1");
        if (fattn_pb1 && !launch_snapshot) fail("mandatory FATTN VEC launch-snapshot function is unavailable");

        const bool recurrent_operator_mode = rms_only || ssm_conv_only || l2_only || gated_norm_only || bf16_only || bf16_candidate;
        const char * fixture = fattn_pb1 ? "deterministic_qwen35_fattn_exact" :
                recurrent_operator_mode ? "deterministic_nonzero_shape_exact" : "deterministic_cancellation_sensitive";
        std::printf("{\"type\":\"config\",\"mode\":\"%s\",\"cases\":%zu,\"replays\":8,\"order\":\"alternating\",\"fixture\":\"%s\",\"comparison\":\"bitwise_f32\"}\n",
                mode, cases.size(), fixture);
        for (const auto & item : cases) {
            if (item.op == operation::nvfp4_head) {
                run_head_candidate(backend.get(), item, reset, snapshot);
            } else if (item.op == operation::nvfp4_ffn_row_invariant ||
                    item.op == operation::nvfp4_down_row_invariant) {
                run_projection_candidate(backend.get(), item, reset, snapshot);
            } else if (fp8_only) {
                run_fp8_case(backend.get(), item, reset, snapshot);
            } else if (item.op == operation::rms_norm || item.op == operation::l2_norm ||
                    item.op == operation::gated_norm) {
                reset();
                run_norm_case(backend.get(), item, snapshot);
            } else if (item.op == operation::ssm_conv) {
                reset();
                run_ssm_conv_case(backend.get(), item, snapshot);
            } else if (item.op == operation::bf16_projection) {
                reset();
                run_bf16_case(backend.get(), item, snapshot);
            } else if (item.op == operation::bf16_projection_candidate) {
                reset();
                run_bf16_candidate_case(backend.get(), item, reset, snapshot);
            } else if (item.op == operation::fattn_vec_cols2_pb1) {
                run_fattn_pb1_case(backend.get(), item, reset, snapshot, launch_snapshot);
                continue;
            } else {
                reset();
                if (item.op == operation::gdn) {
                    if (item.id == "gdn-k1-vs-k8") run_gdn_slot_case(backend.get(), item);
                    else run_gdn_sequence_case(backend.get(), item);
                } else {
                    run_dense_case(backend.get(), item);
                }
            }
            route_counts counts{};
            if (!snapshot(&counts, sizeof(counts))) fail("CUDA route-marker snapshot ABI rejected caller size");
            validate_routes(item, counts);
        }
        std::printf("{\"type\":\"complete\",\"status\":\"passed\",\"mode\":\"%s\",\"cases\":%zu}\n",
                mode, cases.size());
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "row-invariance diagnostic FAILED: %s\n", error.what());
        return 1;
    }
}
