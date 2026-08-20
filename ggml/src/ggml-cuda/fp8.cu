#include "fp8.cuh"
#include "diagnostics.cuh"

#if defined(GGML_CUDA_HAS_CUBLASLT) && CUDART_VERSION >= 12080

#include <cuda_fp8.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace {

constexpr size_t FP8_MAX_WORKSPACE = 32u * 1024u * 1024u;
constexpr uint32_t FP8_MATRIX_ALIGNMENT = 128;

struct fp8_key {
    ggml_backend_cuda_context * ctx;
    int64_t k;
    int64_t n;
    int64_t m;
    const void * weight_scale;
    const void * input_scale;

    bool operator==(const fp8_key & other) const {
        return ctx == other.ctx && k == other.k && n == other.n && m == other.m &&
            weight_scale == other.weight_scale && input_scale == other.input_scale;
    }
};

struct fp8_key_hash {
    size_t operator()(const fp8_key & key) const {
        size_t h = std::hash<void *>{}(key.ctx);
        const auto mix = [&h](size_t v) { h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2); };
        mix(std::hash<int64_t>{}(key.k));
        mix(std::hash<int64_t>{}(key.n));
        mix(std::hash<int64_t>{}(key.m));
        mix(std::hash<const void *>{}(key.weight_scale));
        mix(std::hash<const void *>{}(key.input_scale));
        return h;
    }
};

struct fp8_shape_key {
    ggml_backend_cuda_context * ctx;
    int64_t k;
    int64_t n;
    int64_t m;

    bool operator==(const fp8_shape_key & other) const {
        return ctx == other.ctx && k == other.k && n == other.n && m == other.m;
    }
};

struct fp8_shape_key_hash {
    size_t operator()(const fp8_shape_key & key) const {
        size_t h = std::hash<void *>{}(key.ctx);
        const auto mix = [&h](size_t v) { h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2); };
        mix(std::hash<int64_t>{}(key.k));
        mix(std::hash<int64_t>{}(key.n));
        mix(std::hash<int64_t>{}(key.m));
        return h;
    }
};

struct fp8_algo {
    cublasLtMatmulAlgo_t algo{};
    size_t workspace_size = 0;
};

struct fp8_plan {
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t a = nullptr;
    cublasLtMatrixLayout_t b = nullptr;
    cublasLtMatrixLayout_t d = nullptr;
    cublasLtMatmulAlgo_t algo{};
    size_t workspace_size = 0;

    ~fp8_plan() {
        if (d) { cublasLtMatrixLayoutDestroy(d); }
        if (b) { cublasLtMatrixLayoutDestroy(b); }
        if (a) { cublasLtMatrixLayoutDestroy(a); }
        if (op) { cublasLtMatmulDescDestroy(op); }
    }
};

std::mutex fp8_cache_mutex;
std::unordered_map<fp8_key, std::unique_ptr<fp8_plan>, fp8_key_hash> fp8_cache;
std::unordered_map<fp8_shape_key, fp8_algo, fp8_shape_key_hash> fp8_algo_cache;
std::atomic<uint64_t> fp8_calls{0};
std::atomic<uint64_t> fp8_plan_hits{0};
std::atomic<uint64_t> fp8_plan_misses{0};
std::atomic<uint64_t> fp8_algo_hits{0};
std::atomic<uint64_t> fp8_algo_misses{0};
std::atomic<uint64_t> fp8_m1{0};
std::atomic<uint64_t> fp8_m4{0};
std::atomic<uint64_t> fp8_m8{0};
std::atomic<uint64_t> fp8_m32{0};
std::atomic<uint64_t> fp8_m_large{0};
std::atomic<bool> fp8_route_logged{false};

template <typename T>
void lt_set(cublasLtMatmulDesc_t desc, cublasLtMatmulDescAttributes_t attr, const T & value) {
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(desc, attr, &value, sizeof(value)));
}

std::unique_ptr<fp8_plan> make_plan(ggml_backend_cuda_context & ctx, const fp8_key & key) {
    auto plan = std::make_unique<fp8_plan>();

    CUBLAS_CHECK(cublasLtMatmulDescCreate(&plan->op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const cublasOperation_t trans_a = CUBLAS_OP_T;
    const cublasOperation_t trans_b = CUBLAS_OP_N;
    lt_set(plan->op, CUBLASLT_MATMUL_DESC_TRANSA, trans_a);
    lt_set(plan->op, CUBLASLT_MATMUL_DESC_TRANSB, trans_b);

    // Attributes store device pointer values. Scale addresses are therefore
    // part of fp8_key and cached descriptors are immutable/thread-safe.
    const void * a_scale = key.weight_scale;
    const void * b_scale = key.input_scale;
    lt_set(plan->op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, a_scale);
    lt_set(plan->op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, b_scale);
    const cublasLtMatmulMatrixScale_t scalar_mode = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
    lt_set(plan->op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, scalar_mode);
    lt_set(plan->op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, scalar_mode);

    // GGML weight rows [N,K] are a column-major [K,N] view. TN computes
    // [N,K] x [K,M] -> [N,M] without copies or hidden transposes.
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan->a, CUDA_R_8F_E4M3, key.k, key.n, key.k));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan->b, CUDA_R_8F_E4M3, key.k, key.m, key.k));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan->d, CUDA_R_32F, key.n, key.m, key.n));

    // Scale pointers affect the immutable operation descriptor but not
    // algorithm selection. Cache the expensive heuristic by shape so 208
    // projection-specific scalar pairs do not repeat identical searches.
    const fp8_shape_key shape_key{key.ctx, key.k, key.n, key.m};
    const auto algo_it = fp8_algo_cache.find(shape_key);
    if (algo_it != fp8_algo_cache.end()) {
        plan->algo = algo_it->second.algo;
        plan->workspace_size = algo_it->second.workspace_size;
        fp8_algo_hits.fetch_add(1, std::memory_order_relaxed);
    } else {
        cublasLtMatmulPreference_t preference = nullptr;
        CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
        CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
            preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &FP8_MAX_WORKSPACE, sizeof(FP8_MAX_WORKSPACE)));
        // GGML's CUDA tensor allocator guarantees 128-byte alignment, not
        // cuBLASLt's 256-byte default preference.  Constrain the heuristic to
        // algorithms valid for the actual allocator contract.
        CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
            preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
            &FP8_MATRIX_ALIGNMENT, sizeof(FP8_MATRIX_ALIGNMENT)));
        CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
            preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
            &FP8_MATRIX_ALIGNMENT, sizeof(FP8_MATRIX_ALIGNMENT)));
        CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
            preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
            &FP8_MATRIX_ALIGNMENT, sizeof(FP8_MATRIX_ALIGNMENT)));
        CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
            preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES,
            &FP8_MATRIX_ALIGNMENT, sizeof(FP8_MATRIX_ALIGNMENT)));

        cublasLtMatmulHeuristicResult_t heuristic{};
        int returned = 0;
        const cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(
            // CUDA >= 13.0 Update 2 requires the exact same layout descriptor
            // handle when C and D alias, even if two descriptors describe an
            // identical layout. beta is zero, but pass D as both descriptors
            // to honor that contract.
            ctx.cublaslt_handle(), plan->op, plan->a, plan->b, plan->d, plan->d,
            preference, 1, &heuristic, &returned);
        cublasLtMatmulPreferenceDestroy(preference);
        CUBLAS_CHECK(status);
        if (returned != 1 || heuristic.state != CUBLAS_STATUS_SUCCESS) {
            GGML_ABORT(
                "No native cuBLASLt E4M3 TN algorithm for K=%lld N=%lld M=%lld. "
                "Convert a separate model with --fp8-as-q8 for portable fallback.",
                (long long) key.k, (long long) key.n, (long long) key.m);
        }
        plan->algo = heuristic.algo;
        plan->workspace_size = heuristic.workspaceSize;
        fp8_algo_cache.emplace(shape_key, fp8_algo{heuristic.algo, heuristic.workspaceSize});
        fp8_algo_misses.fetch_add(1, std::memory_order_relaxed);
    }
    return plan;
}

__global__ void quantize_activation_f8_e4m3(
        const float * x, __nv_fp8_e4m3 * y, int64_t n, const float * input_scale) {
    const float scale = *input_scale;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t) blockDim.x * gridDim.x) {
        y[i] = __nv_fp8_e4m3(x[i] / scale);
    }
}

} // namespace

bool ggml_cuda_mul_mat_f8_e4m3(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * weight,
        const ggml_tensor * activation,
        ggml_tensor * dst) {
    if (weight->type != GGML_TYPE_F8_E4M3) {
        return false;
    }

#if CUDART_VERSION < 13020
    // CUDA 13.2 fixed a GeForce FP8-input cuBLASLt issue that could cause
    // illegal memory accesses.  Keep 13.1 useful for compile-only prototype
    // validation, but fail closed before any FP8 GPU work is submitted.
    GGML_ABORT("Native GeForce F8_E4M3 requires a CUDA Toolkit 13.2 or newer build");
#endif

    const ggml_tensor * weight_scale = dst->src[2];
    const ggml_tensor * input_scale  = dst->src[3];
    GGML_ASSERT(weight_scale && input_scale);
    GGML_ASSERT(activation->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(weight_scale->type == GGML_TYPE_F32 && input_scale->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_scalar(weight_scale) && ggml_is_scalar(input_scale));
    GGML_ASSERT(ggml_is_contiguous(weight) && ggml_is_contiguous(activation) && ggml_is_contiguous(dst));
    GGML_ASSERT(weight->ne[2] == 1 && weight->ne[3] == 1);

    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (cc < GGML_CUDA_CC_BLACKWELL || cc >= 1300 ||
        ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_BLACKWELL) {
        GGML_ABORT("F8_E4M3 requires NVIDIA SM120; use a separate --fp8-as-q8 GGUF");
    }

    const int64_t k = weight->ne[0];
    const int64_t n = weight->ne[1];
    const int64_t m = ggml_nrows(activation);
    GGML_ASSERT(activation->ne[0] == k && dst->ne[0] == n && ggml_nrows(dst) == m);
    if (k % 16 != 0 || n % 16 != 0 || m <= 0) {
        GGML_ABORT("Unsupported F8_E4M3 shape K=%lld N=%lld M=%lld",
                   (long long) k, (long long) n, (long long) m);
    }
    if ((reinterpret_cast<uintptr_t>(weight_scale->data) & 15u) != 0 ||
        (reinterpret_cast<uintptr_t>(input_scale->data) & 15u) != 0) {
        GGML_ABORT("cuBLASLt FP8 scalar scale addresses must be 16-byte aligned");
    }

    cudaStream_t stream = ctx.stream();
    cudaStreamCaptureStatus capture_status;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture_status));
    const fp8_key key{&ctx, k, n, m, weight_scale->data, input_scale->data};
    fp8_plan * plan = nullptr;
    {
        std::lock_guard<std::mutex> lock(fp8_cache_mutex);
        auto it = fp8_cache.find(key);
        if (it == fp8_cache.end()) {
            if (capture_status != cudaStreamCaptureStatusNone) {
                GGML_ABORT("F8_E4M3 cuBLASLt plan cache miss during CUDA graph capture");
            }
            it = fp8_cache.emplace(key, make_plan(ctx, key)).first;
            fp8_plan_misses.fetch_add(1, std::memory_order_relaxed);
        } else {
            fp8_plan_hits.fetch_add(1, std::memory_order_relaxed);
        }
        plan = it->second.get();
    }

    ggml_cuda_pool_alloc<__nv_fp8_e4m3> activation_f8(ctx.pool(), k * m);
    // cuBLASLt requires 256-byte workspace alignment while GGML's VMM pool
    // only guarantees 128. Keep the owning base allocation alive and round a
    // deliberately over-allocated interior pointer upward.
    ggml_cuda_pool_alloc<uint8_t> workspace(ctx.pool(), plan->workspace_size + 255);
    void * workspace_aligned = reinterpret_cast<void *>(
        (reinterpret_cast<uintptr_t>(workspace.get()) + 255u) & ~uintptr_t(255u));
    if ((reinterpret_cast<uintptr_t>(weight->data) & (FP8_MATRIX_ALIGNMENT - 1)) != 0 ||
        (reinterpret_cast<uintptr_t>(activation_f8.get()) & (FP8_MATRIX_ALIGNMENT - 1)) != 0 ||
        (reinterpret_cast<uintptr_t>(dst->data) & (FP8_MATRIX_ALIGNMENT - 1)) != 0) {
        GGML_ABORT("F8_E4M3 matrix pointers do not satisfy GGML's 128-byte CUDA alignment contract");
    }
    const int threads = 256;
    const int blocks = (int) std::min<int64_t>((k * m + threads - 1) / threads, 65535);
    quantize_activation_f8_e4m3<<<blocks, threads, 0, stream>>>(
        (const float *) activation->data, activation_f8.get(), k * m,
        (const float *) input_scale->data);
    CUDA_CHECK(cudaGetLastError());

    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_CHECK(cublasLtMatmul(
        ctx.cublaslt_handle(), plan->op, &alpha,
        weight->data, plan->a, activation_f8.get(), plan->b, &beta,
        dst->data, plan->d, dst->data, plan->d, &plan->algo,
        workspace_aligned, plan->workspace_size, stream));
    GGML_CUDA_DIAGNOSTIC_ROUTE_HIT(GGML_CUDA_DIAGNOSTIC_ROUTE_FP8_E4M3);
    if (m > 1) {
        GGML_CUDA_DIAGNOSTIC_ROUTE_HIT(GGML_CUDA_DIAGNOSTIC_ROUTE_FP8_E4M3_BATCH);
    }
    fp8_calls.fetch_add(1, std::memory_order_relaxed);
    if (m == 1) {
        fp8_m1.fetch_add(1, std::memory_order_relaxed);
    } else if (m <= 4) {
        fp8_m4.fetch_add(1, std::memory_order_relaxed);
    } else if (m <= 8) {
        fp8_m8.fetch_add(1, std::memory_order_relaxed);
    } else if (m <= 32) {
        fp8_m32.fetch_add(1, std::memory_order_relaxed);
    } else {
        fp8_m_large.fetch_add(1, std::memory_order_relaxed);
    }
    if (!fp8_route_logged.exchange(true, std::memory_order_relaxed)) {
        GGML_LOG_INFO(
            "ggml_cuda: native ModelOpt E4M3 W8A8 route active "
            "(cuBLASLt TN, FP32 accumulation/output, static scalar scales)\n");
    }
    return true;
}

void ggml_cuda_fp8_cache_clear(ggml_backend_cuda_context * ctx) {
    std::lock_guard<std::mutex> lock(fp8_cache_mutex);
    size_t removed = 0;
    size_t algos_removed = 0;
    for (auto it = fp8_cache.begin(); it != fp8_cache.end();) {
        if (it->first.ctx == ctx) {
            it = fp8_cache.erase(it);
            ++removed;
        } else {
            ++it;
        }
    }
    for (auto it = fp8_algo_cache.begin(); it != fp8_algo_cache.end();) {
        if (it->first.ctx == ctx) {
            it = fp8_algo_cache.erase(it);
            ++algos_removed;
        } else {
            ++it;
        }
    }
    if (removed > 0) {
        GGML_LOG_INFO(
            "ggml_cuda: E4M3 W8A8 stats: context_plans=%zu context_algos=%zu "
            "process_totals{calls=%llu plan_hits=%llu plan_misses=%llu algo_hits=%llu algo_misses=%llu "
            "M{1=%llu,2-4=%llu,5-8=%llu,9-32=%llu,>32=%llu}}\n",
            removed, algos_removed,
            (unsigned long long) fp8_calls.load(std::memory_order_relaxed),
            (unsigned long long) fp8_plan_hits.load(std::memory_order_relaxed),
            (unsigned long long) fp8_plan_misses.load(std::memory_order_relaxed),
            (unsigned long long) fp8_algo_hits.load(std::memory_order_relaxed),
            (unsigned long long) fp8_algo_misses.load(std::memory_order_relaxed),
            (unsigned long long) fp8_m1.load(std::memory_order_relaxed),
            (unsigned long long) fp8_m4.load(std::memory_order_relaxed),
            (unsigned long long) fp8_m8.load(std::memory_order_relaxed),
            (unsigned long long) fp8_m32.load(std::memory_order_relaxed),
            (unsigned long long) fp8_m_large.load(std::memory_order_relaxed));
    }
}

#else

bool ggml_cuda_mul_mat_f8_e4m3(
        ggml_backend_cuda_context &, const ggml_tensor * weight,
        const ggml_tensor *, ggml_tensor *) {
    if (weight->type == GGML_TYPE_F8_E4M3) {
        GGML_ABORT("F8_E4M3 requires NVIDIA CUDA Toolkit >= 12.8 with cuBLASLt");
    }
    return false;
}

void ggml_cuda_fp8_cache_clear(ggml_backend_cuda_context *) {}

#endif
