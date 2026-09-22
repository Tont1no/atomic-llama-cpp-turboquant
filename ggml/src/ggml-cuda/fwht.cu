#include "common.cuh"
#include "fwht.cuh"
#include "../ggml-turbo4.h"

static __constant__ const int8_t fwht_turbo_signs1[GGML_TURBO4_QK] = GGML_TURBO4_SIGNS1_INIT;
static __constant__ const int8_t fwht_turbo_signs2[GGML_TURBO4_QK] = GGML_TURBO4_SIGNS2_INIT;

template <int N, bool TURBO = false, bool INVERSE = false>
__launch_bounds__(4*ggml_cuda_get_physical_warp_size(), 1)
__global__ void fwht_cuda(const float * src, float * dst, const int64_t n_rows, const float scale) {
    static_assert(!TURBO || N == GGML_TURBO4_QK, "TurboQuant requires 128-value groups");
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    const int64_t r = (int64_t) blockIdx.x * blockDim.y + threadIdx.y;

    if (r >= n_rows) {
        return;
    }

    src += r * N;
    dst += r * N;

    static constexpr int el_w = N / warp_size;
    float     reg[el_w];
    const int lane = threadIdx.x;

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        if constexpr (TURBO) {
            const int index = i * warp_size + lane;
            const float sign = INVERSE ? fwht_turbo_signs2[index] : fwht_turbo_signs1[index];
            reg[i] = __fmul_rn(src[index], sign);
        } else {
            reg[i] = src[i * warp_size + lane] * scale;
        }
    }

#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; j++) {
            const float val  = reg[j];
            const float val2 = __shfl_xor_sync(0xFFFFFFFF, val, h, warp_size);

            if constexpr (TURBO) {
                reg[j] = (lane & h) == 0 ? __fadd_rn(val, val2) : __fsub_rn(val2, val);
            } else {
                reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
            }
        }
    }

#pragma unroll
    for (int h = warp_size; h < N; h *= 2) {
        const int step = h / warp_size;
#pragma unroll
        for (int j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];

                if constexpr (TURBO) {
                    reg[j + k]        = __fadd_rn(x, y);
                    reg[j + k + step] = __fsub_rn(x, y);
                } else {
                    reg[j + k]        = x + y;
                    reg[j + k + step] = x - y;
                }
            }
        }
    }

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        if constexpr (TURBO) {
            const int index = i * warp_size + lane;
            const float sign = INVERSE ? fwht_turbo_signs1[index] : fwht_turbo_signs2[index];
            // Match the codec: apply normalization after all butterfly stages.
            dst[index] = __fmul_rn(reg[i], __fmul_rn(GGML_TURBO4_INV_SQRT128, sign));
        } else {
            dst[i * warp_size + lane] = reg[i];
        }
    }
}

bool ggml_cuda_op_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst,
                       const ggml_op_hint hint) {
    const bool turbo = hint == GGML_HINT_SRC0_IS_TURBO_FORWARD || hint == GGML_HINT_SRC0_IS_TURBO_INVERSE;
    if (turbo && (src->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
                  src->ne[0] != GGML_TURBO4_QK || !ggml_are_same_shape(src, dst))) {
        return false;
    }
    GGML_ASSERT(ggml_are_same_shape(src, dst));
    if (!ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        return false;
    }
    const int     n    = src->ne[0];
    const int64_t rows = ggml_nrows(src);

    const float * src_d = (const float *) src->data;
    float *       dst_d = (float *) dst->data;

    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int rows_per_block = 4;

    const int64_t num_blocks = (rows + rows_per_block - 1) / rows_per_block;

    cudaStream_t                         stream = ctx.stream();
    dim3                                 grid_dims(num_blocks, 1, 1);
    dim3                                 block_dims(warp_size, rows_per_block, 1);
    const ggml_cuda_kernel_launch_params launch_params =
        ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);

    if (turbo) {
        if (hint == GGML_HINT_SRC0_IS_TURBO_INVERSE) {
            ggml_cuda_kernel_launch(fwht_cuda<GGML_TURBO4_QK, true, true>, launch_params, src_d, dst_d, rows, GGML_TURBO4_INV_SQRT128);
        } else {
            ggml_cuda_kernel_launch(fwht_cuda<GGML_TURBO4_QK, true, false>, launch_params, src_d, dst_d, rows, GGML_TURBO4_INV_SQRT128);
        }
        return true;
    }

    const float scale = 1 / sqrtf(n);

    switch (n) {
        case 64:
            ggml_cuda_kernel_launch(fwht_cuda<64>, launch_params, src_d, dst_d, rows, scale);
            return true;
        case 128:
            ggml_cuda_kernel_launch(fwht_cuda<128>, launch_params, src_d, dst_d, rows, scale);
            return true;
        case 256:
            ggml_cuda_kernel_launch(fwht_cuda<256>, launch_params, src_d, dst_d, rows, scale);
            return true;
        case 512:
            ggml_cuda_kernel_launch(fwht_cuda<512>, launch_params, src_d, dst_d, rows, scale);
            return true;
        case 1024:
            // prism.hadamard weight folding (Bonsai 2) rotates in 1024 blocks;
            // 32 values per lane still fit in registers
            ggml_cuda_kernel_launch(fwht_cuda<1024>, launch_params, src_d, dst_d, rows, scale);
            return true;
        default:
            return false;
    }
}
