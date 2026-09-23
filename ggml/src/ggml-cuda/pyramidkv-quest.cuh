#include "common.cuh"

void ggml_cuda_op_pyramidkv_quest_update(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_pyramidkv_quest_select(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
