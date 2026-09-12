#pragma once

#include <cstdlib>

// Capture at backend creation; graphs can be created after a scoped override ends.
class ggml_cuda_graph_policy {
public:
    ggml_cuda_graph_policy() : disabled_by_environment(std::getenv("GGML_CUDA_DISABLE_GRAPHS") != nullptr) {}

    bool is_enabled(bool disabled_by_gpu_arch) const {
        return !disabled_by_environment && !disabled_by_gpu_arch;
    }

private:
    const bool disabled_by_environment;
};
