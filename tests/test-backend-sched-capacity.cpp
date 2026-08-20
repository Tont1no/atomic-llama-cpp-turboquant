#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "../ggml/src/ggml-impl.h"
#include "../src/llama-graph.h"

#include <cstddef>

#undef NDEBUG
#include <cassert>

static void test_graph_result_nodes_and_leafs_metadata() {
    // The failing P8 reserve selected max_nodes=1139; accounting for the
    // sampling graph's shared padded-logits node makes the final capacity 1140.
    // A cgraph can independently own this many operation and leaf tensors, so
    // its metadata arena must hold both sets.
    constexpr size_t old_graph_capacity = 1139;
    constexpr size_t graph_capacity = old_graph_capacity + 1;

    const size_t old_meta_size = old_graph_capacity * ggml_tensor_overhead() +
            ggml_graph_overhead_custom(old_graph_capacity, false);
#if defined(_WIN64)
    assert(ggml_tensor_overhead() == 368);
    assert(old_meta_size == 487216);
    assert(old_meta_size + ggml_tensor_overhead() == 487584);
#endif

    const size_t meta_size = llm_graph_result_meta_size(graph_capacity);
#if defined(_WIN64)
    assert(meta_size == 907120);
#endif
    ggml_init_params params = {
        /* .mem_size = */ meta_size,
        /* .mem_base = */ nullptr,
        /* .no_alloc = */ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    assert(ctx);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), graph_capacity, false);
    assert(graph);

    for (uint32_t i = 0; i < graph_capacity; ++i) {
        ggml_tensor * leaf = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
        ggml_tensor * node = ggml_dup(ctx.get(), leaf);
        assert(leaf && node);
        ggml_build_forward_expand(graph, node);
    }
    assert(graph->n_nodes == (int) graph_capacity);
    assert(graph->n_leafs == (int) graph_capacity);
    assert(ggml_used_mem(ctx.get()) == ggml_get_mem_size(ctx.get()));
}

static void test_scheduler_nodes_and_leafs_capacity() {
    // Reproduce the P8 reserve failure condition: the graph has room for 1024
    // nodes and 1024 leafs independently, while their combined count exceeds
    // the old scheduler hash capacity.
    constexpr size_t graph_capacity = 1024;
    constexpr int n_leafs = 520;

    ggml_init_params params = {
        /* .mem_size = */ 2 * n_leafs * ggml_tensor_overhead() + ggml_graph_overhead_custom(graph_capacity, false),
        /* .mem_base = */ nullptr,
        /* .no_alloc = */ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    assert(ctx);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), graph_capacity, false);
    assert(graph);

    ggml_tensor * out = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
    for (int i = 1; i < n_leafs; ++i) {
        ggml_tensor * leaf = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
        out = ggml_add(ctx.get(), out, leaf);
    }
    ggml_build_forward_expand(graph, out);

    assert(graph->n_nodes == n_leafs - 1);
    assert(graph->n_leafs == n_leafs);
    assert(graph->n_nodes + graph->n_leafs == 1039);

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    assert(backend);
    ggml_backend_t backends[] = { backend.get() };
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(
            backends, nullptr, 1, graph_capacity, false, true));
    assert(sched);
    assert(ggml_backend_sched_reserve(sched.get(), graph));
}

int main() {
    test_graph_result_nodes_and_leafs_metadata();
    test_scheduler_nodes_and_leafs_capacity();

    return 0;
}
