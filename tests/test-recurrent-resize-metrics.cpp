#include "llama-memory-recurrent.h"

#include <cstdlib>
#include <cstdint>

static void require(bool condition) {
    if (!condition) {
        std::abort();
    }
}

static llama_memory_recurrent_resize_stats snapshot(
        bool dynamic,
        uint32_t configured,
        uint32_t resident,
        uint32_t required,
        uint32_t pending = UINT32_MAX,
        uint32_t stable = 0,
        uint64_t count = 0) {
    return llama_recurrent_resize_metrics_snapshot(
            dynamic, configured, resident, required, pending, stable, count, count * 10);
}

int main() {
    // A target-only context has no configured or resident snapshot planes.
    const auto target = snapshot(false, 0, 0, 0);
    require(target.configured_depth == 0);
    require(target.resident_depth == 0);
    require(target.required_depth == 0);
    require(target.pending_depth == UINT32_MAX);

    // Fixed/static DSpark allocates its complete ceiling at startup. It must
    // report that physical and required depth even though no resize occurs.
    const auto fixed = snapshot(false, 7, 7, 0);
    require(fixed.configured_depth == 7);
    require(fixed.resident_depth == 7);
    require(fixed.required_depth == 7);
    require(fixed.count == 0);

    // Dynamic storage starts at zero, grows to the current requirement, may
    // defer a shrink, and eventually exposes the smaller resident depth.
    const auto dynamic_zero = snapshot(true, 7, 0, 0);
    require(dynamic_zero.configured_depth == 7);
    require(dynamic_zero.resident_depth == 0);
    require(dynamic_zero.required_depth == 0);

    const auto dynamic_three = snapshot(true, 7, 3, 3, UINT32_MAX, 0, 1);
    require(dynamic_three.resident_depth == 3);
    require(dynamic_three.required_depth == 3);
    require(dynamic_three.count == 1);
    require(dynamic_three.time_us == 10);

    const auto shrink_pending = snapshot(true, 7, 3, 0, 0, 1, 1);
    require(shrink_pending.resident_depth == 3);
    require(shrink_pending.required_depth == 0);
    require(shrink_pending.pending_depth == 0);
    require(shrink_pending.stable_ticks == 1);

    const auto dynamic_zero_again = snapshot(true, 7, 0, 0, UINT32_MAX, 0, 2);
    require(dynamic_zero_again.resident_depth == 0);
    require(dynamic_zero_again.required_depth == 0);
    require(dynamic_zero_again.count == 2);

    // Do not hide corrupt/out-of-range internal observations: the server and
    // qualification runner must see 99 and fail their <= configured gate.
    require(snapshot(true, 7, 7, 99).required_depth == 99);

    return 0;
}
