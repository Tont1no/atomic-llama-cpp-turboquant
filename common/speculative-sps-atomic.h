#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>

namespace common_speculative_sps_detail {

// Win32 error values are stable ABI constants. The production translation
// unit statically verifies these numbers against the active Windows SDK.
constexpr uint32_t WIN_ERROR_ACCESS_DENIED              = 5;
constexpr uint32_t WIN_ERROR_SHARING_VIOLATION          = 32;
constexpr uint32_t WIN_ERROR_LOCK_VIOLATION             = 33;
constexpr uint32_t WIN_ERROR_UNABLE_TO_REMOVE_REPLACED  = 1175;

constexpr uint64_t WINDOWS_REPLACE_RETRY_WINDOW_NS = 500ULL * 1000ULL * 1000ULL;
constexpr uint32_t WINDOWS_REPLACE_RETRY_SLEEP_MS  = 5;

inline bool windows_replace_error_is_retryable(uint32_t error) {
    return error == WIN_ERROR_ACCESS_DENIED ||
           error == WIN_ERROR_SHARING_VIOLATION ||
           error == WIN_ERROR_LOCK_VIOLATION ||
           error == WIN_ERROR_UNABLE_TO_REMOVE_REPLACED;
}

struct windows_replace_retry_result {
    bool     success = false;
    uint32_t error   = 0;
    uint32_t retries = 0;
};

// attempt() returns zero on success or a Win32 error. now_ns() must use a
// monotonic clock. sleep_ms() is injected so the bounded retry contract is
// deterministic in CPU tests and uses Sleep() in production.
template<class Attempt, class NowNs, class SleepMs>
windows_replace_retry_result retry_windows_replace(
        Attempt && attempt,
        NowNs && now_ns,
        SleepMs && sleep_ms,
        uint64_t retry_window_ns = WINDOWS_REPLACE_RETRY_WINDOW_NS,
        uint32_t retry_sleep_ms = WINDOWS_REPLACE_RETRY_SLEEP_MS) {
    const uint64_t start = now_ns();
    const uint64_t deadline = start > std::numeric_limits<uint64_t>::max() - retry_window_ns ?
            std::numeric_limits<uint64_t>::max() : start + retry_window_ns;

    windows_replace_retry_result result;
    result.error = attempt();
    while (result.error != 0 && windows_replace_error_is_retryable(result.error)) {
        const uint64_t now = now_ns();
        if (now >= deadline || retry_sleep_ms == 0) {
            break;
        }
        const uint64_t remaining_ns = deadline - now;
        const uint32_t bounded_sleep_ms = (uint32_t) std::max<uint64_t>(
                1, std::min<uint64_t>(retry_sleep_ms, remaining_ns / (1000ULL * 1000ULL)));
        sleep_ms(bounded_sleep_ms);
        // A millisecond-granularity sleep may cross a sub-millisecond
        // remainder. Never start another ReplaceFileW call after the
        // monotonic retry deadline.
        if (now_ns() >= deadline) {
            break;
        }
        result.retries++;
        result.error = attempt();
    }
    result.success = result.error == 0;
    return result;
}

} // namespace common_speculative_sps_detail
