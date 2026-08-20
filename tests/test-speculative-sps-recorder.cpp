#include "speculative-sps.h"
#include "speculative-sps-atomic.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#undef NDEBUG
#include <cassert>

using json = nlohmann::ordered_json;

template<typename Fn>
static void assert_throws(Fn && fn) {
    bool threw = false;
    try {
        fn();
    } catch (const std::exception &) {
        threw = true;
    }
    assert(threw);
}

static std::string read_file(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary);
    assert(input.good());
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

static common_speculative_sps_profile_sample fake_timed_sample(
        int32_t context,
        int32_t active,
        int32_t rows,
        int64_t begin_us,
        int64_t end_us,
        common_speculative_sps_execution_kind kind = COMMON_SPECULATIVE_SPS_EXECUTION_GRAPH_REPLAY) {
    std::vector<int32_t> prefixes((size_t) active, 0);
    int32_t extras = rows - active;
    if (active > 0 && extras >= 0) {
        const int32_t base = extras / active;
        const int32_t remainder = extras % active;
        for (int32_t i = 0; i < active; ++i) {
            prefixes[(size_t) i] = base + (i < remainder ? 1 : 0);
        }
    }
    return {
        context,
        active,
        rows,
        (double) (end_us - begin_us),
        true,
        true,
        true,
        true,
        kind,
        prefixes,
    };
}

static common_speculative_sps_profile_recorder_config recorder_config();

static void test_cap7_and_heterogeneous_prefixes() {
    auto config = recorder_config();
    config.verify_rows_by_active = { { 2, { 2, 9, 16 } } };
    config.max_draft_tokens_per_slot = 7;
    config.retained_samples_per_coordinate = 1;
    config.warmup_samples_per_coordinate = 0;
    common_speculative_sps_profile_recorder recorder(std::move(config));

    auto anchor = fake_timed_sample(100, 2, 2, 0, 100);
    assert(recorder.observe(anchor) == COMMON_SPECULATIVE_SPS_RECORD_COORDINATE_READY);

    auto heterogeneous = fake_timed_sample(100, 2, 9, 0, 110);
    assert((heterogeneous.prefixes == std::vector<int32_t>{4, 3}));
    assert(recorder.observe(heterogeneous) == COMMON_SPECULATIVE_SPS_RECORD_COORDINATE_READY);

    auto cap7 = fake_timed_sample(100, 2, 16, 0, 120);
    assert((cap7.prefixes == std::vector<int32_t>{7, 7}));
    assert(recorder.observe(cap7) == COMMON_SPECULATIVE_SPS_RECORD_ALL_READY);
    assert(recorder.profile().entries.size() == 3);
}

static common_speculative_sps_profile_recorder_config recorder_config() {
    return {
        /* .source_identity                       = */ std::string(64, 'a'),
        /* .context_buckets                       = */ { 100 },
        /* .verify_rows_by_active                 = */ {
            { 1, { 1, 2 } },
            { 2, { 2 } },
        },
        /* .max_draft_tokens_per_slot             = */ 1,
        /* .retained_samples_per_coordinate       = */ 2,
        /* .warmup_samples_per_coordinate         = */ 1,
        /* .aggregate_quantile                    = */ 1.0,
    };
}

static void test_validation_and_fake_timing() {
    auto invalid = recorder_config();
    invalid.context_buckets = { 100, 100 };
    assert_throws([&] { common_speculative_sps_profile_recorder recorder(invalid); });

    invalid = recorder_config();
    invalid.verify_rows_by_active[1] = { 1, 3 };
    assert_throws([&] { common_speculative_sps_profile_recorder recorder(invalid); });

    common_speculative_sps_profile_recorder recorder(recorder_config());
    assert(recorder.expected_coordinates() == 3);
    assert(recorder.ready_coordinates() == 0);
    assert(!recorder.ready());
    assert_throws([&] { recorder.profile(); });

    auto sample = fake_timed_sample(
            95, 1, 1, 1000, 1250, COMMON_SPECULATIVE_SPS_EXECUTION_DIRECT_WARMUP);
    assert(recorder.observe(sample) == COMMON_SPECULATIVE_SPS_RECORD_SKIPPED_WARMUP);

    sample.execution_kind = COMMON_SPECULATIVE_SPS_EXECUTION_GRAPH_CAPTURE;
    assert(recorder.observe(sample) == COMMON_SPECULATIVE_SPS_RECORD_SKIPPED_CAPTURE);

    sample.execution_kind = COMMON_SPECULATIVE_SPS_EXECUTION_UNKNOWN;
    assert(recorder.observe(sample) == COMMON_SPECULATIVE_SPS_RECORD_SKIPPED_WARMUP);

    sample.execution_kind = COMMON_SPECULATIVE_SPS_EXECUTION_GRAPH_REPLAY;
    sample.pure_generation = false;
    assert(recorder.observe(sample) == COMMON_SPECULATIVE_SPS_RECORD_SKIPPED_INELIGIBLE);

    sample.pure_generation = true;
    sample.prefixes.clear();
    assert(recorder.observe(sample) == COMMON_SPECULATIVE_SPS_RECORD_SKIPPED_INELIGIBLE);
    sample.prefixes = { 0 };
    sample.cost_us = 100.0;
    assert(recorder.observe(sample) == COMMON_SPECULATIVE_SPS_RECORD_RETAINED);
    sample.cost_us = 120.0;
    assert(recorder.observe(sample) == COMMON_SPECULATIVE_SPS_RECORD_COORDINATE_READY);
    assert(recorder.ready_coordinates() == 1);
    assert(recorder.observe(sample) == COMMON_SPECULATIVE_SPS_RECORD_SKIPPED_FULL);

    auto outside = fake_timed_sample(101, 1, 1, 0, 100);
    assert(recorder.observe(outside) == COMMON_SPECULATIVE_SPS_RECORD_SKIPPED_OUTSIDE_GRID);

    // The stable warmup quota applies to graph replay too. Capture itself is
    // excluded independently and does not consume the quota.
    common_speculative_sps_profile_recorder replay_recorder(recorder_config());
    auto replay = fake_timed_sample(
            100, 1, 1, 0, 100, COMMON_SPECULATIVE_SPS_EXECUTION_GRAPH_CAPTURE);
    assert(replay_recorder.observe(replay) == COMMON_SPECULATIVE_SPS_RECORD_SKIPPED_CAPTURE);
    replay.execution_kind = COMMON_SPECULATIVE_SPS_EXECUTION_GRAPH_REPLAY;
    assert(replay_recorder.observe(replay) == COMMON_SPECULATIVE_SPS_RECORD_SKIPPED_WARMUP);
    assert(replay_recorder.observe(replay) == COMMON_SPECULATIVE_SPS_RECORD_RETAINED);
}

static void test_atomic_sidecar_and_restart_resume() {
    const auto nonce = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    const auto dir = std::filesystem::temp_directory_path() /
            ("llama-sps-recorder-" + std::to_string(nonce));
    std::filesystem::create_directories(dir);
    const auto profile_path = dir / "profile.json";
    const auto sidecar_path = dir / "profile.samples.json";

    {
        std::ofstream old_profile(profile_path, std::ios::binary);
        old_profile << "sentinel";
    }

    common_speculative_sps_profile_recorder recorder(recorder_config());

    auto record_pair = [&](int32_t active, int32_t rows, double first, double second) {
        auto sample = fake_timed_sample(100, active, rows, 0, (int64_t) first);
        sample.cost_us = first;
        recorder.observe(sample);
        // The configured stable warmup applies to graph replays as well.
        recorder.observe(sample);
        sample.cost_us = second;
        recorder.observe(sample);
    };

    record_pair(1, 1, 100.0, 120.0);
    assert_throws([&] { recorder.write_atomic(profile_path.string(), sidecar_path.string()); });
    assert(read_file(profile_path) == "sentinel");
    std::filesystem::remove(profile_path);
    recorder.write_atomic(profile_path.string(), sidecar_path.string());
    auto sidecar = json::parse(read_file(sidecar_path));
    assert(!sidecar.at("complete").get<bool>());
    assert(sidecar.at("ready_coordinates").get<size_t>() == 1);
    assert(sidecar.at("coordinates").at(0).at("p50_us").get<double>() == 100.0);
    assert(sidecar.at("coordinates").at(0).at("p95_us").get<double>() == 120.0);
    assert(sidecar.at("coordinates").at(0).at("max_us").get<double>() == 120.0);
    assert(sidecar.at("coordinates").at(1).at("p50_us").is_null());
    const auto incomplete_sidecar = sidecar;

    auto resumed_config = recorder_config();
    resumed_config.resume_sidecar_path = sidecar_path.string();
    common_speculative_sps_profile_recorder resumed(std::move(resumed_config));
    assert(resumed.ready_coordinates() == 1);
    assert(!resumed.ready());
    assert(resumed.retained_samples() == 2);
    assert(resumed.skipped_warmup() == 1);
    assert(resumed.skipped_capture() == 0);
    assert(resumed.skipped_ineligible() == 0);
    assert(resumed.skipped_outside_grid() == 0);
    assert(resumed.skipped_full() == 0);

    auto record_resumed_pair = [&](int32_t active, int32_t rows, double first, double second) {
        auto sample = fake_timed_sample(100, active, rows, 0, (int64_t) first);
        sample.cost_us = first;
        resumed.observe(sample);
        // This coordinate has not consumed its stable warmup before resume.
        resumed.observe(sample);
        sample.cost_us = second;
        resumed.observe(sample);
    };

    // Deliberately non-monotonic raw costs. The emitted profile must lift
    // dominated points instead of lowering any observation.
    record_resumed_pair(1, 2, 40.0, 50.0);
    record_resumed_pair(2, 2, 70.0, 80.0);
    assert(resumed.ready());
    assert(resumed.retained_samples() == 6);
    assert(resumed.skipped_warmup() == 3);

    resumed.write_atomic(profile_path.string(), sidecar_path.string());
    const auto profile = common_speculative_sps_profile_load(profile_path.string());
    assert(profile.schema_version == 2);
    assert(profile.max_draft_tokens_per_slot == 1);
    assert(profile.entries.size() == 3);
    assert(profile.lookup_cost_us(100, 1, 1).value() == 120.0);
    assert(profile.lookup_cost_us(100, 1, 2).value() == 120.0);
    assert(profile.lookup_cost_us(100, 2, 2).value() == 120.0);
    assert(!profile.lookup_cost_us(100, 3, 3));

    sidecar = json::parse(read_file(sidecar_path));
    assert(sidecar.at("complete").get<bool>());
    assert(sidecar.at("ready_coordinates").get<size_t>() == 3);
    assert(sidecar.at("retained_total").get<uint64_t>() == 6);

    // Unknown fields and config mismatches must never be accepted on resume.
    auto corrupt = sidecar;
    corrupt["unknown"] = true;
    const auto corrupt_path = dir / "corrupt.samples.json";
    {
        std::ofstream output(corrupt_path, std::ios::binary);
        output << corrupt.dump();
    }
    auto corrupt_config = recorder_config();
    corrupt_config.resume_sidecar_path = corrupt_path.string();
    assert_throws([&] { common_speculative_sps_profile_recorder rejected(std::move(corrupt_config)); });

    corrupt = sidecar;
    corrupt["coordinates"][0]["samples"][0]["prefixes"] = json::array({ 1 });
    {
        std::ofstream output(corrupt_path, std::ios::binary | std::ios::trunc);
        output << corrupt.dump();
    }
    corrupt_config = recorder_config();
    corrupt_config.resume_sidecar_path = corrupt_path.string();
    assert_throws([&] { common_speculative_sps_profile_recorder rejected(std::move(corrupt_config)); });

    corrupt = sidecar;
    corrupt["coordinates"][0]["warmup_seen"] = 0;
    {
        std::ofstream output(corrupt_path, std::ios::binary | std::ios::trunc);
        output << corrupt.dump();
    }
    corrupt_config = recorder_config();
    corrupt_config.resume_sidecar_path = corrupt_path.string();
    assert_throws([&] { common_speculative_sps_profile_recorder rejected(std::move(corrupt_config)); });

    corrupt = incomplete_sidecar;
    corrupt["coordinates"][1]["skipped_full"] = 1;
    corrupt["skipped_full_total"] = 1;
    corrupt["observations_total"] = corrupt["observations_total"].get<uint64_t>() + 1;
    {
        std::ofstream output(corrupt_path, std::ios::binary | std::ios::trunc);
        output << corrupt.dump();
    }
    corrupt_config = recorder_config();
    corrupt_config.resume_sidecar_path = corrupt_path.string();
    assert_throws([&] { common_speculative_sps_profile_recorder rejected(std::move(corrupt_config)); });

    auto mismatch_config = recorder_config();
    mismatch_config.aggregate_quantile = 0.5;
    mismatch_config.resume_sidecar_path = sidecar_path.string();
    assert_throws([&] { common_speculative_sps_profile_recorder rejected(std::move(mismatch_config)); });

    auto identity_mismatch = recorder_config();
    identity_mismatch.source_identity = std::string(64, 'b');
    identity_mismatch.resume_sidecar_path = sidecar_path.string();
    assert_throws([&] { common_speculative_sps_profile_recorder rejected(std::move(identity_mismatch)); });

    for (const auto & entry : std::filesystem::directory_iterator(dir)) {
        assert(entry.path().filename().string().find(".tmp.") == std::string::npos);
    }
    std::filesystem::remove_all(dir);
}

#ifdef _WIN32
static void test_bounded_windows_replace_retry_contract() {
    using namespace common_speculative_sps_detail;

    assert(windows_replace_error_is_retryable(WIN_ERROR_ACCESS_DENIED));
    assert(windows_replace_error_is_retryable(WIN_ERROR_SHARING_VIOLATION));
    assert(windows_replace_error_is_retryable(WIN_ERROR_LOCK_VIOLATION));
    assert(windows_replace_error_is_retryable(WIN_ERROR_UNABLE_TO_REMOVE_REPLACED));
    assert(!windows_replace_error_is_retryable(ERROR_FILE_NOT_FOUND));
    assert(!windows_replace_error_is_retryable(ERROR_INVALID_PARAMETER));

    uint64_t now_ns = 0;
    size_t attempt_index = 0;
    const std::vector<uint32_t> transient_then_success = {
        WIN_ERROR_ACCESS_DENIED,
        WIN_ERROR_SHARING_VIOLATION,
        WIN_ERROR_LOCK_VIOLATION,
        WIN_ERROR_UNABLE_TO_REMOVE_REPLACED,
        0,
    };
    const auto success = retry_windows_replace(
            [&]() { return transient_then_success.at(attempt_index++); },
            [&]() { return now_ns; },
            [&](uint32_t delay_ms) { now_ns += (uint64_t) delay_ms * 1000ULL * 1000ULL; });
    assert(success.success);
    assert(success.error == 0);
    assert(success.retries == 4);
    assert(attempt_index == 5);
    assert(now_ns == 20ULL * 1000ULL * 1000ULL);

    now_ns = 0;
    size_t permanent_attempts = 0;
    const auto deadline_failure = retry_windows_replace(
            [&]() {
                permanent_attempts++;
                return WIN_ERROR_UNABLE_TO_REMOVE_REPLACED;
            },
            [&]() { return now_ns; },
            [&](uint32_t delay_ms) { now_ns += (uint64_t) delay_ms * 1000ULL * 1000ULL; },
            12ULL * 1000ULL * 1000ULL,
            5);
    assert(!deadline_failure.success);
    assert(deadline_failure.error == WIN_ERROR_UNABLE_TO_REMOVE_REPLACED);
    assert(deadline_failure.retries == 2);
    assert(permanent_attempts == 3);
    assert(now_ns == 12ULL * 1000ULL * 1000ULL);

    for (const uint32_t nonretryable : { (uint32_t) ERROR_FILE_NOT_FOUND, (uint32_t) ERROR_INVALID_PARAMETER }) {
        now_ns = 0;
        size_t calls = 0;
        size_t sleeps = 0;
        const auto immediate = retry_windows_replace(
                [&]() { calls++; return nonretryable; },
                [&]() { return now_ns; },
                [&](uint32_t) { sleeps++; });
        assert(!immediate.success);
        assert(immediate.error == nonretryable);
        assert(immediate.retries == 0);
        assert(calls == 1);
        assert(sleeps == 0);
    }
}

static void test_atomic_replace_with_shared_delete_reader() {
    const auto nonce = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    const auto dir = std::filesystem::temp_directory_path() /
            ("llama-sps-recorder-shared-reader-" + std::to_string(nonce));
    std::filesystem::create_directories(dir);
    const auto profile_path = dir / "profile.json";
    const auto sidecar_path = dir / "profile.samples.json";

    common_speculative_sps_profile_recorder recorder(recorder_config());
    auto sample = fake_timed_sample(100, 1, 1, 0, 100);
    recorder.observe(sample);
    recorder.observe(sample);
    sample.cost_us = 120.0;
    recorder.observe(sample);
    recorder.write_atomic(profile_path.string(), sidecar_path.string());

    const HANDLE reader = CreateFileW(
            sidecar_path.c_str(),
            GENERIC_READ,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
            nullptr,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN,
            nullptr);
    assert(reader != INVALID_HANDLE_VALUE);
    try {
        for (int i = 0; i < 300; ++i) {
            recorder.write_atomic(profile_path.string(), sidecar_path.string());
        }
    } catch (...) {
        CloseHandle(reader);
        std::filesystem::remove_all(dir);
        throw;
    }
    assert(CloseHandle(reader));
    assert(json::parse(read_file(sidecar_path)).at("ready_coordinates").get<size_t>() == 1);
    std::filesystem::remove_all(dir);
}

static void test_permanent_sharing_failure_is_bounded_and_cleans_temp() {
    const auto nonce = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    const auto dir = std::filesystem::temp_directory_path() /
            ("llama-sps-recorder-locked-destination-" + std::to_string(nonce));
    std::filesystem::create_directories(dir);
    const auto profile_path = dir / "profile.json";
    const auto sidecar_path = dir / "profile.samples.json";

    common_speculative_sps_profile_recorder recorder(recorder_config());
    auto sample = fake_timed_sample(100, 1, 1, 0, 100);
    recorder.observe(sample);
    recorder.observe(sample);
    sample.cost_us = 120.0;
    recorder.observe(sample);
    recorder.write_atomic(profile_path.string(), sidecar_path.string());

    // Deliberately omit FILE_SHARE_DELETE so ReplaceFileW sees a permanent,
    // retryable sharing violation until the bounded deadline expires.
    const HANDLE reader = CreateFileW(
            sidecar_path.c_str(),
            GENERIC_READ,
            FILE_SHARE_READ,
            nullptr,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN,
            nullptr);
    assert(reader != INVALID_HANDLE_VALUE);

    bool threw = false;
    std::string message;
    const auto started = std::chrono::steady_clock::now();
    try {
        recorder.write_atomic(profile_path.string(), sidecar_path.string());
    } catch (const std::exception & error) {
        threw = true;
        message = error.what();
    }
    const auto elapsed = std::chrono::steady_clock::now() - started;
    assert(threw);
    assert(message.find("Windows error 32") != std::string::npos ||
           message.find("Windows error 5") != std::string::npos);
    assert(message.find("retries 0") == std::string::npos);
    assert(elapsed >= std::chrono::milliseconds(450));
    assert(elapsed < std::chrono::seconds(2));
    assert(CloseHandle(reader));

    for (const auto & entry : std::filesystem::directory_iterator(dir)) {
        assert(entry.path().filename().string().find(".tmp.") == std::string::npos);
    }
    assert(json::parse(read_file(sidecar_path)).at("ready_coordinates").get<size_t>() == 1);
    std::filesystem::remove_all(dir);
}
#endif

int main() {
    test_validation_and_fake_timing();
    test_cap7_and_heterogeneous_prefixes();
    test_atomic_sidecar_and_restart_resume();
#ifdef _WIN32
    test_bounded_windows_replace_retry_contract();
    test_atomic_replace_with_shared_delete_reader();
    test_permanent_sharing_failure_is_bounded_and_cleans_temp();
#endif
    return 0;
}
