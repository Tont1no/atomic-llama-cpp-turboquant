#include "speculative-sps.h"
#include "speculative-sps-atomic.h"

#include "common.h"
#include "log.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

static_assert(ERROR_ACCESS_DENIED == common_speculative_sps_detail::WIN_ERROR_ACCESS_DENIED);
static_assert(ERROR_SHARING_VIOLATION == common_speculative_sps_detail::WIN_ERROR_SHARING_VIOLATION);
static_assert(ERROR_LOCK_VIOLATION == common_speculative_sps_detail::WIN_ERROR_LOCK_VIOLATION);
static_assert(ERROR_UNABLE_TO_REMOVE_REPLACED ==
        common_speculative_sps_detail::WIN_ERROR_UNABLE_TO_REMOVE_REPLACED);
#endif

using json = nlohmann::ordered_json;

namespace {

// Validation compares every dominance-related pair. Keep startup work bounded;
// a phase-1 profile should be a compact measured grid, not raw benchmark data.
constexpr size_t SPS_MAX_ENTRIES = 4096;

[[noreturn]] void sps_error(const std::string & source, const std::string & message) {
    throw std::runtime_error("invalid SPS profile '" + source + "': " + message);
}

void require_exact_keys(
        const json & value,
        const std::set<std::string> & expected,
        const std::string & source,
        const std::string & where) {
    if (!value.is_object()) {
        sps_error(source, where + " must be an object");
    }

    std::set<std::string> actual;
    for (auto it = value.begin(); it != value.end(); ++it) {
        actual.insert(it.key());
    }
    if (actual != expected) {
        sps_error(source, where + " has missing or unknown fields");
    }
}

int32_t positive_i32(
        const json & value,
        const char * key,
        const std::string & source,
        size_t row) {
    const auto & field = value.at(key);
    if (!field.is_number_integer()) {
        sps_error(source, "entries[" + std::to_string(row) + "]." + key + " must be an integer");
    }

    const int64_t result = field.get<int64_t>();
    if (result <= 0 || result > std::numeric_limits<int32_t>::max()) {
        sps_error(source, "entries[" + std::to_string(row) + "]." + key + " must be in [1, INT32_MAX]");
    }
    return (int32_t) result;
}

int32_t nonnegative_i32(
        const json & value,
        const char * key,
        const std::string & source) {
    const auto & field = value.at(key);
    if (!field.is_number_integer()) {
        sps_error(source, std::string(key) + " must be an integer");
    }

    const int64_t result = field.get<int64_t>();
    if (result < 0 || result > std::numeric_limits<int32_t>::max()) {
        sps_error(source, std::string(key) + " must be in [0, INT32_MAX]");
    }
    return (int32_t) result;
}

double positive_finite(
        const json & value,
        const char * key,
        const std::string & source,
        size_t row) {
    const auto & field = value.at(key);
    if (!field.is_number()) {
        sps_error(source, "entries[" + std::to_string(row) + "]." + key + " must be numeric");
    }

    const double result = field.get<double>();
    if (!std::isfinite(result) || result <= 0.0) {
        sps_error(source, "entries[" + std::to_string(row) + "]." + key + " must be finite and positive");
    }
    return result;
}

bool dominates_or_equals(
        const common_speculative_sps_entry & lhs,
        const common_speculative_sps_entry & rhs) {
    return lhs.context_tokens    <= rhs.context_tokens &&
           lhs.active_slots      <= rhs.active_slots &&
           lhs.total_verify_rows <= rhs.total_verify_rows;
}

int compare_positive_ratios(double lhs_num, double lhs_den, double rhs_num, double rhs_den) {
    int lhs_num_exp = 0;
    int lhs_den_exp = 0;
    int rhs_num_exp = 0;
    int rhs_den_exp = 0;
    const double lhs_num_frac = std::frexp(lhs_num, &lhs_num_exp);
    const double lhs_den_frac = std::frexp(lhs_den, &lhs_den_exp);
    const double rhs_num_frac = std::frexp(rhs_num, &rhs_num_exp);
    const double rhs_den_frac = std::frexp(rhs_den, &rhs_den_exp);

    int lhs_ratio_exp = 0;
    int rhs_ratio_exp = 0;
    const double lhs_ratio_frac = std::frexp(lhs_num_frac / lhs_den_frac, &lhs_ratio_exp);
    const double rhs_ratio_frac = std::frexp(rhs_num_frac / rhs_den_frac, &rhs_ratio_exp);

    const int lhs_exp = lhs_num_exp - lhs_den_exp + lhs_ratio_exp;
    const int rhs_exp = rhs_num_exp - rhs_den_exp + rhs_ratio_exp;
    if (lhs_exp != rhs_exp) {
        return lhs_exp < rhs_exp ? -1 : 1;
    }

    return lhs_ratio_frac < rhs_ratio_frac ? -1 : lhs_ratio_frac > rhs_ratio_frac ? 1 : 0;
}

const char * execution_kind_name(common_speculative_sps_execution_kind kind) {
    switch (kind) {
        case COMMON_SPECULATIVE_SPS_EXECUTION_UNKNOWN:         return "unknown";
        case COMMON_SPECULATIVE_SPS_EXECUTION_DIRECT_WARMUP:   return "direct_warmup";
        case COMMON_SPECULATIVE_SPS_EXECUTION_GRAPH_CAPTURE:   return "graph_capture";
        case COMMON_SPECULATIVE_SPS_EXECUTION_GRAPH_REPLAY:    return "graph_replay";
        case COMMON_SPECULATIVE_SPS_EXECUTION_GRAPHS_DISABLED: return "graphs_disabled";
    }
    return "invalid";
}

common_speculative_sps_execution_kind execution_kind_from_name(
        const std::string & value,
        const std::string & source) {
    if (value == "unknown")         return COMMON_SPECULATIVE_SPS_EXECUTION_UNKNOWN;
    if (value == "direct_warmup")   return COMMON_SPECULATIVE_SPS_EXECUTION_DIRECT_WARMUP;
    if (value == "graph_capture")   return COMMON_SPECULATIVE_SPS_EXECUTION_GRAPH_CAPTURE;
    if (value == "graph_replay")    return COMMON_SPECULATIVE_SPS_EXECUTION_GRAPH_REPLAY;
    if (value == "graphs_disabled") return COMMON_SPECULATIVE_SPS_EXECUTION_GRAPHS_DISABLED;
    throw std::runtime_error("invalid SPS recorder sidecar '" + source + "': unknown execution_kind");
}

json profile_to_json(const common_speculative_sps_profile & profile) {
    json entries = json::array();
    for (const auto & entry : profile.entries) {
        entries.push_back({
            { "context_tokens",    entry.context_tokens },
            { "active_slots",      entry.active_slots },
            { "total_verify_rows", entry.total_verify_rows },
            { "cost_us",           entry.cost_us },
        });
    }

    if (profile.schema_version == 1) {
        return {
            { "schema_version", 1 },
            { "entries", std::move(entries) },
        };
    }
    return {
        { "schema_version", 2 },
        { "max_draft_tokens_per_slot", profile.max_draft_tokens_per_slot },
        { "entries", std::move(entries) },
    };
}

std::filesystem::path utf8_path(const std::string & path) {
    return std::filesystem::u8path(path);
}

void atomic_replace_json(const std::string & destination, const json & document) {
    if (destination.empty()) {
        throw std::runtime_error("SPS recorder output path must not be empty");
    }

    static std::atomic<uint64_t> nonce { 0 };
    const auto destination_path = utf8_path(destination);
    const auto suffix = ".tmp." + std::to_string(
            std::chrono::high_resolution_clock::now().time_since_epoch().count()) + "." +
            std::to_string(nonce.fetch_add(1, std::memory_order_relaxed));
    auto temporary_path = destination_path;
    temporary_path += utf8_path(suffix);

    try {
        {
            std::ofstream output(temporary_path, std::ios::out | std::ios::binary | std::ios::trunc);
            if (!output) {
                throw std::runtime_error("failed to open temporary SPS recorder output '" + temporary_path.u8string() + "'");
            }
            output << document.dump(2) << '\n';
            output.flush();
            if (!output) {
                throw std::runtime_error("failed to write temporary SPS recorder output '" + temporary_path.u8string() + "'");
            }
        }

#ifdef _WIN32
        // MoveFileExW cannot replace an open destination even when every
        // reader opted into FILE_SHARE_DELETE. ReplaceFileW can, and preserves
        // the recorder artifact's one-generation atomicity.
        const auto replace_result = common_speculative_sps_detail::retry_windows_replace(
                [&]() -> uint32_t {
                    return ReplaceFileW(
                            destination_path.c_str(),
                            temporary_path.c_str(),
                            nullptr,
                            REPLACEFILE_WRITE_THROUGH,
                            nullptr,
                            nullptr) ? 0 : GetLastError();
                },
                []() -> uint64_t {
                    return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now().time_since_epoch()).count();
                },
                [](uint32_t delay_ms) { Sleep(delay_ms); });
        if (replace_result.success) {
            if (replace_result.retries > 0) {
                COM_WRN("SPS recorder atomic replace succeeded after %u retries: '%s'\n",
                        replace_result.retries, destination.c_str());
            }
        } else if (replace_result.error == ERROR_FILE_NOT_FOUND) {
            // Initial publication has no destination to replace. Keep this a
            // single fail-closed move: retrying it could overwrite a file
            // concurrently created by another writer.
            if (!MoveFileExW(
                        temporary_path.c_str(),
                        destination_path.c_str(),
                        MOVEFILE_WRITE_THROUGH)) {
                const DWORD move_error = GetLastError();
                COM_ERR("SPS recorder initial atomic move failed after 0 retries with Windows error %lu: '%s'\n",
                        (unsigned long) move_error, destination.c_str());
                throw std::runtime_error("failed to atomically replace SPS recorder output '" + destination +
                        "' (Windows error " + std::to_string(move_error) + ", retries 0)");
            }
        } else {
            COM_ERR("SPS recorder atomic replace failed after %u retries with Windows error %u: '%s'\n",
                    replace_result.retries, replace_result.error, destination.c_str());
            throw std::runtime_error("failed to atomically replace SPS recorder output '" + destination +
                    "' (Windows error " + std::to_string(replace_result.error) + ", retries " +
                    std::to_string(replace_result.retries) + ")");
        }
#else
        std::error_code ec;
        std::filesystem::rename(temporary_path, destination_path, ec);
        if (ec) {
            throw std::runtime_error("failed to atomically replace SPS recorder output '" + destination +
                    "': " + ec.message());
        }
#endif
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporary_path, ignored);
        throw;
    }
}

} // namespace

std::optional<double> common_speculative_sps_profile::lookup_cost_us(
        int32_t context_tokens,
        int32_t active_slots,
        int32_t total_verify_rows) const {
    if (context_tokens <= 0 || active_slots <= 0 || total_verify_rows < active_slots) {
        return std::nullopt;
    }

    const common_speculative_sps_entry * best = nullptr;
    for (const auto & entry : entries) {
        if (entry.context_tokens < context_tokens ||
            (schema_version == 2 ? entry.active_slots != active_slots : entry.active_slots < active_slots) ||
            entry.total_verify_rows < total_verify_rows) {
            continue;
        }

        if (best == nullptr ||
            std::tie(entry.context_tokens, entry.active_slots, entry.total_verify_rows) <
            std::tie(best->context_tokens, best->active_slots, best->total_verify_rows)) {
            best = &entry;
        }
    }

    return best ? std::optional<double>(best->cost_us) : std::nullopt;
}

common_speculative_sps_profile common_speculative_sps_profile_parse(
        const std::string & contents,
        const std::string & source) {
    json root;
    bool duplicate_key = false;
    std::vector<std::set<std::string>> object_keys;
    try {
        root = json::parse(contents, [&](int, json::parse_event_t event, json & parsed) {
            switch (event) {
                case json::parse_event_t::object_start:
                    object_keys.emplace_back();
                    break;
                case json::parse_event_t::key:
                    if (object_keys.empty() || !object_keys.back().insert(parsed.get<std::string>()).second) {
                        duplicate_key = true;
                    }
                    break;
                case json::parse_event_t::object_end:
                    if (!object_keys.empty()) {
                        object_keys.pop_back();
                    }
                    break;
                default:
                    break;
            }
            return true;
        });
    } catch (const std::exception & e) {
        sps_error(source, std::string("JSON parse failed: ") + e.what());
    }
    if (duplicate_key) {
        sps_error(source, "duplicate JSON object member name");
    }

    if (!root.is_object() || !root.contains("schema_version") || !root.at("schema_version").is_number_integer()) {
        sps_error(source, "schema_version must be integer 1 or 2");
    }
    const int64_t schema_version = root.at("schema_version").get<int64_t>();
    if (schema_version == 1) {
        require_exact_keys(root, { "schema_version", "entries" }, source, "root");
    } else if (schema_version == 2) {
        require_exact_keys(root, { "schema_version", "max_draft_tokens_per_slot", "entries" }, source, "root");
    } else {
        sps_error(source, "schema_version must be integer 1 or 2");
    }

    const auto & rows = root.at("entries");
    if (!rows.is_array() || rows.empty() || rows.size() > SPS_MAX_ENTRIES) {
        sps_error(source, "entries must be a non-empty array with at most " + std::to_string(SPS_MAX_ENTRIES) + " rows");
    }

    common_speculative_sps_profile result;
    result.schema_version = (int32_t) schema_version;
    if (schema_version == 2) {
        result.max_draft_tokens_per_slot = nonnegative_i32(
                root, "max_draft_tokens_per_slot", source);
    }
    result.entries.reserve(rows.size());

    std::set<std::tuple<int32_t, int32_t, int32_t>> coordinates;
    for (size_t i = 0; i < rows.size(); ++i) {
        const auto & row = rows[i];
        require_exact_keys(
                row,
                { "context_tokens", "active_slots", "total_verify_rows", "cost_us" },
                source,
                "entries[" + std::to_string(i) + "]");

        common_speculative_sps_entry entry {
            positive_i32(row, "context_tokens", source, i),
            positive_i32(row, "active_slots", source, i),
            positive_i32(row, "total_verify_rows", source, i),
            positive_finite(row, "cost_us", source, i),
        };

        if (entry.total_verify_rows < entry.active_slots) {
            sps_error(source, "entries[" + std::to_string(i) + "].total_verify_rows must be >= active_slots");
        }
        if (schema_version == 2) {
            const int64_t max_rows = (int64_t) entry.active_slots *
                    (1 + (int64_t) result.max_draft_tokens_per_slot);
            if (entry.total_verify_rows > max_rows) {
                sps_error(source, "entries[" + std::to_string(i) +
                        "].total_verify_rows exceeds the schema v2 active-slot row domain");
            }
        }

        const auto key = std::make_tuple(entry.context_tokens, entry.active_slots, entry.total_verify_rows);
        if (!coordinates.insert(key).second) {
            sps_error(source, "duplicate SPS coordinate at entries[" + std::to_string(i) + "]");
        }

        result.entries.push_back(entry);
    }

    // A measured cost surface must not become cheaper when any load dimension
    // increases. Validate every comparable pair so sparse tables remain safe.
    for (size_t i = 0; i < result.entries.size(); ++i) {
        for (size_t j = 0; j < result.entries.size(); ++j) {
            if (i == j || !dominates_or_equals(result.entries[i], result.entries[j])) {
                continue;
            }
            if (result.entries[i].cost_us > result.entries[j].cost_us) {
                sps_error(source, "cost_us is non-monotonic across SPS coordinates");
            }
        }
    }

    if (schema_version == 1) {
        // v1 preserves the original global Cartesian axes and active-slot
        // ceiling lookup for compatibility with existing profiles.
        std::set<int32_t> contexts;
        std::set<int32_t> slot_counts;
        std::set<int32_t> row_counts;
        for (const auto & entry : result.entries) {
            contexts.insert(entry.context_tokens);
            slot_counts.insert(entry.active_slots);
            row_counts.insert(entry.total_verify_rows);
        }
        const std::vector<int32_t> sorted_rows(row_counts.begin(), row_counts.end());
        size_t valid_slot_row_pairs = 0;
        for (const int32_t active : slot_counts) {
            const auto first_valid = std::lower_bound(sorted_rows.begin(), sorted_rows.end(), active);
            const size_t compatible_rows = (size_t) (sorted_rows.end() - first_valid);
            if (compatible_rows > result.entries.size() - valid_slot_row_pairs) {
                sps_error(source, "entries must form a complete valid SPS coordinate grid");
            }
            valid_slot_row_pairs += compatible_rows;
        }

        if (valid_slot_row_pairs == 0 ||
            contexts.size() > result.entries.size() / valid_slot_row_pairs ||
            contexts.size() * valid_slot_row_pairs != result.entries.size()) {
            sps_error(source, "entries must form a complete valid SPS coordinate grid");
        }
    } else {
        // v2 makes active_slots exact. Each active count therefore owns an
        // independent physical row axis and only its context x row grid must
        // be complete.
        std::map<int32_t, std::set<int32_t>> contexts_by_active;
        std::map<int32_t, std::set<int32_t>> rows_by_active;
        std::map<int32_t, size_t> counts_by_active;
        for (const auto & entry : result.entries) {
            contexts_by_active[entry.active_slots].insert(entry.context_tokens);
            rows_by_active[entry.active_slots].insert(entry.total_verify_rows);
            counts_by_active[entry.active_slots]++;
        }
        for (const auto & [active, contexts] : contexts_by_active) {
            const size_t rows_count = rows_by_active.at(active).size();
            if (rows_count == 0 || contexts.size() > SPS_MAX_ENTRIES / rows_count ||
                contexts.size() * rows_count != counts_by_active.at(active)) {
                sps_error(source, "schema v2 entries must form a complete context-by-row grid for each active_slots value");
            }
        }
    }

    std::sort(result.entries.begin(), result.entries.end(), [](const auto & lhs, const auto & rhs) {
        return std::tie(lhs.context_tokens, lhs.active_slots, lhs.total_verify_rows) <
               std::tie(rhs.context_tokens, rhs.active_slots, rhs.total_verify_rows);
    });

    return result;
}

common_speculative_sps_profile common_speculative_sps_profile_load(const std::string & path) {
    std::ifstream input = fs_open_ifstream(path, std::ios::in | std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open SPS profile '" + path + "'");
    }

    std::ostringstream contents;
    contents << input.rdbuf();
    if (!input.good() && !input.eof()) {
        throw std::runtime_error("failed to read SPS profile '" + path + "'");
    }

    return common_speculative_sps_profile_parse(contents.str(), path);
}

struct common_speculative_sps_profile_recorder::impl {
    using coordinate = std::tuple<int32_t, int32_t, int32_t>;

    struct retained_sample {
        int32_t actual_context_tokens = 0;
        double cost_us = 0.0;
        common_speculative_sps_execution_kind execution_kind = COMMON_SPECULATIVE_SPS_EXECUTION_UNKNOWN;
        std::vector<int32_t> prefixes;
    };

    struct coordinate_state {
        std::vector<retained_sample> samples;
        uint32_t warmup_seen         = 0;
        uint64_t skipped_warmup      = 0;
        uint64_t skipped_capture     = 0;
        uint64_t skipped_ineligible  = 0;
        uint64_t skipped_full        = 0;
    };

    explicit impl(common_speculative_sps_profile_recorder_config config_) : config(std::move(config_)) {
        validate_config();
        for (const auto & [active, rows] : config.verify_rows_by_active) {
            for (const int32_t context : config.context_buckets) {
                for (const int32_t row : rows) {
                    states.emplace(coordinate { context, active, row }, coordinate_state {});
                }
            }
        }
        if (!config.resume_sidecar_path.empty()) {
            load_sidecar(config.resume_sidecar_path);
        }
    }

    void validate_config() {
        if (config.source_identity.size() != 64 ||
            !std::all_of(config.source_identity.begin(), config.source_identity.end(), [](unsigned char ch) {
                return std::isxdigit(ch) != 0;
            })) {
            throw std::invalid_argument("SPS recorder source identity must be a 64-character SHA-256 fingerprint");
        }
        if (config.context_buckets.empty() || config.verify_rows_by_active.empty()) {
            throw std::invalid_argument("SPS recorder requires non-empty context and active-slot axes");
        }
        if (config.max_draft_tokens_per_slot < 0) {
            throw std::invalid_argument("SPS recorder max draft tokens must be non-negative");
        }
        if (config.retained_samples_per_coordinate == 0) {
            throw std::invalid_argument("SPS recorder retained sample count must be positive");
        }
        if (!std::isfinite(config.aggregate_quantile) ||
            config.aggregate_quantile <= 0.0 || config.aggregate_quantile > 1.0) {
            throw std::invalid_argument("SPS recorder aggregate quantile must be within (0, 1]");
        }

        std::sort(config.context_buckets.begin(), config.context_buckets.end());
        if (config.context_buckets.front() <= 0 ||
            std::adjacent_find(config.context_buckets.begin(), config.context_buckets.end()) !=
                    config.context_buckets.end()) {
            throw std::invalid_argument("SPS recorder context buckets must be unique and positive");
        }

        size_t coordinate_count = 0;
        for (auto & [active, rows] : config.verify_rows_by_active) {
            if (active <= 0 || rows.empty()) {
                throw std::invalid_argument("SPS recorder active-slot axes must be positive and non-empty");
            }
            std::sort(rows.begin(), rows.end());
            if (std::adjacent_find(rows.begin(), rows.end()) != rows.end()) {
                throw std::invalid_argument("SPS recorder row axes must contain unique values");
            }
            const int64_t max_rows = (int64_t) active *
                    (1 + (int64_t) config.max_draft_tokens_per_slot);
            if (rows.front() < active || rows.back() > max_rows) {
                throw std::invalid_argument("SPS recorder row axis exceeds its physical active-slot domain");
            }
            if (config.context_buckets.size() > SPS_MAX_ENTRIES / rows.size() ||
                config.context_buckets.size() * rows.size() > SPS_MAX_ENTRIES - coordinate_count) {
                throw std::invalid_argument("SPS recorder grid exceeds the SPS profile entry limit");
            }
            coordinate_count += config.context_buckets.size() * rows.size();
        }
    }

    std::optional<coordinate> coordinate_for(const common_speculative_sps_profile_sample & sample) const {
        if (sample.actual_context_tokens <= 0) {
            return std::nullopt;
        }
        const auto context = std::lower_bound(
                config.context_buckets.begin(), config.context_buckets.end(), sample.actual_context_tokens);
        if (context == config.context_buckets.end()) {
            return std::nullopt;
        }
        const auto active = config.verify_rows_by_active.find(sample.active_slots);
        if (active == config.verify_rows_by_active.end() ||
            !std::binary_search(active->second.begin(), active->second.end(), sample.total_verify_rows)) {
            return std::nullopt;
        }
        return coordinate { *context, sample.active_slots, sample.total_verify_rows };
    }

    bool coordinate_ready(const coordinate_state & state) const {
        return state.samples.size() >= config.retained_samples_per_coordinate;
    }

    bool valid_prefixes(
            int32_t active_slots,
            int32_t total_verify_rows,
            const std::vector<int32_t> & prefixes) const {
        if (active_slots <= 0 || prefixes.size() != (size_t) active_slots) {
            return false;
        }
        int64_t rows = active_slots;
        for (const int32_t prefix : prefixes) {
            if (prefix < 0 || prefix > config.max_draft_tokens_per_slot) {
                return false;
            }
            rows += prefix;
        }
        return rows == total_verify_rows;
    }

    size_t ready_coordinates() const {
        size_t result = 0;
        for (const auto & [_, state] : states) {
            result += coordinate_ready(state) ? 1 : 0;
        }
        return result;
    }

    std::optional<double> sample_quantile(const coordinate_state & state, double quantile) const {
        if (state.samples.empty()) {
            return std::nullopt;
        }
        std::vector<double> values;
        values.reserve(state.samples.size());
        for (const auto & sample : state.samples) {
            values.push_back(sample.cost_us);
        }
        std::sort(values.begin(), values.end());
        const size_t rank = std::max<size_t>(
                1, (size_t) std::ceil(quantile * values.size()));
        return values[std::min(rank, values.size()) - 1];
    }

    double aggregate(const coordinate_state & state) const {
        GGML_ASSERT(coordinate_ready(state));
        return *sample_quantile(state, config.aggregate_quantile);
    }

    void load_sidecar(const std::string & path) {
        auto fail = [&](const std::string & message) -> void {
            throw std::runtime_error("invalid SPS recorder sidecar '" + path + "': " + message);
        };
        auto exact_keys = [&](const json & value, const std::set<std::string> & expected, const char * where) {
            if (!value.is_object()) {
                fail(std::string(where) + " must be an object");
            }
            std::set<std::string> actual;
            for (auto it = value.begin(); it != value.end(); ++it) {
                actual.insert(it.key());
            }
            if (actual != expected) {
                fail(std::string(where) + " has missing or unknown fields");
            }
        };
        auto u64 = [&](const json & value, const char * key, const char * where) -> uint64_t {
            const auto & field = value.at(key);
            if (field.is_number_unsigned()) {
                return field.get<uint64_t>();
            }
            if (!field.is_number_integer()) {
                fail(std::string(where) + "." + key + " must be a non-negative integer");
            }
            const int64_t parsed = field.get<int64_t>();
            if (parsed < 0) {
                fail(std::string(where) + "." + key + " must be non-negative");
            }
            return (uint64_t) parsed;
        };
        auto i32 = [&](const json & value, const char * key, const char * where) -> int32_t {
            const auto & field = value.at(key);
            if (!field.is_number_integer()) {
                fail(std::string(where) + "." + key + " must be an integer");
            }
            const int64_t parsed = field.get<int64_t>();
            if (parsed <= 0 || parsed > std::numeric_limits<int32_t>::max()) {
                fail(std::string(where) + "." + key + " must be positive int32");
            }
            return (int32_t) parsed;
        };

        std::ifstream input = fs_open_ifstream(path, std::ios::in | std::ios::binary);
        if (!input) {
            fail("failed to open file");
        }
        std::ostringstream contents;
        contents << input.rdbuf();
        if (!input.good() && !input.eof()) {
            fail("failed to read file");
        }

        json root;
        bool duplicate_key = false;
        std::vector<std::set<std::string>> object_keys;
        try {
            root = json::parse(contents.str(), [&](int, json::parse_event_t event, json & parsed) {
                if (event == json::parse_event_t::object_start) {
                    object_keys.emplace_back();
                } else if (event == json::parse_event_t::key) {
                    if (object_keys.empty() || !object_keys.back().insert(parsed.get<std::string>()).second) {
                        duplicate_key = true;
                    }
                } else if (event == json::parse_event_t::object_end && !object_keys.empty()) {
                    object_keys.pop_back();
                }
                return true;
            });
        } catch (const std::exception & e) {
            fail(std::string("JSON parse failed: ") + e.what());
        }
        if (duplicate_key) {
            fail("duplicate JSON object member name");
        }

        exact_keys(root, {
            "schema_version", "profile_schema_version", "source_identity", "complete",
            "max_draft_tokens_per_slot", "context_buckets", "verify_rows_by_active",
            "retained_samples_per_coordinate", "warmup_samples_per_coordinate",
            "aggregate_quantile", "observations_total", "retained_total",
            "skipped_warmup_total", "skipped_capture_total", "skipped_ineligible_total",
            "skipped_outside_grid_total", "skipped_full_total", "ready_coordinates",
            "expected_coordinates", "coordinates",
        }, "root");

        if (!root.at("schema_version").is_number_integer() || root.at("schema_version").get<int64_t>() != 1 ||
            !root.at("profile_schema_version").is_number_integer() || root.at("profile_schema_version").get<int64_t>() != 2) {
            fail("unsupported schema version");
        }
        if (!root.at("source_identity").is_string() ||
            root.at("source_identity").get<std::string>() != config.source_identity ||
            !root.at("max_draft_tokens_per_slot").is_number_integer() ||
            root.at("max_draft_tokens_per_slot").get<int64_t>() != config.max_draft_tokens_per_slot ||
            root.at("context_buckets") != json(config.context_buckets) ||
            u64(root, "retained_samples_per_coordinate", "root") != config.retained_samples_per_coordinate ||
            u64(root, "warmup_samples_per_coordinate", "root") != config.warmup_samples_per_coordinate ||
            !root.at("aggregate_quantile").is_number() ||
            root.at("aggregate_quantile").get<double>() != config.aggregate_quantile) {
            fail("configuration does not match");
        }
        json expected_row_axes = json::object();
        for (const auto & [active, rows] : config.verify_rows_by_active) {
            expected_row_axes[std::to_string(active)] = rows;
        }
        if (root.at("verify_rows_by_active") != expected_row_axes) {
            fail("row axes do not match");
        }
        if (!root.at("coordinates").is_array() || root.at("coordinates").size() != states.size() ||
            u64(root, "expected_coordinates", "root") != states.size()) {
            fail("coordinate set does not match");
        }

        std::map<coordinate, coordinate_state> loaded;
        for (size_t index = 0; index < root.at("coordinates").size(); ++index) {
            const auto & row = root.at("coordinates").at(index);
            const std::string where = "coordinates[" + std::to_string(index) + "]";
            exact_keys(row, {
                "context_tokens", "active_slots", "total_verify_rows", "ready",
                "retained_samples", "warmup_seen", "skipped_warmup",
                "skipped_capture", "skipped_ineligible", "skipped_full",
                "p50_us", "p95_us", "max_us", "samples",
            }, where.c_str());
            const coordinate key {
                i32(row, "context_tokens", where.c_str()),
                i32(row, "active_slots", where.c_str()),
                i32(row, "total_verify_rows", where.c_str()),
            };
            if (states.find(key) == states.end() || loaded.find(key) != loaded.end()) {
                fail(where + " is duplicate or outside the configured grid");
            }

            coordinate_state state;
            const uint64_t warmup_seen = u64(row, "warmup_seen", where.c_str());
            if (warmup_seen > std::numeric_limits<uint32_t>::max()) {
                fail(where + ".warmup_seen exceeds uint32");
            }
            state.warmup_seen         = (uint32_t) warmup_seen;
            state.skipped_warmup      = u64(row, "skipped_warmup", where.c_str());
            state.skipped_capture     = u64(row, "skipped_capture", where.c_str());
            state.skipped_ineligible  = u64(row, "skipped_ineligible", where.c_str());
            state.skipped_full        = u64(row, "skipped_full", where.c_str());
            if (state.warmup_seen > config.warmup_samples_per_coordinate ||
                state.warmup_seen > state.skipped_warmup || !row.at("samples").is_array()) {
                fail(where + " has invalid warmup counters or samples");
            }

            for (size_t sample_index = 0; sample_index < row.at("samples").size(); ++sample_index) {
                const auto & sample_json = row.at("samples").at(sample_index);
                const std::string sample_where = where + ".samples[" + std::to_string(sample_index) + "]";
                exact_keys(sample_json, {
                    "actual_context_tokens", "cost_us", "execution_kind", "prefixes",
                }, sample_where.c_str());
                if (!sample_json.at("cost_us").is_number() ||
                    !std::isfinite(sample_json.at("cost_us").get<double>()) ||
                    sample_json.at("cost_us").get<double>() <= 0.0 ||
                    !sample_json.at("execution_kind").is_string() ||
                    !sample_json.at("prefixes").is_array()) {
                    fail(sample_where + " has invalid fields");
                }

                retained_sample sample;
                sample.actual_context_tokens = i32(sample_json, "actual_context_tokens", sample_where.c_str());
                sample.cost_us = sample_json.at("cost_us").get<double>();
                sample.execution_kind = execution_kind_from_name(
                        sample_json.at("execution_kind").get<std::string>(), path);
                if (sample.execution_kind == COMMON_SPECULATIVE_SPS_EXECUTION_DIRECT_WARMUP ||
                    sample.execution_kind == COMMON_SPECULATIVE_SPS_EXECUTION_GRAPH_CAPTURE) {
                    fail(sample_where + " retained a warmup or capture execution");
                }
                for (const auto & prefix : sample_json.at("prefixes")) {
                    if (!prefix.is_number_integer()) {
                        fail(sample_where + ".prefixes must contain integers");
                    }
                    const int64_t parsed = prefix.get<int64_t>();
                    if (parsed < 0 || parsed > std::numeric_limits<int32_t>::max()) {
                        fail(sample_where + ".prefixes contains an out-of-range value");
                    }
                    sample.prefixes.push_back((int32_t) parsed);
                }
                if (!valid_prefixes(std::get<1>(key), std::get<2>(key), sample.prefixes)) {
                    fail(sample_where + " has an inconsistent prefix distribution");
                }
                common_speculative_sps_profile_sample coordinate_probe;
                coordinate_probe.actual_context_tokens = sample.actual_context_tokens;
                coordinate_probe.active_slots = std::get<1>(key);
                coordinate_probe.total_verify_rows = std::get<2>(key);
                if (coordinate_for(coordinate_probe) != std::optional<coordinate>(key)) {
                    fail(sample_where + " maps to a different context bucket");
                }
                state.samples.push_back(std::move(sample));
            }
            if (state.samples.size() != u64(row, "retained_samples", where.c_str()) ||
                state.samples.size() > config.retained_samples_per_coordinate ||
                !row.at("ready").is_boolean() ||
                row.at("ready").get<bool>() != coordinate_ready(state)) {
                fail(where + " has inconsistent retained sample state");
            }
            if ((!state.samples.empty() &&
                    state.warmup_seen != config.warmup_samples_per_coordinate) ||
                (state.skipped_full > 0 && !coordinate_ready(state))) {
                fail(where + " contains a sampling history that the recorder cannot produce");
            }
            auto validate_stat = [&](const char * key_name, const std::optional<double> & expected) {
                const auto & value = row.at(key_name);
                if (!expected) {
                    if (!value.is_null()) {
                        fail(where + "." + key_name + " must be null without samples");
                    }
                } else if (!value.is_number() || !std::isfinite(value.get<double>()) ||
                           value.get<double>() != *expected) {
                    fail(where + "." + key_name + " does not match retained samples");
                }
            };
            validate_stat("p50_us", sample_quantile(state, 0.50));
            validate_stat("p95_us", sample_quantile(state, 0.95));
            validate_stat("max_us", sample_quantile(state, 1.00));
            loaded.emplace(key, std::move(state));
        }

        states = std::move(loaded);
        observations_total         = u64(root, "observations_total", "root");
        retained_total             = u64(root, "retained_total", "root");
        skipped_warmup_total       = u64(root, "skipped_warmup_total", "root");
        skipped_capture_total      = u64(root, "skipped_capture_total", "root");
        skipped_ineligible_total   = u64(root, "skipped_ineligible_total", "root");
        skipped_outside_grid_total = u64(root, "skipped_outside_grid_total", "root");
        skipped_full_total         = u64(root, "skipped_full_total", "root");

        uint64_t retained_sum = 0;
        uint64_t warmup_sum = 0;
        uint64_t capture_sum = 0;
        uint64_t ineligible_sum = 0;
        uint64_t full_sum = 0;
        auto checked_add = [&](uint64_t & total, uint64_t value, const char * what) {
            if (value > std::numeric_limits<uint64_t>::max() - total) {
                fail(std::string(what) + " counter sum overflows uint64");
            }
            total += value;
        };
        for (const auto & [_, state] : states) {
            checked_add(retained_sum,   state.samples.size(),       "retained");
            checked_add(warmup_sum,     state.skipped_warmup,       "warmup");
            checked_add(capture_sum,    state.skipped_capture,      "capture");
            checked_add(ineligible_sum, state.skipped_ineligible,   "ineligible");
            checked_add(full_sum,       state.skipped_full,         "full");
        }
        uint64_t observations_sum = 0;
        checked_add(observations_sum, retained_total,             "observations");
        checked_add(observations_sum, skipped_warmup_total,       "observations");
        checked_add(observations_sum, skipped_capture_total,      "observations");
        checked_add(observations_sum, skipped_ineligible_total,   "observations");
        checked_add(observations_sum, skipped_outside_grid_total, "observations");
        checked_add(observations_sum, skipped_full_total,         "observations");
        if (retained_total != retained_sum || skipped_warmup_total != warmup_sum ||
            skipped_capture_total != capture_sum || skipped_ineligible_total != ineligible_sum ||
            skipped_full_total != full_sum ||
            observations_total != observations_sum ||
            u64(root, "ready_coordinates", "root") != ready_coordinates() ||
            !root.at("complete").is_boolean() || root.at("complete").get<bool>() !=
                    (ready_coordinates() == states.size())) {
            fail("global counters or completion state are inconsistent");
        }
    }

    common_speculative_sps_profile make_profile() const {
        if (ready_coordinates() != states.size()) {
            throw std::runtime_error("SPS recorder profile is not sample-ready");
        }

        common_speculative_sps_profile result;
        result.schema_version = 2;
        result.max_draft_tokens_per_slot = config.max_draft_tokens_per_slot;
        result.entries.reserve(states.size());
        for (const auto & [key, state] : states) {
            result.entries.push_back({
                std::get<0>(key),
                std::get<1>(key),
                std::get<2>(key),
                aggregate(state),
            });
        }

        // Timing noise can make independently aggregated coordinates locally
        // non-monotonic. Lift each coordinate to the maximum measured value of
        // every point it dominates; never lower a measurement.
        const auto raw = result.entries;
        for (auto & target : result.entries) {
            for (const auto & source : raw) {
                if (dominates_or_equals(source, target)) {
                    target.cost_us = std::max(target.cost_us, source.cost_us);
                }
            }
        }

        // Keep serializer and strict parser behavior locked together.
        return common_speculative_sps_profile_parse(
                profile_to_json(result).dump(), "<recorder>");
    }

    json make_sidecar() const {
        json coordinates = json::array();
        for (const auto & [key, state] : states) {
            const auto stat_json = [](const std::optional<double> & value) {
                return value ? json(*value) : json(nullptr);
            };
            json samples = json::array();
            for (const auto & sample : state.samples) {
                samples.push_back({
                    { "actual_context_tokens", sample.actual_context_tokens },
                    { "cost_us", sample.cost_us },
                    { "execution_kind", execution_kind_name(sample.execution_kind) },
                    { "prefixes", sample.prefixes },
                });
            }
            coordinates.push_back({
                { "context_tokens", std::get<0>(key) },
                { "active_slots", std::get<1>(key) },
                { "total_verify_rows", std::get<2>(key) },
                { "ready", coordinate_ready(state) },
                { "retained_samples", state.samples.size() },
                { "warmup_seen", state.warmup_seen },
                { "skipped_warmup", state.skipped_warmup },
                { "skipped_capture", state.skipped_capture },
                { "skipped_ineligible", state.skipped_ineligible },
                { "skipped_full", state.skipped_full },
                { "p50_us", stat_json(sample_quantile(state, 0.50)) },
                { "p95_us", stat_json(sample_quantile(state, 0.95)) },
                { "max_us", stat_json(sample_quantile(state, 1.00)) },
                { "samples", std::move(samples) },
            });
        }

        json row_axes = json::object();
        for (const auto & [active, rows] : config.verify_rows_by_active) {
            row_axes[std::to_string(active)] = rows;
        }

        return {
            { "schema_version", 1 },
            { "profile_schema_version", 2 },
            { "source_identity", config.source_identity },
            { "complete", ready_coordinates() == states.size() },
            { "max_draft_tokens_per_slot", config.max_draft_tokens_per_slot },
            { "context_buckets", config.context_buckets },
            { "verify_rows_by_active", std::move(row_axes) },
            { "retained_samples_per_coordinate", config.retained_samples_per_coordinate },
            { "warmup_samples_per_coordinate", config.warmup_samples_per_coordinate },
            { "aggregate_quantile", config.aggregate_quantile },
            { "observations_total", observations_total },
            { "retained_total", retained_total },
            { "skipped_warmup_total", skipped_warmup_total },
            { "skipped_capture_total", skipped_capture_total },
            { "skipped_ineligible_total", skipped_ineligible_total },
            { "skipped_outside_grid_total", skipped_outside_grid_total },
            { "skipped_full_total", skipped_full_total },
            { "ready_coordinates", ready_coordinates() },
            { "expected_coordinates", states.size() },
            { "coordinates", std::move(coordinates) },
        };
    }

    common_speculative_sps_profile_recorder_config config;
    std::map<coordinate, coordinate_state> states;

    uint64_t observations_total          = 0;
    uint64_t retained_total              = 0;
    uint64_t skipped_warmup_total        = 0;
    uint64_t skipped_capture_total       = 0;
    uint64_t skipped_ineligible_total    = 0;
    uint64_t skipped_outside_grid_total  = 0;
    uint64_t skipped_full_total          = 0;
};

common_speculative_sps_profile_recorder::common_speculative_sps_profile_recorder(
        common_speculative_sps_profile_recorder_config config) :
    pimpl(new impl(std::move(config))) {
}

common_speculative_sps_profile_recorder::~common_speculative_sps_profile_recorder() = default;
common_speculative_sps_profile_recorder::common_speculative_sps_profile_recorder(
        common_speculative_sps_profile_recorder &&) noexcept = default;
common_speculative_sps_profile_recorder & common_speculative_sps_profile_recorder::operator=(
        common_speculative_sps_profile_recorder &&) noexcept = default;

common_speculative_sps_record_result common_speculative_sps_profile_recorder::observe(
        const common_speculative_sps_profile_sample & sample) {
    pimpl->observations_total++;

    const auto key = pimpl->coordinate_for(sample);
    if (!key) {
        pimpl->skipped_outside_grid_total++;
        return COMMON_SPECULATIVE_SPS_RECORD_SKIPPED_OUTSIDE_GRID;
    }
    auto & state = pimpl->states.at(*key);

    if (!sample.decode_succeeded || !sample.synchronized ||
        !sample.pure_generation || !sample.whole_batch ||
        !std::isfinite(sample.cost_us) || sample.cost_us <= 0.0 ||
        !pimpl->valid_prefixes(sample.active_slots, sample.total_verify_rows, sample.prefixes)) {
        state.skipped_ineligible++;
        pimpl->skipped_ineligible_total++;
        return COMMON_SPECULATIVE_SPS_RECORD_SKIPPED_INELIGIBLE;
    }
    if (sample.execution_kind == COMMON_SPECULATIVE_SPS_EXECUTION_DIRECT_WARMUP) {
        state.skipped_warmup++;
        pimpl->skipped_warmup_total++;
        return COMMON_SPECULATIVE_SPS_RECORD_SKIPPED_WARMUP;
    }
    if (sample.execution_kind == COMMON_SPECULATIVE_SPS_EXECUTION_GRAPH_CAPTURE) {
        state.skipped_capture++;
        pimpl->skipped_capture_total++;
        return COMMON_SPECULATIVE_SPS_RECORD_SKIPPED_CAPTURE;
    }
    if (state.warmup_seen < pimpl->config.warmup_samples_per_coordinate) {
        state.warmup_seen++;
        state.skipped_warmup++;
        pimpl->skipped_warmup_total++;
        return COMMON_SPECULATIVE_SPS_RECORD_SKIPPED_WARMUP;
    }
    if (pimpl->coordinate_ready(state)) {
        state.skipped_full++;
        pimpl->skipped_full_total++;
        return COMMON_SPECULATIVE_SPS_RECORD_SKIPPED_FULL;
    }

    state.samples.push_back({
        sample.actual_context_tokens,
        sample.cost_us,
        sample.execution_kind,
        sample.prefixes,
    });
    pimpl->retained_total++;

    if (!pimpl->coordinate_ready(state)) {
        return COMMON_SPECULATIVE_SPS_RECORD_RETAINED;
    }
    return ready() ? COMMON_SPECULATIVE_SPS_RECORD_ALL_READY :
            COMMON_SPECULATIVE_SPS_RECORD_COORDINATE_READY;
}

bool common_speculative_sps_profile_recorder::ready() const {
    return pimpl->ready_coordinates() == pimpl->states.size();
}

size_t common_speculative_sps_profile_recorder::expected_coordinates() const {
    return pimpl->states.size();
}

size_t common_speculative_sps_profile_recorder::ready_coordinates() const {
    return pimpl->ready_coordinates();
}

uint64_t common_speculative_sps_profile_recorder::retained_samples() const {
    return pimpl->retained_total;
}

uint64_t common_speculative_sps_profile_recorder::skipped_warmup() const {
    return pimpl->skipped_warmup_total;
}

uint64_t common_speculative_sps_profile_recorder::skipped_capture() const {
    return pimpl->skipped_capture_total;
}

uint64_t common_speculative_sps_profile_recorder::skipped_ineligible() const {
    return pimpl->skipped_ineligible_total;
}

uint64_t common_speculative_sps_profile_recorder::skipped_outside_grid() const {
    return pimpl->skipped_outside_grid_total;
}

uint64_t common_speculative_sps_profile_recorder::skipped_full() const {
    return pimpl->skipped_full_total;
}

common_speculative_sps_profile common_speculative_sps_profile_recorder::profile() const {
    return pimpl->make_profile();
}

void common_speculative_sps_profile_recorder::write_atomic(
        const std::string & profile_path,
        const std::string & sidecar_path) const {
    const bool is_ready = ready();
    if (!is_ready && std::filesystem::exists(utf8_path(profile_path))) {
        throw std::runtime_error(
                "incomplete SPS recorder state cannot coexist with a primary profile at '" + profile_path + "'");
    }
    // Commit the resumable source of truth first. If the process stops before
    // the primary replace, a complete sidecar can regenerate that profile on
    // the next startup; the reverse order can strand an incomplete sidecar
    // beside a new primary and must never be used.
    atomic_replace_json(sidecar_path, pimpl->make_sidecar());
    if (is_ready) {
        atomic_replace_json(profile_path, profile_to_json(profile()));
    }
}

common_speculative_sps_plan common_speculative_sps_plan_prefixes(
        const common_speculative_sps_profile & profile,
        int32_t context_tokens,
        int32_t active_slots,
        int32_t base_verify_rows,
        double base_useful_tokens,
        const std::vector<common_speculative_sps_slot> & slots) {
    common_speculative_sps_plan result;
    result.prefixes.assign(slots.size(), 0);

    if (context_tokens <= 0 || active_slots <= 0 || base_verify_rows < active_slots ||
        !std::isfinite(base_useful_tokens) || base_useful_tokens <= 0.0) {
        result.reason = "invalid planner dimensions";
        return result;
    }

    for (size_t i = 0; i < slots.size(); ++i) {
        const auto & slot = slots[i];
        if (slot.seq_id < 0 || slot.min_prefix < 0 || slot.max_prefix < slot.min_prefix ||
            (size_t) slot.max_prefix > slot.survival.size()) {
            result.reason = "invalid per-slot prefix bounds";
            return result;
        }

        float previous = 1.0f;
        for (int32_t pos = 0; pos < slot.max_prefix; ++pos) {
            const float survival = slot.survival[pos];
            if (!std::isfinite(survival) || survival < 0.0f || survival > 1.0f || survival > previous) {
                result.reason = "invalid or non-monotonic DSpark survival";
                return result;
            }
            previous = survival;
        }
    }

    int32_t max_profile_rows = 0;
    for (const auto & entry : profile.entries) {
        if (profile.schema_version == 2 && entry.active_slots != active_slots) {
            continue;
        }
        max_profile_rows = std::max(max_profile_rows, entry.total_verify_rows);
    }
    if (base_verify_rows > max_profile_rows) {
        result.reason = "SPS table does not cover this decode tick";
        return result;
    }
    const int32_t max_extra_rows = max_profile_rows - base_verify_rows;

    struct dp_state {
        double utility = -std::numeric_limits<double>::infinity();
        std::vector<int32_t> prefixes;
    };

    std::map<int32_t, dp_state> dp;
    dp.emplace(0, dp_state { 0.0, {} });

    for (const auto & slot : slots) {
        const int32_t capped_max_prefix = std::min(slot.max_prefix, max_extra_rows);
        std::vector<double> prefix_utility((size_t) capped_max_prefix + 1, 0.0);
        for (int32_t prefix = 1; prefix <= capped_max_prefix; ++prefix) {
            prefix_utility[prefix] = prefix_utility[prefix - 1] + slot.survival[prefix - 1];
        }

        std::map<int32_t, dp_state> next;
        for (const auto & [rows, state] : dp) {
            for (int64_t prefix_64 = 0; prefix_64 <= capped_max_prefix; ++prefix_64) {
                const int32_t prefix = (int32_t) prefix_64;
                if (prefix > 0 && prefix < slot.min_prefix) {
                    continue;
                }
                const int64_t rows_next_64 = (int64_t) rows + prefix;
                if (rows_next_64 > max_extra_rows) {
                    continue;
                }
                const int32_t rows_next = (int32_t) rows_next_64;
                const double utility_next = state.utility + prefix_utility[prefix];
                auto & dst = next[rows_next];
                if (utility_next > dst.utility) {
                    dst.utility = utility_next;
                    dst.prefixes = state.prefixes;
                    dst.prefixes.push_back(prefix);
                }
            }
        }
        dp = std::move(next);
    }

    bool found = false;
    for (const auto & [extra_rows, state] : dp) {
        const int64_t rows_64 = (int64_t) base_verify_rows + extra_rows;
        if (rows_64 > std::numeric_limits<int32_t>::max()) {
            continue;
        }

        const auto cost = profile.lookup_cost_us(context_tokens, active_slots, (int32_t) rows_64);
        if (!cost) {
            continue;
        }

        const double useful = base_useful_tokens + state.utility;
        if (!std::isfinite(useful)) {
            continue;
        }
        const double score  = useful / *cost;
        const int ratio_order = found ? compare_positive_ratios(
                useful, *cost, result.expected_useful_tokens, result.predicted_cost_us) : 1;
        const bool better = !found ||
                ratio_order > 0 ||
                (ratio_order == 0 &&
                    (useful > result.expected_useful_tokens ||
                     (useful == result.expected_useful_tokens && rows_64 < result.total_verify_rows)));
        if (!better) {
            continue;
        }

        found = true;
        result.valid                  = true;
        result.prefixes               = state.prefixes;
        result.total_verify_rows      = (int32_t) rows_64;
        result.expected_useful_tokens = useful;
        result.predicted_cost_us      = *cost;
        result.expected_tokens_per_us = std::isfinite(score) ? score : std::numeric_limits<double>::max();
        result.reason.clear();
    }

    if (!found) {
        result.reason = "SPS table does not cover this decode tick";
    }

    return result;
}
