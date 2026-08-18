#include "speculative-sps.h"

#include "common.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>

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
            entry.active_slots < active_slots ||
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

    require_exact_keys(root, { "schema_version", "entries" }, source, "root");

    if (!root.at("schema_version").is_number_integer() || root.at("schema_version").get<int64_t>() != 1) {
        sps_error(source, "schema_version must be integer 1");
    }

    const auto & rows = root.at("entries");
    if (!rows.is_array() || rows.empty() || rows.size() > SPS_MAX_ENTRIES) {
        sps_error(source, "entries must be a non-empty array with at most " + std::to_string(SPS_MAX_ENTRIES) + " rows");
    }

    common_speculative_sps_profile result;
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

    // A component-wise ceiling is only unambiguous on a complete measured
    // grid. Reject sparse crossing dimensions instead of selecting an
    // incomparable point that could underestimate the real cost. Coordinates
    // where verify rows are smaller than active slots are structurally invalid
    // and therefore are not required.
    std::set<int32_t> contexts;
    std::set<int32_t> slot_counts;
    std::set<int32_t> row_counts;
    for (const auto & entry : result.entries) {
        contexts.insert(entry.context_tokens);
        slot_counts.insert(entry.active_slots);
        row_counts.insert(entry.total_verify_rows);
    }
    // First compute the number of structurally valid Cartesian coordinates.
    // A complete profile cannot contain more coordinates than entries, so
    // fail before enumerating a potentially huge sparse axis product. Because
    // every parsed entry is itself a valid, unique coordinate, equality is
    // also sufficient to prove that the valid grid is complete.
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
