#include "speculative-sps.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#undef NDEBUG
#include <cassert>

static const char * VALID_PROFILE = R"JSON({
  "schema_version": 1,
  "entries": [
    {"context_tokens": 1024, "active_slots": 2, "total_verify_rows": 2, "cost_us": 1.0},
    {"context_tokens": 1024, "active_slots": 2, "total_verify_rows": 3, "cost_us": 1.1},
    {"context_tokens": 1024, "active_slots": 2, "total_verify_rows": 4, "cost_us": 1.4},
    {"context_tokens": 1024, "active_slots": 2, "total_verify_rows": 5, "cost_us": 2.0},
    {"context_tokens": 1024, "active_slots": 2, "total_verify_rows": 6, "cost_us": 3.0}
  ]
})JSON";

template<typename Fn>
static void assert_throws(Fn && fn) {
    bool threw = false;
    try {
        fn();
    } catch (const std::runtime_error &) {
        threw = true;
    }
    assert(threw);
}

static void test_parse_and_lookup() {
    const auto profile = common_speculative_sps_profile_parse(VALID_PROFILE);
    assert(profile.entries.size() == 5);

    assert(profile.lookup_cost_us(1024, 2, 2).value() == 1.0);
    assert(profile.lookup_cost_us(512, 1, 3).value() == 1.1);
    assert(!profile.lookup_cost_us(2048, 2, 2));
    assert(!profile.lookup_cost_us(1024, 3, 3));
    assert(!profile.lookup_cost_us(1024, 2, 7));

    assert_throws([] {
        common_speculative_sps_profile_parse(R"({"schema_version":1,"entries":[],"extra":true})");
    });
    assert_throws([] {
        common_speculative_sps_profile_parse(R"({"schema_version":2,"entries":[]})");
    });
    assert_throws([] {
        common_speculative_sps_profile_parse(R"({
          "schema_version": 1,
          "entries": [
            {"context_tokens": 1024,"active_slots": 2,"total_verify_rows": 2,"cost_us": 2.0},
            {"context_tokens": 1024,"active_slots": 2,"total_verify_rows": 3,"cost_us": 1.0}
          ]
        })");
    });
    assert_throws([] {
        common_speculative_sps_profile_parse(R"({
          "schema_version": 1,
          "entries": [
            {"context_tokens": 1024,"active_slots": 2,"total_verify_rows": 2,"cost_us": 1.0},
            {"context_tokens": 1024,"active_slots": 2,"total_verify_rows": 2,"cost_us": 1.0}
          ]
        })");
    });
    assert_throws([] {
        common_speculative_sps_profile_parse(R"({
          "schema_version": 1,
          "entries": [
            {"context_tokens": 1024,"active_slots": 2,"total_verify_rows": 2,"cost_us": 1.0},
            {"context_tokens": 2048,"active_slots": 4,"total_verify_rows": 4,"cost_us": 2.0}
          ]
        })");
    });
    assert_throws([] {
        common_speculative_sps_profile_parse(R"({
          "schema_version": 1,
          "schema_version": 1,
          "entries": [
            {"context_tokens": 1024,"active_slots": 1,"total_verify_rows": 1,"cost_us": 1.0}
          ]
        })");
    });
    assert_throws([] {
        common_speculative_sps_profile_parse(R"({
          "schema_version": 1,
          "entries": [
            {"context_tokens": 1024,"active_slots": 1,"total_verify_rows": 1,"cost_us": 1.0,"cost_us": 2.0}
          ]
        })");
    });
}

static void test_file_loader() {
    const auto nonce = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    const auto path = std::filesystem::temp_directory_path() /
            ("llama-sps-profile-" + std::to_string(nonce) + ".json");
    {
        std::ofstream output(path, std::ios::binary);
        assert(output.good());
        output << VALID_PROFILE;
    }

    const auto loaded = common_speculative_sps_profile_load(path.string());
    assert(loaded.entries.size() == 5);
    std::filesystem::remove(path);

    assert_throws([&] {
        common_speculative_sps_profile_load(path.string());
    });
}

static void test_global_prefix_plan() {
    const auto profile = common_speculative_sps_profile_parse(VALID_PROFILE);

    const std::vector<common_speculative_sps_slot> slots = {
        { 0, { 0.9f, 0.8f }, 1, 2 },
        { 1, { 0.7f, 0.1f }, 1, 2 },
    };

    const auto plan = common_speculative_sps_plan_prefixes(
            profile,
            /* context_tokens    = */ 900,
            /* active_slots      = */ 2,
            /* base_verify_rows  = */ 2,
            /* base_useful       = */ 2.0,
            slots);

    assert(plan.valid);
    assert(plan.prefixes.size() == 2);
    assert(plan.prefixes[0] == 2);
    assert(plan.prefixes[1] == 0);
    assert(plan.total_verify_rows == 4);
    assert(plan.expected_useful_tokens > 3.69 && plan.expected_useful_tokens < 3.71);
    assert(plan.predicted_cost_us == 1.4);

    auto invalid_slots = slots;
    invalid_slots[0].survival = { 0.5f, 0.7f };
    const auto invalid = common_speculative_sps_plan_prefixes(profile, 900, 2, 2, 2.0, invalid_slots);
    assert(!invalid.valid);
    assert(!invalid.reason.empty());

    const auto uncovered = common_speculative_sps_plan_prefixes(profile, 2048, 2, 2, 2.0, slots);
    assert(!uncovered.valid);
    assert(!uncovered.reason.empty());
}

static void test_score_order_and_profile_row_cap() {
    const auto tiny_score_profile = common_speculative_sps_profile_parse(R"({
      "schema_version": 1,
      "entries": [
        {"context_tokens": 1024,"active_slots": 1,"total_verify_rows": 1,"cost_us": 1000000000000000.0},
        {"context_tokens": 1024,"active_slots": 1,"total_verify_rows": 2,"cost_us": 2000000000000000.0}
      ]
    })");
    const std::vector<common_speculative_sps_slot> low_value_slot = {
        { 0, { 0.1f }, 1, 1 },
    };
    const auto tiny_score_plan = common_speculative_sps_plan_prefixes(
            tiny_score_profile, 100, 1, 1, 1.0, low_value_slot);
    assert(tiny_score_plan.valid);
    assert(tiny_score_plan.prefixes == std::vector<int32_t>({ 0 }));

    const auto subnormal_cost_profile = common_speculative_sps_profile_parse(R"({
      "schema_version": 1,
      "entries": [
        {"context_tokens": 1024,"active_slots": 1,"total_verify_rows": 1,"cost_us": 1e-320},
        {"context_tokens": 1024,"active_slots": 1,"total_verify_rows": 2,"cost_us": 1e-319}
      ]
    })");
    const std::vector<common_speculative_sps_slot> superficially_useful_slot = {
        { 0, { 1.0f }, 1, 1 },
    };
    const auto subnormal_cost_plan = common_speculative_sps_plan_prefixes(
            subnormal_cost_profile, 100, 1, 1, 1.0, superficially_useful_slot);
    assert(subnormal_cost_plan.valid);
    assert(subnormal_cost_plan.prefixes == std::vector<int32_t>({ 0 }));

    const auto adjacent_exponent_profile = common_speculative_sps_profile_parse(R"({
      "schema_version": 1,
      "entries": [
        {"context_tokens": 1024,"active_slots": 1,"total_verify_rows": 1,"cost_us": 1.0},
        {"context_tokens": 1024,"active_slots": 1,"total_verify_rows": 2,"cost_us": 1.0},
        {"context_tokens": 1024,"active_slots": 1,"total_verify_rows": 3,"cost_us": 1.9}
      ]
    })");
    const std::vector<common_speculative_sps_slot> adjacent_exponent_slot = {
        { 0, { 0.9f, 0.1f }, 1, 2 },
    };
    const auto adjacent_exponent_plan = common_speculative_sps_plan_prefixes(
            adjacent_exponent_profile, 100, 1, 1, 1.0, adjacent_exponent_slot);
    assert(adjacent_exponent_plan.valid);
    assert(adjacent_exponent_plan.prefixes == std::vector<int32_t>({ 1 }));

    const auto capped_profile = common_speculative_sps_profile_parse(R"({
      "schema_version": 1,
      "entries": [
        {"context_tokens": 1024,"active_slots": 1,"total_verify_rows": 1,"cost_us": 1.0},
        {"context_tokens": 1024,"active_slots": 1,"total_verify_rows": 2,"cost_us": 1.1}
      ]
    })");
    const std::vector<common_speculative_sps_slot> long_slot = {
        { 0, std::vector<float>(1024, 0.9f), 1, 1024 },
    };
    const auto capped_plan = common_speculative_sps_plan_prefixes(
            capped_profile, 100, 1, 1, 1.0, long_slot);
    assert(capped_plan.valid);
    assert(capped_plan.prefixes.size() == 1);
    assert(capped_plan.prefixes[0] <= 1);
    assert(capped_plan.total_verify_rows <= 2);
}

int main() {
    test_parse_and_lookup();
    test_file_loader();
    test_global_prefix_plan();
    test_score_order_and_profile_row_cap();
    return 0;
}
