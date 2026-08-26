#pragma once

#include "common.h"

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

struct common_suffix_draft_config {
    uint16_t match_min = 4;
    uint16_t match_max = 64;
    uint16_t draft_min = 2;
    uint16_t draft_max = 64;
    uint16_t max_candidates = 64;
    size_t history_max = 2 * 1024 * 1024;

    float min_confidence = 0.75f;
    float acceptance_alpha = 0.25f;
};

struct common_suffix_draft_metrics {
    uint64_t lookups = 0;
    uint64_t hits = 0;
    uint64_t weak_misses = 0;
    uint64_t proposed_tokens = 0;

    size_t last_match = 0;
    float last_confidence = 0.0f;
};

class common_suffix_drafter {
public:
    explicit common_suffix_drafter(common_suffix_draft_config config = {});

    void begin(const llama_tokens & prompt);
    llama_tokens draft(const llama_tokens & prompt, llama_token sampled, int32_t n_max = -1);
    void accept(uint16_t n_accepted);
    void reset();

    const common_suffix_draft_metrics & metrics() const;
    size_t history_size() const;

private:
    common_suffix_draft_config config;
    common_suffix_draft_metrics counters;

    llama_tokens history;
    llama_tokens active;
    std::unordered_map<uint64_t, std::vector<size_t>> anchors;

    size_t last_offered = 0;
    float acceptance_ema = 1.0f;
    bool has_acceptance = false;

    uint64_t anchor_hash(size_t end) const;
    bool anchor_valid(size_t end) const;
    void append(llama_token token);
    void append_boundary();
    void append_segment(const llama_tokens & tokens);
    void prune_history();
    void rebuild(const llama_tokens & tokens);
    void sync_for_draft(const llama_tokens & prompt, llama_token sampled);
};
