#include "suffix-draft.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

namespace {

struct suffix_candidate {
    size_t end = 0;
    size_t match = 0;
    llama_token next = LLAMA_TOKEN_NULL;
};

static float clamp_probability(float value) {
    return std::max(0.0f, std::min(1.0f, value));
}

}

common_suffix_drafter::common_suffix_drafter(common_suffix_draft_config config) : config(config) {
    if (config.match_min == 0 || config.match_min > config.match_max ||
            config.draft_min == 0 || config.draft_min > config.draft_max ||
            config.max_candidates == 0 || config.history_max < config.match_max ||
            !std::isfinite(config.min_confidence) ||
            config.min_confidence < 0.0f || config.min_confidence > 1.0f ||
            !std::isfinite(config.acceptance_alpha) || config.acceptance_alpha <= 0.0f ||
            config.acceptance_alpha > 1.0f) {
        throw std::invalid_argument("invalid suffix draft configuration");
    }
}

uint64_t common_suffix_drafter::anchor_hash(size_t end) const {
    uint64_t hash = 1469598103934665603ULL;
    const size_t begin = end + 1 - config.match_min;
    for (size_t i = begin; i <= end; ++i) {
        hash ^= static_cast<uint32_t>(history[i]);
        hash *= 1099511628211ULL;
    }
    return hash;
}

void common_suffix_drafter::append(llama_token token) {
    history.push_back(token);
    if (history.size() < config.match_min || !anchor_valid(history.size() - 1)) {
        return;
    }

    const size_t end = history.size() - 1;
    anchors[anchor_hash(end)].push_back(end);
}

bool common_suffix_drafter::anchor_valid(size_t end) const {
    if (end + 1 < config.match_min) {
        return false;
    }
    const size_t begin = end + 1 - config.match_min;
    return std::find(history.begin() + begin, history.begin() + end + 1, LLAMA_TOKEN_NULL) ==
            history.begin() + end + 1;
}

void common_suffix_drafter::append_boundary() {
    if (!history.empty() && history.back() != LLAMA_TOKEN_NULL) {
        history.push_back(LLAMA_TOKEN_NULL);
    }
}

void common_suffix_drafter::append_segment(const llama_tokens & tokens) {
    append_boundary();
    for (llama_token token : tokens) {
        append(token);
    }
    active = tokens;
    prune_history();
}

void common_suffix_drafter::prune_history() {
    if (history.size() > active.size() + config.history_max) {
        rebuild(active);
    }
}

void common_suffix_drafter::rebuild(const llama_tokens & tokens) {
    history.clear();
    anchors.clear();
    history.reserve(tokens.size());
    for (llama_token token : tokens) {
        append(token);
    }
    active = tokens;
}

void common_suffix_drafter::begin(const llama_tokens & prompt) {
    last_offered = 0;

    if (active.size() <= prompt.size() && std::equal(active.begin(), active.end(), prompt.begin())) {
        for (size_t i = active.size(); i < prompt.size(); ++i) {
            append(prompt[i]);
        }
        active = prompt;
        prune_history();
        return;
    }

    append_segment(prompt);
}

void common_suffix_drafter::sync_for_draft(const llama_tokens & prompt, llama_token sampled) {
    if (active.size() <= prompt.size() && std::equal(active.begin(), active.end(), prompt.begin())) {
        for (size_t i = active.size(); i < prompt.size(); ++i) {
            append(prompt[i]);
        }
        active = prompt;
        prune_history();
    } else if (active.size() == prompt.size() + 1 &&
            std::equal(prompt.begin(), prompt.end(), active.begin()) && active.back() == sampled) {
        return;
    } else if (active.size() != prompt.size() || !std::equal(prompt.begin(), prompt.end(), active.begin())) {
        append_segment(prompt);
    }

    append(sampled);
    active.push_back(sampled);
    prune_history();
}

llama_tokens common_suffix_drafter::draft(
        const llama_tokens & prompt, llama_token sampled, int32_t n_max) {
    counters.lookups++;
    counters.last_match = 0;
    counters.last_confidence = 0.0f;
    last_offered = 0;

    const int32_t requested_max = n_max < 0 ? config.draft_max : std::min<int32_t>(n_max, config.draft_max);
    if (requested_max < config.draft_min) {
        counters.weak_misses++;
        return {};
    }

    sync_for_draft(prompt, sampled);
    if (history.size() <= config.match_min) {
        counters.weak_misses++;
        return {};
    }

    const size_t current_end = history.size() - 1;
    const auto anchor_it = anchors.find(anchor_hash(current_end));
    if (anchor_it == anchors.end()) {
        counters.weak_misses++;
        return {};
    }

    std::vector<suffix_candidate> candidates;
    candidates.reserve(std::min<size_t>(anchor_it->second.size(), config.max_candidates));

    size_t checked = 0;
    for (auto it = anchor_it->second.rbegin(); it != anchor_it->second.rend(); ++it) {
        const size_t candidate_end = *it;
        if (candidate_end >= current_end) {
            continue;
        }
        if (++checked > config.max_candidates) {
            break;
        }

        bool anchor_matches = true;
        for (size_t i = 0; i < config.match_min; ++i) {
            if (history[candidate_end - i] != history[current_end - i]) {
                anchor_matches = false;
                break;
            }
        }
        if (!anchor_matches) {
            continue;
        }

        size_t match = config.match_min;
        while (match < config.match_max && candidate_end >= match && current_end >= match &&
                history[candidate_end - match] != LLAMA_TOKEN_NULL &&
                history[current_end - match] != LLAMA_TOKEN_NULL &&
                history[candidate_end - match] == history[current_end - match]) {
            ++match;
        }

        candidates.push_back({ candidate_end, match, history[candidate_end + 1] });
    }

    if (candidates.empty()) {
        counters.weak_misses++;
        return {};
    }

    const size_t best_match = std::max_element(candidates.begin(), candidates.end(),
            [](const suffix_candidate & a, const suffix_candidate & b) {
                return a.match < b.match;
            })->match;

    std::unordered_map<llama_token, size_t> next_counts;
    for (const auto & candidate : candidates) {
        if (candidate.match == best_match) {
            next_counts[candidate.next]++;
        }
    }

    llama_token best_next = LLAMA_TOKEN_NULL;
    size_t best_next_count = 0;
    size_t top_count = 0;
    for (const auto & item : next_counts) {
        top_count += item.second;
        if (item.second > best_next_count ||
                (item.second == best_next_count && (best_next == LLAMA_TOKEN_NULL || item.first < best_next))) {
            best_next = item.first;
            best_next_count = item.second;
        }
    }

    const float length_confidence = std::min(1.0f,
            static_cast<float>(best_match) / static_cast<float>(2 * config.match_min));
    const float agreement = top_count == 0 ? 0.0f :
            static_cast<float>(best_next_count) / static_cast<float>(top_count);
    const float confidence = clamp_probability(length_confidence * agreement);

    counters.last_match = best_match;
    counters.last_confidence = confidence;
    if (confidence < config.min_confidence) {
        counters.weak_misses++;
        return {};
    }

    const auto selected = std::find_if(candidates.begin(), candidates.end(),
            [&](const suffix_candidate & candidate) {
                return candidate.match == best_match && candidate.next == best_next;
            });
    if (selected == candidates.end()) {
        counters.weak_misses++;
        return {};
    }

    const float quality = confidence * (has_acceptance ? acceptance_ema : 1.0f);
    const size_t adaptive_max = config.draft_min + static_cast<size_t>(
            std::floor((requested_max - config.draft_min) * clamp_probability(quality)));
    size_t available = 0;
    while (selected->end + available + 1 < history.size() &&
            history[selected->end + available + 1] != LLAMA_TOKEN_NULL) {
        ++available;
    }
    const size_t count = std::min(adaptive_max, available);
    if (count < config.draft_min) {
        counters.weak_misses++;
        return {};
    }

    llama_tokens result;
    result.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        result.push_back(history[selected->end + 1 + i]);
    }

    last_offered = result.size();
    counters.hits++;
    counters.proposed_tokens += result.size();
    return result;
}

void common_suffix_drafter::accept(uint16_t n_accepted) {
    if (last_offered == 0) {
        return;
    }

    const float sample = static_cast<float>(std::min<size_t>(n_accepted, last_offered)) /
            static_cast<float>(last_offered);
    acceptance_ema = has_acceptance ?
            config.acceptance_alpha * sample + (1.0f - config.acceptance_alpha) * acceptance_ema : sample;
    has_acceptance = true;
    last_offered = 0;
}

void common_suffix_drafter::reset() {
    common_suffix_draft_metrics fresh;
    counters = fresh;
    history.clear();
    active.clear();
    anchors.clear();
    last_offered = 0;
    acceptance_ema = 1.0f;
    has_acceptance = false;
}

const common_suffix_draft_metrics & common_suffix_drafter::metrics() const {
    return counters;
}

size_t common_suffix_drafter::history_size() const {
    return history.size();
}
