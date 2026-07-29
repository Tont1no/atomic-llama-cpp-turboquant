#pragma once

#include <cstddef>
#include <string>
#include <utility>

namespace diffusion_answer_metrics {

struct VisibleAnswer {
    std::string text;
    int token_count = 0;
    bool duplicate_trimmed = false;
};

// Normalize the exact text returned to the client, then count that normalized
// text. The counter is injected so the server can use the model vocabulary
// while Ai-Loader's model-free self-test can verify the ordering contract.
template <typename TokenCounter>
VisibleAnswer normalize(std::string answer, TokenCounter && count_tokens) {
    VisibleAnswer out;
    out.text = std::move(answer);

    std::string candidate = out.text;
    const size_t first = candidate.find_first_not_of(" \n\t");
    if (first != std::string::npos) {
        candidate = candidate.substr(first);
    }

    const size_t half = candidate.size() / 2;
    if (half > 0 &&
        candidate.compare(0, half, candidate, candidate.size() - half, half) == 0) {
        out.text = candidate.substr(0, half);
        out.duplicate_trimmed = true;
    }

    out.token_count = static_cast<int>(count_tokens(out.text));
    return out;
}

} // namespace diffusion_answer_metrics
