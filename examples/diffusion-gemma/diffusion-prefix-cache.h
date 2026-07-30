#pragma once

#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

// Pure, model-free policy for the dedicated DiffusionGemma server's single
// physical KV cache. The HTTP server owns one llama_context and serializes all
// generation, so at most one prompt prefix can be warm at a time.
//
// `scope` is an opaque, loader-authenticated value. The policy deliberately
// refuses an empty scope and requires exact scope/config equality before even
// comparing tokens. No tenant, user, session, prompt, or scope value is logged
// or exported by this type.
template <typename Token>
class diffusion_prompt_prefix_cache {
public:
    struct reuse_plan {
        size_t cached_tokens = 0;
        bool exact_scope = false;
        bool exact_config = false;
    };

    reuse_plan plan(
            const std::string & scope,
            const std::string & config,
            const std::vector<Token> & prompt_tokens) const {
        reuse_plan out;
        if (!valid_ || scope.empty() || config.empty() || prompt_tokens.empty()) {
            return out;
        }
        out.exact_scope = scope == scope_;
        out.exact_config = config == config_;
        if (!out.exact_scope || !out.exact_config) {
            return out;
        }
        const size_t limit = std::min(tokens_.size(), prompt_tokens.size());
        while (out.cached_tokens < limit &&
               tokens_[out.cached_tokens] == prompt_tokens[out.cached_tokens]) {
            ++out.cached_tokens;
        }
        return out;
    }

    void commit(
            std::string scope,
            std::string config,
            std::vector<Token> prompt_tokens) {
        if (scope.empty() || config.empty() || prompt_tokens.empty()) {
            invalidate();
            return;
        }
        scope_ = std::move(scope);
        config_ = std::move(config);
        tokens_ = std::move(prompt_tokens);
        valid_ = true;
    }

    void invalidate() {
        valid_ = false;
        scope_.clear();
        config_.clear();
        tokens_.clear();
    }

    bool valid() const {
        return valid_;
    }

    size_t token_count() const {
        return valid_ ? tokens_.size() : 0;
    }

private:
    bool valid_ = false;
    std::string scope_;
    std::string config_;
    std::vector<Token> tokens_;
};
