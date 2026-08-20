#include "server-task.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <utility>
#include <vector>

static void require(bool condition) {
    if (!condition) {
        std::abort();
    }
}

static json native_partial(llama_token token, const std::string & content, int32_t n_decoded, bool suppress_content = false) {
    completion_token_output token_output;
    token_output.tok          = token;
    token_output.text_to_send = content;
    auto payload = server_make_stream_partial_payload(token_output, suppress_content);

    server_task_result_cmpl_partial result;
    result.content          = std::move(payload.content);
    result.tokens           = std::move(payload.tokens);
    result.n_decoded        = n_decoded;
    result.n_prompt_tokens  = 736;
    result.res_type         = TASK_RESPONSE_TYPE_NONE;
    return result.to_json_non_oaicompat();
}

int main() {
    // An incomplete UTF-8 piece needs a native token-only partial only when
    // the caller explicitly requested raw token IDs.
    require(server_should_emit_native_token_only_partial(TASK_RESPONSE_TYPE_NONE, true, true, true));
    require(!server_should_emit_native_token_only_partial(TASK_RESPONSE_TYPE_NONE, true, false, true));
    require(!server_should_emit_native_token_only_partial(TASK_RESPONSE_TYPE_NONE, true, true, false));
    require(!server_should_emit_native_token_only_partial(TASK_RESPONSE_TYPE_OAI_CMPL, true, true, true));

    // Exercise the same decision and payload seams used by process_token() and
    // send_partial_response(): the first token ends in incomplete UTF-8 and is
    // emitted ID-only; the next token carries the completed text and only its
    // own ID.
    const bool first_is_incomplete = true;
    require(server_should_emit_native_token_only_partial(
            TASK_RESPONSE_TYPE_NONE, true, true, first_is_incomplete));
    const json first = native_partial(7001, "\xC3", 1, true);
    const json second = native_partial(7002, "\xC3\xA9", 2);
    require(first.at("content") == "");
    require(first.at("tokens") == json::array({ 7001 }));
    require(second.at("content") == "\xC3\xA9");
    require(second.at("tokens") == json::array({ 7002 }));

    // Empty content and special-token pieces still carry exactly one raw ID.
    const json empty = native_partial(32001, "", 1);
    require(empty.at("content") == "");
    require(empty.at("tokens") == json::array({ 32001 }));

    // A normal content-bearing partial is unchanged.
    const json normal = native_partial(42, "hello", 2);
    require(normal.at("content") == "hello");
    require(normal.at("tokens") == json::array({ 42 }));

    // Speculative verification may accept several tokens in one backend
    // batch. process_token() visits each accepted ID, so the serialized stream
    // must preserve all 256 IDs exactly once, including token-only chunks.
    std::vector<llama_token> observed;
    observed.reserve(256);
    constexpr size_t accepted_group_sizes[] = { 7, 1, 3, 8, 2, 5, 4, 6 };
    size_t group = 0;
    for (int32_t ordinal = 0; ordinal < 256;) {
        const size_t count = std::min<size_t>(accepted_group_sizes[group++ % 8], 256 - ordinal);
        for (size_t i = 0; i < count; ++i, ++ordinal) {
            const llama_token token = 1000 + ordinal;
            const bool buffered_piece = ordinal % 61 == 0;
            const json partial = native_partial(token, buffered_piece ? "" : "x", ordinal + 1);
            require(partial.at("tokens").size() == 1);
            observed.push_back(partial.at("tokens").at(0).get<llama_token>());
        }
    }
    require(observed.size() == 256);
    for (int32_t ordinal = 0; ordinal < 256; ++ordinal) {
        require(observed[ordinal] == 1000 + ordinal);
    }

    // OpenAI serialization remains content-only and never exposes native IDs.
    server_task_result_cmpl_partial oai;
    oai.content          = "hello";
    oai.tokens           = { 42 };
    oai.n_decoded        = 1;
    oai.n_prompt_tokens  = 736;
    oai.res_type         = TASK_RESPONSE_TYPE_OAI_CMPL;
    oai.oaicompat_model  = "test";
    oai.oaicompat_cmpl_id = "cmpl-test";
    const json oai_json = oai.to_json_oaicompat();
    require(!oai_json.contains("tokens"));
    require(oai_json.at("choices").at(0).at("text") == "hello");

    // The public Prometheus surface must preserve the recurrent depth values,
    // not merely expose correctly named zero-valued gauges.
    server_task_result_metrics metrics_result;
    metrics_result.n_idle_slots                              = 1;
    metrics_result.n_processing_slots                        = 0;
    metrics_result.n_tasks_deferred                          = 0;
    metrics_result.metrics.recurrent_configured_depth        = 7;
    metrics_result.metrics.recurrent_resident_depth          = 7;
    metrics_result.metrics.recurrent_required_depth          = 7;
    metrics_result.metrics.recurrent_pending_depth           = UINT32_MAX;
    const std::string prometheus = metrics_result.to_metrics();
    require(prometheus.find("llamacpp:recurrent_snapshot_configured_depth 7\n") != std::string::npos);
    require(prometheus.find("llamacpp:recurrent_snapshot_resident_depth 7\n") != std::string::npos);
    require(prometheus.find("llamacpp:recurrent_snapshot_required_depth 7\n") != std::string::npos);
    require(prometheus.find("llamacpp:recurrent_snapshot_pending_depth -1\n") != std::string::npos);

    return 0;
}
