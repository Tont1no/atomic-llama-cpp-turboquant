#include "arg.h"
#include "common.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"
#include "../src/llama-ext.h"
#include "log.h"
#include "qwen38-recurrent-parity-utils.h"
#include "sampling.h"

#include <algorithm>
#include <array>
#include <clocale>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace qwen38_recurrent_parity;

namespace {

constexpr const char * GUARD_NAME  = "LLAMACPP_QWEN38_PARITY_GUARD";
constexpr const char * GUARD_VALUE = "ONE_CONTEXT_NO_SERVER";
constexpr const char * TOKENS_NAME = "LLAMACPP_QWEN38_PARITY_REFERENCE_TOKENS";
constexpr uint32_t P1_SEED = 20260819;
constexpr std::array<llama_token, 5> P1_EOG_TOKENS = {
    248044, 248046, 248063, 248064, 248065,
};
constexpr uint32_t CUDA_ROUTE_ABI = 12;
constexpr size_t CUDA_ROUTE_COUNT = 31;
constexpr size_t CUDA_ROUTE_GDN = 8;
constexpr size_t CUDA_ROUTE_GDN_FUSED_CACHE = 9;
constexpr size_t CUDA_ROUTE_FATTN_VEC = 21;
constexpr size_t CUDA_ROUTE_FATTN_MMA_F16 = 22;
constexpr size_t CUDA_ROUTE_FATTN_TILE = 23;
constexpr size_t CUDA_ROUTE_SIGMOID = 24;
constexpr size_t CUDA_ROUTE_SIGMOID_MUL = 25;
constexpr size_t CUDA_ROUTE_SET_ROWS = 26;
constexpr size_t CUDA_ROUTE_ROPE_VIEW_SET_ROWS = 27;
constexpr size_t CUDA_ROUTE_FATTN_QWEN35_D256_Q8_GQA6_VEC = 28;
constexpr size_t CUDA_ROUTE_FATTN_QWEN35_D256_Q8_GQA6_HINT = 29;
constexpr size_t CUDA_ROUTE_FATTN_QWEN35_D256_Q8_GQA6_VEC_COLS2_PB1 = 30;

struct cuda_route_counts {
    uint32_t abi_version;
    uint32_t route_count;
    uint64_t count[CUDA_ROUTE_COUNT];
};

struct cuda_fattn_candidate_observation {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t evaluated;
    uint32_t src4_null;
    uint64_t fail_mask;
    int32_t device_cc;
    int32_t compiled_arch;
    int32_t op;
    int32_t hint;
    int32_t q_type;
    int32_t k_type;
    int32_t v_type;
    int32_t mask_type;
    int32_t dst_type;
    int32_t prec;
    uint32_t scale_bits;
    uint32_t max_bias_bits;
    uint32_t softcap_bits;
    uint32_t mask_contiguous;
    uint32_t dst_contiguous;
    int64_t ne[5][4];
    uint64_t nb[5][4];
};

using cuda_route_reset_fn = void (*)();
using cuda_route_snapshot_fn = bool (*)(cuda_route_counts *, size_t);
using cuda_fattn_candidate_snapshot_fn = bool (*)(cuda_fattn_candidate_observation *, size_t);

struct cuda_route_api {
    cuda_route_reset_fn reset = nullptr;
    cuda_route_snapshot_fn snapshot = nullptr;
    cuda_fattn_candidate_snapshot_fn fattn_candidate_snapshot = nullptr;
};

static cuda_route_api require_cuda_route_api() {
    for (size_t i = 0; i < ggml_backend_reg_count(); ++i) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        auto reset = (cuda_route_reset_fn) ggml_backend_reg_get_proc_address(
                reg, "ggml_backend_cuda_diagnostic_route_reset");
        auto snapshot = (cuda_route_snapshot_fn) ggml_backend_reg_get_proc_address(
                reg, "ggml_backend_cuda_diagnostic_route_snapshot_v2");
        auto fattn_candidate_snapshot = (cuda_fattn_candidate_snapshot_fn) ggml_backend_reg_get_proc_address(
                reg, "ggml_backend_cuda_diagnostic_fattn_candidate_snapshot_v1");
        if (reset && snapshot && fattn_candidate_snapshot) {
            return { reset, snapshot, fattn_candidate_snapshot };
        }
    }
    throw std::runtime_error("mandatory CUDA diagnostic route API is unavailable");
}

static cuda_fattn_candidate_observation require_cuda_fattn_candidate_snapshot(
        cuda_fattn_candidate_snapshot_fn snapshot) {
    cuda_fattn_candidate_observation result{};
    if (!snapshot || !snapshot(&result, sizeof(result)) ||
            result.abi_version != 1 || result.struct_size != sizeof(result) || !result.evaluated) {
        throw std::runtime_error("CUDA full-attention candidate predicate observation ABI is incompatible");
    }
    return result;
}

static cuda_route_counts require_cuda_route_snapshot(cuda_route_snapshot_fn snapshot) {
    cuda_route_counts result{};
    if (!snapshot || !snapshot(&result, sizeof(result)) ||
            result.abi_version != CUDA_ROUTE_ABI || result.route_count != CUDA_ROUTE_COUNT) {
        throw std::runtime_error("CUDA diagnostic route snapshot ABI is incompatible");
    }
    return result;
}

static bool same_cuda_routes(const cuda_route_counts & left, const cuda_route_counts & right) {
    return left.abi_version == right.abi_version && left.route_count == right.route_count &&
            std::equal(std::begin(left.count), std::end(left.count), std::begin(right.count));
}

static std::string trim_ascii(std::string value) {
    const size_t first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    const size_t last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

static bool ends_with(const std::string & value, const char * suffix) {
    const size_t suffix_size = std::strlen(suffix);
    return value.size() >= suffix_size &&
            value.compare(value.size() - suffix_size, suffix_size, suffix) == 0;
}

static std::string model_meta_by_index(const llama_model * model, int32_t index, bool key) {
    char ignored = '\0';
    const int32_t size = key ? llama_model_meta_key_by_index(model, index, &ignored, 0) :
            llama_model_meta_val_str_by_index(model, index, &ignored, 0);
    if (size < 0) {
        throw std::runtime_error("failed to enumerate loaded-model metadata");
    }
    std::vector<char> buffer((size_t) size + 1);
    const int32_t written = key ? llama_model_meta_key_by_index(model, index, buffer.data(), buffer.size()) :
            llama_model_meta_val_str_by_index(model, index, buffer.data(), buffer.size());
    if (written != size) {
        throw std::runtime_error("loaded-model metadata changed during enumeration");
    }
    return std::string(buffer.data(), (size_t) size);
}

static bool parse_recurrent_flag(const std::string & text, uint8_t & flag) {
    const std::string value = trim_ascii(text);
    if (value == "true" || value == "1") {
        flag = 1;
        return true;
    }
    if (value == "false" || value == "0") {
        flag = 0;
        return true;
    }
    return false;
}

static std::vector<uint8_t> parse_recurrent_metadata(const std::string & text, size_t layers) {
    const std::string value = trim_ascii(text);
    std::vector<uint8_t> result;
    if (value.size() >= 2 && value.front() == '[' && value.back() == ']') {
        std::stringstream stream(value.substr(1, value.size() - 2));
        std::string item;
        while (std::getline(stream, item, ',')) {
            uint8_t flag = 0;
            if (!parse_recurrent_flag(item, flag)) {
                throw std::runtime_error("loaded recurrent-layer metadata contains a non-boolean value");
            }
            result.push_back(flag);
        }
        // Qwen3.5 metadata may include appended MTP blocks after the main
        // decoder stack. The boundary diagnostic traces only the public
        // llama_model_n_layer() main-stack prefix.
        if (result.size() < layers || result.size() > 512) {
            throw std::runtime_error("loaded recurrent-layer metadata does not cover the main stack exactly once");
        }
        result.resize(layers);
        return result;
    }

    uint8_t repeated = 0;
    if (!parse_recurrent_flag(value, repeated)) {
        throw std::runtime_error("loaded recurrent-layer metadata is neither a boolean nor a boolean array");
    }
    return std::vector<uint8_t>(layers, repeated);
}

static uint32_t parse_positive_u32(const std::string & text) {
    const std::string value = trim_ascii(text);
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos) {
        throw std::runtime_error("loaded full-attention interval is not an unsigned integer");
    }
    size_t consumed = 0;
    const unsigned long parsed = std::stoul(value, &consumed, 10);
    if (consumed != value.size() || parsed == 0 || parsed > UINT32_MAX) {
        throw std::runtime_error("loaded full-attention interval is out of range");
    }
    return (uint32_t) parsed;
}

static std::vector<uint8_t> loaded_recurrent_layer_map(
        const llama_model * model,
        size_t layers,
        std::string & source) {
    std::string recurrent_value;
    std::string interval_value;
    std::string interval_source;
    int recurrent_matches = 0;
    int interval_matches = 0;
    const int32_t metadata_count = llama_model_meta_count(model);
    if (metadata_count < 0) {
        throw std::runtime_error("failed to query loaded-model metadata count");
    }
    for (int32_t i = 0; i < metadata_count; ++i) {
        const std::string key = model_meta_by_index(model, i, true);
        if (ends_with(key, ".attention.recurrent_layers")) {
            recurrent_value = model_meta_by_index(model, i, false);
            source = key;
            ++recurrent_matches;
        } else if (ends_with(key, ".full_attention_interval")) {
            interval_value = model_meta_by_index(model, i, false);
            interval_source = key;
            ++interval_matches;
        }
    }
    if (recurrent_matches > 1 || interval_matches > 1) {
        throw std::runtime_error("loaded model contains ambiguous recurrent-layer metadata");
    }
    if (recurrent_matches == 1) {
        return parse_recurrent_metadata(recurrent_value, layers);
    }

    uint32_t interval = 4;
    if (interval_matches == 1) {
        interval = parse_positive_u32(interval_value);
        source = interval_source;
    } else {
        source = "qwen35_runtime_default_interval_4";
    }
    std::vector<uint8_t> result(layers);
    for (size_t il = 0; il < layers; ++il) {
        result[il] = ((il + 1) % interval != 0) ? 1 : 0;
    }
    return result;
}

struct state_digest {
    uint64_t hash = 0;
    size_t bytes = 0;
};

struct prediction_observation {
    int32_t prediction_index = -1;
    llama_token input_token = LLAMA_TOKEN_NULL;
    llama_token expected_token = LLAMA_TOKEN_NULL;
    top2_result top2;
};

struct width_result {
    uint32_t width = 0;
    std::vector<prediction_observation> predictions;
    state_digest prefix_state;
    state_digest final_state;
    state_digest full_final_state;
    int32_t final_consumed_token_index = -1;
};

struct oracle_observation {
    top2_result next;
    state_digest state;
};

struct rollback_observation {
    std::vector<prediction_observation> batch_rows;
    state_digest batch_final;
    state_digest selected;
    top2_result continued;
    state_digest continued_state;
};

using boundary_key = std::pair<int32_t, std::string>;

struct boundary_capture_snapshot {
    std::map<boundary_key, std::vector<uint64_t>> rows;
    std::map<boundary_key, std::vector<std::vector<uint8_t>>> row_data;
    std::map<boundary_key, size_t> row_bytes;
    std::map<int32_t, std::vector<uint64_t>> recurrent_state_inputs;
    std::map<int32_t, std::vector<std::vector<uint8_t>>> recurrent_state_data;
    std::map<int32_t, size_t> recurrent_state_bytes;
    std::map<int32_t, std::string> layer_types;
    std::map<boundary_key, std::array<int64_t, 4>> internal_shapes;
    std::map<std::string, std::vector<uint64_t>> internal_cache_inputs;
    std::map<std::string, std::vector<std::vector<uint8_t>>> internal_cache_data;
    std::map<std::string, size_t> internal_cache_bytes;
    std::map<std::string, std::array<int64_t, 4>> internal_cache_shapes;
};

struct boundary_capture {
    enum class stage { coarse, refine, recurrent_internal, full_attention_internal };

    bool active = false;
    bool batched = false;
    int32_t scalar_row = -1;
    int32_t target_layer = -1;
    bool target_layer_recurrent = false;
    size_t expected_features = 0;
    stage capture_stage = stage::coarse;
    std::string failure;
    boundary_capture_snapshot snapshot;

    static bool parse_layer_name(const char * name, const char * prefix, int32_t & layer) {
        const size_t prefix_size = std::strlen(prefix);
        if (std::strncmp(name, prefix, prefix_size) != 0) {
            return false;
        }
        char * end = nullptr;
        const long value = std::strtol(name + prefix_size, &end, 10);
        if (end == name + prefix_size || *end != '\0' || value < 0 || value > INT32_MAX) {
            return false;
        }
        layer = (int32_t) value;
        return true;
    }

    static bool classify(
            ggml_tensor * tensor,
            int32_t & layer,
            std::string & boundary,
            bool & recurrent_state,
            ggml_tensor *& captured,
            size_t expected_features,
            int32_t target_layer,
            bool target_layer_recurrent) {
        recurrent_state = false;
        captured = tensor;
        if (parse_layer_name(tensor->name, "linear_attn_out-", layer)) {
            boundary = "attention_output";
            return true;
        }
        if (parse_layer_name(tensor->name, "attn_output-", layer)) {
            // Recurrent Qwen3.5 blocks also expose a lower-level GDN tensor with
            // this generic name. Singleton token/sequence axes made its scalar
            // [S_v,H_v,1,1] shape look 2-D in the first run. The comparable
            // post-Wo branch always has the model embedding width in ne[0].
            if ((layer == target_layer && target_layer_recurrent) || tensor->ne[0] <= 0 ||
                    (uint64_t) tensor->ne[0] != expected_features) {
                return false;
            }
            boundary = "attention_output";
            return true;
        }
        if (parse_layer_name(tensor->name, "attn_residual-", layer)) {
            boundary = "attention_residual";
            return true;
        }
        if (parse_layer_name(tensor->name, "ffn_out-", layer)) {
            boundary = "ffn_output";
            return true;
        }
        if (parse_layer_name(tensor->name, "l_out-", layer)) {
            boundary = "block_output";
            return true;
        }
        if (parse_layer_name(tensor->name, "state_predelta-", layer)) {
            boundary = "recurrent_state_input";
            recurrent_state = true;
            return true;
        }
        return false;
    }

    static std::vector<uint8_t> read_tensor(ggml_tensor * tensor) {
        if (!tensor || tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensor) ||
                !tensor->buffer || !tensor->data) {
            throw std::runtime_error("layer-boundary capture requires allocated contiguous F32 tensors");
        }
        std::vector<uint8_t> bytes(ggml_nbytes(tensor));
        if (ggml_backend_buffer_is_host(tensor->buffer)) {
            std::memcpy(bytes.data(), tensor->data, bytes.size());
        } else {
            ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
        }
        const float * values = (const float *) bytes.data();
        for (size_t i = 0; i < bytes.size() / sizeof(float); ++i) {
            if (!std::isfinite(values[i])) {
                throw std::runtime_error("layer-boundary capture found a non-finite value");
            }
        }
        return bytes;
    }

    static bool internal_spec(
            ggml_tensor * tensor,
            int32_t target_layer,
            std::string & boundary,
            uint32_t & feature_rank,
            std::array<int64_t, 2> & expected_feature_shape,
            bool & cache_input) {
        struct spec {
            const char * prefix;
            const char * boundary;
            uint32_t feature_rank;
            std::array<int64_t, 2> feature_shape;
            bool cache_input;
        };
        // Do not select a_softplus or conv_output_raw here: ending scheduler
        // graph views at those intermediate nodes would disable the reviewed
        // CUDA SOFTPLUS->MUL and SSM_CONV->SILU fusions. Their fused outputs
        // gate and conv_output_silu are the route-preserving boundaries.
        static constexpr std::array<spec, 16> specs = {{
            { "attn_norm-",                "attn_norm",                1, { 5120,  1 }, false },
            { "linear_attn_qkv_mixed-",    "linear_attn_qkv_mixed",    1, { 10240, 1 }, false },
            { "z-",                        "z",                        1, { 6144,  1 }, false },
            { "beta-",                     "beta",                     2, { 1,    48 }, false },
            { "beta_sigmoid-",             "beta_sigmoid",             2, { 1,    48 }, false },
            { "alpha-",                    "alpha",                    1, { 48,    1 }, false },
            { "gate-",                     "gate",                     1, { 48,    1 }, false },
            { "conv_states-",              "conv_states",              0, { 30720, 1 }, true  },
            { "state_predelta-",           "state_predelta",           0, { 128, 128 }, true  },
            { "conv_output_silu-",          "conv_output_silu",          1, { 10240, 1 }, false },
            { "q_conv_predelta-",           "q_conv_predelta",           2, { 128,  16 }, false },
            { "k_conv_predelta-",           "k_conv_predelta",           2, { 128,  16 }, false },
            { "v_conv_predelta-",           "v_conv_predelta",           2, { 128,  48 }, false },
            { "attn_output-",               "gdn_core_output",           2, { 128,  48 }, false },
            { "final_output-",              "final_output",              1, { 6144,  1 }, false },
            { "linear_attn_out-",           "linear_attn_out",           1, { 5120,  1 }, false },
        }};
        for (const auto & item : specs) {
            int32_t layer = -1;
            if (parse_layer_name(tensor->name, item.prefix, layer) && layer == target_layer) {
                boundary = item.boundary;
                feature_rank = item.feature_rank;
                expected_feature_shape = item.feature_shape;
                cache_input = item.cache_input;
                return true;
            }
        }
        return false;
    }

    static bool full_attention_internal_spec(
            ggml_tensor * tensor,
            int32_t target_layer,
            std::string & boundary,
            uint32_t & feature_rank,
            std::array<int64_t, 2> & expected_feature_shape,
            bool & split_qg_projection) {
        struct spec {
            const char * prefix;
            const char * boundary;
            uint32_t feature_rank;
            std::array<int64_t, 2> feature_shape;
            bool split_qg_projection;
        };
        // Qwen3.8 uses IMROPE. The current CUDA RMS/ROPE/SET_ROWS fusion guards
        // accept NORMAL/NEOX only, so Q/K norm and post-RoPE K are distinct
        // production nodes for this pinned model. gate_sigmoid is deliberately
        // omitted because selecting it would split SIGMOID+MUL. Route equality
        // against callback-free arms below fails closed if future CUDA fusion
        // eligibility changes. Pinned metadata: D=256, Hq=24, Hkv=4.
        static constexpr std::array<spec, 12> specs = {{
            { "attn_norm-",       "attn_norm",              1, { 5120, 1 }, false },
            // Qcur_full is contiguous [Q256,G256] per head. The callback
            // derives logical raw-Q and raw-gate rows offline without selecting
            // the noncontiguous Qcur_reshaped view or gate_reshaped node.
            { "Qcur_full-",       "qg_projection",          1, { 12288, 1 }, true  },
            { "Qcur_normed-",     "q_post_norm",            2, { 256, 24 }, false },
            { "Kcur_projection-", "k_projection",           1, { 1024, 1 }, false },
            { "Kcur_normed-",     "k_post_norm",            2, { 256, 4 }, false },
            { "Vcur_projection-", "v_projection",           1, { 1024, 1 }, false },
            { "Qcur-",            "q_post_norm_rope",       2, { 256, 24 }, false },
            { "Kcur-",            "k_post_norm_rope",       2, { 256, 4 }, false },
            { "Vcur-",            "v_reshaped",             2, { 256, 4 }, false },
            { "attn_pregate-",    "kv_fa_output_pregate",   1, { 6144, 1 }, false },
            { "attn_gated-",      "attn_gated",             1, { 6144, 1 }, false },
            { "attn_output-",     "output_projection",      1, { 5120, 1 }, false },
        }};
        for (const auto & item : specs) {
            int32_t layer = -1;
            if (parse_layer_name(tensor->name, item.prefix, layer) && layer == target_layer) {
                boundary = item.boundary;
                feature_rank = item.feature_rank;
                expected_feature_shape = item.feature_shape;
                split_qg_projection = item.split_qg_projection;
                return true;
            }
        }
        return false;
    }

    static std::vector<std::vector<uint8_t>> read_internal_token_rows(
            ggml_tensor * tensor,
            uint32_t feature_rank,
            const std::array<int64_t, 2> & expected_feature_shape,
            size_t expected_rows,
            size_t & row_bytes) {
        if (!tensor || tensor->type != GGML_TYPE_F32 || !tensor->buffer || !tensor->data ||
                feature_rank < 1 || feature_rank > 2) {
            throw std::runtime_error("internal recurrent boundary requires an allocated F32 tensor and a reviewed feature rank");
        }
        if (tensor->nb[0] != sizeof(float)) {
            throw std::runtime_error("internal recurrent boundary feature axis is not dense F32");
        }
        const std::array<int64_t, 4> shape = {
            tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
        };
        boundary_row_layout layout;
        if (!internal_f32_token_layout(
                shape, feature_rank, expected_feature_shape, expected_rows, layout)) {
            throw std::runtime_error("internal recurrent boundary shape differs from the pinned token-slice contract");
        }
        for (uint32_t axis = 1; axis < feature_rank; ++axis) {
            if (axis > 0 && tensor->nb[axis] != tensor->nb[axis - 1] * (size_t) tensor->ne[axis - 1]) {
                throw std::runtime_error("internal recurrent boundary feature slice is not contiguous");
            }
        }
        const uint32_t token_axis = feature_rank;
        row_bytes = layout.row_bytes;
        if (tensor->nb[token_axis] < row_bytes) {
            throw std::runtime_error("internal recurrent boundary token stride overlaps feature rows");
        }
        std::vector<std::vector<uint8_t>> rows(expected_rows, std::vector<uint8_t>(row_bytes));
        const size_t tensor_bytes = ggml_nbytes(tensor);
        for (size_t row = 0; row < expected_rows; ++row) {
            const size_t offset = row * tensor->nb[token_axis];
            if (offset > tensor_bytes || row_bytes > tensor_bytes - offset) {
                throw std::runtime_error("internal recurrent boundary row exceeds the tensor allocation");
            }
            if (ggml_backend_buffer_is_host(tensor->buffer)) {
                std::memcpy(rows[row].data(), (const uint8_t *) tensor->data + offset, row_bytes);
            } else {
                ggml_backend_tensor_get(tensor, rows[row].data(), offset, row_bytes);
            }
            const float * values = (const float *) rows[row].data();
            for (size_t i = 0; i < row_bytes / sizeof(float); ++i) {
                if (!std::isfinite(values[i])) {
                    throw std::runtime_error("internal recurrent boundary contains a non-finite value");
                }
            }
        }
        return rows;
    }

    bool eval_internal(ggml_tensor * tensor, bool ask) {
        std::string boundary;
        uint32_t feature_rank = 0;
        std::array<int64_t, 2> expected_feature_shape = { 0, 0 };
        bool cache_input = false;
        if (!target_layer_recurrent || !internal_spec(
                tensor, target_layer, boundary, feature_rank, expected_feature_shape, cache_input)) {
            return false;
        }
        if (ask) {
            return true;
        }
        try {
            const std::array<int64_t, 4> shape = {
                tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
            };
            if (cache_input) {
                const std::array<int64_t, 4> expected_cache_shape = boundary == "conv_states" ?
                        std::array<int64_t, 4>{ 30720, 1, 1, 1 } :
                        std::array<int64_t, 4>{ 128, 128, 48, 1 };
                if (shape != expected_cache_shape) {
                    throw std::runtime_error("cache-input shape differs from the pinned Qwen3.8 contract");
                }
                const auto bytes = read_tensor(tensor);
                auto & hashes = snapshot.internal_cache_inputs[boundary];
                auto & raw = snapshot.internal_cache_data[boundary];
                if (batched) {
                    if (!hashes.empty() || !raw.empty()) {
                        throw std::runtime_error("duplicate batched internal recurrent cache boundary");
                    }
                } else if ((int32_t) hashes.size() != scalar_row || (int32_t) raw.size() != scalar_row) {
                    throw std::runtime_error("scalar internal recurrent cache boundary order is not deterministic");
                }
                hashes.push_back(fnv1a64(bytes.data(), bytes.size()));
                raw.push_back(bytes);
                snapshot.internal_cache_bytes[boundary] = bytes.size();
                snapshot.internal_cache_shapes[boundary] = shape;
                return true;
            }

            size_t row_bytes = 0;
            const auto rows = read_internal_token_rows(
                    tensor, feature_rank, expected_feature_shape, batched ? 8 : 1, row_bytes);
            const boundary_key key = { target_layer, boundary };
            auto & hashes = snapshot.rows[key];
            auto & raw_rows = snapshot.row_data[key];
            if (batched) {
                if (!hashes.empty() || !raw_rows.empty()) {
                    throw std::runtime_error("duplicate batched internal recurrent activation boundary");
                }
            } else if ((int32_t) hashes.size() != scalar_row || (int32_t) raw_rows.size() != scalar_row) {
                throw std::runtime_error("scalar internal recurrent activation boundary order is not deterministic");
            }
            for (const auto & row : rows) {
                hashes.push_back(fnv1a64(row.data(), row.size()));
                raw_rows.push_back(row);
            }
            snapshot.row_bytes[key] = row_bytes;
            snapshot.internal_shapes[key] = shape;
            return true;
        } catch (const std::exception & error) {
            failure = std::string("internal recurrent ") + boundary + ": " + error.what();
            return false;
        }
    }

    bool eval_full_attention_internal(ggml_tensor * tensor, bool ask) {
        std::string boundary;
        uint32_t feature_rank = 0;
        std::array<int64_t, 2> expected_feature_shape = { 0, 0 };
        bool split_qg_projection = false;
        if (target_layer_recurrent || !full_attention_internal_spec(
                tensor, target_layer, boundary, feature_rank, expected_feature_shape,
                split_qg_projection)) {
            return false;
        }
        if (ask) {
            return true;
        }
        try {
            const std::array<int64_t, 4> shape = {
                tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
            };
            size_t row_bytes = 0;
            const auto rows = read_internal_token_rows(
                    tensor, feature_rank, expected_feature_shape, batched ? 8 : 1, row_bytes);

            auto store_rows = [&](const std::string & name,
                                  const std::vector<std::vector<uint8_t>> & values,
                                  size_t bytes_per_row,
                                  const std::array<int64_t, 4> & value_shape) {
                const boundary_key key = { target_layer, name };
                auto & hashes = snapshot.rows[key];
                auto & raw_rows = snapshot.row_data[key];
                if (batched) {
                    if (!hashes.empty() || !raw_rows.empty()) {
                        throw std::runtime_error("duplicate batched full-attention internal boundary");
                    }
                } else if ((int32_t) hashes.size() != scalar_row ||
                        (int32_t) raw_rows.size() != scalar_row) {
                    throw std::runtime_error("scalar full-attention internal boundary order is not deterministic");
                }
                for (const auto & value : values) {
                    if (value.size() != bytes_per_row) {
                        throw std::runtime_error("derived full-attention row has an invalid byte count");
                    }
                    hashes.push_back(fnv1a64(value.data(), value.size()));
                    raw_rows.push_back(value);
                }
                snapshot.row_bytes[key] = bytes_per_row;
                snapshot.internal_shapes[key] = value_shape;
            };

            store_rows(boundary, rows, row_bytes, shape);
            if (split_qg_projection) {
                constexpr size_t head_dim = 256;
                constexpr size_t n_head = 24;
                constexpr size_t qg_head = 2 * head_dim;
                constexpr size_t logical_features = head_dim * n_head;
                std::vector<std::vector<uint8_t>> q_rows;
                std::vector<std::vector<uint8_t>> gate_rows;
                q_rows.reserve(rows.size());
                gate_rows.reserve(rows.size());
                for (const auto & value : rows) {
                    if (value.size() != qg_head * n_head * sizeof(float)) {
                        throw std::runtime_error("joint Q/G projection row differs from the pinned interleave");
                    }
                    std::vector<uint8_t> q(logical_features * sizeof(float));
                    std::vector<uint8_t> gate(logical_features * sizeof(float));
                    for (size_t head = 0; head < n_head; ++head) {
                        const uint8_t * src = value.data() + head * qg_head * sizeof(float);
                        std::memcpy(q.data() + head * head_dim * sizeof(float),
                                src, head_dim * sizeof(float));
                        std::memcpy(gate.data() + head * head_dim * sizeof(float),
                                src + head_dim * sizeof(float), head_dim * sizeof(float));
                    }
                    q_rows.push_back(std::move(q));
                    gate_rows.push_back(std::move(gate));
                }
                store_rows("q_pre_norm", q_rows, logical_features * sizeof(float),
                        { (int64_t) head_dim, (int64_t) n_head, batched ? 8 : 1, 1 });
                store_rows("gate_pre_sigmoid", gate_rows, logical_features * sizeof(float),
                        { (int64_t) logical_features, batched ? 8 : 1, 1, 1 });
            }
            return true;
        } catch (const std::exception & error) {
            failure = std::string("internal full-attention ") + boundary + ": " + error.what();
            return false;
        }
    }

    bool eval(ggml_tensor * tensor, bool ask) {
        if (!active || !failure.empty()) {
            return false;
        }
        if (capture_stage == stage::recurrent_internal) {
            return eval_internal(tensor, ask);
        }
        if (capture_stage == stage::full_attention_internal) {
            return eval_full_attention_internal(tensor, ask);
        }
        int32_t layer = -1;
        std::string boundary;
        bool recurrent_state = false;
        ggml_tensor * captured = nullptr;
        if (!classify(tensor, layer, boundary, recurrent_state, captured, expected_features,
                target_layer, target_layer_recurrent)) {
            return false;
        }
        if ((capture_stage == stage::coarse && boundary != "block_output") ||
                (capture_stage == stage::refine && layer != target_layer) ||
                (capture_stage == stage::refine && boundary == "block_input")) {
            return false;
        }
        if (ask) {
            return true;
        }
        try {
            const auto bytes = read_tensor(captured);
            if (recurrent_state) {
                const uint64_t hash = fnv1a64(bytes.data(), bytes.size());
                auto & hashes = snapshot.recurrent_state_inputs[layer];
                auto & raw = snapshot.recurrent_state_data[layer];
                if (batched) {
                    if (!hashes.empty() || !raw.empty()) {
                        throw std::runtime_error("duplicate batched recurrent-state boundary");
                    }
                } else if ((int32_t) hashes.size() != scalar_row || (int32_t) raw.size() != scalar_row) {
                    throw std::runtime_error("scalar recurrent-state boundary order is not deterministic");
                }
                hashes.push_back(hash);
                raw.push_back(bytes);
                snapshot.recurrent_state_bytes[layer] = bytes.size();
                snapshot.layer_types[layer] = "recurrent";
                return true;
            }

            const std::array<int64_t, 4> shape = {
                captured->ne[0], captured->ne[1], captured->ne[2], captured->ne[3],
            };
            boundary_row_layout layout;
            const size_t expected_rows = batched ? 8 : 1;
            if (!canonical_f32_boundary_layout(shape, expected_features, expected_rows, layout)) {
                std::ostringstream message;
                message << "layer-boundary " << boundary << " (" << tensor->name
                        << ") has non-canonical shape [" << shape[0] << ',' << shape[1]
                        << ',' << shape[2] << ',' << shape[3] << "]; expected feature axis "
                        << expected_features << ", " << expected_rows
                        << " token rows on ne[1], and singleton higher axes";
                throw std::runtime_error(message.str());
            }
            const size_t row_bytes = layout.row_bytes;
            if (bytes.size() != layout.total_bytes) {
                throw std::runtime_error("layer-boundary tensor byte layout is not row-contiguous");
            }
            const boundary_key key = { layer, boundary };
            auto & hashes = snapshot.rows[key];
            auto & raw_rows = snapshot.row_data[key];
            if (batched) {
                if (!hashes.empty() || !raw_rows.empty()) {
                    throw std::runtime_error("duplicate batched layer boundary");
                }
                for (size_t row = 0; row < layout.rows; ++row) {
                    hashes.push_back(fnv1a64(bytes.data() + row * row_bytes, row_bytes));
                    raw_rows.emplace_back(
                            bytes.begin() + row * row_bytes,
                            bytes.begin() + (row + 1) * row_bytes);
                }
            } else {
                if ((int32_t) hashes.size() != scalar_row || (int32_t) raw_rows.size() != scalar_row) {
                    throw std::runtime_error("scalar layer-boundary order is not deterministic");
                }
                hashes.push_back(fnv1a64(bytes.data(), row_bytes));
                raw_rows.push_back(bytes);
            }
            snapshot.row_bytes[key] = row_bytes;
            if (boundary == "attention_output") {
                const bool recurrent = std::strncmp(tensor->name, "linear_attn_out-", 16) == 0;
                const std::string type = recurrent ? "recurrent" : "full_attention";
                const auto existing = snapshot.layer_types.find(layer);
                if (existing != snapshot.layer_types.end() && existing->second != type) {
                    throw std::runtime_error("layer emitted conflicting attention types");
                }
                snapshot.layer_types[layer] = type;
            }
            return true;
        } catch (const std::exception & error) {
            failure = error.what();
            return false;
        }
    }

    static bool callback(ggml_tensor * tensor, bool ask, void * user_data) {
        return ((boundary_capture *) user_data)->eval(tensor, ask);
    }

    void begin(
            bool batch,
            stage selected_stage,
            int32_t selected_layer = -1,
            bool selected_layer_recurrent = false) {
        active = true;
        batched = batch;
        scalar_row = batch ? -1 : 0;
        capture_stage = selected_stage;
        target_layer = selected_layer;
        target_layer_recurrent = selected_layer_recurrent;
        failure.clear();
        snapshot = {};
    }

    void require_success() const {
        if (!failure.empty()) {
            throw std::runtime_error("layer-boundary callback failed: " + failure);
        }
    }
};

struct boundary_arm_result {
    boundary_capture_snapshot capture;
    state_digest prefix_state;
    state_digest final_state;
    state_digest full_final_state;
    std::vector<state_digest> scalar_states;
    std::vector<prediction_observation> predictions;
    bool repeat_stable = false;
};

struct diagnostic_context {
    common_threadpools threadpools;
    llama_context_ptr handle;

    llama_context * get() const {
        return handle.get();
    }
};

using diagnostic_context_ptr = std::unique_ptr<diagnostic_context>;

static const char * get_env(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr ? value : "";
}

static int32_t get_env_i32(const char * name, int32_t fallback) {
    const char * value = get_env(name);
    if (*value == '\0') {
        return fallback;
    }
    size_t parsed = 0;
    const long long result = std::stoll(value, &parsed, 10);
    if (parsed != std::strlen(value) || result < 0 || result > std::numeric_limits<int32_t>::max()) {
        throw std::invalid_argument(std::string("invalid integer environment variable ") + name);
    }
    return (int32_t) result;
}

static std::string hex64(uint64_t value) {
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16) << value;
    return out.str();
}

static void print_fattn_candidate_predicate(
        uint32_t width, const cuda_fattn_candidate_observation & observation) {
    std::ostringstream out;
    out << "{\"type\":\"fattn_candidate_predicate\",\"width\":" << width
        << ",\"fail_mask\":\"" << hex64(observation.fail_mask) << "\""
        << ",\"device_cc\":" << observation.device_cc
        << ",\"compiled_arch\":" << observation.compiled_arch
        << ",\"src4_null\":" << (observation.src4_null ? "true" : "false")
        << ",\"hint\":" << observation.hint << ",\"op\":" << observation.op
        << ",\"q_type\":" << observation.q_type << ",\"k_type\":" << observation.k_type
        << ",\"v_type\":" << observation.v_type << ",\"mask_type\":" << observation.mask_type
        << ",\"dst_type\":" << observation.dst_type << ",\"prec\":" << observation.prec
        << ",\"scale_bits\":" << observation.scale_bits
        << ",\"max_bias_bits\":" << observation.max_bias_bits
        << ",\"softcap_bits\":" << observation.softcap_bits
        << ",\"mask_contiguous\":" << (observation.mask_contiguous ? "true" : "false")
        << ",\"dst_contiguous\":" << (observation.dst_contiguous ? "true" : "false");
    static const char * names[5] = { "q", "k", "v", "mask", "dst" };
    for (size_t tensor = 0; tensor < 5; ++tensor) {
        out << ",\"" << names[tensor] << "_ne\":[";
        for (size_t dim = 0; dim < 4; ++dim) {
            if (dim) out << ',';
            out << observation.ne[tensor][dim];
        }
        out << "],\"" << names[tensor] << "_nb\":[";
        for (size_t dim = 0; dim < 4; ++dim) {
            if (dim) out << ',';
            out << observation.nb[tensor][dim];
        }
        out << ']';
    }
    out << ",\"evaluated\":true,\"content_free\":true}";
    std::puts(out.str().c_str());
    std::fflush(stdout);
}

static std::string get_required_sha256(const char * name) {
    const std::string value = get_env(name);
    if (value.size() != 64 || !std::all_of(value.begin(), value.end(), [](char c) {
            return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
        })) {
        throw std::invalid_argument(std::string("missing or invalid SHA-256 environment variable ") + name);
    }
    return value;
}

static bool same_state(const state_digest & left, const state_digest & right) {
    return left.hash == right.hash && left.bytes == right.bytes;
}

static bool same_top2(const top2_result & left, const top2_result & right) {
    return left.top1 == right.top1 && left.top2 == right.top2 &&
            float_bits(left.logit1) == float_bits(right.logit1) &&
            float_bits(left.logit2) == float_bits(right.logit2);
}

static bool same_width_result(const width_result & left, const width_result & right) {
    if (left.width != right.width || left.final_consumed_token_index != right.final_consumed_token_index ||
            !same_state(left.prefix_state, right.prefix_state) || !same_state(left.final_state, right.final_state) ||
            !same_state(left.full_final_state, right.full_final_state) ||
            left.predictions.size() != right.predictions.size()) {
        return false;
    }
    for (size_t i = 0; i < left.predictions.size(); ++i) {
        const auto & a = left.predictions[i];
        const auto & b = right.predictions[i];
        if (a.prediction_index != b.prediction_index || a.input_token != b.input_token ||
                a.expected_token != b.expected_token || !same_top2(a.top2, b.top2)) {
            return false;
        }
    }
    return true;
}

static void print_top2_json(
        const char * type,
        uint32_t width,
        const prediction_observation & observation) {
    std::printf(
            "{\"type\":\"%s\",\"width\":%u,\"prediction_index\":%d,"
            "\"input_token\":%d,\"expected_token\":%d,\"top1\":%d,\"top2\":%d,"
            "\"logit1\":%.9g,\"logit2\":%.9g,\"margin\":%.9g,\"expected_is_top1\":%s}\n",
            type, width, observation.prediction_index,
            observation.input_token, observation.expected_token,
            observation.top2.top1, observation.top2.top2,
            observation.top2.logit1, observation.top2.logit2, observation.top2.margin(),
            observation.expected_token == observation.top2.top1 ? "true" : "false");
}

static common_params_sampling make_p1_sampling_params(
        const llama_model * model,
        const common_params & params,
        int32_t & n_eog_biases) {
    common_params_sampling sampling = params.sampling;
    common_params_sampling_init_from_model(model, sampling);

    if (sampling.ignore_eos || !sampling.logit_bias.empty() || !sampling.logit_bias_eog.empty()) {
        throw std::runtime_error("exact P1 sampler requires empty base logit biases and ignore_eos=false");
    }

    // Exact deterministic_workload request from the pinned p1-final-v5
    // qualification. The server applies these request fields after the model
    // defaults above, then creates a CPU sampler (backend_sampling=false).
    sampling.temp             = 0.0f;
    sampling.seed             = P1_SEED;
    sampling.ignore_eos       = true;
    sampling.backend_sampling = false;

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> eog_tokens;
    for (llama_token token = 0; token < llama_vocab_n_tokens(vocab); ++token) {
        if (llama_vocab_is_eog(vocab, token)) {
            eog_tokens.push_back(token);
            sampling.logit_bias_eog.push_back({ token, -INFINITY });
            sampling.logit_bias.push_back({ token, -INFINITY });
        }
    }
    if (eog_tokens.size() != P1_EOG_TOKENS.size() ||
            !std::equal(eog_tokens.begin(), eog_tokens.end(), P1_EOG_TOKENS.begin())) {
        throw std::runtime_error("target vocabulary EOG token IDs do not match the pinned P1 request contract");
    }
    n_eog_biases = (int32_t) eog_tokens.size();
    return sampling;
}

static diagnostic_context_ptr make_context(
        llama_model * model,
        const common_params & params,
        uint32_t width,
        boundary_capture * capture = nullptr) {
    llama_context_params cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = 1;
    cparams.n_rs_seq = width - 1;
    cparams.n_outputs_max = width;
    cparams.n_outputs_max_per_seq = width;
    cparams.rs_seq_dynamic = false;
    if (capture) {
        cparams.cb_eval = boundary_capture::callback;
        cparams.cb_eval_user_data = capture;
    }
    auto result = std::make_unique<diagnostic_context>();
    result->handle.reset(llama_init_from_model(model, cparams));
    if (!result->handle) {
        return nullptr;
    }

    // llama-server attaches the common threadpools before its startup warmup.
    result->threadpools.init(result->handle.get(), params);
    if (!params.warmup) {
        return result;
    }

    llama_context * ctx = result->handle.get();

    // common_init_from_params(), used by llama-server, performs one decoder
    // warmup before the first request and then clears/synchronizes the memory.
    // Reproduce that lifecycle for every independently created width context.
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_tokens warmup;
    const llama_token bos = llama_vocab_bos(vocab);
    const llama_token eos = llama_vocab_eos(vocab);
    if (bos != LLAMA_TOKEN_NULL) {
        warmup.push_back(bos);
    }
    if (eos != LLAMA_TOKEN_NULL) {
        warmup.push_back(eos);
    }
    if (warmup.empty()) {
        warmup.push_back(0);
    }
    if (llama_model_has_decoder(model)) {
        const int32_t rc = llama_decode(
                ctx, llama_batch_get_one(warmup.data(), std::min(warmup.size(), (size_t) params.n_batch)));
        if (rc != 0) {
            throw std::runtime_error("server-equivalent context warmup failed with rc=" + std::to_string(rc));
        }
    }
    llama_memory_clear(llama_get_memory(ctx), true);
    llama_synchronize(ctx);
    llama_perf_context_reset(ctx);
    return result;
}

static void decode_rows(
        llama_context * ctx,
        const llama_tokens & tokens,
        llama_pos pos_first,
        uint32_t rs_depth,
        bool all_logits) {
    if (tokens.empty()) {
        return;
    }
    llama_batch batch = llama_batch_init((int32_t) tokens.size(), 0, 1);
    std::vector<uint32_t> depths(tokens.size(), rs_depth);
    batch.rs_depth = depths.data();
    for (size_t i = 0; i < tokens.size(); ++i) {
        common_batch_add(batch, tokens[i], pos_first + (llama_pos) i, { 0 },
                batch_row_requires_logits(i, tokens.size(), all_logits));
    }
    const int32_t rc = llama_decode(ctx, batch);
    batch.rs_depth = nullptr;
    llama_batch_free(batch);
    if (rc != 0) {
        throw std::runtime_error("llama_decode failed with rc=" + std::to_string(rc));
    }
}

static state_digest sequence_digest(llama_context * ctx, llama_state_seq_flags flags) {
    const size_t size = llama_state_seq_get_size_ext(ctx, 0, flags);
    if (size == 0) {
        throw std::runtime_error("sequence state is empty");
    }
    std::vector<uint8_t> state(size);
    const size_t copied = llama_state_seq_get_data_ext(ctx, state.data(), state.size(), 0, flags);
    if (copied != state.size()) {
        throw std::runtime_error("sequence state copy was truncated");
    }
    return { fnv1a64(state.data(), state.size()), state.size() };
}

static std::vector<uint8_t> sequence_state(llama_context * ctx, llama_state_seq_flags flags) {
    std::vector<uint8_t> state(llama_state_seq_get_size_ext(ctx, 0, flags));
    if (state.empty()) {
        throw std::runtime_error("sequence state is empty");
    }
    const size_t copied = llama_state_seq_get_data_ext(ctx, state.data(), state.size(), 0, flags);
    if (copied != state.size()) {
        throw std::runtime_error("sequence state copy was truncated");
    }
    return state;
}

static void restore_sequence_state(
        llama_context * ctx,
        const std::vector<uint8_t> & state,
        llama_state_seq_flags flags = LLAMA_STATE_SEQ_FLAGS_NONE) {
    llama_memory_clear(llama_get_memory(ctx), true);
    const size_t restored = llama_state_seq_set_data_ext(
            ctx, state.data(), state.size(), 0, flags);
    if (restored != state.size()) {
        throw std::runtime_error("sequence state restore was truncated");
    }
}

static prediction_observation observe(
        llama_context * ctx,
        int32_t output,
        int32_t n_vocab,
        int32_t prediction_index,
        llama_token input_token,
        llama_token expected_token) {
    return {
        prediction_index,
        input_token,
        expected_token,
        select_top2(llama_get_logits_ith(ctx, output), (size_t) n_vocab),
    };
}

static void prepare_prefix(
        llama_context * ctx,
        llama_token prompt_token,
        int32_t prompt_count,
        const llama_tokens & reference,
        int32_t prediction_start) {
    llama_tokens prompt((size_t) prompt_count, prompt_token);
    decode_rows(ctx, prompt, 0, 0, false);

    // To predict output i, the next input is output i-1. Leave that token for
    // the diagnostic batch and consume only through i-2 here.
    for (int32_t token_index = 0; token_index <= prediction_start - 2; ++token_index) {
        decode_rows(ctx, { reference[(size_t) token_index] }, prompt_count + token_index, 0, true);
    }
}

static std::vector<uint64_t> hash_f32_rows(const float * data, size_t row_floats, size_t rows) {
    if (!data || row_floats == 0 || rows == 0) {
        throw std::invalid_argument("layer-input hashing requires non-empty rows");
    }
    std::vector<uint64_t> result;
    result.reserve(rows);
    for (size_t row = 0; row < rows; ++row) {
        const float * values = data + row * row_floats;
        for (size_t i = 0; i < row_floats; ++i) {
            if (!std::isfinite(values[i])) {
                throw std::runtime_error("block-input capture found a non-finite value");
            }
        }
        result.push_back(fnv1a64((const uint8_t *) values, row_floats * sizeof(float)));
    }
    return result;
}

static bool same_predictions(
        const std::vector<prediction_observation> & left,
        const std::vector<prediction_observation> & right) {
    if (left.size() != right.size()) {
        return false;
    }
    for (size_t i = 0; i < left.size(); ++i) {
        if (left[i].prediction_index != right[i].prediction_index ||
                left[i].input_token != right[i].input_token ||
                left[i].expected_token != right[i].expected_token ||
                !same_top2(left[i].top2, right[i].top2)) {
            return false;
        }
    }
    return true;
}

static bool same_state_vector(const std::vector<state_digest> & left, const std::vector<state_digest> & right) {
    if (left.size() != right.size()) {
        return false;
    }
    for (size_t i = 0; i < left.size(); ++i) {
        if (!same_state(left[i], right[i])) {
            return false;
        }
    }
    return true;
}

static boundary_arm_result run_boundary_arm(
        llama_model * model,
        const common_params & params,
        llama_token prompt_token,
        int32_t prompt_count,
        const llama_tokens & reference,
        int32_t prediction_start,
        uint32_t width,
        int32_t n_vocab,
        boundary_capture::stage stage,
        int32_t target_layer,
        bool target_layer_recurrent,
        cuda_route_api route_api = {},
        cuda_route_counts * route_counts = nullptr,
        bool capture_full_state = false,
        bool per_execute_route = false) {
    boundary_capture capture;
    const int32_t n_embd = llama_model_n_embd(model);
    if (n_embd <= 0) {
        throw std::runtime_error("layer-boundary model embedding width is invalid");
    }
    capture.expected_features = (size_t) n_embd;
    auto ctx = make_context(model, params, width, &capture);
    if (!ctx) {
        throw std::runtime_error("failed to create layer-boundary width-" + std::to_string(width) + " context");
    }
    prepare_prefix(ctx->get(), prompt_token, prompt_count, reference, prediction_start);

    const state_digest prefix_digest = sequence_digest(ctx->get(), LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    // A Qwen3.5 hybrid PARTIAL_ONLY state omits the base KV used by the 16
    // full-attention blocks. Preserve the complete prefix for the replay even
    // though compact comparison records intentionally use the partial digest.
    const std::vector<uint8_t> prefix_state = sequence_state(ctx->get(), LLAMA_STATE_SEQ_FLAGS_NONE);
    if (stage == boundary_capture::stage::refine) {
        if (target_layer < 0 || target_layer >= llama_model_n_layer(model)) {
            throw std::invalid_argument("refinement layer is outside the model");
        }
        llama_set_embeddings_layer_inp(ctx->get(), (uint32_t) target_layer, true);
    }

    auto execute = [&]() {
        boundary_arm_result result;
        result.prefix_state = sequence_digest(ctx->get(), LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        if (!same_state(result.prefix_state, prefix_digest)) {
            throw std::runtime_error("restored layer-boundary prefix differs from the approved prefix");
        }
        capture.begin(width > 1, stage, target_layer, target_layer_recurrent);
        if (width == 1) {
            for (uint32_t row = 0; row < 8; ++row) {
                capture.scalar_row = (int32_t) row;
                const int32_t prediction = prediction_start + (int32_t) row;
                const llama_token input = reference[(size_t) prediction - 1];
                decode_rows(ctx->get(), { input }, prompt_count + prediction - 1, 0, true);
                capture.require_success();
                result.predictions.push_back(observe(
                        ctx->get(), 0, n_vocab, prediction, input, reference[(size_t) prediction]));
                if (stage == boundary_capture::stage::refine) {
                    float * block_input = llama_get_embeddings_layer_inp(ctx->get(), (uint32_t) target_layer);
                    const auto row_hash = hash_f32_rows(block_input, (size_t) n_embd, 1);
                    auto & hashes = capture.snapshot.rows[{ target_layer, "block_input" }];
                    if (hashes.size() != row) {
                        throw std::runtime_error("scalar block-input capture order is not deterministic");
                    }
                    hashes.push_back(row_hash[0]);
                    capture.snapshot.row_data[{ target_layer, "block_input" }].emplace_back(
                            (const uint8_t *) block_input,
                            (const uint8_t *) (block_input + n_embd));
                    capture.snapshot.row_bytes[{ target_layer, "block_input" }] = (size_t) n_embd * sizeof(float);
                }
                result.scalar_states.push_back(sequence_digest(ctx->get(), LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
            }
        } else {
            llama_tokens inputs;
            inputs.reserve(8);
            for (uint32_t row = 0; row < 8; ++row) {
                inputs.push_back(reference[(size_t) prediction_start + row - 1]);
            }
            decode_rows(ctx->get(), inputs, prompt_count + prediction_start - 1, 7, true);
            capture.require_success();
            for (uint32_t row = 0; row < 8; ++row) {
                const int32_t prediction = prediction_start + (int32_t) row;
                result.predictions.push_back(observe(
                        ctx->get(), (int32_t) row, n_vocab, prediction, inputs[row], reference[(size_t) prediction]));
            }
            if (stage == boundary_capture::stage::refine) {
                float * block_input = llama_get_embeddings_layer_inp(ctx->get(), (uint32_t) target_layer);
                capture.snapshot.rows[{ target_layer, "block_input" }] =
                        hash_f32_rows(block_input, (size_t) n_embd, 8);
                auto & raw_rows = capture.snapshot.row_data[{ target_layer, "block_input" }];
                for (uint32_t row = 0; row < 8; ++row) {
                    raw_rows.emplace_back(
                            (const uint8_t *) (block_input + (size_t) row * n_embd),
                            (const uint8_t *) (block_input + (size_t) (row + 1) * n_embd));
                }
                capture.snapshot.row_bytes[{ target_layer, "block_input" }] = (size_t) n_embd * sizeof(float);
            }
            result.scalar_states.push_back(sequence_digest(ctx->get(), LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
        }
        capture.active = false;
        result.capture = capture.snapshot;
        result.final_state = sequence_digest(ctx->get(), LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        if (capture_full_state) {
            result.full_final_state = sequence_digest(ctx->get(), LLAMA_STATE_SEQ_FLAGS_NONE);
        }
        return result;
    };

    if ((route_counts != nullptr) != (route_api.reset != nullptr && route_api.snapshot != nullptr)) {
        throw std::invalid_argument("layer-boundary route capture must bind reset, snapshot, and output together");
    }
    if (per_execute_route && !route_counts) {
        throw std::invalid_argument("per-execute route capture requires a route output");
    }
    if (route_counts) {
        route_api.reset();
    }
    boundary_arm_result result = execute();
    cuda_route_counts first_routes{};
    if (route_counts && per_execute_route) {
        first_routes = require_cuda_route_snapshot(route_api.snapshot);
    }
    restore_sequence_state(ctx->get(), prefix_state, LLAMA_STATE_SEQ_FLAGS_NONE);
    if (route_counts && per_execute_route) {
        route_api.reset();
    }
    boundary_arm_result repeat = execute();
    if (route_counts) {
        const cuda_route_counts final_routes = require_cuda_route_snapshot(route_api.snapshot);
        if (per_execute_route) {
            if (!std::equal(std::begin(first_routes.count), std::end(first_routes.count),
                    std::begin(final_routes.count))) {
                throw std::runtime_error("layer-boundary per-execute routes are not repeat-stable");
            }
            *route_counts = first_routes;
        } else {
            *route_counts = final_routes;
        }
    }
    if (result.capture.rows != repeat.capture.rows || result.capture.row_data != repeat.capture.row_data ||
            result.capture.row_bytes != repeat.capture.row_bytes ||
            result.capture.recurrent_state_inputs != repeat.capture.recurrent_state_inputs ||
            result.capture.recurrent_state_data != repeat.capture.recurrent_state_data ||
            result.capture.recurrent_state_bytes != repeat.capture.recurrent_state_bytes ||
            result.capture.layer_types != repeat.capture.layer_types ||
            result.capture.internal_shapes != repeat.capture.internal_shapes ||
            result.capture.internal_cache_inputs != repeat.capture.internal_cache_inputs ||
            result.capture.internal_cache_data != repeat.capture.internal_cache_data ||
            result.capture.internal_cache_bytes != repeat.capture.internal_cache_bytes ||
            result.capture.internal_cache_shapes != repeat.capture.internal_cache_shapes ||
            !same_state(result.prefix_state, repeat.prefix_state) ||
            !same_state(result.final_state, repeat.final_state) ||
            !same_state(result.full_final_state, repeat.full_final_state) ||
            !same_state_vector(result.scalar_states, repeat.scalar_states) ||
            !same_predictions(result.predictions, repeat.predictions)) {
        throw std::runtime_error("layer-boundary arm repeat is not bitwise stable");
    }
    result.repeat_stable = true;
    return result;
}

static boundary_comparison compare_captured_boundary(
        const boundary_capture_snapshot & scalar,
        const boundary_capture_snapshot & batch,
        const boundary_key & key,
        int32_t prediction_start) {
    const auto scalar_hashes = scalar.rows.find(key);
    const auto batch_hashes = batch.rows.find(key);
    const auto scalar_data = scalar.row_data.find(key);
    const auto batch_data = batch.row_data.find(key);
    const auto scalar_bytes = scalar.row_bytes.find(key);
    const auto batch_bytes = batch.row_bytes.find(key);
    if (scalar_hashes == scalar.rows.end() || batch_hashes == batch.rows.end() ||
            scalar_data == scalar.row_data.end() || batch_data == batch.row_data.end() ||
            scalar_bytes == scalar.row_bytes.end() || batch_bytes == batch.row_bytes.end() ||
            scalar_hashes->second.size() != 8 || batch_hashes->second.size() != 8 ||
            scalar_data->second.size() != 8 || batch_data->second.size() != 8 ||
            scalar_bytes->second == 0 || scalar_bytes->second != batch_bytes->second) {
        throw std::runtime_error("layer-boundary capture is incomplete or shape-inconsistent");
    }
    boundary_comparison result = compare_boundary_rows(
            scalar_hashes->second, batch_hashes->second, prediction_start);
    result.first_mismatch_prediction = -1;
    result.mismatch_count = 0;
    for (size_t row = 0; row < 8; ++row) {
        const bool exact = scalar_data->second[row] == batch_data->second[row];
        const bool digest_equal = scalar_hashes->second[row] == batch_hashes->second[row];
        if (exact != digest_equal) {
            throw std::runtime_error("layer-boundary digest does not agree with exact row bytes");
        }
        if (!exact) {
            if (result.first_mismatch_prediction < 0) {
                result.first_mismatch_prediction = prediction_start + (int32_t) row;
            }
            ++result.mismatch_count;
        }
    }
    return result;
}

static bool same_width_observations(const width_result & baseline, const boundary_arm_result & observed) {
    return same_state(baseline.prefix_state, observed.prefix_state) &&
            same_state(baseline.final_state, observed.final_state) &&
            same_predictions(baseline.predictions, observed.predictions);
}

static bool same_width_observations_full(const width_result & baseline, const boundary_arm_result & observed) {
    return same_width_observations(baseline, observed) &&
            baseline.full_final_state.bytes > 0 && observed.full_final_state.bytes > 0 &&
            same_state(baseline.full_final_state, observed.full_final_state);
}

static width_result run_width(
        llama_model * model,
        const common_params & params,
        llama_token prompt_token,
        int32_t prompt_count,
        const llama_tokens & reference,
        int32_t prediction_start,
        int32_t prediction_count,
        uint32_t width,
        int32_t n_vocab,
        bool capture_full_state = false,
        cuda_route_api route_api = {},
        cuda_route_counts * route_counts = nullptr) {
    auto ctx = make_context(model, params, width);
    if (!ctx) {
        throw std::runtime_error("failed to create width-" + std::to_string(width) + " context");
    }
    prepare_prefix(ctx->get(), prompt_token, prompt_count, reference, prediction_start);

    if ((route_counts != nullptr) != (route_api.reset != nullptr && route_api.snapshot != nullptr)) {
        throw std::invalid_argument("width route capture must bind reset, snapshot, and output together");
    }
    if (route_counts) {
        route_api.reset();
    }

    width_result result;
    result.width = width;
    result.prefix_state = sequence_digest(ctx->get(), LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    bool route_snapshotted = false;
    for (int32_t prediction = prediction_start;
            prediction < prediction_start + prediction_count;) {
        const uint32_t chunk = std::min<uint32_t>(
                width, (uint32_t) (prediction_start + prediction_count - prediction));
        llama_tokens inputs;
        inputs.reserve(chunk);
        for (uint32_t row = 0; row < chunk; ++row) {
            inputs.push_back(reference[(size_t) prediction + row - 1]);
        }
        decode_rows(ctx->get(), inputs, prompt_count + prediction - 1, chunk - 1, true);
        if (route_counts && !route_snapshotted) {
            *route_counts = require_cuda_route_snapshot(route_api.snapshot);
            route_snapshotted = true;
        }
        for (uint32_t row = 0; row < chunk; ++row) {
            result.predictions.push_back(observe(
                    ctx->get(), (int32_t) row, n_vocab, prediction + (int32_t) row,
                    inputs[row], reference[(size_t) prediction + row]));
        }
        prediction += (int32_t) chunk;
    }

    result.final_consumed_token_index = prediction_start + prediction_count - 2;
    result.final_state = sequence_digest(ctx->get(), LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    if (capture_full_state) {
        result.full_final_state = sequence_digest(ctx->get(), LLAMA_STATE_SEQ_FLAGS_NONE);
    }
    if (route_counts && !route_snapshotted) {
        throw std::runtime_error("width route probe did not execute a target batch");
    }
    return result;
}

static cuda_route_counts run_width_route_probe_from_restored_prefix(
        llama_model * model,
        const common_params & params,
        const std::vector<uint8_t> & prefix_state,
        const llama_tokens & reference,
        int32_t prompt_count,
        int32_t prediction_start,
        uint32_t width,
        const cuda_route_api & route_api,
        cuda_fattn_candidate_observation & predicate_observation) {
    auto ctx = make_context(model, params, width);
    if (!ctx) {
        throw std::runtime_error("failed to create graph-cold width route-probe context");
    }
    restore_sequence_state(ctx->get(), prefix_state, LLAMA_STATE_SEQ_FLAGS_NONE);
    route_api.reset();
    llama_tokens inputs;
    inputs.reserve(width);
    for (uint32_t row = 0; row < width; ++row) {
        inputs.push_back(reference[(size_t) prediction_start + row - 1]);
    }
    decode_rows(ctx->get(), inputs, prompt_count + prediction_start - 1, width - 1, true);
    predicate_observation = require_cuda_fattn_candidate_snapshot(
            route_api.fattn_candidate_snapshot);
    return require_cuda_route_snapshot(route_api.snapshot);
}

static std::map<int32_t, oracle_observation> build_scalar_oracle(
        llama_model * model,
        const common_params & params,
        llama_token prompt_token,
        int32_t prompt_count,
        const llama_tokens & reference,
        int32_t prediction_start,
        uint32_t width,
        int32_t n_vocab,
        llama_state_seq_flags state_flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) {
    auto ctx = make_context(model, params, 1);
    if (!ctx) {
        throw std::runtime_error("failed to create scalar oracle context");
    }
    prepare_prefix(ctx->get(), prompt_token, prompt_count, reference, prediction_start);

    std::map<int32_t, oracle_observation> result;
    for (int32_t input_index = prediction_start - 1;
            input_index <= prediction_start + (int32_t) width - 2; ++input_index) {
        decode_rows(ctx->get(), { reference[(size_t) input_index] }, prompt_count + input_index, 0, true);
        const top2_result next = select_top2(llama_get_logits_ith(ctx->get(), 0), (size_t) n_vocab);
        result.emplace(input_index, oracle_observation{
            next,
            sequence_digest(ctx->get(), state_flags),
        });
    }
    return result;
}

static std::vector<top2_result> validate_scalar_trajectory(
        llama_model * model,
        const common_params & params,
        const common_params_sampling & p1_sampling,
        bool reference_runtime_bundle_match,
        llama_token prompt_token,
        int32_t prompt_count,
        const llama_tokens & reference,
        int32_t max_prediction,
        int32_t n_vocab,
        state_digest & final_state,
        int32_t & prompt_tokens_accepted,
        int32_t & generated_tokens_accepted) {
    auto ctx = make_context(model, params, 1);
    if (!ctx) {
        throw std::runtime_error("failed to create scalar reference-validation context");
    }

    llama_tokens prompt((size_t) prompt_count, prompt_token);
    decode_rows(ctx->get(), prompt, 0, 0, false);

    common_params_sampling sampling = p1_sampling;
    common_sampler_ptr sampler(common_sampler_init(model, sampling));
    if (!sampler) {
        throw std::runtime_error("failed to create exact P1 reference sampler");
    }
    prompt_tokens_accepted = 0;
    for (llama_token token : prompt) {
        common_sampler_accept(sampler.get(), token, false);
        ++prompt_tokens_accepted;
    }
    if (prompt_tokens_accepted != prompt_count) {
        throw std::runtime_error("server-equivalent sampler did not accept the complete prompt");
    }

    std::vector<top2_result> result;
    result.reserve((size_t) max_prediction + 1);
    generated_tokens_accepted = 0;
    auto require_reference = [&](int32_t prediction) {
        // The long prompt requests only its final batch row. Negative indexing
        // addresses the last compact output row; it also remains correct for
        // each subsequent single-token batch.
        const top2_result raw = select_top2(llama_get_logits_ith(ctx->get(), -1), (size_t) n_vocab);
        const llama_token sampled = common_sampler_sample(sampler.get(), ctx->get(), -1);
        const llama_token expected = reference[(size_t) prediction];
        if (sampled != expected) {
            const bool raw_top1_is_eog = llama_vocab_is_eog(llama_model_get_vocab(model), raw.top1);
            std::ostringstream message;
            message << std::setprecision(9)
                    << "server-equivalent scalar reference diverges at prediction " << prediction
                    << ": expected_token=" << expected
                    << ", sampler_selected_token=" << sampled
                    << ", raw_top1_token=" << raw.top1
                    << ", raw_top2_token=" << raw.top2
                    << ", raw_top1_logit=" << raw.logit1
                    << ", raw_top2_logit=" << raw.logit2
                    << ", raw_margin=" << raw.margin()
                    << ", raw_top1_is_eog=" << (raw_top1_is_eog ? "true" : "false")
                    << ", reference_runtime_bundle_match="
                    << (reference_runtime_bundle_match ? "true" : "false");
            throw std::runtime_error(message.str());
        }
        common_sampler_accept(sampler.get(), expected, true);
        ++generated_tokens_accepted;
        result.push_back(raw);
    };

    require_reference(0);
    for (int32_t prediction = 1; prediction <= max_prediction; ++prediction) {
        decode_rows(ctx->get(), { reference[(size_t) prediction - 1] },
                prompt_count + prediction - 1, 0, true);
        require_reference(prediction);
    }
    final_state = sequence_digest(ctx->get(), LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    return result;
}

static rollback_observation run_rollback_case(
        llama_context * ctx,
        const std::vector<uint8_t> & prefix_state,
        const llama_tokens & verify_inputs,
        const llama_tokens & reference,
        int32_t prompt_count,
        int32_t prediction_start,
        uint32_t width,
        int32_t n_vocab,
        const rollback_case & item,
        llama_state_seq_flags state_flags) {
    restore_sequence_state(ctx, prefix_state);
    decode_rows(ctx, verify_inputs, prompt_count + prediction_start - 1, width - 1, true);

    rollback_observation result;
    result.batch_rows.reserve(width);
    for (uint32_t row = 0; row < width; ++row) {
        const int32_t input_index = prediction_start - 1 + (int32_t) row;
        result.batch_rows.push_back(observe(
                ctx, (int32_t) row, n_vocab, input_index + 1,
                reference[(size_t) input_index], reference[(size_t) input_index + 1]));
    }
    result.batch_final = sequence_digest(ctx, state_flags);

    const llama_pos rollback_pos = prompt_count + item.continuation_token_index;
    if (!llama_memory_seq_rm(llama_get_memory(ctx), 0, rollback_pos, -1)) {
        throw std::runtime_error("recurrent rollback failed for depth " + std::to_string(item.rollback));
    }
    result.selected = sequence_digest(ctx, state_flags);

    decode_rows(ctx, { reference[(size_t) item.continuation_token_index] },
            prompt_count + item.continuation_token_index, 0, true);
    result.continued = select_top2(llama_get_logits_ith(ctx, 0), (size_t) n_vocab);
    result.continued_state = sequence_digest(ctx, state_flags);
    return result;
}

static bool same_rollback_observation(const rollback_observation & left, const rollback_observation & right) {
    if (!same_state(left.batch_final, right.batch_final) || !same_state(left.selected, right.selected) ||
            !same_top2(left.continued, right.continued) ||
            !same_state(left.continued_state, right.continued_state) ||
            left.batch_rows.size() != right.batch_rows.size()) {
        return false;
    }
    for (size_t i = 0; i < left.batch_rows.size(); ++i) {
        if (!same_top2(left.batch_rows[i].top2, right.batch_rows[i].top2)) {
            return false;
        }
    }
    return true;
}

static bool run_rollbacks(
        llama_model * model,
        const common_params & params,
        llama_token prompt_token,
        int32_t prompt_count,
        const llama_tokens & reference,
        int32_t prediction_start,
        uint32_t width,
        int32_t n_vocab,
        const std::map<int32_t, oracle_observation> & oracle,
        llama_state_seq_flags state_flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) {
    auto ctx = make_context(model, params, width);
    if (!ctx) {
        throw std::runtime_error("failed to create rollback context");
    }
    prepare_prefix(ctx->get(), prompt_token, prompt_count, reference, prediction_start);
    const std::vector<uint8_t> prefix_state = sequence_state(ctx->get(), LLAMA_STATE_SEQ_FLAGS_NONE);

    llama_tokens verify_inputs;
    verify_inputs.reserve(width);
    for (uint32_t row = 0; row < width; ++row) {
        verify_inputs.push_back(reference[(size_t) prediction_start + row - 1]);
    }

    bool batch_observed = false;
    bool exact = true;
    for (const auto & item : make_rollback_cases(prediction_start, width)) {
        const auto result = run_rollback_case(ctx->get(), prefix_state, verify_inputs, reference,
                prompt_count, prediction_start, width, n_vocab, item, state_flags);
        const auto repeat = run_rollback_case(ctx->get(), prefix_state, verify_inputs, reference,
                prompt_count, prediction_start, width, n_vocab, item, state_flags);
        if (!same_rollback_observation(result, repeat)) {
            throw std::runtime_error("rollback repeat is not bitwise stable at depth " + std::to_string(item.rollback));
        }

        if (!batch_observed) {
            bool row_ids_match_scalar = true;
            bool row_values_match_scalar = true;
            for (uint32_t row = 0; row < width; ++row) {
                const auto & observation = result.batch_rows[row];
                print_top2_json("rollback_batch", width, observation);
                const auto & scalar_row = oracle.at(prediction_start - 1 + (int32_t) row).next;
                row_ids_match_scalar &= observation.top2.top1 == scalar_row.top1 &&
                        observation.top2.top2 == scalar_row.top2;
                row_values_match_scalar &= float_bits(observation.top2.logit1) == float_bits(scalar_row.logit1) &&
                        float_bits(observation.top2.logit2) == float_bits(scalar_row.logit2);
            }
            const state_digest oracle_final = oracle.at(prediction_start + (int32_t) width - 2).state;
            const bool batch_state_match = same_state(result.batch_final, oracle_final);
            exact &= row_ids_match_scalar && row_values_match_scalar && batch_state_match;
            std::printf(
                    "{\"type\":\"rollback_batch_summary\",\"width\":%u,"
                    "\"row_top2_ids_match_scalar\":%s,\"row_top2_values_match_scalar\":%s,"
                    "\"final_state_hash\":\"%s\",\"oracle_final_state_hash\":\"%s\","
                    "\"final_state_bytes\":%zu,\"oracle_final_state_bytes\":%zu,"
                    "\"final_state_match_scalar\":%s,\"state_scope\":\"%s\",\"repeat_stable\":true,"
                    "\"intermediate_snapshot_generation_vs_selection_separable\":false}\n",
                    width, row_ids_match_scalar ? "true" : "false", row_values_match_scalar ? "true" : "false",
                    hex64(result.batch_final.hash).c_str(), hex64(oracle_final.hash).c_str(),
                    result.batch_final.bytes, oracle_final.bytes,
                    batch_state_match ? "true" : "false",
                    state_flags == LLAMA_STATE_SEQ_FLAGS_NONE ? "full" : "partial");
            batch_observed = true;
        }

        const auto & committed_oracle = oracle.at(item.committed_token_index);
        const auto & continuation_oracle = oracle.at(item.continuation_token_index);
        const bool continued_top2_match = result.continued.top1 == continuation_oracle.next.top1 &&
                result.continued.top2 == continuation_oracle.next.top2;
        const bool continued_top2_values_match = same_top2(result.continued, continuation_oracle.next);
        const bool selected_state_match = same_state(result.selected, committed_oracle.state);
        const bool continued_state_match = same_state(result.continued_state, continuation_oracle.state);
        exact &= selected_state_match && continued_state_match &&
                continued_top2_match && continued_top2_values_match;

        std::printf(
                "{\"type\":\"rollback\",\"width\":%u,\"rollback\":%u,"
                "\"committed_token_index\":%d,\"continuation_token_index\":%d,"
                "\"continuation_prediction_index\":%d,\"continued_top1\":%d,"
                "\"oracle_top1\":%d,\"continued_margin\":%.9g,\"oracle_margin\":%.9g,"
                "\"selected_state_hash\":\"%s\",\"oracle_selected_state_hash\":\"%s\","
                "\"continued_state_hash\":\"%s\",\"oracle_continued_state_hash\":\"%s\","
                "\"state_bytes\":%zu,\"oracle_selected_state_bytes\":%zu,\"oracle_continued_state_bytes\":%zu,"
                "\"selected_state_match\":%s,\"continued_state_match\":%s,\"continued_top2_match\":%s,"
                "\"continued_top2_values_match\":%s,\"repeat_stable\":true,"
                "\"state_scope\":\"%s\",\"snapshot_generation_vs_selection_separable\":false}\n",
                width, item.rollback, item.committed_token_index, item.continuation_token_index,
                item.continuation_prediction_index, result.continued.top1, continuation_oracle.next.top1,
                result.continued.margin(), continuation_oracle.next.margin(),
                hex64(result.selected.hash).c_str(), hex64(committed_oracle.state.hash).c_str(),
                hex64(result.continued_state.hash).c_str(), hex64(continuation_oracle.state.hash).c_str(),
                result.selected.bytes, committed_oracle.state.bytes, continuation_oracle.state.bytes,
                selected_state_match ? "true" : "false",
                continued_state_match ? "true" : "false",
                continued_top2_match ? "true" : "false",
                continued_top2_values_match ? "true" : "false",
                state_flags == LLAMA_STATE_SEQ_FLAGS_NONE ? "full" : "partial");
    }
    return exact;
}

static void print_usage(int, char ** argv) {
    LOG("\nGuarded Qwen3.8 recurrent parity diagnostic.\n");
    LOG("Reference tokens and the safety guard are supplied by scripts/Invoke-Qwen38ParityDiagnostic.ps1.\n");
    LOG("Example (normally use the wrapper):\n  %s -m model.gguf -c 2048 -b 2048 -ub 128 -ctk q8_0 -ctv q8_0 -fa on -ngl all --device CUDA0 --split-mode none --fit off --cache-ram 0 --ctx-checkpoints 0 --reasoning off --kv-unified\n\n", argv[0]);
}

} // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    common_init();
    bool backend_initialized = false;
    const char * failure_stage = "guard_validation";

    try {
        if (std::strcmp(get_env(GUARD_NAME), GUARD_VALUE) != 0) {
            throw std::runtime_error(std::string("refusing to run: set ") +
                    GUARD_NAME + "=" + GUARD_VALUE + " through the guarded wrapper");
        }
        if (std::strcmp(get_env("AI_LOADER_EXCLUSIVE_GPU_GUARD"), "1") != 0 ||
                *get_env("AI_LOADER_EXCLUSIVE_GPU_UUID") == '\0' ||
                *get_env("AI_LOADER_EXCLUSIVE_GPU_LEASE_PATH") == '\0' ||
                *get_env("AI_LOADER_EXCLUSIVE_GPU_LEASE_NONCE") == '\0' ||
                *get_env("AI_LOADER_EXCLUSIVE_GPU_OWNER_PID") == '\0' ||
                *get_env("AI_LOADER_EXCLUSIVE_GPU_JOB_NAME") == '\0' ||
                std::strcmp(get_env("CUDA_VISIBLE_DEVICES"), get_env("AI_LOADER_EXCLUSIVE_GPU_UUID")) != 0) {
            throw std::runtime_error("refusing to run outside scripts/Invoke-ExclusiveGpuTask.ps1");
        }

        failure_stage = "identity_validation";
        const std::string executable_sha256 = get_required_sha256("LLAMACPP_QWEN38_PARITY_EXECUTABLE_SHA256");
        const std::string model_sha256       = get_required_sha256("LLAMACPP_QWEN38_PARITY_MODEL_SHA256");
        const std::string reference_sha256   = get_required_sha256("LLAMACPP_QWEN38_PARITY_REFERENCE_SHA256");
        const std::string manifest_sha256    = get_required_sha256("LLAMACPP_QWEN38_PARITY_MANIFEST_SHA256");
        const std::string command_sha256     = get_required_sha256("LLAMACPP_QWEN38_PARITY_COMMAND_SHA256");
        const std::string receipt_sha256     = get_required_sha256("LLAMACPP_QWEN38_PARITY_RECEIPT_SHA256");
        const std::string current_llama_sha256 = get_required_sha256("LLAMACPP_QWEN38_PARITY_CURRENT_LLAMA_SHA256");
        const std::string reference_llama_sha256 = get_required_sha256("LLAMACPP_QWEN38_PARITY_REFERENCE_LLAMA_SHA256");
        const std::string current_common_sha256 = get_required_sha256("LLAMACPP_QWEN38_PARITY_CURRENT_COMMON_SHA256");
        const std::string reference_common_sha256 = get_required_sha256("LLAMACPP_QWEN38_PARITY_REFERENCE_COMMON_SHA256");
        const std::string current_bundle_sha256 = get_required_sha256("LLAMACPP_QWEN38_PARITY_CURRENT_BUNDLE_SHA256");
        const std::string reference_bundle_sha256 = get_required_sha256("LLAMACPP_QWEN38_PARITY_REFERENCE_BUNDLE_SHA256");
        const std::string current_server_sha256 = get_required_sha256("LLAMACPP_QWEN38_PARITY_CURRENT_SERVER_SHA256");
        const std::string reference_server_sha256 = get_required_sha256("LLAMACPP_QWEN38_PARITY_REFERENCE_SERVER_SHA256");
        const int32_t runtime_bundle_count = get_env_i32("LLAMACPP_QWEN38_PARITY_BUNDLE_COUNT", 0);
        const std::string runtime_match_text = get_env("LLAMACPP_QWEN38_PARITY_RUNTIME_BUNDLE_MATCH");
        const std::string diagnostic_candidate = get_env("LLAMACPP_QWEN38_PARITY_DIAGNOSTIC_CANDIDATE");
        const std::string layer_boundary_text = get_env("LLAMACPP_QWEN38_PARITY_LAYER_BOUNDARY_BISECTION");
        const std::string recurrent_internal_text = get_env("LLAMACPP_QWEN38_PARITY_RECURRENT_INTERNAL_REFINE");
        const std::string full_attention_internal_text =
                get_env("LLAMACPP_QWEN38_PARITY_FULL_ATTENTION_INTERNAL_REFINE");
        const bool projection_candidate_marker =
                diagnostic_candidate == "QWEN35_NVFP4_FFN_PROJECTIONS_V1" ||
                diagnostic_candidate == "QWEN35_NVFP4_FFN_PLUS_BF16_RECURRENT_PROJECTIONS_V2" ||
                diagnostic_candidate == "QWEN35_NVFP4_FFN_PLUS_BF16_RECURRENT_PLUS_FATTN_VEC_V3";
        const bool bf16_recurrent_candidate_marker =
                diagnostic_candidate == "QWEN35_BF16_RECURRENT_PROJECTIONS_V1" ||
                diagnostic_candidate == "QWEN35_NVFP4_FFN_PLUS_BF16_RECURRENT_PROJECTIONS_V2" ||
                diagnostic_candidate == "QWEN35_NVFP4_FFN_PLUS_BF16_RECURRENT_PLUS_FATTN_VEC_V3";
        const bool fattn_vec_candidate_marker =
                diagnostic_candidate == "QWEN35_NVFP4_FFN_PLUS_BF16_RECURRENT_PLUS_FATTN_VEC_V3";
        const bool projection_opt_in =
                std::strcmp(get_env("LLAMA_QWEN35_NVFP4_ROW_INVARIANT_PROJECTIONS"), "1") == 0;
        const bool bf16_recurrent_opt_in = std::strcmp(
                get_env("LLAMA_QWEN35_BF16_ROW_INVARIANT_RECURRENT_PROJECTIONS"), "1") == 0;
        const bool fattn_vec_opt_in = std::strcmp(
                get_env("LLAMA_QWEN35_FATTN_D256_Q8_GQA6_VEC_QCOLS3_8"), "1") == 0;
        if ((!diagnostic_candidate.empty() && !projection_candidate_marker &&
                    !bf16_recurrent_candidate_marker && !fattn_vec_candidate_marker) ||
                projection_candidate_marker != projection_opt_in ||
                bf16_recurrent_candidate_marker != bf16_recurrent_opt_in ||
                fattn_vec_candidate_marker != fattn_vec_opt_in) {
            throw std::invalid_argument("Qwen3.5 diagnostic candidate identity and graph opt-ins must be bound exactly");
        }
        const bool projection_candidate = projection_candidate_marker && projection_opt_in;
        const bool bf16_recurrent_candidate = bf16_recurrent_candidate_marker && bf16_recurrent_opt_in;
        const bool fattn_vec_candidate = fattn_vec_candidate_marker && fattn_vec_opt_in;
        if (!layer_boundary_text.empty() && layer_boundary_text != "1") {
            throw std::invalid_argument("layer-boundary bisection marker must be empty or 1");
        }
        const bool layer_boundary_bisection = layer_boundary_text == "1";
        if (!recurrent_internal_text.empty() && recurrent_internal_text != "1") {
            throw std::invalid_argument("recurrent-internal refinement marker must be empty or 1");
        }
        const bool recurrent_internal_refine = recurrent_internal_text == "1";
        if (!full_attention_internal_text.empty() && full_attention_internal_text != "1") {
            throw std::invalid_argument("full-attention-internal refinement marker must be empty or 1");
        }
        const bool full_attention_internal_refine = full_attention_internal_text == "1";
        if (layer_boundary_bisection && !projection_candidate && !bf16_recurrent_candidate) {
            throw std::invalid_argument("layer-boundary bisection requires at least one explicit reviewed Qwen3.5 candidate");
        }
        if (recurrent_internal_refine && !layer_boundary_bisection) {
            throw std::invalid_argument("recurrent-internal refinement requires layer-boundary bisection");
        }
        if (full_attention_internal_refine && !layer_boundary_bisection) {
            throw std::invalid_argument("full-attention-internal refinement requires layer-boundary bisection");
        }
        if (full_attention_internal_refine && (!projection_candidate || !bf16_recurrent_candidate)) {
            throw std::invalid_argument("full-attention-internal refinement requires the explicit combined V2 candidate");
        }
        if (full_attention_internal_refine && fattn_vec_candidate) {
            throw std::invalid_argument("full-attention internal refinement is pinned to the pre-selector V2 candidate");
        }
        if (recurrent_internal_refine && full_attention_internal_refine) {
            throw std::invalid_argument("recurrent and full-attention internal refinement are mutually exclusive");
        }
        const bool reference_runtime_bundle_match = runtime_match_text == "true" &&
                current_bundle_sha256 == reference_bundle_sha256 &&
                current_server_sha256 == reference_server_sha256;
        if ((!reference_runtime_bundle_match && !projection_candidate && !bf16_recurrent_candidate &&
                    !fattn_vec_candidate) || runtime_bundle_count <= 0) {
            throw std::invalid_argument("reference trace runtime bundle is not identical to the complete adjacent runtime bundle");
        }

        const llama_tokens reference = [&]() {
            const auto parsed = parse_token_csv(get_env(TOKENS_NAME));
            return llama_tokens(parsed.begin(), parsed.end());
        }();
        const int32_t prompt_token = get_env_i32("LLAMACPP_QWEN38_PARITY_PROMPT_TOKEN", 1);
        const int32_t prompt_count = get_env_i32("LLAMACPP_QWEN38_PARITY_PROMPT_COUNT", 736);
        const int32_t prediction_start = get_env_i32("LLAMACPP_QWEN38_PARITY_PREDICTION_START", 176);
        const int32_t prediction_count = get_env_i32("LLAMACPP_QWEN38_PARITY_PREDICTION_COUNT", 24);
        const int32_t rollback_start = get_env_i32("LLAMACPP_QWEN38_PARITY_ROLLBACK_START", 184);

        if (prompt_count <= 0 || prediction_start <= 0 || prediction_count <= 0 ||
                prediction_count % 8 != 0 || rollback_start <= 0 ||
                (size_t) (prediction_start + prediction_count) > reference.size() ||
                (size_t) (rollback_start + 8) > reference.size()) {
            throw std::invalid_argument("diagnostic window is outside the deterministic reference trace");
        }
        if (layer_boundary_bisection && prediction_count != 8) {
            throw std::invalid_argument("layer-boundary bisection requires exactly one width-8 window");
        }
        if (full_attention_internal_refine &&
                (prediction_start != 176 || prediction_count != 8 || rollback_start != 184)) {
            throw std::invalid_argument(
                    "full-attention-internal refinement is pinned to predictions 176..183 and rollback start 184");
        }

        common_params params;
        failure_stage = "argument_parse";
        // The diagnostic reproduces the server's target-context contract. In
        // particular, --kv-unified is intentionally a server-scoped option.
        if (!common_params_parse_no_system_config(argc, argv, params, LLAMA_EXAMPLE_SERVER, print_usage)) {
            return 1;
        }
        if (params.model.path.empty()) {
            throw std::invalid_argument("-m <target.gguf> is required");
        }
        failure_stage = "argument_contract";
        const bool exact_device = params.devices.size() == 2 &&
                params.devices[0] != nullptr && params.devices[1] == nullptr &&
                std::strcmp(ggml_backend_dev_name(params.devices[0]), "CUDA0") == 0;
        if (params.n_parallel != 1 || params.n_ctx != 2048 || params.n_batch != 2048 || params.n_ubatch != 128 ||
                params.cache_type_k != GGML_TYPE_Q8_0 || params.cache_type_v != GGML_TYPE_Q8_0 ||
                params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_ENABLED || !params.kv_unified ||
                !params.warmup ||
                params.fit_params || params.cache_ram_mib != 0 || params.n_ctx_checkpoints != 0 ||
                params.cache_idle_slots || params.cache_prompt || params.ui || !params.endpoint_metrics ||
                params.enable_reasoning != 0 || params.verbosity != 4 ||
                params.n_gpu_layers != -2 || params.split_mode != LLAMA_SPLIT_MODE_NONE ||
                !exact_device || !llama_supports_gpu_offload()) {
            throw std::invalid_argument(
                    "diagnostic requires exact P1 target settings: -np 1 -c 2048 -b 2048 -ub 128 "
                    "-ctk q8_0 -ctv q8_0 -fa on -ngl all --device CUDA0 --split-mode none "
                    "--fit off --cache-ram 0 --ctx-checkpoints 0 --no-cache-idle-slots --no-cache-prompt "
                    "--no-webui --metrics --reasoning off -lv 4 --kv-unified");
        }

        failure_stage = "backend_init";
        llama_backend_init();
        backend_initialized = true;
        llama_numa_init(params.numa);
        failure_stage = "model_load";
        llama_model_ptr model(llama_model_load_from_file(
                params.model.path.c_str(), common_model_params_to_llama(params)));
        if (!model) {
            throw std::runtime_error("failed to load target model");
        }
        if (!llama_model_is_recurrent(model.get()) && !llama_model_is_hybrid(model.get())) {
            throw std::invalid_argument("target model is not recurrent or hybrid");
        }

        failure_stage = "diagnostic";
        const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        for (llama_token token : reference) {
            if (token < 0 || token >= n_vocab) {
                throw std::invalid_argument("reference trace contains an out-of-vocabulary token");
            }
        }
        if (prompt_token < 0 || prompt_token >= n_vocab) {
            throw std::invalid_argument("prompt token is out of vocabulary");
        }

        int32_t n_eog_biases = 0;
        const common_params_sampling p1_sampling = make_p1_sampling_params(
                model.get(), params, n_eog_biases);

        std::vector<uint8_t> recurrent_layer_map;
        std::string recurrent_layer_map_source;
        uint32_t recurrent_layer_count = 0;
        uint32_t full_attention_layer_count = 0;
        if (layer_boundary_bisection) {
            if (llama_model_n_layer(model.get()) != 64) {
                throw std::runtime_error("reviewed layer-boundary contract requires exactly 64 Qwen3.5 blocks");
            }
            recurrent_layer_map = loaded_recurrent_layer_map(model.get(), 64, recurrent_layer_map_source);
            for (int32_t il = 0; il < 64; ++il) {
                const bool recurrent = recurrent_layer_map[(size_t) il] != 0;
                recurrent_layer_count += recurrent ? 1 : 0;
                full_attention_layer_count += recurrent ? 0 : 1;
                if (recurrent != ((il + 1) % 4 != 0)) {
                    throw std::runtime_error("loaded model recurrent-layer metadata differs from the pinned Qwen3.8 map");
                }
            }
            if (recurrent_layer_count != 48 || full_attention_layer_count != 16) {
                throw std::runtime_error("loaded model does not have the reviewed 48 recurrent / 16 full-attention layout");
            }
        }
        const uint64_t recurrent_layer_map_hash = recurrent_layer_map.empty() ? 0 :
                fnv1a64(recurrent_layer_map.data(), recurrent_layer_map.size());

        std::printf(
                "{\"type\":\"config\",\"prompt_token\":%d,\"prompt_count\":%d,"
                "\"reference_tokens\":%zu,\"prediction_start\":%d,\"prediction_count\":%d,"
                "\"rollback_start\":%d,\"one_context_at_a_time\":true,"
                "\"exclusive_gpu_guard\":true,\"repeat_stability_required\":true,"
                "\"executable_sha256\":\"%s\",\"model_sha256\":\"%s\","
                "\"reference_sha256\":\"%s\",\"manifest_sha256\":\"%s\","
                "\"command_sha256\":\"%s\",\"receipt_sha256\":\"%s\"," 
                "\"reference_sampler\":\"production_cpu_chain\"," 
                "\"temperature\":0,\"seed\":%u,\"ignore_eos\":true," 
                "\"backend_sampling\":false,\"eog_biases\":%d," 
                "\"server_warmup_replayed\":true," 
                "\"server_threadpools_attached\":true," 
                "\"full_reference_trajectory_required\":true," 
                "\"prompt_tokens_accepted_per_pass\":%d,\"generated_tokens_accepted_per_pass\":%zu," 
                "\"reference_runtime_bundle_match\":%s," 
                "\"runtime_bundle_count\":%d," 
                "\"current_bundle_sha256\":\"%s\",\"reference_bundle_sha256\":\"%s\"," 
                "\"current_server_sha256\":\"%s\",\"reference_server_sha256\":\"%s\"," 
                "\"current_llama_sha256\":\"%s\",\"reference_llama_sha256\":\"%s\"," 
                "\"current_common_sha256\":\"%s\",\"reference_common_sha256\":\"%s\"," 
                "\"diagnostic_candidate\":\"%s\",\"qwen35_nvfp4_ffn_projections\":%s,"
                "\"qwen35_bf16_recurrent_projections\":%s,"
                "\"qwen35_fattn_d256_q8_gqa6_vec_qcols3_8\":%s,"
                "\"qwen35_nvfp4_head_hint_compiled\":true,"
                "\"layer_boundary_bisection\":%s,\"layer_boundary_width\":%d,"
                "\"recurrent_internal_refine\":%s,\"recurrent_internal_layer\":%d,"
                "\"full_attention_internal_refine\":%s,\"full_attention_internal_layer\":%d,"
                "\"layer_type_map_attested\":%s,\"recurrent_layer_map_hash\":\"%s\","
                "\"recurrent_layer_map_source\":\"%s\","
                "\"recurrent_layers\":%u,\"full_attention_layers\":%u}\n",
                prompt_token, prompt_count, reference.size(), prediction_start, prediction_count, rollback_start,
                executable_sha256.c_str(), model_sha256.c_str(), reference_sha256.c_str(),
                manifest_sha256.c_str(), command_sha256.c_str(), receipt_sha256.c_str(),
                P1_SEED, n_eog_biases,
                prompt_count, reference.size(),
                reference_runtime_bundle_match ? "true" : "false",
                runtime_bundle_count,
                current_bundle_sha256.c_str(), reference_bundle_sha256.c_str(),
                current_server_sha256.c_str(), reference_server_sha256.c_str(),
                current_llama_sha256.c_str(), reference_llama_sha256.c_str(),
                current_common_sha256.c_str(), reference_common_sha256.c_str(),
                diagnostic_candidate.c_str(), projection_opt_in ? "true" : "false",
                bf16_recurrent_opt_in ? "true" : "false",
                fattn_vec_opt_in ? "true" : "false",
                layer_boundary_bisection ? "true" : "false", layer_boundary_bisection ? 8 : 0,
                recurrent_internal_refine ? "true" : "false", recurrent_internal_refine ? 0 : -1,
                full_attention_internal_refine ? "true" : "false", full_attention_internal_refine ? 3 : -1,
                layer_boundary_bisection ? "true" : "false", hex64(recurrent_layer_map_hash).c_str(),
                recurrent_layer_map_source.c_str(),
                recurrent_layer_count, full_attention_layer_count);

        const int32_t max_diagnostic_prediction = (int32_t) reference.size() - 1;
        state_digest trajectory_final;
        state_digest trajectory_repeat_final;
        int32_t prompt_tokens_accepted = 0;
        int32_t generated_tokens_accepted = 0;
        int32_t repeat_prompt_tokens_accepted = 0;
        int32_t repeat_generated_tokens_accepted = 0;
        const auto trajectory = validate_scalar_trajectory(model.get(), params, p1_sampling,
                reference_runtime_bundle_match, prompt_token, prompt_count,
                reference, max_diagnostic_prediction, n_vocab, trajectory_final,
                prompt_tokens_accepted, generated_tokens_accepted);
        const auto trajectory_repeat = validate_scalar_trajectory(model.get(), params, p1_sampling,
                reference_runtime_bundle_match, prompt_token, prompt_count,
                reference, max_diagnostic_prediction, n_vocab, trajectory_repeat_final,
                repeat_prompt_tokens_accepted, repeat_generated_tokens_accepted);
        if (trajectory.size() != trajectory_repeat.size() ||
                !same_state(trajectory_final, trajectory_repeat_final) ||
                prompt_tokens_accepted != prompt_count || repeat_prompt_tokens_accepted != prompt_count ||
                generated_tokens_accepted != (int32_t) reference.size() ||
                repeat_generated_tokens_accepted != (int32_t) reference.size()) {
            throw std::runtime_error("scalar reference trajectory repeat is not stable");
        }
        for (size_t i = 0; i < trajectory.size(); ++i) {
            if (!same_top2(trajectory[i], trajectory_repeat[i])) {
                throw std::runtime_error(
                        "scalar reference trajectory repeat differs at prediction " + std::to_string(i));
            }
        }
        std::printf(
                "{\"type\":\"reference_validation\",\"first_prediction\":0,"
                "\"last_prediction\":%d,\"predictions_checked\":%zu,"
                "\"prompt_tokens_accepted_per_pass\":%d,\"generated_tokens_accepted_per_pass\":%d,"
                "\"final_state_hash\":\"%s\",\"final_state_bytes\":%zu,\"repeat_stable\":true}\n",
                max_diagnostic_prediction, trajectory.size(),
                prompt_tokens_accepted, generated_tokens_accepted,
                hex64(trajectory_final.hash).c_str(), trajectory_final.bytes);

        if (layer_boundary_bisection) {
            const int32_t n_layer = llama_model_n_layer(model.get());
            const cuda_route_api full_attention_route_api = full_attention_internal_refine ?
                    require_cuda_route_api() : cuda_route_api{};
            cuda_route_counts scalar_baseline_routes{};
            cuda_route_counts batch_baseline_routes{};
            // Establish callback-free observations first. Each helper destroys
            // its context before the next one is created.
            const width_result scalar_baseline = run_width(
                    model.get(), params, prompt_token, prompt_count, reference,
                    prediction_start, 8, 1, n_vocab, full_attention_internal_refine,
                    full_attention_route_api,
                    full_attention_internal_refine ? &scalar_baseline_routes : nullptr);
            const width_result batch_baseline = run_width(
                    model.get(), params, prompt_token, prompt_count, reference,
                    prediction_start, 8, 8, n_vocab, full_attention_internal_refine,
                    full_attention_route_api,
                    full_attention_internal_refine ? &batch_baseline_routes : nullptr);

            // Coarse pass: only block outputs for all layers. The scalar and
            // width-8 contexts never coexist, and each pass is repeated from a
            // restored identical prefix inside its own context.
            const boundary_arm_result scalar_coarse = run_boundary_arm(
                    model.get(), params, prompt_token, prompt_count, reference,
                    prediction_start, 1, n_vocab, boundary_capture::stage::coarse, -1, false);
            const boundary_arm_result batch_coarse = run_boundary_arm(
                    model.get(), params, prompt_token, prompt_count, reference,
                    prediction_start, 8, n_vocab, boundary_capture::stage::coarse, -1, false);
            if (!same_width_observations(scalar_baseline, scalar_coarse) ||
                    !same_width_observations(batch_baseline, batch_coarse)) {
                throw std::runtime_error("coarse callback changed logits or partial state");
            }
            if (scalar_coarse.capture.rows.size() != (size_t) n_layer ||
                    batch_coarse.capture.rows.size() != (size_t) n_layer ||
                    !scalar_coarse.capture.recurrent_state_inputs.empty() ||
                    !batch_coarse.capture.recurrent_state_inputs.empty()) {
                throw std::runtime_error("coarse layer-boundary capture did not isolate exactly 64 block outputs");
            }

            int32_t first_bad_layer = -1;
            int32_t first_bad_prediction = -1;
            for (int32_t il = 0; il < n_layer; ++il) {
                const boundary_key key = { il, "block_output" };
                const auto comparison = compare_captured_boundary(
                        scalar_coarse.capture, batch_coarse.capture, key, prediction_start);
                if (first_bad_layer < 0 && comparison.mismatch_count != 0) {
                    first_bad_layer = il;
                    first_bad_prediction = comparison.first_mismatch_prediction;
                }
                const bool recurrent = recurrent_layer_map[(size_t) il] != 0;
                std::printf(
                        "{\"type\":\"layer_block\",\"layer\":%d,\"layer_type\":\"%s\","
                        "\"boundary\":\"block_output\",\"rows\":8,\"row_bytes\":%zu,"
                        "\"first_mismatch_prediction\":%d,\"mismatch_count\":%u,"
                        "\"scalar_window_hash\":\"%s\",\"batch_window_hash\":\"%s\","
                        "\"finite\":true,\"exact_row_comparison\":true,\"repeat_stable\":true}\n",
                        il, recurrent ? "recurrent" : "full_attention",
                        scalar_coarse.capture.row_bytes.at(key),
                        comparison.first_mismatch_prediction, comparison.mismatch_count,
                        hex64(comparison.scalar_window_hash).c_str(),
                        hex64(comparison.batch_window_hash).c_str());
            }

            std::printf(
                    "{\"type\":\"observer_invariance\",\"stage\":\"coarse\","
                    "\"scalar_top2_match_unobserved\":true,\"batch_top2_match_unobserved\":true,"
                    "\"scalar_partial_state_digest_match_unobserved\":true,"
                    "\"batch_partial_state_digest_match_unobserved\":true,\"repeat_stable\":true}\n");

            int32_t first_boundary_prediction = first_bad_prediction;
            std::string first_boundary = first_bad_layer >= 0 ? "block_output" : "";
            bool refine_executed = false;
            if (first_bad_layer >= 0) {
                const boundary_arm_result scalar_refine = run_boundary_arm(
                        model.get(), params, prompt_token, prompt_count, reference,
                        prediction_start, 1, n_vocab, boundary_capture::stage::refine, first_bad_layer,
                        recurrent_layer_map[(size_t) first_bad_layer] != 0);
                const boundary_arm_result batch_refine = run_boundary_arm(
                        model.get(), params, prompt_token, prompt_count, reference,
                        prediction_start, 8, n_vocab, boundary_capture::stage::refine, first_bad_layer,
                        recurrent_layer_map[(size_t) first_bad_layer] != 0);
                if (!same_width_observations(scalar_baseline, scalar_refine) ||
                        !same_width_observations(batch_baseline, batch_refine)) {
                    throw std::runtime_error("refinement callback changed logits or partial state");
                }
                const bool recurrent = recurrent_layer_map[(size_t) first_bad_layer] != 0;
                const auto type_scalar = scalar_refine.capture.layer_types.find(first_bad_layer);
                const auto type_batch = batch_refine.capture.layer_types.find(first_bad_layer);
                const std::string expected_type = recurrent ? "recurrent" : "full_attention";
                if (type_scalar == scalar_refine.capture.layer_types.end() ||
                        type_batch == batch_refine.capture.layer_types.end() ||
                        type_scalar->second != expected_type || type_batch->second != expected_type ||
                        scalar_refine.capture.rows.size() != 5 || batch_refine.capture.rows.size() != 5) {
                    throw std::runtime_error("refinement capture did not attest the exact five layer boundaries");
                }

                first_boundary.clear();
                first_boundary_prediction = -1;
                for (const char * boundary : std::array<const char *, 5>{
                        "block_input", "attention_output", "attention_residual", "ffn_output", "block_output" }) {
                    const boundary_key key = { first_bad_layer, boundary };
                    const auto comparison = compare_captured_boundary(
                            scalar_refine.capture, batch_refine.capture, key, prediction_start);
                    if (first_boundary.empty() && comparison.mismatch_count != 0) {
                        first_boundary = boundary;
                        first_boundary_prediction = comparison.first_mismatch_prediction;
                    }
                    std::printf(
                            "{\"type\":\"layer_boundary\",\"layer\":%d,\"layer_type\":\"%s\","
                            "\"boundary\":\"%s\",\"rows\":8,\"row_bytes\":%zu,"
                            "\"first_mismatch_prediction\":%d,\"mismatch_count\":%u,"
                            "\"scalar_window_hash\":\"%s\",\"batch_window_hash\":\"%s\","
                            "\"finite\":true,\"exact_row_comparison\":true,\"repeat_stable\":true}\n",
                            first_bad_layer, expected_type.c_str(), boundary,
                            scalar_refine.capture.row_bytes.at(key),
                            comparison.first_mismatch_prediction, comparison.mismatch_count,
                            hex64(comparison.scalar_window_hash).c_str(),
                            hex64(comparison.batch_window_hash).c_str());
                }

                if (recurrent) {
                    const auto & scalar_hashes = scalar_refine.capture.recurrent_state_inputs.at(first_bad_layer);
                    const auto & batch_hashes = batch_refine.capture.recurrent_state_inputs.at(first_bad_layer);
                    const auto & scalar_raw = scalar_refine.capture.recurrent_state_data.at(first_bad_layer);
                    const auto & batch_raw = batch_refine.capture.recurrent_state_data.at(first_bad_layer);
                    if (scalar_hashes.size() != 8 || batch_hashes.size() != 1 ||
                            scalar_raw.size() != 8 || batch_raw.size() != 1 ||
                            (scalar_hashes[0] == batch_hashes[0]) != (scalar_raw[0] == batch_raw[0])) {
                        throw std::runtime_error("recurrent state-input capture is incomplete or hash-inconsistent");
                    }
                    std::printf(
                            "{\"type\":\"recurrent_state_boundary\",\"layer\":%d,"
                            "\"boundary\":\"state_predelta\",\"scalar_steps\":8,\"batch_prefix_states\":1,"
                            "\"state_bytes\":%zu,\"prefix_exact_match\":%s,"
                            "\"scalar_evolution_hash\":\"%s\",\"batch_prefix_hash\":\"%s\","
                            "\"finite\":true,\"repeat_stable\":true}\n",
                            first_bad_layer, scalar_refine.capture.recurrent_state_bytes.at(first_bad_layer),
                            scalar_raw[0] == batch_raw[0] ? "true" : "false",
                            hex64(hash_u64_sequence(scalar_hashes)).c_str(), hex64(batch_hashes[0]).c_str());
                } else if (!scalar_refine.capture.recurrent_state_inputs.empty() ||
                        !batch_refine.capture.recurrent_state_inputs.empty()) {
                    throw std::runtime_error("full-attention refinement unexpectedly captured recurrent state");
                }

                std::printf(
                        "{\"type\":\"observer_invariance\",\"stage\":\"refine\","
                        "\"scalar_top2_match_unobserved\":true,\"batch_top2_match_unobserved\":true,"
                        "\"scalar_partial_state_digest_match_unobserved\":true,"
                        "\"batch_partial_state_digest_match_unobserved\":true,\"repeat_stable\":true}\n");
                refine_executed = true;
            }

            if (recurrent_internal_refine) {
                if (first_bad_layer != 0 || recurrent_layer_map[0] == 0 ||
                        first_boundary != "attention_output" || first_boundary_prediction != prediction_start) {
                    throw std::runtime_error(
                            "recurrent-internal refinement requires the reviewed layer-0 attention-output mismatch");
                }
                const cuda_route_api route_api = require_cuda_route_api();
                cuda_route_counts scalar_routes{};
                const boundary_arm_result scalar_internal = run_boundary_arm(
                        model.get(), params, prompt_token, prompt_count, reference,
                        prediction_start, 1, n_vocab, boundary_capture::stage::recurrent_internal, 0, true,
                        route_api, &scalar_routes);
                cuda_route_counts batch_routes{};
                const boundary_arm_result batch_internal = run_boundary_arm(
                        model.get(), params, prompt_token, prompt_count, reference,
                        prediction_start, 8, n_vocab, boundary_capture::stage::recurrent_internal, 0, true,
                        route_api, &batch_routes);
                if (scalar_routes.count[CUDA_ROUTE_GDN] != 0 || batch_routes.count[CUDA_ROUTE_GDN] != 0 ||
                        scalar_routes.count[CUDA_ROUTE_GDN_FUSED_CACHE] != 768 ||
                        batch_routes.count[CUDA_ROUTE_GDN_FUSED_CACHE] != 96) {
                    throw std::runtime_error("recurrent-internal callbacks did not preserve the exact fused-cache GDN routes");
                }
                if (!same_width_observations(scalar_baseline, scalar_internal) ||
                        !same_width_observations(batch_baseline, batch_internal)) {
                    throw std::runtime_error("recurrent-internal callbacks changed logits or recurrent state");
                }

                static constexpr std::array<const char *, 14> internal_boundaries = {{
                    "attn_norm", "linear_attn_qkv_mixed", "z", "beta", "beta_sigmoid",
                    "alpha", "gate", "conv_output_silu",
                    "q_conv_predelta", "k_conv_predelta", "v_conv_predelta", "gdn_core_output",
                    "final_output", "linear_attn_out",
                }};
                if (scalar_internal.capture.rows.size() != internal_boundaries.size() ||
                        batch_internal.capture.rows.size() != internal_boundaries.size() ||
                        scalar_internal.capture.internal_shapes.size() != internal_boundaries.size() ||
                        batch_internal.capture.internal_shapes.size() != internal_boundaries.size()) {
                    throw std::runtime_error("recurrent-internal capture did not emit the exact activation set");
                }

                std::string first_internal_boundary;
                int32_t first_internal_prediction = -1;
                for (size_t boundary_index = 0; boundary_index < internal_boundaries.size(); ++boundary_index) {
                    const char * boundary = internal_boundaries[boundary_index];
                    const boundary_key key = { 0, boundary };
                    const auto comparison = compare_captured_boundary(
                            scalar_internal.capture, batch_internal.capture, key, prediction_start);
                    if (first_internal_boundary.empty() && comparison.mismatch_count != 0) {
                        first_internal_boundary = boundary;
                        first_internal_prediction = comparison.first_mismatch_prediction;
                    }
                    const auto scalar_shape = scalar_internal.capture.internal_shapes.at(key);
                    const auto batch_shape = batch_internal.capture.internal_shapes.at(key);
                    std::printf(
                            "{\"type\":\"recurrent_internal_boundary\",\"layer\":0,"
                            "\"boundary\":\"%s\",\"source_sequence_index\":%zu,"
                            "\"rows\":8,\"row_bytes\":%zu,"
                            "\"scalar_shape\":[%lld,%lld,%lld,%lld],"
                            "\"batch_shape\":[%lld,%lld,%lld,%lld],"
                            "\"first_mismatch_prediction\":%d,\"mismatch_count\":%u,"
                            "\"scalar_window_hash\":\"%s\",\"batch_window_hash\":\"%s\","
                            "\"finite\":true,\"exact_token_slice_comparison\":true,"
                            "\"repeat_stable\":true}\n",
                            boundary, boundary_index, scalar_internal.capture.row_bytes.at(key),
                            (long long) scalar_shape[0], (long long) scalar_shape[1],
                            (long long) scalar_shape[2], (long long) scalar_shape[3],
                            (long long) batch_shape[0], (long long) batch_shape[1],
                            (long long) batch_shape[2], (long long) batch_shape[3],
                            comparison.first_mismatch_prediction, comparison.mismatch_count,
                            hex64(comparison.scalar_window_hash).c_str(),
                            hex64(comparison.batch_window_hash).c_str());
                }
                if (first_internal_boundary.empty()) {
                    throw std::runtime_error("recurrent-internal capture did not locate the known attention-output mismatch");
                }

                static constexpr std::array<const char *, 2> cache_boundaries = {{
                    "conv_states", "state_predelta",
                }};
                if (scalar_internal.capture.internal_cache_inputs.size() != cache_boundaries.size() ||
                        batch_internal.capture.internal_cache_inputs.size() != cache_boundaries.size()) {
                    throw std::runtime_error("recurrent-internal capture did not emit both cache inputs");
                }
                for (const char * boundary : cache_boundaries) {
                    const auto & scalar_hashes = scalar_internal.capture.internal_cache_inputs.at(boundary);
                    const auto & batch_hashes = batch_internal.capture.internal_cache_inputs.at(boundary);
                    const auto & scalar_raw = scalar_internal.capture.internal_cache_data.at(boundary);
                    const auto & batch_raw = batch_internal.capture.internal_cache_data.at(boundary);
                    if (scalar_hashes.size() != 8 || batch_hashes.size() != 1 ||
                            scalar_raw.size() != 8 || batch_raw.size() != 1 ||
                            scalar_raw[0] != batch_raw[0] || scalar_hashes[0] != batch_hashes[0] ||
                            scalar_internal.capture.internal_cache_bytes.at(boundary) !=
                                batch_internal.capture.internal_cache_bytes.at(boundary)) {
                        throw std::runtime_error(std::string("recurrent-internal cache prefix mismatch at ") + boundary);
                    }
                    const auto scalar_shape = scalar_internal.capture.internal_cache_shapes.at(boundary);
                    const auto batch_shape = batch_internal.capture.internal_cache_shapes.at(boundary);
                    std::printf(
                            "{\"type\":\"recurrent_internal_cache\",\"layer\":0,"
                            "\"boundary\":\"%s\",\"scope\":\"cache_input_prefix_and_scalar_evolution\","
                            "\"scalar_steps\":8,\"batch_prefix_states\":1,"
                            "\"state_bytes\":%zu,\"prefix_exact_match\":true,"
                            "\"scalar_shape\":[%lld,%lld,%lld,%lld],"
                            "\"batch_shape\":[%lld,%lld,%lld,%lld],"
                            "\"scalar_evolution_hash\":\"%s\",\"batch_prefix_hash\":\"%s\","
                            "\"finite\":true,\"repeat_stable\":true}\n",
                            boundary, scalar_internal.capture.internal_cache_bytes.at(boundary),
                            (long long) scalar_shape[0], (long long) scalar_shape[1],
                            (long long) scalar_shape[2], (long long) scalar_shape[3],
                            (long long) batch_shape[0], (long long) batch_shape[1],
                            (long long) batch_shape[2], (long long) batch_shape[3],
                            hex64(hash_u64_sequence(scalar_hashes)).c_str(),
                            hex64(batch_hashes[0]).c_str());
                }
                std::printf(
                        "{\"type\":\"observer_invariance\",\"stage\":\"recurrent_internal\","
                        "\"scalar_top2_match_unobserved\":true,\"batch_top2_match_unobserved\":true,"
                        "\"scalar_partial_state_digest_match_unobserved\":true,"
                        "\"batch_partial_state_digest_match_unobserved\":true,\"repeat_stable\":true}\n");
                std::printf(
                        "{\"type\":\"recurrent_internal_summary\",\"layer\":0,\"rows\":8,"
                        "\"boundaries\":14,\"cache_boundaries\":2,"
                        "\"first_bad_observed_boundary\":\"%s\",\"first_bad_prediction\":%d,"
                        "\"boundary_order_semantic_not_causal\":true,"
                        "\"scalar_gdn\":%llu,\"scalar_gdn_fused_cache\":%llu,"
                        "\"batch_gdn\":%llu,\"batch_gdn_fused_cache\":%llu,"
                        "\"one_context_at_a_time\":true,\"content_free\":true,"
                        "\"observer_top2_and_state_digest_invariant\":true}\n",
                        first_internal_boundary.c_str(), first_internal_prediction,
                        (unsigned long long) scalar_routes.count[CUDA_ROUTE_GDN],
                        (unsigned long long) scalar_routes.count[CUDA_ROUTE_GDN_FUSED_CACHE],
                        (unsigned long long) batch_routes.count[CUDA_ROUTE_GDN],
                        (unsigned long long) batch_routes.count[CUDA_ROUTE_GDN_FUSED_CACHE]);
            }

            if (full_attention_internal_refine) {
                if (first_bad_layer != 3 || recurrent_layer_map[3] != 0 ||
                        first_boundary != "attention_output" || first_boundary_prediction != prediction_start) {
                    throw std::runtime_error(
                            "full-attention internal refinement requires the reviewed layer-3 attention-output mismatch");
                }

                cuda_route_counts scalar_routes{};
                const boundary_arm_result scalar_internal = run_boundary_arm(
                        model.get(), params, prompt_token, prompt_count, reference,
                        prediction_start, 1, n_vocab, boundary_capture::stage::full_attention_internal, 3, false,
                        full_attention_route_api, &scalar_routes, true, true);
                cuda_route_counts batch_routes{};
                const boundary_arm_result batch_internal = run_boundary_arm(
                        model.get(), params, prompt_token, prompt_count, reference,
                        prediction_start, 8, n_vocab, boundary_capture::stage::full_attention_internal, 3, false,
                        full_attention_route_api, &batch_routes, true, true);

                if (!same_cuda_routes(scalar_baseline_routes, scalar_routes) ||
                        !same_cuda_routes(batch_baseline_routes, batch_routes)) {
                    throw std::runtime_error(
                            "full-attention callbacks changed the callback-free CUDA route snapshot");
                }
                if (scalar_routes.count[CUDA_ROUTE_FATTN_VEC] != 128 ||
                        scalar_routes.count[CUDA_ROUTE_FATTN_MMA_F16] != 0 ||
                        scalar_routes.count[CUDA_ROUTE_FATTN_TILE] != 0 ||
                        batch_routes.count[CUDA_ROUTE_FATTN_VEC] != 0 ||
                        batch_routes.count[CUDA_ROUTE_FATTN_MMA_F16] != 16 ||
                        batch_routes.count[CUDA_ROUTE_FATTN_TILE] != 0 ||
                        scalar_routes.count[CUDA_ROUTE_SIGMOID] != 0 ||
                        scalar_routes.count[CUDA_ROUTE_SIGMOID_MUL] != 128 ||
                        batch_routes.count[CUDA_ROUTE_SIGMOID] != 0 ||
                        batch_routes.count[CUDA_ROUTE_SIGMOID_MUL] != 16 ||
                        scalar_routes.count[CUDA_ROUTE_SET_ROWS] != 256 ||
                        batch_routes.count[CUDA_ROUTE_SET_ROWS] != 32 ||
                        scalar_routes.count[CUDA_ROUTE_ROPE_VIEW_SET_ROWS] != 0 ||
                        batch_routes.count[CUDA_ROUTE_ROPE_VIEW_SET_ROWS] != 0) {
                    throw std::runtime_error(
                            "full-attention refinement did not preserve exact FA, sigmoid-mul, and Q8 cache-write routes");
                }
                if (!same_width_observations_full(scalar_baseline, scalar_internal) ||
                        !same_width_observations_full(batch_baseline, batch_internal)) {
                    throw std::runtime_error(
                            "full-attention callbacks changed top2, recurrent state, or complete sequence state");
                }

                // This order is a source-reviewed semantic sequence, not a claim
                // that parallel Q/K/V/gate branches are totally ordered.
                static constexpr std::array<const char *, 14> internal_boundaries = {{
                    "attn_norm", "qg_projection", "q_pre_norm", "q_post_norm",
                    "k_projection", "k_post_norm", "v_projection", "gate_pre_sigmoid",
                    "q_post_norm_rope", "k_post_norm_rope", "v_reshaped",
                    "kv_fa_output_pregate", "attn_gated", "output_projection",
                }};
                if (scalar_internal.capture.rows.size() != internal_boundaries.size() ||
                        batch_internal.capture.rows.size() != internal_boundaries.size() ||
                        scalar_internal.capture.internal_shapes.size() != internal_boundaries.size() ||
                        batch_internal.capture.internal_shapes.size() != internal_boundaries.size() ||
                        !scalar_internal.capture.internal_cache_inputs.empty() ||
                        !batch_internal.capture.internal_cache_inputs.empty()) {
                    throw std::runtime_error(
                            "full-attention internal capture did not emit the exact reviewed activation set");
                }

                std::string first_internal_boundary;
                int32_t first_internal_prediction = -1;
                for (size_t boundary_index = 0; boundary_index < internal_boundaries.size(); ++boundary_index) {
                    const char * boundary = internal_boundaries[boundary_index];
                    const boundary_key key = { 3, boundary };
                    const auto comparison = compare_captured_boundary(
                            scalar_internal.capture, batch_internal.capture, key, prediction_start);
                    if (first_internal_boundary.empty() && comparison.mismatch_count != 0) {
                        first_internal_boundary = boundary;
                        first_internal_prediction = comparison.first_mismatch_prediction;
                    }
                    const auto scalar_shape = scalar_internal.capture.internal_shapes.at(key);
                    const auto batch_shape = batch_internal.capture.internal_shapes.at(key);
                    std::printf(
                            "{\"type\":\"full_attention_internal_boundary\",\"layer\":3,"
                            "\"boundary\":\"%s\",\"source_sequence_index\":%zu,"
                            "\"rows\":8,\"row_bytes\":%zu,"
                            "\"scalar_shape\":[%lld,%lld,%lld,%lld],"
                            "\"batch_shape\":[%lld,%lld,%lld,%lld],"
                            "\"first_mismatch_prediction\":%d,\"mismatch_count\":%u,"
                            "\"scalar_window_hash\":\"%s\",\"batch_window_hash\":\"%s\","
                            "\"finite\":true,\"exact_token_slice_comparison\":true,"
                            "\"repeat_stable\":true}\n",
                            boundary, boundary_index, scalar_internal.capture.row_bytes.at(key),
                            (long long) scalar_shape[0], (long long) scalar_shape[1],
                            (long long) scalar_shape[2], (long long) scalar_shape[3],
                            (long long) batch_shape[0], (long long) batch_shape[1],
                            (long long) batch_shape[2], (long long) batch_shape[3],
                            comparison.first_mismatch_prediction, comparison.mismatch_count,
                            hex64(comparison.scalar_window_hash).c_str(),
                            hex64(comparison.batch_window_hash).c_str());
                }
                if (first_internal_boundary.empty()) {
                    throw std::runtime_error(
                            "full-attention internal capture did not locate the known attention-output mismatch");
                }

                std::printf(
                        "{\"type\":\"observer_invariance\",\"stage\":\"full_attention_internal\","
                        "\"scalar_top2_match_unobserved\":true,\"batch_top2_match_unobserved\":true,"
                        "\"scalar_partial_state_digest_match_unobserved\":true,"
                        "\"batch_partial_state_digest_match_unobserved\":true,"
                        "\"scalar_full_state_digest_match_unobserved\":true,"
                        "\"batch_full_state_digest_match_unobserved\":true,"
                        "\"route_snapshot_matches_unobserved\":true,\"repeat_stable\":true}\n");
                std::printf(
                        "{\"type\":\"full_attention_internal_summary\",\"layer\":3,\"rows\":8,"
                        "\"boundaries\":14,\"first_bad_observed_boundary\":\"%s\","
                        "\"first_bad_prediction\":%d,\"boundary_order_semantic_not_causal\":true,"
                        "\"scalar_fattn_vec\":%llu,\"scalar_fattn_mma_f16\":%llu,"
                        "\"scalar_fattn_tile\":%llu,\"batch_fattn_vec\":%llu,"
                        "\"batch_fattn_mma_f16\":%llu,\"batch_fattn_tile\":%llu,"
                        "\"scalar_sigmoid\":%llu,\"scalar_sigmoid_mul\":%llu,"
                        "\"batch_sigmoid\":%llu,\"batch_sigmoid_mul\":%llu,"
                        "\"scalar_set_rows\":%llu,\"batch_set_rows\":%llu,"
                        "\"scalar_rope_view_set_rows\":%llu,\"batch_rope_view_set_rows\":%llu,"
                        "\"route_snapshot_matches_unobserved\":true,"
                        "\"full_state_digest_matches_unobserved\":true,"
                        "\"one_context_at_a_time\":true,\"content_free\":true}\n",
                        first_internal_boundary.c_str(), first_internal_prediction,
                        (unsigned long long) scalar_routes.count[CUDA_ROUTE_FATTN_VEC],
                        (unsigned long long) scalar_routes.count[CUDA_ROUTE_FATTN_MMA_F16],
                        (unsigned long long) scalar_routes.count[CUDA_ROUTE_FATTN_TILE],
                        (unsigned long long) batch_routes.count[CUDA_ROUTE_FATTN_VEC],
                        (unsigned long long) batch_routes.count[CUDA_ROUTE_FATTN_MMA_F16],
                        (unsigned long long) batch_routes.count[CUDA_ROUTE_FATTN_TILE],
                        (unsigned long long) scalar_routes.count[CUDA_ROUTE_SIGMOID],
                        (unsigned long long) scalar_routes.count[CUDA_ROUTE_SIGMOID_MUL],
                        (unsigned long long) batch_routes.count[CUDA_ROUTE_SIGMOID],
                        (unsigned long long) batch_routes.count[CUDA_ROUTE_SIGMOID_MUL],
                        (unsigned long long) scalar_routes.count[CUDA_ROUTE_SET_ROWS],
                        (unsigned long long) batch_routes.count[CUDA_ROUTE_SET_ROWS],
                        (unsigned long long) scalar_routes.count[CUDA_ROUTE_ROPE_VIEW_SET_ROWS],
                        (unsigned long long) batch_routes.count[CUDA_ROUTE_ROPE_VIEW_SET_ROWS]);
            }

            std::vector<uint64_t> scalar_state_hashes;
            for (const auto & state : scalar_coarse.scalar_states) {
                scalar_state_hashes.push_back(state.hash);
            }
            std::printf(
                    "{\"type\":\"recurrent_partial_state\",\"scope\":\"recurrent_only\","
                    "\"prefix_hash\":\"%s\",\"prefix_bytes\":%zu,"
                    "\"scalar_evolution_hash\":\"%s\",\"scalar_final_hash\":\"%s\","
                    "\"batch_final_hash\":\"%s\",\"final_bytes\":%zu,"
                    "\"prefix_match\":%s,\"final_match\":%s,\"repeat_stable\":true}\n",
                    hex64(scalar_coarse.prefix_state.hash).c_str(), scalar_coarse.prefix_state.bytes,
                    hex64(hash_u64_sequence(scalar_state_hashes)).c_str(),
                    hex64(scalar_coarse.final_state.hash).c_str(), hex64(batch_coarse.final_state.hash).c_str(),
                    scalar_coarse.final_state.bytes,
                    same_state(scalar_coarse.prefix_state, batch_coarse.prefix_state) ? "true" : "false",
                    same_state(scalar_coarse.final_state, batch_coarse.final_state) ? "true" : "false");

            std::printf(
                    "{\"type\":\"layer_boundary_summary\",\"window_start\":%d,\"window_width\":8,"
                    "\"layers\":64,\"recurrent_layers\":%u,\"full_attention_layers\":%u,"
                    "\"recurrent_layer_map_hash\":\"%s\",\"recurrent_layer_map_source\":\"%s\","
                    "\"layer_type_map_attested\":true,"
                    "\"first_bad_layer\":%d,\"first_bad_prediction\":%d,"
                    "\"refine_executed\":%s,\"first_bad_boundary\":\"%s\","
                    "\"first_boundary_mismatch_prediction\":%d,\"one_context_at_a_time\":true,"
                    "\"content_free\":true,\"observer_top2_and_state_digest_invariant\":true}\n",
                    prediction_start, recurrent_layer_count, full_attention_layer_count,
                    hex64(recurrent_layer_map_hash).c_str(), recurrent_layer_map_source.c_str(),
                    first_bad_layer, first_bad_prediction,
                    refine_executed ? "true" : "false", first_boundary.c_str(), first_boundary_prediction);
            std::printf("{\"type\":\"complete\",\"operational_success\":true,\"exact_parity_gate_bypassed\":false}\n");
            model.reset();
            llama_backend_free();
            backend_initialized = false;
            return 0;
        }

        std::map<uint32_t, width_result> widths;
        const cuda_route_api fattn_candidate_route_api = fattn_vec_candidate ?
                require_cuda_route_api() : cuda_route_api {};
        const std::vector<uint32_t> diagnostic_widths = fattn_vec_candidate ?
                std::vector<uint32_t>{ 1, 2, 3, 4, 5, 6, 7, 8 } :
                std::vector<uint32_t>{ 1, 2, 4, 8 };
        std::vector<uint8_t> fattn_route_prefix_state;
        cuda_route_counts fattn_scalar_prefix_routes{};
        cuda_fattn_candidate_observation fattn_scalar_prefix_predicate{};
        if (fattn_vec_candidate) {
            // The prompt graph has a different query width. Reset immediately
            // after it, then snapshot the first scalar decode before any scalar
            // graph capture/replay can hide the unchanged M1 VEC route.
            auto prefix_producer = make_context(model.get(), params, 1);
            if (!prefix_producer) {
                throw std::runtime_error("failed to create full-attention route-prefix producer");
            }
            llama_tokens prompt((size_t) prompt_count, prompt_token);
            decode_rows(prefix_producer->get(), prompt, 0, 0, false);
            fattn_candidate_route_api.reset();
            decode_rows(prefix_producer->get(), { reference[0] }, prompt_count, 0, true);
            fattn_scalar_prefix_routes = require_cuda_route_snapshot(
                    fattn_candidate_route_api.snapshot);
            fattn_scalar_prefix_predicate = require_cuda_fattn_candidate_snapshot(
                    fattn_candidate_route_api.fattn_candidate_snapshot);
            for (int32_t token_index = 1; token_index <= prediction_start - 2; ++token_index) {
                decode_rows(prefix_producer->get(), { reference[(size_t) token_index] },
                        prompt_count + token_index, 0, true);
            }
            fattn_route_prefix_state = sequence_state(
                    prefix_producer->get(), LLAMA_STATE_SEQ_FLAGS_NONE);
        }
        for (uint32_t width : diagnostic_widths) {
            cuda_route_counts routes{};
            cuda_fattn_candidate_observation predicate_observation{};
            if (fattn_vec_candidate) {
                if (width == 1) {
                    routes = fattn_scalar_prefix_routes;
                    predicate_observation = fattn_scalar_prefix_predicate;
                } else {
                    routes = run_width_route_probe_from_restored_prefix(
                            model.get(), params, fattn_route_prefix_state, reference,
                            prompt_count, prediction_start, width, fattn_candidate_route_api,
                            predicate_observation);
                }
            }
            auto result = run_width(model.get(), params, prompt_token, prompt_count, reference,
                    prediction_start, prediction_count, width, n_vocab, fattn_vec_candidate);
            const auto repeat = run_width(model.get(), params, prompt_token, prompt_count, reference,
                    prediction_start, prediction_count, width, n_vocab, fattn_vec_candidate);
            if (!same_width_result(result, repeat)) {
                throw std::runtime_error("width-" + std::to_string(width) + " repeat is not bitwise stable");
            }
            if (fattn_vec_candidate) {
                print_fattn_candidate_predicate(width, predicate_observation);
                const uint64_t selected = routes.count[CUDA_ROUTE_FATTN_QWEN35_D256_Q8_GQA6_VEC];
                const uint64_t hinted = routes.count[CUDA_ROUTE_FATTN_QWEN35_D256_Q8_GQA6_HINT];
                const bool candidate_width = width >= 3 && width <= 8;
                std::printf(
                        "{\"type\":\"fattn_graph_hint_route\",\"width\":%u,"
                        "\"hint_seen\":%llu,\"hint_expected\":%s,"
                        "\"first_single_batch_probe\":true,\"content_free\":true}\n",
                        width, (unsigned long long) hinted, candidate_width ? "true" : "false");
                std::printf(
                        "{\"type\":\"fattn_candidate_route\",\"width\":%u,"
                        "\"candidate_vec\":%llu,\"fattn_vec\":%llu,"
                        "\"fattn_mma_f16\":%llu,\"fattn_tile\":%llu,"
                        "\"candidate_selected\":%s,\"first_single_batch_probe\":true,"
                        "\"shape_guard_attested\":true,\"content_free\":true}\n",
                        width, (unsigned long long) selected,
                        (unsigned long long) routes.count[CUDA_ROUTE_FATTN_VEC],
                        (unsigned long long) routes.count[CUDA_ROUTE_FATTN_MMA_F16],
                        (unsigned long long) routes.count[CUDA_ROUTE_FATTN_TILE],
                        selected > 0 ? "true" : "false");
                std::fflush(stdout);
                if ((candidate_width && (hinted != 16 || selected != 16 ||
                            routes.count[CUDA_ROUTE_FATTN_VEC] != 16 ||
                            routes.count[CUDA_ROUTE_FATTN_MMA_F16] != 0 ||
                            routes.count[CUDA_ROUTE_FATTN_TILE] != 0)) ||
                        (!candidate_width && (hinted != 0 || selected != 0 ||
                            routes.count[CUDA_ROUTE_FATTN_VEC] != 16 ||
                            routes.count[CUDA_ROUTE_FATTN_MMA_F16] != 0 ||
                            routes.count[CUDA_ROUTE_FATTN_TILE] != 0))) {
                    throw std::runtime_error(
                            "Qwen3.5 full-attention first-batch route probe failed for width " +
                            std::to_string(width));
                }
            }
            for (const auto & observation : result.predictions) {
                print_top2_json("width", width, observation);
            }
            const state_digest & reported_state = fattn_vec_candidate ? result.full_final_state : result.final_state;
            std::printf(
                    "{\"type\":\"width_state\",\"width\":%u,\"consumed_token_index\":%d,"
                    "\"prefix_state_hash\":\"%s\",\"prefix_state_bytes\":%zu,"
                    "\"state_hash\":\"%s\",\"state_bytes\":%zu,\"state_scope\":\"%s\",\"repeat_stable\":true}\n",
                    width, result.final_consumed_token_index,
                    hex64(result.prefix_state.hash).c_str(), result.prefix_state.bytes,
                    hex64(reported_state.hash).c_str(), reported_state.bytes,
                    fattn_vec_candidate ? "full" : "partial");
            widths.emplace(width, std::move(result));
        }

        const auto & scalar = widths.at(1);
        // Raw logits intentionally remain unmodified for the numerical width
        // locator. Reference provenance was already checked above with the
        // exact production sampler, including ignore_eos logit biases.
        bool fattn_exact_parity = true;
        const std::vector<uint32_t> comparison_widths = fattn_vec_candidate ?
                std::vector<uint32_t>{ 2, 3, 4, 5, 6, 7, 8 } :
                std::vector<uint32_t>{ 2, 4, 8 };
        for (uint32_t width : comparison_widths) {
            const auto & candidate = widths.at(width);
            int32_t first_top1_difference = -1;
            int32_t first_top2_difference = -1;
            int32_t first_top2_value_difference = -1;
            for (size_t i = 0; i < scalar.predictions.size(); ++i) {
                if (first_top1_difference < 0 && scalar.predictions[i].top2.top1 != candidate.predictions[i].top2.top1) {
                    first_top1_difference = scalar.predictions[i].prediction_index;
                }
                if (first_top2_difference < 0 &&
                        (scalar.predictions[i].top2.top1 != candidate.predictions[i].top2.top1 ||
                         scalar.predictions[i].top2.top2 != candidate.predictions[i].top2.top2)) {
                    first_top2_difference = scalar.predictions[i].prediction_index;
                }
                if (first_top2_value_difference < 0 &&
                        (float_bits(scalar.predictions[i].top2.logit1) != float_bits(candidate.predictions[i].top2.logit1) ||
                         float_bits(scalar.predictions[i].top2.logit2) != float_bits(candidate.predictions[i].top2.logit2))) {
                    first_top2_value_difference = scalar.predictions[i].prediction_index;
                }
            }
            const bool prefix_state_match = scalar.prefix_state.hash == candidate.prefix_state.hash &&
                    scalar.prefix_state.bytes == candidate.prefix_state.bytes;
            const state_digest & scalar_comparison_state = fattn_vec_candidate ? scalar.full_final_state : scalar.final_state;
            const state_digest & candidate_comparison_state = fattn_vec_candidate ? candidate.full_final_state : candidate.final_state;
            const bool final_state_match = same_state(scalar_comparison_state, candidate_comparison_state);
            fattn_exact_parity &= first_top1_difference < 0 && first_top2_difference < 0 &&
                    first_top2_value_difference < 0 && prefix_state_match && final_state_match;
            std::printf(
                    "{\"type\":\"width_summary\",\"width\":%u,\"first_top1_difference\":%d,"
                    "\"first_top2_difference\":%d,\"first_top2_value_difference\":%d,"
                    "\"prefix_state_match_scalar\":%s,\"final_state_match_scalar\":%s,\"state_scope\":\"%s\"}\n",
                    width, first_top1_difference, first_top2_difference, first_top2_value_difference,
                    prefix_state_match ? "true" : "false", final_state_match ? "true" : "false",
                    fattn_vec_candidate ? "full" : "partial");
        }

        const llama_state_seq_flags comparison_state_flags = fattn_vec_candidate ?
                LLAMA_STATE_SEQ_FLAGS_NONE : LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
        const auto oracle = build_scalar_oracle(model.get(), params, prompt_token, prompt_count,
                reference, rollback_start, 8, n_vocab, comparison_state_flags);
        const bool rollback_exact_parity = run_rollbacks(model.get(), params, prompt_token, prompt_count, reference,
                rollback_start, 8, n_vocab, oracle, comparison_state_flags);

        if (fattn_vec_candidate && (!fattn_exact_parity || !rollback_exact_parity)) {
            throw std::runtime_error("Qwen3.5 full-attention V3 exact parity promotion gate failed");
        }

        std::printf("{\"type\":\"complete\",\"operational_success\":true,\"exact_parity_gate_bypassed\":false}\n");
        model.reset();
        llama_backend_free();
        backend_initialized = false;
        return 0;
    } catch (const std::exception & error) {
        const std::string escaped = json_escape(error.what());
        std::fprintf(stderr,
                "{\"type\":\"error\",\"stage\":\"%s\",\"message\":\"%s\"}\n",
                failure_stage, escaped.c_str());
        std::fflush(stderr);
        if (backend_initialized) {
            llama_backend_free();
        }
        return 2;
    }
}
