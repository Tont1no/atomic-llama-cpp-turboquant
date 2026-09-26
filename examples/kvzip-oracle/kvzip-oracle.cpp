#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "nlohmann/json.hpp"
extern "C" {
#include "sha256/sha256.h"
}

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// KVzap training oracle: one full chat sequence, source keys and copied-assistant queries.
// The output is private raw arrays; oracle_scores require the separate effective W_O export.
namespace fs = std::filesystem;
using json = nlohmann::json;

static std::string sha256_file(const fs::path & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot hash private input file");
    sha256_t hash;
    sha256_init(&hash);
    char block[65536];
    while (in.read(block, sizeof(block)) || in.gcount()) {
        sha256_update(&hash, reinterpret_cast<const unsigned char *>(block), static_cast<size_t>(in.gcount()));
    }
    if (!in.eof()) throw std::runtime_error("private input hash read failed");
    unsigned char digest[SHA256_DIGEST_SIZE];
    sha256_final(&hash, digest);
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (auto byte : digest) out << std::setw(2) << static_cast<int>(byte);
    return out.str();
}

static std::vector<llama_token> read_ids(const fs::path & path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot read private token file");
    std::vector<llama_token> ids;
    int64_t value;
    while (in >> value) {
        if (value < 0 || value > std::numeric_limits<llama_token>::max()) throw std::runtime_error("invalid token ID");
        ids.push_back(static_cast<llama_token>(value));
    }
    if (!in.eof()) throw std::runtime_error("invalid private token file");
    return ids;
}

static float f16_to_f32(uint16_t bits) {
    uint32_t sign = (uint32_t(bits) & 0x8000u) << 16;
    uint32_t exp = (bits >> 10) & 31u;
    uint32_t mant = bits & 1023u;
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) out = sign;
        else {
            int e = -14;
            while ((mant & 1024u) == 0) { mant <<= 1; --e; }
            out = sign | (uint32_t(e + 127) << 23) | ((mant & 1023u) << 13);
        }
    } else if (exp == 31) out = sign | 0x7f800000u | (mant << 13);
    else out = sign | ((exp - 15u + 127u) << 23) | (mant << 13);
    float value;
    std::memcpy(&value, &out, sizeof(value));
    return value;
}

static float get_f32(const ggml_tensor * t, const std::vector<uint8_t> & bytes, int64_t i, int64_t j = 0, int64_t k = 0) {
    const size_t offset = size_t(i) * t->nb[0] + size_t(j) * t->nb[1] + size_t(k) * t->nb[2];
    if (t->type == GGML_TYPE_F32) {
        if (offset + 4 > bytes.size()) throw std::runtime_error("tensor stride exceeds buffer");
        float value;
        std::memcpy(&value, bytes.data() + offset, 4);
        return value;
    }
    if (t->type == GGML_TYPE_F16) {
        if (offset + 2 > bytes.size()) throw std::runtime_error("tensor stride exceeds buffer");
        uint16_t bits;
        std::memcpy(&bits, bytes.data() + offset, 2);
        return f16_to_f32(bits);
    }
    throw std::runtime_error("oracle tensor must be F16 or F32");
}

struct layer_data {
    std::ofstream features;
    std::ofstream values;
    std::ofstream probe_gated;
    std::ofstream probe_output;
    std::vector<float> max_attention; // [source_position, query_head]
    std::vector<float> query_norm;
    int64_t feature_rows = 0;
    int64_t value_rows = 0;
    int64_t replay_rows = 0;
    int64_t probe_gated_rows = 0;
    int64_t probe_output_rows = 0;
};

struct capture {
    enum phase_t { ignore, source, replay } phase = ignore;
    std::vector<layer_data> layers;
    int64_t source_len = 0;
    int64_t source_start = 0;
    int64_t source_end = 0;
    int64_t repeat_start = 0;
    int64_t decode_pos = 0;
    int64_t decode_len = 0;
    int64_t replay_rows = 0;
    int32_t n_head = 0;
    int32_t n_kv_head = 0;
    int32_t head_dim = 256;
    int32_t hidden = 5120;
    std::string error;

    bool wants(const ggml_tensor * t) const {
        const std::string name(t->name);
        const size_t dash = name.rfind('-');
        if (dash == std::string::npos) return false;
        const std::string layer = name.substr(dash + 1);
        if (layer.empty() || layer.find_first_not_of("0123456789") != std::string::npos) return false;
        const int il = std::stoi(layer);
        if (il < 0 || il >= static_cast<int>(layers.size()) || (il + 1) % 4 != 0) return false;
        if (phase == source && (name.rfind("attn_gated-", 0) == 0 || name.rfind("attn_output-", 0) == 0)) {
            return decode_pos < source_start + 4 || decode_pos + decode_len > source_end - 4;
        }
        if (phase == source) return name.rfind("attn_norm-", 0) == 0 || name.rfind("Vcur-", 0) == 0;
        if (phase == replay) return name.rfind("attn_norm-", 0) == 0 || name.rfind("kq_soft_max-", 0) == 0;
        return false;
    }

    void collect(const ggml_tensor * t) {
        const std::string name(t->name);
        const size_t dash = name.rfind('-');
        if (dash == std::string::npos) throw std::runtime_error("oracle callback name has no layer");
        const int il = std::stoi(name.substr(dash + 1));
        if (il < 0 || il >= static_cast<int>(layers.size()) || (il + 1) % 4 != 0) {
            throw std::runtime_error("unexpected oracle attention layer");
        }
        if (ggml_nbytes(t) > 64u * 1024u * 1024u) throw std::runtime_error("oracle callback tensor exceeds 64 MiB");
        std::vector<uint8_t> bytes(ggml_nbytes(t));
        ggml_backend_tensor_get(t, bytes.data(), 0, bytes.size());
        auto & layer = layers[il];
        if (name.rfind("attn_norm-", 0) == 0) {
            if (t->ne[0] != hidden || t->ne[1] != decode_len || t->ne[2] != 1 || t->ne[3] != 1) {
                throw std::runtime_error("unexpected attn_norm shape");
            }
            if (phase == source) {
                for (int64_t j = 0; j < decode_len; ++j) {
                    for (int32_t d = 0; d < hidden; ++d) {
                        const float x = get_f32(t, bytes, d, j);
                        if (!std::isfinite(x)) throw std::runtime_error("nonfinite source feature");
                        layer.features.write(reinterpret_cast<const char *>(&x), sizeof(x));
                    }
                }
                layer.feature_rows += decode_len;
            } else {
                layer.query_norm.resize(static_cast<size_t>(decode_len));
                for (int64_t j = 0; j < decode_len; ++j) {
                    double squared = 0;
                    for (int32_t d = 0; d < hidden; ++d) {
                        const float x = get_f32(t, bytes, d, j);
                        squared += double(x) * x;
                    }
                    const float norm = static_cast<float>(std::sqrt(squared));
                    if (!std::isfinite(norm) || norm <= 0) throw std::runtime_error("invalid replay hidden norm");
                    layer.query_norm[static_cast<size_t>(j)] = norm;
                }
            }
        } else if (name.rfind("Vcur-", 0) == 0) {
            // The first Vcur callback is flat; capture only the final [head_dim,Hkv,N] view.
            if (t->ne[0] != head_dim || t->ne[1] != n_kv_head || t->ne[2] != decode_len || t->ne[3] != 1) return;
            for (int64_t j = 0; j < decode_len; ++j) {
                for (int32_t h = 0; h < n_kv_head; ++h) {
                    for (int32_t d = 0; d < head_dim; ++d) {
                        const float v = get_f32(t, bytes, d, h, j);
                        if (!std::isfinite(v)) throw std::runtime_error("nonfinite source value");
                        layer.values.write(reinterpret_cast<const char *>(&v), sizeof(v));
                    }
                }
            }
            layer.value_rows += decode_len;
        } else if (name.rfind("attn_gated-", 0) == 0 || name.rfind("attn_output-", 0) == 0) {
            const bool gated = name.rfind("attn_gated-", 0) == 0;
            const int32_t width = gated ? n_head * head_dim : hidden;
            if (phase != source || t->ne[0] != width || t->ne[1] != decode_len || t->ne[2] != 1 || t->ne[3] != 1) {
                throw std::runtime_error("unexpected W_O probe shape");
            }
            auto & out = gated ? layer.probe_gated : layer.probe_output;
            auto & rows = gated ? layer.probe_gated_rows : layer.probe_output_rows;
            for (int64_t j = 0; j < decode_len; ++j) {
                const int64_t pos = decode_pos + j;
                if (pos >= source_start + 4 && pos < source_end - 4) continue;
                for (int32_t d = 0; d < width; ++d) {
                    const float value = get_f32(t, bytes, d, j);
                    if (!std::isfinite(value)) throw std::runtime_error("nonfinite W_O probe");
                    out.write(reinterpret_cast<const char *>(&value), 4);
                }
                ++rows;
            }
        } else if (name.rfind("kq_soft_max-", 0) == 0) {
            if (phase != replay || t->ne[1] != decode_len || t->ne[2] != n_head || t->ne[3] != 1 ||
                t->ne[0] <= decode_pos + decode_len - 1 || layer.query_norm.size() != static_cast<size_t>(decode_len)) {
                throw std::runtime_error("unexpected replay softmax shape or missing norm");
            }
            for (int64_t j = 0; j < decode_len; ++j) {
                for (int32_t h = 0; h < n_head; ++h) {
                    for (int64_t key = source_start; key < source_end; ++key) {
                        const float probability = get_f32(t, bytes, key, j, h);
                        if (!std::isfinite(probability) || probability < 0) throw std::runtime_error("invalid replay attention");
                        const float score = probability / layer.query_norm[static_cast<size_t>(j)];
                        auto & maximum = layer.max_attention[static_cast<size_t>((key - source_start) * n_head + h)];
                        maximum = std::max(maximum, score);
                    }
                }
            }
            layer.replay_rows += decode_len;
        }
    }
};

static bool callback(ggml_tensor * t, bool ask, void * user_data) {
    auto & state = *static_cast<capture *>(user_data);
    if (!state.error.empty()) return false;
    if (ask) return state.wants(t);
    try {
        state.collect(t);
        return true;
    } catch (const std::exception & e) {
        state.error = e.what();
        return false;
    }
}

static void decode(llama_context * ctx, capture & state, const std::vector<llama_token> & ids, int64_t start, int64_t end, int32_t batch) {
    for (int64_t pos = start; pos < end; pos += batch) {
        const int64_t count = std::min<int64_t>(batch, end - pos);
        state.decode_pos = pos;
        state.decode_len = count;
        auto * tokens = const_cast<llama_token *>(ids.data() + pos);
        if (llama_decode(ctx, llama_batch_get_one(tokens, static_cast<int32_t>(count))) != 0 || !state.error.empty()) {
            throw std::runtime_error(state.error.empty() ? "llama_decode failed" : state.error);
        }
    }
}

struct capture_request {
    fs::path ids_path;
    fs::path protected_path;
    fs::path output;
    int64_t source_start;
    int64_t source_end;
    int64_t repeat_start;
    int64_t repeat_end;
    std::string template_sha256;
    std::string expected_ids_sha256;
};

static bool lowercase_sha256(const std::string & digest) {
    return digest.size() == 64 && digest.find_first_not_of("0123456789abcdef") == std::string::npos;
}

static fs::path checked_private(const fs::path & path, const fs::path & repo, const std::string & target_sha256) {
    const fs::path resolved = fs::weakly_canonical(path);
    const fs::path local = fs::weakly_canonical(repo / "tmp" / "bonsai-training-20260924");
    if (resolved != local && !resolved.lexically_relative(local).empty() &&
        *resolved.lexically_relative(local).begin() != "..") return resolved;
    const fs::path secondary = fs::weakly_canonical("Y:/Ai-Loader-training-20260924");
    if (secondary != fs::path("Y:/Ai-Loader-training-20260924").lexically_normal() ||
        resolved == secondary || resolved.lexically_relative(secondary).empty() ||
        *resolved.lexically_relative(secondary).begin() == "..") {
        throw std::runtime_error("path outside exact private training roots");
    }
    std::ifstream marker_file(secondary / "storage.json");
    if (!marker_file) throw std::runtime_error("secondary training marker missing");
    const json marker = json::parse(marker_file);
    if (marker.value("schema", "") != "ai-loader-private-training-storage/v1" ||
        fs::weakly_canonical(fs::path(marker.value("project", ""))) != fs::weakly_canonical(repo) ||
        marker.value("target_sha256", "") != target_sha256 ||
        marker.value("minimum_free_bytes", uint64_t(0)) != 40ull * 1024ull * 1024ull * 1024ull) {
        throw std::runtime_error("secondary training marker mismatch");
    }
    return resolved;
}

static void capture_one(llama_model * model, const fs::path & repo, const capture_request & request,
                        const std::string & sha256, const std::string & binary_sha256) {
    const fs::path ids_path = checked_private(request.ids_path, repo, sha256);
    const fs::path protected_path = checked_private(request.protected_path, repo, sha256);
    const fs::path output = checked_private(request.output, repo, sha256);
    if (!fs::is_regular_file(ids_path) || !fs::is_regular_file(protected_path)) throw std::runtime_error("private input missing");
    const std::string ids_sha256 = sha256_file(ids_path);
    if (!request.expected_ids_sha256.empty() && ids_sha256 != request.expected_ids_sha256) {
        throw std::runtime_error("token file differs from input receipt");
    }
    const int64_t source_start = request.source_start, source_end = request.source_end;
    const int64_t repeat_start = request.repeat_start, repeat_end = request.repeat_end;
    const std::string & template_sha256 = request.template_sha256;
    if (!lowercase_sha256(template_sha256)) throw std::runtime_error("invalid template hash");

    {
        const auto ids = read_ids(ids_path);
        const auto protected_positions = read_ids(protected_path);
        const int64_t source_len = source_end - source_start;
        const int64_t repeat_len = repeat_end - repeat_start;
        if (ids.size() > 4096 || source_start < 0 || source_len < 32 || source_len > 1250 ||
            source_end > repeat_start || repeat_start < 0 || repeat_len < 1 || repeat_end > static_cast<int64_t>(ids.size())) {
            throw std::runtime_error("invalid full-chat source/repeat token bounds");
        }
        const int32_t microbatch = 64;
        const int32_t recent = 256;
        const int32_t sink = 4;
        std::vector<uint8_t> eligible(static_cast<size_t>(source_len), 0);
        for (int64_t key = source_start; key < source_end - recent; ++key) {
            if (key >= sink) eligible[static_cast<size_t>(key - source_start)] = 1;
        }
        for (auto key : protected_positions) {
            if (key < source_start || key >= source_end) throw std::runtime_error("protected position outside original source");
            eligible[static_cast<size_t>(key - source_start)] = 0;
        }
        if (std::none_of(eligible.begin(), eligible.end(), [](uint8_t x) { return x != 0; })) throw std::runtime_error("no eligible source positions");

        capture state;
        state.layers.resize(static_cast<size_t>(llama_model_n_layer(model)));
        state.source_len = source_len;
        state.source_start = source_start;
        state.source_end = source_end;
        state.repeat_start = repeat_start;
        state.n_head = llama_model_n_head(model);
        state.n_kv_head = llama_model_n_head_kv(model);
        state.hidden = llama_model_n_embd(model);
        if (state.layers.size() != 64 || state.hidden != 5120 || state.n_head != 24 || state.n_kv_head != 4) {
            throw std::runtime_error("model is not the Bonsai Qwen35 pilot geometry");
        }
        if (fs::exists(output) || !fs::create_directory(output)) throw std::runtime_error("output directory already exists or cannot be created");
        for (int il = 3; il < 64; il += 4) {
            auto & layer = state.layers[il];
            const std::string base = "layer-" + std::to_string(il);
            layer.features.open(output / (base + ".features.f32"), std::ios::binary);
            layer.values.open(output / (base + ".values.f32"), std::ios::binary);
            layer.probe_gated.open(output / (base + ".probe_gated.f32"), std::ios::binary);
            layer.probe_output.open(output / (base + ".probe_output.f32"), std::ios::binary);
            layer.max_attention.resize(static_cast<size_t>(source_len * state.n_head), 0.0f);
            if (!layer.features || !layer.values || !layer.probe_gated || !layer.probe_output) throw std::runtime_error("cannot create oracle layer files");
        }
        auto cp = llama_context_default_params();
        cp.n_ctx = 4096;
        cp.n_batch = microbatch;
        cp.n_ubatch = microbatch;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.type_k = GGML_TYPE_F16;
        cp.type_v = GGML_TYPE_F16;
        cp.cb_eval = callback;
        cp.cb_eval_user_data = &state;
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model, cp), llama_free);
        if (!ctx) throw std::runtime_error("context init failed");
        decode(ctx.get(), state, ids, 0, source_start, microbatch);
        state.phase = capture::source;
        decode(ctx.get(), state, ids, source_start, source_end, microbatch);
        state.phase = capture::ignore;
        decode(ctx.get(), state, ids, source_end, repeat_start, microbatch);
        state.phase = capture::replay;
        decode(ctx.get(), state, ids, repeat_start, repeat_end, microbatch);
        state.phase = capture::ignore;
        decode(ctx.get(), state, ids, repeat_end, static_cast<int64_t>(ids.size()), microbatch);
        for (int il = 3; il < 64; il += 4) {
            auto & layer = state.layers[il];
            if (layer.feature_rows != source_len || layer.value_rows != source_len || layer.replay_rows != repeat_len ||
                layer.probe_gated_rows != 8 || layer.probe_output_rows != 8) {
                throw std::runtime_error("incomplete layer capture");
            }
            layer.features.close();
            layer.values.close();
            layer.probe_gated.close();
            layer.probe_output.close();
            if (!layer.features || !layer.values || !layer.probe_gated || !layer.probe_output) throw std::runtime_error("oracle layer write failed");
            const std::string base = "layer-" + std::to_string(il);
            std::ofstream out(output / (base + ".attention.f32"), std::ios::binary);
            out.write(reinterpret_cast<const char *>(layer.max_attention.data()), static_cast<std::streamsize>(layer.max_attention.size() * sizeof(float)));
            if (!out) throw std::runtime_error("oracle attention write failed");
        }
        {
            std::ofstream out(output / "eligible.u8", std::ios::binary);
            out.write(reinterpret_cast<const char *>(eligible.data()), static_cast<std::streamsize>(eligible.size()));
            if (!out) throw std::runtime_error("eligibility write failed");
        }
        {
            std::ofstream source_pos(output / "source_positions.i32", std::ios::binary);
            std::ofstream query_pos(output / "query_positions.i32", std::ios::binary);
            std::ofstream query_origin(output / "query_origin.i32", std::ios::binary);
            for (int32_t i = 0; i < source_len; ++i) {
                const int32_t pos = static_cast<int32_t>(source_start + i);
                source_pos.write(reinterpret_cast<const char *>(&pos), 4);
            }
            for (int32_t i = 0; i < repeat_len; ++i) {
                const int32_t pos = static_cast<int32_t>(repeat_start + i);
                const int32_t origin = -1; // Same copied text, but tokenizer positions need not align 1:1.
                query_pos.write(reinterpret_cast<const char *>(&pos), 4);
                query_origin.write(reinterpret_cast<const char *>(&origin), 4);
            }
            if (!source_pos || !query_pos || !query_origin) throw std::runtime_error("position mapping write failed");
        }
        {
            std::ofstream out(output / "receipt.json");
            out << "{\n  \"kind\": \"kvzap_full_chat_native_attention_capture_v3\",\n"
                << "  \"target_gguf_sha256\": \"" << sha256 << "\",\n"
                << "  \"native_binary_sha256\": \"" << binary_sha256 << "\",\n"
                << "  \"ids_sha256\": \"" << ids_sha256 << "\",\n"
                << "  \"replay_template_sha256\": \"" << template_sha256 << "\",\n"
                << "  \"input_length\": " << ids.size() << ",\n"
                << "  \"source_length\": " << source_len << ",\n"
                << "  \"source_start_position\": " << source_start << ",\n"
                << "  \"source_end_position\": " << source_end << ",\n"
                << "  \"repeat_start_position\": " << repeat_start << ",\n"
                << "  \"repeat_end_position\": " << repeat_end << ",\n"
                << "  \"repeat_length\": " << repeat_len << ",\n"
                << "  \"replay_microbatch\": " << microbatch << ",\n"
                << "  \"replay_microbatch_count\": " << (repeat_len + microbatch - 1) / microbatch << ",\n"
                << "  \"sink\": " << sink << ",\n"
                << "  \"recent_eligibility_mask\": " << recent << ",\n"
                << "  \"protected_count\": " << protected_positions.size() << ",\n"
                << "  \"n_head\": " << state.n_head << ",\n"
                << "  \"n_kv_head\": " << state.n_kv_head << ",\n"
                << "  \"head_dim\": " << state.head_dim << ",\n"
                << "  \"hidden\": " << state.hidden << ",\n"
                << "  \"wo_probe_positions\": [" << source_start << ", " << source_start + 1 << ", " << source_start + 2 << ", " << source_start + 3 << ", " << source_end - 4 << ", " << source_end - 3 << ", " << source_end - 2 << ", " << source_end - 1 << "],\n"
                << "  \"wo_probe_gated_shape\": [8, " << state.n_head * state.head_dim << "],\n"
                << "  \"wo_probe_output_shape\": [8, " << state.hidden << "],\n"
                << "  \"flash_attention\": false,\n"
                << "  \"kv_cache_type\": \"f16\",\n"
                << "  \"attention_mask\": \"native_full_causal; query=assistant_copy_only; key=user_source_only\",\n"
                << "  \"query_origin_kind\": \"copied_text_no_token_alignment\",\n"
                << "  \"attention_shape\": [" << source_len << ", " << state.n_head << "],\n"
                << "  \"value_shape\": [" << source_len << ", " << state.n_kv_head << ", " << state.head_dim << "],\n"
                << "  \"feature_shape\": [" << source_len << ", " << state.hidden << "]\n} \n";
            if (!out) throw std::runtime_error("receipt write failed");
        }
        std::cout << "oracle capture complete; private receipt: " << (output / "receipt.json").string() << '\n';
    }
}

int main(int argc, char ** argv) {
    try {
        const bool validate_only = argc == 14 && std::string(argv[13]) == "--validate-only";
        const bool batch = (argc == 13 || validate_only) && std::string(argv[5]) == "--runs";
        if (!batch && argc != 27) throw std::runtime_error("usage: --repo REPO --model GGUF --runs PRIVATE_RUNS_JSON --gpu-layers N --sha256 HASH --binary-sha256 HASH; or the v2 one-context flags");
        const fs::path repo = fs::weakly_canonical(argv[2]);
        const fs::path model_path = argv[4];
        if (std::string(argv[1]) != "--repo" || std::string(argv[3]) != "--model" ||
            !fs::is_directory(repo / "tmp" / "bonsai-training-20260924") || !fs::is_regular_file(model_path)) {
            throw std::runtime_error("invalid repository or model path");
        }
        const int gpu_layers = std::stoi(argv[batch ? 8 : 20]);
        const std::string sha256 = argv[batch ? 10 : 22], binary_sha256 = argv[batch ? 12 : 24];
        if (gpu_layers < 0 || gpu_layers > 256 || !lowercase_sha256(sha256) || !lowercase_sha256(binary_sha256)) {
            throw std::runtime_error("invalid GPU layer count or SHA256");
        }
        std::vector<capture_request> requests;
        if (batch) {
            const char * flags[] = {"--repo", "--model", "--runs", "--gpu-layers", "--sha256", "--binary-sha256"};
            for (int i = 0; i < 6; ++i) if (std::string(argv[1 + i * 2]) != flags[i]) throw std::runtime_error("invalid batch argument order");
            const fs::path runs_path = checked_private(argv[6], repo, sha256);
            std::ifstream stream(runs_path);
            if (!stream) throw std::runtime_error("private runs manifest missing");
            const json runs = json::parse(stream);
            if (!runs.is_array() || runs.empty() || runs.size() > 110) throw std::runtime_error("batch must contain 1 to 110 contexts");
            const std::vector<std::string> required = {"conversation_id", "similarity_group", "source", "prompt_sha256", "response_sha256", "input_receipt", "capture_receipt", "oracle_dir"};
            for (const auto & run : runs) {
                if (!run.is_object() || run.size() != required.size()) throw std::runtime_error("invalid runs row schema");
                for (const auto & key : required) if (!run.contains(key) || !run[key].is_string()) throw std::runtime_error("invalid runs row field");
                if (run["response_sha256"] != run["prompt_sha256"] ||
                    (run["source"] != "curated" && run["source"] != "synthetic")) throw std::runtime_error("invalid run source/response identity");
                const fs::path input_receipt = checked_private(run["input_receipt"].get<std::string>(), repo, sha256);
                const fs::path capture_receipt = checked_private(run["capture_receipt"].get<std::string>(), repo, sha256);
                checked_private(run["oracle_dir"].get<std::string>(), repo, sha256);
                if (input_receipt.filename() != "input-receipt.json" || capture_receipt.filename() != "receipt.json") throw std::runtime_error("invalid receipt filename");
                std::ifstream input(input_receipt);
                if (!input) throw std::runtime_error("input receipt missing");
                const json receipt = json::parse(input);
                if (receipt.value("kind", "") != "kvzap_full_chat_token_spans_v1" ||
                    receipt.value("conversation_id", "") != run["conversation_id"] ||
                    receipt.value("similarity_group", "") != run["similarity_group"] ||
                    receipt.value("source", "") != run["source"] ||
                    receipt.value("prompt_sha256", "") != run["prompt_sha256"]) throw std::runtime_error("run/input provenance mismatch");
                capture_request request {input_receipt.parent_path() / "ids.txt", input_receipt.parent_path() / "protected.txt",
                                         capture_receipt.parent_path(), receipt.at("source_start").get<int64_t>(),
                                         receipt.at("source_end").get<int64_t>(), receipt.at("repeat_start").get<int64_t>(),
                                         receipt.at("repeat_end").get<int64_t>(), receipt.at("template_sha256").get<std::string>(),
                                         receipt.at("ids_sha256").get<std::string>()};
                checked_private(request.ids_path, repo, sha256);
                checked_private(request.protected_path, repo, sha256);
                if (!lowercase_sha256(request.expected_ids_sha256)) throw std::runtime_error("invalid input ID hash");
                if (sha256_file(request.ids_path) != request.expected_ids_sha256) throw std::runtime_error("batch token file hash mismatch");
                requests.push_back(request);
            }
        } else {
            const char * flags[] = {"--repo", "--model", "--ids", "--protected", "--output", "--source-start", "--source-end", "--repeat-start", "--repeat-end", "--gpu-layers", "--sha256", "--binary-sha256", "--template-sha256"};
            for (int i = 0; i < 13; ++i) if (std::string(argv[1 + i * 2]) != flags[i]) throw std::runtime_error("invalid one-context argument order");
            const std::string output_name = argv[10];
            if (output_name.empty() || output_name == "." || output_name == ".." || output_name.find_first_of("/\\") != std::string::npos) throw std::runtime_error("output must be one directory name");
            requests.push_back({argv[6], argv[8], repo / "tmp" / "bonsai-training-20260924" / output_name,
                                std::stoll(argv[12]), std::stoll(argv[14]), std::stoll(argv[16]), std::stoll(argv[18]), argv[26], ""});
        }
        if (validate_only) {
            std::cout << "private batch validated; contexts: " << requests.size() << '\n';
            return 0;
        }
        if (sha256_file(argv[0]) != binary_sha256 || sha256_file(model_path) != sha256) {
            throw std::runtime_error("native executable or target GGUF hash mismatch");
        }
        llama_backend_init();
        try {
            auto params = llama_model_default_params();
            params.n_gpu_layers = gpu_layers;
            std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(model_path.string().c_str(), params), llama_model_free);
            if (!model) throw std::runtime_error("model load failed");
            for (const auto & request : requests) {
                if (batch && fs::is_regular_file(request.output / "receipt.json")) {
                    std::ifstream done_file(request.output / "receipt.json");
                    const json done = json::parse(done_file);
                    if (done.value("ids_sha256", "") != request.expected_ids_sha256 ||
                        done.value("target_gguf_sha256", "") != sha256 ||
                        done.value("native_binary_sha256", "") != binary_sha256 ||
                        done.value("replay_template_sha256", "") != request.template_sha256 ||
                        done.value("source_start_position", int64_t(-1)) != request.source_start ||
                        done.value("source_end_position", int64_t(-1)) != request.source_end ||
                        done.value("repeat_start_position", int64_t(-1)) != request.repeat_start ||
                        done.value("repeat_end_position", int64_t(-1)) != request.repeat_end) {
                        throw std::runtime_error("existing capture receipt mismatches this batch");
                    }
                    continue;
                }
                capture_one(model.get(), repo, request, sha256, binary_sha256);
            }
            model.reset();
            llama_backend_free();
        } catch (...) {
            llama_backend_free();
            throw;
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "oracle capture failed: " << error.what() << '\n';
        return 1;
    }
}
