#include "llama-kv-cache.h"

#include "llama-impl.h"
#include "llama-io.h"
#include "llama-model.h"
#include "llama-context.h"

#include "../ggml/src/ggml-turbo4.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

static bool ggml_is_power_of_2(int n) {
    return (n & (n - 1)) == 0;
}

static bool llama_kv_type_is_turbo(ggml_type type) {
    return type == GGML_TYPE_TURBO4_0 || type == GGML_TYPE_TURBO3_5;
}

static uint32_t llama_turbo4_padded_head(uint32_t head_dim) {
    if (head_dim == 64 || head_dim == GGML_TURBO4_QK) {
        return GGML_TURBO4_QK;
    }
    if (head_dim == 2*GGML_TURBO4_QK) {
        return 2*GGML_TURBO4_QK;
    }
    return 0;
}

// orthonormal Walsh-Hadamard rotation matrix
// note: res^2 == I
static void ggml_gen_hadamard(ggml_tensor * tensor) {
    assert(tensor->type == GGML_TYPE_F32);

    const int n = tensor->ne[0];

    assert(ggml_is_power_of_2(n));
    assert(tensor->ne[1] == n);
    assert(tensor->ne[2] == 1);
    assert(tensor->ne[3] == 1);

    std::vector<float> data_f32;

    float * data = (float *) tensor->data;

    if (tensor->type != GGML_TYPE_F32) {
        data_f32.resize(n*n);
        data = data_f32.data();
    }

    data[0*n + 0] = 1.0 / sqrtf(n);

    for (int s = 1; s < n; s *= 2) {
        for (int i = 0; i < s; i++) {
            for (int j = 0; j < s; j++) {
                const float val = data[i*n + j];

                data[(i + s)*n + (j    )] =  val;
                data[(i    )*n + (j + s)] =  val;
                data[(i + s)*n + (j + s)] = -val;
            }
        }
    }

    if (tensor->type != GGML_TYPE_F32) {
        ggml_quantize_chunk(tensor->type, data, tensor->data, 0, 1, n*n, nullptr);
    }
}

//
// llama_kv_cache
//

llama_kv_cache::llama_kv_cache(
        const llama_model & model,
        const llama_hparams & hparams,
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                     bool   offload,
                     bool   unified,
                 uint32_t   kv_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
    llama_memory_t   mem_other,
    const layer_filter_cb & filter,
    const  layer_reuse_cb & reuse,
    const  layer_share_cb & share,
    ggml_backend_buffer_type_t kv_buffer_type,
    llama_pyramidkv_c1_config pyramidkv_c1,
    bool tq4_key_center) :
    model(model), hparams(hparams), v_trans(v_trans),
    n_seq_max(n_seq_max), n_stream(unified ? 1 : n_seq_max), n_pad(n_pad), n_swa(n_swa), swa_type(swa_type),
    other(static_cast<llama_kv_cache *>(mem_other)),
    v_cells_impl(other ? other->v_cells_impl : std::make_shared<llama_kv_cells_vec>()),
    v_cells(*v_cells_impl),
    kv_buffer_type(kv_buffer_type),
    pyramidkv_c1_config(std::move(pyramidkv_c1)),
    tq4_key_center_enabled_flag(tq4_key_center) {

    if (kv_buffer_type != nullptr) {
        if (mem_other != nullptr) {
            throw std::runtime_error("kv_buffer_type cannot be combined with a shared KV cache");
        }
        ggml_backend_dev_t kv_dev = ggml_backend_buft_get_device(kv_buffer_type);
        if (kv_dev == nullptr) {
            throw std::runtime_error("kv_buffer_type has no backend device");
        }
        if (!ggml_backend_dev_supports_buft(kv_dev, kv_buffer_type)) {
            throw std::runtime_error("kv_buffer_type is not supported by its backend device");
        }
    }

    // shared cells view the source cache's K/V tensors, so the cell count
    // follows the source allocation: a fitted target can be smaller than the
    // draft default and oversized views would overflow the source tensors
    if (other) {
        const uint32_t size_other = other->get_size();
        if (kv_size != size_other) {
            LLAMA_LOG_WARN("%s: kv_size = %u overridden to %u to match the shared source cache\n", __func__, kv_size, size_other);
            kv_size = size_other;
        }
    }

    GGML_ASSERT(kv_size % n_pad == 0);

    const uint32_t n_layer = hparams.n_layer_all;
    const bool has_turbo4 = llama_kv_type_is_turbo(type_k) || llama_kv_type_is_turbo(type_v);
    const auto & c1_config = pyramidkv_c1_config;

    if (tq4_key_center_enabled_flag) {
        ggml_backend_dev_t kv_dev = kv_buffer_type == nullptr ? nullptr : ggml_backend_buft_get_device(kv_buffer_type);
        if (kv_dev == nullptr || ggml_backend_dev_type(kv_dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            throw std::runtime_error("TQ4 key center requires an explicit GPU KV buffer");
        }
        if (mem_other != nullptr || filter || reuse || share) {
            throw std::runtime_error("TQ4 key center rejects shared, filtered, reused, or aliased KV caches");
        }
        if (!unified || n_seq_max != 1 || n_stream != 1 || n_swa != 0 || swa_type != LLAMA_SWA_TYPE_NONE) {
            throw std::runtime_error("TQ4 key center requires one unified non-SWA sequence");
        }
        if (model.arch != LLM_ARCH_QWEN2) {
            throw std::runtime_error("TQ4 key center is limited to dense Qwen2");
        }
        if (hparams.no_alloc) {
            throw std::runtime_error("TQ4 key center requires allocated KV tensors");
        }
        if (type_k != type_v || (type_k != GGML_TYPE_F16 && type_k != GGML_TYPE_TURBO4_0)) {
            throw std::runtime_error("TQ4 key center requires symmetric F16 or TQ4 K/V");
        }
        for (uint32_t il = 0; il < n_layer; ++il) {
            if (!hparams.has_kv(il)) {
                continue;
            }
            if (hparams.n_embd_head_k(il) != GGML_TURBO4_QK ||
                    hparams.n_embd_head_v(il) != GGML_TURBO4_QK ||
                    hparams.n_head_kv(il) == 0) {
                throw std::runtime_error("TQ4 key center requires D128 KV heads on every dense layer");
            }
        }
    }

    if (pyramidkv_c1_config.enabled) {
        if (!llama_kv_type_is_turbo(type_k) || type_v != type_k) {
            throw std::runtime_error("PyramidKV C1 requires symmetric TQ4 or TQ3.5 cold K/V");
        }
        // Preserve the full prompt until its first requested output. Earlier
        // chunks cannot score facts against a question that has not arrived.
        // Compaction releases this full TQ4 allocation after prompt completion.
        pyramidkv_c1_initial_capacity = kv_size;
        if (pyramidkv_c1_initial_capacity == 0) {
            throw std::runtime_error("PyramidKV C1 initial physical capacity is empty");
        }
        pyramidkv_c1_hot_enabled = true;
    }

    // define a comparator for the buft -> ctx map to ensure that the order is well-defined:
    struct ggml_backend_buft_comparator {
        bool operator()(const ggml_backend_buffer_type_t & lhs, const ggml_backend_buffer_type_t & rhs) const {
            return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };
    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> ctx_map;

    // create a context for each buffer type
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/ size_t((pyramidkv_c1_hot_enabled ? 4u : 2u)*(1 + n_stream)*n_layer*ggml_tensor_overhead() +
                    (has_turbo4 ? 2u : 0u)*ggml_tensor_overhead()),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                return nullptr;
            }

            ctx_map.emplace(buft, ctx);

            return ctx;
        }

        return it->second.get();
    };

    GGML_ASSERT(n_stream == 1 || n_stream == n_seq_max);

    v_heads.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_heads[s] = 0;
    }

    v_cells.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_cells[s].resize(kv_size);
    }

    // by default, all sequence ids are mapped to the 0th stream
    seq_to_stream.resize(LLAMA_MAX_SEQ, 0);

    if (n_stream > 1) {
        seq_to_stream.resize(n_stream, 0);
        for (uint32_t s = 0; s < n_stream; ++s) {
            seq_to_stream[s] = s;
        }
    }

    // [TAG_V_CACHE_VARIABLE]
    if (v_trans && hparams.is_n_embd_v_gqa_variable()) {
        LLAMA_LOG_WARN("%s: the V embeddings have different sizes across layers and FA is not enabled - padding V cache to %d\n",
                __func__, hparams.n_embd_v_gqa_max());
    }

    const bool is_mla = hparams.is_mla();

    for (uint32_t il = 0; il < n_layer; il++) {
        if (!hparams.has_kv(il)) {
            LLAMA_LOG_DEBUG("%s: layer %3d: does not have KV cache\n", __func__, il);
            continue;
        }

        if (filter && !filter(il)) {
            LLAMA_LOG_DEBUG("%s: layer %3d: filtered\n", __func__, il);
            continue;
        }

        if (share && other) {
            const int32_t il_share = share(il);

            if (il_share >= 0) {
                const auto & layer_share = other->layers[other->map_layer_ids[il_share]];

                LLAMA_LOG_WARN("%s: layer %3d: sharing with layer %d. k = %p, v = %p\n", __func__, il, il_share,
                        layer_share.k->data, layer_share.v->data);

                map_layer_ids[il] = layers.size();

                layers.push_back(layer_share);
                layers.back().il = il;

                continue;
            }
        }

        if (n_embd_head_k_all == 0) {
            n_embd_head_k_all = (int32_t) hparams.n_embd_head_k(il);
        } else if (n_embd_head_k_all > 0 && n_embd_head_k_all != (int32_t) hparams.n_embd_head_k(il)) {
            n_embd_head_k_all = -1;
        }

        if (!is_mla) {
            if (n_embd_head_v_all == 0) {
                n_embd_head_v_all = (int32_t) hparams.n_embd_head_v(il);
            } else if (n_embd_head_v_all > 0 && n_embd_head_v_all != (int32_t) hparams.n_embd_head_v(il)) {
                n_embd_head_v_all = -1;
            }
        }

        // [TAG_V_CACHE_VARIABLE]
        const uint32_t n_embd_head_k = hparams.n_embd_head_k(il);
        const uint32_t n_embd_head_v = hparams.n_embd_head_v(il);
        const uint32_t n_head_kv     = hparams.n_head_kv(il);

        const uint32_t n_embd_k_gqa = llama_kv_type_is_turbo(type_k) ?
            n_head_kv * llama_turbo4_padded_head(n_embd_head_k) : hparams.n_embd_k_gqa(il);
        const uint32_t n_embd_v_gqa = llama_kv_type_is_turbo(type_v) ?
            n_head_kv * llama_turbo4_padded_head(n_embd_head_v) :
            (!v_trans ? hparams.n_embd_v_gqa(il) : hparams.n_embd_v_gqa_max());

        if ((llama_kv_type_is_turbo(type_k) && llama_turbo4_padded_head(n_embd_head_k) == 0) ||
            (llama_kv_type_is_turbo(type_v) && llama_turbo4_padded_head(n_embd_head_v) == 0)) {
            throw std::runtime_error("TurboQuant4 requires per-head dimensions 64, 128, or 256");
        }

        const char * dev_name = "CPU";

        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();

        if (kv_buffer_type != nullptr) {
            buft = kv_buffer_type;
            dev_name = ggml_backend_dev_name(ggml_backend_buft_get_device(kv_buffer_type));
        } else if (offload) {
            auto * dev = model.dev_layer(il);
            buft = ggml_backend_dev_buffer_type(dev);

            dev_name = ggml_backend_dev_name(dev);
        }

        LLAMA_LOG_DEBUG("%s: layer %3d: dev = %s\n", __func__, il, dev_name);

        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            throw std::runtime_error("failed to create ggml context for kv cache");
        }

        const bool has_k = true;
        const bool has_v = !is_mla;

        const uint32_t physical_kv_size = pyramidkv_c1_hot_enabled ?
            pyramidkv_c1_initial_capacity : kv_size;
        ggml_tensor * k = has_k ? ggml_new_tensor_3d(ctx, type_k, n_embd_k_gqa,
            physical_kv_size, n_stream) : nullptr;
        ggml_tensor * v = has_v ? ggml_new_tensor_3d(ctx, type_v, n_embd_v_gqa,
            physical_kv_size, n_stream) : nullptr;
        ggml_tensor * k_hot = nullptr;
        ggml_tensor * v_hot = nullptr;

        has_k && ggml_format_name(k, "cache_k_l%d", il);
        has_v && ggml_format_name(v, "cache_v_l%d", il);

        std::vector<ggml_tensor *> k_stream;
        std::vector<ggml_tensor *> v_stream;
        std::vector<ggml_tensor *> k_hot_stream;
        std::vector<ggml_tensor *> v_hot_stream;

        for (uint32_t s = 0; s < n_stream; ++s) {
            k_stream.push_back(has_k ? ggml_view_2d(ctx, k, n_embd_k_gqa,
                physical_kv_size, k->nb[1], s*k->nb[2]) : nullptr);
            v_stream.push_back(has_v ? ggml_view_2d(ctx, v, n_embd_v_gqa,
                physical_kv_size, v->nb[1], s*v->nb[2]) : nullptr);
        }

        if (pyramidkv_c1_hot_enabled) {
            const uint32_t hot_k_head = llama_turbo4_padded_head(n_embd_head_k);
            const uint32_t hot_v_head = llama_turbo4_padded_head(n_embd_head_v);
            if (hot_k_head == 0 || hot_v_head == 0 ||
                    c1_config.hot_capacity > std::numeric_limits<uint32_t>::max() / n_head_kv) {
                throw std::runtime_error("PyramidKV C1 hot-row geometry is unsupported");
            }
            const uint32_t hot_rows = static_cast<uint32_t>(c1_config.hot_capacity * n_head_kv);
            k_hot = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, hot_k_head, hot_rows, n_stream);
            v_hot = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, hot_v_head, hot_rows, n_stream);
            ggml_format_name(k_hot, "cache_k_hot_l%d", il);
            ggml_format_name(v_hot, "cache_v_hot_l%d", il);
            for (uint32_t s = 0; s < n_stream; ++s) {
                k_hot_stream.push_back(ggml_view_2d(ctx, k_hot, hot_k_head, hot_rows,
                    k_hot->nb[1], s*k_hot->nb[2]));
                v_hot_stream.push_back(ggml_view_2d(ctx, v_hot, hot_v_head, hot_rows,
                    v_hot->nb[1], s*v_hot->nb[2]));
            }
        }

        map_layer_ids[il] = layers.size();

        layers.push_back({ il, k, v, k_stream, v_stream, k_hot, v_hot, k_hot_stream, v_hot_stream });

        if (!other && has_turbo4 && turbo_rotation == nullptr) {
            turbo_rotation = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, GGML_TURBO4_QK, GGML_TURBO4_QK);
            ggml_format_name(turbo_rotation, "turbo4_rotation");
            turbo_rotation_inv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, GGML_TURBO4_QK, GGML_TURBO4_QK);
            ggml_format_name(turbo_rotation_inv, "turbo4_rotation_inv");
        }
    }

    if (reuse) {
        LLAMA_LOG_DEBUG("%s: reusing layers:\n", __func__);

        for (uint32_t il = 0; il < n_layer; il++) {
            const int32_t il_reuse = reuse(il);

            if (il_reuse < 0) {
                LLAMA_LOG_DEBUG("%s: - layer %3d: no reuse\n", __func__, il);
                continue;
            }

            if (filter && !filter(il)) {
                LLAMA_LOG_DEBUG("%s: - layer %3d: filtered\n", __func__, il);
                continue;
            }

            GGML_ASSERT(map_layer_ids.find(il_reuse) != map_layer_ids.end());

            map_layer_ids[il] = map_layer_ids[il_reuse];

            LLAMA_LOG_DEBUG("%s: - layer %3d: reuse layer %d, is_swa = %d\n", __func__, il, il_reuse, hparams.is_swa(il));
        }
    }

    pyramidkv_c1_layers.resize(layers.size());
    if (pyramidkv_c1_hot_enabled) {
        for (size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
            auto & state = pyramidkv_c1_layers[layer_index];
            state.kv_heads = hparams.n_head_kv(layers[layer_index].il);
            state.hot_row_capacity = static_cast<uint32_t>(c1_config.hot_capacity);
            state.hot_recent = static_cast<uint32_t>(c1_config.recent_window);
            state.hot_heads.resize(state.kv_heads);
            for (auto & head : state.hot_heads) {
                head.row_capacity = state.hot_row_capacity;
                head.logical_to_physical.assign(kv_size, pyramidkv_c1_invalid_cell);
                head.physical_to_logical.assign(state.hot_row_capacity, pyramidkv_c1_invalid_cell);
            }
        }
        pyramidkv_c1_paged_reset();
        if (pyramidkv_c1_local_prefill()) {
            LLAMA_LOG_INFO("%s: PYRAMIDKV_LOCAL_PREFILL enabled, packed TQ4 gather for single-sequence prompts\n", __func__);
        }
    }

    // allocate tensors and initialize the buffers to avoid NaNs in the padding
    for (auto & [buft, ctx] : ctx_map) {
        ggml_backend_buffer_t buf;
        if (hparams.no_alloc) {
            buf = ggml_backend_buft_alloc_buffer(buft, /*size =*/ 0); // dummy buffer
            for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
                t->buffer = buf; // set dummy buffer for KV cache so that the backend scheduler won't try to allocate it
            }
        } else {
            buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft); // real buffer
        }
        if (!buf) {
            throw std::runtime_error("failed to allocate buffer for kv cache");
        }

        LLAMA_LOG_INFO("%s: %10s KV buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);

        ggml_backend_buffer_clear(buf, 0);

        if (turbo_rotation != nullptr && turbo_rotation->buffer == buf && !model.hparams.no_alloc) {
            std::array<float, GGML_TURBO4_QK*GGML_TURBO4_QK> rotation;
            std::array<float, GGML_TURBO4_QK*GGML_TURBO4_QK> rotation_inv;
            ggml_turbo4_rotation_matrix(rotation.data(), false);
            ggml_turbo4_rotation_matrix(rotation_inv.data(), true);

            ggml_backend_tensor_set(turbo_rotation, rotation.data(), 0, rotation.size()*sizeof(float));
            ggml_backend_tensor_set(turbo_rotation_inv, rotation_inv.data(), 0, rotation_inv.size()*sizeof(float));
        }

        ctxs_bufs.emplace_back(std::move(ctx), buf);
    }

    if (pyramidkv_c1_hot_enabled && !hparams.no_alloc) {
        std::string aux_error;
        if (!pyramidkv_c1_aux_rebuild(aux_error)) {
            throw std::runtime_error(aux_error);
        }
    }

    if (tq4_key_center_enabled_flag) {
        if (layers.empty()) {
            throw std::runtime_error("TQ4 key center found no dense KV layers");
        }
        ggml_init_params anchor_params = {
            /*.mem_size   =*/ (layers.size() + 1)*ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        tq4_key_anchor_ctx.reset(ggml_init(anchor_params));
        if (!tq4_key_anchor_ctx) {
            throw std::runtime_error("failed to create TQ4 key-center anchor context");
        }

        tq4_key_anchors.reserve(layers.size());
        for (const auto & layer : layers) {
            const uint32_t n_head_kv = hparams.n_head_kv(layer.il);
            ggml_tensor * anchor = ggml_new_tensor_2d(
                    tq4_key_anchor_ctx.get(), GGML_TYPE_F32, GGML_TURBO4_QK, n_head_kv);
            if (anchor == nullptr) {
                throw std::runtime_error("failed to create TQ4 key-center anchor tensor");
            }
            ggml_format_name(anchor, "tq4_key_anchor_l%d", layer.il);
            tq4_key_anchors.push_back(anchor);
        }

        tq4_key_anchor_buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(
                tq4_key_anchor_ctx.get(), kv_buffer_type));
        if (!tq4_key_anchor_buf) {
            throw std::runtime_error("failed to allocate TQ4 key-center anchor buffer");
        }
        ggml_backend_buffer_clear(tq4_key_anchor_buf.get(), 0);
    }

    {
        const size_t memory_size_k = size_k_bytes();
        const size_t memory_size_v = size_v_bytes();

        LLAMA_LOG_INFO("%s: size = %7.2f MiB (%6u cells, %3d layers, %2u/%u seqs), K (%s): %7.2f MiB, V (%s): %7.2f MiB\n", __func__,
                (float)(memory_size_k + memory_size_v) / (1024.0f * 1024.0f), kv_size, (int) layers.size(), n_seq_max, n_stream,
                ggml_type_name(type_k), (float)memory_size_k / (1024.0f * 1024.0f),
                ggml_type_name(type_v), (float)memory_size_v / (1024.0f * 1024.0f));
    }

    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        n_embd_head_k_all = other->n_embd_head_k_all;
        n_embd_head_v_all = other->n_embd_head_v_all;

        attn_rot_k = other->attn_rot_k;
        attn_rot_v = other->attn_rot_v;
        turbo_rotation = other->turbo_rotation;
        turbo_rotation_inv = other->turbo_rotation_inv;
        if (has_turbo4) {
            attn_rot_k = false;
            attn_rot_v = false;
            if (turbo_rotation == nullptr || turbo_rotation_inv == nullptr) {
                throw std::runtime_error("TurboQuant4 shared KV cache has no rotation matrices");
            }
        }
    } else {
        const char * LLAMA_ATTN_ROT_DISABLE = getenv("LLAMA_ATTN_ROT_DISABLE");
        const bool attn_rot_disable = LLAMA_ATTN_ROT_DISABLE ? atoi(LLAMA_ATTN_ROT_DISABLE) : false;
        if (attn_rot_disable) {
            LLAMA_LOG_WARN("%s: attention rotation force disabled (LLAMA_ATTN_ROT_DISABLE)\n", __func__);
        }

        attn_rot_k =
            !attn_rot_disable &&
            !llama_kv_type_is_turbo(type_k) &&
            n_embd_head_k_all > 0 &&
            ggml_is_quantized(type_k) &&
            hparams.n_embd_head_k() % 64 == 0;

        // always create Hadamard rotation tensors for DeepSeek lightning indexers
        if ((model.arch == LLM_ARCH_DEEPSEEK32 || model.arch == LLM_ARCH_DEEPSEEK4 ||
                model.arch == LLM_ARCH_GLM_DSA || model.arch == LLM_ARCH_DOTS3NOTE) &&
                hparams.n_embd_head_k_full == hparams.indexer_head_size &&
                !llama_kv_type_is_turbo(type_k)) {
            attn_rot_k = true;
        }

        attn_rot_v =
            !attn_rot_disable &&
            !llama_kv_type_is_turbo(type_v) &&
            n_embd_head_v_all > 0 &&
            ggml_is_quantized(type_v) &&
            hparams.n_embd_head_v() % 64 == 0;
    }

    LLAMA_LOG_INFO("%s: attn_rot_k = %d, n_embd_head_k_all = %d\n", __func__, attn_rot_k, n_embd_head_k_all);
    LLAMA_LOG_INFO("%s: attn_rot_v = %d, n_embd_head_k_all = %d\n", __func__, attn_rot_v, n_embd_head_v_all);

    // pre-compute the haramard matrices and keep them in host memory
    // TODO: in the future, we can make copies in the backend buffers to avoid host -> device transfers
    if (attn_rot_k || attn_rot_v) {
        for (int64_t n = 64; n <= std::max(n_embd_head_k_all, n_embd_head_v_all); n *= 2) {
            attn_rot_hadamard[n] = std::vector<float>(n*n);

            ggml_init_params params = {
                /* .mem_size   = */ 1*ggml_tensor_overhead(),
                /* .mem_buffer = */ nullptr,
                /* .no_alloc   = */ true,
            };

            ggml_context_ptr ctx { ggml_init(params) };

            ggml_tensor * tmp = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n, n);
            tmp->data = attn_rot_hadamard[n].data();

            ggml_gen_hadamard(tmp);
        }
    }

    const char * LLAMA_KV_CACHE_DEBUG = getenv("LLAMA_KV_CACHE_DEBUG");
    debug = LLAMA_KV_CACHE_DEBUG ? atoi(LLAMA_KV_CACHE_DEBUG) : 0;
}

llama_kv_cache::~llama_kv_cache() {
    // The event belongs to the device; the context may already have freed its backend.
    pyramidkv_c1_aux.wait_upload();
}

bool llama_kv_cache::tq4_key_center_enabled() const {
    return tq4_key_center_enabled_flag;
}

bool llama_kv_cache::tq4_key_center_capture() const {
    return tq4_key_center_enabled_flag && !tq4_key_center_valid_flag;
}

bool llama_kv_cache::tq4_key_center_ready() const {
    return tq4_key_center_enabled_flag && tq4_key_center_valid_flag && !tq4_key_center_failed_flag;
}

ggml_tensor * llama_kv_cache::get_tq4_key_anchor(int32_t il) const {
    if (!tq4_key_center_enabled_flag) {
        return nullptr;
    }
    const auto it = map_layer_ids.find(il);
    if (it == map_layer_ids.end() || it->second < 0 ||
            static_cast<size_t>(it->second) >= tq4_key_anchors.size()) {
        return nullptr;
    }
    return tq4_key_anchors[it->second];
}

bool llama_kv_cache::tq4_key_center_cache_empty() const {
    for (const auto & cells : v_cells) {
        if (cells.get_used() != 0) {
            return false;
        }
    }
    return true;
}

void llama_kv_cache::tq4_key_center_invalidate() noexcept {
    if (!tq4_key_center_enabled_flag) {
        return;
    }
    tq4_key_center_valid_flag = false;
    tq4_key_center_capture_pending = false;
}

bool llama_kv_cache::tq4_key_center_prepare(const llama_ubatch & ubatch, std::string & error) {
    if (!tq4_key_center_enabled_flag) {
        return true;
    }
    if (tq4_key_center_failed_flag) {
        error = "TQ4 key-center capture is failed and requires a full clear";
        return false;
    }
    if (tq4_key_center_capture_pending) {
        error = "TQ4 key-center capture is already pending device completion";
        return false;
    }
    if (ubatch.n_tokens == 0 || ubatch.n_pos != 1 || ubatch.n_seqs_unq != 1 ||
            ubatch.pos == nullptr || ubatch.n_seq_id == nullptr || ubatch.seq_id == nullptr) {
        error = "TQ4 key center requires one dense sequence with one position per token";
        return false;
    }
    if (layers.empty() || tq4_key_anchors.size() != layers.size()) {
        error = "TQ4 key center has no complete per-layer anchor allocation";
        return false;
    }
    for (size_t i = 0; i < layers.size(); ++i) {
        const ggml_tensor * anchor = tq4_key_anchors[i];
        if (anchor == nullptr || anchor->type != GGML_TYPE_F32 ||
                anchor->ne[0] != GGML_TURBO4_QK ||
                anchor->ne[1] != hparams.n_head_kv(layers[i].il)) {
            error = "TQ4 key center anchor geometry is invalid";
            return false;
        }
    }

    llama_pos previous = -1;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        if (ubatch.n_seq_id[i] != 1 || ubatch.seq_id[i] == nullptr ||
                ubatch.seq_id[i][0] != 0 || ubatch.pos[i] < 0 ||
                (i != 0 && ubatch.pos[i] <= previous)) {
            error = "TQ4 key center requires strictly increasing sequence-0 positions";
            return false;
        }
        previous = ubatch.pos[i];
    }

    if (tq4_key_center_capture()) {
        if (!tq4_key_center_cache_empty()) {
            error = "TQ4 key center capture requires an empty cache history";
            return false;
        }
        for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
            if (ubatch.pos[i] != static_cast<llama_pos>(i)) {
                error = "TQ4 key center capture requires positions starting at zero without gaps";
                return false;
            }
        }
        tq4_key_center_capture_pending = true;
    }

    return true;
}

void llama_kv_cache::tq4_key_center_commit_capture() {
    if (!tq4_key_center_enabled_flag || tq4_key_center_failed_flag ||
            !tq4_key_center_capture_pending) {
        return;
    }
    tq4_key_center_capture_pending = false;
    tq4_key_center_valid_flag = true;
}

void llama_kv_cache::tq4_key_center_fail() {
    if (!tq4_key_center_enabled_flag) {
        return;
    }
    tq4_key_center_capture_pending = false;
    tq4_key_center_valid_flag = false;
    tq4_key_center_failed_flag = true;
}

void llama_kv_cache::pyramidkv_c1_bind_aux_backend(ggml_backend_t backend) {
    auto & aux = pyramidkv_c1_aux;
    aux.wait_upload();
    aux.upload_done.reset();
    aux.backend = backend;
    if (backend != nullptr && aux.buf) {
        aux.upload_done.reset(ggml_backend_event_new(ggml_backend_get_device(backend)));
    }
}

bool llama_kv_cache::pyramidkv_c1_aux_rebuild(std::string & error) {
    auto & aux = pyramidkv_c1_aux;
    ggml_backend_t backend = aux.backend; // bound once by the context, survives rebuilds
    aux.wait_upload();
    aux = pyramidkv_c1_aux_tensors{};
    aux.backend = backend;
    if (!pyramidkv_c1_hot_enabled || kv_buffer_type == nullptr || pyramidkv_c1_layers.empty()) {
        return true; // host graph inputs remain the fallback
    }
    uint64_t pos_stride = 0;
    uint64_t heads_max  = 0;
    for (const auto & state : pyramidkv_c1_layers) {
        const uint64_t rows = static_cast<uint64_t>(state.cold_row_capacity) + state.hot_row_capacity;
        pos_stride = std::max(pos_stride, rows*state.kv_heads);
        heads_max  = std::max<uint64_t>(heads_max, state.kv_heads);
    }
    const uint64_t n_ubatch   = std::max<uint64_t>(1, pyramidkv_c1_config.continuation_headroom);
    const uint64_t hot_stride = n_ubatch*heads_max;
    const uint64_t n_layers   = pyramidkv_c1_layers.size();
    if (heads_max == 0 || hot_stride == 0 || pos_stride > (1ull << 31) || hot_stride > (1ull << 31) ||
            n_layers > (1ull << 20)) {
        error = "PyramidKV C1 device input geometry is invalid";
        return false;
    }

    // Paged lists: one (arena cell, position, hot row) triple per entry, per head and
    // per sequence slot of the ubatch (at most n_seq_max), per layer.
    const bool paged = pyramidkv_c1_config.paged;
    const uint64_t list_capacity = paged ? std::max<uint64_t>(1, pyramidkv_c1_config.list_capacity) : 0;
    const uint64_t list_stride = paged ? 3ull*list_capacity*heads_max*n_seq_max : 0;
    const uint64_t len_stride  = paged ? static_cast<uint64_t>(heads_max)*n_seq_max : 0;
    if (list_stride > (1ull << 31) || len_stride > (1ull << 31)) {
        error = "PyramidKV C1 paged list geometry exceeds the I32 index range";
        return false;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ 8*ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx { ggml_init(params) };
    if (!ctx) {
        error = "PyramidKV C1 device input context allocation failed";
        return false;
    }
    ggml_tensor * k_positions = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, pos_stride, n_layers);
    ggml_tensor * hot_write   = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, hot_stride, n_layers);
    ggml_tensor * q_positions = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, n_ubatch);
    ggml_format_name(k_positions, "pyramidkv_c1_aux_k_positions");
    ggml_format_name(hot_write,   "pyramidkv_c1_aux_hot_write_idxs");
    ggml_format_name(q_positions, "pyramidkv_c1_aux_q_positions");
    ggml_tensor * k_list = nullptr;
    ggml_tensor * k_list_len = nullptr;
    ggml_tensor * q_meta = nullptr;
    ggml_tensor * ext = nullptr;
    if (paged) {
        k_list     = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, list_stride, n_layers);
        k_list_len = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, len_stride, n_layers);
        q_meta     = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 2*n_ubatch);
        ext        = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, get_size() + n_ubatch);
        ggml_format_name(ext, "pyramidkv_c1_aux_ext");
        ggml_format_name(k_list,     "pyramidkv_c1_aux_k_list");
        ggml_format_name(k_list_len, "pyramidkv_c1_aux_k_list_len");
        ggml_format_name(q_meta,     "pyramidkv_c1_aux_q_meta");
    }
    ggml_backend_buffer_ptr buf { ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), kv_buffer_type) };
    if (!buf) {
        error = "PyramidKV C1 device input buffer allocation failed";
        return false;
    }
    ggml_backend_buffer_clear(buf.get(), 0);

    aux.ctx = std::move(ctx);
    aux.buf = std::move(buf);
    aux.k_positions    = k_positions;
    aux.hot_write_idxs = hot_write;
    aux.q_positions    = q_positions;
    aux.pos_stride = static_cast<uint32_t>(pos_stride);
    aux.hot_stride = static_cast<uint32_t>(hot_stride);
    aux.n_ubatch   = static_cast<uint32_t>(n_ubatch);
    aux.stage_pos.assign(pos_stride*n_layers, -1);
    aux.stage_hot.assign(hot_stride*n_layers, 0);
    aux.stage_q.assign(n_ubatch, 0);
    if (paged) {
        aux.k_list = k_list;
        aux.k_list_len = k_list_len;
        aux.q_meta = q_meta;
        aux.list_stride = static_cast<uint32_t>(list_stride);
        aux.len_stride  = static_cast<uint32_t>(len_stride);
        aux.heads_max   = static_cast<uint32_t>(heads_max);
        aux.stage_list.assign(list_stride*n_layers, -1);
        aux.stage_len.assign(len_stride*n_layers, 0);
        aux.stage_meta.assign(2*n_ubatch, -1);
        aux.staged_valid.assign(static_cast<size_t>(n_layers)*heads_max*n_seq_max, 0);
        aux.staged_seq.assign(static_cast<size_t>(n_layers)*n_seq_max, -1);
        aux.ext = ext;
        aux.stage_ext.assign(n_ubatch, -1);
    }
    quest_pending_resets.clear();
    quest_meta_dirty = true;
    if (pyramidkv_c1_quest()) {
        const uint32_t page = static_cast<uint32_t>(pyramidkv_c1_config.quest_page_size);
        const uint64_t n_cells = get_size();
        const uint64_t n_pages = (n_cells + page - 1)/page;
        int64_t head_dim = -1;
        for (size_t slot = 0; slot < pyramidkv_c1_layers.size(); ++slot) {
            const int64_t d = hparams.n_embd_head_k(layers[slot].il);
            if ((head_dim >= 0 && d != head_dim) || pyramidkv_c1_layers[slot].kv_heads != heads_max) {
                error = "PyramidKV Quest requires one head size and KV head count over the attention layers";
                return false;
            }
            head_dim = d;
        }
        if (head_dim != 128 && head_dim != 256) {
            error = "PyramidKV Quest requires a 128 or 256 key head size";
            return false;
        }
        ggml_init_params qparams = {
            /*.mem_size   =*/ 8*ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_context_ptr qctx { ggml_init(qparams) };
        if (!qctx) {
            error = "PyramidKV Quest context allocation failed";
            return false;
        }
        // 8-bit ordered bounds (ggml-quest8.h) halve the table; F16 for A/B
        static const bool f16_bounds = [] {
            const char * value = std::getenv("LLAMA_PYRAMIDKV_QUEST_F16");
            return value != nullptr && std::strcmp(value, "1") == 0;
        }();
        ggml_tensor * bounds = ggml_new_tensor_4d(qctx.get(), f16_bounds ? GGML_TYPE_F16 : GGML_TYPE_I8,
            2*head_dim, heads_max, n_pages, n_layers);
        ggml_tensor * page_seqs = ggml_new_tensor_1d(qctx.get(), GGML_TYPE_I32, n_pages);
        ggml_tensor * cell_meta = ggml_new_tensor_2d(qctx.get(), GGML_TYPE_I32, 2, n_cells);
        ggml_tensor * writes = ggml_new_tensor_2d(qctx.get(), GGML_TYPE_I32, 3, n_ubatch);
        ggml_tensor * resets = ggml_new_tensor_1d(qctx.get(), GGML_TYPE_I32, 1 + n_ubatch);
        ggml_tensor * seq_ids = ggml_new_tensor_1d(qctx.get(), GGML_TYPE_I32, n_seq_max);
        ggml_format_name(bounds, "pyramidkv_quest_bounds");
        ggml_format_name(page_seqs, "pyramidkv_quest_page_seqs");
        ggml_format_name(cell_meta, "pyramidkv_quest_cell_meta");
        ggml_format_name(writes, "pyramidkv_quest_writes");
        ggml_format_name(resets, "pyramidkv_quest_resets");
        ggml_format_name(seq_ids, "pyramidkv_quest_seq_ids");
        ggml_backend_buffer_ptr qbuf { ggml_backend_alloc_ctx_tensors_from_buft(qctx.get(), kv_buffer_type) };
        if (!qbuf) {
            error = "PyramidKV Quest buffer allocation failed";
            return false;
        }
        ggml_backend_buffer_clear(qbuf.get(), 0);
        LLAMA_LOG_INFO("%s: PyramidKV Quest %zu pages x %zu cells per step, %llu pages, bounds %.1f MiB\n",
            __func__, pyramidkv_c1_config.quest_pages, pyramidkv_c1_config.quest_page_size,
            (unsigned long long) n_pages, ggml_nbytes(bounds)/1048576.0);
        aux.quest_ctx = std::move(qctx);
        aux.quest_buf = std::move(qbuf);
        aux.quest_bounds = bounds;
        aux.quest_page_seqs = page_seqs;
        aux.quest_cell_meta = cell_meta;
        aux.quest_writes = writes;
        aux.quest_resets = resets;
        aux.quest_seq_ids = seq_ids;
        aux.stage_quest_writes.assign(3*n_ubatch, -1);
        aux.stage_quest_resets.assign(1 + n_ubatch, 0);
        aux.stage_quest_seq_ids.assign(n_seq_max, -1);
    }
    pyramidkv_c1_bind_aux_backend(backend);
    return true;
}

//
// Paged C1
//

bool llama_kv_cache::pyramidkv_c1_set_protected(llama_seq_id seq_id, std::vector<std::pair<int32_t, int32_t>> ranges) {
    if (seq_id < 0 || static_cast<size_t>(seq_id) >= pyramidkv_c1_protected.size()) {
        return false;
    }
    pyramidkv_c1_protected[seq_id] = std::move(ranges);
    return true;
}

bool llama_kv_cache::pyramidkv_c1_is_protected(llama_seq_id seq_id, llama_pos pos) const {
    if (seq_id < 0 || static_cast<size_t>(seq_id) >= pyramidkv_c1_protected.size()) {
        return false;
    }
    for (const auto & r : pyramidkv_c1_protected[seq_id]) {
        if (pos >= r.first && pos < r.second) {
            return true;
        }
    }
    return false;
}

void llama_kv_cache::pyramidkv_c1_paged_reset() {
    pyramidkv_c1_paged_compacted.assign(n_seq_max, 0);
    pyramidkv_c1_protected.assign(n_seq_max, {});
    pyramidkv_c1_paged_lists.assign(pyramidkv_c1_layers.size(), {});
    pyramidkv_c1_paged_dirty.assign(pyramidkv_c1_layers.size(), {});
    for (size_t layer_index = 0; layer_index < pyramidkv_c1_layers.size(); ++layer_index) {
        const auto & state = pyramidkv_c1_layers[layer_index];
        pyramidkv_c1_paged_lists[layer_index].assign(state.kv_heads,
            std::vector<std::vector<pyramidkv_c1_list_entry>>(n_seq_max));
        pyramidkv_c1_paged_dirty[layer_index].assign(state.kv_heads, std::vector<uint32_t>(n_seq_max, 0));
    }
}

bool llama_kv_cache::pyramidkv_c1_paged_seq_compacted(llama_seq_id seq_id) const {
    return pyramidkv_c1_paged() && seq_id >= 0 &&
        static_cast<size_t>(seq_id) < pyramidkv_c1_paged_compacted.size() &&
        pyramidkv_c1_paged_compacted[seq_id] != 0;
}

bool llama_kv_cache::pyramidkv_c1_local_prefill() const {
    static const bool enabled = [] {
        const char * value = std::getenv("LLAMA_PYRAMIDKV_LOCAL_PREFILL");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    return enabled && pyramidkv_c1_paged();
}

bool llama_kv_cache::pyramidkv_c1_paged_ubatch_ready(const llama_ubatch & ubatch) const {
    if (!pyramidkv_c1_paged()) {
        return false;
    }
    if (pyramidkv_c1_reserve_paged) {
        return true;
    }
    if (ubatch.n_seqs_unq == 0 || ubatch.seq_id_unq == nullptr) {
        return false;
    }
    bool selected = false;
    bool unselected = false;
    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
        if (pyramidkv_c1_paged_seq_compacted(ubatch.seq_id_unq[s])) {
            selected = true;
        } else {
            unselected = true;
        }
    }
    if (selected && unselected) {
        throw std::runtime_error("PyramidKV paged attention requires separate selected decode and unselected prefill ubatches");
    }
    return selected;
}

size_t llama_kv_cache::pyramidkv_c1_paged_list_size(llama_seq_id seq_id) const {
    size_t longest = 0;
    if (seq_id < 0 || static_cast<size_t>(seq_id) >= n_seq_max) {
        return 0;
    }
    for (const auto & layer : pyramidkv_c1_paged_lists) {
        for (const auto & head : layer) {
            longest = std::max(longest, head[seq_id].size());
        }
    }
    return longest;
}

void llama_kv_cache::pyramidkv_c1_paged_trim(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if (seq_id < 0 || static_cast<size_t>(seq_id) >= n_seq_max) {
        return;
    }
    for (size_t layer = 0; layer < pyramidkv_c1_paged_lists.size(); ++layer) {
        for (size_t head = 0; head < pyramidkv_c1_paged_lists[layer].size(); ++head) {
            auto & list = pyramidkv_c1_paged_lists[layer][head][seq_id];
            const auto first = std::find_if(list.begin(), list.end(),
                [&](const pyramidkv_c1_list_entry & e) { return e.pos >= p0 && e.pos < p1; });
            if (first == list.end()) {
                continue;
            }
            uint32_t & dirty = pyramidkv_c1_paged_dirty[layer][head][seq_id];
            dirty = std::min(dirty, static_cast<uint32_t>(first - list.begin()));
            list.erase(std::remove_if(first, list.end(),
                [&](const pyramidkv_c1_list_entry & e) { return e.pos >= p0 && e.pos < p1; }), list.end());
        }
    }
}

bool llama_kv_cache::pyramidkv_c1_paged_apply_selection(
        llama_seq_id seq_id,
        const std::vector<llama_pyramidkv_c1_layer_selection> & selections,
        std::string & error) {
    if (!pyramidkv_c1_paged()) {
        error = "PyramidKV C1 paged selection on a non-paged cache";
        return false;
    }
    if (seq_id < 0 || static_cast<size_t>(seq_id) >= n_seq_max) {
        error = "PyramidKV C1 paged selection names an unknown sequence";
        return false;
    }
    if (selections.empty()) {
        error = "PyramidKV C1 received no layer selections";
        return false;
    }
    auto & cells = v_cells[seq_to_stream[seq_id]];
    std::vector<const llama_pyramidkv_c1_layer_selection *> by_layer(layers.size(), nullptr);
    for (const auto & selection : selections) {
        const auto map_it = map_layer_ids.find(selection.il);
        if (map_it == map_layer_ids.end() || static_cast<size_t>(map_it->second) >= layers.size() ||
                by_layer[map_it->second] != nullptr) {
            error = "PyramidKV C1 selection was built for a different cache or layer set";
            return false;
        }
        by_layer[map_it->second] = &selection;
    }
    for (size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
        if (by_layer[layer_index] == nullptr) {
            error = "PyramidKV C1 has no score selection for a cached layer";
            return false;
        }
        if (by_layer[layer_index]->heads.size() != pyramidkv_c1_layers[layer_index].kv_heads) {
            error = "PyramidKV C1 selection does not match layer KV-head geometry";
            return false;
        }
    }

    // Per-head selections differ, so their union over 16 layers x 4 heads
    // covers nearly the whole prompt and would free nothing. Memory only
    // comes back when a cell is dropped for every layer and head: rank the
    // cells by how many (layer, head) selections keep them (ties: newer
    // first), keep max_capacity_prompt of them, and cut every head's list to
    // that set. Protected recent rows are in every selection and stay.
    std::vector<uint32_t> votes(cells.size(), 0);
    for (size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
        const auto & selection = *by_layer[layer_index];
        for (size_t head = 0; head < selection.heads.size(); ++head) {
            const auto & selected = selection.heads[head];
            if (selected.keep_cells.size() != selected.keep_positions.size()) {
                error = "PyramidKV C1 selection has an inconsistent head list";
                return false;
            }
            for (size_t i = 0; i < selected.keep_cells.size(); ++i) {
                const size_t logical = selected.keep_cells[i];
                if (logical >= cells.size() || cells.is_empty(static_cast<uint32_t>(logical)) ||
                        !cells.seq_has(static_cast<uint32_t>(logical), seq_id) ||
                        cells.pos_get(static_cast<uint32_t>(logical)) != selected.keep_positions[i]) {
                    error = "PyramidKV C1 selection does not preserve logical positions";
                    return false;
                }
                ++votes[logical];
            }
        }
    }
    std::vector<uint32_t> candidates;
    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (votes[i] != 0) {
            candidates.push_back(i);
        }
    }
    const size_t union_cap = std::max<size_t>(1, pyramidkv_c1_config.max_capacity_prompt) *
        std::max<size_t>(1, pyramidkv_c1_config.paged_union_factor);
    if (candidates.size() > union_cap) {
        std::nth_element(candidates.begin(), candidates.begin() + union_cap, candidates.end(),
            [&](uint32_t a, uint32_t b) {
                if (votes[a] != votes[b]) {
                    return votes[a] > votes[b];
                }
                return cells.pos_get(a) > cells.pos_get(b);
            });
        candidates.resize(union_cap);
    }
    std::vector<uint8_t> kept(cells.size(), 0);
    for (const uint32_t cell : candidates) {
        kept[cell] = 1;
    }
    // Protected ranges (images) stay whole and join every head's list.
    std::vector<uint32_t> protected_cells;
    if (static_cast<size_t>(seq_id) < pyramidkv_c1_protected.size() && !pyramidkv_c1_protected[seq_id].empty()) {
        const auto & ranges = pyramidkv_c1_protected[seq_id];
        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (cells.is_empty(i) || !cells.seq_has(i, seq_id) || cells.seq_count(i) != 1) {
                continue;
            }
            const llama_pos p = cells.pos_get(i);
            for (const auto & r : ranges) {
                if (p >= r.first && p < r.second) {
                    protected_cells.push_back(i);
                    kept[i] = 1;
                    break;
                }
            }
        }
    }
    for (size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
        const auto & selection = *by_layer[layer_index];
        for (size_t head = 0; head < selection.heads.size(); ++head) {
            const auto & selected = selection.heads[head];
            auto & list = pyramidkv_c1_paged_lists[layer_index][head][seq_id];
            list.clear();
            pyramidkv_c1_paged_dirty[layer_index][head][seq_id] = 0;
            list.reserve(selected.keep_cells.size() + protected_cells.size());
            for (size_t i = 0; i < selected.keep_cells.size(); ++i) {
                const size_t logical = selected.keep_cells[i];
                if (!kept[logical]) {
                    continue;
                }
                list.push_back({ static_cast<uint32_t>(logical), static_cast<int32_t>(selected.keep_positions[i]) });
            }
            for (const uint32_t cell : protected_cells) {
                // keep_cells is ordered by cell
                if (!std::binary_search(selected.keep_cells.begin(), selected.keep_cells.end(), static_cast<size_t>(cell))) {
                    list.push_back({ cell, static_cast<int32_t>(cells.pos_get(cell)) });
                }
            }
            // The selection orders by cell; a prompt that landed in cells freed
            // by an earlier sequence is not cell-ordered by position. The
            // lists are position-ordered (appends and trims rely on it).
            std::sort(list.begin(), list.end(), [](const pyramidkv_c1_list_entry & a, const pyramidkv_c1_list_entry & b) {
                return a.pos < b.pos;
            });
            if (list.empty() || list.size() > pyramidkv_c1_config.list_capacity) {
                error = "PyramidKV C1 selection left an empty or oversized head list";
                return false;
            }
        }
    }
    uint32_t freed = 0;
    auto & head_pos = v_heads[seq_to_stream[seq_id]];
    // Quest attends pages of the whole prompt: nothing is freed.
    for (uint32_t i = 0; i < cells.size() && !pyramidkv_c1_quest(); ++i) {
        if (kept[i] || cells.is_empty(i) || !cells.seq_has(i, seq_id) || cells.seq_count(i) != 1) {
            continue;
        }
        cells.rm(i);
        ++freed;
        head_pos = std::min(head_pos, i);
    }
    pyramidkv_c1_paged_compacted[seq_id] = 1;
    int32_t pos_min = std::numeric_limits<int32_t>::max(), pos_max = -1;
    size_t recent = 0;
    if (!pyramidkv_c1_paged_lists.empty() && !pyramidkv_c1_paged_lists[0].empty()) {
        for (const auto & e : pyramidkv_c1_paged_lists[0][0][seq_id]) {
            pos_min = std::min(pos_min, e.pos);
            pos_max = std::max(pos_max, e.pos);
        }
        for (const auto & e : pyramidkv_c1_paged_lists[0][0][seq_id]) {
            recent += e.pos + static_cast<int32_t>(pyramidkv_c1_config.recent_window) > pos_max;
        }
    }
    LLAMA_LOG_INFO("%s: PyramidKV paged selection seq=%d kept_cells=%zu freed_cells=%u longest_list=%zu pos=[%d,%d] recent_l0h0=%zu\n",
        __func__, (int) seq_id, static_cast<size_t>(std::count(kept.begin(), kept.end(), 1)), freed,
        pyramidkv_c1_paged_list_size(seq_id), pos_min, pos_max, recent);
    return true;
}

ggml_tensor * llama_kv_cache::get_k_paged(ggml_context * ctx, int32_t il) const {
    const int32_t ikv = map_layer_ids.at(il);
    const ggml_tensor * storage = layers[ikv].k;
    const auto & state = pyramidkv_c1_layers[ikv];
    const int64_t d = storage->ne[0] / state.kv_heads;
    return ggml_view_4d(ctx, const_cast<ggml_tensor *>(storage), d, storage->ne[1], state.kv_heads, 1,
        storage->nb[1], ggml_row_size(storage->type, d), storage->nb[2], 0);
}

ggml_tensor * llama_kv_cache::get_v_paged(ggml_context * ctx, int32_t il) const {
    const int32_t ikv = map_layer_ids.at(il);
    const ggml_tensor * storage = layers[ikv].v;
    const auto & state = pyramidkv_c1_layers[ikv];
    const int64_t d = storage->ne[0] / state.kv_heads;
    return ggml_view_4d(ctx, const_cast<ggml_tensor *>(storage), d, storage->ne[1], state.kv_heads, 1,
        storage->nb[1], ggml_row_size(storage->type, d), storage->nb[2], 0);
}

bool llama_kv_cache::pyramidkv_c1_reinitialize(std::string & error) {
    try {
    if (!pyramidkv_c1_hot_enabled || !pyramidkv_c1_compacted) {
        error = "PyramidKV C1 reset requires an active compacted layout";
        return false;
    }
    if (kv_buffer_type == nullptr || model.hparams.no_alloc || n_stream != 1 || layers.empty()) {
        error = "PyramidKV C1 reset requires one allocated explicit KV pool";
        return false;
    }
    if (pyramidkv_c1_initial_capacity == 0 || v_cells.empty() ||
            pyramidkv_c1_initial_capacity > v_cells[0].size() ||
            pyramidkv_c1_layers.size() != layers.size()) {
        error = "PyramidKV C1 reset has no valid initial physical capacity or layer state";
        return false;
    }

    const auto & config = pyramidkv_c1_config;

    size_t tensor_count = layers.size();
    if (tensor_count > std::numeric_limits<size_t>::max() / (4u * (1u + n_stream))) {
        error = "PyramidKV C1 reset tensor count overflows";
        return false;
    }
    tensor_count *= 4u * (1u + n_stream);
    if (turbo_rotation != nullptr && tensor_count > std::numeric_limits<size_t>::max() - 2u) {
        error = "PyramidKV C1 reset rotation tensor count overflows";
        return false;
    }
    tensor_count += turbo_rotation != nullptr ? 2u : 0u;
    if (tensor_count > std::numeric_limits<size_t>::max() / ggml_tensor_overhead()) {
        error = "PyramidKV C1 reset context size overflows";
        return false;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ tensor_count * ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr new_ctx(ggml_init(params));
    if (!new_ctx) {
        error = "PyramidKV C1 reset failed to create ggml context";
        return false;
    }

    std::vector<kv_layer> new_layers;
    new_layers.reserve(layers.size());
    for (const auto & old_layer : layers) {
        const uint32_t kv_heads = hparams.n_head_kv(old_layer.il);
        if (kv_heads == 0 || old_layer.k == nullptr || old_layer.v == nullptr ||
                old_layer.k_hot == nullptr || old_layer.v_hot == nullptr ||
                old_layer.k->ne[0] == 0 || old_layer.v->ne[0] == 0 ||
                old_layer.k->ne[0] > std::numeric_limits<uint32_t>::max() / kv_heads ||
                old_layer.v->ne[0] > std::numeric_limits<uint32_t>::max() / kv_heads ||
                old_layer.k_hot->ne[1] > std::numeric_limits<uint32_t>::max() ||
                old_layer.v_hot->ne[1] > std::numeric_limits<uint32_t>::max() ||
                old_layer.k_hot->ne[1] % kv_heads != 0 ||
                old_layer.v_hot->ne[1] % kv_heads != 0) {
            error = "PyramidKV C1 reset found invalid compacted layer geometry";
            return false;
        }

        const uint32_t cold_k_width = static_cast<uint32_t>(old_layer.k->ne[0]) * kv_heads;
        const uint32_t cold_v_width = static_cast<uint32_t>(old_layer.v->ne[0]) * kv_heads;
        const uint32_t hot_k_rows = static_cast<uint32_t>(old_layer.k_hot->ne[1]);
        const uint32_t hot_v_rows = static_cast<uint32_t>(old_layer.v_hot->ne[1]);
        ggml_tensor * k = ggml_new_tensor_3d(new_ctx.get(), old_layer.k->type,
                cold_k_width, pyramidkv_c1_initial_capacity, n_stream);
        ggml_tensor * v = ggml_new_tensor_3d(new_ctx.get(), old_layer.v->type,
                cold_v_width, pyramidkv_c1_initial_capacity, n_stream);
        ggml_tensor * k_hot = ggml_new_tensor_3d(new_ctx.get(), GGML_TYPE_F16,
                old_layer.k_hot->ne[0], hot_k_rows, n_stream);
        ggml_tensor * v_hot = ggml_new_tensor_3d(new_ctx.get(), GGML_TYPE_F16,
                old_layer.v_hot->ne[0], hot_v_rows, n_stream);
        if (k == nullptr || v == nullptr || k_hot == nullptr || v_hot == nullptr) {
            error = "PyramidKV C1 reset failed to create cache tensors";
            return false;
        }
        ggml_format_name(k, "cache_k_l%d", old_layer.il);
        ggml_format_name(v, "cache_v_l%d", old_layer.il);
        ggml_format_name(k_hot, "cache_k_hot_l%d", old_layer.il);
        ggml_format_name(v_hot, "cache_v_hot_l%d", old_layer.il);

        std::vector<ggml_tensor *> k_stream;
        std::vector<ggml_tensor *> v_stream;
        std::vector<ggml_tensor *> k_hot_stream;
        std::vector<ggml_tensor *> v_hot_stream;
        for (uint32_t stream = 0; stream < n_stream; ++stream) {
            k_stream.push_back(ggml_view_2d(new_ctx.get(), k, k->ne[0], k->ne[1],
                    k->nb[1], stream * k->nb[2]));
            v_stream.push_back(ggml_view_2d(new_ctx.get(), v, v->ne[0], v->ne[1],
                    v->nb[1], stream * v->nb[2]));
            k_hot_stream.push_back(ggml_view_2d(new_ctx.get(), k_hot, k_hot->ne[0], k_hot->ne[1],
                    k_hot->nb[1], stream * k_hot->nb[2]));
            v_hot_stream.push_back(ggml_view_2d(new_ctx.get(), v_hot, v_hot->ne[0], v_hot->ne[1],
                    v_hot->nb[1], stream * v_hot->nb[2]));
        }
        new_layers.push_back({ old_layer.il, k, v, k_stream, v_stream,
                k_hot, v_hot, k_hot_stream, v_hot_stream });
    }

    ggml_tensor * new_rotation = nullptr;
    ggml_tensor * new_rotation_inv = nullptr;
    if (turbo_rotation != nullptr) {
        new_rotation = ggml_new_tensor_2d(new_ctx.get(), GGML_TYPE_F32,
                GGML_TURBO4_QK, GGML_TURBO4_QK);
        new_rotation_inv = ggml_new_tensor_2d(new_ctx.get(), GGML_TYPE_F32,
                GGML_TURBO4_QK, GGML_TURBO4_QK);
        if (new_rotation == nullptr || new_rotation_inv == nullptr) {
            error = "PyramidKV C1 reset failed to create rotation tensors";
            return false;
        }
        ggml_format_name(new_rotation, "turbo4_rotation");
        ggml_format_name(new_rotation_inv, "turbo4_rotation_inv");
    }

    const size_t new_bytes = ggml_backend_alloc_ctx_tensors_from_buft_size(new_ctx.get(), kv_buffer_type);
    size_t old_bytes = total_size();
    for (const auto & [_, retired_buf] : pyramidkv_c1_retired_ctxs_bufs) {
        const size_t retired_bytes = ggml_backend_buffer_get_size(retired_buf.get());
        if (old_bytes > std::numeric_limits<size_t>::max() - retired_bytes) {
            error = "PyramidKV C1 reset old-buffer size overflows";
            return false;
        }
        old_bytes += retired_bytes;
    }
    size_t transition_bytes = old_bytes;
    const auto charge_reset = [&](size_t count, size_t element_size) {
        if (element_size != 0 && count > std::numeric_limits<size_t>::max() / element_size) {
            return false;
        }
        const size_t bytes = count * element_size;
        if (transition_bytes > config.transition_max_bytes ||
                bytes > config.transition_max_bytes - transition_bytes) {
            return false;
        }
        transition_bytes += bytes;
        return true;
    };
    // Match compaction's bounded observer/scratch allowance. Old and new map
    // payloads coexist until commit; count retained vector capacities too.
    bool reset_fits = charge_reset(new_bytes, 1) &&
        charge_reset(config.observer_max_bytes, 1) &&
        charge_reset(config.observer_max_bytes, 1) &&
        charge_reset(pyramidkv_c1_layers.capacity(), sizeof(pyramidkv_c1_layer_state)) &&
        charge_reset(layers.size(), sizeof(pyramidkv_c1_layer_state));
    for (const auto & state : pyramidkv_c1_layers) {
        reset_fits = reset_fits &&
            charge_reset(state.hot_heads.capacity(), sizeof(pyramidkv_c1_head_state)) &&
            charge_reset(state.cold_heads.capacity(), sizeof(pyramidkv_c1_head_state)) &&
            charge_reset(state.kv_heads, sizeof(pyramidkv_c1_head_state));
        for (const auto * maps : { &state.hot_heads, &state.cold_heads }) {
            for (const auto & map : *maps) {
                reset_fits = reset_fits &&
                    charge_reset(map.logical_to_physical.capacity(), sizeof(uint32_t)) &&
                    charge_reset(map.physical_to_logical.capacity(), sizeof(uint32_t));
            }
        }
        for (uint32_t head = 0; head < state.kv_heads; ++head) {
            reset_fits = reset_fits &&
                charge_reset(v_cells[0].size(), sizeof(uint32_t)) &&
                charge_reset(state.hot_row_capacity, sizeof(uint32_t));
        }
    }
    if (!reset_fits) {
        error = "PyramidKV C1 reset transition budget rejects old+new KV, observer, graph scratch, or map metadata";
        return false;
    }

    ggml_backend_buffer_ptr new_buf(
            ggml_backend_alloc_ctx_tensors_from_buft(new_ctx.get(), kv_buffer_type));
    if (!new_buf) {
        error = "PyramidKV C1 reset failed to allocate replacement KV pool";
        return false;
    }
    ggml_backend_buffer_clear(new_buf.get(), 0);

    if (new_rotation != nullptr) {
        std::array<float, GGML_TURBO4_QK*GGML_TURBO4_QK> rotation;
        std::array<float, GGML_TURBO4_QK*GGML_TURBO4_QK> rotation_inv;
        ggml_turbo4_rotation_matrix(rotation.data(), false);
        ggml_turbo4_rotation_matrix(rotation_inv.data(), true);
        ggml_backend_tensor_set(new_rotation, rotation.data(), 0, rotation.size()*sizeof(float));
        ggml_backend_tensor_set(new_rotation_inv, rotation_inv.data(), 0, rotation_inv.size()*sizeof(float));
    }

    std::vector<pyramidkv_c1_layer_state> new_states(layers.size());
    const size_t logical_size = v_cells.empty() ? 0 : v_cells[0].size();
    for (size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
        const auto & old_state = pyramidkv_c1_layers[layer_index];
        auto & state = new_states[layer_index];
        if (old_state.kv_heads == 0 || old_state.kv_heads != hparams.n_head_kv(layers[layer_index].il) ||
                old_state.hot_heads.size() != old_state.kv_heads ||
                old_state.hot_row_capacity == 0 ||
                old_state.hot_row_capacity > std::numeric_limits<uint32_t>::max() / old_state.kv_heads ||
                layers[layer_index].k_hot->ne[1] / old_state.kv_heads != old_state.hot_row_capacity ||
                layers[layer_index].v_hot->ne[1] / old_state.kv_heads != old_state.hot_row_capacity) {
            error = "PyramidKV C1 reset found incomplete hot-map state";
            return false;
        }
        state.compacted = false;
        state.kv_heads = old_state.kv_heads;
        state.hot_row_capacity = old_state.hot_row_capacity;
        state.hot_recent = old_state.hot_recent;
        state.hot_heads.resize(state.kv_heads);
        for (auto & head : state.hot_heads) {
            head.row_capacity = state.hot_row_capacity;
            head.next_row = 0;
            head.logical_to_physical.assign(logical_size, pyramidkv_c1_invalid_cell);
            head.physical_to_logical.assign(state.hot_row_capacity, pyramidkv_c1_invalid_cell);
        }
    }

    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> new_ctxs_bufs;
    new_ctxs_bufs.emplace_back(std::move(new_ctx), std::move(new_buf));

    // Commit only after all new objects exist. Keep old objects alive until the
    // context has reset scheduler and graph results. Reserve before moving any
    // old handle so a metadata allocation failure leaves the old layout intact.
    pyramidkv_c1_retired_ctxs_bufs.reserve(
            pyramidkv_c1_retired_ctxs_bufs.size() + ctxs_bufs.size());
    for (auto & entry : ctxs_bufs) {
        pyramidkv_c1_retired_ctxs_bufs.emplace_back(std::move(entry));
    }
    ctxs_bufs = std::move(new_ctxs_bufs);
    layers = std::move(new_layers);
    pyramidkv_c1_layers = std::move(new_states);
    turbo_rotation = new_rotation;
    turbo_rotation_inv = new_rotation_inv;
    sc_info = {};
    pyramidkv_c1_compacted = false;
    pyramidkv_c1_tokens_since_compact = 0;
    pyramidkv_c1_graph_reset_needed = true;
    if (!pyramidkv_c1_aux_rebuild(error)) {
        return false;
    }
    pyramidkv_c1_reset_failed_flag = false;
    pyramidkv_c1_reset_error.clear();
        return true;
    } catch (const std::exception & ex) {
        error = std::string("PyramidKV C1 reset transaction failed: ") + ex.what();
        return false;
    } catch (...) {
        error = "PyramidKV C1 reset transaction failed with an unknown exception";
        return false;
    }
}

void llama_kv_cache::pyramidkv_c1_graph_reset_complete() {
    if (!pyramidkv_c1_graph_reset_needed) {
        return;
    }
    pyramidkv_c1_retired_ctxs_bufs.clear();
    pyramidkv_c1_graph_reset_needed = false;
}

bool llama_kv_cache::pyramidkv_c1_reset_failed(std::string & error) const {
    if (!pyramidkv_c1_reset_failed_flag) {
        return false;
    }
    error = pyramidkv_c1_reset_error.empty()
        ? "PyramidKV C1 reset failed"
        : pyramidkv_c1_reset_error;
    return true;
}

void llama_kv_cache::pyramidkv_c1_fail_transition(const std::string & error) {
    if (pyramidkv_c1_hot_enabled) {
        pyramidkv_c1_reset_failed_flag = true;
        pyramidkv_c1_reset_error = error;
    }
    if (tq4_key_center_enabled_flag) {
        tq4_key_center_fail();
    }
}

void llama_kv_cache::clear(bool data) {
    quest_meta_dirty = true;
    if (pyramidkv_c1_hot_enabled && pyramidkv_c1_compacted) {
        std::string error;
        if (!pyramidkv_c1_reinitialize(error)) {
            pyramidkv_c1_reset_failed_flag = true;
            pyramidkv_c1_reset_error = std::move(error);
            pyramidkv_c1_graph_reset_needed = true;
            tq4_key_center_fail();
            LLAMA_LOG_ERROR("%s: PyramidKV C1 full clear cannot restore initial layout: %s\n",
                    __func__, pyramidkv_c1_reset_error.c_str());
            return;
        }
    }

    for (uint32_t s = 0; s < n_stream; ++s) {
        v_cells[s].reset();
        v_heads[s] = 0;
    }

    if (pyramidkv_c1_hot_enabled) {
        pyramidkv_c1_tokens_since_compact = 0;
        pyramidkv_c1_paged_reset();
        for (auto & state : pyramidkv_c1_layers) {
            state.compacted = false;
            state.cold_row_capacity = 0;
            state.cold_heads.clear();
            for (auto & head : state.hot_heads) {
                head.next_row = 0;
                head.logical_to_physical.assign(v_cells.empty() ? 0 : v_cells[0].size(),
                        pyramidkv_c1_invalid_cell);
                head.physical_to_logical.assign(state.hot_row_capacity,
                        pyramidkv_c1_invalid_cell);
            }
        }
        // Even data=false must invalidate graph tensor handles after a C1
        // layout replacement. The newly allocated initial buffer is empty.
        pyramidkv_c1_graph_reset_needed = true;
        pyramidkv_c1_reset_failed_flag = false;
        pyramidkv_c1_reset_error.clear();
    }

    if (data) {
        for (auto & [_, buf] : ctxs_bufs) {
            ggml_backend_buffer_clear(buf.get(), 0);

            if (turbo_rotation != nullptr && turbo_rotation->buffer == buf.get() && !model.hparams.no_alloc) {
                std::array<float, GGML_TURBO4_QK*GGML_TURBO4_QK> rotation;
                std::array<float, GGML_TURBO4_QK*GGML_TURBO4_QK> rotation_inv;
                ggml_turbo4_rotation_matrix(rotation.data(), false);
                ggml_turbo4_rotation_matrix(rotation_inv.data(), true);
                ggml_backend_tensor_set(turbo_rotation, rotation.data(), 0, rotation.size()*sizeof(float));
                ggml_backend_tensor_set(turbo_rotation_inv, rotation_inv.data(), 0, rotation_inv.size()*sizeof(float));
            }
        }
        if (tq4_key_anchor_buf) {
            ggml_backend_buffer_clear(tq4_key_anchor_buf.get(), 0);
        }
    }

    tq4_key_center_invalidate();
    tq4_key_center_failed_flag = false;
}

bool llama_kv_cache::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    quest_meta_dirty = true;
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return true;
    }

    // TODO: fix incosistent handling of `seq_id < 0` and `seq_id == -1` in the codebase [TAG_LLAMA_SEQ_ID_NEG]
    GGML_ASSERT(seq_id == -1 || (seq_id >= 0 && (size_t) seq_id < seq_to_stream.size()));

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    if (pyramidkv_c1_reset_failed_flag) {
        LLAMA_LOG_ERROR("%s: rejecting seq_rm while PyramidKV C1 reset is failed: %s\n",
                __func__, pyramidkv_c1_reset_error.c_str());
        return false;
    }

    if (pyramidkv_c1_paged() && p0 > 0) {
        const auto rollback_supported = [&](llama_seq_id s) {
            const auto & cells = v_cells[seq_to_stream[s]];
            const llama_pos last = cells.seq_pos_max(s);
            const llama_pos first = cells.seq_pos_min(s);
            if (!pyramidkv_c1_paged_seq_compacted(s) || p0 <= first || p0 > last || p1 <= last) {
                return true;
            }
            // Replacing the last token needs only R rows; deeper rollback needs the declared reserve.
            const uint64_t depth_limit = std::max<std::size_t>(1, pyramidkv_c1_config.rollback_headroom);
            return static_cast<uint64_t>(last - p0) + 1 <= depth_limit;
        };
        for (uint32_t s = 0; s < n_seq_max; ++s) {
            if ((seq_id < 0 || static_cast<uint32_t>(seq_id) == s) && !rollback_supported(s)) {
                LLAMA_LOG_ERROR("%s: paged tail removal exceeds rollback_headroom for sequence %u\n", __func__, s);
                return false;
            }
        }
    }

    // Paged C1: the arena keeps its layout; only this sequence's lists are
    // trimmed (a full removal also clears its selection state).
    if (pyramidkv_c1_paged()) {
        const llama_pos rm_p0 = p0;
        const llama_pos rm_p1 = p1;
        const auto trim_seq = [&](llama_seq_id s) {
            pyramidkv_c1_paged_trim(s, rm_p0, rm_p1);
            bool any = false;
            for (const auto & layer : pyramidkv_c1_paged_lists) {
                for (const auto & head : layer) {
                    if (s >= 0 && static_cast<size_t>(s) < head.size() && !head[s].empty()) {
                        any = true;
                    }
                }
            }
            if (!any && s >= 0 && static_cast<size_t>(s) < pyramidkv_c1_paged_compacted.size()) {
                pyramidkv_c1_paged_compacted[s] = 0;
            }
        };
        if (seq_id >= 0) {
            trim_seq(seq_id);
        } else {
            for (uint32_t s = 0; s < n_seq_max; ++s) {
                trim_seq(static_cast<llama_seq_id>(s));
            }
        }
    }

    // A complete removal must restore the constructor layout before the next
    // prompt. Partial removal keeps the compact layout, but still forces a
    // graph rebuild so prepare_batch_rows can clear stale per-head mappings.
    if (pyramidkv_c1_hot_enabled && pyramidkv_c1_compacted) {
        bool removes_all = true;
        for (uint32_t s = 0; s < n_stream && removes_all; ++s) {
            const auto & cells = v_cells[s];
            for (uint32_t i = 0; i < cells.size(); ++i) {
                if (cells.is_empty(i)) {
                    continue;
                }
                if (!cells.pos_in(i, p0, p1)) {
                    removes_all = false;
                    break;
                }
                if (seq_id >= 0 && !cells.seq_has(i, seq_id)) {
                    removes_all = false;
                    break;
                }
            }
        }

        if (removes_all) {
            std::string error;
            if (!pyramidkv_c1_reinitialize(error)) {
                pyramidkv_c1_reset_failed_flag = true;
                pyramidkv_c1_reset_error = std::move(error);
                pyramidkv_c1_graph_reset_needed = true;
                tq4_key_center_fail();
                LLAMA_LOG_ERROR("%s: PyramidKV C1 full seq_rm cannot restore initial layout: %s\n",
                        __func__, pyramidkv_c1_reset_error.c_str());
                return false;
            }
        }
    }

    if (seq_id >= 0) {
        auto & cells = v_cells[seq_to_stream[seq_id]];
        auto & head  = v_heads[seq_to_stream[seq_id]];

        uint32_t new_head = cells.size();

        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.pos_in(i, p0, p1)) {
                continue;
            }

            if (cells.seq_has(i, seq_id) && cells.seq_rm(i, seq_id)) {
                if (new_head == cells.size()) {
                    new_head = i;
                }
            }
        }

        // If we freed up a slot, set head to it so searching can start there.
        if (new_head != cells.size() && new_head < head) {
            head = new_head;
        }
    } else {
        // match any sequence
        for (uint32_t s = 0; s < n_stream; ++s) {
            auto & cells = v_cells[s];
            auto & head  = v_heads[s];

            uint32_t new_head = cells.size();

            for (uint32_t i = 0; i < cells.size(); ++i) {
                if (!cells.pos_in(i, p0, p1)) {
                    continue;
                }

                cells.rm(i);

                if (new_head == cells.size()) {
                    new_head = i;
                }
            }

            // If we freed up a slot, set head to it so searching can start there.
            if (new_head != cells.size() && new_head < head) {
                head = new_head;
            }
        }
    }

    if (pyramidkv_c1_hot_enabled) {
        // A partial removal (speculative rollback) leaves the cold pool and
        // the hot ring in place: prepare_batch_rows frees the hot and cold
        // rows of emptied cells before the next ubatch reuses them, and the
        // graph shape does not depend on cell occupancy. Removing rows never
        // brings the ring closer to overwriting an unpromoted row, so the
        // maintenance counter stands. Only a selection recorded before this
        // call is stale; a full removal was reinitialized above and already
        // requested its graph reset.
        if (pyramidkv_c1_paged()) {
            if (pyramidkv_c1_selection_stale_seqs.size() != n_seq_max) {
                pyramidkv_c1_selection_stale_seqs.assign(n_seq_max, 0);
            }
            if (seq_id >= 0 && static_cast<uint32_t>(seq_id) < n_seq_max) {
                pyramidkv_c1_selection_stale_seqs[seq_id] = 1;
            } else {
                std::fill(pyramidkv_c1_selection_stale_seqs.begin(), pyramidkv_c1_selection_stale_seqs.end(), 1);
            }
        } else {
            pyramidkv_c1_selection_stale = true;
        }
        if (!pyramidkv_c1_compacted) {
            pyramidkv_c1_tokens_since_compact = 0;
        }
    }
    if (tq4_key_center_enabled_flag && tq4_key_center_cache_empty()) {
        tq4_key_center_invalidate();
    }
    return true;
}

void llama_kv_cache::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    quest_meta_dirty = true;
    if ((pyramidkv_c1_hot_enabled || tq4_key_center_enabled_flag) && seq_id_src != seq_id_dst) {
        pyramidkv_c1_fail_transition(tq4_key_center_enabled_flag
            ? "TQ4 key center cannot copy cache rows to another sequence"
            : "PyramidKV C1 cannot copy cache rows to another sequence");
        return;
    }
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id_src >= 0 && (size_t) seq_id_src < seq_to_stream.size());
    GGML_ASSERT(seq_id_dst >= 0 && (size_t) seq_id_dst < seq_to_stream.size());

    const auto s0 = seq_to_stream[seq_id_src];
    const auto s1 = seq_to_stream[seq_id_dst];

    if (s0 == s1) {
        // since both sequences are in the same stream, no data copy is necessary
        // we just have to update the cells meta data

        auto & cells = v_cells[s0];

        if (seq_id_src == seq_id_dst) {
            return;
        }

        if (p0 < 0) {
            p0 = 0;
        }

        if (p1 < 0) {
            p1 = std::numeric_limits<llama_pos>::max();
        }

        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.pos_in(i, p0, p1)) {
                continue;
            }

            if (cells.seq_has(i, seq_id_src)) {
                cells.seq_add(i, seq_id_dst);
            }
        }

        return;
    }

    // cross-stream sequence copies require to copy the actual buffer data

    bool is_full = true;

    if (p0 > 0 && p0 + 1 < (int) get_size()) {
        is_full = false;
    }

    if (p1 > 0 && p1 + 1 < (int) get_size()) {
        is_full = false;
    }

    GGML_ASSERT(is_full && "seq_cp() is only supported for full KV buffers");

    // enqueue the copy operation - the buffer copy will be performed during the next update
    sc_info.ssrc.push_back(s0);
    sc_info.sdst.push_back(s1);

    v_cells[s1].reset();
    for (uint32_t i = 0; i < v_cells[s0].size(); ++i) {
        if (v_cells[s0].seq_has(i, seq_id_src)) {
            llama_pos pos   = v_cells[s0].pos_get(i);
            llama_pos shift = v_cells[s0].get_shift(i);

            llama_kv_cell_ext ext = v_cells[s0].ext_get(i);

            if (shift != 0) {
                pos -= shift;
                assert(pos >= 0);
            }

            v_cells[s1].pos_set(i, pos);
            v_cells[s1].seq_add(i, seq_id_dst);

            if (shift != 0) {
                v_cells[s1].pos_add(i, shift);
            }

            v_cells[s1].ext_set(i, ext);
        }
    }

    v_heads[s1] = v_heads[s0];

    //for (uint32_t s = 0; s < n_stream; ++s) {
    //    LLAMA_LOG_WARN("%s: seq %d: min = %d, max = %d\n", __func__, s, v_cells[s].seq_pos_min(s), v_cells[s].seq_pos_max(s));
    //}
}

void llama_kv_cache::seq_keep(llama_seq_id seq_id) {
    quest_meta_dirty = true;
    if ((pyramidkv_c1_hot_enabled || tq4_key_center_enabled_flag) && seq_id != 0) {
        pyramidkv_c1_fail_transition(tq4_key_center_enabled_flag
            ? "TQ4 key center supports only sequence 0"
            : "PyramidKV C1 supports only sequence 0");
        return;
    }
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    auto & cells = v_cells[seq_to_stream[seq_id]];
    auto & head  = v_heads[seq_to_stream[seq_id]];

    uint32_t new_head = cells.size();

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (cells.seq_keep(i, seq_id)) {
            if (new_head == cells.size()) {
                new_head = i;
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != cells.size() && new_head < head) {
        head = new_head;
    }
    if (tq4_key_center_enabled_flag && tq4_key_center_cache_empty()) {
        tq4_key_center_invalidate();
    }
}

void llama_kv_cache::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    quest_meta_dirty = true;
    const llama_pos range_begin = std::max<llama_pos>(0, p0);
    const llama_pos range_end = p1 < 0 ? std::numeric_limits<llama_pos>::max() : p1;
    if ((pyramidkv_c1_hot_enabled || tq4_key_center_enabled_flag) && shift != 0 && range_begin != range_end) {
        pyramidkv_c1_fail_transition(tq4_key_center_enabled_flag
            ? "TQ4 key center cannot shift original cache positions"
            : "PyramidKV C1 cannot shift original cache positions");
        return;
    }
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());
    GGML_ASSERT(hparams.n_pos_per_embd() == 1 && "seq_add() is only supported for n_pos_per_embd() == 1");

    auto & cells = v_cells[seq_to_stream[seq_id]];
    auto & head  = v_heads[seq_to_stream[seq_id]];

    if (shift == 0) {
        return;
    }

    uint32_t new_head = cells.size();

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over all cells.
    if (p0 == p1) {
        return;
    }

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.pos_in(i, p0, p1)) {
            continue;
        }

        if (cells.seq_has(i, seq_id)) {
            if (cells.pos_add(i, shift)) {
                if (new_head == cells.size()) {
                    new_head = i;
                }
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    // Otherwise we just start the next search from the beginning.
    head = new_head != cells.size() ? new_head : 0;
}

void llama_kv_cache::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    quest_meta_dirty = true;
    const llama_pos range_begin = std::max<llama_pos>(0, p0);
    const llama_pos range_end = p1 < 0 ? std::numeric_limits<llama_pos>::max() : p1;
    if ((pyramidkv_c1_hot_enabled || tq4_key_center_enabled_flag) && d != 1 && range_begin != range_end) {
        pyramidkv_c1_fail_transition(tq4_key_center_enabled_flag
            ? "TQ4 key center cannot divide original cache positions"
            : "PyramidKV C1 cannot divide original cache positions");
        return;
    }
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());
    GGML_ASSERT(hparams.n_pos_per_embd() == 1 && "seq_div() is only supported for n_pos_per_embd() == 1");

    auto & cells = v_cells[seq_to_stream[seq_id]];

    if (d == 1) {
        return;
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over the cache.
    if (p0 == p1) {
        return;
    }

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.pos_in(i, p0, p1)) {
            continue;
        }

        if (cells.seq_has(i, seq_id)) {
            cells.pos_div(i, d);
        }
    }
}

llama_pos llama_kv_cache::seq_pos_min(llama_seq_id seq_id) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return other->seq_pos_min(seq_id);
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    const auto & cells = v_cells[seq_to_stream[seq_id]];

    return cells.seq_pos_min(seq_id);
}

int64_t llama_kv_cache::seq_n_cells(llama_seq_id seq_id) const {
    if (other) {
        return other->seq_n_cells(seq_id);
    }
    if (seq_id < 0 || (size_t) seq_id >= seq_to_stream.size()) {
        return -1;
    }
    const auto & cells = v_cells[seq_to_stream[seq_id]];
    int64_t n = 0;
    for (uint32_t i = 0; i < cells.used_max_p1(); ++i) {
        n += !cells.is_empty(i) && cells.seq_has(i, seq_id);
    }
    return n;
}

int32_t llama_kv_cache::seq_positions(llama_seq_id seq_id, llama_pos * pos, int32_t cap) const {
    if (other) {
        return other->seq_positions(seq_id, pos, cap);
    }
    if (seq_id < 0 || (size_t) seq_id >= seq_to_stream.size() || cap < 0) {
        return -1;
    }
    const auto & cells = v_cells[seq_to_stream[seq_id]];
    std::vector<llama_pos> found;
    for (uint32_t i = 0; i < cells.used_max_p1(); ++i) {
        if (!cells.is_empty(i) && cells.seq_has(i, seq_id)) {
            found.push_back(cells.pos_get(i));
        }
    }
    std::sort(found.begin(), found.end());
    const int32_t n = (int32_t) std::min<size_t>(found.size(), (size_t) cap);
    if (pos != nullptr) {
        std::copy(found.begin(), found.begin() + n, pos);
    }
    return (int32_t) found.size();
}

bool llama_kv_cache::seq_keep_positions(llama_seq_id seq_id, const llama_pos * pos, int32_t n) {
    if (other) {
        return true;
    }
    if (seq_id < 0 || (size_t) seq_id >= seq_to_stream.size() || (n > 0 && pos == nullptr)) {
        return false;
    }
    // The bounded C1 layouts track cells through their own maps/lists; only a
    // plain cache (draft contexts) takes a cell-wise removal here.
    if (pyramidkv_c1_hot_enabled || pyramidkv_c1_compacted || pyramidkv_c1_paged()) {
        return false;
    }
    auto & cells = v_cells[seq_to_stream[seq_id]];
    auto & head  = v_heads[seq_to_stream[seq_id]];
    uint32_t new_head = cells.size();
    for (uint32_t i = 0; i < cells.used_max_p1(); ++i) {
        if (cells.is_empty(i) || !cells.seq_has(i, seq_id)) {
            continue;
        }
        if (std::binary_search(pos, pos + n, cells.pos_get(i))) {
            continue;
        }
        if (cells.seq_rm(i, seq_id) && new_head == cells.size()) {
            new_head = i;
        }
    }
    if (new_head != cells.size() && new_head < head) {
        head = new_head;
    }
    return true;
}

llama_pos llama_kv_cache::seq_pos_max(llama_seq_id seq_id) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return other->seq_pos_max(seq_id);
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    const auto & cells = v_cells[seq_to_stream[seq_id]];

    return cells.seq_pos_max(seq_id);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> ret;
    const auto add_buffers = [&](const auto & buffers) {
        for (const auto & [ctx, buf] : buffers) {
            ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf.get());

            if (hparams.no_alloc) {
                GGML_ASSERT(ggml_backend_buffer_get_base(buf.get()) == nullptr);
                ret[buft] += ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), buft);
            } else {
                // Retired C1 buffers remain physically resident until the
                // scheduler reset completes, so include them in the transient
                // breakdown while they are still owned.
                ret[buft] += ggml_backend_buffer_get_size(buf.get());
            }
        }
    };
    add_buffers(ctxs_bufs);
    add_buffers(pyramidkv_c1_retired_ctxs_bufs);
    if (tq4_key_anchor_buf) {
        const ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(tq4_key_anchor_buf.get());
        ret[buft] += ggml_backend_buffer_get_size(tq4_key_anchor_buf.get());
    }
    return ret;
}

llama_memory_context_ptr llama_kv_cache::init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) {
    if (pyramidkv_c1_reset_failed_flag || tq4_key_center_failed_flag) {
        LLAMA_LOG_ERROR("%s: refusing batch while KV transition is failed: %s\n",
                __func__, pyramidkv_c1_reset_error.empty()
                    ? "TQ4 key-center capture failed" : pyramidkv_c1_reset_error.c_str());
        return std::make_unique<llama_kv_cache_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    }

    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        if (pyramidkv_c1_paged()) {
            ubatches = pyramidkv_c1_split_batch(balloc, n_ubatch, 0, embd_all);
        } else {
            while (true) {
                auto ubatch = n_stream == 1 ? balloc.split_simple(n_ubatch) : balloc.split_equal(n_ubatch, true, 0);

                if (ubatch.n_tokens == 0) {
                    break;
                }

                ubatches.push_back(std::move(ubatch)); // NOLINT
            }
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        auto sinfos = prepare(ubatches);
        if (sinfos.empty()) {
            break;
        }

        return std::make_unique<llama_kv_cache_context>(
                this, std::move(sinfos), std::move(ubatches));
    } while (false);

    return std::make_unique<llama_kv_cache_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

std::vector<llama_ubatch> llama_kv_cache::pyramidkv_c1_split_batch(
        llama_batch_allocr & balloc, uint32_t n_ubatch, uint32_t n_keep_tail, bool single_seq) const {
    GGML_ASSERT(pyramidkv_c1_paged());
    const auto window = pyramidkv_c1_config.observation_window;
    if (window > n_ubatch || n_keep_tail > n_ubatch) {
        LLAMA_LOG_ERROR("%s: PyramidKV observer and recurrent tails must fit in n_ubatch\n", __func__);
        return {};
    }
    const auto & batch = balloc.get_batch();
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        if (batch.n_seq_id[i] != 1) {
            LLAMA_LOG_ERROR("%s: PyramidKV paged batches require one sequence per token\n", __func__);
            return {};
        }
    }
    const uint32_t n_prefill_tail = std::max(n_keep_tail, static_cast<uint32_t>(window));
    std::vector<llama_ubatch> ubatches;
    while (true) {
        // Selected sequences share paged attention; each prefill keeps its own observer window.
        auto ubatch = balloc.split_equal(n_ubatch, false, n_keep_tail,
            [&](llama_seq_id seq) { return pyramidkv_c1_paged_seq_compacted(seq); }, single_seq);
        if (ubatch.n_tokens == 0) {
            ubatch = balloc.split_equal(n_ubatch, false, n_prefill_tail,
                [&](llama_seq_id seq) { return !pyramidkv_c1_paged_seq_compacted(seq); }, true);
        }
        if (ubatch.n_tokens == 0) {
            break;
        }
        ubatches.push_back(std::move(ubatch));
    }
    return ubatches;
}

llama_memory_context_ptr llama_kv_cache::init_full() {
    return std::make_unique<llama_kv_cache_context>(this);
}

llama_memory_context_ptr llama_kv_cache::init_update(llama_context * lctx, bool optimize) {
    GGML_UNUSED(optimize);

    bool do_shift = get_has_shift();

    return std::make_unique<llama_kv_cache_context>(this, lctx, do_shift, std::move(sc_info));
}

llama_kv_cache::slot_info_vec_t llama_kv_cache::prepare(const std::vector<llama_ubatch> & ubatches) {
    llama_kv_cache::slot_info_vec_t res;

    struct state_t {
        slot_info sinfo; // slot info for the ubatch

        std::vector<uint32_t> v_heads_old; // old positions of the heads, before placing the ubatch

        std::vector<llama_kv_cells> v_cells; // copy of the old cells, before placing the ubatch
    };

    // remember the old state of the cells so we can restore it in the end
    std::vector<state_t> states;

    bool success = true;

    for (const auto & ubatch : ubatches) {
        // only find a suitable slot for the ubatch. don't modify the cells yet
        const auto sinfo_new = find_slot(ubatch, false);
        if (sinfo_new.empty()) {
            success = false;
            break;
        }

        // remember the position that we found
        res.push_back(sinfo_new);

        // store the old state of the cells in the recovery stack
        {
            state_t state = { sinfo_new, v_heads, {} };

            for (uint32_t s = 0; s < sinfo_new.n_stream(); ++s) {
                auto & cells = v_cells[sinfo_new.strm[s]];

                state.v_cells.push_back(cells.cp(sinfo_new.idxs[s]));
            }

            states.push_back(std::move(state));
        }

        // now emplace the ubatch
        apply_ubatch(sinfo_new, ubatch);
    }

    GGML_ASSERT(!states.empty() || !success);

    // iterate backwards and restore the cells to their original state
    for (auto it = states.rbegin(); it != states.rend(); ++it) {
        const auto & sinfo = it->sinfo;

        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            auto & cells = v_cells[sinfo.strm[s]];
            auto & head  = v_heads[sinfo.strm[s]];

            cells.set(sinfo.idxs[s], it->v_cells[s]);
            head = it->v_heads_old[s];
        }
    }

    if (!success) {
        return {};
    }

    return res;
}

bool llama_kv_cache::update(llama_context * lctx, bool do_shift, const stream_copy_info & sc_info) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return true;
    }

    bool updated = false;

    auto * sched = lctx->get_sched();

    if (!sc_info.empty()) {
        assert(n_stream > 1 && "stream copy should never happen with a single stream");

        llama_synchronize(lctx);

        const size_t n_copy = sc_info.ssrc.size();

        for (size_t i = 0; i < n_copy; ++i) {
            const auto ssrc = sc_info.ssrc[i];
            const auto sdst = sc_info.sdst[i];

            assert(ssrc < n_stream);
            assert(sdst < n_stream);

            LLAMA_LOG_DEBUG("%s: copying KV buffer: stream %d to stream %d\n", __func__, ssrc, sdst);

            assert(ssrc != sdst);

            for (uint32_t il = 0; il < layers.size(); ++il) {
                const auto & layer = layers[il];

                ggml_backend_tensor_copy(layer.k_stream[ssrc], layer.k_stream[sdst]);

                if (layer.v_stream[ssrc]) {
                    ggml_backend_tensor_copy(layer.v_stream[ssrc], layer.v_stream[sdst]);
                }
            }
        }
    }

    if (do_shift) {
        if (!get_can_shift()) {
            if (llama_kv_type_is_turbo(type_k())) {
                throw std::runtime_error("TurboQuant4 KV cache does not support K-shift");
            }
            GGML_ABORT("The current KV cache / model configuration does not support K-shift");
        }

        LLAMA_LOG_DEBUG("%s: applying K-shift\n", __func__);

        // apply K-shift if needed
        if (hparams.rope_type != LLAMA_ROPE_TYPE_NONE) {
            ggml_backend_sched_reset(sched);

            auto * res = lctx->get_gf_res_reserve();

            res->reset();

            auto * gf = build_graph_shift(res, lctx);
            if (!ggml_backend_sched_alloc_graph(sched, gf)) {
                LLAMA_LOG_ERROR("%s: failed to allocate compute graph for K-shift\n", __func__);
                return updated;
            }

            res->set_inputs(nullptr);

            if (lctx->graph_compute(gf, false) != GGML_STATUS_SUCCESS) {
                LLAMA_LOG_ERROR("%s: failed to compute K-shift\n", __func__);
                return updated;
            }

            updated = true;
        }

        for (uint32_t s = 0; s < n_stream; ++s) {
            auto & cells = v_cells[s];

            cells.reset_shift();
        }
    }

    return updated;
}

llama_kv_cache::slot_info llama_kv_cache::find_slot(const llama_ubatch & ubatch, bool cont) const {

    if (debug > 0) {
        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
            const auto seq_id = ubatch.seq_id_unq[s];
            const auto stream_id = seq_to_stream[seq_id];
            const auto & cells = v_cells[stream_id];
            const uint32_t head_cur = v_heads[stream_id];

            LLAMA_LOG_DEBUG("%s: stream[%d], n = %5d, used = %5d, head = %5d, size = %5d, n_swa = %5d\n",
                    __func__, stream_id, cells.used_max_p1(), cells.get_used(), head_cur, get_size(), n_swa);

            if ((debug == 2 && n_swa > 0) || debug > 2) {
                std::string ss;
                for (uint32_t i = 0; i < cells.size(); ++i) {
                    if (cells.is_empty(i)) {
                        ss += '.';
                    } else {
                        assert(cells.seq_count(i) >= 1);

                        if (cells.seq_count(i) == 1) {
                            ss += std::to_string(cells.seq_get(i));
                        } else {
                            ss += 'M';
                        }
                    }
                    if (i%256 == 255) {
                        ss += " *";
                        ss += '\n';
                    }
                }
                LLAMA_LOG_DEBUG("\n%s\n", ss.c_str());
            }

            if ((debug == 2 && n_swa > 0) || debug > 2) {
                std::string ss;
                for (uint32_t i = 0; i < cells.size(); ++i) {
                    std::string cur;
                    if (cells.is_empty(i)) {
                        cur = '.';
                    } else {
                        cur = std::to_string(cells.pos_get(i));
                    }
                    const int n = cur.size();
                    for (int j = 0; j < 5 - n; ++j) {
                        cur += ' ';
                    }
                    ss += cur;
                    if (i%256 == 255) {
                        ss += " *";
                    }
                    if (i%64 == 63) {
                        ss += '\n';
                    }
                }
                LLAMA_LOG_DEBUG("\n%s\n", ss.c_str());
            }

            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                if (cells.seq_pos_min(s) < 0) {
                    continue;
                }

                LLAMA_LOG_DEBUG("%s: stream[%d] min[%d] = %5d, max[%d] = %5d\n", __func__, stream_id, s, cells.seq_pos_min(s), s, cells.seq_pos_max(s));
            }
        }
    }

    uint32_t n_tokens = ubatch.n_tokens;
    uint32_t n_seqs   = 1;

    if (n_stream > 1) {
        GGML_ASSERT(n_tokens % ubatch.n_seqs_unq == 0);

        n_seqs   = ubatch.n_seqs_unq;
        n_tokens = n_tokens / n_seqs;
    }

    slot_info res = {
        /*.s0   =*/ LLAMA_MAX_SEQ,
        /*.s1   =*/ 0,
        /*.strm =*/ { },
        /*.idxs =*/ { },
    };

    res.resize(n_seqs);

    for (uint32_t s = 0; s < n_seqs; ++s) {
        const auto seq_id = ubatch.seq_id_unq[s];

        if (n_stream > 1) {
            GGML_ASSERT(ubatch.n_seq_id[s*n_tokens]    == 1);
            GGML_ASSERT(ubatch.seq_id  [s*n_tokens][0] == seq_id);
        }

        res.s0 = std::min<uint32_t>(res.s0, seq_to_stream[seq_id]);
        res.s1 = std::max<uint32_t>(res.s1, seq_to_stream[seq_id]);

        res.strm[s] = seq_to_stream[seq_id];
        res.idxs[s].reserve(n_tokens);

        const auto & cells = v_cells[seq_to_stream[seq_id]];

        uint32_t head_cur = v_heads[seq_to_stream[seq_id]];

        // if we have enough unused cells before the current head ->
        //   better to start searching from the beginning of the cache, hoping to fill it
        if (head_cur > cells.get_used() + 2*n_tokens) {
            head_cur = 0;
        }

        const uint32_t search_limit = pyramidkv_c1_hot_enabled && !pyramidkv_c1_compacted
            ? std::min(cells.size(), pyramidkv_c1_initial_capacity) : cells.size();
        if (search_limit == 0 || n_tokens > search_limit) {
            LLAMA_LOG_ERROR("%s: n_tokens = %d > size = %u\n", __func__, n_tokens, cells.size());
            return { };
        }

        uint32_t n_tested = 0;

        // for continuous slots, we test that all tokens in the ubatch fit, starting from the current head
        // for non-continuous slots, we test the tokens one by one
        const uint32_t n_test = cont ? n_tokens : 1;

        while (true) {
            if (head_cur + n_test > search_limit) {
                n_tested += search_limit - head_cur;
                head_cur = 0;
                continue;
            }

            for (uint32_t i = 0; i < n_test; i++) {
                const auto idx = head_cur;

                head_cur++;
                n_tested++;

                //const llama_pos    pos    = ubatch.pos[i];
                //const llama_seq_id seq_id = ubatch.seq_id[i][0];

                // can we use this cell? either:
                //  - the cell is empty
                //  - the cell is occupied only by one sequence:
                //    - (disabled) mask causally, if the sequence is the same as the one we are inserting
                //    - mask SWA, using current max pos for that sequence in the cache
                //                always insert in the cell with minimum pos
                bool can_use = cells.is_empty(idx);

                if (!can_use && cells.seq_count(idx) == 1) {
                    const llama_pos pos_cell = cells.pos_get(idx);

                    // (disabled) causal mask
                    // note: it's better to purge any "future" tokens beforehand
                    //if (cells.seq_has(idx, seq_id)) {
                    //    can_use = pos_cell >= pos;
                    //}

                    if (!can_use) {
                        const llama_seq_id seq_id_cell = cells.seq_get(idx);

                        // SWA mask
                        if (llama_hparams::is_masked_swa(n_swa, swa_type, pos_cell, cells.seq_pos_max(seq_id_cell) + 1)) {
                            can_use = true;
                        }
                    }
                }

                if (can_use) {
                    res.idxs[s].push_back(idx);
                } else {
                    if (cont) {
                        break;
                    }
                }
            }

            if (res.idxs[s].size() == n_tokens) {
                break;
            }

            if (cont) {
                res.idxs[s].clear();
            }

            if (n_tested >= search_limit) {
                //LLAMA_LOG_ERROR("%s: failed to find a slot for %d tokens\n", __func__, n_tokens);
                return { };
            }
        }

        // we didn't find a suitable slot - return empty result
        if (res.idxs[s].size() < n_tokens) {
            return { };
        }
    }

    assert(res.s1 >= res.s0);

    return res;
}

void llama_kv_cache::apply_ubatch(const slot_info & sinfo, const llama_ubatch & ubatch) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    // keep track of the max sequence position that we would overwrite with this ubatch
    // for non-SWA cache, this would be always empty
    llama_seq_id seq_pos_max_rm[LLAMA_MAX_SEQ];
    for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
        seq_pos_max_rm[s] = -1;
    }

    assert(ubatch.n_tokens == sinfo.n_stream()*sinfo.size());

    if (pyramidkv_c1_quest() && sinfo.n_stream() == 1) {
        const auto & cells = v_cells[sinfo.strm[0]];
        const uint32_t page = static_cast<uint32_t>(pyramidkv_c1_config.quest_page_size);
        int64_t last_page = -1;
        for (uint32_t ii = 0; ii < sinfo.size(); ++ii) {
            const int64_t p = sinfo.idxs[0][ii]/page;
            if (p == last_page || std::find(quest_pending_resets.begin(), quest_pending_resets.end(), (int32_t) p) != quest_pending_resets.end()) {
                last_page = p;
                continue;
            }
            last_page = p;
            bool empty = true;
            for (uint32_t c = (uint32_t) p*page; c < std::min<uint32_t>(((uint32_t) p + 1)*page, cells.size()) && empty; ++c) {
                empty = cells.is_empty(c);
            }
            if (empty) {
                quest_pending_resets.push_back((int32_t) p);
            }
        }
    }

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        for (uint32_t ii = 0; ii < sinfo.size(); ++ii) {
            const uint32_t i = s*sinfo.size() + ii;

            auto & cells = v_cells[sinfo.strm[s]];

            const auto idx = sinfo.idxs[s][ii];

            if (!cells.is_empty(idx)) {
                quest_meta_dirty = true;
                assert(cells.seq_count(idx) == 1);

                const llama_seq_id seq_id = cells.seq_get(idx);
                const llama_pos    pos    = cells.pos_get(idx);

                seq_pos_max_rm[seq_id] = std::max(seq_pos_max_rm[seq_id], pos);

                cells.rm(idx);
            }

            cells.pos_set(idx, ubatch.pos[i]);

            if (ubatch.is_pos_2d() || ubatch.token) {
                llama_kv_cell_ext ext;

                if (ubatch.is_pos_2d()) {
                    ext.x = ubatch.pos[i + ubatch.n_tokens*2];
                    ext.y = ubatch.pos[i + ubatch.n_tokens];
                }

                if (ubatch.token) {
                    ext.tok = ubatch.token[i];
                }

                cells.ext_set(idx, ext);
            }

            for (int32_t s = 0; s < ubatch.n_seq_id[i]; s++) {
                cells.seq_add(idx, ubatch.seq_id[i][s]);
            }
        }
    }

    // note: we want to preserve the invariant that all positions between [pos_min, pos_max] for each sequence
    //       will be present in the cache. so we have to purge any position which is less than those we would overwrite
    //       ref: https://github.com/ggml-org/llama.cpp/pull/13746#issuecomment-2916057092
    for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
        if (seq_pos_max_rm[s] == -1) {
            continue;
        }

        GGML_ASSERT(s < seq_to_stream.size());

        auto & cells = v_cells[seq_to_stream[s]];

        if (cells.seq_pos_min(s) <= seq_pos_max_rm[s]) {
            LLAMA_LOG_DEBUG("%s: purging positions [%d, %d] of sequence %d from KV cache\n",
                    __func__, cells.seq_pos_min(s), seq_pos_max_rm[s], s);

            seq_rm(s, cells.seq_pos_min(s), seq_pos_max_rm[s] + 1);
        }
    }

    // move the head at the end of the slot
    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        auto & head = v_heads[sinfo.strm[s]];

        head = sinfo.idxs[s].back() + 1;
    }
}

bool llama_kv_cache::get_can_shift() const {
    if (tq4_key_center_enabled_flag || llama_kv_type_is_turbo(type_k())) {
        return false;
    }

    // Step35 uses per-layer RoPE dims; K-shift assumes a single global n_rot.
    if (model.arch == LLM_ARCH_STEP35) {
        return false;
    }
    if (hparams.n_pos_per_embd() > 1) {
        return false;
    }
    return true;
}

uint32_t llama_kv_cache::get_size() const {
    const auto & cells = v_cells[seq_to_stream[0]];

    return cells.size();
}

uint32_t llama_kv_cache::get_n_stream() const {
    return n_stream;
}

bool llama_kv_cache::get_has_shift() const {
    bool result = false;

    for (uint32_t s = 0; s < n_stream; ++s) {
        result |= v_cells[s].get_has_shift();
    }

    return result;
}

ggml_type llama_kv_cache::type_k() const {
    return layers[0].k->type;
}

ggml_type llama_kv_cache::type_v() const {
    return layers[0].v->type;
}

ggml_tensor * llama_kv_cache::get_turbo_rotation() const {
    return turbo_rotation;
}

ggml_tensor * llama_kv_cache::get_turbo_rotation_inv() const {
    return turbo_rotation_inv;
}

std::vector<uint32_t> llama_kv_cache::get_layer_ids() const {
    std::vector<uint32_t> res;
    res.reserve(layers.size());

    for (const auto & layer : layers) {
        res.push_back(layer.il);
    }

    return res;
}

ggml_tensor * llama_kv_cache::get_k_storage(int32_t il) const {
    const int32_t ikv = map_layer_ids.at(il);

    return layers[ikv].k;
}

const llama_kv_cells & llama_kv_cache::get_cells(llama_seq_id seq_id) const {
    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    return v_cells[seq_to_stream[seq_id]];
}

bool llama_kv_cache::pyramidkv_c1_supported(std::string & error) const {
    if (tq4_key_center_enabled_flag &&
            (!tq4_key_anchor_buf || tq4_key_anchors.size() != layers.size())) {
        error = "TQ4 key center has no complete persistent anchor allocation";
        return false;
    }
    if (kv_buffer_type == nullptr) {
        error = "PyramidKV C1 requires an explicit kv_buffer_type pool";
        return false;
    }
    if (model.hparams.no_alloc) {
        error = "PyramidKV C1 cannot compact a no_alloc cache";
        return false;
    }
    if (other != nullptr) {
        error = "PyramidKV C1 does not support a shared KV cache";
        return false;
    }
    if (n_stream != 1 || (n_seq_max != 1 && !pyramidkv_c1_config.paged)) {
        error = "PyramidKV C1 requires one sequence and one cache stream";
        return false;
    }
    if (n_swa != 0 || swa_type != LLAMA_SWA_TYPE_NONE) {
        error = "PyramidKV C1 rejects SWA caches";
        return false;
    }
    if (v_trans) {
        error = "PyramidKV C1 requires non-transposed V rows";
        return false;
    }
    if (!sc_info.empty()) {
        error = "PyramidKV C1 cannot compact with a pending stream copy";
        return false;
    }
    if (get_has_shift()) {
        error = "PyramidKV C1 cannot compact while a K-shift is pending";
        return false;
    }
    if (layers.empty()) {
        error = "PyramidKV C1 found no dense KV layers";
        return false;
    }
    if (!pyramidkv_c1_hot_enabled || !llama_kv_type_is_turbo(type_k()) ||
            type_v() != type_k()) {
        error = "PyramidKV C1 requires TQ4 or TQ3.5 cold K/V plus an F16 hot window";
        return false;
    }
    if (hparams.n_layer_all == 0) {
        error = "PyramidKV C1 has no dense layer geometry";
        return false;
    }

    for (size_t i = 0; i < layers.size(); ++i) {
        if (layers[i].k == nullptr || layers[i].v == nullptr ||
                layers[i].k_hot == nullptr || layers[i].v_hot == nullptr) {
            error = "PyramidKV C1 requires cold and F16 hot K/V for every cached layer";
            return false;
        }
        for (size_t j = 0; j < i; ++j) {
            if (layers[i].k == layers[j].k || layers[i].v == layers[j].v ||
                    layers[i].k_hot == layers[j].k_hot || layers[i].v_hot == layers[j].v_hot) {
                error = "PyramidKV C1 rejects reused or aliased layer storage";
                return false;
            }
        }
    }
    return true;
}

bool llama_kv_cache::pyramidkv_c1_should_maintain(uint32_t n_tokens, uint32_t n_ubatch) {
    if (!pyramidkv_c1_compacted) {
        return true;
    }

    uint32_t hot_capacity = std::numeric_limits<uint32_t>::max();
    for (const auto & layer : pyramidkv_c1_layers) {
        hot_capacity = std::min(hot_capacity, layer.hot_row_capacity);
    }
    if (pyramidkv_c1_layers.empty() || n_ubatch == 0 || n_ubatch >= hot_capacity) {
        return true;
    }

    // Leave room for a full next ubatch after any single-token prefix.
    const uint32_t interval = std::min(n_ubatch, hot_capacity - n_ubatch + 1);
    pyramidkv_c1_tokens_since_compact = static_cast<uint32_t>(std::min<uint64_t>(
        static_cast<uint64_t>(pyramidkv_c1_tokens_since_compact) + n_tokens,
        std::numeric_limits<uint32_t>::max()));
    return pyramidkv_c1_tokens_since_compact >= interval;
}

bool llama_kv_cache::pyramidkv_c1_keep_all(
        std::vector<llama_pyramidkv_c1_layer_selection> & selections,
        std::string & error) const {
    selections.clear();
    if (!pyramidkv_c1_compacted) {
        error = "PyramidKV append maintenance requires the initial prompt selection";
        return false;
    }
    for (size_t index = 0; index < layers.size(); ++index) {
        const auto & state = pyramidkv_c1_layers[index];
        std::vector<std::vector<std::int64_t>> positions;
        std::vector<std::vector<std::size_t>> cells;
        std::vector<std::vector<std::size_t>> slots;
        uint32_t active = 0;
        if (!pyramidkv_c1_key_positions(layers[index].il,
                state.cold_row_capacity + state.hot_row_capacity,
                positions, cells, slots, active, error)) {
            return false;
        }
        llama_pyramidkv_c1_layer_selection selection;
        selection.il = layers[index].il;
        selection.source_tokens = active;
        selection.kv_heads = state.kv_heads;
        selection.continuation_headroom = pyramidkv_c1_config.continuation_headroom;
        selection.heads.resize(state.kv_heads);
        for (uint32_t head = 0; head < state.kv_heads; ++head) {
            auto & kept = selection.heads[head];
            // Compaction addresses logical cells, which need not be in position order after seq_rm.
            std::vector<std::pair<size_t, int64_t>> ordered;
            ordered.reserve(cells[head].size());
            for (size_t row = 0; row < cells[head].size(); ++row) {
                ordered.emplace_back(cells[head][row], positions[head][row]);
            }
            std::sort(ordered.begin(), ordered.end());
            for (const auto & row : ordered) {
                kept.keep_cells.push_back(row.first);
                kept.keep_positions.push_back(row.second);
            }
        }
        selections.push_back(std::move(selection));
    }
    return true;
}

bool llama_kv_cache::pyramidkv_c1_hybrid_ready(int32_t il) const {
    if (!pyramidkv_c1_compacted || !pyramidkv_c1_hot_enabled) {
        return false;
    }
    const auto map_it = map_layer_ids.find(il);
    if (map_it == map_layer_ids.end() ||
            static_cast<size_t>(map_it->second) >= layers.size() ||
            static_cast<size_t>(map_it->second) >= pyramidkv_c1_layers.size()) {
        return false;
    }
    const auto & layer = layers[map_it->second];
    const auto & state = pyramidkv_c1_layers[map_it->second];
    return state.compacted && state.kv_heads == hparams.n_head_kv(il) &&
        state.cold_heads.size() == state.kv_heads &&
        state.hot_heads.size() == state.kv_heads &&
        layer.k != nullptr && layer.v != nullptr &&
        layer.k_hot != nullptr && layer.v_hot != nullptr;
}

bool llama_kv_cache::pyramidkv_c1_key_positions(
        int32_t il,
        uint32_t n_kv,
        std::vector<std::vector<std::int64_t>> & positions,
        std::vector<std::vector<std::size_t>> & logical_cells,
        std::vector<std::vector<std::size_t>> & score_slots,
        uint32_t & active_tokens,
        std::string & error,
        llama_seq_id seq_id,
        bool sequence_local) const {
    if (!pyramidkv_c1_supported(error)) {
        return false;
    }
    const llama_seq_id want_seq = seq_id < 0 ? 0 : seq_id;

    const auto layer_it = map_layer_ids.find(il);
    if (layer_it == map_layer_ids.end() ||
            static_cast<std::size_t>(layer_it->second) >= pyramidkv_c1_layers.size()) {
        error = "PyramidKV C1 score layer is not present in the cache map";
        return false;
    }

    const auto & cells = v_cells[0];
    active_tokens = cells.used_max_p1();
    const auto & layer_state = pyramidkv_c1_layers[layer_it->second];
    if (sequence_local && (!pyramidkv_c1_paged() || seq_id < 0 || layer_state.compacted)) {
        error = "PyramidKV local score mapping requires an unselected paged sequence";
        return false;
    }
    if (active_tokens == 0 || (!layer_state.compacted &&
                ((!sequence_local && active_tokens > n_kv) || n_kv > cells.size())) ||
            (layer_state.compacted && n_kv != layer_state.cold_row_capacity + layer_state.hot_row_capacity)) {
        error = "PyramidKV C1 score rows do not match the cache cell range";
        return false;
    }

    const uint32_t kv_heads = layer_state.kv_heads != 0 ? layer_state.kv_heads : hparams.n_head_kv(il);
    positions.assign(kv_heads, {});
    logical_cells.assign(kv_heads, {});
    score_slots.assign(kv_heads, {});
    for (uint32_t head = 0; head < kv_heads; ++head) {
        auto & head_positions = positions[head];
        auto & head_cells = logical_cells[head];
        auto & head_slots = score_slots[head];
        uint32_t local_slot = 0;
        std::vector<std::tuple<llama_pos, uint32_t, uint32_t>> entries;
        if (layer_state.compacted) {
            if (head >= layer_state.cold_heads.size() || head >= layer_state.hot_heads.size()) {
                error = "PyramidKV C1 layer has no per-head cold/hot maps";
                return false;
            }
        }
        for (uint32_t i = 0; i < active_tokens; ++i) {
            if (cells.is_empty(i) || cells.seq_count(i) != 1 || !cells.seq_has(i, want_seq)) {
                continue;
            }
            uint32_t slot = sequence_local ? local_slot++ : i;
            if (sequence_local && slot >= n_kv) {
                error = "PyramidKV observer row mapping exceeds its score tensor";
                return false;
            }
            if (layer_state.compacted) {
                const auto & hot = layer_state.hot_heads[head].logical_to_physical;
                const auto & cold = layer_state.cold_heads[head].logical_to_physical;
                if (i < hot.size() && hot[i] < layer_state.hot_row_capacity) {
                    slot = layer_state.cold_row_capacity + hot[i];
                } else if (i < cold.size() && cold[i] < layer_state.cold_row_capacity) {
                    slot = cold[i];
                } else {
                    continue;
                }
            }
            const llama_pos position = cells.pos_get(i);
            if (position < 0) {
                error = "PyramidKV C1 head has a negative original position";
                return false;
            }
            entries.emplace_back(position, i, slot);
        }
        // Sorted by (position, logical cell). An M-RoPE image puts all of its
        // cells on one sequence position (the 2-D place lives in the cell's
        // ext), so equal positions are legal; one logical cell twice is not.
        std::sort(entries.begin(), entries.end());
        for (const auto & [position, logical, slot] : entries) {
            if (!head_positions.empty() && (position < head_positions.back() ||
                    (position == head_positions.back() && logical == head_cells.back()))) {
                error = "PyramidKV C1 head contains duplicate original positions";
                return false;
            }
            head_positions.push_back(position);
            head_cells.push_back(logical);
            head_slots.push_back(slot);
        }
        if (head_positions.empty()) {
            error = "PyramidKV C1 layer has no valid logical row for a KV head";
            return false;
        }
    }
    return true;
}

bool llama_kv_cache::pyramidkv_c1_prepare_batch_rows(
        const slot_info & sinfo,
        const llama_ubatch & ubatch,
        std::string & error) {
    if (!pyramidkv_c1_hot_enabled && !pyramidkv_c1_compacted) {
        return true;
    }
    if (sinfo.n_stream() != 1 || ubatch.n_tokens != sinfo.size()) {
        error = "PyramidKV C1 batch row preparation requires one cache stream";
        return false;
    }
    if (pyramidkv_c1_hot_enabled && pyramidkv_c1_compacted &&
            (pyramidkv_c1_layers.empty() ||
             pyramidkv_c1_layers.front().hot_row_capacity <= pyramidkv_c1_layers.front().hot_recent ||
             ubatch.n_tokens > pyramidkv_c1_layers.front().hot_row_capacity -
                 pyramidkv_c1_layers.front().hot_recent)) {
        error = "PyramidKV C1 hot window has no bounded batch headroom";
        return false;
    }

    auto & cells = v_cells[sinfo.strm[0]];
    const bool paged = pyramidkv_c1_paged();
    // Oldest (position, cell) of each sequence that stays protected; older hot
    // rows may be released.
    std::vector<std::pair<int64_t, uint32_t>> protected_from;
    if (paged) {
        const uint64_t protected_per_seq = static_cast<uint64_t>(pyramidkv_c1_config.recent_window) +
            pyramidkv_c1_config.rollback_headroom;
        const uint64_t required_hot = static_cast<uint64_t>(n_seq_max)*protected_per_seq + ubatch.n_tokens;
        if (ubatch.n_tokens > pyramidkv_c1_config.continuation_headroom ||
                required_hot > pyramidkv_c1_config.hot_capacity) {
            error = "PyramidKV paged hot cache cannot protect recent and rollback rows plus this ubatch";
            return false;
        }
        // Protect each sequence's newest protected_per_seq CELLS (a later
        // ubatch can run before the caller rejects this sequence's draft
        // tail). Counting cells rather than positions matters for M-RoPE
        // images, whose cells share one position: a position window kept a
        // whole screenshot protected and the ring overflowed. For text the
        // two are the same. Every head holds the same logical cells, so the
        // first ring decides for all.
        protected_from.assign(n_seq_max, { std::numeric_limits<int64_t>::min(), 0 });
        if (!pyramidkv_c1_layers.empty() && !pyramidkv_c1_layers.front().hot_heads.empty() &&
                protected_per_seq > 0) {
            std::vector<std::vector<std::pair<int64_t, uint32_t>>> rows(n_seq_max);
            for (const uint32_t logical : pyramidkv_c1_layers.front().hot_heads.front().physical_to_logical) {
                if (logical == pyramidkv_c1_invalid_cell || logical >= cells.size() ||
                        cells.is_empty(logical) || cells.seq_count(logical) != 1) {
                    continue;
                }
                const llama_seq_id seq = cells.seq_get(logical);
                if (seq >= 0 && static_cast<uint32_t>(seq) < n_seq_max) {
                    rows[seq].emplace_back(static_cast<int64_t>(cells.pos_get(logical)), logical);
                }
            }
            for (uint32_t seq = 0; seq < n_seq_max; ++seq) {
                auto & r = rows[seq];
                if (r.size() <= protected_per_seq) {
                    continue;
                }
                const auto nth = r.begin() + static_cast<std::ptrdiff_t>(protected_per_seq - 1);
                std::nth_element(r.begin(), nth, r.end(), std::greater<std::pair<int64_t, uint32_t>>());
                protected_from[seq] = *nth;
            }
        }
    }
    for (auto & state : pyramidkv_c1_layers) {
        if (state.kv_heads == 0 || state.hot_heads.size() != state.kv_heads) {
            error = "PyramidKV C1 layer has no per-head hot map";
            return false;
        }

        if (state.compacted && state.cold_heads.size() != state.kv_heads) {
            error = "PyramidKV C1 compacted layer has no per-head cold map";
            return false;
        }

        for (uint32_t head = 0; head < state.kv_heads; ++head) {
            auto & hot = state.hot_heads[head];
            if (hot.logical_to_physical.size() != cells.size()) {
                hot.logical_to_physical.assign(cells.size(), pyramidkv_c1_invalid_cell);
            }
            uint32_t free_hot_rows = 0;
            for (uint32_t physical = 0; physical < hot.physical_to_logical.size(); ++physical) {
                const uint32_t logical = hot.physical_to_logical[physical];
                bool release = logical != pyramidkv_c1_invalid_cell &&
                    (logical >= cells.size() || cells.is_empty(logical));
                if (paged && logical != pyramidkv_c1_invalid_cell && !release) {
                    if (cells.seq_count(logical) != 1) {
                        error = "PyramidKV paged hot rows require one owner per cell";
                        return false;
                    }
                    const llama_seq_id seq = cells.seq_get(logical);
                    if (seq < 0 || static_cast<uint32_t>(seq) >= n_seq_max) {
                        error = "PyramidKV paged hot row has an invalid sequence owner";
                        return false;
                    }
                    release = std::make_pair(static_cast<int64_t>(cells.pos_get(logical)), logical) <
                        protected_from[seq];
                }
                if (release) {
                    hot.physical_to_logical[physical] = pyramidkv_c1_invalid_cell;
                    if (logical < hot.logical_to_physical.size()) {
                        hot.logical_to_physical[logical] = pyramidkv_c1_invalid_cell;
                    }
                }
                free_hot_rows += hot.physical_to_logical[physical] == pyramidkv_c1_invalid_cell;
            }
            for (uint32_t logical : sinfo.idxs[0]) {
                if (logical >= hot.logical_to_physical.size()) {
                    error = "PyramidKV C1 batch cell exceeds the logical head map";
                    return false;
                }
                if (hot.logical_to_physical[logical] != pyramidkv_c1_invalid_cell) {
                    continue;
                }
                if (hot.physical_to_logical.empty()) {
                    error = "PyramidKV C1 hot map has no physical row";
                    return false;
                }
                const uint32_t capacity = static_cast<uint32_t>(hot.physical_to_logical.size());
                uint32_t physical = pyramidkv_c1_invalid_cell;
                if (free_hot_rows != 0) {
                    // Partial seq_rm can leave holes away from next_row. Fill
                    // those holes before discarding still-valid history.
                    for (uint32_t offset = 0; offset < capacity; ++offset) {
                        const uint32_t candidate = (hot.next_row + offset) % capacity;
                        if (hot.physical_to_logical[candidate] == pyramidkv_c1_invalid_cell) {
                            physical = candidate;
                            --free_hot_rows;
                            break;
                        }
                    }
                } else if (paged) {
                    error = "PyramidKV paged hot cache would overwrite a protected recent or rollback row";
                    return false;
                } else {
                    // Hole reuse can change physical age order. Evict the
                    // oldest original position, never a row just assigned to
                    // this ubatch (whose logical cell is not committed yet).
                    llama_pos oldest = std::numeric_limits<llama_pos>::max();
                    for (uint32_t candidate = 0; candidate < capacity; ++candidate) {
                        const uint32_t existing = hot.physical_to_logical[candidate];
                        if (existing < cells.size() && !cells.is_empty(existing) &&
                                cells.pos_get(existing) < oldest) {
                            oldest = cells.pos_get(existing);
                            physical = candidate;
                        }
                    }
                }
                if (physical == pyramidkv_c1_invalid_cell) {
                    error = "PyramidKV C1 hot window has no free or evictable row outside the active batch";
                    return false;
                }
                const uint32_t evicted = hot.physical_to_logical[physical];
                if (state.compacted && evicted != pyramidkv_c1_invalid_cell &&
                        (evicted >= state.cold_heads[head].logical_to_physical.size() ||
                         state.cold_heads[head].logical_to_physical[evicted] >= state.cold_row_capacity)) {
                    error = "PyramidKV append maintenance did not preserve a live hot row before overwrite";
                    return false;
                }
                if (evicted != pyramidkv_c1_invalid_cell && evicted < hot.logical_to_physical.size()) {
                    hot.logical_to_physical[evicted] = pyramidkv_c1_invalid_cell;
                }
                hot.physical_to_logical[physical] = logical;
                hot.logical_to_physical[logical] = physical;
                hot.next_row = (physical + 1) % static_cast<uint32_t>(hot.physical_to_logical.size());
            }
            if (pyramidkv_c1_paged() && ubatch.seq_id != nullptr && ubatch.pos != nullptr) {
                // Paged: a token of a selected sequence joins that sequence's
                // list for this layer/head (its arena row is written by the
                // same ubatch). The list must stay position-ordered.
                auto & lists = pyramidkv_c1_paged_lists[&state - pyramidkv_c1_layers.data()][head];
                for (uint32_t token = 0; token < ubatch.n_tokens; ++token) {
                    const llama_seq_id seq = ubatch.n_seq_id[token] > 0 ? ubatch.seq_id[token][0] : -1;
                    if (!pyramidkv_c1_paged_seq_compacted(seq)) {
                        continue;
                    }
                    auto & list = lists[seq];
                    const int32_t pos = static_cast<int32_t>(ubatch.pos[token]);
                    // The cells of an M-RoPE image (embedding rows) share one
                    // position; text must strictly increase.
                    const bool image_row = ubatch.token == nullptr && ubatch.embd != nullptr && ubatch.n_pos == 4;
                    if (!list.empty() && (list.back().pos > pos || (list.back().pos == pos && !image_row))) {
                        error = "PyramidKV C1 paged append is not position-ordered (rollback missed?)";
                        return false;
                    }
                    if (list.size() >= pyramidkv_c1_config.list_capacity) {
                        error = "PyramidKV C1 paged list_capacity exhausted for a sequence";
                        return false;
                    }
                    list.push_back({ sinfo.idxs[0][token], pos });
                }
            }
            if (state.compacted) {
                auto & cold = state.cold_heads[head];
                if (cold.logical_to_physical.size() != cells.size()) {
                    error = "PyramidKV C1 cold map is not aligned with logical cells";
                    return false;
                }
                for (uint32_t physical = 0; physical < cold.physical_to_logical.size(); ++physical) {
                    const uint32_t logical = cold.physical_to_logical[physical];
                    if (logical != pyramidkv_c1_invalid_cell && cells.is_empty(logical)) {
                        cold.physical_to_logical[physical] = pyramidkv_c1_invalid_cell;
                        cold.logical_to_physical[logical] = pyramidkv_c1_invalid_cell;
                    }
                }
            }
        }
    }
    return true;
}

bool llama_kv_cache::pyramidkv_c1_compact(
        llama_context * lctx,
        const std::vector<llama_pyramidkv_c1_layer_selection> & selections,
        std::string & error) {
    if (lctx == nullptr || !pyramidkv_c1_supported(error)) {
        return false;
    }
    if (selections.empty()) {
        error = "PyramidKV C1 received no layer selections";
        return false;
    }
    const bool initial_selection = !pyramidkv_c1_compacted;

    const auto & old_cells = v_cells[0];
    const uint32_t old_active = old_cells.used_max_p1();
    if (old_active == 0) {
        error = "PyramidKV C1 cannot compact an empty cache";
        return false;
    }

#if LLAMA_PYRAMIDKV_C1_PHASE_TIMING
    auto & phase = pyramidkv_c1_phase_timing_stats;
    ++phase.compaction_calls;
#endif

    struct head_plan {
        std::vector<uint32_t> source_rows;
        std::vector<int32_t> hot_source_rows;
        std::vector<int64_t> hot_destination_rows;
        uint32_t capacity = 0;
    };
    struct layer_plan {
        size_t layer_index = 0;
        const llama_pyramidkv_c1_layer_selection * selection = nullptr;
        std::vector<head_plan> heads;
        uint32_t cold_capacity = 0;
    };

    std::vector<const llama_pyramidkv_c1_layer_selection *> by_layer(layers.size(), nullptr);
    for (const auto & selection : selections) {
        const auto map_it = map_layer_ids.find(selection.il);
        if (selection.source_tokens != old_active || map_it == map_layer_ids.end() ||
                static_cast<size_t>(map_it->second) >= layers.size()) {
            error = "PyramidKV C1 selection was built for a different cache or layer set";
            return false;
        }
        const size_t layer_index = static_cast<size_t>(map_it->second);
        if (by_layer[layer_index] != nullptr || selection.kv_heads == 0 ||
                selection.heads.size() != selection.kv_heads ||
                selection.continuation_headroom == 0) {
            error = "PyramidKV C1 selection has duplicate layers or invalid per-head metadata";
            return false;
        }
        by_layer[layer_index] = &selection;

        for (size_t head = 0; head < selection.heads.size(); ++head) {
            const auto & selected = selection.heads[head];
            if (selected.keep_cells.empty() ||
                    selected.keep_cells.size() != selected.keep_positions.size() ||
                    selected.protected_recent_cells.size() != selected.protected_recent_positions.size()) {
                error = "PyramidKV C1 selection has an empty or inconsistent head list";
                return false;
            }
            size_t previous = 0;
            bool first = true;
            for (size_t i = 0; i < selected.keep_cells.size(); ++i) {
                const size_t logical = selected.keep_cells[i];
                if (logical >= old_active || (!first && logical <= previous) ||
                        old_cells.is_empty(static_cast<uint32_t>(logical)) ||
                        old_cells.pos_get(static_cast<uint32_t>(logical)) != selected.keep_positions[i]) {
                    error = "PyramidKV C1 selection does not preserve logical positions";
                    return false;
                }
                first = false;
                previous = logical;
            }
            for (size_t i = 0; i < selected.protected_recent_cells.size(); ++i) {
                const size_t logical = selected.protected_recent_cells[i];
                if (logical >= old_active || old_cells.is_empty(static_cast<uint32_t>(logical)) ||
                        old_cells.pos_get(static_cast<uint32_t>(logical)) != selected.protected_recent_positions[i] ||
                        !std::binary_search(selected.keep_cells.begin(), selected.keep_cells.end(), logical)) {
                    error = "PyramidKV C1 selection dropped a protected recent row";
                    return false;
                }
            }
        }
    }
    for (const auto & layer : layers) {
        const auto map_it = map_layer_ids.find(static_cast<int32_t>(layer.il));
        if (map_it == map_layer_ids.end() || by_layer[map_it->second] == nullptr) {
            error = "PyramidKV C1 has no score selection for a cached layer";
            return false;
        }
    }

    std::vector<layer_plan> plans;
    size_t promoted_rows = 0;
    size_t copied_cold_rows = 0;
    plans.reserve(layers.size());
    for (size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
        const auto & layer = layers[layer_index];
        const auto & selection = *by_layer[layer_index];
        const auto & old_state = pyramidkv_c1_layers[layer_index];
        const uint32_t kv_heads = hparams.n_head_kv(layer.il);
        if (selection.kv_heads != kv_heads || old_state.kv_heads != kv_heads ||
                selection.heads.size() != kv_heads) {
            error = "PyramidKV C1 selection does not match layer KV-head geometry";
            return false;
        }
        if (old_state.hot_heads.size() != kv_heads ||
                old_state.compacted && old_state.cold_heads.size() != kv_heads) {
            error = "PyramidKV C1 existing layer has no complete per-head maps";
            return false;
        }

        layer_plan plan;
        plan.layer_index = layer_index;
        plan.selection = &selection;
        plan.heads.resize(kv_heads);
        for (uint32_t head = 0; head < kv_heads; ++head) {
            const auto & selected = selection.heads[head];
            auto & hp = plan.heads[head];
            if (selected.keep_cells.size() > std::numeric_limits<uint32_t>::max() -
                    selection.continuation_headroom) {
                error = "PyramidKV C1 per-head capacity overflows uint32_t";
                return false;
            }
            hp.capacity = GGML_PAD(static_cast<uint32_t>(selected.keep_cells.size() +
                selection.continuation_headroom), n_pad);
            if (hp.capacity == 0 || hp.capacity > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
                error = "PyramidKV C1 per-head capacity cannot be addressed by I32 indices";
                return false;
            }
            hp.source_rows.reserve(selected.keep_cells.size());
            for (const size_t logical_size : selected.keep_cells) {
                const uint32_t logical = static_cast<uint32_t>(logical_size);
                uint32_t source = logical;
                if (old_state.compacted) {
                    const auto & old_head = old_state.cold_heads[head];
                    if (logical >= old_head.logical_to_physical.size() ||
                            old_head.logical_to_physical[logical] == pyramidkv_c1_invalid_cell) {
                        const auto & old_hot = old_state.hot_heads[head];
                        if (logical >= old_hot.logical_to_physical.size() ||
                                old_hot.logical_to_physical[logical] >= old_hot.row_capacity) {
                            error = "PyramidKV selection has neither a cold nor a hot source row";
                            return false;
                        }
                        const uint64_t hot_source = static_cast<uint64_t>(head) * old_hot.row_capacity +
                            old_hot.logical_to_physical[logical];
                        if (hot_source >= static_cast<uint64_t>(layer.k_hot->ne[1]) ||
                                hot_source > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
                            error = "PyramidKV hot promotion row exceeds the physical cache";
                            return false;
                        }
                        hp.hot_source_rows.push_back(static_cast<int32_t>(hot_source));
                        hp.hot_destination_rows.push_back(static_cast<int64_t>(hp.source_rows.size()));
                        hp.source_rows.push_back(pyramidkv_c1_invalid_cell);
                        continue;
                    }
                    if (old_head.row_capacity == 0 ||
                            old_head.logical_to_physical[logical] >= old_head.row_capacity ||
                            head > (std::numeric_limits<uint32_t>::max() -
                                old_head.logical_to_physical[logical]) / old_head.row_capacity) {
                        error = "PyramidKV old per-head row address overflows";
                        return false;
                    }
                    source = head * old_head.row_capacity + old_head.logical_to_physical[logical];
                }
                if (source >= layers[layer_index].k->ne[1]) {
                    error = "PyramidKV source row exceeds the physical cold tensor";
                    return false;
                }
                hp.source_rows.push_back(source);
            }
            plan.cold_capacity = std::max(plan.cold_capacity, hp.capacity);
        }
        if (plan.cold_capacity == 0 || plan.cold_capacity >
                static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) / kv_heads) {
            error = "PyramidKV layer cold capacity is invalid";
            return false;
        }
        for (const auto & head : plan.heads) {
            promoted_rows += head.hot_source_rows.size();
            copied_cold_rows += head.source_rows.size() - head.hot_source_rows.size();
        }
        plans.push_back(std::move(plan));
    }

    const auto & transition_config = pyramidkv_c1_config;

    size_t tensor_count = layers.size();
    if (tensor_count > std::numeric_limits<size_t>::max() /
            (4u * (1u + n_stream))) {
        error = "PyramidKV C1 replacement tensor count overflows";
        return false;
    }
    tensor_count *= 4u * (1u + n_stream);
    if (turbo_rotation != nullptr && tensor_count > std::numeric_limits<size_t>::max() - 2u) {
        error = "PyramidKV C1 rotation tensor count overflows";
        return false;
    }
    const size_t overhead_tensors = tensor_count + (turbo_rotation != nullptr ? 2u : 0u);
    if (overhead_tensors > std::numeric_limits<size_t>::max() / ggml_tensor_overhead()) {
        error = "PyramidKV C1 replacement context size overflows";
        return false;
    }
    ggml_init_params params = {
        /*.mem_size   =*/ overhead_tensors * ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr new_ctx(ggml_init(params));
    if (!new_ctx) {
        error = "PyramidKV C1 failed to create the replacement ggml context";
        return false;
    }

    std::vector<kv_layer> new_layers;
    new_layers.reserve(layers.size());
    for (const auto & plan : plans) {
        const auto & old_layer = layers[plan.layer_index];
        const uint32_t kv_heads = hparams.n_head_kv(old_layer.il);
        const bool head_major = pyramidkv_c1_layers[plan.layer_index].compacted;
        const uint32_t padded_k = static_cast<uint32_t>(old_layer.k->ne[0] / (head_major ? 1 : kv_heads));
        const uint32_t padded_v = static_cast<uint32_t>(old_layer.v->ne[0] / (head_major ? 1 : kv_heads));
        const uint64_t cold_rows = static_cast<uint64_t>(plan.cold_capacity) * kv_heads;
        const uint64_t hot_rows = old_layer.k_hot->ne[1];
        if (cold_rows > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) ||
                hot_rows > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
            error = "PyramidKV replacement row count exceeds ggml index range";
            return false;
        }
        ggml_tensor * k = ggml_new_tensor_3d(new_ctx.get(), old_layer.k->type,
            padded_k, cold_rows, n_stream);
        ggml_tensor * v = ggml_new_tensor_3d(new_ctx.get(), old_layer.v->type,
            padded_v, cold_rows, n_stream);
        ggml_tensor * k_hot = ggml_new_tensor_3d(new_ctx.get(), GGML_TYPE_F16,
            old_layer.k_hot->ne[0], hot_rows, n_stream);
        ggml_tensor * v_hot = ggml_new_tensor_3d(new_ctx.get(), GGML_TYPE_F16,
            old_layer.v_hot->ne[0], hot_rows, n_stream);
        ggml_format_name(k, "cache_k_l%d", old_layer.il);
        ggml_format_name(v, "cache_v_l%d", old_layer.il);
        ggml_format_name(k_hot, "cache_k_hot_l%d", old_layer.il);
        ggml_format_name(v_hot, "cache_v_hot_l%d", old_layer.il);

        std::vector<ggml_tensor *> k_stream;
        std::vector<ggml_tensor *> v_stream;
        std::vector<ggml_tensor *> k_hot_stream;
        std::vector<ggml_tensor *> v_hot_stream;
        for (uint32_t stream = 0; stream < n_stream; ++stream) {
            k_stream.push_back(ggml_view_2d(new_ctx.get(), k, k->ne[0], k->ne[1],
                k->nb[1], stream * k->nb[2]));
            v_stream.push_back(ggml_view_2d(new_ctx.get(), v, v->ne[0], v->ne[1],
                v->nb[1], stream * v->nb[2]));
            k_hot_stream.push_back(ggml_view_2d(new_ctx.get(), k_hot, k_hot->ne[0], k_hot->ne[1],
                k_hot->nb[1], stream * k_hot->nb[2]));
            v_hot_stream.push_back(ggml_view_2d(new_ctx.get(), v_hot, v_hot->ne[0], v_hot->ne[1],
                v_hot->nb[1], stream * v_hot->nb[2]));
        }
        new_layers.push_back({ old_layer.il, k, v, k_stream, v_stream,
            k_hot, v_hot, k_hot_stream, v_hot_stream });
    }

    ggml_tensor * new_rotation = nullptr;
    ggml_tensor * new_rotation_inv = nullptr;
    if (turbo_rotation != nullptr) {
        new_rotation = ggml_new_tensor_2d(new_ctx.get(), GGML_TYPE_F32,
            GGML_TURBO4_QK, GGML_TURBO4_QK);
        new_rotation_inv = ggml_new_tensor_2d(new_ctx.get(), GGML_TYPE_F32,
            GGML_TURBO4_QK, GGML_TURBO4_QK);
        ggml_format_name(new_rotation, "turbo4_rotation");
        ggml_format_name(new_rotation_inv, "turbo4_rotation_inv");
    }

    const size_t new_bytes = ggml_backend_alloc_ctx_tensors_from_buft_size(
        new_ctx.get(), kv_buffer_type);
    const size_t old_bytes = total_size();
    const size_t observer_bytes = transition_config.observer_max_bytes;
    const size_t graph_scratch_bytes = observer_bytes;
    size_t copy_scratch_bytes = 0;
    size_t metadata_bytes = 0;
    bool metadata_ok = true;
    const auto add_metadata_vector = [&](size_t logical_count, size_t physical_count) {
        if (!metadata_ok || logical_count > std::numeric_limits<size_t>::max() - physical_count) {
            metadata_ok = false;
            return;
        }
        const size_t entries = logical_count + physical_count;
        if (entries > std::numeric_limits<size_t>::max() / sizeof(uint32_t) ||
                metadata_bytes > std::numeric_limits<size_t>::max() -
                    entries * sizeof(uint32_t)) {
            metadata_ok = false;
            return;
        }
        metadata_bytes += entries * sizeof(uint32_t);
    };
    for (const auto & plan : plans) {
        const auto & old_layer = layers[plan.layer_index];
        for (const auto * tensor : { old_layer.k, old_layer.v }) {
            const size_t bytes = ggml_nbytes(tensor);
            if (bytes > std::numeric_limits<size_t>::max() / 2) {
                error = "PyramidKV C1 copy scratch size overflows";
                return false;
            }
            copy_scratch_bytes = std::max(copy_scratch_bytes, bytes * 2);
        }
        copy_scratch_bytes = std::max({ copy_scratch_bytes,
            ggml_nbytes(old_layer.k_hot), ggml_nbytes(old_layer.v_hot) });
        const auto & old_state = pyramidkv_c1_layers[plan.layer_index];
        for (uint32_t head = 0; head < plan.heads.size(); ++head) {
            const auto & old_hot = old_state.hot_heads[head];
            // The old hot map and its copied replacement are simultaneously
            // live until the atomic state swap.
            add_metadata_vector(old_hot.logical_to_physical.size(),
                old_hot.physical_to_logical.size());
            add_metadata_vector(old_hot.logical_to_physical.size(),
                old_hot.physical_to_logical.size());
            if (old_state.compacted) {
                const auto & old_cold = old_state.cold_heads[head];
                add_metadata_vector(old_cold.logical_to_physical.size(),
                    old_cold.physical_to_logical.size());
            }
            // New cold maps retain the logical cell index range and add the
            // head-local physical map.
            add_metadata_vector(old_cells.size(), plan.cold_capacity);
        }
    }
    size_t transition_bytes = old_bytes;
    const auto add_transition = [&](size_t bytes) {
        if (transition_bytes > std::numeric_limits<size_t>::max() - bytes) {
            return false;
        }
        transition_bytes += bytes;
        return true;
    };
    bool plan_bytes_ok = add_transition(params.mem_size);
    const auto charge_plan = [&](size_t count, size_t bytes) {
        plan_bytes_ok = plan_bytes_ok && count <= std::numeric_limits<size_t>::max() / bytes &&
            add_transition(count * bytes);
    };
    charge_plan(plans.capacity(), sizeof(layer_plan));
    charge_plan(layers.size(), sizeof(pyramidkv_c1_layer_state));
    for (const auto & plan : plans) {
        charge_plan(plan.heads.capacity(), sizeof(head_plan));
        charge_plan(plan.heads.size() * 2, sizeof(pyramidkv_c1_head_state));
        for (const auto & head : plan.heads) {
            charge_plan(head.source_rows.capacity(), sizeof(uint32_t));
            charge_plan(head.hot_source_rows.capacity(), sizeof(int32_t));
            charge_plan(head.hot_destination_rows.capacity() * 2, sizeof(int64_t));
        }
    }
    if (!metadata_ok || !add_transition(new_bytes) || !add_transition(observer_bytes) ||
            !add_transition(graph_scratch_bytes) || !add_transition(metadata_bytes) ||
            !plan_bytes_ok || !add_transition(copy_scratch_bytes) ||
            transition_bytes > transition_config.transition_max_bytes) {
        error = "PyramidKV C1 transition budget rejects old+new KV, observer, scratch, or map metadata";
        return false;
    }

    llama_synchronize(lctx);
    ggml_backend_sched_reset(lctx->get_sched());
    ggml_backend_buffer_ptr new_buf(
        ggml_backend_alloc_ctx_tensors_from_buft(new_ctx.get(), kv_buffer_type));
    if (!new_buf) {
        error = "PyramidKV C1 failed to allocate replacement kv_buffer_type buffers";
        return false;
    }
    ggml_backend_buffer_clear(new_buf.get(), 0);

    auto copy_hot = [&](ggml_tensor * source, ggml_tensor * destination) {
#if LLAMA_PYRAMIDKV_C1_PHASE_TIMING
        const int64_t phase_start_us = ggml_time_us();
#endif
        const size_t bytes = ggml_nbytes(source);
        std::vector<uint8_t> host(bytes);
        ggml_backend_tensor_get(source, host.data(), 0, bytes);
        ggml_backend_tensor_set(destination, host.data(), 0, bytes);
#if LLAMA_PYRAMIDKV_C1_PHASE_TIMING
        ++phase.hot_copy_calls;
        // Count both the device-to-host read and host-to-device write, matching
        // cold-copy accounting for the total host payload moved.
        phase.hot_copy_bytes += static_cast<uint64_t>(bytes) * 2;
        phase.hot_copy_us += static_cast<uint64_t>(ggml_time_us() - phase_start_us);
#endif
    };
    auto copy_head_runs = [&](ggml_tensor * source, ggml_tensor * destination,
            const std::vector<uint32_t> & rows, uint32_t head,
            uint32_t old_capacity, uint32_t new_capacity, bool old_head_major) {
        const size_t source_row_bytes = ggml_row_size(source->type, source->ne[0]);
        const size_t destination_row_bytes = ggml_row_size(destination->type, destination->ne[0]);
        const size_t source_head_bytes = ggml_row_size(source->type,
            destination->ne[0]);
        size_t begin = 0;
        while (begin < rows.size()) {
            if (rows[begin] == pyramidkv_c1_invalid_cell) {
                ++begin;
                continue;
            }
            size_t run = 1;
            while (begin + run < rows.size() && rows[begin + run] != pyramidkv_c1_invalid_cell &&
                    rows[begin + run] == rows[begin] + run) {
                ++run;
            }
            if (run > std::numeric_limits<size_t>::max() / source_row_bytes ||
                    run > std::numeric_limits<size_t>::max() / destination_row_bytes) {
                throw std::runtime_error("PyramidKV row-run copy size overflows");
            }
#if LLAMA_PYRAMIDKV_C1_PHASE_TIMING
            const int64_t phase_start_us = ggml_time_us();
#endif
            std::vector<uint8_t> source_host(run * source_row_bytes);
            ggml_backend_tensor_get(source, source_host.data(),
                static_cast<size_t>(rows[begin]) * source_row_bytes, source_host.size());
            std::vector<uint8_t> destination_host(run * destination_row_bytes, 0);
            for (size_t i = 0; i < run; ++i) {
                const uint8_t * src = source_host.data() + i * source_row_bytes;
                if (!old_head_major) {
                    src += static_cast<size_t>(head) * source_head_bytes;
                }
                if (source_head_bytes != destination_row_bytes) {
                    throw std::runtime_error("PyramidKV head row byte geometry changed");
                }
                std::memcpy(destination_host.data() + i * destination_row_bytes,
                    src, destination_row_bytes);
            }
            const size_t destination_row = static_cast<size_t>(head) * new_capacity + begin;
            ggml_backend_tensor_set(destination, destination_host.data(),
                destination_row * destination_row_bytes, destination_host.size());
#if LLAMA_PYRAMIDKV_C1_PHASE_TIMING
            ++phase.cold_copy_calls;
            phase.cold_copy_bytes += source_host.size() + destination_host.size();
            phase.cold_copy_us += static_cast<uint64_t>(ggml_time_us() - phase_start_us);
#endif
            begin += run;
        }
        GGML_UNUSED(old_capacity);
    };

    auto promote_hot = [&](const kv_layer & source, const kv_layer & destination,
            const head_plan & plan, uint32_t head, uint32_t capacity) {
        if (plan.hot_source_rows.empty()) {
            return;
        }
        if (turbo_rotation_inv == nullptr || plan.hot_source_rows.size() != plan.hot_destination_rows.size()) {
            throw std::runtime_error("PyramidKV hot promotion has no rotation or complete row map");
        }
        const auto device = ggml_backend_buft_get_device(kv_buffer_type);
        ggml_backend_t backend = nullptr;
        for (int i = 0; i < ggml_backend_sched_get_n_backends(lctx->get_sched()); ++i) {
            auto candidate = ggml_backend_sched_get_backend(lctx->get_sched(), i);
            if (ggml_backend_get_device(candidate) == device) {
                backend = candidate;
                break;
            }
        }
        if (backend == nullptr || ggml_backend_buffer_is_host(source.k_hot->buffer)) {
            throw std::runtime_error("PyramidKV hot promotion requires its KV GPU backend");
        }
#if LLAMA_PYRAMIDKV_C1_PHASE_TIMING
        const int64_t phase_start_us = ggml_time_us();
        ++phase.promotion_calls;
        phase.promotion_rows += plan.hot_source_rows.size();
#endif
        const size_t max_nodes = 32;
        const ggml_init_params promotion_params = {
            /*.mem_size   =*/ max_nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(max_nodes, false),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_context_ptr promotion_ctx(ggml_init(promotion_params));
        if (!promotion_ctx) {
            throw std::runtime_error("PyramidKV hot promotion context allocation failed");
        }
        auto * graph = ggml_new_graph_custom(promotion_ctx.get(), max_nodes, false);
        auto * source_indices = ggml_new_tensor_1d(promotion_ctx.get(), GGML_TYPE_I32, plan.hot_source_rows.size());
        auto * destination_indices = ggml_new_tensor_1d(promotion_ctx.get(), GGML_TYPE_I64, plan.hot_destination_rows.size());
        auto encode = [&](ggml_tensor * hot, ggml_tensor * cold) {
            auto * hot_rows = ggml_reshape_2d(promotion_ctx.get(), hot, hot->ne[0], hot->ne[1]);
            auto * selected = ggml_get_rows(promotion_ctx.get(), hot_rows, source_indices);
            // Hot is already in the WHT basis. Undo that rotation before the
            // existing TQ4 encoder applies it once to newly promoted rows.
            auto * flat = ggml_reshape_2d(promotion_ctx.get(), selected,
                GGML_TURBO4_QK, ggml_nelements(selected) / GGML_TURBO4_QK);
            auto * unrotated = ggml_mul_mat(promotion_ctx.get(), turbo_rotation_inv, flat);
            ggml_mul_mat_set_prec(unrotated, GGML_PREC_F32);
            ggml_mul_mat_set_hint(unrotated, GGML_HINT_SRC0_IS_TURBO_INVERSE);
            unrotated = ggml_reshape_2d(promotion_ctx.get(), unrotated, cold->ne[0], plan.hot_source_rows.size());
            auto * cold_rows = ggml_reshape_2d(promotion_ctx.get(), cold, cold->ne[0], cold->ne[1]);
            auto * encoded = ggml_set_rows(promotion_ctx.get(), cold_rows, unrotated, destination_indices);
            ggml_format_name(encoded, "pyramidkv_promote_l%d_%s", source.il, hot == source.k_hot ? "k" : "v");
            ggml_build_forward_expand(graph, encoded);
        };
        encode(source.k_hot, destination.k);
        encode(source.v_hot, destination.v);

        const size_t promotion_bytes = ggml_backend_alloc_ctx_tensors_from_buft_size(promotion_ctx.get(), kv_buffer_type);
        if (promotion_bytes > graph_scratch_bytes || promotion_params.mem_size > graph_scratch_bytes - promotion_bytes) {
            throw std::runtime_error("PyramidKV hot promotion exceeds the reserved transition scratch budget");
        }
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            if (!ggml_backend_supports_op(backend, ggml_graph_node(graph, i))) {
                throw std::runtime_error("PyramidKV hot promotion contains an unsupported KV-device operation");
            }
        }
        ggml_backend_buffer_ptr promotion_buffer(
            ggml_backend_alloc_ctx_tensors_from_buft(promotion_ctx.get(), kv_buffer_type));
        if (!promotion_buffer) {
            throw std::runtime_error("PyramidKV hot promotion GPU allocation failed");
        }
        auto destination_rows = plan.hot_destination_rows;
        for (auto & row : destination_rows) {
            row += static_cast<int64_t>(head) * capacity;
        }
        ggml_backend_tensor_set(source_indices, plan.hot_source_rows.data(), 0,
            plan.hot_source_rows.size() * sizeof(int32_t));
        ggml_backend_tensor_set(destination_indices, destination_rows.data(), 0,
            destination_rows.size() * sizeof(int64_t));
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("PyramidKV hot promotion GPU computation failed");
        }
        ggml_backend_synchronize(backend);
#if LLAMA_PYRAMIDKV_C1_PHASE_TIMING
        ++phase.promotion_syncs;
        phase.promotion_us += static_cast<uint64_t>(ggml_time_us() - phase_start_us);
#endif
    };

    try {
        for (const auto & plan : plans) {
            const auto & old_layer = layers[plan.layer_index];
            const auto & old_state = pyramidkv_c1_layers[plan.layer_index];
            const bool old_head_major = old_state.compacted;
            const uint32_t old_capacity = old_state.compacted && !old_state.cold_heads.empty()
                ? old_state.cold_heads[0].row_capacity : 0;
            for (uint32_t head = 0; head < plan.heads.size(); ++head) {
                copy_head_runs(old_layer.k, new_layers[plan.layer_index].k,
                    plan.heads[head].source_rows, head, old_capacity,
                    plan.cold_capacity, old_head_major);
                copy_head_runs(old_layer.v, new_layers[plan.layer_index].v,
                    plan.heads[head].source_rows, head, old_capacity,
                    plan.cold_capacity, old_head_major);
                promote_hot(old_layer, new_layers[plan.layer_index], plan.heads[head], head, plan.cold_capacity);
            }
            copy_hot(old_layer.k_hot, new_layers[plan.layer_index].k_hot);
            copy_hot(old_layer.v_hot, new_layers[plan.layer_index].v_hot);
        }
        if (new_rotation != nullptr) {
            std::array<float, GGML_TURBO4_QK*GGML_TURBO4_QK> rotation;
            std::array<float, GGML_TURBO4_QK*GGML_TURBO4_QK> rotation_inv;
            ggml_turbo4_rotation_matrix(rotation.data(), false);
            ggml_turbo4_rotation_matrix(rotation_inv.data(), true);
            ggml_backend_tensor_set(new_rotation, rotation.data(), 0, rotation.size()*sizeof(float));
            ggml_backend_tensor_set(new_rotation_inv, rotation_inv.data(), 0, rotation_inv.size()*sizeof(float));
        }
    } catch (const std::exception & ex) {
        error = std::string("PyramidKV C1 replacement copy failed: ") + ex.what();
        return false;
    }

    std::vector<pyramidkv_c1_layer_state> new_states(layers.size());
    for (const auto & plan : plans) {
        const auto & selection = *plan.selection;
        const auto & old_state = pyramidkv_c1_layers[plan.layer_index];
        auto & state = new_states[plan.layer_index];
        state.compacted = true;
        state.kv_heads = selection.kv_heads;
        state.cold_row_capacity = plan.cold_capacity;
        state.hot_row_capacity = old_state.hot_row_capacity;
        state.hot_recent = old_state.hot_recent;
        state.hot_heads = old_state.hot_heads;
        state.cold_heads.resize(selection.kv_heads);
        for (uint32_t head = 0; head < selection.kv_heads; ++head) {
            auto & map = state.cold_heads[head];
            map.row_capacity = plan.cold_capacity;
            map.logical_to_physical.assign(old_cells.size(), pyramidkv_c1_invalid_cell);
            map.physical_to_logical.assign(plan.cold_capacity, pyramidkv_c1_invalid_cell);
            const auto & selected = selection.heads[head];
            if (!old_state.compacted) {
                auto & hot = state.hot_heads[head];
                for (uint32_t physical = 0; physical < hot.physical_to_logical.size(); ++physical) {
                    const uint32_t logical = hot.physical_to_logical[physical];
                    if (logical != pyramidkv_c1_invalid_cell &&
                            !std::binary_search(selected.keep_cells.begin(), selected.keep_cells.end(), logical)) {
                        hot.physical_to_logical[physical] = pyramidkv_c1_invalid_cell;
                        hot.logical_to_physical[logical] = pyramidkv_c1_invalid_cell;
                    }
                }
            }
            for (size_t physical = 0; physical < selected.keep_cells.size(); ++physical) {
                const uint32_t logical = static_cast<uint32_t>(selected.keep_cells[physical]);
                map.logical_to_physical[logical] = static_cast<uint32_t>(physical);
                map.physical_to_logical[physical] = logical;
            }
        }
    }

    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> new_ctxs_bufs;
    new_ctxs_bufs.emplace_back(std::move(new_ctx), std::move(new_buf));
    ctxs_bufs = std::move(new_ctxs_bufs);
    layers = std::move(new_layers);
    pyramidkv_c1_layers = std::move(new_states);
    turbo_rotation = new_rotation;
    turbo_rotation_inv = new_rotation_inv;
    sc_info = {};
    pyramidkv_c1_compacted = true;
    pyramidkv_c1_tokens_since_compact = 0;
    if (!pyramidkv_c1_aux_rebuild(error)) {
        return false;
    }

    LLAMA_LOG_INFO("%s: PyramidKV C1 installed per-layer/per-KV-head cold rows with F16 hot headroom using %s; copied=%zu promoted_on_kv_gpu=%zu stage=%s\n",
        __func__, ggml_backend_buft_name(kv_buffer_type), copied_cold_rows, promoted_rows,
        initial_selection ? "prefill" : "append");
    return true;
}


uint32_t llama_kv_cache::get_n_kv(const slot_info & sinfo) const {
    uint32_t result = 0;

    // pad the n_kv value so that the graph remains constant across batches and can be reused
    // note: this also helps some backends with performance (f.ex https://github.com/ggml-org/llama.cpp/pull/16812#issuecomment-3455112220)
    const uint32_t n_pad_cur = std::max(n_pad, 256u);

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        const auto & cells = v_cells[sinfo.strm[s]];
        uint32_t physical_limit = cells.size();
        if (pyramidkv_c1_hot_enabled && !pyramidkv_c1_compacted && !layers.empty() &&
                layers.front().k != nullptr) {
            physical_limit = std::min<uint32_t>(physical_limit,
                static_cast<uint32_t>(layers.front().k->ne[1]));
        }
        const uint32_t pad_floor = std::min(physical_limit, n_pad_cur);
        const uint32_t used_padded = GGML_PAD(cells.used_max_p1(), n_pad_cur);
        result = std::max(std::min(physical_limit, std::max(pad_floor, used_padded)), result);
    }

    return result;
}

ggml_tensor * llama_kv_cache::get_k(
        ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo,
        ggml_tensor * read_idxs) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto * k = layers[ikv].k;

    const uint64_t kv_size      = pyramidkv_c1_hot_enabled && !pyramidkv_c1_compacted
        ? k->ne[1] : get_size();
    const uint64_t n_embd_k_gqa = k->ne[0];

    if (n_kv > k->ne[1]) {
        throw std::runtime_error("PyramidKV C1 logical K rows exceed the bounded physical cache");
    }

    if (!llama_kv_type_is_turbo(k->type)) {
        assert(n_embd_k_gqa == hparams.n_embd_k_gqa(il));
    }

    const uint32_t n_head_kv = hparams.n_head_kv(il);
    const uint32_t n_embd_head_k = llama_kv_type_is_turbo(k->type) ?
        (pyramidkv_c1_compacted ? static_cast<uint32_t>(k->ne[0]) : n_embd_k_gqa / n_head_kv) :
        hparams.n_embd_head_k(il);

    if (pyramidkv_c1_compacted) {
        if (read_idxs == nullptr || sinfo.n_stream() != 1 || sinfo.s0 != 0 || sinfo.s1 != 0) {
            throw std::runtime_error("PyramidKV C1 K view requires one compacted stream and read indices");
        }
        // A compacted layer has its own physical row count. Gather logical
        // rows before exposing the unchanged logical [head, kv-head, key]
        // graph shape. Logical dropped rows are redirected to a harmless
        // spare row; the hybrid lane uses original-position metadata for
        // validity instead of a common logical mask.
        ggml_tensor * rows = ggml_reshape_2d(ctx, k, k->ne[0], k->ne[1] * k->ne[2]);
        rows = ggml_get_rows(ctx, rows, read_idxs);
        return ggml_reshape_4d(ctx, rows, n_embd_head_k, n_head_kv, n_kv, 1);
    }

    const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;

    return ggml_view_4d(ctx, k,
            n_embd_head_k, n_head_kv, n_kv, ns,
            ggml_row_size(k->type, n_embd_head_k),
            ggml_row_size(k->type, n_embd_k_gqa),
            ggml_row_size(k->type, n_embd_k_gqa*kv_size),
            ggml_row_size(k->type, n_embd_k_gqa*kv_size)*sinfo.s0);
}

ggml_tensor * llama_kv_cache::get_v(
        ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo,
        ggml_tensor * read_idxs) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto * v = layers[ikv].v;

    const uint64_t kv_size      = pyramidkv_c1_hot_enabled && !pyramidkv_c1_compacted
        ? v->ne[1] : get_size();
    const uint64_t n_embd_v_gqa = v->ne[0];

    if (n_kv > v->ne[1]) {
        throw std::runtime_error("PyramidKV C1 logical V rows exceed the bounded physical cache");
    }

    // [TAG_V_CACHE_VARIABLE]
    assert(n_embd_v_gqa >= hparams.n_embd_v_gqa(il));

    const uint32_t n_head_kv = hparams.n_head_kv(il);
    const uint32_t n_embd_head_v = llama_kv_type_is_turbo(v->type) ?
        (pyramidkv_c1_compacted ? static_cast<uint32_t>(v->ne[0]) : n_embd_v_gqa / n_head_kv) :
        hparams.n_embd_head_v(il);

    if (pyramidkv_c1_compacted) {
        if (read_idxs == nullptr || v_trans || sinfo.n_stream() != 1 || sinfo.s0 != 0 || sinfo.s1 != 0) {
            throw std::runtime_error("PyramidKV C1 V view requires one non-transposed stream and read indices");
        }
        ggml_tensor * rows = ggml_reshape_2d(ctx, v, v->ne[0], v->ne[1] * v->ne[2]);
        rows = ggml_get_rows(ctx, rows, read_idxs);
        return ggml_reshape_4d(ctx, rows, n_embd_head_v, n_head_kv, n_kv, 1);
    }

    const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;

    if (!v_trans) {
        // note: v->nb[1] <= v->nb[2]
        return ggml_view_4d(ctx, v,
                n_embd_head_v, n_head_kv, n_kv, ns,
                ggml_row_size(v->type, n_embd_head_v),          // v->nb[1]
                ggml_row_size(v->type, n_embd_v_gqa),                   // v->nb[2]
                ggml_row_size(v->type, n_embd_v_gqa*kv_size),           // v->nb[3]
                ggml_row_size(v->type, n_embd_v_gqa*kv_size)*sinfo.s0);
    }

    // note: v->nb[1] > v->nb[2]
    return ggml_view_4d(ctx, v,
            n_kv, n_head_kv, n_embd_head_v, ns,
            ggml_row_size(v->type, kv_size*n_embd_head_v),  // v->nb[1]
            ggml_row_size(v->type, kv_size),                        // v->nb[2]
            ggml_row_size(v->type, kv_size*n_embd_v_gqa),           // v->nb[3]
            ggml_row_size(v->type, kv_size*n_embd_v_gqa)*sinfo.s0);
}

ggml_tensor * llama_kv_cache::get_k_hot(
        ggml_context * ctx, int32_t il, uint32_t n_kv, ggml_tensor * read_idxs) const {
    const int32_t ikv = map_layer_ids.at(il);
    ggml_tensor * k_hot = layers[ikv].k_hot;
    const uint32_t n_head_kv = hparams.n_head_kv(il);
    if (!pyramidkv_c1_hot_enabled || k_hot == nullptr || read_idxs == nullptr ||
            read_idxs->ne[0] != static_cast<int64_t>(n_kv)*n_head_kv) {
        throw std::runtime_error("PyramidKV C1 hot K view has invalid head-local indices");
    }
    ggml_tensor * rows = ggml_reshape_2d(ctx, k_hot, k_hot->ne[0], k_hot->ne[1]*k_hot->ne[2]);
    rows = ggml_get_rows(ctx, rows, read_idxs);
    return ggml_reshape_4d(ctx, rows, k_hot->ne[0], n_head_kv, n_kv, 1);
}

ggml_tensor * llama_kv_cache::get_v_hot(
        ggml_context * ctx, int32_t il, uint32_t n_kv, ggml_tensor * read_idxs) const {
    const int32_t ikv = map_layer_ids.at(il);
    ggml_tensor * v_hot = layers[ikv].v_hot;
    const uint32_t n_head_kv = hparams.n_head_kv(il);
    if (!pyramidkv_c1_hot_enabled || v_hot == nullptr || read_idxs == nullptr ||
            read_idxs->ne[0] != static_cast<int64_t>(n_kv)*n_head_kv) {
        throw std::runtime_error("PyramidKV C1 hot V view has invalid head-local indices");
    }
    ggml_tensor * rows = ggml_reshape_2d(ctx, v_hot, v_hot->ne[0], v_hot->ne[1]*v_hot->ne[2]);
    rows = ggml_get_rows(ctx, rows, read_idxs);
    return ggml_reshape_4d(ctx, rows, v_hot->ne[0], n_head_kv, n_kv, 1);
}

ggml_tensor * llama_kv_cache::get_k_hybrid(ggml_context * ctx, int32_t il) const {
    const int32_t ikv = map_layer_ids.at(il);
    if (!pyramidkv_c1_hybrid_ready(il)) {
        throw std::runtime_error("PyramidKV C1 cold K hybrid view is not ready");
    }
    const auto & state = pyramidkv_c1_layers[ikv];
    const ggml_tensor * storage = layers[ikv].k;
    return ggml_reshape_4d(ctx, const_cast<ggml_tensor *>(storage), storage->ne[0],
        state.cold_row_capacity, state.kv_heads, storage->ne[2]);
}

ggml_tensor * llama_kv_cache::get_v_hybrid(ggml_context * ctx, int32_t il) const {
    const int32_t ikv = map_layer_ids.at(il);
    if (!pyramidkv_c1_hybrid_ready(il)) {
        throw std::runtime_error("PyramidKV C1 cold V hybrid view is not ready");
    }
    const auto & state = pyramidkv_c1_layers[ikv];
    const ggml_tensor * storage = layers[ikv].v;
    return ggml_reshape_4d(ctx, const_cast<ggml_tensor *>(storage), storage->ne[0],
        state.cold_row_capacity, state.kv_heads, storage->ne[2]);
}

ggml_tensor * llama_kv_cache::get_k_hot_ring(ggml_context * ctx, int32_t il) const {
    const int32_t ikv = map_layer_ids.at(il);
    const auto & state = pyramidkv_c1_layers[ikv];
    const ggml_tensor * storage = layers[ikv].k_hot;
    if (storage == nullptr || state.kv_heads == 0) {
        throw std::runtime_error("PyramidKV C1 hot K ring is not allocated");
    }
    return ggml_reshape_4d(ctx, const_cast<ggml_tensor *>(storage), storage->ne[0],
        state.hot_row_capacity, state.kv_heads, storage->ne[2]);
}

ggml_tensor * llama_kv_cache::get_v_hot_ring(ggml_context * ctx, int32_t il) const {
    const int32_t ikv = map_layer_ids.at(il);
    const auto & state = pyramidkv_c1_layers[ikv];
    const ggml_tensor * storage = layers[ikv].v_hot;
    if (storage == nullptr || state.kv_heads == 0) {
        throw std::runtime_error("PyramidKV C1 hot V ring is not allocated");
    }
    return ggml_reshape_4d(ctx, const_cast<ggml_tensor *>(storage), storage->ne[0],
        state.hot_row_capacity, state.kv_heads, storage->ne[2]);
}

ggml_tensor * llama_kv_cache::get_k_hot_hybrid(ggml_context * ctx, int32_t il) const {
    const int32_t ikv = map_layer_ids.at(il);
    if (!pyramidkv_c1_hybrid_ready(il)) {
        throw std::runtime_error("PyramidKV C1 hot K hybrid view is not ready");
    }
    const auto & state = pyramidkv_c1_layers[ikv];
    const ggml_tensor * storage = layers[ikv].k_hot;
    return ggml_reshape_4d(ctx, const_cast<ggml_tensor *>(storage), storage->ne[0],
        state.hot_row_capacity, state.kv_heads, storage->ne[2]);
}

ggml_tensor * llama_kv_cache::get_v_hot_hybrid(ggml_context * ctx, int32_t il) const {
    const int32_t ikv = map_layer_ids.at(il);
    if (!pyramidkv_c1_hybrid_ready(il)) {
        throw std::runtime_error("PyramidKV C1 hot V hybrid view is not ready");
    }
    const auto & state = pyramidkv_c1_layers[ikv];
    const ggml_tensor * storage = layers[ikv].v_hot;
    return ggml_reshape_4d(ctx, const_cast<ggml_tensor *>(storage), storage->ne[0],
        state.hot_row_capacity, state.kv_heads, storage->ne[2]);
}

ggml_tensor * llama_kv_cache::cpy_k(
        ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il,
        const slot_info & sinfo, ggml_tensor * physical_idxs) const {
    GGML_UNUSED(sinfo);

    const int32_t ikv = map_layer_ids.at(il);

    ggml_tensor * k = layers[ikv].k;

    const int64_t n_embd_head = k_cur->ne[0];
    const int64_t n_head      = k_cur->ne[1];
    const int64_t n_tokens    = k_cur->ne[2];

    if (llama_kv_type_is_turbo(k->type)) {
        const int64_t padded_head = pyramidkv_c1_compacted ? k->ne[0] : k->ne[0] / n_head;
        if (n_embd_head < padded_head) {
            k_cur = ggml_pad(ctx, k_cur, padded_head - n_embd_head, 0, 0, 0);
        }
    }

    const int64_t n_embd_gqa = k_cur->ne[0]*n_head;

    // we can merge dims 0 and 1
    // TODO: add ggml helper function for this?
    if (ggml_row_size(k_cur->type, k_cur->ne[0]) != k_cur->nb[1]) {
        k_cur = ggml_cont(ctx, k_cur);
    }

    k_cur = ggml_view_2d(ctx, k_cur, n_embd_gqa, n_tokens, k_cur->nb[2], 0);

    if (pyramidkv_c1_compacted) {
        if (physical_idxs == nullptr) {
            throw std::runtime_error("PyramidKV C1 K write requires physical indices");
        }
        if (k->ne[2] != 1 || physical_idxs->ne[0] != n_tokens*n_head) {
            throw std::runtime_error("PyramidKV C1 K write index geometry is invalid");
        }
        if (k_cur->ne[0] % n_head != 0) {
            throw std::runtime_error("PyramidKV C1 K write rows are not head-major");
        }
        k_cur = ggml_reshape_2d(ctx, k_cur, k_cur->ne[0] / n_head, n_tokens*n_head);
        k = ggml_reshape_2d(ctx, k, k->ne[0], k->ne[1] * k->ne[2]);
        return ggml_set_rows(ctx, k, k_cur, physical_idxs);
    }

    const int64_t n_stream = k->ne[2];

    if (n_stream > 1) {
        const int64_t kv_size = get_size();

        assert(n_embd_gqa == k->ne[0]);
        assert(kv_size    == k->ne[1]);

        // merge the buffer across all streams because the idxs are global
        k = ggml_reshape_2d(ctx, k, n_embd_gqa, kv_size*n_stream);
    }

    // store the current K values into the cache
    return ggml_set_rows(ctx, k, k_cur, k_idxs);
}

ggml_tensor * llama_kv_cache::cpy_v(
        ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il,
        const slot_info & sinfo, ggml_tensor * physical_idxs) const {
    GGML_UNUSED(sinfo);

    const int32_t ikv = map_layer_ids.at(il);

    auto * v = layers[ikv].v;

    const int64_t n_embd_head = v_cur->ne[0];
    const int64_t n_head      = v_cur->ne[1];
    const int64_t n_tokens    = v_cur->ne[2];

    if (llama_kv_type_is_turbo(v->type)) {
        const int64_t padded_head = pyramidkv_c1_compacted ? v->ne[0] : v->ne[0] / n_head;
        if (n_embd_head < padded_head) {
            v_cur = ggml_pad(ctx, v_cur, padded_head - n_embd_head, 0, 0, 0);
        }
    }

    const int64_t n_embd_gqa = v_cur->ne[0]*n_head;

    // we can merge dims 0 and 1
    if (ggml_row_size(v_cur->type, v_cur->ne[0]) != v_cur->nb[1]) {
        v_cur = ggml_cont(ctx, v_cur);
    }

    const int64_t n_stream = v->ne[2];

    // take this branch when FA is enabled (the V cache is not transposed)
    if (!v_trans) {
        v_cur = ggml_view_2d(ctx, v_cur, n_embd_gqa, n_tokens, v_cur->nb[2], 0);

        if (pyramidkv_c1_compacted) {
            if (physical_idxs == nullptr) {
                throw std::runtime_error("PyramidKV C1 V write requires physical indices");
            }
            if (v->ne[2] != 1 || physical_idxs->ne[0] != n_tokens*n_head) {
                throw std::runtime_error("PyramidKV C1 V write index geometry is invalid");
            }
            if (v_cur->ne[0] % n_head != 0) {
                throw std::runtime_error("PyramidKV C1 V write rows are not head-major");
            }
            v_cur = ggml_reshape_2d(ctx, v_cur, v_cur->ne[0] / n_head, n_tokens*n_head);
            v = ggml_reshape_2d(ctx, v, v->ne[0], v->ne[1] * v->ne[2]);
            return ggml_set_rows(ctx, v, v_cur, physical_idxs);
        }

        if (n_stream > 1) {
            const int64_t kv_size = get_size();

            assert(n_embd_gqa == v->ne[0]);
            assert(kv_size    == v->ne[1]);

            // merge the buffer across all streams because the idxs are global
            v = ggml_reshape_2d(ctx, v, n_embd_gqa, kv_size*n_stream);
        }

        return ggml_set_rows(ctx, v, v_cur, v_idxs);
    }

    if (pyramidkv_c1_compacted) {
        throw std::runtime_error("PyramidKV C1 does not support transposed V writes");
    }

    if (ggml_row_size(v_cur->type, n_embd_gqa) == v_cur->nb[2]) {
        // we can merge dims 0, 1 and 2
        v_cur = ggml_reshape_2d(ctx, v_cur, n_embd_gqa, n_tokens);
    } else {
        // otherwise -> make a copy to get contiguous data
        v_cur = ggml_cont_2d   (ctx, v_cur, n_embd_gqa, n_tokens);
    }

    // [TAG_V_CACHE_VARIABLE]
    if (n_embd_gqa < v->ne[0]) {
        v_cur = ggml_pad(ctx, v_cur, v->ne[0] - n_embd_gqa, 0, 0, 0);
    }

    // in this branch the v_idxs are constructed in such a way that each row is a single head element
    ggml_tensor * v_view = ggml_reshape_2d(ctx, v, 1, ggml_nelements(v));

    v_cur = ggml_reshape_2d(ctx, v_cur, 1, ggml_nelements(v_cur));

    return ggml_set_rows(ctx, v_view, v_cur, v_idxs);
}

ggml_tensor * llama_kv_cache::cpy_k_hot(
        ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const {
    const int32_t ikv = map_layer_ids.at(il);
    ggml_tensor * k_hot = layers[ikv].k_hot;
    if (!pyramidkv_c1_hot_enabled || k_hot == nullptr || k_idxs == nullptr) {
        throw std::runtime_error("PyramidKV C1 hot K storage is unavailable");
    }
    const int64_t n_head = k_cur->ne[1];
    const int64_t n_tokens = k_cur->ne[2];
    const int64_t padded_head = k_hot->ne[0];
    if (n_head <= 0 || n_tokens <= 0 || k_cur->ne[0] > padded_head ||
            k_idxs->ne[0] != n_head*n_tokens) {
        throw std::runtime_error("PyramidKV C1 hot K write geometry is invalid");
    }
    if (k_cur->ne[0] < padded_head) {
        k_cur = ggml_pad(ctx, k_cur, padded_head - k_cur->ne[0], 0, 0, 0);
    }
    if (k_cur->type != GGML_TYPE_F16) {
        k_cur = ggml_cast(ctx, k_cur, GGML_TYPE_F16);
    }
    k_cur = ggml_cont(ctx, k_cur);
    k_cur = ggml_reshape_2d(ctx, k_cur, padded_head, n_head*n_tokens);
    k_hot = ggml_reshape_2d(ctx, k_hot, padded_head, k_hot->ne[1]*k_hot->ne[2]);
    return ggml_set_rows(ctx, k_hot, k_cur, k_idxs);
}

ggml_tensor * llama_kv_cache::cpy_v_hot(
        ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const {
    const int32_t ikv = map_layer_ids.at(il);
    ggml_tensor * v_hot = layers[ikv].v_hot;
    if (!pyramidkv_c1_hot_enabled || v_hot == nullptr || v_idxs == nullptr) {
        throw std::runtime_error("PyramidKV C1 hot V storage is unavailable");
    }
    const int64_t n_head = v_cur->ne[1];
    const int64_t n_tokens = v_cur->ne[2];
    const int64_t padded_head = v_hot->ne[0];
    if (n_head <= 0 || n_tokens <= 0 || v_cur->ne[0] > padded_head ||
            v_idxs->ne[0] != n_head*n_tokens) {
        throw std::runtime_error("PyramidKV C1 hot V write geometry is invalid");
    }
    if (v_cur->ne[0] < padded_head) {
        v_cur = ggml_pad(ctx, v_cur, padded_head - v_cur->ne[0], 0, 0, 0);
    }
    if (v_cur->type != GGML_TYPE_F16) {
        v_cur = ggml_cast(ctx, v_cur, GGML_TYPE_F16);
    }
    v_cur = ggml_cont(ctx, v_cur);
    v_cur = ggml_reshape_2d(ctx, v_cur, padded_head, n_head*n_tokens);
    v_hot = ggml_reshape_2d(ctx, v_hot, padded_head, v_hot->ne[1]*v_hot->ne[2]);
    return ggml_set_rows(ctx, v_hot, v_cur, v_idxs);
}

ggml_tensor * llama_kv_cache::build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    const uint32_t n_tokens = ubatch.n_tokens;

    ggml_tensor * k_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);

    ggml_set_input(k_idxs);

    return k_idxs;
}

ggml_tensor * llama_kv_cache::build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    const uint32_t n_tokens = ubatch.n_tokens;

    ggml_tensor * v_idxs;

    if (!v_trans) {
        v_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    } else {
        v_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens*hparams.n_embd_v_gqa_max());
    }

    ggml_set_input(v_idxs);

    return v_idxs;
}

ggml_tensor * llama_kv_cache::build_input_k_rot(ggml_context * ctx) const {
    ggml_tensor * res = nullptr;

    if (attn_rot_k) {
        int nrot = 64;

        // TODO: investigate if using the smallest rotation matrix is beneficial also for K (similar as for V)
        // ref: https://github.com/ggml-org/llama.cpp/pull/21038#issuecomment-4141323088
        do {
            nrot *= 2;
        } while (n_embd_head_k_all % nrot == 0);
        nrot /= 2;

        res = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nrot, nrot);
        ggml_set_input(res);
        ggml_set_name(res, "attn_inp_k_rot");
    }

    return res;
}

ggml_tensor * llama_kv_cache::build_input_v_rot(ggml_context * ctx) const {
    ggml_tensor * res = nullptr;

    if (attn_rot_v) {
        int nrot = 64;
        // using smaller rotation matrices for V seems beneficial
        // ref: https://github.com/ggml-org/llama.cpp/pull/21038#issuecomment-4146397570
        //do {
        //    nrot *= 2;
        //} while (hparams.n_embd_head_v() % nrot == 0);
        //nrot /= 2;

        res = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nrot, nrot);
        ggml_set_input(res);
        ggml_set_name(res, "attn_inp_v_rot");
    }

    return res;
}

void llama_kv_cache::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const {
    const uint32_t n_tokens = ubatch->n_tokens;
    GGML_ASSERT(n_tokens == (int64_t) sinfo.size()*sinfo.n_stream());

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    int64_t * data = (int64_t *) dst->data;

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        const int64_t offs = sinfo.strm[s]*get_size();

        for (uint32_t i = 0; i < sinfo.size(); ++i) {
            data[s*sinfo.size() + i] = offs + sinfo.idxs[s][i];
        }
    }
}

void llama_kv_cache::set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const {
    const uint32_t n_tokens = ubatch->n_tokens;
    GGML_ASSERT(n_tokens == (int64_t) sinfo.size()*sinfo.n_stream());

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    int64_t * data = (int64_t *) dst->data;

    if (!v_trans) {
        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            const int64_t offs = sinfo.strm[s]*get_size();

            for (uint32_t i = 0; i < sinfo.size(); ++i) {
                data[s*sinfo.size() + i] = offs + sinfo.idxs[s][i];
            }
        }
    } else {
        // note: the V cache is transposed when not using flash attention
        const int64_t kv_size = get_size();

        const int64_t n_embd_v_gqa = hparams.n_embd_v_gqa_max();

        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            const int64_t offs = sinfo.strm[s]*kv_size*n_embd_v_gqa;

            for (uint32_t i = 0; i < sinfo.size(); ++i) {
                for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                    data[s*sinfo.size()*n_embd_v_gqa + i*n_embd_v_gqa + j] = offs + j*kv_size + sinfo.idxs[s][i];
                }
            }
        }
    }
}

void llama_kv_cache::set_input_k_shift(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    int32_t * data = (int32_t *) dst->data;

    for (uint32_t s = 0; s < n_stream; ++s) {
        const auto & cells = v_cells[s];

        for (uint32_t i = 0; i < cells.size(); ++i) {
            data[s*cells.size() + i] = cells.is_empty(i) ? 0 : cells.get_shift(i);
        }
    }
}

struct args_set_input_kq_mask {
    const llama_hparams & hparams;
    const llama_ubatch  * ubatch;

    const std::vector<llama_kv_cells> & v_cells;
    const std::vector<uint32_t>       & seq_to_stream;

    uint32_t       n_swa;
    llama_swa_type swa_type;

    int64_t n_kv;
    int64_t n_stream;
    int64_t n_tps;
    const std::vector<int32_t> * local_cells;
};

template<typename T, bool causal, bool swa, bool is_2d, bool alibi>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
  //const auto & hparams = args.hparams;
    const auto & ubatch  = args.ubatch;

    const auto & v_cells       = args.v_cells;
    const auto & seq_to_stream = args.seq_to_stream;

    const uint32_t       n_swa    = args.n_swa;
    const llama_swa_type swa_type = args.swa_type;

    const int64_t n_kv     = args.n_kv;
    const int64_t n_stream = args.n_stream;
    const int64_t n_tps    = args.n_tps;

    const T mask_keep = llama_cast<T>(0.0f);
    const T mask_drop = llama_cast<T>(-INFINITY);

    // the min position in the batch for each sequence
    llama_pos seq_pos_min[LLAMA_MAX_SEQ];
    std::fill(seq_pos_min, seq_pos_min + LLAMA_MAX_SEQ, INT32_MAX);

    for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
        const llama_seq_id seq_id = ubatch->seq_id[i][0];

        seq_pos_min[seq_id] = std::min(seq_pos_min[seq_id], ubatch->pos[i]);
    }

    for (uint32_t s = 0; s < n_stream; ++s) {
        // bookkeeping of the KQ mask cells that could change for other tokens of the same sequence
        std::unordered_map<llama_seq_id, uint32_t>              seq_srct;
        std::unordered_map<llama_seq_id, std::vector<uint32_t>> seq_idxs;

        for (uint32_t ii = 0; ii < n_tps; ++ii) {
            const uint32_t i = s*n_tps + ii;

            const llama_seq_id seq_id = ubatch->seq_id[i][0];

            const auto & cells = v_cells.at(seq_to_stream[seq_id]);

                  llama_pos p0 = -1;
            const llama_pos p1 = ubatch->pos[i];

            // for M-RoPE
            const llama_pos p1_x = is_2d ? ubatch->pos[i + ubatch->n_tokens*2] : 0;
            const llama_pos p1_y = is_2d ? ubatch->pos[i + ubatch->n_tokens]   : 0;

            const uint64_t idst = n_kv*i;

            // for tokens of the same sequence, the mask is mostly the same, so we can reuse it
            // the only cells that could change are the ones that are with similar positions as the
            //   ones in the batch (i.e. due to causal masking, SWA, etc.)
            // keep track of those cells and shortcut the loop to save time
            // note: this optimization is not compatible with Alibi position encoding
            // ref:  https://github.com/ggml-org/llama.cpp/pull/18842
            bool prev = false;

            auto & idxs = seq_idxs[seq_id];

            if (!alibi) {
                if (seq_srct.find(seq_id) != seq_srct.end()) {
                    const uint32_t srct = seq_srct[seq_id];

                    const uint64_t idst_prev = n_kv*srct;

                    std::copy(data + idst_prev, data + idst_prev + n_kv, data + idst);

                    prev = true;
                } else {
                    idxs.clear();
                    idxs.reserve(ubatch->n_tokens + n_swa + 32);

                    seq_srct[seq_id] = i;
                }
            }

            for (uint32_t jj = 0; jj < n_kv; ++jj) {
                uint32_t j = jj;
                uint32_t cell_j;

                // we have an exiting mask for this sequence -> update just seq_idxs
                if (!alibi) {
                    if (prev) {
                        if (jj >= idxs.size()) {
                            break;
                        }

                        j = idxs[jj];
                    }
                }

                if (args.local_cells != nullptr && j >= args.local_cells->size()) {
                    goto skip;
                }
                cell_j = args.local_cells != nullptr ? (*args.local_cells)[j] : j;
                if (cells.is_empty(cell_j)) {
                    goto skip;
                }

                // mask the token if not the same sequence
                if (!cells.seq_has(cell_j, seq_id)) {
                    goto skip;
                }

                p0 = cells.pos_get(cell_j);

                if (!alibi) {
                    if (!prev) {
                        // record all cells for which: p0 >= seq_pos_min[seq_id] - n_swa - 32
                        if (p0 + (int32_t) (n_swa + 32) >= seq_pos_min[seq_id]) {
                            idxs.push_back(j);
                        }
                    }
                }

                if (causal) {
                    // mask future tokens
                    if (p0 > p1) {
                        goto skip;
                    }

                    // M-RoPE causal mask
                    if (is_2d) {
                        if (p0 == p1) {
                            const auto & p0_ext = cells.ext_get(cell_j);

                            if (p0_ext.is_2d_gt(p1_x, p1_y)) {
                                goto skip;
                            }
                        }
                    }
                }

                // apply SWA if any
                if (swa) {
                    if (llama_hparams::is_masked_swa(n_swa, swa_type, p0, p1)) {
                        goto skip;
                    }
                }

                if (alibi) {
                    data[idst + j] = llama_cast<T>(static_cast<float>(-std::abs(p0 - p1)));
                } else {
                    data[idst + j] = mask_keep;
                }

                continue;
skip:
                data[idst + j] = mask_drop;
            }
        }
    }
}

template<typename T, bool causal, bool swa, bool is_2d>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
    const bool alibi = args.hparams.use_alibi;
    if (alibi) {
        set_input_kq_mask_impl<T, causal, swa, is_2d, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, causal, swa, is_2d, false>(args, data);
    }
}

template<typename T, bool causal, bool swa>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
    const bool is_2d = args.ubatch->is_pos_2d();
    if (is_2d) {
        set_input_kq_mask_impl<T, causal, swa, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, causal, swa, false>(args, data);
    }
}

template<typename T, bool causal>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
    const bool swa = args.swa_type != LLAMA_SWA_TYPE_NONE;
    if (swa) {
        set_input_kq_mask_impl<T, causal, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, causal, false>(args, data);
    }
}

template<typename T>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data, bool causal_attn) {
    if (causal_attn) {
        set_input_kq_mask_impl<T, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, false>(args, data);
    }
}

bool llama_kv_cache::supports_compact_mask(const llama_ubatch & ubatch) const {
    // Narrow exact lane: one stream/owner, ordinary positions, no ALiBi or SWA.
    // Other geometries keep their existing host implementation.
    if (n_stream != 1 || hparams.use_alibi || swa_type != LLAMA_SWA_TYPE_NONE ||
            ubatch.is_pos_2d() || ubatch.n_seqs_unq != 1 || ubatch.n_tokens < 32 || ubatch.n_tokens > 4096) return false;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        if (ubatch.n_seq_id[i] != 1 || ubatch.seq_id[i][0] != ubatch.seq_id[0][0] || ubatch.pos[i] < 0) return false;
    }
    return true;
}

void llama_kv_cache::set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn,
        const std::vector<int32_t> * local_cells) const {
    const uint32_t n_tokens = ubatch->n_tokens;

    if (std::strcmp(dst->name, "attn_inp_kq_mask_compact") == 0) {
        GGML_ASSERT(local_cells == nullptr);
        if (!supports_compact_mask(*ubatch) || dst->ne[0] <= 0 || dst->ne[0] > (1 << 20))
            throw std::runtime_error("compact mask geometry changed");
        ggml_backend_t backend = nullptr;
        std::memcpy(&backend, dst->op_params, sizeof(backend));
        using Fill = bool (*)(ggml_backend_t, ggml_tensor *, const int32_t *, const int32_t *, bool);
        auto device = ggml_backend_get_device(backend);
        auto fill = (Fill) ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(device),
                "ggml_backend_kq_mask_v1");
        if (!fill) throw std::runtime_error("compact mask backend unavailable");
        const auto seq = ubatch->seq_id[0][0];
        const auto & cells = v_cells[0];
        std::vector<int32_t> positions(dst->ne[0], -1);
        for (size_t i = 0; i < positions.size(); ++i)
            if (!cells.is_empty(i) && cells.seq_has(i, seq)) positions[i] = cells.pos_get(i);
        if (!fill(backend, dst, positions.data(), ubatch->pos, causal_attn))
            throw std::runtime_error("compact mask transfer failed");
        return;
    }

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    const int64_t n_kv     = dst->ne[0];
    const int64_t n_stream = dst->ne[3]; // num streams in the current ubatch

    GGML_ASSERT(n_tokens%n_stream == 0);

    // n_tps == n_tokens_per_stream
    const int64_t n_tps = n_tokens/n_stream;

    //const int64_t t_start = ggml_time_us();

    const args_set_input_kq_mask args = {
        /*.hparams          =*/ hparams,
        /*.ubatch           =*/ ubatch,
        /*.v_cells          =*/ v_cells,
        /*.seq_to_stream    =*/ seq_to_stream,
        /*.n_swa            =*/ n_swa,
        /*.swa_type         =*/ swa_type,
        /*.n_kv             =*/ n_kv,
        /*.n_stream         =*/ n_stream,
        /*.n_tps            =*/ n_tps,
        /*.local_cells      =*/ local_cells,
    };

    if (dst->type == GGML_TYPE_F16) {
        set_input_kq_mask_impl<ggml_fp16_t>(args, (ggml_fp16_t *) dst->data, causal_attn);
    } else {
        set_input_kq_mask_impl<float>(args, (float *) dst->data, causal_attn);
    }

    //const int64_t t_end = ggml_time_us();

    //LLAMA_LOG_ERROR("%s: kq mask time: %0.3f ms\n", __func__, (t_end - t_start)/1000.0);
}

void llama_kv_cache::set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(n_stream == 1 && "TODO: support multiple streams");
    const auto & cells = v_cells[0];

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    GGML_ASSERT(!ubatch->equal_seqs()); // TODO: use ubatch->n_seqs instead of failing

    int32_t * data = (int32_t *) dst->data;

    const int32_t n_kv = dst->ne[0];

    for (int h = 0; h < 1; ++h) {
        for (int i = 0; i < n_tokens; ++i) {
            for (int j = 0; j < n_kv; ++j) {
                // the position when the cells is empty is irrelevant - it will be masked out later in the attention
                const llama_pos p0 = cells.is_empty(j) ? -1 : cells.pos_get(j);

                data[h*(n_kv*n_tokens) + i*n_kv + j] = llama_relative_position_bucket(p0, ubatch->pos[i], hparams.n_rel_attn_bkts, false);
            }
        }
    }
}

void llama_kv_cache::set_input_k_rot(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    const auto n_rot = dst->ne[0];
    GGML_ASSERT(attn_rot_hadamard.count(dst->ne[0]));

    memcpy(dst->data, attn_rot_hadamard.at(n_rot).data(), ggml_nbytes(dst));
}

void llama_kv_cache::set_input_v_rot(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    const auto n_rot = dst->ne[0];
    GGML_ASSERT(attn_rot_hadamard.count(dst->ne[0]));

    memcpy(dst->data, attn_rot_hadamard.at(n_rot).data(), ggml_nbytes(dst));
}

bool llama_kv_cache::has_cell_ext() const {
    return hparams.n_pos_per_embd() > 1;
}

void llama_kv_cache::get_prev_tokens(const llama_ubatch & ubatch, uint32_t n, std::vector<llama_token> & res) const {
    const uint32_t n_tokens = ubatch.n_tokens;

    res.clear();
    res.resize(n_tokens*n, LLAMA_TOKEN_NULL);

    if (n == 0) {
        return;
    }

    // note: apply_ubatch() has already stored the current ubatch
    //       the window below thus covers tokens of this very ubatch as well, which is what we want
    llama_pos p_min = std::numeric_limits<llama_pos>::max();
    llama_pos p_max = std::numeric_limits<llama_pos>::min();

    std::bitset<LLAMA_MAX_SEQ> seqs;

    for (uint32_t i = 0; i < n_tokens; ++i) {
        p_min = std::min(p_min, ubatch.pos[i]);
        p_max = std::max(p_max, ubatch.pos[i]);
    }

    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
        seqs.set(ubatch.seq_id_unq[s]);
    }

    // (seq_id, pos) -> token, for every cell that could be a predecessor of a ubatch token
    std::unordered_map<uint64_t, llama_token> hist;

    const auto key = [](llama_seq_id seq_id, llama_pos pos) {
        return ((uint64_t) seq_id << 32) | (uint32_t) pos;
    };

    for (uint32_t s = 0; s < n_stream; ++s) {
        v_cells[s].for_each_token_in(seqs, p_min - (llama_pos) n, p_max,
            [&](llama_seq_id seq_id, llama_pos pos, llama_token tok) {
                hist[key(seq_id, pos)] = tok;
            });
    }

    for (uint32_t i = 0; i < n_tokens; ++i) {
        // TODO: a token that belongs to more than one sequence has an ambiguous history.
        //       the n-gram architectures have to reject such batches
        const llama_seq_id seq_id = ubatch.seq_id[i][0];

        for (uint32_t j = 0; j < n; ++j) {
            const llama_pos p = ubatch.pos[i] - (llama_pos) (n - j);
            if (p < 0) {
                continue;
            }

            const auto it = hist.find(key(seq_id, p));
            if (it != hist.end()) {
                res[i*n + j] = it->second;
            }
        }
    }
}

size_t llama_kv_cache::total_size() const {
    size_t size = 0;

    for (const auto & [_, buf] : ctxs_bufs) {
        size += ggml_backend_buffer_get_size(buf.get());
    }
    if (tq4_key_anchor_buf) {
        size += ggml_backend_buffer_get_size(tq4_key_anchor_buf.get());
    }

    return size;
}

size_t llama_kv_cache::size_k_bytes() const {
    size_t size_k_bytes = 0;

    for (const auto & layer : layers) {
        size_k_bytes += ggml_nbytes(layer.k);
        size_k_bytes += layer.k_hot ? ggml_nbytes(layer.k_hot) : 0;
    }

    return size_k_bytes;
}

size_t llama_kv_cache::size_v_bytes() const {
    size_t size_v_bytes = 0;

    for (const auto & layer : layers) {
        size_v_bytes += layer.v ? ggml_nbytes(layer.v) : 0;
        size_v_bytes += layer.v_hot ? ggml_nbytes(layer.v_hot) : 0;
    }

    return size_v_bytes;
}

ggml_tensor * llama_kv_cache::build_rope_shift(
        const llama_cparams & cparams,
               ggml_context * ctx,
                ggml_tensor * cur,
                ggml_tensor * shift,
                ggml_tensor * rot,
                ggml_tensor * factors,
                      float   freq_base,
                      float   freq_scale,
                   uint32_t   il) const {
    if (llama_kv_type_is_turbo(cur->type)) {
        throw std::runtime_error("TurboQuant4 KV cache does not support K-shift re-quantization");
    }

    const auto & n_ctx_orig = cparams.n_ctx_orig_yarn;

    const auto & yarn_ext_factor  = cparams.yarn_ext_factor;
    const auto & yarn_beta_fast   = cparams.yarn_beta_fast;
    const auto & yarn_beta_slow   = cparams.yarn_beta_slow;
    const auto & yarn_attn_factor = cparams.yarn_attn_factor;

    const auto & n_rot     = hparams.n_rot(il);
    const auto & rope_type = hparams.rope_type == LLAMA_ROPE_TYPE_MROPE || hparams.rope_type == LLAMA_ROPE_TYPE_IMROPE
                                // @ngxson : this is a workaround
                                // for M-RoPE, we want to rotate the whole vector when doing KV shift
                                // a normal RoPE should work, we just need to use the correct ordering
                                // ref: https://github.com/ggml-org/llama.cpp/pull/13870
                                ? LLAMA_ROPE_TYPE_NEOX
                                : hparams.rope_type;
    ggml_tensor * tmp;

    if (ggml_is_quantized(cur->type)) {
        // dequantize to f32 -> RoPE -> quantize back
        tmp = ggml_cast(ctx, cur, GGML_TYPE_F32);

        // rotate back
        tmp = llama_mul_mat_hadamard(ctx, tmp, rot);

        tmp = ggml_rope_ext(ctx, tmp,
                shift, factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                yarn_ext_factor, yarn_attn_factor, yarn_beta_fast, yarn_beta_slow);

        // rotate fwd
        tmp = llama_mul_mat_hadamard(ctx, tmp, rot);

        tmp = ggml_cpy(ctx, tmp, cur);
    } else {
        // we rotate only the first n_rot dimensions
        tmp = ggml_rope_ext_inplace(ctx, cur,
                shift, factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                yarn_ext_factor, yarn_attn_factor, yarn_beta_fast, yarn_beta_slow);
    }

    return tmp;
}

class llm_graph_input_k_shift : public llm_graph_input_i {
public:
    llm_graph_input_k_shift(const llama_kv_cache * kv_self) : kv_self(kv_self) {}
    virtual ~llm_graph_input_k_shift() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * k_shift; // I32 [kv_size*n_stream]

    // note: assumes k_rot^2 == I
    ggml_tensor * k_rot = nullptr;

    const llama_kv_cache * kv_self;
};

void llm_graph_input_k_shift::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    if (k_shift) {
        kv_self->set_input_k_shift(k_shift);
    }

    if (k_rot) {
        kv_self->set_input_k_rot(k_rot);
    }
}

ggml_cgraph * llama_kv_cache::build_graph_shift(llm_graph_result * res, llama_context * lctx) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    GGML_ASSERT(!other);

    if (llama_kv_type_is_turbo(type_k())) {
        throw std::runtime_error("TurboQuant4 KV cache does not support K-shift");
    }

    auto * ctx = res->get_ctx();
    auto * gf  = res->get_gf();

    auto inp = std::make_unique<llm_graph_input_k_shift>(this);

    inp->k_shift = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) get_size()*n_stream);
    ggml_set_input(inp->k_shift);

    inp->k_rot = build_input_k_rot(ctx);

    const auto & cparams = lctx->get_cparams();

    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        if (!hparams.has_rope(il)) {
            continue;
        }

        const int64_t n_head_kv    = hparams.n_head_kv(il);
        const int64_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        const auto n_rot         = hparams.n_rot(il);
        const auto n_embd_head_k = hparams.n_embd_head_k(il);
        const auto n_embd_nope   = hparams.n_lora_kv > 0 ? n_embd_head_k - n_rot : 0;

        const float freq_base_l  = model.get_rope_freq_base (cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);

        ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

        ggml_tensor * k =
            ggml_view_3d(ctx, layer.k,
                n_rot, n_head_kv, get_size()*n_stream,
                ggml_row_size(layer.k->type, n_embd_head_k),
                ggml_row_size(layer.k->type, n_embd_k_gqa),
                ggml_row_size(layer.k->type, n_embd_nope));

        ggml_tensor * cur = build_rope_shift(cparams, ctx, k, inp->k_shift, inp->k_rot, rope_factors, freq_base_l, freq_scale_l, il);

        ggml_build_forward_expand(gf, cur);
    }

    res->add_input(std::move(inp));

    return gf;
}

void llama_kv_cache::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    if (tq4_key_center_enabled_flag) {
        throw std::runtime_error(
            "TQ4 key-center state_write is disabled until the anchor has a versioned state format");
    }
    if (pyramidkv_c1_hot_enabled) {
        throw std::runtime_error(
            "PyramidKV C1 state_write is disabled: the bounded cold/hot layout has no format version/remap");
    }

    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_UNUSED(flags);

    io.write(&n_stream, sizeof(n_stream));

    for (uint32_t s = 0; s < n_stream; ++s) {
        cell_ranges_t cr { s, {} };

        uint32_t cell_count = 0;

        const auto & cells = v_cells[s];

        // Count the number of cells with the specified seq_id
        // Find all the ranges of cells with this seq id (or all, when -1)
        uint32_t cell_range_begin = cells.size();

        for (uint32_t i = 0; i < cells.size(); ++i) {
            bool add_cell = true;

            add_cell = add_cell && !cells.is_empty(i);
            add_cell = add_cell && (seq_id == -1 || cells.seq_has(i, seq_id));

            // check the cell is not SWA-masked
            if (add_cell && seq_id != -1) {
                const bool is_masked = llama_hparams::is_masked_swa(n_swa, swa_type, cells.pos_get(i), cells.seq_pos_max(seq_id));

                add_cell = !is_masked;
            }

            if (add_cell) {
                ++cell_count;
                if (cell_range_begin == cells.size()) {
                    cell_range_begin = i;
                }
            } else {
                if (cell_range_begin != cells.size()) {
                    cr.data.emplace_back(cell_range_begin, i);
                    cell_range_begin = cells.size();
                }
            }
        }

        if (cell_range_begin != cells.size()) {
            cr.data.emplace_back(cell_range_begin, cells.size());
        }

        // DEBUG CHECK: Sum of cell counts in ranges should equal the total cell count
        uint32_t cell_count_check = 0;
        for (const auto & range : cr.data) {
            cell_count_check += range.second - range.first;
        }
        GGML_ASSERT(cell_count == cell_count_check);

        io.write(&cell_count, sizeof(cell_count));

        // skip empty streams
        if (cell_count == 0) {
            continue;
        }

        state_write_meta(io, cr, seq_id);
        state_write_data(io, cr);
    }
}

void llama_kv_cache::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    quest_meta_dirty = true;
    if (tq4_key_center_enabled_flag) {
        throw std::runtime_error(
            "TQ4 key-center state_read is disabled until the anchor has a versioned state format");
    }
    if (pyramidkv_c1_hot_enabled) {
        throw std::runtime_error(
            "PyramidKV C1 state_read is disabled: the bounded cold/hot layout has no format version/remap");
    }

    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_UNUSED(flags);

    // TODO: fix incosistent handling of `seq_id < 0` and `seq_id == -1` in the codebase [TAG_LLAMA_SEQ_ID_NEG]
    GGML_ASSERT(seq_id == -1 || (seq_id >= 0 && (size_t) seq_id < seq_to_stream.size()));

    uint32_t n_stream_cur;
    io.read(&n_stream_cur, sizeof(n_stream_cur));
    if (n_stream_cur != n_stream) {
        throw std::runtime_error("n_stream mismatch");
    }

    for (uint32_t s = 0; s < n_stream; ++s) {
        uint32_t cell_count;
        io.read(&cell_count, sizeof(cell_count));

        if (cell_count == 0) {
            continue;
        }

        const uint32_t strm = seq_id == -1 ? s : seq_to_stream[seq_id];

        slot_info sinfo;

        bool res = true;
        res = res && state_read_meta(io, strm, cell_count, sinfo, seq_id);

        try {
            res = res && state_read_data(io, strm, cell_count, sinfo);
        } catch (...) {
            res = false;
        }

        if (!res) {
            if (seq_id == -1) {
                clear(true);
            } else {
                seq_rm(seq_id, -1, -1);
            }
            throw std::runtime_error("failed to restore kv cache");
        }
    }
}

void llama_kv_cache::state_write_meta(llama_io_write_i & io, const cell_ranges_t & cr, llama_seq_id seq_id) const {
    const auto & cells = v_cells[cr.strm];

    for (const auto & range : cr.data) {
        for (uint32_t i = range.first; i < range.second; ++i) {
            std::vector<llama_seq_id> seq_ids;

            for (llama_seq_id cur = 0; cur < (int) n_seq_max; ++cur) {
                if (cur == seq_id || seq_id == -1) {
                    if (cells.seq_has(i, cur)) {
                        seq_ids.push_back(cur);
                    }
                }
            }

            const llama_pos pos     = cells.pos_get(i);
            const uint32_t n_seq_id = seq_ids.size();

            io.write(&pos,      sizeof(pos));
            io.write(&n_seq_id, sizeof(n_seq_id));

            if (has_cell_ext()) {
                const llama_kv_cell_ext ext = cells.ext_get(i);
                io.write(&ext, sizeof(ext));
            }

            for (const auto & seq_id : seq_ids) {
                io.write(&seq_id, sizeof(seq_id));
            }
        }
    }
}

void llama_kv_cache::state_write_data(llama_io_write_i & io, const cell_ranges_t & cr) const {
    const auto & cells = v_cells[cr.strm];

    const uint32_t v_trans = this->v_trans ? 1 : 0;
    const uint32_t n_layer = layers.size();

    io.write(&v_trans, sizeof(v_trans));
    io.write(&n_layer, sizeof(n_layer));

    // Iterate and write all the keys first, each row is a cell
    // Get whole range at a time
    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        auto * k = layer.k_stream[cr.strm];
        const uint32_t n_embd_k_gqa = llama_kv_type_is_turbo(k->type) ? k->ne[0] : hparams.n_embd_k_gqa(il);

        // Write key type
        const int32_t k_type_i = (int32_t) k->type;
        io.write(&k_type_i, sizeof(k_type_i));

        // Write row size of key
        const uint64_t k_size_row = ggml_row_size(k->type, n_embd_k_gqa);
        io.write(&k_size_row, sizeof(k_size_row));

        // Read each range of cells of k_size length and write out
        for (const auto & range : cr.data) {
            const size_t range_size = range.second - range.first;
            const size_t buf_size = range_size * k_size_row;
            io.write_tensor(k, range.first * k_size_row, buf_size);
        }
    }

    if (!v_trans) {
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            auto * v = layer.v_stream[cr.strm];
            if (!v) {
                continue;
            }
            const uint32_t n_embd_v_gqa = llama_kv_type_is_turbo(v->type) ? v->ne[0] : hparams.n_embd_v_gqa(il);

            // Write value type
            const int32_t v_type_i = (int32_t) v->type;
            io.write(&v_type_i, sizeof(v_type_i));

            // Write row size of value
            const uint64_t v_size_row = ggml_row_size(v->type, n_embd_v_gqa);
            io.write(&v_size_row, sizeof(v_size_row));

            // Read each range of cells of v_size length and write out
            for (const auto & range : cr.data) {
                const size_t range_size = range.second - range.first;
                const size_t buf_size = range_size * v_size_row;
                io.write_tensor(v, range.first * v_size_row, buf_size);
            }
        }
    } else {
        // When v is transposed, we also need the element size and get the element ranges from each row
        const uint32_t kv_size = cells.size();

        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            auto * v = layer.v_stream[cr.strm];
            if (!v) {
                continue;
            }
            const uint32_t n_embd_v_gqa = llama_kv_type_is_turbo(v->type) ? v->ne[0] : hparams.n_embd_v_gqa(il);

            // Write value type
            const int32_t v_type_i = (int32_t) v->type;
            io.write(&v_type_i, sizeof(v_type_i));

            // Write element size
            const uint32_t v_size_el = ggml_type_size(v->type);
            io.write(&v_size_el, sizeof(v_size_el));

            // Write GQA embedding size
            io.write(&n_embd_v_gqa, sizeof(n_embd_v_gqa));

            // For each row, we get the element values of each cell
            for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                // Read each range of cells of v_size_el length and write out
                for (const auto & range : cr.data) {
                    const size_t range_size = range.second - range.first;
                    const size_t src_offset = (range.first + j * kv_size) * v_size_el;
                    const size_t buf_size = range_size * v_size_el;
                    io.write_tensor(v, src_offset, buf_size);
                }
            }
        }
    }
}

bool llama_kv_cache::state_read_meta(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, slot_info & sinfo, llama_seq_id dest_seq_id) {
    auto & cells = v_cells[strm];
    auto & head  = v_heads[strm];

    if (dest_seq_id != -1) {
        // single sequence
        seq_rm(dest_seq_id, -1, -1);

        llama_batch_allocr balloc(hparams.n_pos_per_embd());

        llama_ubatch ubatch = balloc.ubatch_reserve(cell_count, 1);

        ubatch.seq_id_unq[0] = dest_seq_id;

        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t n_seq_id;

            io.read(&pos,      sizeof(pos));
            io.read(&n_seq_id, sizeof(n_seq_id));

            if (n_seq_id != 1) {
                LLAMA_LOG_ERROR("%s: invalid seq_id-agnostic kv cell\n", __func__);
                return false;
            }

            if (has_cell_ext()) {
                llama_kv_cell_ext ext;
                io.read(&ext, sizeof(ext));

                if (hparams.n_pos_per_embd() > 1) {
                    ubatch.pos[i + ubatch.n_tokens]   = ext.y;
                    ubatch.pos[i + ubatch.n_tokens*2] = ext.x;
                }

                // apply_ubatch() below restores ext.tok from the ubatch tokens
                ubatch.token[i] = ext.tok;
            }

            // read the sequence id, but directly discard it - we will use dest_seq_id instead
            {
                llama_seq_id seq_id;
                io.read(&seq_id, sizeof(seq_id));
            }

            ubatch.pos[i]      = pos;
            ubatch.n_seq_id[i] = n_seq_id;
            ubatch.seq_id[i]   = &dest_seq_id;
        }

        sinfo = find_slot(ubatch, false);
        if (sinfo.empty()) {
            LLAMA_LOG_ERROR("%s: failed to find %d available cells in kv cache\n", __func__,  cell_count);
            return false;
        }

        // note: apply_ubatch() rebuilds llama_kv_cell_ext from the ubatch
        //       only ext.tok and the M-RoPE 2D position round-trip through it
        //       see: https://github.com/ggml-org/llama.cpp/pull/16825#issuecomment-3460868350
        apply_ubatch(sinfo, ubatch);

        LLAMA_LOG_DEBUG("%s: cell_count = %d, dest_seq_id = %d\n", __func__, cell_count, dest_seq_id);

        // DEBUG CHECK: verify that all cells were allocated and have correct seq_id and pos values
        GGML_ASSERT(sinfo.n_stream() == 1);
        GGML_ASSERT(sinfo.idxs[0].size() == cell_count);
        for (uint32_t i = 0; i < cell_count; ++i) {
            const uint32_t idx = sinfo.idxs[0][i];
            GGML_ASSERT(cells.pos_get(idx) == ubatch.pos[i]);
            GGML_ASSERT(cells.seq_has(idx, dest_seq_id));
        }
    } else {
        // whole KV cache restore

        if (cell_count > cells.size()) {
            LLAMA_LOG_ERROR("%s: not enough cells in kv cache\n", __func__);
            return false;
        }

        clear(true);

        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t  n_seq_id;

            io.read(&pos,      sizeof(pos));
            io.read(&n_seq_id, sizeof(n_seq_id));

            cells.pos_set(i, pos);

            if (has_cell_ext()) {
                llama_kv_cell_ext ext;
                io.read(&ext, sizeof(ext));
                cells.ext_set(i, ext);
            }

            for (uint32_t j = 0; j < n_seq_id; ++j) {
                llama_seq_id seq_id;
                io.read(&seq_id, sizeof(seq_id));

                if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max) {
                    LLAMA_LOG_ERROR("%s: invalid seq_id, %d is out of range [0, %u)\n", __func__, seq_id, n_seq_max);
                    return false;
                }

                cells.seq_add(i, seq_id);
            }
        }

        // Create contiguous slot_info for whole cache restore
        sinfo.s0 = strm;
        sinfo.s1 = strm;
        sinfo.resize(1);
        sinfo.strm[0] = strm;
        sinfo.idxs[0].resize(cell_count);
        for (uint32_t i = 0; i < cell_count; ++i) {
            sinfo.idxs[0][i] = i;
        }

        head = 0;
    }

    return true;
}

bool llama_kv_cache::state_read_data(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, const slot_info & sinfo) {
    auto & cells = v_cells[strm];

    uint32_t v_trans;
    uint32_t n_layer;

    io.read(&v_trans, sizeof(v_trans));
    io.read(&n_layer, sizeof(n_layer));

    if (n_layer != layers.size()) {
        LLAMA_LOG_ERROR("%s: mismatched layer count (%u instead of %u)\n", __func__, n_layer, (uint32_t) layers.size());
        return false;
    }

    if (cell_count > cells.size()) {
        LLAMA_LOG_ERROR("%s: not enough cells in kv cache to restore state (%u > %u)\n", __func__, cell_count, cells.size());
        return false;
    }

    if (this->v_trans != (bool) v_trans) {
        LLAMA_LOG_ERROR("%s: incompatible V transposition\n", __func__);
        return false;
    }

    // For each layer, read the keys for each cell, one row is one cell, read as one contiguous block
    for (const auto & layer : layers) {
        const uint32_t il = layer.il;
        auto * k = layer.k_stream[strm];
        const uint32_t n_embd_k_gqa = llama_kv_type_is_turbo(k->type) ? k->ne[0] : hparams.n_embd_k_gqa(il);

        // Read type of key
        int32_t k_type_i_ref;
        io.read(&k_type_i_ref, sizeof(k_type_i_ref));
        const int32_t k_type_i = (int32_t) k->type;
        if (k_type_i != k_type_i_ref) {
            LLAMA_LOG_ERROR("%s: mismatched key type (%d != %d, layer %d)\n", __func__, k_type_i, k_type_i_ref, il);
            return false;
        }

        // Read row size of key
        uint64_t k_size_row_ref;
        io.read(&k_size_row_ref, sizeof(k_size_row_ref));
        const size_t k_size_row = ggml_row_size(k->type, n_embd_k_gqa);
        if (k_size_row != k_size_row_ref) {
            LLAMA_LOG_ERROR("%s: mismatched key row size (%zu != %zu, layer %d)\n", __func__, k_size_row, (size_t) k_size_row_ref, il);
            return false;
        }

        if (cell_count) {
            if (sinfo.is_contiguous()) {
                // Fast path: contiguous cells, single memcpy
                io.read_tensor(k, sinfo.head() * k_size_row, cell_count * k_size_row);
            } else {
                // Slow path: scatter to non-contiguous positions
                for (uint32_t i = 0; i < cell_count; ++i) {
                    const size_t dst_offset = sinfo.idxs[0][i] * k_size_row;
                    io.read_tensor(k, dst_offset, k_size_row);
                }
            }
        }
    }

    if (!this->v_trans) {
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;
            auto * v = layer.v_stream[strm];
            if (!v) {
                continue;
            }
            const uint32_t n_embd_v_gqa = llama_kv_type_is_turbo(v->type) ? v->ne[0] : hparams.n_embd_v_gqa(il);

            // Read type of value
            int32_t v_type_i_ref;
            io.read(&v_type_i_ref, sizeof(v_type_i_ref));
            const int32_t v_type_i = (int32_t) v->type;
            if (v_type_i != v_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)\n", __func__, v_type_i, v_type_i_ref, il);
                return false;
            }

            // Read row size of value
            uint64_t v_size_row_ref;
            io.read(&v_size_row_ref, sizeof(v_size_row_ref));
            const size_t v_size_row = ggml_row_size(v->type, n_embd_v_gqa);
            if (v_size_row != v_size_row_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value row size (%zu != %zu, layer %d)\n", __func__, v_size_row, (size_t) v_size_row_ref, il);
                return false;
            }

            if (cell_count) {
                if (sinfo.is_contiguous()) {
                    // Fast path: contiguous cells, single memcpy
                    io.read_tensor(v, sinfo.head() * v_size_row, cell_count * v_size_row);
                } else {
                    // Slow path: scatter to non-contiguous positions
                    for (uint32_t i = 0; i < cell_count; ++i) {
                        const size_t dst_offset = sinfo.idxs[0][i] * v_size_row;
                        io.read_tensor(v, dst_offset, v_size_row);
                    }
                }
            }
        }
    } else {
        // For each layer, read the values for each cell (transposed)
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;
            auto * v = layer.v_stream[strm];
            if (!v) {
                continue;
            }
            const uint32_t n_embd_v_gqa = llama_kv_type_is_turbo(v->type) ? v->ne[0] : hparams.n_embd_v_gqa(il);

            // Read type of value
            int32_t v_type_i_ref;
            io.read(&v_type_i_ref, sizeof(v_type_i_ref));
            const int32_t v_type_i = (int32_t) v->type;
            if (v_type_i != v_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)\n", __func__, v_type_i, v_type_i_ref, il);
                return false;
            }

            // Read element size of value
            uint32_t v_size_el_ref;
            io.read(&v_size_el_ref, sizeof(v_size_el_ref));
            const size_t v_size_el = ggml_type_size(v->type);
            if (v_size_el != v_size_el_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value element size (%zu != %zu, layer %d)\n", __func__, v_size_el, (size_t) v_size_el_ref, il);
                return false;
            }

            // Read GQA embedding size
            uint32_t n_embd_v_gqa_ref;
            io.read(&n_embd_v_gqa_ref, sizeof(n_embd_v_gqa_ref));
            if (n_embd_v_gqa != n_embd_v_gqa_ref) {
                LLAMA_LOG_ERROR("%s: mismatched GQA embedding size (%u != %u, layer %d)\n", __func__, n_embd_v_gqa, n_embd_v_gqa_ref, il);
                return false;
            }

            if (cell_count) {
                if (sinfo.is_contiguous()) {
                    // Fast path: contiguous cells
                    const uint32_t h = sinfo.head();
                    for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                        const size_t dst_offset = (h + j * cells.size()) * v_size_el;
                        io.read_tensor(v, dst_offset, cell_count * v_size_el);
                    }
                } else {
                    // Slow path: scatter to non-contiguous positions
                    for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                        for (uint32_t i = 0; i < cell_count; ++i) {
                            const size_t dst_offset = (sinfo.idxs[0][i] + j * cells.size()) * v_size_el;
                            io.read_tensor(v, dst_offset, v_size_el);
                        }
                    }
                }
            }
        }
    }

    return true;
}

//
// llama_kv_cache_context
//

llama_kv_cache_context::llama_kv_cache_context(llama_memory_status status) : status(status) {}

llama_kv_cache_context::llama_kv_cache_context(
        llama_kv_cache * kv) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv) {
    n_kv = kv->get_size();
    if (kv->pyramidkv_c1_hot_enabled && !kv->pyramidkv_c1_compacted) {
        n_kv = std::min(n_kv, static_cast<int32_t>(kv->pyramidkv_c1_initial_capacity));
    }
    if (kv->pyramidkv_c1_local_prefill()) {
        if (4096u % kv->n_pad != 0) {
            throw std::runtime_error("PyramidKV local prefill requires KV padding that divides 4096");
        }
        const size_t live_bound = kv->pyramidkv_c1_config.max_prefill_cells == 0 ? n_kv :
            std::min<size_t>(n_kv, kv->pyramidkv_c1_config.max_prefill_cells);
        std::string error;
        if (!llama_pyramidkv_c1_prefill_rows(kv->pyramidkv_c1_config, n_kv,
                live_bound, pyramidkv_prefill_n_kv, error)) {
            throw std::runtime_error(error);
        }
        // The reserve graph uses the request bound; runtime gathers only live rows.
        pyramidkv_prefill_cells.resize(pyramidkv_prefill_n_kv);
        for (uint32_t i = 0; i < pyramidkv_prefill_n_kv; ++i) {
            pyramidkv_prefill_cells[i] = i;
        }
    }

    const uint32_t n_stream = kv->get_n_stream();

    // create a dummy slot info - the actual data is irrelevant. we just need to build the graph
    sinfos.resize(1);
    sinfos[0].s0 = 0;
    sinfos[0].s1 = n_stream - 1;
    sinfos[0].idxs.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        sinfos[0].strm.push_back(s);
        sinfos[0].idxs[s].resize(1, 0);
    }
}

llama_kv_cache_context::llama_kv_cache_context(
        llama_kv_cache * kv,
        llama_context * lctx,
        bool do_shift,
        stream_copy_info sc_info) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv), lctx(lctx), do_shift(do_shift), sc_info(std::move(sc_info)) {
    if (!do_shift && this->sc_info.empty()) {
        status = LLAMA_MEMORY_STATUS_NO_UPDATE;
    }
}

llama_kv_cache_context::llama_kv_cache_context(
        llama_kv_cache * kv,
        llama_kv_cache::slot_info_vec_t sinfos,
        std::vector<llama_ubatch> ubatches) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv), sinfos(std::move(sinfos)), ubatches(std::move(ubatches)) {
}

llama_kv_cache_context::~llama_kv_cache_context() = default;

llama_kv_cache_context::pyramidkv_graph_inputs & llama_kv_cache_context::pyramidkv_input(int32_t il) const {
    auto it = std::find_if(pyramidkv_inputs.begin(), pyramidkv_inputs.end(),
        [il](const pyramidkv_graph_inputs & input) { return input.il == il; });
    if (it == pyramidkv_inputs.end()) {
        pyramidkv_inputs.push_back({ il, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr });
        it = pyramidkv_inputs.end();
        --it;
    }
    return *it;
}

ggml_tensor * llama_kv_cache_context::pyramidkv_read_idxs(ggml_context * ctx, int32_t il) const {
    if (!kv->pyramidkv_c1_is_compacted()) {
        return nullptr;
    }

    auto & input = pyramidkv_input(il);
    const auto map_it = kv->map_layer_ids.find(il);
    if (map_it == kv->map_layer_ids.end()) {
        throw std::runtime_error("PyramidKV C1 read index layer is unknown");
    }
    const auto & state = kv->pyramidkv_c1_layers[map_it->second];
    if (state.kv_heads == 0 || n_kv > std::numeric_limits<size_t>::max() / state.kv_heads) {
        throw std::runtime_error("PyramidKV C1 read index count overflows");
    }
    const size_t count = static_cast<size_t>(n_kv) * state.kv_heads;
    if (input.read_idxs == nullptr) {
        input.read_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, count);
        ggml_set_input(input.read_idxs);
    } else if (static_cast<size_t>(input.read_idxs->ne[0]) != count) {
        throw std::runtime_error("PyramidKV C1 read index shape changed while reusing a graph");
    }
    return input.read_idxs;
}

ggml_tensor * llama_kv_cache_context::pyramidkv_write_idxs(
        ggml_context * ctx, int32_t il, size_t n) const {
    if (!kv->pyramidkv_c1_is_compacted()) {
        return nullptr;
    }

    auto & input = pyramidkv_input(il);
    const auto map_it = kv->map_layer_ids.find(il);
    if (map_it == kv->map_layer_ids.end()) {
        throw std::runtime_error("PyramidKV C1 write index layer is unknown");
    }
    const auto & state = kv->pyramidkv_c1_layers[map_it->second];
    if (state.kv_heads == 0 || n > std::numeric_limits<size_t>::max() / state.kv_heads) {
        throw std::runtime_error("PyramidKV C1 write index count overflows");
    }
    n *= state.kv_heads;
    if (input.write_idxs == nullptr) {
        input.write_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
        ggml_set_input(input.write_idxs);
    } else if (static_cast<size_t>(input.write_idxs->ne[0]) != n) {
        throw std::runtime_error("PyramidKV C1 write index shape changed while reusing a graph");
    }
    return input.write_idxs;
}

ggml_tensor * llama_kv_cache_context::pyramidkv_hot_read_idxs(ggml_context * ctx, int32_t il) const {
    if (!kv->pyramidkv_c1_is_compacted()) {
        return nullptr;
    }
    auto & input = pyramidkv_input(il);
    const auto map_it = kv->map_layer_ids.find(il);
    if (map_it == kv->map_layer_ids.end()) {
        throw std::runtime_error("PyramidKV C1 hot read index layer is unknown");
    }
    const auto & state = kv->pyramidkv_c1_layers[map_it->second];
    if (state.kv_heads == 0 || n_kv > std::numeric_limits<size_t>::max() / state.kv_heads) {
        throw std::runtime_error("PyramidKV C1 hot read index count overflows");
    }
    const size_t count = static_cast<size_t>(n_kv) * state.kv_heads;
    if (input.hot_read_idxs == nullptr) {
        input.hot_read_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, count);
        ggml_set_input(input.hot_read_idxs);
    } else if (static_cast<size_t>(input.hot_read_idxs->ne[0]) != count) {
        throw std::runtime_error("PyramidKV C1 hot read index shape changed while reusing a graph");
    }
    return input.hot_read_idxs;
}

ggml_tensor * llama_kv_cache_context::pyramidkv_hot_write_idxs(
        ggml_context * ctx, int32_t il, size_t n) const {
    if (!kv->pyramidkv_c1_hot_enabled && !kv->pyramidkv_c1_is_compacted()) {
        return nullptr;
    }
    auto & input = pyramidkv_input(il);
    const auto map_it = kv->map_layer_ids.find(il);
    if (map_it == kv->map_layer_ids.end()) {
        throw std::runtime_error("PyramidKV C1 hot write index layer is unknown");
    }
    const auto & state = kv->pyramidkv_c1_layers[map_it->second];
    if (state.kv_heads == 0 || n > std::numeric_limits<size_t>::max() / state.kv_heads) {
        throw std::runtime_error("PyramidKV C1 hot write index count overflows");
    }
    n *= state.kv_heads;
    const auto & aux = kv->pyramidkv_c1_aux;
    if (input.hot_write_idxs == nullptr) {
        if (aux.hot_write_idxs != nullptr && n <= aux.hot_stride &&
                static_cast<size_t>(map_it->second) < static_cast<size_t>(aux.hot_write_idxs->ne[1])) {
            // Device-resident slot of this layer; filled through the staging
            // copy in set_input, never a scheduler input.
            input.hot_write_idxs = ggml_view_1d(ctx, aux.hot_write_idxs, n,
                static_cast<size_t>(map_it->second)*aux.hot_write_idxs->nb[1]);
        } else {
            input.hot_write_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
            ggml_set_input(input.hot_write_idxs);
        }
    } else if (static_cast<size_t>(input.hot_write_idxs->ne[0]) != n) {
        throw std::runtime_error("PyramidKV C1 hot write index shape changed while reusing a graph");
    }
    return input.hot_write_idxs;
}

ggml_tensor * llama_kv_cache_context::pyramidkv_k_positions(ggml_context * ctx, int32_t il) const {
    if (!kv->pyramidkv_c1_is_compacted()) {
        return nullptr;
    }
    auto & input = pyramidkv_input(il);
    const auto map_it = kv->map_layer_ids.find(il);
    if (map_it == kv->map_layer_ids.end()) {
        throw std::runtime_error("PyramidKV C1 position layer is unknown");
    }
    const auto & state = kv->pyramidkv_c1_layers[map_it->second];
    const size_t rows = static_cast<size_t>(state.cold_row_capacity) + state.hot_row_capacity;
    if (state.kv_heads == 0 || rows == 0 || rows > std::numeric_limits<size_t>::max() / state.kv_heads) {
        throw std::runtime_error("PyramidKV C1 position shape overflows");
    }
    const auto & aux = kv->pyramidkv_c1_aux;
    if (input.k_positions == nullptr) {
        if (aux.k_positions != nullptr && rows*state.kv_heads <= aux.pos_stride &&
                static_cast<size_t>(map_it->second) < static_cast<size_t>(aux.k_positions->ne[1])) {
            input.k_positions = ggml_view_2d(ctx, aux.k_positions, rows, state.kv_heads,
                rows*ggml_type_size(GGML_TYPE_I32),
                static_cast<size_t>(map_it->second)*aux.k_positions->nb[1]);
        } else {
            input.k_positions = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, rows, state.kv_heads, 1, 1);
            ggml_set_input(input.k_positions);
        }
    } else if (static_cast<size_t>(input.k_positions->ne[0]) != rows ||
            static_cast<size_t>(input.k_positions->ne[1]) != state.kv_heads) {
        throw std::runtime_error("PyramidKV C1 key position shape changed while reusing a graph");
    }
    return input.k_positions;
}

ggml_tensor * llama_kv_cache_context::pyramidkv_q_positions(ggml_context * ctx, int32_t il, size_t n) const {
    if (!kv->pyramidkv_c1_is_compacted() || n == 0) {
        return nullptr;
    }
    auto & input = pyramidkv_input(il);
    const auto & aux = kv->pyramidkv_c1_aux;
    if (input.q_positions == nullptr) {
        if (aux.q_positions != nullptr && n <= aux.n_ubatch) {
            // The ubatch positions are the same for every layer: one device
            // tensor, one upload, one view per layer.
            input.q_positions = ggml_view_1d(ctx, aux.q_positions, n, 0);
        } else {
            input.q_positions = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, n, 1, 1, 1);
            ggml_set_input(input.q_positions);
        }
    } else if (static_cast<size_t>(input.q_positions->ne[0]) != n) {
        throw std::runtime_error("PyramidKV C1 query position shape changed while reusing a graph");
    }
    return input.q_positions;
}

ggml_tensor * llama_kv_cache_context::get_pyramidkv_observer_mask(
        ggml_context * ctx, int32_t il, size_t n) const {
    if (!kv->pyramidkv_c1_is_compacted()) {
        return nullptr;
    }
    auto * positions = pyramidkv_k_positions(ctx, il);
    const size_t rows = static_cast<size_t>(positions->ne[0]);
    const size_t query_heads = kv->hparams.n_head(il);
    const auto & config = kv->pyramidkv_c1_get_config();
    if (n == 0 || query_heads == 0 || rows > config.observer_max_bytes / sizeof(float) / query_heads / n) {
        throw std::runtime_error("PyramidKV hybrid observer mask exceeds its byte budget");
    }

    auto & input = pyramidkv_input(il);
    if (input.valid_mask == nullptr) {
        input.valid_mask = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, rows, n, query_heads);
        ggml_format_name(input.valid_mask, "pyramidkv_observer_mask_l%d", il);
        ggml_set_input(input.valid_mask);
    } else if (input.valid_mask->ne[0] != static_cast<int64_t>(rows) ||
            input.valid_mask->ne[1] != static_cast<int64_t>(n) ||
            input.valid_mask->ne[2] != static_cast<int64_t>(query_heads)) {
        throw std::runtime_error("PyramidKV hybrid observer mask shape changed while reusing a graph");
    }
    return input.valid_mask;
}

void llama_kv_cache_context::set_input_pyramidkv_indices(const llama_ubatch * ubatch) const {
    if (!kv->pyramidkv_c1_hot_enabled && !kv->pyramidkv_c1_is_compacted()) {
        return;
    }
    // Qwen35 image rows (embeddings, see llama_context::extract_pyramidkv_scores)
    // map by their sequence position on axis 0 like text.
    const bool image_rows = kv->model.arch == LLM_ARCH_QWEN35 && ubatch != nullptr &&
        ubatch->token == nullptr && ubatch->embd != nullptr && ubatch->n_pos == 4;
    if (ubatch == nullptr || i_cur >= sinfos.size() || sinfos[i_cur].n_stream() != 1 ||
            ubatch->pos == nullptr ||
            (ubatch->n_pos != 1 && !(kv->model.arch == LLM_ARCH_QWEN35 && ubatch->token && ubatch->n_pos == 4) &&
                !image_rows)) {
        throw std::runtime_error("PyramidKV C1 input mapping requires one stream and one-dimensional positions");
    }
    for (uint32_t axis = 1; axis < ubatch->n_pos && !image_rows; ++axis) {
        for (uint32_t token = 0; token < ubatch->n_tokens; ++token) {
            if (ubatch->pos[axis * ubatch->n_tokens + token] != ubatch->pos[token]) {
                throw std::runtime_error("PyramidKV C1 only supports broadcast text positions");
            }
        }
    }

#if LLAMA_PYRAMIDKV_C1_PHASE_TIMING
    const int64_t phase_start_us = ggml_time_us();
    auto & phase = kv->pyramidkv_c1_phase_timing_stats;
#endif

    const auto & sinfo = sinfos[i_cur];
    const auto & cells = kv->v_cells[sinfo.strm[0]];
    const uint32_t active = cells.used_max_p1();
    // Device-resident inputs are staged on the host per layer and uploaded
    // once per kind after the loop (see pyramidkv_c1_aux_tensors).
    auto & aux = kv->pyramidkv_c1_aux;
    // Only wait for the previous uploads, not the graph submitted after them.
    aux.wait_upload();
    const auto is_aux_view = [](const ggml_tensor * t, const ggml_tensor * root) {
        return t != nullptr && root != nullptr && t->view_src == root;
    };
    bool aux_pos_dirty = false, aux_hot_dirty = false, aux_q_dirty = false;
    bool aux_list_dirty = false, aux_meta_dirty = false;
    size_t aux_pos_layers = 0, aux_hot_layers = 0, aux_list_layers = 0;
    size_t aux_list_max_len = 0;
    // lowest list entry restaged by this call (upload starts there)
    size_t aux_list_min_start = std::numeric_limits<size_t>::max();
    std::vector<int64_t> query_min(kv->n_seq_max, std::numeric_limits<int64_t>::max());
    std::vector<int64_t> query_max(kv->n_seq_max, -1);
    if (kv->pyramidkv_c1_paged()) {
        if (ubatch->n_seq_id == nullptr || ubatch->seq_id == nullptr) {
            throw std::runtime_error("PyramidKV paged queries have no sequence owners");
        }
        for (uint32_t token = 0; token < ubatch->n_tokens; ++token) {
            if (ubatch->n_seq_id[token] != 1 || ubatch->seq_id[token] == nullptr) {
                throw std::runtime_error("PyramidKV paged queries require one sequence per token");
            }
            const llama_seq_id seq = ubatch->seq_id[token][0];
            if (seq < 0 || static_cast<uint32_t>(seq) >= kv->n_seq_max) {
                throw std::runtime_error("PyramidKV paged query has an invalid sequence");
            }
            query_min[seq] = std::min(query_min[seq], static_cast<int64_t>(ubatch->pos[token]));
            query_max[seq] = std::max(query_max[seq], static_cast<int64_t>(ubatch->pos[token]));
        }
    }
    // LLAMA_PYRAMIDKV_INPUT_TIMING=1: log the host time of this staging
    static const bool input_timing = [] {
        const char * value = std::getenv("LLAMA_PYRAMIDKV_INPUT_TIMING");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    const int64_t input_timing_start = input_timing ? ggml_time_us() : 0;
    for (auto & input : pyramidkv_inputs) {
        const auto map_it = kv->map_layer_ids.find(input.il);
        if (map_it == kv->map_layer_ids.end() ||
                static_cast<size_t>(map_it->second) >= kv->pyramidkv_c1_layers.size()) {
            throw std::runtime_error("PyramidKV C1 graph input references an unknown layer");
        }
        const size_t slot = static_cast<size_t>(map_it->second);
        const auto & state = kv->pyramidkv_c1_layers[map_it->second];
        if (state.kv_heads == 0 || state.hot_heads.size() != state.kv_heads ||
                state.hot_row_capacity == 0 ||
                (state.compacted && (state.cold_heads.size() != state.kv_heads ||
                    state.cold_row_capacity == 0))) {
            throw std::runtime_error("PyramidKV C1 graph input has no per-head maps");
        }

        const uint32_t heads = state.kv_heads;
        const uint32_t cold_scratch = state.cold_row_capacity - 1;
        const uint32_t hot_scratch = state.hot_row_capacity - 1;
        // A live cell owned by exactly one sequence - any sequence. The old
        // seq-0-only test sent every hot write of a second sequence to the
        // scratch row, so paged multi-sequence decodes read stale hot rows.
        const auto valid_cell = [&](size_t logical) {
            return logical < active && logical < cells.size() && !cells.is_empty(static_cast<uint32_t>(logical)) &&
                cells.seq_count(static_cast<uint32_t>(logical)) == 1;
        };
        const auto cold_row = [&](uint32_t head, size_t logical) {
            const auto & map = state.cold_heads[head];
            if (valid_cell(logical) && logical < map.logical_to_physical.size() &&
                    map.logical_to_physical[logical] != llama_kv_cache::pyramidkv_c1_invalid_cell &&
                    map.logical_to_physical[logical] < map.row_capacity) {
                return head * state.cold_row_capacity + map.logical_to_physical[logical];
            }
            return head * state.cold_row_capacity + cold_scratch;
        };
        const auto hot_row = [&](uint32_t head, size_t logical) {
            const auto & map = state.hot_heads[head];
            if (valid_cell(logical) && logical < map.logical_to_physical.size() &&
                    map.logical_to_physical[logical] != llama_kv_cache::pyramidkv_c1_invalid_cell &&
                    map.logical_to_physical[logical] < map.row_capacity) {
                return head * state.hot_row_capacity + map.logical_to_physical[logical];
            }
            return head * state.hot_row_capacity + hot_scratch;
        };
        const auto hot_valid = [&](uint32_t head, size_t logical) {
            const auto & map = state.hot_heads[head];
            return valid_cell(logical) && logical < map.logical_to_physical.size() &&
                map.logical_to_physical[logical] != llama_kv_cache::pyramidkv_c1_invalid_cell &&
                map.logical_to_physical[logical] < map.row_capacity;
        };

        if (input.read_idxs != nullptr) {
            const size_t count = static_cast<size_t>(input.read_idxs->ne[0]);
#if LLAMA_PYRAMIDKV_C1_PHASE_TIMING
            phase.input_map_read_elems += count;
#endif
            if (count % heads != 0) {
                throw std::runtime_error("PyramidKV C1 cold read index count is not head-major");
            }
            GGML_ASSERT(ggml_backend_buffer_is_host(input.read_idxs->buffer));
            auto * data = static_cast<int32_t *>(input.read_idxs->data);
            for (size_t logical = 0; logical < count / heads; ++logical) {
                for (uint32_t head = 0; head < heads; ++head) {
                    data[logical*heads + head] = static_cast<int32_t>(cold_row(head, logical));
                }
            }
        }
        if (input.hot_read_idxs != nullptr) {
            const size_t count = static_cast<size_t>(input.hot_read_idxs->ne[0]);
#if LLAMA_PYRAMIDKV_C1_PHASE_TIMING
            phase.input_map_hot_read_elems += count;
#endif
            if (count % heads != 0) {
                throw std::runtime_error("PyramidKV C1 hot read index count is not head-major");
            }
            GGML_ASSERT(ggml_backend_buffer_is_host(input.hot_read_idxs->buffer));
            auto * data = static_cast<int32_t *>(input.hot_read_idxs->data);
            for (size_t logical = 0; logical < count / heads; ++logical) {
                for (uint32_t head = 0; head < heads; ++head) {
                    data[logical*heads + head] = static_cast<int32_t>(hot_row(head, logical));
                }
            }
        }
        if (input.write_idxs != nullptr) {
            const size_t count = static_cast<size_t>(input.write_idxs->ne[0]);
#if LLAMA_PYRAMIDKV_C1_PHASE_TIMING
            phase.input_map_write_elems += count;
#endif
            if (count != static_cast<size_t>(sinfo.size())*heads) {
                throw std::runtime_error("PyramidKV C1 cold write index count does not match the active batch");
            }
            GGML_ASSERT(ggml_backend_buffer_is_host(input.write_idxs->buffer));
            auto * data = static_cast<int32_t *>(input.write_idxs->data);
            for (size_t token = 0; token < sinfo.size(); ++token) {
                const uint32_t logical = sinfo.idxs[0][token];
                for (uint32_t head = 0; head < heads; ++head) {
                    data[token*heads + head] = static_cast<int32_t>(cold_row(head, logical));
                }
            }
        }
        if (input.hot_write_idxs != nullptr) {
            const size_t count = static_cast<size_t>(input.hot_write_idxs->ne[0]);
#if LLAMA_PYRAMIDKV_C1_PHASE_TIMING
            phase.input_map_hot_write_elems += count;
#endif
            if (count != static_cast<size_t>(sinfo.size())*heads) {
                throw std::runtime_error("PyramidKV C1 hot write index count does not match the active batch");
            }
            int32_t * data = nullptr;
            if (is_aux_view(input.hot_write_idxs, aux.hot_write_idxs)) {
                data = aux.stage_hot.data() + slot*aux.hot_stride;
                aux_hot_dirty = true;
                aux_hot_layers = std::max(aux_hot_layers, slot + 1);
            } else {
                GGML_ASSERT(ggml_backend_buffer_is_host(input.hot_write_idxs->buffer));
                data = static_cast<int32_t *>(input.hot_write_idxs->data);
            }
            for (size_t token = 0; token < sinfo.size(); ++token) {
                const uint32_t logical = sinfo.idxs[0][token];
                for (uint32_t head = 0; head < heads; ++head) {
                    const auto & map = state.hot_heads[head];
                    if (logical >= map.logical_to_physical.size() ||
                            map.logical_to_physical[logical] == llama_kv_cache::pyramidkv_c1_invalid_cell) {
                        throw std::runtime_error("PyramidKV C1 batch row has no hot physical destination");
                    }
                    data[token*heads + head] = static_cast<int32_t>(hot_row(head, logical));
                }
            }
        }
        if (input.k_positions != nullptr) {
            const size_t rows = static_cast<size_t>(state.cold_row_capacity) + state.hot_row_capacity;
#if LLAMA_PYRAMIDKV_C1_PHASE_TIMING
            phase.input_map_position_elems += rows * heads;
#endif
            int32_t * data = nullptr;
            if (is_aux_view(input.k_positions, aux.k_positions)) {
                data = aux.stage_pos.data() + slot*aux.pos_stride;
                aux_pos_dirty = true;
                aux_pos_layers = std::max(aux_pos_layers, slot + 1);
            } else {
                GGML_ASSERT(ggml_backend_buffer_is_host(input.k_positions->buffer));
                data = static_cast<int32_t *>(input.k_positions->data);
            }
            for (uint32_t head = 0; head < heads; ++head) {
                const auto & cold = state.cold_heads[head];
                const auto & hot = state.hot_heads[head];
                for (uint32_t row = 0; row < state.cold_row_capacity; ++row) {
                    const uint32_t logical = row < cold.physical_to_logical.size()
                        ? cold.physical_to_logical[row] : llama_kv_cache::pyramidkv_c1_invalid_cell;
                    // A logical token must be consumed exactly once. While
                    // it is present in the recent hot ring, hide the older
                    // retained cold copy; hot eviction exposes it again.
                    data[row + rows*head] = valid_cell(logical) && !hot_valid(head, logical)
                        ? cells.pos_get(logical) : -1;
                }
                for (uint32_t row = 0; row < state.hot_row_capacity; ++row) {
                    const uint32_t logical = row < hot.physical_to_logical.size()
                        ? hot.physical_to_logical[row] : llama_kv_cache::pyramidkv_c1_invalid_cell;
                    data[state.cold_row_capacity + row + rows*head] = valid_cell(logical) ? cells.pos_get(logical) : -1;
                }
            }
        }
        if (input.k_list != nullptr && kv->pyramidkv_c1_paged()) {
            // Each query chooses precision by position, independently of hot-slot eviction.
            const auto & lists = kv->pyramidkv_c1_paged_lists[slot];
            const uint32_t cap = kv->pyramidkv_c1_paged_list_capacity();
            int32_t * list_data = aux.stage_list.data() + static_cast<size_t>(slot)*aux.list_stride;
            int32_t * len_data  = aux.stage_len.data()  + static_cast<size_t>(slot)*aux.len_stride;
            std::fill(len_data, len_data + aux.len_stride, 0);
            for (uint32_t s = 0; s < ubatch->n_seqs_unq; ++s) {
                const llama_seq_id seq = ubatch->seq_id_unq[s];
                int32_t * staged_seqs = aux.staged_seq.data() + static_cast<size_t>(slot)*kv->n_seq_max;
                const bool same_seq = staged_seqs[s] == seq;
                // the dirty marks are consumed here, so no other slot may
                // keep counting as a copy of this sequence
                for (uint32_t t = 0; t < kv->n_seq_max; ++t) {
                    if (staged_seqs[t] == seq) {
                        staged_seqs[t] = -1;
                    }
                }
                staged_seqs[s] = seq;
                for (uint32_t head = 0; head < heads; ++head) {
                    const auto & list = lists[head][seq];
                    const auto & hot = state.hot_heads[head];
                    int32_t * out = list_data + 3ull*cap*(head + static_cast<size_t>(heads)*s);
                    const size_t n = list.size();
                    if (n > cap) {
                        throw std::runtime_error("PyramidKV paged list exceeds its reserved capacity");
                    }
                    aux_list_max_len = std::max(aux_list_max_len, n);
                    // Only the entries the device copy lacks are restaged: new
                    // ones, those after a trim or selection, and the recent
                    // window, whose hot rows follow the ring (the kernel reads
                    // a hot row only within recent_window of the query).
                    uint32_t & staged = aux.staged_valid[(static_cast<size_t>(slot)*aux.heads_max + head)*kv->n_seq_max + s];
                    uint32_t & dirty = kv->pyramidkv_c1_paged_dirty[slot][head][seq];
                    size_t start = same_seq ? std::min<size_t>({ staged, dirty, n }) : 0;
                    if (n > 0 && start > 0) {
                        const int64_t window = static_cast<int64_t>(list.back().pos) -
                            static_cast<int64_t>(kv->pyramidkv_c1_config.recent_window) -
                            static_cast<int64_t>(ubatch->n_tokens) - 1;
                        const auto recent = std::lower_bound(list.begin(), list.end(), window,
                            [](const auto & e, int64_t p) { return e.pos < p; });
                        start = std::min<size_t>(start, static_cast<size_t>(recent - list.begin()));
                    }
                    staged = static_cast<uint32_t>(n);
                    dirty = UINT32_MAX;
                    aux_list_min_start = std::min(aux_list_min_start, start);
                    for (size_t i = start; i < n; ++i) {
                        const uint32_t cell = list[i].cell;
                        if (!valid_cell(cell) || !cells.seq_has(cell, seq) || cells.pos_get(cell) != list[i].pos) {
                            throw std::runtime_error("PyramidKV paged list references a stale arena cell");
                        }
                        int32_t hot_row = -1;
                        if (cell < hot.logical_to_physical.size()) {
                            const uint32_t phys = hot.logical_to_physical[cell];
                            if (phys != llama_kv_cache::pyramidkv_c1_invalid_cell && phys < hot.row_capacity &&
                                    phys < hot.physical_to_logical.size() && hot.physical_to_logical[phys] == cell) {
                                hot_row = static_cast<int32_t>(phys);
                            }
                        }
                        // A recent key without a hot row (image cells beyond
                        // the ring's newest-cells window) reads its arena copy.
                        out[3*i + 0] = static_cast<int32_t>(cell);
                        out[3*i + 1] = list[i].pos;
                        out[3*i + 2] = hot_row;
                    }
                    len_data[head + heads*s] = static_cast<int32_t>(n);
                }
            }
            aux_list_dirty = true;
            aux_list_layers = std::max(aux_list_layers, slot + 1);
        }
        if (input.q_meta != nullptr && kv->pyramidkv_c1_paged()) {
            for (uint32_t token = 0; token < ubatch->n_tokens; ++token) {
                const llama_seq_id seq = ubatch->n_seq_id[token] > 0 ? ubatch->seq_id[token][0] : -1;
                aux.stage_meta[2*token + 0] = static_cast<int32_t>(ubatch->pos[token]);
                aux.stage_meta[2*token + 1] = seq >= 0 ? ubatch->seq_idx[seq] : -1;
            }
            aux_meta_dirty = true;
        }
        if (input.q_positions != nullptr) {
#if LLAMA_PYRAMIDKV_C1_PHASE_TIMING
            phase.input_map_position_elems += ubatch->n_tokens;
#endif
            int32_t * data = nullptr;
            if (is_aux_view(input.q_positions, aux.q_positions)) {
                data = aux.stage_q.data();
                aux_q_dirty = true;
            } else {
                GGML_ASSERT(ggml_backend_buffer_is_host(input.q_positions->buffer));
                data = static_cast<int32_t *>(input.q_positions->data);
            }
            for (uint32_t token = 0; token < ubatch->n_tokens; ++token) {
                data[token] = static_cast<int32_t>(ubatch->pos[token]);
            }
        }
        if (input.valid_mask != nullptr) {
            const size_t rows = static_cast<size_t>(input.valid_mask->ne[0]);
            const size_t queries = static_cast<size_t>(input.valid_mask->ne[1]);
            const size_t query_heads = static_cast<size_t>(input.valid_mask->ne[2]);
#if LLAMA_PYRAMIDKV_C1_PHASE_TIMING
            phase.input_map_mask_elems += rows * queries * query_heads;
#endif
            if (input.k_positions == nullptr || rows != static_cast<size_t>(input.k_positions->ne[0]) ||
                    queries != ubatch->n_tokens || query_heads % heads != 0) {
                throw std::runtime_error("PyramidKV hybrid observer mask has invalid GQA geometry");
            }
            GGML_ASSERT(ggml_backend_buffer_is_host(input.valid_mask->buffer));
            const int32_t * positions = is_aux_view(input.k_positions, aux.k_positions)
                ? aux.stage_pos.data() + slot*aux.pos_stride
                : static_cast<const int32_t *>(input.k_positions->data);
            auto * mask = static_cast<float *>(input.valid_mask->data);
            const size_t group = query_heads / heads;
            for (size_t qhead = 0; qhead < query_heads; ++qhead) {
                for (size_t query = 0; query < queries; ++query) {
                    for (size_t row = 0; row < rows; ++row) {
                        const int32_t position = positions[row + rows * (qhead / group)];
                        mask[row + rows * (query + queries * qhead)] =
                            position >= 0 && position <= ubatch->pos[query] ? 0.0f : -INFINITY;
                    }
                }
            }
        }
    }
    // M-RoPE order for the paged operator: this ubatch's query codes, and
    // the codes of the arena cells an image ubatch writes.
    bool ext_dirty = false;
    std::vector<std::pair<uint32_t, uint32_t>> ext_runs;
    if (kv->pyramidkv_c1_paged() && aux.ext != nullptr) {
        if (ubatch->n_tokens > aux.stage_ext.size()) {
            throw std::runtime_error("PyramidKV paged M-RoPE codes exceed the staged capacity");
        }
        const bool two_d = image_rows && ubatch->n_pos >= 3;
        for (uint32_t token = 0; token < ubatch->n_tokens; ++token) {
            aux.stage_ext[token] = two_d
                ? static_cast<int32_t>(ubatch->pos[token + ubatch->n_tokens]*65536 + ubatch->pos[token + 2*ubatch->n_tokens])
                : -1;
        }
        if (two_d) {
            // contiguous runs of destination cells, uploaded from stage_ext
            for (uint32_t token = 0; token < ubatch->n_tokens; ) {
                uint32_t end = token + 1;
                while (end < ubatch->n_tokens && sinfo.idxs[0][end] == sinfo.idxs[0][end - 1] + 1) {
                    ++end;
                }
                ext_runs.emplace_back(token, end);
                token = end;
            }
        }
        ext_dirty = true;
    }

    // Quest: the ubatch's (cell, position, sequence) writes, the pages to
    // reset first, the slot -> sequence map; the whole cell table after any
    // change other than an append.
    bool quest_dirty = false, quest_full = false;
    size_t quest_resets = 0;
    if (kv->pyramidkv_c1_quest() && aux.quest_writes != nullptr) {
        if (ubatch->n_tokens > aux.n_ubatch) {
            throw std::runtime_error("PyramidKV Quest ubatch exceeds the staged write capacity");
        }
        for (uint32_t token = 0; token < ubatch->n_tokens; ++token) {
            const bool one = ubatch->n_seq_id != nullptr && ubatch->n_seq_id[token] == 1 && ubatch->seq_id[token] != nullptr;
            aux.stage_quest_writes[3*token + 0] = static_cast<int32_t>(sinfo.idxs[0][token]);
            aux.stage_quest_writes[3*token + 1] = static_cast<int32_t>(ubatch->pos[token]);
            aux.stage_quest_writes[3*token + 2] = one ? ubatch->seq_id[token][0] : -1;
        }
        quest_resets = std::min<size_t>(kv->quest_pending_resets.size(), aux.stage_quest_resets.size() - 1);
        aux.stage_quest_resets[0] = static_cast<int32_t>(quest_resets);
        for (size_t i = 0; i < quest_resets; ++i) {
            aux.stage_quest_resets[1 + i] = kv->quest_pending_resets[i];
        }
        kv->quest_pending_resets.clear();
        std::fill(aux.stage_quest_seq_ids.begin(), aux.stage_quest_seq_ids.end(), -1);
        for (uint32_t s = 0; s < ubatch->n_seqs_unq && s < aux.stage_quest_seq_ids.size(); ++s) {
            aux.stage_quest_seq_ids[s] = ubatch->seq_id_unq[s];
        }
        if (kv->quest_meta_dirty) {
            const uint32_t page = static_cast<uint32_t>(kv->pyramidkv_c1_config.quest_page_size);
            aux.stage_quest_cell_meta.assign(2ull*aux.quest_cell_meta->ne[1], -1);
            aux.stage_quest_page_seqs.assign(aux.quest_page_seqs->ne[0], 0);
            for (uint32_t cell = 0; cell < cells.size(); ++cell) {
                if (cells.is_empty(cell) || cells.seq_count(cell) != 1) {
                    continue;
                }
                const llama_seq_id seq = cells.seq_get(cell);
                aux.stage_quest_cell_meta[2*cell + 0] = cells.pos_get(cell);
                aux.stage_quest_cell_meta[2*cell + 1] = seq;
                if (seq >= 0 && seq < 32) {
                    aux.stage_quest_page_seqs[cell/page] |= (int32_t) (1u << seq);
                }
            }
            kv->quest_meta_dirty = false;
            quest_full = true;
        }
        quest_dirty = true;
    }

    // One upload per kind replaces per-layer inputs. Ordered on the KV
    // backend's compute stream: inside one decode call
    // the previous ubatch's graph can still be reading these tensors while
    // this ubatch is staged (a plain tensor_set copies on another stream and
    // raced with it - nondeterministic C1 prefills). The event below keeps
    // the staging vectors unchanged until every asynchronous copy finishes.
    const auto upload = [&](ggml_tensor * tensor, const void * data, size_t bytes) {
        if (aux.backend != nullptr) {
            ggml_backend_tensor_set_async(aux.backend, tensor, data, 0, bytes);
        } else {
            ggml_backend_tensor_set(tensor, data, 0, bytes);
        }
    };
    if (aux_pos_dirty) {
        upload(aux.k_positions, aux.stage_pos.data(), aux_pos_layers*aux.k_positions->nb[1]);
    }
    if (aux_hot_dirty) {
        upload(aux.hot_write_idxs, aux.stage_hot.data(), aux_hot_layers*aux.hot_write_idxs->nb[1]);
    }
    if (aux_q_dirty) {
        upload(aux.q_positions, aux.stage_q.data(), static_cast<size_t>(ubatch->n_tokens)*sizeof(int32_t));
    }
    if (aux_list_dirty) {
        if (aux_list_max_len > 0) {
            const size_t row_elems = 3ull*kv->pyramidkv_c1_paged_list_capacity();
            GGML_ASSERT(row_elems > 0 && aux.list_stride % row_elems == 0);
            const size_t pitch = row_elems*sizeof(int32_t);
            // Columns before the lowest restaged entry are already on the
            // device (the host staging mirrors every uploaded column).
            const size_t first = std::min(aux_list_min_start, aux_list_max_len);
            const size_t width = 3ull*(aux_list_max_len - first)*sizeof(int32_t);
            const size_t offset = 3ull*first*sizeof(int32_t);
            const size_t rows = aux_list_layers*(aux.list_stride/row_elems);
            // All heads and sequence slots share this pitch, including layer padding.
            // The separately uploaded lengths exclude every uncopied tail.
            if (width > 0 && aux.backend != nullptr) {
                ggml_backend_tensor_set_2d_async(aux.backend, aux.k_list, aux.stage_list.data() + 3*first,
                    offset, width, rows, pitch, pitch);
            } else if (width > 0) {
                ggml_backend_tensor_set_2d(aux.k_list, aux.stage_list.data() + 3*first,
                    offset, width, rows, pitch, pitch);
            }
#if LLAMA_PYRAMIDKV_C1_PHASE_TIMING
            ++phase.list_upload_calls;
            phase.list_upload_bytes += rows*width;
#endif
        }
        upload(aux.k_list_len, aux.stage_len.data(), aux_list_layers*aux.k_list_len->nb[1]);
    }
    if (aux_meta_dirty) {
        upload(aux.q_meta, aux.stage_meta.data(), 2ull*ubatch->n_tokens*sizeof(int32_t));
    }
    if (ext_dirty) {
        const size_t cells = static_cast<size_t>(aux.ext->ne[0]) - aux.stage_ext.size();
        for (const auto & run : ext_runs) {
            const size_t offset = static_cast<size_t>(sinfo.idxs[0][run.first])*sizeof(int32_t);
            const size_t bytes = static_cast<size_t>(run.second - run.first)*sizeof(int32_t);
            if (aux.backend != nullptr) {
                ggml_backend_tensor_set_async(aux.backend, aux.ext, aux.stage_ext.data() + run.first, offset, bytes);
            } else {
                ggml_backend_tensor_set(aux.ext, aux.stage_ext.data() + run.first, offset, bytes);
            }
        }
        const size_t q_offset = cells*sizeof(int32_t);
        const size_t q_bytes = static_cast<size_t>(ubatch->n_tokens)*sizeof(int32_t);
        if (aux.backend != nullptr) {
            ggml_backend_tensor_set_async(aux.backend, aux.ext, aux.stage_ext.data(), q_offset, q_bytes);
        } else {
            ggml_backend_tensor_set(aux.ext, aux.stage_ext.data(), q_offset, q_bytes);
        }
    }
    if (quest_dirty) {
        upload(aux.quest_writes, aux.stage_quest_writes.data(), 3ull*ubatch->n_tokens*sizeof(int32_t));
        upload(aux.quest_resets, aux.stage_quest_resets.data(), (1 + quest_resets)*sizeof(int32_t));
        upload(aux.quest_seq_ids, aux.stage_quest_seq_ids.data(), aux.stage_quest_seq_ids.size()*sizeof(int32_t));
        if (quest_full) {
            upload(aux.quest_cell_meta, aux.stage_quest_cell_meta.data(), ggml_nbytes(aux.quest_cell_meta));
            upload(aux.quest_page_seqs, aux.stage_quest_page_seqs.data(), ggml_nbytes(aux.quest_page_seqs));
        }
    }
    if (aux.backend != nullptr &&
            (aux_pos_dirty || aux_hot_dirty || aux_q_dirty || aux_list_dirty || aux_meta_dirty || quest_dirty || ext_dirty)) {
        if (aux.upload_done) {
            ggml_backend_event_record(aux.upload_done.get(), aux.backend);
            aux.upload_pending = true;
        } else {
            // Backends without events must finish reading the reusable host staging.
            ggml_backend_synchronize(aux.backend);
        }
    }
#if LLAMA_PYRAMIDKV_C1_PHASE_TIMING
    ++phase.input_map_calls;
    phase.input_map_us += static_cast<uint64_t>(ggml_time_us() - phase_start_us);
#endif
    if (input_timing) {
        static int64_t total_us = 0, paged_us = 0;
        static uint64_t calls = 0, paged_calls = 0;
        const int64_t us = ggml_time_us() - input_timing_start;
        total_us += us;
        ++calls;
        if (aux_list_dirty) {
            paged_us += us;
            ++paged_calls;
        }
        if (calls % 256 == 0) {
            LLAMA_LOG_INFO("%s: PyramidKV input staging: %llu calls, %.2f ms avg; paged %llu calls, %.2f ms avg\n",
                __func__, (unsigned long long) calls, total_us/1000.0/calls,
                (unsigned long long) paged_calls, paged_calls ? paged_us/1000.0/paged_calls : 0.0);
        }
    }
}


ggml_tensor * llama_kv_cache_context::get_kq_mask(
        ggml_context * ctx, ggml_tensor * base_mask, int32_t il) const {
    if (!kv->pyramidkv_c1_is_compacted()) {
        return base_mask;
    }
    GGML_UNUSED(ctx);
    GGML_UNUSED(base_mask);
    if (!kv->pyramidkv_c1_hybrid_ready(il)) {
        throw std::runtime_error("PyramidKV C1 compacted attention requires the TQ4/F16 hybrid kernel");
    }
    // A single logical mask cannot represent head-local cold rows.  The
    // hybrid kernel applies causal validity from q_positions/k_positions.
    return nullptr;
}



bool llama_kv_cache_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    if (++i_cur >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_kv_cache_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    // no ubatches -> this is a KV cache update
    if (ubatches.empty()) {
        kv->update(lctx, do_shift, sc_info);

        return true;
    }

    std::string error;
    if (!kv->pyramidkv_c1_prepare_batch_rows(sinfos[i_cur], ubatches[i_cur], error)) {
        LLAMA_LOG_ERROR("%s: failed to prepare PyramidKV C1 batch rows: %s\n", __func__, error.c_str());
        kv->pyramidkv_c1_fail_transition(error);
        status = LLAMA_MEMORY_STATUS_FAILED_PREPARE;
        return false;
    }

    // Headroom and physical destinations are checked before mutating the
    // logical cell arena.  A rejected hot batch therefore cannot leave cells
    // committed without matching per-head row maps.
    kv->apply_ubatch(sinfos[i_cur], ubatches[i_cur]);

    n_kv = kv->get_n_kv(sinfos[i_cur]);
    if (!prepare_pyramidkv_prefill(error)) {
        LLAMA_LOG_ERROR("%s: failed to prepare PyramidKV local prefill: %s\n", __func__, error.c_str());
        kv->pyramidkv_c1_fail_transition(error);
        status = LLAMA_MEMORY_STATUS_FAILED_PREPARE;
        return false;
    }

    return true;
}

llama_memory_status llama_kv_cache_context::get_status() const {
    return status;
}

const llama_ubatch & llama_kv_cache_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ubatches[i_cur];
}

uint32_t llama_kv_cache_context::get_n_kv() const {
    if (pyramidkv_local_prefill_ready()) {
        return pyramidkv_prefill_n_kv;
    }
    if (kv->pyramidkv_c1_local_prefill() && !pyramidkv_paged_ready()) {
        throw std::runtime_error("PyramidKV local prefill cannot fall back to the full arena");
    }
    return n_kv;
}

bool llama_kv_cache_context::pyramidkv_local_prefill_ready() const {
    return !pyramidkv_prefill_cells.empty() && !kv->pyramidkv_c1_reserve_paged_active();
}

bool llama_kv_cache_context::prepare_pyramidkv_prefill(std::string & error) {
    pyramidkv_prefill_cells.clear();
    pyramidkv_prefill_n_kv = 0;
    if (!kv->pyramidkv_c1_local_prefill()) {
        return true;
    }
    if (i_cur >= ubatches.size()) {
        error = "PyramidKV local prefill has no current ubatch";
        return false;
    }
    const auto & ubatch = ubatches[i_cur];
    if (kv->pyramidkv_c1_paged_ubatch_ready(ubatch)) {
        return true;
    }
    if (ubatch.n_seqs_unq != 1 || ubatch.n_tokens == 0 || ubatch.seq_id == nullptr ||
            ubatch.n_seq_id == nullptr || ubatch.seq_id_unq == nullptr) {
        error = "PyramidKV local prefill requires one unselected sequence per ubatch";
        return false;
    }
    const llama_seq_id seq = ubatch.seq_id_unq[0];
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        if (ubatch.n_seq_id[i] != 1 || ubatch.seq_id[i] == nullptr || ubatch.seq_id[i][0] != seq) {
            error = "PyramidKV local prefill requires unshared sequence tokens";
            return false;
        }
    }
    const size_t live_bound = kv->pyramidkv_c1_config.max_prefill_cells == 0 ? kv->get_size() :
        std::min<size_t>(kv->get_size(), kv->pyramidkv_c1_config.max_prefill_cells);
    const auto & cells = kv->get_cells(seq);
    for (uint32_t cell = 0; cell < cells.used_max_p1(); ++cell) {
        if (cells.is_empty(cell) || !cells.seq_has(cell, seq)) {
            continue;
        }
        if (cells.seq_count(cell) != 1) {
            error = "PyramidKV local prefill requires unshared cache cells";
            return false;
        }
        if (pyramidkv_prefill_cells.size() >= live_bound) {
            error = "PyramidKV local prefill exceeds max_prefill_cells";
            return false;
        }
        pyramidkv_prefill_cells.push_back(static_cast<int32_t>(cell));
    }
    if (!llama_pyramidkv_c1_prefill_rows(kv->pyramidkv_c1_config, kv->get_size(),
            pyramidkv_prefill_cells.size(), pyramidkv_prefill_n_kv, error)) {
        return false;
    }
    if (ubatch.output != nullptr && std::any_of(ubatch.output, ubatch.output + ubatch.n_tokens,
            [](int8_t output) { return output != 0; })) {
        LLAMA_LOG_INFO("%s: PYRAMIDKV_LOCAL_PREFILL seq=%d live_cells=%zu local_rows=%u arena_rows=%d\n",
            __func__, (int) seq, pyramidkv_prefill_cells.size(), pyramidkv_prefill_n_kv, n_kv);
    }
    return true;
}

void llama_kv_cache_context::set_input_prefill_idxs(ggml_tensor * dst) const {
    if (!pyramidkv_local_prefill_ready() || dst->type != GGML_TYPE_I32 ||
            dst->ne[0] != pyramidkv_prefill_n_kv || !ggml_backend_buffer_is_host(dst->buffer)) {
        throw std::runtime_error("PyramidKV local prefill gather indices do not match the current sequence");
    }
    auto * data = static_cast<int32_t *>(dst->data);
    std::copy(pyramidkv_prefill_cells.begin(), pyramidkv_prefill_cells.end(), data);
    // Padding reads a valid row; the common causal mask excludes it.
    std::fill(data + pyramidkv_prefill_cells.size(), data + dst->ne[0], pyramidkv_prefill_cells.front());
}

static ggml_tensor * llama_pyramidkv_prefill_gather(ggml_context * ctx, ggml_tensor * cache,
        ggml_tensor * read_idxs, uint32_t heads) {
    GGML_ASSERT(llama_kv_type_is_turbo(cache->type) && cache->ne[2] == 1 && cache->ne[3] == 1);
    GGML_ASSERT(heads > 0 && cache->ne[0] % heads == 0);
    ggml_tensor * rows = ggml_view_2d(ctx, cache, cache->ne[0], cache->ne[1], cache->nb[1], 0);
    rows = ggml_get_rows_quantized(ctx, rows, read_idxs);
    return ggml_reshape_4d(ctx, rows, cache->ne[0]/heads, heads, read_idxs->ne[0], 1);
}

ggml_tensor * llama_kv_cache_context::get_k_prefill(ggml_context * ctx, int32_t il, ggml_tensor * read_idxs) const {
    const auto & layer = kv->layers[kv->map_layer_ids.at(il)];
    const size_t row_bytes = ggml_row_size(layer.k->type, layer.k->ne[0]) + ggml_row_size(layer.v->type, layer.v->ne[0]);
    if (!pyramidkv_local_prefill_ready() || read_idxs == nullptr || read_idxs->ne[0] != get_n_kv() ||
            row_bytes > (std::numeric_limits<size_t>::max() - sizeof(int32_t))/2 ||
            get_n_kv() > kv->pyramidkv_c1_config.transition_max_bytes/(2*row_bytes + sizeof(int32_t))) {
        throw std::runtime_error("PyramidKV local prefill gather exceeds its transition scratch budget");
    }
    return llama_pyramidkv_prefill_gather(ctx, layer.k, read_idxs, kv->hparams.n_head_kv(il));
}

ggml_tensor * llama_kv_cache_context::get_v_prefill(ggml_context * ctx, int32_t il, ggml_tensor * read_idxs) const {
    GGML_ASSERT(pyramidkv_local_prefill_ready() && read_idxs != nullptr && read_idxs->ne[0] == get_n_kv());
    const auto & layer = kv->layers[kv->map_layer_ids.at(il)];
    return llama_pyramidkv_prefill_gather(ctx, layer.v, read_idxs, kv->hparams.n_head_kv(il));
}

ggml_type llama_kv_cache_context::type_k() const {
    return kv->type_k();
}

ggml_type llama_kv_cache_context::type_v() const {
    return kv->type_v();
}

ggml_tensor * llama_kv_cache_context::get_turbo_rotation() const {
    return kv->get_turbo_rotation();
}

ggml_tensor * llama_kv_cache_context::get_turbo_rotation_inv() const {
    return kv->get_turbo_rotation_inv();
}

ggml_tensor * llama_kv_cache_context::get_k(ggml_context * ctx, int32_t il) const {
    return kv->get_k(ctx, il, n_kv, sinfos[i_cur], pyramidkv_read_idxs(ctx, il));
}

ggml_tensor * llama_kv_cache_context::get_v(ggml_context * ctx, int32_t il) const {
    return kv->get_v(ctx, il, n_kv, sinfos[i_cur], pyramidkv_read_idxs(ctx, il));
}

ggml_tensor * llama_kv_cache_context::get_k_hot(ggml_context * ctx, int32_t il) const {
    GGML_UNUSED(n_kv);
    return kv->get_k_hot_hybrid(ctx, il);
}

ggml_tensor * llama_kv_cache_context::get_v_hot(ggml_context * ctx, int32_t il) const {
    GGML_UNUSED(n_kv);
    return kv->get_v_hot_hybrid(ctx, il);
}

ggml_tensor * llama_kv_cache_context::get_k_hybrid(ggml_context * ctx, int32_t il) const {
    return kv->get_k_hybrid(ctx, il);
}

ggml_tensor * llama_kv_cache_context::get_v_hybrid(ggml_context * ctx, int32_t il) const {
    return kv->get_v_hybrid(ctx, il);
}

ggml_tensor * llama_kv_cache_context::get_k_positions(ggml_context * ctx, int32_t il) const {
    return pyramidkv_k_positions(ctx, il);
}

ggml_tensor * llama_kv_cache_context::get_q_positions(ggml_context * ctx, int32_t il, size_t n) const {
    return pyramidkv_q_positions(ctx, il, n);
}

bool llama_kv_cache_context::pyramidkv_hybrid_ready(int32_t il) const {
    return kv->pyramidkv_c1_hybrid_ready(il);
}

bool llama_kv_cache_context::pyramidkv_paged_ready() const {
    if (!kv->pyramidkv_c1_paged()) {
        return false;
    }
    if (kv->pyramidkv_c1_reserve_paged_active()) {
        return true;
    }
    return i_cur < ubatches.size() && kv->pyramidkv_c1_paged_ubatch_ready(ubatches[i_cur]);
}

ggml_tensor * llama_kv_cache_context::get_k_paged(ggml_context * ctx, int32_t il) const {
    return kv->get_k_paged(ctx, il);
}

ggml_tensor * llama_kv_cache_context::get_v_paged(ggml_context * ctx, int32_t il) const {
    return kv->get_v_paged(ctx, il);
}

ggml_tensor * llama_kv_cache_context::get_k_list(ggml_context * ctx, int32_t il) const {
    auto & input = pyramidkv_input(il);
    const auto & aux = kv->pyramidkv_c1_aux;
    const auto map_it = kv->map_layer_ids.find(il);
    if (aux.k_list == nullptr || map_it == kv->map_layer_ids.end()) {
        throw std::runtime_error("PyramidKV C1 paged list tensor is not allocated");
    }
    if (input.k_list == nullptr) {
        const auto & state = kv->pyramidkv_c1_layers[map_it->second];
        const size_t slot = static_cast<size_t>(map_it->second);
        const uint32_t cap = kv->pyramidkv_c1_paged_list_capacity();
        // [3, cap, heads, n_seq_max], contiguous within the layer's region
        input.k_list = ggml_view_4d(ctx, aux.k_list, 3, cap, state.kv_heads, kv->n_seq_max,
            3*sizeof(int32_t), 3ull*cap*sizeof(int32_t), 3ull*cap*state.kv_heads*sizeof(int32_t),
            slot*aux.k_list->nb[1]);
    }
    return input.k_list;
}

ggml_tensor * llama_kv_cache_context::get_k_list_len(ggml_context * ctx, int32_t il) const {
    auto & input = pyramidkv_input(il);
    const auto & aux = kv->pyramidkv_c1_aux;
    const auto map_it = kv->map_layer_ids.find(il);
    if (aux.k_list_len == nullptr || map_it == kv->map_layer_ids.end()) {
        throw std::runtime_error("PyramidKV C1 paged list length tensor is not allocated");
    }
    if (input.k_list_len == nullptr) {
        const auto & state = kv->pyramidkv_c1_layers[map_it->second];
        const size_t slot = static_cast<size_t>(map_it->second);
        input.k_list_len = ggml_view_2d(ctx, aux.k_list_len, state.kv_heads, kv->n_seq_max,
            state.kv_heads*sizeof(int32_t), slot*aux.k_list_len->nb[1]);
    }
    return input.k_list_len;
}

ggml_tensor * llama_kv_cache_context::get_paged_ext(ggml_context * ctx, size_t n) const {
    const auto & aux = kv->pyramidkv_c1_aux;
    if (aux.ext == nullptr || n > aux.stage_ext.size()) {
        throw std::runtime_error("PyramidKV paged M-RoPE code tensor is not allocated for this ubatch");
    }
    // cells, then this ubatch's n queries
    const size_t cells = static_cast<size_t>(aux.ext->ne[0]) - aux.stage_ext.size();
    return ggml_view_1d(ctx, aux.ext, static_cast<int64_t>(cells + n), 0);
}

ggml_tensor * llama_kv_cache_context::get_q_meta(ggml_context * ctx, int32_t il, size_t n) const {
    auto & input = pyramidkv_input(il);
    const auto & aux = kv->pyramidkv_c1_aux;
    if (aux.q_meta == nullptr || n > aux.n_ubatch) {
        throw std::runtime_error("PyramidKV C1 paged query metadata tensor is not allocated for this ubatch");
    }
    if (input.q_meta == nullptr) {
        input.q_meta = ggml_view_2d(ctx, aux.q_meta, 2, n, 2*sizeof(int32_t), 0);
    } else if (static_cast<size_t>(input.q_meta->ne[1]) != n) {
        throw std::runtime_error("PyramidKV C1 query metadata shape changed while reusing a graph");
    }
    return input.q_meta;
}

ggml_tensor * llama_kv_cache_context::quest_update(ggml_context * ctx, ggml_tensor * k_cur, int32_t il) const {
    const auto & aux = kv->pyramidkv_c1_aux;
    const auto map_it = kv->map_layer_ids.find(il);
    if (aux.quest_bounds == nullptr || map_it == kv->map_layer_ids.end()) {
        throw std::runtime_error("PyramidKV Quest tensors are not allocated");
    }
    const size_t slot = static_cast<size_t>(map_it->second);
    const int64_t n_tokens = k_cur->ne[2];
    ggml_tensor * bounds = ggml_view_3d(ctx, aux.quest_bounds, aux.quest_bounds->ne[0], aux.quest_bounds->ne[1],
        aux.quest_bounds->ne[2], aux.quest_bounds->nb[1], aux.quest_bounds->nb[2], slot*aux.quest_bounds->nb[3]);
    ggml_tensor * writes = ggml_view_2d(ctx, aux.quest_writes, 3, n_tokens, aux.quest_writes->nb[1], 0);
    // Resets are sized for the whole ubatch; the count in element 0 bounds them.
    ggml_tensor * resets = ggml_view_1d(ctx, aux.quest_resets, 1 + n_tokens, 0);
    return ggml_pyramidkv_quest_update(ctx, bounds, aux.quest_page_seqs, aux.quest_cell_meta, k_cur, writes, resets,
        static_cast<int32_t>(kv->pyramidkv_c1_config.quest_page_size), slot == 0);
}

ggml_tensor * llama_kv_cache_context::quest_select(ggml_context * ctx, ggml_tensor * q_cur, int32_t il) const {
    const auto & aux = kv->pyramidkv_c1_aux;
    const auto map_it = kv->map_layer_ids.find(il);
    if (aux.quest_bounds == nullptr || map_it == kv->map_layer_ids.end()) {
        throw std::runtime_error("PyramidKV Quest tensors are not allocated");
    }
    const size_t slot = static_cast<size_t>(map_it->second);
    ggml_tensor * bounds = ggml_view_3d(ctx, aux.quest_bounds, aux.quest_bounds->ne[0], aux.quest_bounds->ne[1],
        aux.quest_bounds->ne[2], aux.quest_bounds->nb[1], aux.quest_bounds->nb[2], slot*aux.quest_bounds->nb[3]);
    return ggml_pyramidkv_quest_select(ctx, q_cur, bounds, aux.quest_page_seqs, aux.quest_cell_meta,
        get_q_meta(ctx, il, static_cast<size_t>(q_cur->ne[2])), aux.quest_seq_ids,
        get_k_list(ctx, il), get_k_list_len(ctx, il),
        static_cast<int32_t>(kv->pyramidkv_c1_config.quest_pages),
        static_cast<int32_t>(kv->pyramidkv_c1_config.quest_page_size));
}

ggml_tensor * llama_kv_cache_context::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const {
    return kv->cpy_k(ctx, k_cur, k_idxs, il, sinfos[i_cur],
        pyramidkv_write_idxs(ctx, il, static_cast<size_t>(k_cur->ne[2])));
}

ggml_tensor * llama_kv_cache_context::cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const {
    return kv->cpy_v(ctx, v_cur, v_idxs, il, sinfos[i_cur],
        pyramidkv_write_idxs(ctx, il, static_cast<size_t>(v_cur->ne[2])));
}

ggml_tensor * llama_kv_cache_context::cpy_k_hot(ggml_context * ctx, ggml_tensor * k_cur, int32_t il) const {
    return kv->cpy_k_hot(ctx, k_cur,
        pyramidkv_hot_write_idxs(ctx, il, static_cast<size_t>(k_cur->ne[2])), il);
}

ggml_tensor * llama_kv_cache_context::cpy_v_hot(ggml_context * ctx, ggml_tensor * v_cur, int32_t il) const {
    return kv->cpy_v_hot(ctx, v_cur,
        pyramidkv_hot_write_idxs(ctx, il, static_cast<size_t>(v_cur->ne[2])), il);
}

ggml_tensor * llama_kv_cache_context::build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    return kv->build_input_k_idxs(ctx, ubatch);
}

ggml_tensor * llama_kv_cache_context::build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    return kv->build_input_v_idxs(ctx, ubatch);
}

ggml_tensor * llama_kv_cache_context::build_input_k_rot(ggml_context * ctx) const {
    return kv->build_input_k_rot(ctx);
}

ggml_tensor * llama_kv_cache_context::build_input_v_rot(ggml_context * ctx) const {
    return kv->build_input_v_rot(ctx);
}

void llama_kv_cache_context::set_input_k_shift(ggml_tensor * dst) const {
    kv->set_input_k_shift(dst);
}

void llama_kv_cache_context::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_k_idxs(dst, ubatch, sinfos[i_cur]);
}

void llama_kv_cache_context::set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_v_idxs(dst, ubatch, sinfos[i_cur]);
}

void llama_kv_cache_context::set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const {
    kv->set_input_kq_mask(dst, ubatch, causal_attn, pyramidkv_local_prefill_ready() ? &pyramidkv_prefill_cells : nullptr);
}

bool llama_kv_cache_context::supports_compact_mask(const llama_ubatch & ubatch) const {
    return !pyramidkv_local_prefill_ready() && kv->supports_compact_mask(ubatch);
}

void llama_kv_cache_context::set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_pos_bucket(dst, ubatch);
}

void llama_kv_cache_context::set_input_k_rot(ggml_tensor * dst) const {
    kv->set_input_k_rot(dst);
}

void llama_kv_cache_context::set_input_v_rot(ggml_tensor * dst) const {
    kv->set_input_v_rot(dst);
}

void llama_kv_cache_context::get_prev_tokens(const llama_ubatch & ubatch, uint32_t n, std::vector<llama_token> & res) const {
    kv->get_prev_tokens(ubatch, n, res);
}
