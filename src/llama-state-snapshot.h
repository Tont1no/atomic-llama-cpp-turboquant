#pragma once

#include "llama-io.h"
#include "llama.h"
#include "ggml-cpp.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>

// Copies metadata immediately. Tensor ranges become independent backend buffers
// at freeze(); only the resulting immutable copies may be read on another thread.
struct llama_state_snapshot_error : std::runtime_error {
    llama_state_seq_snapshot_status status;
    llama_state_snapshot_error(llama_state_seq_snapshot_status value, const char * message)
        : std::runtime_error(message), status(value) {}
};

struct llama_state_seq_snapshot : llama_io_write_i {
    struct part {
        std::unique_ptr<uint8_t[]> metadata;
        ggml_tensor * source = nullptr;
        ggml_tensor * copy = nullptr;
        ggml_backend_buffer_type_t copy_buft = nullptr;
        size_t offset = 0;
        size_t size = 0;
    };
    struct group {
        size_t count = 0;
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buffer;
    };
    std::vector<part> parts;
    std::map<ggml_backend_buffer_type_t, group> groups;
    size_t limit;
    size_t charged = sizeof(llama_state_seq_snapshot);
    size_t bytes = 0;
    bool frozen = false;

    explicit llama_state_seq_snapshot(size_t max_bytes) : limit(max_bytes) {
        if (charged > limit) throw llama_state_snapshot_error(LLAMA_STATE_SEQ_SNAPSHOT_BUDGET, "snapshot budget exceeded");
    }
    void charge(size_t size) {
        if (size > limit - charged) throw llama_state_snapshot_error(LLAMA_STATE_SEQ_SNAPSHOT_BUDGET, "snapshot budget exceeded");
        charged += size;
    }
    void append(part value) {
        if (frozen || value.size > std::numeric_limits<size_t>::max() - bytes)
            throw llama_state_snapshot_error(LLAMA_STATE_SEQ_SNAPSHOT_INVALID, "invalid snapshot write");
        if (parts.size() == parts.capacity()) {
            const size_t capacity = parts.capacity() ? parts.capacity() * 2 : 32;
            if (capacity < parts.capacity() || capacity > limit / sizeof(part))
                throw llama_state_snapshot_error(LLAMA_STATE_SEQ_SNAPSHOT_BUDGET, "snapshot budget exceeded");
            charge(capacity * sizeof(part)); // Includes the reallocation peak.
            const size_t old = parts.capacity();
            parts.reserve(capacity);
            charged -= old * sizeof(part);
        }
        bytes += value.size;
        parts.push_back(std::move(value));
    }
    void write(const void * src, size_t size) override {
        charge(size);
        part value;
        value.metadata.reset(new uint8_t[size]);
        value.size = size;
        if (size) memcpy(value.metadata.get(), src, size);
        append(std::move(value));
    }
    static ggml_backend_buffer_type_t copy_buffer_type(ggml_backend_buffer_type_t source) {
        if (!source) return nullptr;
        if (source == ggml_backend_cpu_buffer_type()) return source;
        const auto device = ggml_backend_buft_get_device(source);
        const auto reg = device ? ggml_backend_dev_backend_reg(device) : nullptr;
        if (!reg || strcmp(ggml_backend_reg_name(reg), "CUDA") != 0) return nullptr;
        const auto standard = ggml_backend_dev_buffer_type(device);
        if (source == standard) return standard;
        using kv_buffer_type_fn = ggml_backend_buffer_type_t (*)(int);
        const auto kv_buffer_type = reinterpret_cast<kv_buffer_type_fn>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_kv_buffer_type"));
        if (!kv_buffer_type) return nullptr;
        for (size_t i = 0; i < ggml_backend_reg_dev_count(reg) && i <= size_t(std::numeric_limits<int>::max()); ++i) {
            if (ggml_backend_reg_dev_get(reg, i) == device) {
                // Only the explicit KV pool is supported. Copies use the standard same-device buffer.
                return source == kv_buffer_type(int(i)) ? standard : nullptr;
            }
        }
        return nullptr;
    }
    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (!size) return;
        if (!tensor || !tensor->buffer || offset > ggml_nbytes(tensor) || size > ggml_nbytes(tensor) - offset ||
            size % ggml_type_size(tensor->type) || offset % ggml_type_size(tensor->type))
            throw llama_state_snapshot_error(LLAMA_STATE_SEQ_SNAPSHOT_INVALID, "invalid snapshot tensor range");
        const auto copy_buft = copy_buffer_type(ggml_backend_buffer_get_type(tensor->buffer));
        if (!copy_buft)
            throw llama_state_snapshot_error(LLAMA_STATE_SEQ_SNAPSHOT_UNSUPPORTED, "snapshot backend is not supported");
        auto found = groups.find(copy_buft);
        if (found == groups.end()) {
            charge(sizeof(group) + 256);
            found = groups.emplace(copy_buft, group{}).first;
        }
        ++found->second.count;
        part value;
        value.source = tensor;
        value.copy_buft = copy_buft;
        value.offset = offset;
        value.size = size;
        append(std::move(value));
    }
    size_t n_bytes() override { return bytes; }

    void freeze() {
        if (frozen) throw llama_state_snapshot_error(LLAMA_STATE_SEQ_SNAPSHOT_INVALID, "snapshot already frozen");
        for (auto & item : groups) {
            auto & g = item.second;
            if (g.count > limit / (2 * ggml_tensor_overhead()))
                throw llama_state_snapshot_error(LLAMA_STATE_SEQ_SNAPSHOT_BUDGET, "snapshot budget exceeded");
            const size_t size = 2 * g.count * ggml_tensor_overhead();
            charge(size);
            g.ctx.reset(ggml_init({size, nullptr, true}));
            if (!g.ctx) throw std::bad_alloc();
        }
        for (auto & p : parts) {
            if (!p.source) continue;
            auto & g = groups.at(p.copy_buft);
            const int64_t n = (p.size / ggml_type_size(p.source->type)) * ggml_blck_size(p.source->type);
            p.copy = ggml_new_tensor_1d(g.ctx.get(), p.source->type, n);
        }
        // Check every allocation before allocating any independent tensor buffer.
        for (auto & item : groups) {
            charge(ggml_backend_alloc_ctx_tensors_from_buft_size(item.second.ctx.get(), item.first));
        }
        for (auto & item : groups) {
            auto & g = item.second;
            g.buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(g.ctx.get(), item.first));
            if (!g.buffer) throw std::bad_alloc();
        }
        for (auto & p : parts) {
            if (!p.source) continue;
            auto & g = groups.at(p.copy_buft);
            auto * view = ggml_view_1d(g.ctx.get(), p.source, p.copy->ne[0], p.offset);
            ggml_backend_view_init(view);
            ggml_backend_tensor_copy(view, p.copy);
            p.source = nullptr;
        }
        frozen = true;
    }
    size_t materialize(uint8_t * dst, size_t size) const {
        if (!frozen || !dst || size < bytes) return 0;
        for (const auto & p : parts) {
            if (p.copy) ggml_backend_tensor_get(p.copy, dst, 0, p.size);
            else if (p.size) memcpy(dst, p.metadata.get(), p.size);
            dst += p.size;
        }
        return bytes;
    }
};
