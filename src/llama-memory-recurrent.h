#pragma once

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-memory.h"

#include <map>
#include <set>
#include <vector>

//
// llama_memory_recurrent
//

inline llama_memory_recurrent_resize_stats llama_recurrent_resize_metrics_snapshot(
        bool dynamic,
        uint32_t configured_depth,
        uint32_t resident_depth,
        uint32_t last_required_depth,
        uint32_t pending_depth,
        uint32_t stable_ticks,
        uint64_t resize_count,
        uint64_t resize_time_us) {
    llama_memory_recurrent_resize_stats result;
    result.count            = resize_count;
    result.time_us          = resize_time_us;
    result.resident_depth   = resident_depth;
    result.configured_depth = configured_depth;
    result.pending_depth    = pending_depth;
    result.stable_ticks     = stable_ticks;
    result.required_depth   = dynamic ? last_required_depth : configured_depth;
    return result;
}

// TODO: extract the cache state used for graph computation into llama_memory_recurrent_context_i
//       see the implementation of llama_kv_cache_context_i for an example how to do it
class llama_memory_recurrent : public llama_memory_i {
public:
    llama_memory_recurrent(
            const llama_model & model,
                    ggml_type   type_r,
                    ggml_type   type_s,
                         bool   offload,
                     uint32_t   mem_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_rs_seq,
                         bool   rs_seq_dynamic,
        const layer_filter_cb & filter);

    ~llama_memory_recurrent() = default;

    bool prepare_batch(
            llama_context * lctx,
            const llama_batch & batch,
            bool embd_all) override;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    llama_memory_recurrent_resize_stats recurrent_resize_stats() const override;

    bool prepare(const std::vector<llama_ubatch> & ubatches);

    // find a contiguous slot of memory cells and emplace the ubatch there
    bool find_slot(const llama_ubatch & ubatch);

    bool get_can_shift() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    uint32_t head = 0; // the location where the batch will be placed in the cache (see find_slot())
    uint32_t size = 0; // total number of cells, shared across all sequences
    uint32_t used = 0; // used cells (i.e. at least one seq_id)

    // number of recurrent-state snapshots per seq for rollback; tensors are widened to (1 + n_rs_seq) groups
    uint32_t n_rs_seq = 0;

    // Physical rollback planes currently resident. This can be smaller than
    // n_rs_seq for the dynamic Qwen/DFlash path; plane zero is always present.
    uint32_t n_rs_seq_alloc = 0;

    // When enabled, each decode executes only the snapshot depth required by
    // the rows present for a sequence in that batch. Supported compact Qwen
    // contexts also resize the resident rollback planes between decode ticks;
    // the logical rollback limit remains the configured n_rs_seq maximum.
    const bool rs_seq_dynamic = false;

    // Resolve a safe execution depth for the sanitized batch. Invalid or
    // ambiguous metadata fails closed to the configured maximum.
    uint32_t active_rs_depth(const llama_batch & batch, bool embd_all, bool * used_fallback = nullptr);

    // per-seq rollback index
    std::vector<uint32_t> rs_idx;

    // Per-sequence snapshot depth known to have been written by the most
    // recent dynamic graph. Prevents selecting stale max-allocation planes.
    std::vector<uint32_t> rs_valid_depth;

    void set_rs_idx(llama_seq_id seq_id, uint32_t idx);
    void commit_rs_depth(const llama_ubatch & ubatch, uint32_t active_depth);
    void invalidate_rs_depth(const llama_ubatch & ubatch);

    // computed before each graph build
    uint32_t n = 0;

    // first zero-ed state
    int32_t rs_z = -1;

    // TODO: optimize for recurrent state needs
    struct mem_cell {
        llama_pos pos  = -1;
        int32_t   src  = -1; // used to know where states should be copied from
        int32_t   src0 = -1; // like src, but only used when setting the inputs (allowing to copy once)
        int32_t   tail = -1;

        std::set<llama_seq_id> seq_id;

        bool has_seq_id(const llama_seq_id & id) const {
            return seq_id.find(id) != seq_id.end();
        }

        bool is_empty() const {
            return seq_id.empty();
        }

        bool is_same_seq(const mem_cell & other) const {
            return seq_id == other.seq_id;
        }
    };

    std::vector<mem_cell> cells;

    // per layer
    std::vector<ggml_tensor *> r_l;
    std::vector<ggml_tensor *> s_l;

private:
    //const llama_model & model;
    const llama_hparams & hparams;

    const uint32_t n_seq_max = 1;

    // Qwen hybrid recurrent states are large enough that retaining every
    // configured rollback plane penalizes cap-zero multi-slot throughput.
    // This bounded path changes only the resident plane count between ticks.
    const bool rs_seq_compact = false;

    ggml_type type_r;
    ggml_type type_s;
    std::vector<ggml_backend_buffer_type_t> buft_l;

    uint32_t last_reported_active_n_rs_seq = UINT32_MAX;

    // Exact-target shrink hysteresis. Growth is always immediate when the
    // incoming graph needs more resident planes; shrink happens only after
    // two consecutive logical batches request the same lower safe depth.
    static constexpr uint32_t rs_shrink_dwell_ticks = 2;
    uint32_t rs_shrink_pending_depth = UINT32_MAX;
    uint32_t rs_shrink_pending_load  = UINT32_MAX;
    uint32_t rs_shrink_stable_ticks  = 0;
    uint32_t rs_last_required_depth  = 0;
    uint64_t rs_shrink_last_epoch    = UINT64_MAX;
    uint64_t rs_standalone_epoch     = 0;

    uint64_t rs_resize_count   = 0;
    uint64_t rs_resize_time_us = 0;

    // ggml contexts for the KV cache along with the allocated backend buffers:
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> ctxs_bufs;

    bool resize_rs_storage(uint32_t depth);
    bool materialize_pending_rollbacks(uint32_t retained_depth);

    size_t total_size() const;

    size_t size_r_bytes() const;
    size_t size_s_bytes() const;

    void state_write_meta(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, llama_seq_id seq_id = -1) const;
    void state_write_data(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges) const;

    bool state_read_meta(llama_io_read_i & io, uint32_t cell_count, llama_seq_id dest_seq_id = -1);
    bool state_read_data(llama_io_read_i & io, uint32_t cell_count);
};

class llama_memory_recurrent_context : public llama_memory_context_i {
public:
    // used for errors
    llama_memory_recurrent_context(llama_memory_status status);

    // used to create a full-cache or update context
    llama_memory_recurrent_context(
            llama_memory_recurrent * mem);

    // used to create a batch processing context from a batch
    llama_memory_recurrent_context(
            llama_memory_recurrent * mem,
            std::vector<llama_ubatch> ubatches,
            uint32_t active_n_rs_seq);

    virtual ~llama_memory_recurrent_context();

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;
    void finalize(bool success) override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    //
    // llama_memory_recurrent_context specific API
    //

    uint32_t get_n_rs() const;
    uint32_t get_active_n_rs_seq() const;
    uint32_t get_head() const;
    int32_t  get_rs_z() const;
    uint32_t get_size() const;

    ggml_tensor * get_r_l(int32_t il) const;
    ggml_tensor * get_s_l(int32_t il) const;

    int32_t s_copy(int i) const;

private:
    const llama_memory_status status;

    llama_memory_recurrent * mem;

    size_t i_next = 0;

    std::vector<llama_ubatch> ubatches;

    // Maximum active depth requested for this logical batch. Individual
    // ragged ubatches may safely use a smaller depth based on their row count.
    const uint32_t active_n_rs_seq = 0;

    //
    // data needed for building the compute graph for the current ubatch:
    // TODO: extract all the state like `head` and `n` here
    //

    const bool is_full = false;
};

// Pure batch-metadata helper used by the dynamic rollback implementation and
// its unit tests. Returns configured_max and sets used_fallback on malformed
// metadata; otherwise returns min(configured_max, max_rows_per_seq - 1).
uint32_t llama_recurrent_batch_active_rs_depth(
        const llama_batch & batch,
        uint32_t configured_max,
        uint32_t n_seq_max,
        bool * used_fallback = nullptr);

// Pure fail-closed rollback guard shared by the cache implementation and
// metadata tests. Dynamic contexts may address a snapshot only when both the
// physical plane and the per-sequence last-written depth still cover it.
bool llama_recurrent_rollback_is_valid(
        bool dynamic,
        uint32_t rollback,
        uint32_t configured_max,
        uint32_t resident_depth,
        uint32_t valid_depth);
