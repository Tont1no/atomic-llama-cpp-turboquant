#pragma once

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-kv-cells.h"
#include "llama-memory.h"
#include "llama-pyramidkv-c1.h"

#include <limits>
#include <unordered_map>
#include <vector>

struct llama_cparams;
struct llama_hparams;
struct llama_model;
struct llama_context;
class llama_kv_cache_context;

//
// llama_kv_cache
//

class llama_kv_cache : public llama_memory_i {
public:
    struct stream_copy_info {
        bool empty() const {
            assert(ssrc.size() == sdst.size());
            return ssrc.empty();
        }

        std::vector<uint32_t> ssrc;
        std::vector<uint32_t> sdst;
    };

    // for each ubatch, create a slot_info that contains information about where the ubatch should be inserted in the
    //   KV cells. for example, cell indices for each token, such that: token[i] -> goes to cells[idxs[i]]
    struct slot_info {
        // data for ggml_set_rows
        using idx_vec_t = std::vector<uint32_t>;

        // number of streams: ns = s1 - s0 + 1
        uint32_t s0;
        uint32_t s1;

        std::vector<llama_seq_id> strm; // [ns]
        std::vector<idx_vec_t>    idxs; // [ns]

        uint32_t head() const {
            GGML_ASSERT(idxs.size() == 1);
            GGML_ASSERT(!idxs[0].empty());

            return idxs[0][0];
        }

        void resize(size_t n) {
            strm.resize(n);
            idxs.resize(n);
        }

        size_t size() const {
            GGML_ASSERT(idxs.size() == strm.size());
            GGML_ASSERT(!idxs.empty());

            return idxs[0].size();
        }

        size_t n_stream() const {
            return strm.size();
        }

        bool empty() const {
            return idxs.empty();
        }

        void clear() {
            idxs.clear();
        }

        // check if indices are contiguous starting from head()
        bool is_contiguous() const {
            if (idxs.empty() || idxs[0].empty()) {
                return true;
            }
            if (idxs.size() > 1) {
                return false;
            }
            const uint32_t h = idxs[0][0];
            for (size_t i = 0; i < idxs[0].size(); ++i) {
                if (idxs[0][i] != h + i) {
                    return false;
                }
            }
            return true;
        }
    };

    using slot_info_vec_t = std::vector<slot_info>;

    // TODO: refactor the memory instances to not depend on `llama_model`
    //       instead pass all necessary info (e.g. hparams, dev layers, arch, etc.) directly
    //       likely through `struct llama_memory_params`
    llama_kv_cache(
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
        ggml_backend_buffer_type_t kv_buffer_type = nullptr,
        llama_pyramidkv_c1_config pyramidkv_c1 = {},
        bool tq4_key_center = false);

    ~llama_kv_cache();

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;
    int64_t   seq_n_cells(llama_seq_id seq_id) const override;
    int32_t   seq_positions(llama_seq_id seq_id, llama_pos * pos, int32_t cap) const override;
    bool      seq_keep_positions(llama_seq_id seq_id, const llama_pos * pos, int32_t n) override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    //
    // llama_kv_cache specific API
    //

    uint32_t get_size()     const;
    uint32_t get_n_stream() const;

    bool get_has_shift() const;

    // Optional dense Qwen2 K-anchor seam.  The anchor is a separate F32
    // allocation and is populated by the graph owner after the first
    // successful device synchronization.  It is intentionally fail-closed:
    // state I/O, sequence remapping and shared caches are not part of this
    // first format/version contract.
    bool tq4_key_center_enabled() const;
    bool tq4_key_center_capture() const;
    bool tq4_key_center_ready() const;
    ggml_tensor * get_tq4_key_anchor(int32_t il) const;
    bool tq4_key_center_prepare(const llama_ubatch & ubatch, std::string & error);
    void tq4_key_center_commit_capture();
    void tq4_key_center_fail();

    ggml_type type_k() const;
    ggml_type type_v() const;

    // Signed 128-value rotation used by the native TurboQuant-4 graph path.
    // The matrices live in the same backend buffer as the cache tensors.
    ggml_tensor * get_turbo_rotation() const;
    ggml_tensor * get_turbo_rotation_inv() const;

    std::vector<uint32_t> get_layer_ids() const;
    ggml_tensor * get_k_storage(int32_t il) const;

    const llama_kv_cells & get_cells(llama_seq_id seq_id) const;

    // C1 applies to this attention cache, including the attention part of a hybrid model.
    bool pyramidkv_c1_supported(std::string & error) const;
    const llama_pyramidkv_c1_config & pyramidkv_c1_get_config() const { return pyramidkv_c1_config; }
    // seq_id < 0 lists every cell of sequence 0 (single-sequence C1); a
    // paged cache lists the cells of that sequence only.
    bool pyramidkv_c1_key_positions(
            int32_t il,
            uint32_t n_kv,
            std::vector<std::vector<std::int64_t>> & positions,
            std::vector<std::vector<std::size_t>> & logical_cells,
            std::vector<std::vector<std::size_t>> & score_slots,
            uint32_t & active_tokens,
            std::string & error,
            llama_seq_id seq_id = -1,
            bool sequence_local = false) const;
    bool pyramidkv_c1_prepare_batch_rows(
            const slot_info & sinfo,
            const llama_ubatch & ubatch,
            std::string & error);
    bool pyramidkv_c1_compact(
            llama_context * lctx,
            const std::vector<llama_pyramidkv_c1_layer_selection> & selections,
            std::string & error);
    bool pyramidkv_c1_is_compacted() const { return pyramidkv_c1_compacted; }
    // Call after each computed continuation ubatch; maintenance never reselects tokens.
    bool pyramidkv_c1_should_maintain(uint32_t n_tokens, uint32_t n_ubatch);
    bool pyramidkv_c1_keep_all(
            std::vector<llama_pyramidkv_c1_layer_selection> & selections,
            std::string & error) const;
    bool pyramidkv_c1_hybrid_ready(int32_t il) const;

    // Paged C1 (several sequences share the cache). The TQ4 arena keeps every
    // token of every sequence, the selection is per layer/head bookkeeping
    // (row lists), and decode of a selected sequence runs the paged hybrid
    // operator over its lists. No layout replacement ever happens.
    bool pyramidkv_c1_paged() const { return pyramidkv_c1_hot_enabled && pyramidkv_c1_config.paged; }
    // Quest: per-step page choice on top of the paged lists (keeps every cell).
    bool pyramidkv_c1_quest() const { return pyramidkv_c1_paged() && pyramidkv_c1_config.quest_pages > 0; }
    bool pyramidkv_c1_local_prefill() const;
    bool pyramidkv_c1_paged_seq_compacted(llama_seq_id seq_id) const;
    // Every sequence of the ubatch is selected: the ubatch runs the paged path.
    bool pyramidkv_c1_paged_ubatch_ready(const llama_ubatch & ubatch) const;
    std::vector<llama_ubatch> pyramidkv_c1_split_batch(
            llama_batch_allocr & balloc, uint32_t n_ubatch, uint32_t n_keep_tail, bool single_seq) const;
    bool pyramidkv_c1_paged_apply_selection(
            llama_seq_id seq_id,
            const std::vector<llama_pyramidkv_c1_layer_selection> & selections,
            std::string & error);
    bool pyramidkv_c1_set_protected(llama_seq_id seq_id, std::vector<std::pair<int32_t, int32_t>> ranges);
    bool pyramidkv_c1_is_protected(llama_seq_id seq_id, llama_pos pos) const;
    // Graph reserve builds the paged decode graph once with this set.
    void pyramidkv_c1_set_reserve_paged(bool value) { pyramidkv_c1_reserve_paged = value; }
    bool pyramidkv_c1_reserve_paged_active() const { return pyramidkv_c1_reserve_paged; }
    uint32_t pyramidkv_c1_paged_list_capacity() const { return static_cast<uint32_t>(pyramidkv_c1_config.list_capacity); }
    size_t pyramidkv_c1_paged_list_size(llama_seq_id seq_id) const;

    // A layout replacement keeps old graph tensors alive until the scheduler
    // has been reset. Callers use this state to fail closed after an
    // allocation failure instead of evaluating with stale compacted rows.
    bool pyramidkv_c1_graph_reset_pending() const { return pyramidkv_c1_graph_reset_needed; }
    void pyramidkv_c1_graph_reset_complete();
    // A partial seq_rm (draft rollback) keeps the layout; only a selection
    // the context recorded before the removal is stale. The context drops it
    // and clears the flag; no graph reserve and no compaction is forced.
    bool pyramidkv_c1_take_selection_stale() {
        const bool stale = pyramidkv_c1_selection_stale;
        pyramidkv_c1_selection_stale = false;
        return stale;
    }
    // Paged: a partial seq_rm only touches that sequence's cells, so only its
    // pending selection is stale; another sequence's prompt-end selection in
    // the same batch stays valid (a draft rollback per step would otherwise
    // starve every later prompt of its selection). Returns and clears the set.
    std::vector<llama_seq_id> pyramidkv_c1_take_stale_seqs() {
        std::vector<llama_seq_id> out;
        for (size_t s = 0; s < pyramidkv_c1_selection_stale_seqs.size(); ++s) {
            if (pyramidkv_c1_selection_stale_seqs[s]) {
                out.push_back(static_cast<llama_seq_id>(s));
                pyramidkv_c1_selection_stale_seqs[s] = 0;
            }
        }
        return out;
    }
    bool pyramidkv_c1_reset_failed(std::string & error) const;
    void pyramidkv_c1_fail_transition(const std::string & error);
    const llama_pyramidkv_c1_phase_timing & pyramidkv_c1_get_phase_timing() const {
        return pyramidkv_c1_phase_timing_stats;
    }
    void pyramidkv_c1_reset_phase_timing() {
        pyramidkv_c1_phase_timing_stats = {};
    }

    //
    // graph_build API
    //

    uint32_t get_n_kv(const slot_info & sinfo) const;

    // get views of the current state of the cache
    ggml_tensor * get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo,
                        ggml_tensor * read_idxs = nullptr) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo,
                        ggml_tensor * read_idxs = nullptr) const;
    ggml_tensor * get_k_hot(ggml_context * ctx, int32_t il, uint32_t n_kv,
                            ggml_tensor * read_idxs = nullptr) const;
    ggml_tensor * get_v_hot(ggml_context * ctx, int32_t il, uint32_t n_kv,
                            ggml_tensor * read_idxs = nullptr) const;
    ggml_tensor * get_k_hybrid(ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_v_hybrid(ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_k_hot_hybrid(ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_v_hot_hybrid(ggml_context * ctx, int32_t il) const;
    // Paged views: the arena as [D, rows, kv_heads] (row stride = whole
    // row, head stride = one head's TQ4 blocks) and the hot ring head-major.
    ggml_tensor * get_k_paged(ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_v_paged(ggml_context * ctx, int32_t il) const;
    // hot ring head-major [D, hot_capacity, kv_heads, 1], no readiness gate
    ggml_tensor * get_k_hot_ring(ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_v_hot_ring(ggml_context * ctx, int32_t il) const;

    // store k_cur and v_cur in the cache based on the provided head location
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il, const slot_info & sinfo,
                        ggml_tensor * physical_idxs = nullptr) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il, const slot_info & sinfo,
                        ggml_tensor * physical_idxs = nullptr) const;
    ggml_tensor * cpy_k_hot(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const;
    ggml_tensor * cpy_v_hot(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const;

    //
    // preparation API
    //

    // find places for the provided ubatches in the cache, returns the slot infos
    // return empty vector on failure
    slot_info_vec_t prepare(const std::vector<llama_ubatch> & ubatches);

    bool update(llama_context * lctx, bool do_shift, const stream_copy_info & sc_info);

    // find a slot of kv cells that can hold the ubatch
    // if cont == true, then the slot must be continuous
    // return empty slot_info on failure
    slot_info find_slot(const llama_ubatch & ubatch, bool cont) const;

    // emplace the ubatch context into slot: [sinfo.idxs[0...ubatch.n_tokens - 1]]
    void apply_ubatch(const slot_info & sinfo, const llama_ubatch & ubatch);

    //
    // input API
    //

    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;

    ggml_tensor * build_input_k_rot(ggml_context * ctx) const;
    ggml_tensor * build_input_v_rot(ggml_context * ctx) const;

    void set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const;
    void set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const;

    void set_input_k_shift(ggml_tensor * dst) const;

    bool supports_compact_mask(const llama_ubatch & ubatch) const;

    void set_input_kq_mask   (ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn,
                             const std::vector<int32_t> * local_cells = nullptr) const;
    void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const;

    void set_input_k_rot(ggml_tensor * dst) const;
    void set_input_v_rot(ggml_tensor * dst) const;

    // true if llama_kv_cell_ext holds information that has to survive a state save/restore
    bool has_cell_ext() const;

    // for every token of the ubatch, the ids of the n tokens that precede it in its sequence
    // entries with no matching cell are set to LLAMA_TOKEN_NULL
    // note: used by n-gram input embeddings
    void get_prev_tokens(const llama_ubatch & ubatch, uint32_t n, std::vector<llama_token> & res) const;

private:
    friend class llama_kv_cache_context;

    const llama_model & model;
    const llama_hparams & hparams;

    struct kv_layer {
        // layer index in the model
        // note: can be different from the layer index in the KV cache
        uint32_t il;

        ggml_tensor * k;
        ggml_tensor * v;

        std::vector<ggml_tensor *> k_stream;
        std::vector<ggml_tensor *> v_stream;

        // C1 hybrid lane: F16 hot rows are separate from the TQ4 cold rows.
        ggml_tensor * k_hot = nullptr;
        ggml_tensor * v_hot = nullptr;
        std::vector<ggml_tensor *> k_hot_stream;
        std::vector<ggml_tensor *> v_hot_stream;
    };

    struct pyramidkv_c1_head_state {
        uint32_t row_capacity = 0;
        uint32_t next_row = 0;
        std::vector<uint32_t> logical_to_physical;
        std::vector<uint32_t> physical_to_logical;
    };

    struct pyramidkv_c1_layer_state {
        bool compacted = false;
        uint32_t kv_heads = 0;
        uint32_t cold_row_capacity = 0;
        uint32_t hot_row_capacity = 0;
        uint32_t hot_recent = 0;
        std::vector<pyramidkv_c1_head_state> cold_heads;
        std::vector<pyramidkv_c1_head_state> hot_heads;
    };

    bool pyramidkv_c1_reinitialize(std::string & error);
    void tq4_key_center_invalidate() noexcept;
    bool tq4_key_center_cache_empty() const;

    bool v_trans = true;  // the value tensor is transposed

    const uint32_t n_seq_max = 1;
    const uint32_t n_stream  = 1;

    // required padding
    const uint32_t n_pad = 1;

    // SWA
    const uint32_t n_swa = 0;

    // env: LLAMA_ATTN_ROT_DISABLE
    bool attn_rot_k = false;
    bool attn_rot_v = false;

    // if all layers participating in the cache have constant head size, the value is stored here
    // otherwise the value is -1
    int32_t n_embd_head_k_all = 0;
    int32_t n_embd_head_v_all = 0;

    // pre-computed hadamard martrices
    std::unordered_map<int64_t, std::vector<float>> attn_rot_hadamard;

    // TurboQuant-4 signed forward/inverse rotations (128 x 128).
    ggml_tensor * turbo_rotation     = nullptr;
    ggml_tensor * turbo_rotation_inv = nullptr;

    // env: LLAMA_KV_CACHE_DEBUG
    int debug = 0;

    // this is the SWA type of the cache - not to be confused with the model SWA type
    const llama_swa_type swa_type = LLAMA_SWA_TYPE_NONE;

    // ggml contexts for the KV cache along with the allocated backend buffers:
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> ctxs_bufs;

    // The K anchors are deliberately outside ctxs_bufs: C1 replaces that
    // vector during compaction, while this allocation must keep its identity.
    ggml_context_ptr tq4_key_anchor_ctx;
    ggml_backend_buffer_ptr tq4_key_anchor_buf;
    std::vector<ggml_tensor *> tq4_key_anchors;

    // Retired C1 buffers stay alive until llama_context resets scheduler and
    // graph results. This prevents old graph tensor handles becoming dangling.
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> pyramidkv_c1_retired_ctxs_bufs;

    // A failed reinitialization is sticky until a later successful full clear.
    // init_batch and context scheduling use it as a fail-closed gate.
    bool pyramidkv_c1_graph_reset_needed = false;
    bool pyramidkv_c1_selection_stale = false;
    std::vector<uint8_t> pyramidkv_c1_selection_stale_seqs; // paged: per sequence
    bool pyramidkv_c1_reset_failed_flag = false;
    std::string pyramidkv_c1_reset_error;

    // the current index from where we start searching for a free slot in the ring buffer of KV cells (see find_slot())
    // note: this is not part of the KV state and it's only used to speed-up the find_slot() method
    std::vector<uint32_t> v_heads;

    // TODO: temporary until we refactor to be able to share the same cells between 2 kv caches [TAG_KV_CACHE_SHARE_CELLS]
    llama_kv_cache * other;

    std::shared_ptr<llama_kv_cells_vec> v_cells_impl;

    llama_kv_cells_vec & v_cells;

    // maps from a sequence id to a stream id
    std::vector<uint32_t> seq_to_stream;

    // pending stream copies that will be applied during the next update
    stream_copy_info sc_info;

    // Retaining this handle is required for C1 reallocation.  The normal
    // constructor path still uses per-layer placement when it is null.
    ggml_backend_buffer_type_t kv_buffer_type = nullptr;

    // Immutable profile copied from llama_context_params.
    llama_pyramidkv_c1_config pyramidkv_c1_config;
    mutable llama_pyramidkv_c1_phase_timing pyramidkv_c1_phase_timing_stats;

    bool tq4_key_center_enabled_flag = false;
    bool tq4_key_center_valid_flag = false;
    bool tq4_key_center_capture_pending = false;
    bool tq4_key_center_failed_flag = false;

    // The existing state format has no physical-row remap/version field.
    // Once rows are compacted, state I/O is therefore explicitly rejected.
    bool pyramidkv_c1_compacted = false;
    bool pyramidkv_c1_hot_enabled = false;
    // Keep the full configured cold context until the prompt is complete.
    uint32_t pyramidkv_c1_initial_capacity = 0;
    uint32_t pyramidkv_c1_tokens_since_compact = 0;

    static constexpr uint32_t pyramidkv_c1_invalid_cell = std::numeric_limits<uint32_t>::max();

    std::vector<pyramidkv_c1_layer_state> pyramidkv_c1_layers;

    // Paged C1 bookkeeping: per sequence the selection state, per
    // layer/head/sequence the ordered (cell, position) list the paged
    // operator walks. Decode tokens of a selected sequence are appended in
    // prepare_batch_rows; seq_rm trims by position.
    struct pyramidkv_c1_list_entry {
        uint32_t cell;
        int32_t pos;
    };
    std::vector<uint8_t> pyramidkv_c1_paged_compacted;                          // [n_seq_max]
    // Position ranges a paged selection keeps whole (llama_pyramidkv_c1_protect_positions)
    std::vector<std::vector<std::pair<int32_t, int32_t>>> pyramidkv_c1_protected; // [n_seq_max]
    std::vector<std::vector<std::vector<std::vector<pyramidkv_c1_list_entry>>>>
        pyramidkv_c1_paged_lists;                                               // [layer][head][seq]
    bool pyramidkv_c1_reserve_paged = false;
    void pyramidkv_c1_paged_reset();
    void pyramidkv_c1_paged_trim(llama_seq_id seq_id, llama_pos p0, llama_pos p1);

    // C1 index and position tensors the graph reads per layer, resident on
    // the KV device. As per-layer graph inputs they lived in host memory and
    // ggml_backend_sched synchronised the device once per input before
    // copying it (about 150 stream syncs per decode token on a 36-layer
    // model). They are rebuilt together with the pools; the graph views them
    // per layer and set_input uploads each kind once per ubatch.
    struct pyramidkv_c1_aux_tensors {
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buf;
        ggml_tensor * k_positions    = nullptr; // I32 [pos_stride, n_layers]
        ggml_tensor * hot_write_idxs = nullptr; // I32 [hot_stride, n_layers]
        ggml_tensor * q_positions    = nullptr; // I32 [n_ubatch]
        // paged lists: [3*list_capacity*heads_max*n_seq_max, n_layers],
        // lengths [heads_max*n_seq_max, n_layers], q_meta [2*n_ubatch]
        ggml_tensor * k_list         = nullptr;
        ggml_tensor * k_list_len     = nullptr;
        ggml_tensor * q_meta         = nullptr;
        // paged M-RoPE order: I32 [n_cells + n_ubatch], (y, x) code of every
        // arena cell, then one per query of the ubatch (-1 for text)
        ggml_tensor * ext            = nullptr;
        std::vector<int32_t> stage_ext;
        // Quest (see ggml_pyramidkv_quest_update/_select): page bounds
        // F16 [2*D, heads, n_pages, n_layers], page_seqs I32 [n_pages],
        // cell_meta I32 [2, n_cells], per-ubatch writes I32 [3, n_ubatch],
        // resets I32 [1 + n_ubatch], slot sequence ids I32 [n_seq_max].
        ggml_context_ptr quest_ctx;
        ggml_backend_buffer_ptr quest_buf;
        ggml_tensor * quest_bounds    = nullptr;
        ggml_tensor * quest_page_seqs = nullptr;
        ggml_tensor * quest_cell_meta = nullptr;
        ggml_tensor * quest_writes    = nullptr;
        ggml_tensor * quest_resets    = nullptr;
        ggml_tensor * quest_seq_ids   = nullptr;
        std::vector<int32_t> stage_quest_writes;
        std::vector<int32_t> stage_quest_resets;
        std::vector<int32_t> stage_quest_seq_ids;
        std::vector<int32_t> stage_quest_cell_meta;
        std::vector<int32_t> stage_quest_page_seqs;
        uint32_t pos_stride = 0;
        uint32_t hot_stride = 0;
        uint32_t n_ubatch   = 0;
        uint32_t list_stride = 0; // I32 elements (triples) per layer
        uint32_t len_stride  = 0;
        uint32_t heads_max   = 0;
        std::vector<int32_t> stage_pos;
        std::vector<int32_t> stage_hot;
        std::vector<int32_t> stage_q;
        std::vector<int32_t> stage_list;
        std::vector<int32_t> stage_len;
        std::vector<int32_t> stage_meta;
        // Backend that computes on the KV device. The per-ubatch uploads go
        // through its stream (tensor_set_async) so they are ordered after the
        // previous ubatch's graph, which may still be reading these tensors:
        // the buffer's synchronous tensor_set copies on a different stream.
        ggml_backend_t backend = nullptr;
        ggml_backend_event_ptr upload_done;
        bool upload_pending = false;

        void wait_upload() {
            if (upload_pending) {
                ggml_backend_event_synchronize(upload_done.get());
                upload_pending = false;
            }
        }
    };
    pyramidkv_c1_aux_tensors pyramidkv_c1_aux;
    bool pyramidkv_c1_aux_rebuild(std::string & error);
    // Quest: pages first written since they were empty (reset before the
    // bounds widen), and whether cells changed other than by appends (the
    // device cell table is then rebuilt from the host cells).
    std::vector<int32_t> quest_pending_resets;
    bool quest_meta_dirty = true;
public:
    void pyramidkv_c1_bind_aux_backend(ggml_backend_t backend);
private:

    std::vector<kv_layer> layers;

    // model layer id -> KV cache layer id
    std::unordered_map<int32_t, int32_t> map_layer_ids;

    size_t total_size() const;

    size_t size_k_bytes() const;
    size_t size_v_bytes() const;

    ggml_tensor * build_rope_shift(
            const llama_cparams & cparams,
                   ggml_context * ctx,
                    ggml_tensor * cur,
                    ggml_tensor * shift,
                    ggml_tensor * rot,
                    ggml_tensor * factors,
                          float   freq_base,
                          float   freq_scale,
                       uint32_t   il) const;

    ggml_cgraph * build_graph_shift(
               llm_graph_result * res,
                  llama_context * lctx) const;

    struct cell_ranges_t {
        uint32_t strm;

        std::vector<std::pair<uint32_t, uint32_t>> data; // ranges, from inclusive, to exclusive
    };

    void state_write_meta(llama_io_write_i & io, const cell_ranges_t & cr, llama_seq_id seq_id = -1) const;
    void state_write_data(llama_io_write_i & io, const cell_ranges_t & cr) const;

    bool state_read_meta(llama_io_read_i & io, uint32_t strm, uint32_t cell_count,       slot_info & sinfo, llama_seq_id dest_seq_id = -1);
    bool state_read_data(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, const slot_info & sinfo);
};

class llama_kv_cache_context : public llama_memory_context_i {
public:
    // some shorthands
    using slot_info_vec_t  = llama_kv_cache::slot_info_vec_t;
    using stream_copy_info = llama_kv_cache::stream_copy_info;

    // used for errors
    llama_kv_cache_context(llama_memory_status status);

    // used to create a full-cache context
    llama_kv_cache_context(
            llama_kv_cache * kv);

    // used to create an update context
    llama_kv_cache_context(
            llama_kv_cache * kv,
            llama_context * lctx,
            bool do_shift,
            stream_copy_info sc_info);

    // used to create a batch processing context from a batch
    llama_kv_cache_context(
            llama_kv_cache * kv,
            slot_info_vec_t sinfos,
            std::vector<llama_ubatch> ubatches);

    virtual ~llama_kv_cache_context();

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    //
    // llama_kv_cache_context specific API
    //

    uint32_t get_n_kv() const;

    ggml_type type_k() const;
    ggml_type type_v() const;

    ggml_tensor * get_turbo_rotation() const;
    ggml_tensor * get_turbo_rotation_inv() const;
    ggml_tensor * get_tq4_key_anchor(int32_t il) const { return kv == nullptr ? nullptr : kv->get_tq4_key_anchor(il); }
    bool tq4_key_center_enabled() const { return kv != nullptr && kv->tq4_key_center_enabled(); }
    bool tq4_key_center_capture() const { return kv != nullptr && kv->tq4_key_center_capture(); }
    bool tq4_key_center_ready() const { return kv != nullptr && kv->tq4_key_center_ready(); }

    // get views of the current state of the cache
    ggml_tensor * get_k(ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il) const;
    bool pyramidkv_local_prefill_ready() const;
    ggml_tensor * get_k_prefill(ggml_context * ctx, int32_t il, ggml_tensor * read_idxs) const;
    ggml_tensor * get_v_prefill(ggml_context * ctx, int32_t il, ggml_tensor * read_idxs) const;
    void set_input_prefill_idxs(ggml_tensor * dst) const;
    ggml_tensor * get_k_hot(ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_v_hot(ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_k_hybrid(ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_v_hybrid(ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_k_positions(ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_q_positions(ggml_context * ctx, int32_t il, size_t n) const;
    ggml_tensor * get_pyramidkv_observer_mask(ggml_context * ctx, int32_t il, size_t n) const;
    bool pyramidkv_hybrid_ready(int32_t il) const;
    // Paged C1: the current ubatch runs the paged operator.
    bool pyramidkv_paged_ready() const;
    bool pyramidkv_seq_compacted(llama_seq_id seq_id) const { return kv->pyramidkv_c1_paged_seq_compacted(seq_id); }
    ggml_tensor * get_k_paged(ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_v_paged(ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_k_hot_ring(ggml_context * ctx, int32_t il) const { return kv->get_k_hot_ring(ctx, il); }
    ggml_tensor * get_v_hot_ring(ggml_context * ctx, int32_t il) const { return kv->get_v_hot_ring(ctx, il); }
    ggml_tensor * get_k_list(ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_k_list_len(ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_q_meta(ggml_context * ctx, int32_t il, size_t n) const;
    ggml_tensor * get_paged_ext(ggml_context * ctx, size_t n) const;
    bool pyramidkv_c1_compacted() const { return kv->pyramidkv_c1_is_compacted(); }
    // Quest: widen the page bounds with this ubatch's unrotated keys, and the
    // per-step page list for the paged operator (unrotated queries).
    bool pyramidkv_quest() const { return kv->pyramidkv_c1_quest(); }
    ggml_tensor * quest_update(ggml_context * ctx, ggml_tensor * k_cur, int32_t il) const;
    ggml_tensor * quest_select(ggml_context * ctx, ggml_tensor * q_cur, int32_t il) const;

    ggml_tensor * get_kq_mask(ggml_context * ctx, ggml_tensor * base_mask, int32_t il) const;

    // store k_cur and v_cur in the cache based on the provided head location
    // note: the heads in k_cur and v_cur should be laid out contiguously in memory
    //   - k_cur  [n_embd_head_k, n_head_k, n_tokens]
    //   - k_idxs [n_tokens]
    //   - v_cur  [n_embd_head_v, n_head_v, n_tokens]
    //   - v_idxs [n_tokens] or [n_tokens*n_embd_v_gqa] depending if V cache is transposed
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const;
    ggml_tensor * cpy_k_hot(ggml_context * ctx, ggml_tensor * k_cur, int32_t il) const;
    ggml_tensor * cpy_v_hot(ggml_context * ctx, ggml_tensor * v_cur, int32_t il) const;

    // create destination indices for each head of the current batch for where it would be written in the KV cache
    // the indices address the global KV cache (not per stream) - this is not relevant for the user of this API, but
    //   helps understand the implementation logic of cpy_k and cpy_v
    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;

    ggml_tensor * build_input_k_rot(ggml_context * ctx) const;
    ggml_tensor * build_input_v_rot(ggml_context * ctx) const;

    void set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;
    void set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;

    void set_input_k_shift   (ggml_tensor * dst) const;
    bool supports_compact_mask(const llama_ubatch & ubatch) const;
    void set_input_kq_mask   (ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;
    void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const;

    void set_input_k_rot(ggml_tensor * dst) const;
    void set_input_v_rot(ggml_tensor * dst) const;

    // see llama_kv_cache::get_prev_tokens()
    void get_prev_tokens(const llama_ubatch & ubatch, uint32_t n, std::vector<llama_token> & res) const;

private:
    llama_memory_status status;

    llama_kv_cache * kv;
    llama_context * lctx;

    //
    // update context
    //

    bool do_shift = false;

    stream_copy_info sc_info;

    //
    // batch processing context
    //

    // the index of the cur ubatch to process
    size_t i_cur = 0;

    slot_info_vec_t sinfos;

    std::vector<llama_ubatch> ubatches;

    //
    // data needed for building the compute graph for the current ubatch:
    //

    // a heuristic, to avoid attending the full cache if it is not yet utilized
    // as the cache gets filled, the benefit from this heuristic disappears
    int32_t n_kv;

    // Arena order is shared by the gather, mask and observer score mapping.
    std::vector<int32_t> pyramidkv_prefill_cells;
    uint32_t pyramidkv_prefill_n_kv = 0;
    bool prepare_pyramidkv_prefill(std::string & error);

    using pyramidkv_graph_inputs = llm_graph_pyramidkv_inputs;

    mutable std::vector<pyramidkv_graph_inputs> pyramidkv_inputs;

    pyramidkv_graph_inputs & pyramidkv_input(int32_t il) const;

    ggml_tensor * pyramidkv_read_idxs(ggml_context * ctx, int32_t il) const;
    ggml_tensor * pyramidkv_write_idxs(ggml_context * ctx, int32_t il, std::size_t n) const;
    ggml_tensor * pyramidkv_hot_read_idxs(ggml_context * ctx, int32_t il) const;
    ggml_tensor * pyramidkv_hot_write_idxs(ggml_context * ctx, int32_t il, std::size_t n) const;
    ggml_tensor * pyramidkv_k_positions(ggml_context * ctx, int32_t il) const;
    ggml_tensor * pyramidkv_q_positions(ggml_context * ctx, int32_t il, size_t n) const;
    void set_input_pyramidkv_indices(const llama_ubatch * ubatch) const;

    friend class llm_graph_input_attn_kv;
    // The hybrid memory input routes its attention half through the same
    // PyramidKV input list when a reused graph changes memory context.
    friend class llm_graph_input_mem_hybrid;
};
