#pragma once

// Experimental, opt-in C1 PyramidKV runtime contract.
//
// A score and a physical cache map are per KV head.  There is no union of
// retained positions between heads or layers.  The hybrid attention seam
// consumes the resulting head-local position lists.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct llama_pyramidkv_c1_params;

// Local, compile-time opt-in.  The default build keeps all timers and the
// summary log disabled; no runtime or process-wide setting is added.
#ifndef LLAMA_PYRAMIDKV_C1_PHASE_TIMING
#define LLAMA_PYRAMIDKV_C1_PHASE_TIMING 0
#endif

struct llama_pyramidkv_c1_phase_timing {
    // Host-side input mapping.
    uint64_t input_map_calls = 0;
    uint64_t input_map_us = 0;
    uint64_t input_map_read_elems = 0;
    uint64_t input_map_hot_read_elems = 0;
    uint64_t input_map_write_elems = 0;
    uint64_t input_map_hot_write_elems = 0;
    uint64_t input_map_position_elems = 0;
    uint64_t input_map_mask_elems = 0;
    uint64_t list_upload_calls = 0;
    uint64_t list_upload_bytes = 0;

    // Observer host wall-time while waiting for and reading scores.
    uint64_t observer_events = 0;
    uint64_t observer_sync_us = 0;
    uint64_t observer_readback_us = 0;
    uint64_t observer_readback_calls = 0;
    uint64_t observer_readback_bytes = 0;

    // Host wall-time for llama_context synchronization calls.  The observer
    // synchronize value is a subset of this aggregate.
    uint64_t synchronize_us = 0;
    uint64_t synchronize_calls = 0;

    // Host-side graph stages in llama_context::process_ubatch().
    uint64_t graph_build_us = 0;
    uint64_t graph_build_calls = 0;
    uint64_t graph_alloc_us = 0;
    uint64_t graph_alloc_calls = 0;
    uint64_t graph_set_inputs_us = 0;
    uint64_t graph_set_inputs_calls = 0;
    uint64_t graph_compute_submit_us = 0;
    uint64_t graph_compute_submit_calls = 0;
    uint64_t process_ubatch_observer_calls = 0;
    uint64_t process_ubatch_nonobserver_calls = 0;
    uint64_t graph_build_observer_calls = 0;
    uint64_t graph_build_nonobserver_calls = 0;
    uint64_t graph_reuse_observer_calls = 0;
    uint64_t graph_reuse_nonobserver_calls = 0;

    // Compaction transition host wall-time and transfer/work counters.
    uint64_t compaction_calls = 0;
    uint64_t cold_copy_us = 0;
    uint64_t cold_copy_calls = 0;
    uint64_t cold_copy_bytes = 0;
    uint64_t hot_copy_us = 0;
    uint64_t hot_copy_calls = 0;
    uint64_t hot_copy_bytes = 0;
    uint64_t promotion_us = 0;
    uint64_t promotion_calls = 0;
    uint64_t promotion_rows = 0;
    uint64_t promotion_syncs = 0;

    // Reserve work surrounding C1 layout changes.
    uint64_t post_compaction_reserve_us = 0;
    uint64_t post_compaction_reserve_calls = 0;
    uint64_t sched_reserve_us = 0;
    uint64_t sched_reserve_calls = 0;
};

struct llama_pyramidkv_c1_config {
    bool enabled = false;
    std::size_t layer_count = 0;
    std::size_t layer_index = 0;
    std::size_t max_capacity_prompt = 4096;
    std::size_t beta = 20;
    std::size_t observation_window = 64;
    std::size_t recent_window = 128;
    std::size_t pooling_kernel = 3;
    std::size_t observer_max_bytes = 64u*1024u*1024u;
    std::size_t observer_chunk = 1;
    std::size_t transition_max_bytes = 8ull*1024ull*1024ull*1024ull;
    std::size_t continuation_headroom = 1;
    std::size_t hot_capacity = 384;
    std::size_t rollback_headroom = 0;
    bool paged = false;
    std::size_t list_capacity = 0;
    std::size_t paged_union_factor = 4;
    std::size_t max_prefill_cells = 0;
    std::size_t quest_pages = 0;
    std::size_t quest_page_size = 64;
};

struct llama_pyramidkv_c1_score {
    int32_t il = -1;
    std::size_t query_heads = 0;
    std::size_t kv_heads = 0;
    std::size_t query_tokens = 0;
    // key_tokens is the largest head-local key list.  logical_key_tokens is
    // the global cell range; head-local cell lists may be shorter.
    std::size_t key_tokens = 0;
    std::size_t logical_key_tokens = 0;
    std::size_t key_stride = 0;
    std::size_t observation_window = 0;
    // F32 scores reduced over observed queries and Q-heads. Initial scores
    // use logical cells; hybrid scores use explicit physical score slots.
    std::vector<float> head_scores;
    // Each head has its own valid score slots and original positions.  A
    // missing physical row is absent from the list, never silently unioned.
    std::vector<std::vector<std::int64_t>> key_positions_per_head;
    std::vector<std::vector<std::size_t>> key_cells_per_head;
    std::vector<std::vector<std::size_t>> key_score_slots_per_head;
    std::vector<std::int64_t> query_positions;
};

struct llama_pyramidkv_c1_head_selection {
    std::vector<std::size_t> keep_cells;
    std::vector<std::int64_t> keep_positions;
    std::vector<std::size_t> protected_recent_cells;
    std::vector<std::int64_t> protected_recent_positions;
};

struct llama_pyramidkv_c1_layer_selection {
    int32_t il = -1;
    std::size_t source_tokens = 0;
    std::size_t kv_heads = 0;
    bool would_compact = false;
    std::vector<llama_pyramidkv_c1_head_selection> heads;
    std::size_t continuation_headroom = 1;
};

bool llama_pyramidkv_c1_make_config(
        const llama_pyramidkv_c1_params & params,
        std::size_t layer_count,
        std::size_t continuation_headroom,
        std::size_t n_seq_max,
        llama_pyramidkv_c1_config & output,
        std::string & error);

bool llama_pyramidkv_c1_prefill_rows(
        const llama_pyramidkv_c1_config & config,
        uint32_t arena_cells,
        std::size_t live_cells,
        uint32_t & rows,
        std::string & error);

bool llama_pyramidkv_c1_select(
        const llama_pyramidkv_c1_score & score,
        const llama_pyramidkv_c1_config & config,
        llama_pyramidkv_c1_layer_selection & output,
        std::string & error);
