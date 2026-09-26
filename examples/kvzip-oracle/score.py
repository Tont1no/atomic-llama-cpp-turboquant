"""Convert private native KVzip+ capture and effective Bonsai W_O to oracle NPZ.

This intentionally refuses an unverified W_O export. Raw attention alone is not
KVzip+: value/output norms and GQA max are required for the final labels.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


LAYERS = tuple(range(3, 64, 4))
ORACLE_SOURCE_COMMIT = "7331c23da9e6f1510d89ea651d0dea77a57b3252"


def private(path: Path, root: Path) -> Path:
    sys.path.insert(0, str(root.parent.parent / "Inference engine" / "scripts" / "bonsai_training"))
    from data import require_private

    return require_private(path, root)


@lru_cache(maxsize=None)
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=None)
def effective_weight(path: Path, device: str) -> torch.Tensor:
    weights = np.load(path, mmap_mode="r", allow_pickle=False)
    if weights.shape != (5120, 6144) or weights.dtype != np.float16:
        raise ValueError("Effective W_O dtype/shape mismatch")
    return torch.as_tensor(np.array(weights, dtype=np.float32), device=device)


def read_raw(path: Path, dtype: str, shape: tuple[int, ...]) -> np.ndarray:
    expected = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if path.stat().st_size != expected:
        raise ValueError(f"Oracle raw shape mismatch: {path.name}")
    return np.memmap(path, dtype=dtype, mode="r", shape=shape)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--export-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()
    root = (args.repo.resolve() / "tmp" / "bonsai-training-20260924").resolve()
    capture = private(args.capture, root)
    export_receipt = private(args.export_receipt, root)
    output = private(args.output, root)
    if output.exists():
        raise ValueError("Oracle output already exists")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    receipt_path = capture / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    export = json.loads(export_receipt.read_text(encoding="utf-8"))
    if receipt.get("kind") not in ("kvzap_full_chat_native_attention_capture_v2", "kvzap_full_chat_native_attention_capture_v3"):
        raise ValueError("Unexpected native capture")
    if receipt["kind"].endswith("_v3") and (not isinstance(receipt.get("ids_sha256"), str) or len(receipt["ids_sha256"]) != 64):
        raise ValueError("Batch capture lacks decoded token-file hash")
    if any(len(receipt.get(key, "")) != 64 for key in ("native_binary_sha256", "replay_template_sha256")):
        raise ValueError("Capture lacks binary/template provenance")
    if receipt.get("target_gguf_sha256") != export.get("model_sha256"):
        raise ValueError("Target GGUF hash differs from W_O export")
    if export.get("basis") != "runtime-effective":
        raise ValueError("W_O export is not in the runtime-effective basis")
    if (receipt.get("n_head"), receipt.get("n_kv_head"), receipt.get("head_dim"), receipt.get("hidden")) != (24, 4, 256, 5120):
        raise ValueError("Unexpected Bonsai oracle geometry")
    if receipt.get("flash_attention") is not False or receipt.get("kv_cache_type") != "f16":
        raise ValueError("Capture did not use the reference attention path")
    if (receipt.get("attention_mask") != "native_full_causal; query=assistant_copy_only; key=user_source_only" or
        receipt.get("query_origin_kind") != "copied_text_no_token_alignment" or
        receipt.get("sink") != 4 or receipt.get("recent_eligibility_mask") != 256 or
        receipt.get("replay_microbatch") != 64):
        raise ValueError("Unexpected replay/mask contract")
    n = receipt["source_length"]
    source_start = receipt["source_start_position"]
    source_end = receipt["source_end_position"]
    repeat_start = receipt["repeat_start_position"]
    repeat_end = receipt["repeat_end_position"]
    repeat = receipt["repeat_length"]
    if (not 32 <= n <= 1250 or not 0 <= source_start < source_end <= repeat_start < repeat_end <= receipt["input_length"] <= 4096 or
        source_end - source_start != n or repeat_end - repeat_start != repeat):
        raise ValueError("Invalid source/replay mapping")
    if receipt.get("wo_probe_positions") != [source_start, source_start + 1, source_start + 2, source_start + 3,
                                               source_end - 4, source_end - 3, source_end - 2, source_end - 1]:
        raise ValueError("Unexpected W_O probe positions")
    positions = read_raw(capture / "source_positions.i32", "<i4", (n,))
    query_positions = read_raw(capture / "query_positions.i32", "<i4", (repeat,))
    query_origin = read_raw(capture / "query_origin.i32", "<i4", (repeat,))
    eligible = read_raw(capture / "eligible.u8", "u1", (n,))
    if (not np.array_equal(positions, np.arange(source_start, source_end)) or
        not np.array_equal(query_origin, np.full(repeat, -1)) or
        not np.array_equal(query_positions, np.arange(repeat_start, repeat_end)) or
        not np.isin(eligible, [0, 1]).all() or not eligible.any()):
        raise ValueError("Source/query positions or eligibility are inconsistent")

    # Check all inputs before writing any final oracle shard.
    weight_paths: dict[int, Path] = {}
    for layer in LAYERS:
        tensor = export["tensors"].get(f"blk.{layer}.attn_output.weight")
        if tensor is None or tensor.get("shape") != [5120, 6144]:
            raise ValueError("Missing effective W_O tensor")
        path = private(Path(tensor["path"]), root)
        if sha256(path) != tensor["sha256"]:
            raise ValueError("Effective W_O hash mismatch")
        weight_paths[layer] = path
        for suffix, shape in (("features.f32", (n, 5120)), ("values.f32", (n, 4, 256)), ("attention.f32", (n, 24))):
            read_raw(capture / f"layer-{layer}.{suffix}", "<f4", shape)
        read_raw(capture / f"layer-{layer}.probe_gated.f32", "<f4", (8, 6144))
        read_raw(capture / f"layer-{layer}.probe_output.f32", "<f4", (8, 5120))

    probe_metrics = {}
    with torch.inference_mode():
        for layer in LAYERS:
            gated = read_raw(capture / f"layer-{layer}.probe_gated.f32", "<f4", (8, 6144))
            native = read_raw(capture / f"layer-{layer}.probe_output.f32", "<f4", (8, 5120))
            w = effective_weight(weight_paths[layer], args.device)
            x = torch.as_tensor(np.array(gated), device=args.device)
            y = torch.as_tensor(np.array(native), device=args.device)
            predicted = x @ w.T
            delta = predicted - y
            row_error = torch.linalg.vector_norm(delta, dim=1) / torch.linalg.vector_norm(y, dim=1).clamp_min(1e-12)
            row_cosine = torch.nn.functional.cosine_similarity(predicted, y, dim=1)
            metrics = {"relative_l2": float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(y).clamp_min(1e-12)),
                       "max_row_relative_l2": float(row_error.max()),
                       "min_row_cosine": float(row_cosine.min())}
            if (not all(np.isfinite(v) for v in metrics.values()) or
                metrics["relative_l2"] > 0.01 or metrics["max_row_relative_l2"] > 0.02 or
                metrics["min_row_cosine"] < 0.9999):
                raise ValueError(f"Native W_O probe failed for layer {layer}: {metrics}")
            probe_metrics[str(layer)] = metrics

    output.mkdir(parents=False)
    with torch.inference_mode():
        for layer in LAYERS:
            features = read_raw(capture / f"layer-{layer}.features.f32", "<f4", (n, 5120))
            values = read_raw(capture / f"layer-{layer}.values.f32", "<f4", (n, 4, 256))
            attention = read_raw(capture / f"layer-{layer}.attention.f32", "<f4", (n, 24))
            if not np.isfinite(features).all() or not np.isfinite(values).all() or not np.isfinite(attention).all() or np.any(attention < 0):
                raise ValueError("Nonfinite or negative native capture")
            weights = effective_weight(weight_paths[layer], args.device)
            score = np.zeros((n, 4), dtype=np.float32)
            # The native cache stores F16 values. Round pre-cache Vcur to that basis.
            value_t = torch.as_tensor(np.asarray(values, dtype=np.float16).astype(np.float32), device=args.device)
            for qhead in range(24):
                kvhead = qhead // 6
                w = weights[:, qhead * 256:(qhead + 1) * 256]
                norm = torch.linalg.vector_norm(value_t[:, kvhead] @ w.T, dim=1).cpu().numpy()
                score[:, kvhead] = np.maximum(score[:, kvhead], np.asarray(attention[:, qhead]) * norm)
            if not np.isfinite(score).all() or np.any(score < 0):
                raise ValueError("Invalid KVzip+ oracle scores")
            valid = np.broadcast_to(np.asarray(eligible, dtype=np.bool_)[:, None], (n, 4)).copy()
            np.savez_compressed(output / f"layer-{layer}.npz", layer_id=np.int32(layer),
                                attn_norm=np.asarray(features, dtype=np.float16),
                                oracle_scores=score, eligible=valid,
                                positions=np.asarray(positions, dtype=np.int32),
                                sequence_offsets=np.asarray([0, n], dtype=np.int64))
    meta = {
        "oracle_kind": "kvzip_plus_replay_max_gqa",
        "feature_kind": "qwen35_post_attn_norm",
        "normalization": "max_over_replay_and_gqa(attention*norm(W_OV)/norm(h))",
        "replay_passes": 1,
        "replay_microbatch": receipt["replay_microbatch"],
        "replay_microbatch_count": receipt["replay_microbatch_count"],
        "replay_template_sha256": receipt["replay_template_sha256"],
        "attention_layer_ids": list(LAYERS),
        "n_head_kv": 4,
        "model_name": "Hf",
        "target_gguf_sha256": receipt["target_gguf_sha256"],
        "oracle_source_commit": ORACLE_SOURCE_COMMIT,
        "provenance": {"capture_receipt_sha256": sha256(receipt_path),
                       "export_receipt_sha256": sha256(export_receipt),
                       "native_binary_sha256": receipt["native_binary_sha256"],
                       "wo_native_equivalence": "PASS",
                       "wo_probe": probe_metrics},
        "source_length": n,
        "source_start_position": source_start,
        "source_end_position": source_end,
        "repeat_start_position": repeat_start,
        "repeat_end_position": repeat_end,
        "repeat_length": repeat,
        "sink": receipt["sink"],
        "recent_eligibility_mask": receipt["recent_eligibility_mask"],
        "protected_count": receipt["protected_count"],
        "attention_mask": receipt["attention_mask"],
        "query_origin_kind": receipt["query_origin_kind"],
        "ids_sha256": receipt.get("ids_sha256"),
    }
    (output / "oracle_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({"oracle_layers": len(LAYERS), "source_length": n,
                      "meta": str(output / "oracle_meta.json")}))


if __name__ == "__main__":
    main()
