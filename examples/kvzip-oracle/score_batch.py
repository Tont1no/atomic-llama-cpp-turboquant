"""Score prepared KVzap runs one context at a time with cached effective W_O."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import score


FIELDS = {"conversation_id", "similarity_group", "source", "prompt_sha256", "response_sha256",
          "input_receipt", "capture_receipt", "oracle_dir"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--export-receipt", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--limit", type=int, default=110)
    args = parser.parse_args()
    root = (args.repo.resolve() / "tmp" / "bonsai-training-20260924").resolve()
    runs_path = score.private(args.runs, root)
    export_path = score.private(args.export_receipt, root)
    runs = json.loads(runs_path.read_text(encoding="utf-8"))
    if not isinstance(runs, list) or not 1 <= len(runs) <= 110 or not 1 <= args.limit <= 110:
        raise ValueError("Invalid KVzap batch size or limit")
    checked = []
    for run in runs[:args.limit]:
        if not isinstance(run, dict) or set(run) != FIELDS:
            raise ValueError("Unexpected runs.json row schema")
        input_path = score.private(Path(run["input_receipt"]), root)
        capture_path = score.private(Path(run["capture_receipt"]), root)
        oracle_dir = score.private(Path(run["oracle_dir"]), root)
        input_receipt = json.loads(input_path.read_text(encoding="utf-8"))
        capture = json.loads(capture_path.read_text(encoding="utf-8"))
        if (capture_path.name != "receipt.json" or input_path.name != "input-receipt.json" or
            capture.get("kind") != "kvzap_full_chat_native_attention_capture_v3" or
            capture.get("ids_sha256") != input_receipt.get("ids_sha256") or
            capture.get("replay_template_sha256") != input_receipt.get("template_sha256") or
            capture.get("source_start_position") != input_receipt.get("source_start") or
            capture.get("source_end_position") != input_receipt.get("source_end") or
            capture.get("repeat_start_position") != input_receipt.get("repeat_start") or
            capture.get("repeat_end_position") != input_receipt.get("repeat_end") or
            any(run[key] != input_receipt.get(key) for key in ("conversation_id", "similarity_group", "source", "prompt_sha256")) or
            run["response_sha256"] != run["prompt_sha256"]):
            raise ValueError("Batch input/capture provenance mismatch")
        checked.append((capture_path.parent, oracle_dir, capture["ids_sha256"]))
    scored = skipped = 0
    for capture_dir, oracle_dir, ids_sha256 in checked:
        meta_path = oracle_dir / "oracle_meta.json"
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("ids_sha256") != ids_sha256 or any(not (oracle_dir / f"layer-{layer}.npz").is_file() for layer in score.LAYERS):
                raise ValueError("Existing oracle output is incomplete or from different IDs")
            skipped += 1
            continue
        if oracle_dir.exists():
            raise ValueError("Partial oracle directory requires operator review")
        argv = sys.argv
        try:
            sys.argv = ["score.py", "--repo", str(args.repo), "--capture", str(capture_dir),
                        "--export-receipt", str(export_path), "--output", str(oracle_dir),
                        "--device", args.device]
            score.main()
        finally:
            sys.argv = argv
        scored += 1
    print(json.dumps({"scored": scored, "skipped": skipped, "runs": str(runs_path)}))


if __name__ == "__main__":
    main()
