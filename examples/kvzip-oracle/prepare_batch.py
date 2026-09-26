"""Prepare private KVzap candidate/heldout runs with one tokenizer load."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from prepare_input import prepare_one, private


FIELDS = {"conversation_id", "similarity_group", "source", "prompt_text", "prompt_sha256"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--name", choices=("candidate64", "heldout46"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = (args.repo.resolve() / "tmp" / "bonsai-training-20260924").resolve()
    source = private(args.jsonl, root)
    output = private(args.output_root, root)
    expected_count = 64 if args.name == "candidate64" else 46
    if not source.is_file() or not output.is_dir():
        raise ValueError("Source JSONL or marked output root missing")
    runs_path = output / f"{args.name}-runs.json"
    batch_root = output / args.name
    if runs_path.exists() or batch_root.exists():
        raise ValueError("Batch output already exists")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    with source.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    if len(records) != expected_count:
        raise ValueError("Unexpected selected context count")
    tokenizer = AutoTokenizer.from_pretrained(str(root / "target-tokenizer-qwen38-27b"),
                                               local_files_only=True, use_fast=True)
    if not tokenizer.is_fast or not tokenizer.chat_template:
        raise ValueError("Verified local fast tokenizer with chat template is required")
    (batch_root / "input").mkdir(parents=True)
    (batch_root / "capture").mkdir()
    (batch_root / "oracle").mkdir()
    runs = []
    for index, record in enumerate(records):
        if set(record) != FIELDS or record["source"] not in ("curated", "synthetic"):
            raise ValueError("Invalid selected prompt schema/source")
        if hashlib.sha256(record["prompt_text"].encode()).hexdigest() != record["prompt_sha256"]:
            raise ValueError("Selected prompt hash mismatch")
        metadata = {key: record[key] for key in ("conversation_id", "similarity_group", "source")}
        metadata.update(candidate_index=index, candidate_path=str(source), candidate_sha256=source_sha)
        item = f"{index:03d}"
        input_dir = batch_root / "input" / item
        receipt = prepare_one(record["prompt_text"], metadata, input_dir, tokenizer)
        if receipt["prompt_sha256"] != record["prompt_sha256"] or not 750 < receipt["text_token_length"] < 1250:
            raise ValueError("Selected original prompt does not meet pinned KVzap length/hash")
        runs.append({"conversation_id": record["conversation_id"],
                     "similarity_group": record["similarity_group"],
                     "source": record["source"],
                     "prompt_sha256": receipt["prompt_sha256"],
                     "response_sha256": receipt["prompt_sha256"],
                     "input_receipt": str(input_dir / "input-receipt.json"),
                     "capture_receipt": str(batch_root / "capture" / item / "receipt.json"),
                     "oracle_dir": str(batch_root / "oracle" / item)})
    runs_path.write_text(json.dumps(runs, indent=2), encoding="utf-8")
    print(json.dumps({"contexts": len(runs), "runs": str(runs_path), "source_sha256": source_sha}))


if __name__ == "__main__":
    main()
