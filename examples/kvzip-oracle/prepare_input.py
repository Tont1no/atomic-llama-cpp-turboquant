"""Prepare one private full-chat KVzap oracle sequence with exact token spans."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer


REPEAT = "\n\nRepeat the previous context exactly."


def private(path: Path, root: Path) -> Path:
    sys.path.insert(0, str(root.parent.parent / "Inference engine" / "scripts" / "bonsai_training"))
    from data import require_private

    return require_private(path, root)


def boundary(offsets: list[tuple[int, int]], char_pos: int) -> int:
    # Match pinned kvzap/data.py: first token whose start is at/after char_pos.
    return next((i for i, (start, _) in enumerate(offsets) if start >= char_pos), len(offsets))


def checked_span(offsets: list[tuple[int, int]], start: int, end: int, char_start: int, char_end: int) -> None:
    if end <= start:
        raise ValueError("Prompt token span is empty")
    for lo, hi in offsets[start:end]:
        if lo < char_start or hi > char_end or hi <= lo:
            raise ValueError("Tokenizer crossed a prompt/template boundary")


def prepare_one(prompt: str, metadata: dict, output: Path, tokenizer) -> dict:
    if output.exists():
        raise ValueError("Output already exists")
    prompt = prompt.strip()
    if not prompt or any(marker in prompt for marker in ("<|image_pad|>", "<|vision_start|>", "<|video_pad|>")):
        raise ValueError("Only nonempty text-only prompts are accepted")
    messages = [{"role": "user", "content": prompt + REPEAT}, {"role": "assistant", "content": prompt}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False)
    pieces = rendered.split(prompt)
    if len(pieces) != 3:
        raise ValueError("Cannot locate exactly two full prompt occurrences in the chat template")
    source_char_start = len(pieces[0])
    source_char_end = source_char_start + len(prompt)
    repeat_char_start = source_char_end + len(pieces[1])
    repeat_char_end = repeat_char_start + len(prompt)
    encoded = tokenizer(rendered, return_offsets_mapping=True)
    ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    source_start = boundary(offsets, source_char_start)
    source_end = boundary(offsets, source_char_end)
    repeat_start = boundary(offsets, repeat_char_start)
    repeat_end = boundary(offsets, repeat_char_end)
    checked_span(offsets, source_start, source_end, source_char_start, source_char_end)
    checked_span(offsets, repeat_start, repeat_end, repeat_char_start, repeat_char_end)
    if not 0 <= source_start < source_end <= repeat_start < repeat_end <= len(ids) <= 4096:
        raise ValueError("Full-chat token spans overlap or exceed context")
    output.mkdir(parents=False)
    (output / "ids.txt").write_text("\n".join(map(str, ids)) + "\n", encoding="ascii")
    (output / "protected.txt").write_bytes(b"")
    receipt = {
        "kind": "kvzap_full_chat_token_spans_v1",
        "source_start": source_start,
        "source_end": source_end,
        "repeat_start": repeat_start,
        "repeat_end": repeat_end,
        "input_length": len(ids),
        "source_length": source_end - source_start,
        "repeat_length": repeat_end - repeat_start,
        "text_token_length": len(tokenizer.encode(prompt)),
        "template_sha256": hashlib.sha256(tokenizer.chat_template.encode("utf-8")).hexdigest(),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "rendered_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        "ids_sha256": hashlib.sha256((output / "ids.txt").read_bytes()).hexdigest(),
        **metadata,
    }
    (output / "input-receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", type=Path)
    source.add_argument("--jsonl", type=Path)
    parser.add_argument("--index", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = (args.repo.resolve() / "tmp" / "bonsai-training-20260924").resolve()
    output = private(args.output, root)
    metadata = {}
    if args.text is not None:
        if args.index is not None:
            raise ValueError("--index requires --jsonl")
        prompt = private(args.text, root).read_text(encoding="utf-8")
    else:
        if args.index is None or args.index < 0:
            raise ValueError("--jsonl requires a nonnegative --index")
        candidate = private(args.jsonl, root)
        with candidate.open(encoding="utf-8") as stream:
            record = next((json.loads(line) for i, line in enumerate(stream) if i == args.index), None)
        if record is None or set(record) != {"conversation_id", "similarity_group", "source", "prompt_text", "prompt_sha256"}:
            raise ValueError("Missing candidate row or invalid five-field schema")
        prompt = record["prompt_text"]
        if hashlib.sha256(prompt.encode("utf-8")).hexdigest() != record["prompt_sha256"]:
            raise ValueError("Candidate prompt hash mismatch")
        metadata = {key: record[key] for key in ("conversation_id", "similarity_group", "source")}
        metadata.update(candidate_index=args.index, candidate_path=str(candidate),
                        candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest())
    tokenizer_dir = root / "target-tokenizer-qwen38-27b"
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True, use_fast=True)
    if not tokenizer.is_fast or not tokenizer.chat_template:
        raise ValueError("Verified local fast tokenizer with chat template is required")
    receipt = prepare_one(prompt, metadata, output, tokenizer)
    print(json.dumps({"source_length": receipt["source_length"], "repeat_length": receipt["repeat_length"],
                      "receipt": str(output / "input-receipt.json")}))


if __name__ == "__main__":
    main()
