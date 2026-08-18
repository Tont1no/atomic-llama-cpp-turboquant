#!/usr/bin/env python3
"""Verify lossless ModelOpt E4M3 conversion against its source checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "gguf-py"))

from conversion import get_model_class, load_all_models  # noqa: E402
from conversion.base import LazyTorchTensor, ModelBase, ModelType, get_model_architecture  # noqa: E402
from gguf import GGMLQuantizationType, GGUFReader, LlamaFileType  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=None)
    args = parser.parse_args()

    load_all_models()
    hparams = ModelBase.load_hparams(args.model_dir, False)
    architecture = get_model_architecture(hparams, ModelType.TEXT)
    model_class = get_model_class(architecture, mmproj=False)
    model = model_class(
        args.model_dir,
        LlamaFileType.GUESSED,
        Path("unused.gguf"),
        dry_run=True,
        keep_fp8_e4m3=True,
    )

    reader = GGUFReader(args.gguf)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    expected_hash = hashlib.sha256()
    actual_hash = hashlib.sha256()
    mismatches: list[str] = []
    count = 0

    with torch.inference_mode():
        for name in sorted(model.model_tensors):
            if not name.endswith(".weight"):
                continue
            weight = LazyTorchTensor.to_eager(model.model_tensors[name]())
            if weight.dtype != torch.float8_e4m3fn:
                continue

            bid = next((int(part) for part in name.split(".") if part.isdecimal()), None)
            transformed = list(model.modify_tensors(weight, name, bid))
            if len(transformed) != 1:
                mismatches.append(f"{name}: transform produced {len(transformed)} tensors")
                continue
            new_name, transformed_weight = transformed[0]
            raw = transformed_weight.contiguous().view(torch.uint8).cpu().numpy()
            actual = tensors.get(new_name)
            if actual is None or actual.tensor_type != GGMLQuantizationType.F8_E4M3:
                mismatches.append(f"{name}: missing F8_E4M3 {new_name}")
                continue
            if tuple(actual.data.shape) != tuple(raw.shape) or not (actual.data == raw).all():
                mismatches.append(f"{name}: raw byte/shape mismatch in {new_name}")
                continue

            expected_hash.update(raw.tobytes())
            actual_hash.update(actual.data.tobytes())
            base_src = name.removesuffix(".weight")
            base_dst = new_name.removesuffix(".weight")
            for source_suffix, target_suffix in ((".weight_scale", ".scale"), (".input_scale", ".input_scale")):
                expected_scale = float(
                    LazyTorchTensor.to_eager(model.model_tensors[base_src + source_suffix]()).float().reshape(-1)[0]
                )
                scale_tensor = tensors.get(base_dst + target_suffix)
                if scale_tensor is None or scale_tensor.tensor_type != GGMLQuantizationType.F32 or \
                        scale_tensor.n_elements != 1 or float(scale_tensor.data.reshape(-1)[0]) != expected_scale:
                    mismatches.append(f"{name}: scale mismatch for {base_dst + target_suffix}")
            count += 1

    if args.expected_count is not None and count != args.expected_count:
        mismatches.append(f"expected {args.expected_count} FP8 weights, checked {count}")
    if expected_hash.digest() != actual_hash.digest():
        mismatches.append("aggregate raw-byte hash mismatch")

    print(f"architecture={architecture} checked={count} mismatches={len(mismatches)}")
    print(f"expected_sha256={expected_hash.hexdigest()}")
    print(f"actual_sha256={actual_hash.hexdigest()}")
    for mismatch in mismatches[:20]:
        print(mismatch, file=sys.stderr)
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
