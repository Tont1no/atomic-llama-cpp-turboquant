#!/usr/bin/env python3
"""Add an exact row-packed DFlash/DSpark KV projection to a Q8_0 GGUF.

The source file is never modified. Existing tensor payloads are copied without
dequantization or requantization. The added tensor is the bytewise row
concatenation ``K0,V0,K1,V1,...`` of the existing Q8_0 projection tensors.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# Prefer the in-tree gguf package when this script is run from a checkout.
if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).parent.parent.parent.parent / "gguf-py").exists():
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import gguf


logger = logging.getLogger("gguf-add-dflash-stacked-kv")
STACKED_NAME = "dflash.attn_kv_stacked.weight"
MIN_ROWS_KEY = "dflash.stacked_kv_min_rows"
HASH_CHUNK_BYTES = 16 * 1024 * 1024


def field_value(reader: gguf.GGUFReader, name: str) -> Any:
    field = reader.get_field(name)
    if field is None:
        raise ValueError(f"required GGUF metadata {name!r} is missing")
    return field.contents()


def tensor_map(reader: gguf.GGUFReader) -> dict[str, gguf.ReaderTensor]:
    result = {tensor.name: tensor for tensor in reader.tensors}
    if len(result) != len(reader.tensors):
        raise ValueError("input GGUF contains duplicate tensor names")
    return result


def metadata_fields(reader: gguf.GGUFReader) -> Iterable[gguf.ReaderField]:
    for field in reader.fields.values():
        if not field.name.startswith("GGUF."):
            yield field


def copy_metadata(reader: gguf.GGUFReader, writer: gguf.GGUFWriter) -> None:
    for field in metadata_fields(reader):
        # GGUFWriter creates this key from its arch constructor argument.
        if field.name == gguf.Keys.General.ARCHITECTURE:
            continue
        value_type = field.types[0]
        sub_type = field.types[-1] if value_type == gguf.GGUFValueType.ARRAY else None
        value = field.contents()
        if value_type == gguf.GGUFValueType.ARRAY and (
                not value or any(isinstance(item, (list, tuple)) for item in value)):
            raise ValueError(f"metadata array {field.name!r} cannot be reproduced safely")
        writer.add_key_value(field.name, value, value_type, sub_type=sub_type)


def hash_array(data: np.ndarray[Any, Any]) -> str:
    if not data.flags.c_contiguous:
        raise ValueError("cannot hash a non-contiguous tensor payload safely")
    raw = memoryview(data).cast("B")
    digest = hashlib.sha256()
    for offset in range(0, len(raw), HASH_CHUNK_BYTES):
        digest.update(raw[offset:offset + HASH_CHUNK_BYTES])
    return digest.hexdigest()


def validate_metadata(source: gguf.GGUFReader, output: gguf.GGUFReader, min_rows: int) -> None:
    source_fields = list(metadata_fields(source))
    output_fields = list(metadata_fields(output))
    source_names = [field.name for field in source_fields]
    output_names = [field.name for field in output_fields]
    expected_names = source_names if MIN_ROWS_KEY in source_names else [*source_names, MIN_ROWS_KEY]
    if output_names != expected_names:
        raise ValueError("metadata key order changed during repack")

    output_by_name = {field.name: field for field in output_fields}
    for old in source_fields:
        new = output_by_name[old.name]
        if old.name == MIN_ROWS_KEY:
            continue
        if old.types != new.types or old.contents() != new.contents():
            raise ValueError(f"metadata field {old.name!r} changed during repack")

    threshold = output_by_name[MIN_ROWS_KEY]
    if threshold.types != [gguf.GGUFValueType.UINT32] or int(threshold.contents()) != min_rows:
        raise ValueError(f"metadata field {MIN_ROWS_KEY!r} has the wrong type or value")


def validate_output(
        source: gguf.GGUFReader,
        output_path: Path,
        projections: list[gguf.ReaderTensor],
        packed: np.ndarray[Any, Any],
        min_rows: int) -> dict[str, Any]:
    output = gguf.GGUFReader(output_path, "r")
    validate_metadata(source, output, min_rows)

    old_names = [tensor.name for tensor in source.tensors]
    new_names = [tensor.name for tensor in output.tensors]
    if new_names != [*old_names, STACKED_NAME]:
        raise ValueError("original tensor order was not preserved or packed tensor is not last")

    for old, new in zip(source.tensors, output.tensors[:-1]):
        if old.tensor_type != new.tensor_type or tuple(old.shape) != tuple(new.shape) or old.n_bytes != new.n_bytes:
            raise ValueError(f"tensor descriptor changed for {old.name!r}")
        if new.data_offset % output.alignment != 0:
            raise ValueError(f"tensor payload is misaligned for {old.name!r}")
        if hash_array(old.data) != hash_array(new.data):
            raise ValueError(f"tensor payload changed for {old.name!r}")

    stacked = output.tensors[-1]
    expected_ne0 = int(projections[0].shape[0])
    expected_ne1 = sum(int(tensor.shape[1]) for tensor in projections)
    if stacked.tensor_type != gguf.GGMLQuantizationType.Q8_0:
        raise ValueError("packed tensor is not Q8_0")
    if tuple(int(value) for value in stacked.shape) != (expected_ne0, expected_ne1):
        raise ValueError(f"packed tensor has unexpected shape {tuple(stacked.shape)}")
    if stacked.data_offset % output.alignment != 0:
        raise ValueError("packed tensor payload is misaligned")
    if hash_array(stacked.data) != hash_array(packed):
        raise ValueError("packed tensor payload changed while writing")

    row = 0
    for projection in projections:
        next_row = row + projection.data.shape[0]
        if hash_array(stacked.data[row:next_row]) != hash_array(projection.data):
            raise ValueError(f"packed row segment differs from {projection.name!r}")
        row = next_row

    return {
        "source_tensor_count": len(source.tensors),
        "output_tensor_count": len(output.tensors),
        "packed_shape_ggml": [expected_ne0, expected_ne1],
        "packed_bytes": stacked.n_bytes,
        "packed_sha256": hash_array(stacked.data),
        "stacked_kv_min_rows": min_rows,
    }


def publish_without_overwrite(temporary: Path, output: Path) -> None:
    # A hard link is atomic and fails if output already exists. The temporary
    # file is deliberately created in the output directory, so this is a
    # same-volume operation on supported filesystems (including NTFS).
    try:
        os.link(temporary, output)
    except FileExistsError:
        raise FileExistsError(f"refusing to overwrite existing output: {output}") from None
    except OSError as exc:
        raise OSError(
            f"could not publish output atomically without overwrite ({exc}); "
            f"validated temporary file remains at {temporary}"
        ) from exc
    temporary.unlink()


def repack(input_path: Path, output_path: Path, min_rows: int | None = None) -> dict[str, Any]:
    if min_rows is not None and (min_rows < 0 or min_rows > 0xffffffff):
        raise ValueError("min_rows must be between 0 and 4294967295")

    input_path = input_path.resolve(strict=True)
    output_path = output_path.resolve(strict=False)
    if input_path == output_path:
        raise ValueError("input and output must be different files")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")
    if not output_path.parent.is_dir():
        raise FileNotFoundError(f"output directory does not exist: {output_path.parent}")

    reader = gguf.GGUFReader(input_path, "r")
    if field_value(reader, gguf.Keys.General.ARCHITECTURE) != "dflash":
        raise ValueError("only GGUFs with general.architecture=dflash are supported")
    if any(field.name.startswith("split.") for field in metadata_fields(reader)):
        raise ValueError("split GGUF input is not supported; join it before repacking")
    if any(tensor.name == STACKED_NAME for tensor in reader.tensors):
        raise ValueError(f"input already contains {STACKED_NAME!r}")

    block_count = int(field_value(reader, "dflash.block_count"))
    by_name = tensor_map(reader)
    if min_rows is None:
        existing_threshold = reader.get_field(MIN_ROWS_KEY)
        if existing_threshold is not None:
            if existing_threshold.types != [gguf.GGUFValueType.UINT32]:
                raise ValueError(f"existing metadata {MIN_ROWS_KEY!r} must be UINT32")
            min_rows = int(existing_threshold.contents())
        else:
            hc_field = reader.get_field("dflash.hyper_connection.count")
            hc_count = int(hc_field.contents()) if hc_field is not None else 0
            is_dspark = "markov_w1.weight" in by_name or hc_count > 0
            min_rows = 0 if is_dspark else 16
    projections: list[gguf.ReaderTensor] = []
    for block in range(block_count):
        for kind in ("k", "v"):
            name = f"blk.{block}.attn_{kind}.weight"
            tensor = by_name.get(name)
            if tensor is None:
                raise ValueError(f"required projection tensor {name!r} is missing")
            projections.append(tensor)

    first = projections[0]
    k_shape = tuple(int(value) for value in projections[0].shape)
    v_shape = tuple(int(value) for value in projections[1].shape)
    if first.tensor_type != gguf.GGMLQuantizationType.Q8_0 or len(first.shape) != 2:
        raise ValueError("stacked repack currently supports 2D Q8_0 K/V tensors only")
    if int(first.shape[0]) % 32 != 0:
        raise ValueError("Q8_0 reduction dimension must be divisible by 32")
    expected_row_bytes = int(first.shape[0]) // 32 * 34
    for index, tensor in enumerate(projections):
        expected_shape = k_shape if index % 2 == 0 else v_shape
        if tensor.tensor_type != first.tensor_type:
            raise ValueError(f"projection {tensor.name!r} has type {tensor.tensor_type.name}, expected Q8_0")
        if tuple(int(value) for value in tensor.shape) != expected_shape or int(tensor.shape[0]) != int(first.shape[0]):
            raise ValueError(f"projection {tensor.name!r} has incompatible shape {tuple(tensor.shape)}")
        if tensor.data.dtype != np.uint8 or tensor.data.ndim != 2 or tensor.data.shape != (int(tensor.shape[1]), expected_row_bytes):
            raise ValueError(f"projection {tensor.name!r} has incompatible raw row layout")
        if tensor.data.nbytes != tensor.n_bytes or not tensor.data.flags.c_contiguous:
            raise ValueError(f"projection {tensor.name!r} is not row-contiguous")

    logger.info("Packing %d exact Q8_0 projections from %s", len(projections), input_path)
    packed_rows = sum(tensor.data.shape[0] for tensor in projections)
    packed_path = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.stack.tmp")
    packed = np.memmap(packed_path, dtype=np.uint8, mode="w+", shape=(packed_rows, expected_row_bytes))
    row = 0
    for tensor in projections:
        next_row = row + tensor.data.shape[0]
        packed[row:next_row] = tensor.data
        row = next_row
    packed.flush()

    required_free = input_path.stat().st_size + 2 * packed.nbytes + 64 * 1024 * 1024
    if shutil.disk_usage(output_path.parent).free < required_free:
        packed._mmap.close()
        packed_path.unlink()
        raise OSError(f"insufficient free space; need at least {required_free} bytes")

    temporary = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
    writer: gguf.GGUFWriter | None = None
    validated = False
    try:
        writer = gguf.GGUFWriter(temporary, arch="dflash", endianess=reader.endianess)
        writer.data_alignment = reader.alignment
        copy_metadata(reader, writer)
        writer.add_uint32(MIN_ROWS_KEY, min_rows)

        for tensor in reader.tensors:
            writer.add_tensor_info(
                tensor.name, tensor.data.shape, tensor.data.dtype, tensor.data.nbytes,
                raw_dtype=tensor.tensor_type,
            )
        writer.add_tensor_info(
            STACKED_NAME, packed.shape, packed.dtype, packed.nbytes,
            raw_dtype=gguf.GGMLQuantizationType.Q8_0,
        )

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        for index, tensor in enumerate(reader.tensors, start=1):
            logger.info("Copying tensor %d/%d: %s", index, len(reader.tensors), tensor.name)
            writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)
        writer.write_tensor_data(packed, tensor_endianess=reader.endianess)
        writer.flush()
        assert writer.fout is not None
        for fout in writer.fout:
            os.fsync(fout.fileno())
        writer.close()
        writer = None

        logger.info("Validating metadata, tensor order, descriptors, and SHA-256 payloads")
        result = validate_output(reader, temporary, projections, packed, min_rows)
        validated = True
        result.update({
            "input": str(input_path),
            "output": str(output_path),
            "input_size": input_path.stat().st_size,
            "output_size": temporary.stat().st_size,
        })
        publish_without_overwrite(temporary, output_path)
        return result
    except Exception:
        if writer is not None:
            writer.close()
        # Invalid partial writes are removed. A fully validated file is retained
        # only when atomic publication itself is unsupported, as reported above.
        if temporary.exists() and not validated:
            temporary.unlink()
        raise
    finally:
        packed._mmap.close()
        if packed_path.exists():
            packed_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy a single-file Q8_0 DFlash/DSpark GGUF and add an exact stacked KV tensor",
    )
    parser.add_argument("input", type=Path, help="existing single-file DFlash/DSpark Q8_0 GGUF")
    parser.add_argument("output", type=Path, help="new GGUF path; must not already exist")
    parser.add_argument(
        "--min-rows", type=int,
        help="minimum logical rows for stacked projection; 0 disables (auto: DFlash 16, DSpark 0)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    try:
        result = repack(args.input, args.output, args.min_rows)
    except Exception as exc:
        if args.verbose:
            raise
        logger.error("%s", exc)
        raise SystemExit(1) from None
    for key, value in result.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
