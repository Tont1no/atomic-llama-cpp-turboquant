import struct
import numpy as np
import pytest

from gguf.constants import GGMLQuantizationType
from gguf.gguf_reader import GGUFReader
from gguf.gguf_writer import GGUFWriter
from gguf.quants import dequantize, quantize


def _write_gguf(path, n_dims_field, dims):
    buf = b'GGUF' + struct.pack('<IQQ', 3, 1, 0)  # version 3, 1 tensor, 0 kv
    name = b'bad_tensor'
    buf += struct.pack('<Q', len(name)) + name
    buf += struct.pack('<I', n_dims_field)
    for d in dims:
        buf += struct.pack('<Q', d)
    buf += struct.pack('<I', 0)  # dtype F32
    buf += struct.pack('<Q', 0)  # tensor offset
    buf += b'\x00' * 64
    path.write_bytes(buf)


def test_n_dims_upper_bound(tmp_path):
    # crafted file claims 1_000_000 dims; must be rejected, not read past EOF
    p = tmp_path / 'evil_ndims.gguf'
    _write_gguf(p, 1_000_000, [1] * 8)
    with pytest.raises(ValueError, match='exceeds GGML_MAX_DIMS'):
        GGUFReader(p)


def test_dims_product_no_uint64_wraparound(tmp_path):
    # dims whose true product overflows uint64; np.prod would wrap to 4 and
    # silently pass an undersized read. The reader must not accept it.
    dims = [4194305, 4194305, 211106198978564]
    assert int(np.prod(np.array(dims, dtype=np.uint64))) == 4  # the wrap bug
    p = tmp_path / 'evil_overflow.gguf'
    _write_gguf(p, len(dims), dims)
    with pytest.raises(ValueError):
        GGUFReader(p)


def test_f8_e4m3_raw_roundtrip_and_special_values(tmp_path):
    p = tmp_path / 'f8-e4m3.gguf'
    # Includes positive/negative zero, normal values, the finite maxima, and NaNs.
    codes = np.array([
        [0x00, 0x80, 0x38, 0xB8],
        [0x40, 0xC0, 0x7E, 0xFE],
        [0x7F, 0xFF, 0x01, 0x81],
    ], dtype=np.uint8)
    weight_scale = np.array([0.125], dtype=np.float32)
    input_scale = np.array([0.25], dtype=np.float32)

    writer = GGUFWriter(p, 'qwen35')
    writer.add_tensor('blk.0.attn_q.weight', codes, raw_dtype=GGMLQuantizationType.F8_E4M3)
    writer.add_tensor('blk.0.attn_q.scale', weight_scale)
    writer.add_tensor('blk.0.attn_q.input_scale', input_scale)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    reader = GGUFReader(p)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    weight = tensors['blk.0.attn_q.weight']
    assert weight.tensor_type == GGMLQuantizationType.F8_E4M3
    assert tuple(weight.shape) == (4, 3)
    assert np.array_equal(weight.data, codes)
    assert np.array_equal(tensors['blk.0.attn_q.scale'].data, weight_scale)
    assert np.array_equal(tensors['blk.0.attn_q.input_scale'].data, input_scale)

    values = dequantize(codes, GGMLQuantizationType.F8_E4M3)
    assert np.signbit(values[0, 1])
    assert values[0, 2] == 1.0 and values[0, 3] == -1.0
    assert values[1, 2] == 448.0 and values[1, 3] == -448.0
    assert np.isnan(values[2, 0]) and np.isnan(values[2, 1])
    assert values[2, 2] == 2.0 ** -9 and values[2, 3] == -(2.0 ** -9)

    midpoints = np.array([1.0625, 1.1875, -1.0625, -1.1875, -0.0], dtype=np.float32)
    midpoint_codes = quantize(midpoints, GGMLQuantizationType.F8_E4M3)
    assert np.array_equal(midpoint_codes, np.array([0x38, 0x3A, 0xB8, 0xBA, 0x80], dtype=np.uint8))
