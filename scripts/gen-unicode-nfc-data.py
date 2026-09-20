#!/usr/bin/env python3
"""Generate and verify the Unicode 9 NFC tables matching HF tokenizers 0.22.1.

The generated C++ header is deliberately separate from the existing llama.cpp
Unicode tables.  Those tables only contain the old one-code-point NFD shortcut,
which cannot implement recursive canonical decomposition and composition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


UNICODE_MAX = 0x10FFFF
SBASE, LBASE, VBASE, TBASE = 0xAC00, 0x1100, 0x1161, 0x11A7
LCOUNT, VCOUNT, TCOUNT = 19, 21, 28
NCOUNT, SCOUNT = VCOUNT * TCOUNT, LCOUNT * VCOUNT * TCOUNT
UNICODE_VERSION = "9.0.0"
EXPECTED_NORMALIZATION_TEST_ROWS = 18722

# The generator is intentionally offline and refuses to consume a silently
# changed UCD snapshot.  Update these pins only with a deliberate Unicode-data
# review and a new generated table.
PINNED_INPUT_SHA256 = {
    "UnicodeData.txt": "68DFC414D28257B9B5D6DDBB8B466C768C00EBDF6CBF7784364A9B6CAD55EE8F",
    "DerivedNormalizationProps.txt": "02D8E12CEA7C61A9F3CC5FBF8EACEADF8DA23FE83D60C64CF647088830B810CC",
    "NormalizationTest.txt": "2D48D848656B3CF889DF59980AB13551988950C4CA8C5190A17C691A17F8B9BB",
}


def parse_code_range(value: str) -> tuple[int, int]:
    value = value.strip()
    if ".." in value:
        first, last = value.split("..", 1)
        return int(first, 16), int(last, 16)
    point = int(value, 16)
    return point, point


def verify_pinned_input(name: str, path: Path) -> str:
    actual = hashlib.sha256(path.read_bytes()).hexdigest().upper()
    expected = PINNED_INPUT_SHA256[name]
    if actual != expected:
        raise SystemExit(f"{name}: SHA-256 mismatch; expected {expected}, got {actual}")
    return actual


def parse_ucd(unicode_data: Path, derived_props: Path):
    decomp: dict[int, list[int]] = {}
    ccc: dict[int, int] = {}
    for raw in unicode_data.read_text(encoding="utf-8").splitlines():
        fields = raw.split(";")
        if len(fields) < 6:
            continue
        point = int(fields[0], 16)
        name = fields[1]
        if name.endswith(", First>"):
            continue
        if name.endswith(", Last>"):
            continue
        value = int(fields[3])
        if value:
            ccc[point] = value
        mapping = fields[5].strip()
        if mapping and not mapping.startswith("<"):
            decomp[point] = [int(item, 16) for item in mapping.split()]

    exclusions: set[int] = set()
    in_exclusions = False
    for raw in derived_props.read_text(encoding="utf-8").splitlines():
        if raw.startswith("# Derived Property:"):
            in_exclusions = raw.strip() == "# Derived Property: Full_Composition_Exclusion"
            continue
        if not in_exclusions or not raw.strip() or raw.lstrip().startswith("#"):
            continue
        data = raw.split("#", 1)[0].split(";", 1)
        if len(data) != 2 or data[1].strip() != "Full_Composition_Exclusion":
            continue
        first, last = parse_code_range(data[0])
        exclusions.update(range(first, last + 1))

    return decomp, ccc, exclusions


def parse_test_codepoints(value: str) -> list[int]:
    value = value.strip()
    return [] if not value else [int(item, 16) for item in value.split()]


def make_normalizer(decomp: dict[int, list[int]], ccc: dict[int, int], exclusions: set[int]):
    composition: dict[tuple[int, int], int] = {}
    for composite, mapping in decomp.items():
        if len(mapping) != 2 or composite in exclusions or ccc.get(mapping[0], 0) != 0:
            continue
        key = (mapping[0], mapping[1])
        old = composition.setdefault(key, composite)
        if old != composite:
            raise ValueError(f"duplicate canonical composition {key!r}: {old:04X}/{composite:04X}")

    def decompose_one(point: int, output: list[int]) -> None:
        if SBASE <= point < SBASE + SCOUNT:
            index = point - SBASE
            output.append(LBASE + index // NCOUNT)
            output.append(VBASE + (index % NCOUNT) // TCOUNT)
            tail = index % TCOUNT
            if tail:
                output.append(TBASE + tail)
            return
        mapping = decomp.get(point)
        if mapping is None:
            output.append(point)
            return
        for item in mapping:
            decompose_one(item, output)

    def normalize(points: list[int]) -> list[int]:
        decomposed: list[int] = []
        for point in points:
            decompose_one(point, decomposed)

        # Canonical ordering is stable within each starter segment.  Inserting
        # each mark backwards is small, portable, and preserves equal-CCC order.
        for index in range(1, len(decomposed)):
            mark_ccc = ccc.get(decomposed[index], 0)
            if mark_ccc == 0:
                continue
            cursor = index
            while cursor:
                previous_ccc = ccc.get(decomposed[cursor - 1], 0)
                if previous_ccc == 0 or previous_ccc <= mark_ccc:
                    break
                decomposed[cursor], decomposed[cursor - 1] = decomposed[cursor - 1], decomposed[cursor]
                cursor -= 1

        result: list[int] = []
        starter = None
        last_ccc = 0
        for point in decomposed:
            point_ccc = ccc.get(point, 0)
            if starter is not None and (last_ccc == 0 or last_ccc < point_ccc):
                composite = composition.get((result[starter], point))
                if composite is None and point_ccc == 0:
                    first = result[starter]
                    if LBASE <= first < LBASE + LCOUNT and VBASE <= point < VBASE + VCOUNT:
                        composite = SBASE + ((first - LBASE) * VCOUNT + (point - VBASE)) * TCOUNT
                    elif SBASE <= first < SBASE + SCOUNT and (first - SBASE) % TCOUNT == 0 and TBASE < point <= TBASE + (TCOUNT - 1):
                        composite = first + (point - TBASE)
                if composite is not None:
                    result[starter] = composite
                    continue
            if point_ccc == 0:
                starter = len(result)
                last_ccc = 0
            else:
                last_ccc = point_ccc
            result.append(point)
        return result

    return composition, normalize


def verify(normalize, normalization_test: Path) -> tuple[int, int]:
    rows = 0
    failures = 0
    for raw in normalization_test.read_text(encoding="utf-8").splitlines():
        if not raw or raw.startswith("#"):
            continue
        fields = raw.split("#", 1)[0].split(";")
        if len(fields) < 5:
            continue
        columns = [parse_test_codepoints(field) for field in fields[:5]]
        expected_nfc = columns[1]
        for index in (0, 1, 2):
            if normalize(columns[index]) != expected_nfc:
                failures += 1
                if failures <= 5:
                    print(f"NFC failure row {rows + 1}, column c{index + 1}")
        expected_nfc_compat = columns[3]
        for index in (3, 4):
            if normalize(columns[index]) != expected_nfc_compat:
                failures += 1
                if failures <= 5:
                    print(f"NFC failure row {rows + 1}, column c{index + 1}")
        rows += 1
    return rows, failures


def format_values(values: list[int], width: int = 12) -> str:
    rows = []
    for offset in range(0, len(values), width):
        rows.append("    " + ", ".join(f"0x{value:04X}u" for value in values[offset : offset + width]) + ",")
    return "\n".join(rows)


def generate_header(output: Path, decomp: dict[int, list[int]], ccc: dict[int, int], composition: dict[tuple[int, int], int], hashes: dict[str, str]) -> None:
    values: list[int] = []
    decomp_entries = []
    for point in sorted(decomp):
        mapping = decomp[point]
        offset = len(values)
        values.extend(mapping)
        decomp_entries.append((point, offset, len(mapping)))
    ccc_entries = sorted(ccc.items())
    composition_entries = sorted((first, second, composite) for (first, second), composite in composition.items())

    lines = [
        "// Generated by scripts/gen-unicode-nfc-data.py from pinned Unicode 9.0.0 UCD files.",
        f"// UnicodeData.txt SHA-256: {hashes['UnicodeData.txt']}",
        f"// DerivedNormalizationProps.txt SHA-256: {hashes['DerivedNormalizationProps.txt']}",
        f"// NormalizationTest.txt SHA-256: {hashes['NormalizationTest.txt']}",
        "// Full_Composition_Exclusion is read from DerivedNormalizationProps.txt.",
        "// Do not regenerate or overwrite src/unicode-data.* with this table.",
        "#pragma once",
        "",
        "#include <cstdint>",
        "",
        "struct unicode_nfc_decomp_entry { uint32_t cpt; uint32_t offset; uint16_t length; };",
        "struct unicode_nfc_ccc_entry { uint32_t cpt; uint8_t ccc; };",
        "struct unicode_nfc_composition_entry { uint32_t first; uint32_t second; uint32_t composite; };",
        "",
        "static constexpr uint32_t unicode_nfc_decomp_values[] = {",
        format_values(values),
        "};",
        "",
        "static constexpr unicode_nfc_decomp_entry unicode_nfc_decomp_entries[] = {",
    ]
    lines.extend(f"    {{ 0x{point:04X}u, {offset}u, {length}u }}," for point, offset, length in decomp_entries)
    lines.extend([
        "};",
        "",
        "static constexpr unicode_nfc_ccc_entry unicode_nfc_ccc_entries[] = {",
    ])
    lines.extend(f"    {{ 0x{point:04X}u, {value}u }}," for point, value in ccc_entries)
    lines.extend([
        "};",
        "",
        "static constexpr unicode_nfc_composition_entry unicode_nfc_composition_entries[] = {",
    ])
    lines.extend(f"    {{ 0x{first:04X}u, 0x{second:04X}u, 0x{composite:04X}u }}," for first, second, composite in composition_entries)
    lines.extend(["};", ""])
    output.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ucd", type=Path, required=True)
    parser.add_argument("--derived", type=Path, required=True)
    parser.add_argument("--tests", type=Path, required=True)
    parser.add_argument("--header", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    hashes = {
        "UnicodeData.txt": verify_pinned_input("UnicodeData.txt", args.ucd),
        "DerivedNormalizationProps.txt": verify_pinned_input("DerivedNormalizationProps.txt", args.derived),
        "NormalizationTest.txt": verify_pinned_input("NormalizationTest.txt", args.tests),
    }
    decomp, ccc, exclusions = parse_ucd(args.ucd, args.derived)
    composition, normalize = make_normalizer(decomp, ccc, exclusions)
    rows, failures = verify(normalize, args.tests)
    report = {
        "generator": "scripts/gen-unicode-nfc-data.py",
        "unicode_version": UNICODE_VERSION,
        "normalization": "NFC",
        "normalization_test_rows": rows,
        "normalization_test_failures": failures,
        "canonical_decomposition_entries": len(decomp),
        "canonical_decomposition_values": sum(len(value) for value in decomp.values()),
        "canonical_combining_class_entries": len(ccc),
        "composition_entries": len(composition),
        "full_composition_exclusion_codepoints": len(exclusions),
        "inputs": hashes,
        "input_hashes_verified": True,
        "algorithm": "recursive canonical decomposition; stable CCC ordering; blocked canonical composition; algorithmic Hangul L/V/T",
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, sort_keys=True))
    if failures != 0 or rows != EXPECTED_NORMALIZATION_TEST_ROWS:
        return 1
    generate_header(args.header, decomp, ccc, composition, hashes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
