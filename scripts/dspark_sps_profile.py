#!/usr/bin/env python3
"""Build and validate deterministic DSpark SPS profiling plans.

This module is CPU-only. The server-side recorder is implemented, while the
``run`` command remains a fail-closed integration point for the future guarded
multi-arm runner and does not start a server.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable


PLAN_SCHEMA = "llama.cpp-dspark-sps-profile-plan/v1"
SCHEDULE_STRATEGY = "sha256-paired-reverse-rotate/v1"
MAX_PLAN_BYTES = 16 * 1024 * 1024
MAX_CELLS = 4096
MAX_SCHEDULE_ENTRIES = 65536
ALLOWED_ACTIVE_SLOTS = {1, 2, 4, 8}
ROOT_KEYS = {
    "schema",
    "seed",
    "schedule_strategy",
    "active_slots",
    "context_ceilings",
    "prefix_caps",
    "max_draft_tokens_per_slot",
    "repetitions",
    "samples_per_cell",
    "warmup_shape_runs",
    "row_axes_by_active",
    "cells",
    "schedule",
    "schedule_sha256",
}
CELL_KEYS = {
    "cell_id",
    "context_ceiling",
    "active_slots",
    "prefix_cap",
    "total_verify_rows",
}
SCHEDULE_KEYS = {
    "ordinal",
    "repetition",
    "position",
    "direction",
    "cell_id",
}


class ContractError(RuntimeError):
    pass


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON object member: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> Any:
    try:
        size = path.stat().st_size
    except OSError as error:
        raise ContractError(f"cannot stat plan: {path}") from error
    if size <= 0 or size > MAX_PLAN_BYTES:
        raise ContractError(f"plan size must be in [1, {MAX_PLAN_BYTES}] bytes")
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_keys)
    except ContractError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot read strict UTF-8 JSON plan: {path}") from error


def write_json(path: Path, value: Any, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise ContractError(f"refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=path.name + ".",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def strict_positive_int(value: Any, field: str, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError(f"{field} must be a positive integer")
    if maximum is not None and value > maximum:
        raise ContractError(f"{field} must be <= {maximum}")
    return value


def strict_nonnegative_int(value: Any, field: str, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError(f"{field} must be a non-negative integer")
    if maximum is not None and value > maximum:
        raise ContractError(f"{field} must be <= {maximum}")
    return value


def integer_list(value: Any, field: str, *, allow_zero: bool = False) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ContractError(f"{field} must be a non-empty integer array")
    result: list[int] = []
    for index, item in enumerate(value):
        if allow_zero:
            result.append(strict_nonnegative_int(item, f"{field}[{index}]"))
        else:
            result.append(strict_positive_int(item, f"{field}[{index}]"))
    if result != sorted(set(result)):
        raise ContractError(f"{field} must be unique and strictly increasing")
    return result


def parse_csv_integers(
    source: str,
    field: str,
    *,
    allow_zero: bool = False,
) -> list[int]:
    parts = source.split(",")
    if not parts or any(not part.strip() for part in parts):
        raise ContractError(f"{field} must be a non-empty comma-separated integer list")
    values: list[int] = []
    for index, part in enumerate(parts):
        try:
            value = int(part.strip(), 10)
        except ValueError as error:
            raise ContractError(f"{field}[{index}] is not an integer") from error
        if value < (0 if allow_zero else 1):
            qualifier = "non-negative" if allow_zero else "positive"
            raise ContractError(f"{field}[{index}] must be {qualifier}")
        values.append(value)
    result = sorted(values)
    if len(result) != len(set(result)):
        raise ContractError(f"{field} must not contain duplicates")
    return result


def require_exact_keys(value: Any, expected: set[str], field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{field} must be an object")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ContractError(f"{field} keys mismatch; missing={missing}, unknown={unknown}")
    return value


def build_cells(
    active_slots: Iterable[int],
    context_ceilings: Iterable[int],
    prefix_caps: Iterable[int],
) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for active in active_slots:
        for context in context_ceilings:
            for prefix in prefix_caps:
                rows = active * (1 + prefix)
                cells.append(
                    {
                        "cell_id": f"ctx{context}-a{active}-k{prefix}-r{rows}",
                        "context_ceiling": context,
                        "active_slots": active,
                        "prefix_cap": prefix,
                        "total_verify_rows": rows,
                    }
                )
    if len(cells) > MAX_CELLS:
        raise ContractError(f"plan has {len(cells)} cells; maximum is {MAX_CELLS}")
    return cells


def seeded_cell_order(cells: list[dict[str, Any]], seed: int) -> list[str]:
    def order_key(cell: dict[str, Any]) -> tuple[str, str]:
        cell_id = cell["cell_id"]
        digest = hashlib.sha256(f"{seed}\0{cell_id}".encode("ascii")).hexdigest()
        return digest, cell_id

    return [cell["cell_id"] for cell in sorted(cells, key=order_key)]


def build_schedule(
    cells: list[dict[str, Any]],
    repetitions: int,
    seed: int,
) -> list[dict[str, Any]]:
    if repetitions < 2 or repetitions % 2 != 0:
        raise ContractError("repetitions must be even and at least 2")
    base = seeded_cell_order(cells, seed)
    if not base:
        raise ContractError("cannot schedule an empty cell set")
    if len(base) * repetitions > MAX_SCHEDULE_ENTRIES:
        raise ContractError(
            f"schedule has {len(base) * repetitions} entries; maximum is {MAX_SCHEDULE_ENTRIES}"
        )

    schedule: list[dict[str, Any]] = []
    ordinal = 0
    for pair_index in range(repetitions // 2):
        rotation = pair_index % len(base)
        forward = base[rotation:] + base[:rotation]
        for repetition, direction, order in (
            (2 * pair_index + 1, "forward", forward),
            (2 * pair_index + 2, "reverse", list(reversed(forward))),
        ):
            for position, cell_id in enumerate(order):
                schedule.append(
                    {
                        "ordinal": ordinal,
                        "repetition": repetition,
                        "position": position,
                        "direction": direction,
                        "cell_id": cell_id,
                    }
                )
                ordinal += 1
    return schedule


def build_plan(
    *,
    seed: int,
    active_slots: list[int],
    context_ceilings: list[int],
    prefix_caps: list[int],
    max_draft_tokens_per_slot: int,
    repetitions: int,
    samples_per_cell: int,
    warmup_shape_runs: int,
) -> dict[str, Any]:
    if set(active_slots) - ALLOWED_ACTIVE_SLOTS:
        raise ContractError("active_slots may only contain 1, 2, 4, and 8")
    if prefix_caps[0] != 0:
        raise ContractError("prefix_caps must include target-only cap 0")
    if prefix_caps[-1] != max_draft_tokens_per_slot:
        raise ContractError("prefix_caps must include max_draft_tokens_per_slot")
    if any(cap > max_draft_tokens_per_slot for cap in prefix_caps):
        raise ContractError("prefix_caps must not exceed max_draft_tokens_per_slot")
    if warmup_shape_runs < 3:
        raise ContractError("warmup_shape_runs must be at least 3")

    cells = build_cells(active_slots, context_ceilings, prefix_caps)
    schedule = build_schedule(cells, repetitions, seed)
    row_axes = {
        str(active): [active * (1 + prefix) for prefix in prefix_caps]
        for active in active_slots
    }
    return {
        "schema": PLAN_SCHEMA,
        "seed": seed,
        "schedule_strategy": SCHEDULE_STRATEGY,
        "active_slots": active_slots,
        "context_ceilings": context_ceilings,
        "prefix_caps": prefix_caps,
        "max_draft_tokens_per_slot": max_draft_tokens_per_slot,
        "repetitions": repetitions,
        "samples_per_cell": samples_per_cell,
        "warmup_shape_runs": warmup_shape_runs,
        "row_axes_by_active": row_axes,
        "cells": cells,
        "schedule": schedule,
        "schedule_sha256": sha256_json(schedule),
    }


def validate_plan(value: Any) -> dict[str, Any]:
    root = require_exact_keys(value, ROOT_KEYS, "root")
    if root["schema"] != PLAN_SCHEMA:
        raise ContractError(f"unsupported plan schema: {root['schema']!r}")
    if root["schedule_strategy"] != SCHEDULE_STRATEGY:
        raise ContractError(f"unsupported schedule strategy: {root['schedule_strategy']!r}")

    seed = strict_nonnegative_int(root["seed"], "seed", (1 << 63) - 1)
    active_slots = integer_list(root["active_slots"], "active_slots")
    context_ceilings = integer_list(root["context_ceilings"], "context_ceilings")
    prefix_caps = integer_list(root["prefix_caps"], "prefix_caps", allow_zero=True)
    max_draft = strict_positive_int(
        root["max_draft_tokens_per_slot"], "max_draft_tokens_per_slot", 16
    )
    repetitions = strict_positive_int(root["repetitions"], "repetitions", 100)
    samples = strict_positive_int(root["samples_per_cell"], "samples_per_cell", 1000000)
    warmup = strict_positive_int(root["warmup_shape_runs"], "warmup_shape_runs", 1000000)

    row_axes = root["row_axes_by_active"]
    if not isinstance(row_axes, dict):
        raise ContractError("row_axes_by_active must be an object")
    expected_row_keys = {str(active) for active in active_slots}
    if set(row_axes) != expected_row_keys:
        raise ContractError("row_axes_by_active keys must exactly match active_slots")
    for active in active_slots:
        rows = integer_list(row_axes[str(active)], f"row_axes_by_active.{active}")
        if any(row < active or row > active * (1 + max_draft) for row in rows):
            raise ContractError(f"row axis for active={active} exceeds its physical domain")

    cells = root["cells"]
    if not isinstance(cells, list) or not cells:
        raise ContractError("cells must be a non-empty array")
    if len(cells) > MAX_CELLS:
        raise ContractError(f"cells exceeds maximum {MAX_CELLS}")
    cell_ids: set[str] = set()
    for index, raw_cell in enumerate(cells):
        cell = require_exact_keys(raw_cell, CELL_KEYS, f"cells[{index}]")
        cell_id = cell["cell_id"]
        if not isinstance(cell_id, str) or not cell_id or len(cell_id) > 160:
            raise ContractError(f"cells[{index}].cell_id must be bounded non-empty text")
        if cell_id in cell_ids:
            raise ContractError(f"duplicate cell_id: {cell_id}")
        cell_ids.add(cell_id)
        strict_positive_int(cell["context_ceiling"], f"cells[{index}].context_ceiling")
        active = strict_positive_int(cell["active_slots"], f"cells[{index}].active_slots")
        prefix = strict_nonnegative_int(cell["prefix_cap"], f"cells[{index}].prefix_cap")
        rows = strict_positive_int(cell["total_verify_rows"], f"cells[{index}].total_verify_rows")
        if rows != active * (1 + prefix):
            raise ContractError(f"cells[{index}] has inconsistent total_verify_rows")

    schedule = root["schedule"]
    if not isinstance(schedule, list) or not schedule:
        raise ContractError("schedule must be a non-empty array")
    if len(schedule) > MAX_SCHEDULE_ENTRIES:
        raise ContractError(f"schedule exceeds maximum {MAX_SCHEDULE_ENTRIES}")
    for index, raw_entry in enumerate(schedule):
        entry = require_exact_keys(raw_entry, SCHEDULE_KEYS, f"schedule[{index}]")
        if strict_nonnegative_int(entry["ordinal"], f"schedule[{index}].ordinal") != index:
            raise ContractError("schedule ordinals must be contiguous and zero-based")
        strict_positive_int(entry["repetition"], f"schedule[{index}].repetition")
        strict_nonnegative_int(entry["position"], f"schedule[{index}].position")
        if entry["direction"] not in {"forward", "reverse"}:
            raise ContractError(f"schedule[{index}].direction is invalid")
        if entry["cell_id"] not in cell_ids:
            raise ContractError(f"schedule[{index}] references an unknown cell")

    schedule_hash = root["schedule_sha256"]
    if (
        not isinstance(schedule_hash, str)
        or len(schedule_hash) != 64
        or any(character not in "0123456789abcdef" for character in schedule_hash)
    ):
        raise ContractError("schedule_sha256 must be lowercase SHA-256 hex")
    if schedule_hash != sha256_json(schedule):
        raise ContractError("schedule_sha256 does not match schedule")

    expected = build_plan(
        seed=seed,
        active_slots=active_slots,
        context_ceilings=context_ceilings,
        prefix_caps=prefix_caps,
        max_draft_tokens_per_slot=max_draft,
        repetitions=repetitions,
        samples_per_cell=samples,
        warmup_shape_runs=warmup,
    )
    for field in ROOT_KEYS:
        if root[field] != expected[field]:
            raise ContractError(f"plan field does not match deterministic contract: {field}")
    return root


def command_plan(args: argparse.Namespace) -> int:
    active_slots = parse_csv_integers(args.active_slots, "active_slots")
    context_ceilings = parse_csv_integers(args.context_ceilings, "context_ceilings")
    prefix_caps = parse_csv_integers(args.prefix_caps, "prefix_caps", allow_zero=True)
    plan = build_plan(
        seed=args.seed,
        active_slots=active_slots,
        context_ceilings=context_ceilings,
        prefix_caps=prefix_caps,
        max_draft_tokens_per_slot=args.max_draft_tokens_per_slot,
        repetitions=args.repetitions,
        samples_per_cell=args.samples_per_cell,
        warmup_shape_runs=args.warmup_shape_runs,
    )
    validate_plan(plan)
    write_json(args.output.resolve(), plan, args.force)
    print(
        canonical(
            {
                "ok": True,
                "output": str(args.output.resolve()),
                "cells": len(plan["cells"]),
                "schedule_entries": len(plan["schedule"]),
                "schedule_sha256": plan["schedule_sha256"],
            }
        )
    )
    return 0


def command_validate(args: argparse.Namespace) -> int:
    plan = validate_plan(load_json(args.plan.resolve()))
    receipt = {
        "ok": True,
        "schema": plan["schema"],
        "cells": len(plan["cells"]),
        "schedule_entries": len(plan["schedule"]),
        "schedule_sha256": plan["schedule_sha256"],
    }
    if args.receipt is not None:
        write_json(args.receipt.resolve(), receipt, args.force)
    print(canonical(receipt))
    return 0


def command_run(args: argparse.Namespace) -> int:
    validate_plan(load_json(args.plan.resolve()))
    raise ContractError(
        "GPU execution is disabled: the guarded multi-arm SPS runner is not implemented in this branch"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="write a deterministic CPU-only profiling plan")
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--active-slots", default="1,2,4,8")
    plan.add_argument("--context-ceilings", required=True)
    plan.add_argument("--prefix-caps", default="0,1,2,3,7")
    plan.add_argument("--max-draft-tokens-per-slot", type=int, default=7)
    plan.add_argument("--repetitions", type=int, default=4)
    plan.add_argument("--samples-per-cell", type=int, default=64)
    plan.add_argument("--warmup-shape-runs", type=int, default=3)
    plan.add_argument("--seed", type=int, default=20260818)
    plan.add_argument("--force", action="store_true")
    plan.set_defaults(handler=command_plan)

    validate = subparsers.add_parser("validate", help="validate and rederive a profiling plan")
    validate.add_argument("--plan", type=Path, required=True)
    validate.add_argument("--receipt", type=Path)
    validate.add_argument("--force", action="store_true")
    validate.set_defaults(handler=command_validate)

    run = subparsers.add_parser("run", help="reserved guarded recorder integration point")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--server", type=Path, required=True)
    run.add_argument("--model", type=Path, required=True)
    run.add_argument("--draft-model", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.set_defaults(handler=command_run)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.handler(args))
    except ContractError as error:
        print(canonical({"ok": False, "error": str(error)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
