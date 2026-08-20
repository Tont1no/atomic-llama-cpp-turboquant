#!/usr/bin/env python3
"""Build and validate deterministic DSpark SPS profiling plans.

The plan and validation commands are CPU-only. The run command is available
only inside the repository exclusive GPU guard.
"""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Iterable
import urllib.error
import urllib.parse
import urllib.request


PLAN_SCHEMA = "llama.cpp-dspark-sps-profile-plan/v1"
SCHEDULE_STRATEGY = "sha256-paired-reverse-rotate/v1"
MAX_PLAN_BYTES = 16 * 1024 * 1024
MAX_CELLS = 4096
MAX_SCHEDULE_ENTRIES = 65536
ALLOWED_ACTIVE_SLOTS = {1, 2, 4, 8}
GUARD_MARKER_ENV = "AI_LOADER_EXCLUSIVE_GPU_GUARD"
GUARD_UUID_ENV = "AI_LOADER_EXCLUSIVE_GPU_UUID"
GUARD_LEASE_PATH_ENV = "AI_LOADER_EXCLUSIVE_GPU_LEASE_PATH"
GUARD_LEASE_NONCE_ENV = "AI_LOADER_EXCLUSIVE_GPU_LEASE_NONCE"
GUARD_OWNER_PID_ENV = "AI_LOADER_EXCLUSIVE_GPU_OWNER_PID"
GUARD_JOB_NAME_ENV = "AI_LOADER_EXCLUSIVE_GPU_JOB_NAME"
MAX_HTTP_BYTES = 16 * 1024 * 1024
RUN_STATE_SCHEMA = "llama.cpp-dspark-sps-run-state/v1"
MANIFEST_SCHEMA = "llama.cpp-dspark-sps-run-manifest/v1"
ARM_SCHEMA = "llama.cpp-dspark-sps-arm/v1"
AUDIT_SCHEMA = "llama.cpp-dspark-sps-profile-audit/v1"
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


def write_text(path: Path, value: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise ContractError(f"refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", prefix=path.name + ".",
            suffix=".tmp", dir=path.parent, delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(value)
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while True:
                block = stream.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
    except OSError as error:
        raise ContractError(f"cannot hash file: {path}") from error
    return digest.hexdigest()


def require_file(path: Path, field: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ContractError(f"{field} is not a file: {resolved}")
    return resolved


def nearest_rank(values: list[float], quantile: float) -> float:
    if not values or not 0.0 < quantile <= 1.0:
        raise ContractError("nearest-rank requires samples and a quantile in (0, 1]")
    ordered = sorted(values)
    rank = max(1, math.ceil(quantile * len(ordered)))
    return ordered[rank - 1]


def subprocess_text(
    command: list[str],
    *,
    cwd: Path,
    timeout_s: float,
    label: str,
) -> str:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=timeout_s,
        )
    except (OSError, subprocess.SubprocessError, UnicodeError) as error:
        raise ContractError(f"{label} failed") from error
    if result.returncode != 0:
        raise ContractError(f"{label} exited with code {result.returncode}")
    return result.stdout.strip()


def source_commit(repo_root: Path) -> str:
    value = subprocess_text(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, timeout_s=15, label="git rev-parse"
    )
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ContractError("git rev-parse returned an invalid commit")
    return value


def gpu_identity(repo_root: Path, expected_uuid: str) -> dict[str, str]:
    output = subprocess_text(
        [
            "nvidia-smi",
            f"--id={expected_uuid}",
            "--query-gpu=uuid,name,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        cwd=repo_root,
        timeout_s=15,
        label="nvidia-smi identity query",
    )
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ContractError("nvidia-smi identity query must return exactly one GPU")
    parts = [part.strip() for part in lines[0].split(",")]
    if len(parts) != 4 or parts[0] != expected_uuid or any(not part for part in parts):
        raise ContractError("nvidia-smi identity does not match the exclusive guard")
    banner = subprocess_text(
        ["nvidia-smi"], cwd=repo_root, timeout_s=15, label="nvidia-smi runtime query"
    )
    runtime_match = re.search(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)*)", banner)
    if runtime_match is None:
        raise ContractError("nvidia-smi did not report the CUDA driver runtime")
    return {
        "uuid": parts[0],
        "name": parts[1],
        "driver_version": parts[2],
        "compute_capability": parts[3],
        "cuda_driver_runtime": runtime_match.group(1),
    }


def windows_process_start_time(pid: int) -> datetime:
    if os.name != "nt":
        raise ContractError("exclusive GPU guard requires Windows process identity")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    open_process = kernel32.OpenProcess
    open_process.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32)
    open_process.restype = ctypes.c_void_p
    get_process_times = kernel32.GetProcessTimes
    get_process_times.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
    )
    get_process_times.restype = ctypes.c_int
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (ctypes.c_void_p,)
    close_handle.restype = ctypes.c_int
    handle = open_process(0x1000, 0, pid)
    if not handle:
        raise ContractError("exclusive GPU lease owner process is unavailable")
    creation = ctypes.c_uint64()
    exit_time = ctypes.c_uint64()
    kernel = ctypes.c_uint64()
    user = ctypes.c_uint64()
    try:
        if not get_process_times(
            handle,
            ctypes.byref(creation),
            ctypes.byref(exit_time),
            ctypes.byref(kernel),
            ctypes.byref(user),
        ):
            raise ContractError("exclusive GPU lease owner lifetime is unavailable")
    finally:
        close_handle(handle)
    unix_100ns = creation.value - 116444736000000000
    return datetime.fromtimestamp(unix_100ns / 10_000_000, timezone.utc)


def verify_windows_job_membership(job_name: str) -> None:
    if os.name != "nt":
        raise ContractError("exclusive GPU guard requires Windows Job Object membership")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    open_job = kernel32.OpenJobObjectW
    open_job.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_wchar_p)
    open_job.restype = ctypes.c_void_p
    current_process = kernel32.GetCurrentProcess
    current_process.argtypes = ()
    current_process.restype = ctypes.c_void_p
    is_process_in_job = kernel32.IsProcessInJob
    is_process_in_job.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int))
    is_process_in_job.restype = ctypes.c_int
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (ctypes.c_void_p,)
    close_handle.restype = ctypes.c_int
    handle = open_job(0x0004, 0, job_name)
    if not handle:
        raise ContractError("exclusive GPU Job Object is not active")
    member = ctypes.c_int()
    try:
        if not is_process_in_job(current_process(), handle, ctypes.byref(member)):
            raise ContractError("exclusive GPU Job Object membership query failed")
    finally:
        close_handle(handle)
    if not member.value:
        raise ContractError("runner is not a member of the exclusive GPU Job Object")


def verify_windows_write_lock(path: Path) -> None:
    if os.name != "nt":
        raise ContractError("exclusive GPU lease lock requires Windows")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    )
    create_file.restype = ctypes.c_void_p
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (ctypes.c_void_p,)
    close_handle.restype = ctypes.c_int
    handle = create_file(str(path), 0x40000000, 0x00000001, None, 3, 0x00000080, None)
    invalid_handle = ctypes.c_void_p(-1).value
    if handle != invalid_handle:
        close_handle(handle)
        raise ContractError("exclusive GPU lease is not actively write-locked")
    if ctypes.get_last_error() not in {32, 33}:
        raise ContractError("exclusive GPU lease is not actively write-locked")


def verify_guard_lease(repo_root: Path) -> str:
    expected_path = (repo_root / ".codex-deploy" / "gpu-exclusive.lock").resolve()
    raw_path = os.environ.get(GUARD_LEASE_PATH_ENV, "")
    nonce = os.environ.get(GUARD_LEASE_NONCE_ENV, "")
    owner_raw = os.environ.get(GUARD_OWNER_PID_ENV, "")
    uuid = os.environ.get(GUARD_UUID_ENV, "")
    job_name = os.environ.get(GUARD_JOB_NAME_ENV, "")
    if (
        os.environ.get(GUARD_MARKER_ENV) != "1"
        or not raw_path
        or len(nonce) != 64
        or any(character not in "0123456789abcdef" for character in nonce)
        or not owner_raw.isdecimal()
        or not uuid.startswith("GPU-")
        or os.environ.get("CUDA_VISIBLE_DEVICES") != uuid
        or job_name != f"Local\\AiLoaderGpuGuard-{nonce}"
    ):
        raise ContractError("run requires a complete exclusive GPU guard proof")
    try:
        supplied_path = Path(raw_path).resolve(strict=True)
    except OSError as error:
        raise ContractError("exclusive GPU lease path is unavailable") from error
    if supplied_path != expected_path:
        raise ContractError("exclusive GPU lease is outside the current workspace")
    owner_pid = int(owner_raw, 10)
    if owner_pid != os.getppid():
        raise ContractError("exclusive GPU lease owner is not the runner parent")
    lease = require_exact_keys(
        load_json(expected_path),
        {
            "LeaseVersion", "Nonce", "OwnerPid", "CreatedAt", "OwnerStartTimeUtc",
            "JobName", "Executable", "GpuIndex", "GpuUuid",
        },
        "exclusive GPU lease",
    )
    if (
        lease["LeaseVersion"] != 2
        or lease["Nonce"] != nonce
        or lease["OwnerPid"] != owner_pid
        or lease["GpuUuid"] != uuid
        or lease["JobName"] != job_name
    ):
        raise ContractError("exclusive GPU lease content does not match the child proof")
    # The guard keeps the lease open with FileShare.Read. A copied/forged stale
    # JSON file is readable but remains writable and must not authorize a run.
    verify_windows_write_lock(expected_path)
    try:
        recorded_owner_start = datetime.fromisoformat(str(lease["OwnerStartTimeUtc"]).replace("Z", "+00:00"))
    except ValueError as error:
        raise ContractError("exclusive GPU lease owner lifetime is invalid") from error
    if recorded_owner_start.tzinfo is None:
        raise ContractError("exclusive GPU lease owner lifetime is invalid")
    actual_owner_start = windows_process_start_time(owner_pid)
    if abs((actual_owner_start - recorded_owner_start.astimezone(timezone.utc)).total_seconds()) > 0.01:
        raise ContractError("exclusive GPU lease owner lifetime does not match")
    verify_windows_job_membership(job_name)
    return uuid


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        del req, fp, code, msg, headers, newurl
        return None


class AbortableHttpConnections:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._connections: set[http.client.HTTPConnection] = set()
        self._closed = False

    def close_all(self) -> None:
        with self._lock:
            self._closed = True
            connections = list(self._connections)
        for connection in connections:
            connection.close()

    def post_json(self, base_url: str, path: str, timeout_s: float, body: dict[str, Any]) -> dict[str, Any]:
        parsed = urllib.parse.urlsplit(base_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"} or parsed.path not in {"", "/"}:
            raise ContractError("abortable HTTP endpoint must be a local HTTP origin")
        connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=timeout_s)
        try:
            connection.connect()
            with self._lock:
                if self._closed:
                    raise ContractError("completion endpoint was closed")
                self._connections.add(connection)
            connection.request(
                "POST",
                path,
                body=canonical(body).encode("utf-8"),
                headers={"Content-Type": "application/json", "Accept": "application/json"},
            )
            response = connection.getresponse()
            if response.status != 200:
                raise ContractError(f"{path} returned HTTP {response.status}")
            payload = response.read(MAX_HTTP_BYTES + 1)
        except ContractError:
            raise
        except (OSError, TimeoutError, http.client.HTTPException) as error:
            raise ContractError(f"HTTP request failed: {path}") from error
        finally:
            with self._lock:
                self._connections.discard(connection)
            connection.close()
        if len(payload) > MAX_HTTP_BYTES:
            raise ContractError(f"HTTP response exceeded {MAX_HTTP_BYTES} bytes: {path}")
        try:
            value = json.loads(payload.decode("utf-8", errors="strict"), object_pairs_hook=reject_duplicate_keys)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ContractError(f"HTTP response is not strict UTF-8 JSON: {path}") from error
        if not isinstance(value, dict):
            raise ContractError(f"HTTP response must be an object: {path}")
        return value


def http_json(
    base_url: str,
    path: str,
    timeout_s: float,
    body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = canonical(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(
        base_url + path,
        data=data,
        method="POST" if body is not None else "GET",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.build_opener(NoRedirect()).open(request, timeout=timeout_s) as response:
            if int(response.status) != 200:
                raise ContractError(f"{path} returned HTTP {response.status}")
            payload = response.read(MAX_HTTP_BYTES + 1)
    except ContractError:
        raise
    except (OSError, TimeoutError, urllib.error.URLError) as error:
        raise ContractError(f"HTTP request failed: {path}") from error
    if len(payload) > MAX_HTTP_BYTES:
        raise ContractError(f"HTTP response exceeded {MAX_HTTP_BYTES} bytes: {path}")
    try:
        value = json.loads(payload.decode("utf-8", errors="strict"), object_pairs_hook=reject_duplicate_keys)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ContractError(f"HTTP response is not strict UTF-8 JSON: {path}") from error
    if not isinstance(value, dict):
        raise ContractError(f"HTTP response must be an object: {path}")
    return value


def http_text(base_url: str, path: str, timeout_s: float) -> str:
    request = urllib.request.Request(base_url + path, method="GET")
    try:
        with urllib.request.build_opener(NoRedirect()).open(request, timeout=timeout_s) as response:
            if int(response.status) != 200:
                raise ContractError(f"{path} returned HTTP {response.status}")
            payload = response.read(MAX_HTTP_BYTES + 1)
    except ContractError:
        raise
    except (OSError, TimeoutError, urllib.error.URLError) as error:
        raise ContractError(f"HTTP request failed: {path}") from error
    if len(payload) > MAX_HTTP_BYTES:
        raise ContractError(f"HTTP response exceeded {MAX_HTTP_BYTES} bytes: {path}")
    try:
        return payload.decode("utf-8", errors="strict")
    except UnicodeError as error:
        raise ContractError(f"HTTP response is not UTF-8: {path}") from error


def parse_metrics(text: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        try:
            key, raw = line.rsplit(" ", 1)
            value = float(raw)
        except ValueError:
            continue
        if not math.isfinite(value):
            raise ContractError(f"metric is non-finite: {key}")
        result[key] = value
    return result


def port_is_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.25):
            return True
    except OSError:
        return False


def remaining(deadline: float, label: str) -> float:
    value = deadline - time.monotonic()
    if value <= 0:
        raise ContractError(f"{label} timeout expired")
    return value


def start_server(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
) -> tuple[subprocess.Popen[Any], Any, Any]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_stream = stdout_path.open("a", encoding="utf-8", newline="\n")
    stderr_stream = stderr_path.open("a", encoding="utf-8", newline="\n")
    kwargs: dict[str, Any] = {
        "cwd": cwd,
        "env": environment,
        "stdin": subprocess.DEVNULL,
        "stdout": stdout_stream,
        "stderr": stderr_stream,
    }
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    try:
        process = subprocess.Popen(command, **kwargs)
    except OSError:
        stdout_stream.close()
        stderr_stream.close()
        raise
    return process, stdout_stream, stderr_stream


def stop_process_tree(process: subprocess.Popen[Any], timeout_s: float) -> None:
    if process.poll() is not None:
        return
    deadline = time.monotonic() + timeout_s
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                capture_output=True,
                timeout=remaining(deadline, "server process cleanup"),
            )
        except (OSError, subprocess.SubprocessError):
            process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=remaining(deadline, "server process cleanup"))
            return
        except (OSError, subprocess.SubprocessError):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                process.kill()
    try:
        process.wait(timeout=remaining(deadline, "server process cleanup"))
    except subprocess.TimeoutExpired as error:
        raise ContractError(f"server process tree did not exit: pid={process.pid}") from error


def wait_ready(
    process: subprocess.Popen[Any],
    base_url: str,
    deadline: float,
    http_timeout_s: float,
) -> dict[str, Any]:
    last_error = "not attempted"
    while True:
        if process.poll() is not None:
            raise ContractError(f"llama-server exited during startup with code {process.returncode}")
        try:
            health = http_json(base_url, "/health", min(http_timeout_s, remaining(deadline, "readiness")))
            if health.get("status") == "ok":
                return health
            last_error = f"unexpected health: {health}"
        except ContractError as error:
            last_error = str(error)
        if remaining(deadline, "readiness") <= 0.25:
            raise ContractError(f"server did not become ready: {last_error}")
        time.sleep(0.25)


def validate_props(props: dict[str, Any], parallel: int, minimum_context: int) -> None:
    total_slots = props.get("total_slots")
    settings = props.get("default_generation_settings")
    if isinstance(total_slots, bool) or total_slots != parallel:
        raise ContractError("/props total_slots does not match --parallel")
    if props.get("endpoint_metrics") is not True:
        raise ContractError("/props does not prove that metrics are enabled")
    if not isinstance(settings, dict):
        raise ContractError("/props default_generation_settings is missing")
    n_ctx = settings.get("n_ctx")
    if isinstance(n_ctx, bool) or not isinstance(n_ctx, int) or n_ctx < minimum_context:
        raise ContractError("/props per-slot context is smaller than the profiling cell")


def validate_sidecar(
    value: Any,
    *,
    source_identity_value: str,
    cell: dict[str, Any],
    max_draft_tokens_per_slot: int,
    samples_per_cell: int,
    warmup_shape_runs: int,
    require_ready: bool,
) -> dict[str, Any]:
    expected_root = {
        "schema_version", "profile_schema_version", "source_identity", "complete",
        "max_draft_tokens_per_slot", "context_buckets", "verify_rows_by_active",
        "retained_samples_per_coordinate", "warmup_samples_per_coordinate",
        "aggregate_quantile", "observations_total", "retained_total",
        "skipped_warmup_total", "skipped_capture_total", "skipped_ineligible_total",
        "skipped_outside_grid_total", "skipped_full_total", "ready_coordinates",
        "expected_coordinates", "coordinates",
    }
    root = require_exact_keys(value, expected_root, "sidecar")
    if root["schema_version"] != 1 or root["profile_schema_version"] != 2:
        raise ContractError("sidecar schema version is invalid")
    if root["source_identity"] != source_identity_value:
        raise ContractError("sidecar source identity mismatch")
    if root["max_draft_tokens_per_slot"] != max_draft_tokens_per_slot:
        raise ContractError("sidecar draft depth mismatch")
    if root["context_buckets"] != [cell["context_ceiling"]]:
        raise ContractError("sidecar context bucket mismatch")
    expected_rows = {str(cell["active_slots"]): [cell["total_verify_rows"]]}
    if root["verify_rows_by_active"] != expected_rows:
        raise ContractError("sidecar row axis mismatch")
    if root["retained_samples_per_coordinate"] != samples_per_cell:
        raise ContractError("sidecar sample quota mismatch")
    if root["warmup_samples_per_coordinate"] != warmup_shape_runs:
        raise ContractError("sidecar warmup quota mismatch")
    if root["aggregate_quantile"] != 0.95:
        raise ContractError("sidecar aggregation quantile mismatch")
    if root["expected_coordinates"] != 1 or not isinstance(root["coordinates"], list) or len(root["coordinates"]) != 1:
        raise ContractError("sidecar must contain exactly one coordinate")
    coordinate = root["coordinates"][0]
    expected_coordinate_keys = {
        "context_tokens", "active_slots", "total_verify_rows", "ready",
        "retained_samples", "warmup_seen", "skipped_warmup", "skipped_capture",
        "skipped_ineligible", "skipped_full", "p50_us", "p95_us", "max_us", "samples",
    }
    coordinate = require_exact_keys(coordinate, expected_coordinate_keys, "sidecar.coordinates[0]")
    if (
        coordinate["context_tokens"] != cell["context_ceiling"]
        or coordinate["active_slots"] != cell["active_slots"]
        or coordinate["total_verify_rows"] != cell["total_verify_rows"]
    ):
        raise ContractError("sidecar coordinate mismatch")
    samples = coordinate["samples"]
    if not isinstance(samples, list) or coordinate["retained_samples"] != len(samples):
        raise ContractError("sidecar retained sample count mismatch")
    if len(samples) > samples_per_cell:
        raise ContractError("sidecar contains more samples than configured")
    counter_names = (
        "observations_total", "retained_total", "skipped_warmup_total",
        "skipped_capture_total", "skipped_ineligible_total",
        "skipped_outside_grid_total", "skipped_full_total", "ready_coordinates",
    )
    if any(
        isinstance(root[name], bool) or not isinstance(root[name], int) or root[name] < 0
        for name in counter_names
    ):
        raise ContractError("sidecar contains an invalid root counter")
    coordinate_counters = (
        "warmup_seen", "skipped_warmup", "skipped_capture", "skipped_ineligible", "skipped_full",
    )
    if any(
        isinstance(coordinate[name], bool)
        or not isinstance(coordinate[name], int)
        or coordinate[name] < 0
        for name in coordinate_counters
    ):
        raise ContractError("sidecar contains an invalid coordinate counter")
    for index, raw_sample in enumerate(samples):
        sample = require_exact_keys(
            raw_sample,
            {"actual_context_tokens", "cost_us", "execution_kind", "prefixes"},
            f"sidecar.samples[{index}]",
        )
        actual_context = sample["actual_context_tokens"]
        cost = sample["cost_us"]
        prefixes = sample["prefixes"]
        if (
            isinstance(actual_context, bool)
            or not isinstance(actual_context, int)
            or actual_context <= 0
            or actual_context > cell["context_ceiling"]
            or isinstance(cost, bool)
            or not isinstance(cost, (int, float))
            or not math.isfinite(float(cost))
            or float(cost) <= 0.0
            or sample["execution_kind"] not in {"graph_replay", "graphs_disabled"}
            or not isinstance(prefixes, list)
            or len(prefixes) != cell["active_slots"]
            or any(
                isinstance(prefix, bool)
                or not isinstance(prefix, int)
                or prefix < 0
                or prefix > max_draft_tokens_per_slot
                for prefix in prefixes
            )
            or sum(prefixes) + cell["active_slots"] != cell["total_verify_rows"]
        ):
            raise ContractError(f"sidecar sample {index} violates the coordinate contract")
    ready = len(samples) == samples_per_cell
    costs = [float(sample["cost_us"]) for sample in samples]
    expected_statistics: tuple[float | None, float | None, float | None]
    if costs:
        expected_statistics = (
            nearest_rank(costs, 0.50), nearest_rank(costs, 0.95), nearest_rank(costs, 1.00)
        )
    else:
        expected_statistics = (None, None, None)
    if tuple(coordinate[name] for name in ("p50_us", "p95_us", "max_us")) != expected_statistics:
        raise ContractError("sidecar statistics do not match raw samples")
    if (
        root["retained_total"] != len(samples)
        or root["ready_coordinates"] != int(ready)
        or root["skipped_warmup_total"] != coordinate["skipped_warmup"]
        or root["skipped_capture_total"] != coordinate["skipped_capture"]
        or root["skipped_ineligible_total"] != coordinate["skipped_ineligible"]
        or root["skipped_full_total"] != coordinate["skipped_full"]
        or root["observations_total"] != sum(
            root[name] for name in (
                "retained_total", "skipped_warmup_total", "skipped_capture_total",
                "skipped_ineligible_total", "skipped_outside_grid_total", "skipped_full_total",
            )
        )
        or coordinate["warmup_seen"] > warmup_shape_runs
        or (samples and coordinate["warmup_seen"] != warmup_shape_runs)
    ):
        raise ContractError("sidecar counters are inconsistent")
    if coordinate["ready"] is not ready or root["complete"] is not ready:
        raise ContractError("sidecar completion flags are inconsistent")
    if require_ready and not ready:
        raise ContractError("sidecar coordinate is incomplete")
    return root


def validate_arm_profile(value: Any, cell: dict[str, Any], max_draft: int) -> dict[str, Any]:
    root = require_exact_keys(value, {"schema_version", "max_draft_tokens_per_slot", "entries"}, "arm profile")
    if root["schema_version"] != 2 or root["max_draft_tokens_per_slot"] != max_draft:
        raise ContractError("arm profile schema or depth mismatch")
    if not isinstance(root["entries"], list) or len(root["entries"]) != 1:
        raise ContractError("arm profile must contain exactly one entry")
    entry = require_exact_keys(
        root["entries"][0], {"context_tokens", "active_slots", "total_verify_rows", "cost_us"}, "arm profile entry"
    )
    if (
        entry["context_tokens"] != cell["context_ceiling"]
        or entry["active_slots"] != cell["active_slots"]
        or entry["total_verify_rows"] != cell["total_verify_rows"]
        or isinstance(entry["cost_us"], bool)
        or not isinstance(entry["cost_us"], (int, float))
        or not math.isfinite(float(entry["cost_us"]))
        or float(entry["cost_us"]) <= 0.0
    ):
        raise ContractError("arm profile entry mismatch")
    return root


def validate_final_profile(value: Any, plan: dict[str, Any]) -> dict[str, Any]:
    root = require_exact_keys(
        value,
        {"schema_version", "max_draft_tokens_per_slot", "entries"},
        "final profile",
    )
    if (
        root["schema_version"] != 2
        or root["max_draft_tokens_per_slot"] != plan["max_draft_tokens_per_slot"]
        or not isinstance(root["entries"], list)
        or len(root["entries"]) != len(plan["cells"])
    ):
        raise ContractError("final profile header or entry count mismatch")
    expected_coordinates = {
        (cell["context_ceiling"], cell["active_slots"], cell["total_verify_rows"])
        for cell in plan["cells"]
    }
    observed: dict[tuple[int, int, int], float] = {}
    for index, raw_entry in enumerate(root["entries"]):
        entry = require_exact_keys(
            raw_entry,
            {"context_tokens", "active_slots", "total_verify_rows", "cost_us"},
            f"final profile entries[{index}]",
        )
        coordinate = (
            entry["context_tokens"], entry["active_slots"], entry["total_verify_rows"]
        )
        cost = entry["cost_us"]
        if (
            coordinate not in expected_coordinates
            or coordinate in observed
            or isinstance(cost, bool)
            or not isinstance(cost, (int, float))
            or not math.isfinite(float(cost))
            or float(cost) <= 0.0
        ):
            raise ContractError(f"final profile entry {index} is invalid")
        observed[coordinate] = float(cost)
    for left, left_cost in observed.items():
        for right, right_cost in observed.items():
            if all(a <= b for a, b in zip(left, right)) and left_cost > right_cost:
                raise ContractError("final profile is not monotonic")
    return root


def request_seed(source_identity_value: str, ordinal: int, client_index: int) -> int:
    digest = hashlib.sha256(
        f"{source_identity_value}:{ordinal}:{client_index}".encode("ascii")
    ).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF or 1


def completion_request(
    *,
    base_url: str,
    model_alias: str,
    prompt_token_id: int,
    prompt_tokens: int,
    output_tokens: int,
    seed: int,
    barrier: threading.Barrier,
    timeout_s: float,
    connections: AbortableHttpConnections,
) -> dict[str, Any]:
    barrier.wait(timeout=timeout_s)
    started = time.monotonic()
    response = connections.post_json(
        base_url,
        "/completion",
        timeout_s,
        {
            "model": model_alias,
            "prompt": [prompt_token_id] * prompt_tokens,
            "n_predict": output_tokens,
            "temperature": 0,
            "seed": seed,
            "ignore_eos": True,
            "cache_prompt": False,
            "stream": False,
        },
    )
    timings = response.get("timings")
    if (
        response.get("stop_type") != "limit"
        or response.get("truncated") is not False
        or not isinstance(timings, dict)
        or timings.get("predicted_n") != output_tokens
    ):
        raise ContractError("completion did not satisfy the exact output contract")
    return {
        "seed": seed,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": output_tokens,
        "wall_ms": (time.monotonic() - started) * 1000.0,
        "predicted_ms": timings.get("predicted_ms"),
        "predicted_per_second": timings.get("predicted_per_second"),
        "content_sha256": hashlib.sha256(str(response.get("content", "")).encode("utf-8")).hexdigest(),
    }


def server_arguments(
    args: argparse.Namespace,
    cell: dict[str, Any],
    record_path: Path,
    source_identity_value: str,
) -> list[str]:
    values = [
        str(args.server.resolve()), "--model", str(args.model.resolve()),
        "--host", "127.0.0.1", "--port", str(args.port), "--alias", args.model_alias,
        "--ctx-size", str(args.ctx_size), "--parallel", str(args.parallel),
        "--cache-type-k", args.target_kv, "--cache-type-v", args.target_kv,
        "--flash-attn", "on", "--batch-size", str(args.batch_size),
        "--ubatch-size", str(args.ubatch_size), "--n-gpu-layers", "all",
        "--device", "CUDA0", "--split-mode", "none", "--fit", "off",
        "--cache-ram", "0", "--ctx-checkpoints", "0", "--no-cache-idle-slots",
        "--no-cache-prompt", "--no-webui", "--metrics", "--reasoning", "off",
        "-lv", str(args.log_verbosity),
        "--spec-type", "draft-dspark", "--spec-draft-model", str(args.draft_model.resolve()),
        "--spec-draft-n-max", str(args.max_draft_tokens_per_slot),
        "--spec-draft-n-min", "0", "--spec-draft-p-min", "0",
        "--spec-draft-type-k", args.draft_kv, "--spec-draft-type-v", args.draft_kv,
        "--spec-draft-ngl", "all", "--spec-draft-device", "CUDA0",
        "--spec-draft-sps-record", str(record_path),
        "--spec-draft-sps-record-identity", source_identity_value,
        "--spec-draft-sps-record-context-buckets", str(cell["context_ceiling"]),
        "--spec-draft-sps-record-active", str(cell["active_slots"]),
        "--spec-draft-sps-record-caps", str(cell["prefix_cap"]),
        "--spec-draft-sps-force-verify-rows", str(cell["total_verify_rows"]),
        "--spec-draft-sps-record-samples", str(args.samples_per_cell),
        "--spec-draft-sps-record-warmup", str(args.warmup_shape_runs),
    ]
    values.append("--no-kv-unified" if args.disable_unified_kv else "--kv-unified")
    return values


def immutable_runtime_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "runtime_label": args.runtime_label,
        "host": "127.0.0.1",
        "port": args.port,
        "model_alias": args.model_alias,
        "ctx_size": args.ctx_size,
        "parallel": args.parallel,
        "batch_size": args.batch_size,
        "ubatch_size": args.ubatch_size,
        "target_kv": args.target_kv,
        "draft_kv": args.draft_kv,
        "disable_unified_kv": args.disable_unified_kv,
        "disable_cuda_graphs": args.disable_cuda_graphs,
        "max_draft_tokens_per_slot": args.max_draft_tokens_per_slot,
        "prompt_token_id": args.prompt_token_id,
        "output_tokens": args.output_tokens,
        "prompt_safety_tokens": args.prompt_safety_tokens,
        "log_verbosity": args.log_verbosity,
    }


def runtime_bundle_identity(server: Path) -> list[dict[str, str]]:
    server = server.resolve()
    libraries = sorted(
        (
            path for path in server.parent.iterdir()
            if path.is_file()
            and path.suffix.lower() == ".dll"
        ),
        key=lambda path: path.name.lower(),
    )
    names = [path.name.lower() for path in libraries]
    if len(names) != len(set(names)):
        raise ContractError("runtime bundle contains case-colliding DLLs")
    return [{"name": path.name, "sha256": sha256_file(path)} for path in libraries]


def build_identity(
    args: argparse.Namespace,
    plan_path: Path,
    repo_root: Path,
    gpu: dict[str, str],
) -> tuple[str, dict[str, Any]]:
    components = {
        "schema": "llama.cpp-dspark-sps-source-identity/v1",
        "source_commit": source_commit(repo_root),
        "driver_sha256": sha256_file(Path(__file__).resolve()),
        "wrapper_sha256": sha256_file(Path(__file__).with_name("Invoke-DsparkSpsProfile.ps1")),
        "guard_sha256": sha256_file(Path(__file__).with_name("Invoke-ExclusiveGpuTask.ps1")),
        "plan_sha256": sha256_file(plan_path),
        "server_sha256": sha256_file(args.server.resolve()),
        "runtime_bundle": runtime_bundle_identity(args.server),
        "target_model_sha256": sha256_file(args.model.resolve()),
        "draft_model_sha256": sha256_file(args.draft_model.resolve()),
        "gpu": gpu,
        "runtime": immutable_runtime_config(args),
        "python": sys.version.split()[0],
    }
    return sha256_json(components), components


def make_manifest(
    source_identity_value: str,
    components: dict[str, Any],
    plan: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": MANIFEST_SCHEMA,
        "source_identity": source_identity_value,
        "identity_components": components,
        "plan_schema": plan["schema"],
        "schedule_sha256": plan["schedule_sha256"],
        "schedule_entries": len(plan["schedule"]),
        "created_at": utc_now(),
    }


def validate_existing_manifest(value: Any, expected: dict[str, Any]) -> dict[str, Any]:
    root = require_exact_keys(
        value,
        {"schema", "source_identity", "identity_components", "plan_schema", "schedule_sha256", "schedule_entries", "created_at"},
        "manifest",
    )
    for key in ("schema", "source_identity", "identity_components", "plan_schema", "schedule_sha256", "schedule_entries"):
        if root[key] != expected[key]:
            raise ContractError(f"existing manifest is incompatible: {key}")
    if not isinstance(root["created_at"], str) or not root["created_at"]:
        raise ContractError("existing manifest has an invalid created_at")
    return root


def new_state(source_identity_value: str, plan: dict[str, Any]) -> dict[str, Any]:
    now = utc_now()
    return {
        "schema": RUN_STATE_SCHEMA,
        "source_identity": source_identity_value,
        "schedule_sha256": plan["schedule_sha256"],
        "status": "running",
        "created_at": now,
        "updated_at": now,
        "next_ordinal": 0,
        "completed_ordinals": [],
        "current_ordinal": None,
        "error": None,
    }


def validate_state(value: Any, source_identity_value: str, plan: dict[str, Any]) -> dict[str, Any]:
    root = require_exact_keys(
        value,
        {"schema", "source_identity", "schedule_sha256", "status", "created_at", "updated_at", "next_ordinal", "completed_ordinals", "current_ordinal", "error"},
        "run state",
    )
    if root["schema"] != RUN_STATE_SCHEMA or root["source_identity"] != source_identity_value or root["schedule_sha256"] != plan["schedule_sha256"]:
        raise ContractError("run state is incompatible with plan or runtime identity")
    if root["status"] not in {"running", "failed", "completed"}:
        raise ContractError("run state status is invalid")
    if any(not isinstance(root[key], str) or not root[key] for key in ("created_at", "updated_at")):
        raise ContractError("run state timestamps are invalid")
    if root["current_ordinal"] is not None and (
        isinstance(root["current_ordinal"], bool)
        or not isinstance(root["current_ordinal"], int)
        or root["current_ordinal"] < 0
        or root["current_ordinal"] >= len(plan["schedule"])
    ):
        raise ContractError("run state current_ordinal is invalid")
    if root["error"] is not None and not isinstance(root["error"], str):
        raise ContractError("run state error is invalid")
    completed = root["completed_ordinals"]
    if not isinstance(completed, list) or any(isinstance(item, bool) or not isinstance(item, int) for item in completed):
        raise ContractError("run state completed_ordinals is invalid")
    if completed != sorted(set(completed)) or any(item < 0 or item >= len(plan["schedule"]) for item in completed):
        raise ContractError("run state completed_ordinals is out of range")
    if root["next_ordinal"] != len(completed):
        raise ContractError("run state next_ordinal is inconsistent")
    if completed != list(range(len(completed))):
        raise ContractError("run state may only contain a completed ordinal prefix")
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


def arm_paths(output_dir: Path, ordinal: int) -> dict[str, Path]:
    root = output_dir / "runs" / f"{ordinal:05d}"
    return {
        "root": root,
        "command": root / "command.json",
        "stdout": root / "server.stdout.log",
        "stderr": root / "server.stderr.log",
        "client": root / "client.json",
        "props": root / "props.json",
        "metrics_before": root / "metrics-before.txt",
        "metrics_after": root / "metrics-after.txt",
        "record": root / "record.json",
        "sidecar": root / "record.samples.json",
        "receipt": root / "receipt.json",
    }


def validate_completion_artifact(value: Any, entry: dict[str, Any], cell: dict[str, Any], output_tokens: int) -> dict[str, Any]:
    root = require_exact_keys(
        value,
        {"schema", "ordinal", "cell_id", "active_slots", "prompt_token_id", "prompt_tokens", "output_tokens", "common_wall_ms", "requests"},
        "client artifact",
    )
    if (
        root["schema"] != ARM_SCHEMA
        or root["ordinal"] != entry["ordinal"]
        or root["cell_id"] != cell["cell_id"]
        or root["active_slots"] != cell["active_slots"]
        or root["output_tokens"] != output_tokens
        or not isinstance(root["requests"], list)
        or len(root["requests"]) != cell["active_slots"]
        or isinstance(root["common_wall_ms"], bool)
        or not isinstance(root["common_wall_ms"], (int, float))
        or root["common_wall_ms"] <= 0
    ):
        raise ContractError("client artifact does not match the arm")
    for request in root["requests"]:
        if request.get("generated_tokens") != output_tokens or request.get("prompt_tokens") != root["prompt_tokens"]:
            raise ContractError("client request did not meet its token contract")
    return root


def validate_completed_arm(
    paths: dict[str, Path],
    *,
    entry: dict[str, Any],
    cell: dict[str, Any],
    source_identity_value: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    command = {
        "schema": ARM_SCHEMA,
        "source_identity": source_identity_value,
        "ordinal": entry["ordinal"],
        "cell": cell,
        "arguments": server_arguments(args, cell, paths["record"], source_identity_value),
    }
    if load_json(paths["command"]) != command:
        raise ContractError(f"completed arm command mismatch: ordinal={entry['ordinal']}")
    props = load_json(paths["props"])
    validate_props(props, args.parallel, cell["context_ceiling"])
    metrics_before = parse_metrics(paths["metrics_before"].read_text(encoding="utf-8"))
    metrics_after = parse_metrics(paths["metrics_after"].read_text(encoding="utf-8"))
    retained_metric = "llamacpp:spec_decode_sps_record_retained_samples_total"
    if (
        retained_metric not in metrics_before
        or retained_metric not in metrics_after
        or metrics_after[retained_metric] < metrics_before[retained_metric]
    ):
        raise ContractError("completed arm recorder metrics are invalid")
    sidecar = validate_sidecar(
        load_json(paths["sidecar"]),
        source_identity_value=source_identity_value,
        cell=cell,
        max_draft_tokens_per_slot=args.max_draft_tokens_per_slot,
        samples_per_cell=args.samples_per_cell,
        warmup_shape_runs=args.warmup_shape_runs,
        require_ready=True,
    )
    arm_profile = validate_arm_profile(
        load_json(paths["record"]), cell, args.max_draft_tokens_per_slot
    )
    if float(arm_profile["entries"][0]["cost_us"]) != float(sidecar["coordinates"][0]["p95_us"]):
        raise ContractError("arm profile cost does not match the raw sidecar p95")
    client = validate_completion_artifact(load_json(paths["client"]), entry, cell, args.output_tokens)
    receipt = require_exact_keys(
        load_json(paths["receipt"]),
        {"schema", "source_identity", "ordinal", "cell_id", "status", "command_sha256", "props_sha256", "metrics_before_sha256", "metrics_after_sha256", "record_sha256", "sidecar_sha256", "client_sha256", "completed_at"},
        "arm receipt",
    )
    if (
        receipt["schema"] != ARM_SCHEMA
        or receipt["source_identity"] != source_identity_value
        or receipt["ordinal"] != entry["ordinal"]
        or receipt["cell_id"] != cell["cell_id"]
        or receipt["status"] != "completed"
        or receipt["command_sha256"] != sha256_file(paths["command"])
        or receipt["props_sha256"] != sha256_file(paths["props"])
        or receipt["metrics_before_sha256"] != sha256_file(paths["metrics_before"])
        or receipt["metrics_after_sha256"] != sha256_file(paths["metrics_after"])
        or receipt["record_sha256"] != sha256_file(paths["record"])
        or receipt["sidecar_sha256"] != sha256_file(paths["sidecar"])
        or receipt["client_sha256"] != sha256_file(paths["client"])
    ):
        raise ContractError(f"completed arm receipt mismatch: ordinal={entry['ordinal']}")
    return {"sidecar": sidecar, "client": client, "receipt": receipt}


def run_wave(
    *,
    base_url: str,
    source_identity_value: str,
    entry: dict[str, Any],
    cell: dict[str, Any],
    args: argparse.Namespace,
    prompt_tokens: int,
    sidecar_path: Path,
    arm_deadline: float,
    abort_server: Any,
    guard_check: Any = None,
) -> dict[str, Any]:
    active = cell["active_slots"]
    barrier = threading.Barrier(active + 1)
    common_started = time.monotonic()
    connections = AbortableHttpConnections()
    results: list[dict[str, Any] | None] = [None] * active
    worker_errors: list[Exception | None] = [None] * active
    workers: list[threading.Thread] = []
    abort_lock = threading.Lock()
    abort_state: dict[str, Any] = {"called": False, "error": None, "deadline": None}
    failed = True

    request_timeout_s = min(
        args.http_timeout_s,
        remaining(arm_deadline, "arm"),
        args.progress_timeout_s + args.poll_interval_s,
    )

    def abort_endpoint() -> None:
        with abort_lock:
            if abort_state["called"]:
                return
            abort_state["called"] = True
            abort_state["deadline"] = time.monotonic() + args.stop_timeout_s
        connections.close_all()
        try:
            abort_server()
        except Exception as error:
            abort_state["error"] = error

    def request_worker(client_index: int) -> None:
        try:
            results[client_index] = completion_request(
                base_url=base_url,
                model_alias=args.model_alias,
                prompt_token_id=args.prompt_token_id,
                prompt_tokens=prompt_tokens,
                output_tokens=args.output_tokens,
                seed=request_seed(source_identity_value, entry["ordinal"], client_index),
                barrier=barrier,
                timeout_s=request_timeout_s,
                connections=connections,
            )
        except Exception as error:
            worker_errors[client_index] = error
            abort_endpoint()

    try:
        for client_index in range(active):
            worker = threading.Thread(
                target=request_worker,
                args=(client_index,),
                name=f"sps-client-{client_index}",
                daemon=True,
            )
            workers.append(worker)
            worker.start()
        barrier.wait(timeout=request_timeout_s)
        last_retained = -1
        last_progress = time.monotonic()
        last_metrics_poll = 0.0
        retained_metric = "llamacpp:spec_decode_sps_record_retained_samples_total"
        last_metric_value = -1.0
        recorder_ready = False
        while any(worker.is_alive() for worker in workers):
            remaining(arm_deadline, "arm")
            if guard_check is not None:
                guard_check()
            for error in worker_errors:
                if error is not None:
                    raise ContractError(f"completion request failed: {error}") from error
            if sidecar_path.is_file():
                sidecar = validate_sidecar(
                    load_json(sidecar_path),
                    source_identity_value=source_identity_value,
                    cell=cell,
                    max_draft_tokens_per_slot=args.max_draft_tokens_per_slot,
                    samples_per_cell=args.samples_per_cell,
                    warmup_shape_runs=args.warmup_shape_runs,
                    require_ready=False,
                )
                retained = sidecar["retained_total"]
                recorder_ready = bool(sidecar["complete"])
                if retained != last_retained:
                    last_retained = retained
                    last_progress = time.monotonic()
            now = time.monotonic()
            if now - last_metrics_poll >= max(1.0, args.poll_interval_s):
                progress_remaining = max(
                    0.001,
                    args.progress_timeout_s - (now - last_progress),
                )
                metrics = parse_metrics(
                    http_text(
                        base_url,
                        "/metrics",
                        min(
                            args.http_timeout_s,
                            remaining(arm_deadline, "arm"),
                            progress_remaining,
                            max(0.05, args.poll_interval_s),
                        ),
                    )
                )
                if retained_metric not in metrics or metrics[retained_metric] < last_metric_value:
                    raise ContractError("recorder retained-sample metric is missing or decreased")
                if metrics[retained_metric] > last_metric_value:
                    last_progress = now
                last_metric_value = metrics[retained_metric]
                last_metrics_poll = now
            if not recorder_ready and time.monotonic() - last_progress > args.progress_timeout_s:
                raise ContractError("recorder made no progress before progress timeout")
            for error in worker_errors:
                if error is not None:
                    raise ContractError(f"completion request failed: {error}") from error
            time.sleep(args.poll_interval_s)
        for error in worker_errors:
            if error is not None:
                raise ContractError(f"completion request failed: {error}") from error
        if any(result is None for result in results):
            raise ContractError("completion worker exited without a result")
        requests = [result for result in results if result is not None]
        failed = False
    finally:
        if failed:
            barrier.abort()
            abort_endpoint()
            cleanup_deadline = float(abort_state["deadline"])
            for worker in workers:
                worker.join(timeout=max(0.0, cleanup_deadline - time.monotonic()))
            if any(worker.is_alive() for worker in workers):
                raise ContractError("completion workers did not exit after endpoint cleanup")
            if abort_state["error"] is not None:
                raise ContractError("server endpoint cleanup failed") from abort_state["error"]
        else:
            connections.close_all()
            for worker in workers:
                worker.join(timeout=0)
    return {
        "schema": ARM_SCHEMA,
        "ordinal": entry["ordinal"],
        "cell_id": cell["cell_id"],
        "active_slots": active,
        "prompt_token_id": args.prompt_token_id,
        "prompt_tokens": prompt_tokens,
        "output_tokens": args.output_tokens,
        "common_wall_ms": (time.monotonic() - common_started) * 1000.0,
        "requests": requests,
    }


def run_arm(
    *,
    repo_root: Path,
    output_dir: Path,
    source_identity_value: str,
    entry: dict[str, Any],
    cell: dict[str, Any],
    args: argparse.Namespace,
    job_deadline: float,
) -> None:
    paths = arm_paths(output_dir, entry["ordinal"])
    paths["root"].mkdir(parents=True, exist_ok=True)
    command = server_arguments(args, cell, paths["record"], source_identity_value)
    command_artifact = {
        "schema": ARM_SCHEMA,
        "source_identity": source_identity_value,
        "ordinal": entry["ordinal"],
        "cell": cell,
        "arguments": command,
    }
    if paths["command"].exists():
        if load_json(paths["command"]) != command_artifact:
            raise ContractError(f"arm command changed during resume: ordinal={entry['ordinal']}")
    else:
        write_json(paths["command"], command_artifact, False)

    if port_is_open("127.0.0.1", args.port):
        raise ContractError(f"profiling port is already occupied: {args.port}")
    arm_deadline = min(job_deadline, time.monotonic() + args.arm_timeout_s)
    environment = os.environ.copy()
    if args.disable_cuda_graphs:
        environment["GGML_CUDA_DISABLE_GRAPHS"] = "1"
    else:
        environment.pop("GGML_CUDA_DISABLE_GRAPHS", None)

    process: subprocess.Popen[Any] | None = None
    stdout_stream = None
    stderr_stream = None
    try:
        process, stdout_stream, stderr_stream = start_server(
            command,
            cwd=repo_root,
            environment=environment,
            stdout_path=paths["stdout"],
            stderr_path=paths["stderr"],
        )
        base_url = f"http://127.0.0.1:{args.port}"
        wait_ready(
            process,
            base_url,
            min(arm_deadline, time.monotonic() + args.readiness_timeout_s),
            args.http_timeout_s,
        )
        props = http_json(base_url, "/props", min(args.http_timeout_s, remaining(arm_deadline, "arm")))
        validate_props(props, args.parallel, cell["context_ceiling"])
        write_json(paths["props"], props, True)
        metrics_before = http_text(base_url, "/metrics", min(args.http_timeout_s, remaining(arm_deadline, "arm")))
        write_text(paths["metrics_before"], metrics_before, True)

        prompt_tokens = cell["context_ceiling"] - args.output_tokens - args.prompt_safety_tokens
        if prompt_tokens <= 0:
            raise ContractError(
                f"context ceiling {cell['context_ceiling']} cannot fit output and safety tokens"
            )
        client = run_wave(
            base_url=base_url,
            source_identity_value=source_identity_value,
            entry=entry,
            cell=cell,
            args=args,
            prompt_tokens=prompt_tokens,
            sidecar_path=paths["sidecar"],
            arm_deadline=arm_deadline,
            abort_server=lambda: stop_process_tree(process, args.stop_timeout_s),
            guard_check=lambda: verify_guard_lease(repo_root),
        )
        write_json(paths["client"], client, True)
        sidecar = validate_sidecar(
            load_json(paths["sidecar"]),
            source_identity_value=source_identity_value,
            cell=cell,
            max_draft_tokens_per_slot=args.max_draft_tokens_per_slot,
            samples_per_cell=args.samples_per_cell,
            warmup_shape_runs=args.warmup_shape_runs,
            require_ready=True,
        )
        arm_profile = validate_arm_profile(
            load_json(paths["record"]), cell, args.max_draft_tokens_per_slot
        )
        if float(arm_profile["entries"][0]["cost_us"]) != float(sidecar["coordinates"][0]["p95_us"]):
            raise ContractError("arm profile cost does not match the raw sidecar p95")
        metrics_after = http_text(base_url, "/metrics", min(args.http_timeout_s, remaining(arm_deadline, "arm")))
        write_text(paths["metrics_after"], metrics_after, True)
        before_values = parse_metrics(metrics_before)
        after_values = parse_metrics(metrics_after)
        retained_metric = "llamacpp:spec_decode_sps_record_retained_samples_total"
        if (
            retained_metric not in before_values
            or retained_metric not in after_values
            or after_values[retained_metric] < before_values[retained_metric]
        ):
            raise ContractError("recorder retained-sample metric is missing or decreased")
        receipt = {
            "schema": ARM_SCHEMA,
            "source_identity": source_identity_value,
            "ordinal": entry["ordinal"],
            "cell_id": cell["cell_id"],
            "status": "completed",
            "command_sha256": sha256_file(paths["command"]),
            "props_sha256": sha256_file(paths["props"]),
            "metrics_before_sha256": sha256_file(paths["metrics_before"]),
            "metrics_after_sha256": sha256_file(paths["metrics_after"]),
            "record_sha256": sha256_file(paths["record"]),
            "sidecar_sha256": sha256_file(paths["sidecar"]),
            "client_sha256": sha256_file(paths["client"]),
            "completed_at": utc_now(),
        }
        write_json(paths["receipt"], receipt, True)
    finally:
        if process is not None:
            stop_process_tree(process, args.stop_timeout_s)
        if stdout_stream is not None:
            stdout_stream.close()
        if stderr_stream is not None:
            stderr_stream.close()
        deadline = time.monotonic() + args.stop_timeout_s
        while port_is_open("127.0.0.1", args.port) and time.monotonic() < deadline:
            time.sleep(0.1)
        if port_is_open("127.0.0.1", args.port):
            raise ContractError(f"server port remained open after cleanup: {args.port}")


def merge_profiles(
    *,
    output_dir: Path,
    plan: dict[str, Any],
    source_identity_value: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    cells = {cell["cell_id"]: cell for cell in plan["cells"]}
    grouped: dict[str, list[dict[str, Any]]] = {cell_id: [] for cell_id in cells}
    client_artifacts: list[dict[str, Any]] = []
    arm_hashes: list[dict[str, Any]] = []
    for entry in plan["schedule"]:
        cell = cells[entry["cell_id"]]
        paths = arm_paths(output_dir, entry["ordinal"])
        validated = validate_completed_arm(
            paths,
            entry=entry,
            cell=cell,
            source_identity_value=source_identity_value,
            args=args,
        )
        coordinate = validated["sidecar"]["coordinates"][0]
        grouped[cell["cell_id"]].extend(coordinate["samples"])
        client_artifacts.append(validated["client"])
        arm_hashes.append({"ordinal": entry["ordinal"], "receipt_sha256": sha256_file(paths["receipt"])})

    raw_entries: list[dict[str, Any]] = []
    for cell in plan["cells"]:
        samples = grouped[cell["cell_id"]]
        expected = args.samples_per_cell * plan["repetitions"]
        if len(samples) != expected:
            raise ContractError(f"merged sample count mismatch: {cell['cell_id']}")
        costs = [float(sample["cost_us"]) for sample in samples]
        raw_entries.append(
            {
                "context_tokens": cell["context_ceiling"],
                "active_slots": cell["active_slots"],
                "total_verify_rows": cell["total_verify_rows"],
                "raw_p50_us": nearest_rank(costs, 0.50),
                "raw_p95_us": nearest_rank(costs, 0.95),
                "raw_max_us": nearest_rank(costs, 1.00),
                "samples": samples,
            }
        )
    profile_entries: list[dict[str, Any]] = []
    for target in raw_entries:
        envelope_sources = [
            source["raw_p95_us"]
            for source in raw_entries
            if source["context_tokens"] <= target["context_tokens"]
            and source["active_slots"] <= target["active_slots"]
            and source["total_verify_rows"] <= target["total_verify_rows"]
        ]
        profile_entries.append(
            {
                "context_tokens": target["context_tokens"],
                "active_slots": target["active_slots"],
                "total_verify_rows": target["total_verify_rows"],
                "cost_us": max(envelope_sources),
            }
        )
    profile_entries.sort(key=lambda row: (row["context_tokens"], row["active_slots"], row["total_verify_rows"]))
    profile = {
        "schema_version": 2,
        "max_draft_tokens_per_slot": args.max_draft_tokens_per_slot,
        "entries": profile_entries,
    }
    validate_final_profile(profile, plan)
    audit = {
        "schema": AUDIT_SCHEMA,
        "source_identity": source_identity_value,
        "schedule_sha256": plan["schedule_sha256"],
        "aggregate": "nearest-rank-p95 then conservative monotone upper envelope",
        "samples_per_coordinate": args.samples_per_cell * plan["repetitions"],
        "coordinates": raw_entries,
        "arms": arm_hashes,
    }
    total_tokens = sum(
        request["generated_tokens"]
        for client in client_artifacts
        for request in client["requests"]
    )
    total_wall_s = sum(client["common_wall_ms"] for client in client_artifacts) / 1000.0
    return {
        "profile": profile,
        "audit": audit,
        "summary": {
            "schema": "llama.cpp-dspark-sps-run-summary/v1",
            "status": "completed",
            "source_identity": source_identity_value,
            "schedule_sha256": plan["schedule_sha256"],
            "arms_completed": len(plan["schedule"]),
            "coordinates": len(plan["cells"]),
            "samples": sum(len(row["samples"]) for row in raw_entries),
            "generated_tokens": total_tokens,
            "sum_arm_wall_seconds": total_wall_s,
            "aggregate_goodput_tps_over_sum_arm_walls": total_tokens / total_wall_s,
            "completed_at": utc_now(),
        },
    }


def command_run(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    guard_uuid = verify_guard_lease(repo_root)
    for field in (
        "port", "ctx_size", "parallel", "batch_size", "ubatch_size",
        "max_draft_tokens_per_slot", "samples_per_cell", "warmup_shape_runs",
        "prompt_safety_tokens", "job_timeout_s", "arm_timeout_s",
        "readiness_timeout_s", "http_timeout_s", "progress_timeout_s",
        "stop_timeout_s", "poll_interval_s",
    ):
        value = getattr(args, field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ContractError(f"--{field.replace('_', '-')} must be positive")
    if not 1 <= args.port <= 65535 or args.prompt_token_id < 0 or args.output_tokens < 0:
        raise ContractError("port, prompt token, or output token setting is out of range")
    if any(
        not isinstance(value, str) or not value.strip() or "\x00" in value or len(value) > 256
        for value in (args.runtime_label, args.model_alias)
    ):
        raise ContractError("runtime label and model alias must be non-empty bounded strings")
    plan_path = require_file(args.plan, "plan")
    args.server = require_file(args.server, "server")
    args.model = require_file(args.model, "model")
    args.draft_model = require_file(args.draft_model, "draft model")
    plan = validate_plan(load_json(plan_path))
    if args.max_draft_tokens_per_slot != plan["max_draft_tokens_per_slot"]:
        raise ContractError("runtime draft depth does not match plan")
    if args.samples_per_cell != plan["samples_per_cell"] or args.warmup_shape_runs != plan["warmup_shape_runs"]:
        raise ContractError("runtime recorder quotas do not match plan")
    if args.parallel < max(plan["active_slots"]):
        raise ContractError("--parallel is smaller than the plan active-slot axis")
    if args.ubatch_size > args.batch_size:
        raise ContractError("--ubatch-size must not exceed --batch-size")
    minimum_output = (args.samples_per_cell + args.warmup_shape_runs + 16) * (1 + args.max_draft_tokens_per_slot)
    if args.output_tokens == 0:
        args.output_tokens = minimum_output
    if args.output_tokens < minimum_output:
        raise ContractError(f"--output-tokens must be at least {minimum_output}")
    minimum_ceiling = min(plan["context_ceilings"])
    if args.output_tokens + args.prompt_safety_tokens >= minimum_ceiling:
        raise ContractError("smallest context ceiling cannot fit output and safety tokens")
    per_slot_context = args.ctx_size // args.parallel
    if per_slot_context < max(plan["context_ceilings"]):
        raise ContractError("per-slot context does not cover the largest context ceiling")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    gpu = gpu_identity(repo_root, guard_uuid)
    source_identity_value, components = build_identity(args, plan_path, repo_root, gpu)
    manifest_expected = make_manifest(source_identity_value, components, plan)
    manifest_path = output_dir / "manifest.json"
    state_path = output_dir / "run-state.json"
    if manifest_path.exists():
        manifest = validate_existing_manifest(load_json(manifest_path), manifest_expected)
    else:
        if any(output_dir.iterdir()):
            allowed = {"guard.stdout.log", "guard.stderr.log"}
            if any(path.name not in allowed for path in output_dir.iterdir()):
                raise ContractError("non-empty output directory has no compatible manifest")
        manifest = manifest_expected
        write_json(manifest_path, manifest, False)

    if state_path.exists():
        if not args.resume:
            raise ContractError("run state exists but --no-resume was requested")
        state = validate_state(load_json(state_path), source_identity_value, plan)
    else:
        if manifest_path.exists() and (output_dir / "runs").exists():
            raise ContractError("compatible manifest has run artifacts but no run state")
        state = new_state(source_identity_value, plan)
        write_json(state_path, state, False)

    cells = {cell["cell_id"]: cell for cell in plan["cells"]}
    job_deadline = time.monotonic() + args.job_timeout_s
    try:
        for entry in plan["schedule"]:
            remaining(job_deadline, "job")
            if runtime_bundle_identity(args.server) != components["runtime_bundle"]:
                raise ContractError("server runtime DLL bundle changed during profiling")
            ordinal = entry["ordinal"]
            cell = cells[entry["cell_id"]]
            if ordinal in state["completed_ordinals"]:
                validate_completed_arm(
                    arm_paths(output_dir, ordinal),
                    entry=entry,
                    cell=cell,
                    source_identity_value=source_identity_value,
                    args=args,
                )
                continue
            if ordinal != state["next_ordinal"]:
                raise ContractError("resume state is not a contiguous completed prefix")
            if verify_guard_lease(repo_root) != guard_uuid:
                raise ContractError("exclusive GPU guard identity changed before server arm")
            state["current_ordinal"] = ordinal
            state["updated_at"] = utc_now()
            state["error"] = None
            write_json(state_path, state, True)
            run_arm(
                repo_root=repo_root,
                output_dir=output_dir,
                source_identity_value=source_identity_value,
                entry=entry,
                cell=cell,
                args=args,
                job_deadline=job_deadline,
            )
            if runtime_bundle_identity(args.server) != components["runtime_bundle"]:
                raise ContractError("server runtime DLL bundle changed during profiling")
            validate_completed_arm(
                arm_paths(output_dir, ordinal),
                entry=entry,
                cell=cell,
                source_identity_value=source_identity_value,
                args=args,
            )
            state["completed_ordinals"].append(ordinal)
            state["next_ordinal"] = ordinal + 1
            state["current_ordinal"] = None
            state["updated_at"] = utc_now()
            write_json(state_path, state, True)

        if runtime_bundle_identity(args.server) != components["runtime_bundle"]:
            raise ContractError("server runtime DLL bundle changed during profiling")
        merged = merge_profiles(
            output_dir=output_dir,
            plan=plan,
            source_identity_value=source_identity_value,
            args=args,
        )
        audit_path = output_dir / "profile.samples.json"
        profile_path = output_dir / "profile.json"
        validation_path = output_dir / "validation.json"
        summary_path = output_dir / "summary.json"
        write_json(audit_path, merged["audit"], True)
        write_json(profile_path, merged["profile"], True)
        validate_final_profile(load_json(profile_path), plan)
        validation = {
            "schema": "llama.cpp-dspark-sps-run-validation/v1",
            "status": "pass",
            "source_identity": source_identity_value,
            "profile_sha256": sha256_file(profile_path),
            "audit_sha256": sha256_file(audit_path),
            "arms_revalidated": len(plan["schedule"]),
            "validated_at": utc_now(),
        }
        write_json(validation_path, validation, True)
        merged["summary"]["validation_sha256"] = sha256_file(validation_path)
        write_json(summary_path, merged["summary"], True)
        state["status"] = "completed"
        state["current_ordinal"] = None
        state["updated_at"] = utc_now()
        state["error"] = None
        write_json(state_path, state, True)
        print(canonical({"ok": True, "status": "completed", "summary": str(summary_path)}))
        return 0
    except BaseException as error:
        state["status"] = "failed"
        state["updated_at"] = utc_now()
        state["error"] = f"{type(error).__name__}: {error}"[:1000]
        write_json(state_path, state, True)
        raise


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

    run = subparsers.add_parser("run", help="run all server arms inside the exclusive GPU guard")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--server", type=Path, required=True)
    run.add_argument("--model", type=Path, required=True)
    run.add_argument("--draft-model", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--runtime-label", required=True)
    run.add_argument("--port", type=int, default=18136)
    run.add_argument("--model-alias", default="dspark-sps-profiler")
    run.add_argument("--ctx-size", type=int, default=65536)
    run.add_argument("--parallel", type=int, default=8)
    run.add_argument("--batch-size", type=int, default=2048)
    run.add_argument("--ubatch-size", type=int, default=512)
    run.add_argument("--target-kv", choices=("q4_0", "q8_0", "f16"), default="q4_0")
    run.add_argument("--draft-kv", choices=("q4_0", "q8_0", "f16"), default="q4_0")
    run.add_argument("--disable-unified-kv", action="store_true")
    run.add_argument("--disable-cuda-graphs", action="store_true")
    run.add_argument("--max-draft-tokens-per-slot", type=int, default=7)
    run.add_argument("--samples-per-cell", type=int, default=64)
    run.add_argument("--warmup-shape-runs", type=int, default=3)
    run.add_argument("--prompt-token-id", type=int, default=1)
    run.add_argument("--output-tokens", type=int, default=0)
    run.add_argument("--prompt-safety-tokens", type=int, default=16)
    run.add_argument("--log-verbosity", type=int, default=4)
    run.add_argument("--job-timeout-s", type=float, default=18000)
    run.add_argument("--arm-timeout-s", type=float, default=1200)
    run.add_argument("--readiness-timeout-s", type=float, default=300)
    run.add_argument("--http-timeout-s", type=float, default=900)
    run.add_argument("--progress-timeout-s", type=float, default=300)
    run.add_argument("--stop-timeout-s", type=float, default=30)
    run.add_argument("--poll-interval-s", type=float, default=0.25)
    run.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
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
    except Exception as error:
        print(
            canonical({"ok": False, "error": f"{type(error).__name__}: runner failed"}),
            file=sys.stderr,
        )
        return 3


if __name__ == "__main__":
    sys.exit(main())
