#!/usr/bin/env python3
"""Fail-closed Qwen3.8 DSpark production qualification runner.

``validate`` is CPU-only. ``run`` and ``target-trace`` are intentionally usable
only as the single child of Invoke-ExclusiveGpuTask.ps1. ``run`` owns the full
sequential qualification tier; ``target-trace`` owns only the P1 target server.
Neither command ever permits two servers to overlap.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import sys
import threading
import time
from typing import Any, Callable
import urllib.parse

import dspark_sps_profile as base


PLAN_SCHEMA = "ai-loader-qwen38-dspark-production-qualification/v1"
MANIFEST_SCHEMA = "ai-loader-qwen38-dspark-qualification-manifest/v1"
STATE_SCHEMA = "ai-loader-qwen38-dspark-qualification-state/v1"
SUMMARY_SCHEMA = "ai-loader-qwen38-dspark-qualification-summary/v1"
TARGET_TRACE_SUMMARY_SCHEMA = "ai-loader-qwen38-target-trace-summary/v1"
TARGET_TRACE_VALIDATION_SCHEMA = "ai-loader-qwen38-target-trace-validation/v1"
ARM_SCHEMA = "ai-loader-qwen38-dspark-qualification-arm/v1"
ALLOWED_TIERS = {"P1": (1, 2048), "P4": (4, 8192), "P8": (8, 8192)}
ARM_NAMES = {
    "P1": ["target-only", "static-dspark", "sps-execution"],
    "P4": ["target-only", "static-dspark-matched-cap3", "static-dspark-best-cap2", "sps-execution"],
    "P8": ["target-only", "static-dspark-cap1", "sps-execution"],
}
ARM_CAPS = {"P1": [None, 7, 7], "P4": [None, 3, 2, 3], "P8": [None, 1, 1]}
REQUIRED_METRICS = {
    "tokens": "llamacpp:tokens_predicted_total",
    "decode_seconds": "llamacpp:tokens_predicted_seconds_total",
    "prompt_tokens": "llamacpp:prompt_tokens_total",
    "cached_prompt_tokens": "llamacpp:prompt_tokens_cached_total",
    "draft_tokens": "llamacpp:spec_decode_num_draft_tokens_total",
    "accepted_tokens": "llamacpp:spec_decode_num_accepted_tokens_total",
    "draft_steps": "llamacpp:spec_decode_num_drafts_total",
    "rs_resizes": "llamacpp:recurrent_snapshot_resizes_total",
    "sps_ticks": "llamacpp:spec_decode_sps_plan_ticks_total",
    "sps_fallback": "llamacpp:spec_decode_sps_fallback_ticks_total",
    "sps_shadow": "llamacpp:spec_decode_sps_shadow_ticks_total",
    "static_rows": "llamacpp:spec_decode_sps_static_verify_rows_total",
    "planned_rows": "llamacpp:spec_decode_sps_planned_verify_rows_total",
    "executed_rows": "llamacpp:spec_decode_sps_executed_verify_rows_total",
    "requests_processing": "llamacpp:requests_processing",
    "requests_deferred": "llamacpp:requests_deferred",
    "rs_resident_depth": "llamacpp:recurrent_snapshot_resident_depth",
    "rs_configured_depth": "llamacpp:recurrent_snapshot_configured_depth",
    "rs_required_depth": "llamacpp:recurrent_snapshot_required_depth",
    "rs_pending_depth": "llamacpp:recurrent_snapshot_pending_depth",
}
GAUGE_METRICS = {"requests_processing", "requests_deferred", "rs_resident_depth", "rs_configured_depth", "rs_required_depth", "rs_pending_depth"}
TARGET_TRACE_REQUIRED_RUNTIME_DLLS = {
    "cublas64_13.dll", "cublaslt64_13.dll", "ggml-base.dll", "ggml-cpu.dll",
    "ggml-cuda.dll", "ggml.dll", "llama-common.dll", "llama-server-impl.dll",
    "llama.dll", "mtmd.dll",
}
MAX_SSE_BYTES = 16 * 1024 * 1024


ContractError = base.ContractError


class CompletionContractError(ContractError):
    def __init__(self, diagnostic: dict[str, Any]):
        self.diagnostic = diagnostic
        super().__init__("completion output contract failed: " + base.canonical(diagnostic))


class ParityContractError(ContractError):
    def __init__(self, diagnostic: dict[str, Any]):
        self.diagnostic = diagnostic
        super().__init__("exact parity failed: " + base.canonical(diagnostic))


def completion_error(failed_predicates: list[str], observed: dict[str, Any]) -> CompletionContractError:
    return CompletionContractError({
        "schema": "ai-loader-qwen38-completion-contract-diagnostic/v1",
        "failed_predicates": failed_predicates,
        "observed": observed,
    })


def _integer(value: Any, field: str, expected: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError(f"{field} must be a non-negative integer")
    if expected is not None and value != expected:
        raise ContractError(f"{field} must be {expected}")
    return value


def _finite(value: Any, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ContractError(f"{field} must be finite")
    result = float(value)
    if positive and result <= 0:
        raise ContractError(f"{field} must be positive")
    return result


def _resolve_configured(plan_path: Path, raw: str) -> Path:
    path = Path(raw)
    return (path if path.is_absolute() else plan_path.parent / path).resolve()


def validate_plan(value: Any, plan_path: Path, tier_name: str | None = None) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if not isinstance(value, dict) or value.get("schema") != PLAN_SCHEMA:
        raise ContractError(f"qualification plan schema must be {PLAN_SCHEMA}")
    base.require_exact_keys(value, {
        "schema", "status", "created_for", "execution_policy", "artifacts",
        "common_server_args", "deterministic_workload", "tiers", "correctness_gates",
        "reported_metrics", "promotion_gates", "estimated_runtime_minutes",
    }, "qualification plan")
    for key in ("execution_policy", "artifacts", "common_server_args", "deterministic_workload", "tiers", "correctness_gates", "promotion_gates"):
        if key not in value:
            raise ContractError(f"qualification plan is missing {key}")

    policy = value["execution_policy"]
    common = value["common_server_args"]
    workload = value["deterministic_workload"]
    if not isinstance(policy, dict) or not isinstance(common, dict) or not isinstance(workload, dict):
        raise ContractError("qualification policy, server args and workload must be objects")
    required_policy = {
        "gpu_processes_max": 1, "server_arms_sequential_only": True,
        "outer_guard_required": True, "preflight_requires_port_free": True,
        "preflight_requires_zero_llama_processes": True, "postflight_requires_zero_llama_processes": True,
        "stop_on_guard_violation": True, "stop_on_any_http_or_server_error": True,
        "port": 18136, "max_used_vram_mib": 27000, "min_free_vram_mib": 5000,
        "min_free_system_ram_gib": 18, "poll_interval_ms": 250,
    }
    for key, expected in required_policy.items():
        if policy.get(key) != expected:
            raise ContractError(f"execution_policy.{key} must be {expected!r}")
    expected_common = {
        "host": "127.0.0.1", "port": 18136, "batch_size": 2048,
        "ubatch_size": 128, "target_kv": "q8_0", "draft_kv": "q8_0",
        "flash_attn": True, "n_gpu_layers": "all", "device": "CUDA0",
        "split_mode": "none", "fit": "off", "cache_ram": 0,
        "ctx_checkpoints": 0, "cache_idle_slots": False,
        "cache_prompt": False, "kv_unified": True, "metrics": True,
        "reasoning": "off",
    }
    for key, expected in expected_common.items():
        if common.get(key) != expected:
            raise ContractError(f"common_server_args.{key} must be {expected!r}")
    expected_workload = {
        "endpoint": "/completion", "stream": True, "prompt_token_id": 1,
        "prompt_tokens_per_request": 736, "output_tokens_per_request": 256,
        "temperature": 0, "seed_base": 20260819,
        "seed_rule": "seed_base + client_index", "ignore_eos": True,
        "cache_prompt": False, "warmup_waves": 1, "measured_waves": 6,
        "request_barrier": True, "exact_stop_type": "limit", "truncated": False,
    }
    for key, expected in expected_workload.items():
        if workload.get(key) != expected:
            raise ContractError(f"deterministic_workload.{key} must be {expected!r}")

    tiers = value["tiers"]
    if not isinstance(tiers, list) or len(tiers) != 3:
        raise ContractError("qualification plan must contain exactly P1, P4 and P8")
    selected = None
    observed: set[str] = set()
    for raw_tier in tiers:
        if not isinstance(raw_tier, dict) or raw_tier.get("name") not in ALLOWED_TIERS:
            raise ContractError("qualification tier name is invalid")
        name = raw_tier["name"]
        if name in observed:
            raise ContractError(f"duplicate qualification tier: {name}")
        observed.add(name)
        parallel, ctx_size = ALLOWED_TIERS[name]
        _integer(raw_tier.get("parallel"), f"{name}.parallel", parallel)
        _integer(raw_tier.get("ctx_size"), f"{name}.ctx_size", ctx_size)
        arms = raw_tier.get("arms")
        if not isinstance(arms, list) or len(arms) != len(ARM_NAMES[name]):
            raise ContractError(f"{name}.arms has the wrong sequential arm count")
        if [arm.get("name") if isinstance(arm, dict) else None for arm in arms] != ARM_NAMES[name]:
            raise ContractError(f"{name}.arms are not in the required target/static/SPS order")
        if arms[0].get("speculation") != "none":
            raise ContractError(f"{name} target arm must disable speculation")
        for index, cap in enumerate(ARM_CAPS[name][1:], start=1):
            arm = arms[index]
            if arm.get("speculation") != "draft-dspark" or arm.get("n_max") != cap:
                raise ContractError(f"{name} arm {index} has the wrong DSpark cap")
            # False is required, not merely the current server default. This
            # protects qualification identity from environment/default drift.
            if arm.get("dynamic_rs") is not False:
                raise ContractError(f"{name} arm {index} must explicitly set dynamic_rs=false")
        if not isinstance(arms[-1].get("sps_profile"), str) or not arms[-1]["sps_profile"]:
            raise ContractError(f"{name} SPS arm is missing its profile")
        if name == tier_name:
            selected = raw_tier
    if observed != set(ALLOWED_TIERS):
        raise ContractError("qualification plan must contain exactly P1, P4 and P8")
    if tier_name is not None and selected is None:
        raise ContractError(f"unknown qualification tier: {tier_name}")
    return value, selected


def validate_profile(value: Any, tier: dict[str, Any]) -> dict[str, Any]:
    expected_max = int(tier["arms"][-1]["n_max"])
    root = base.require_exact_keys(value, {"schema_version", "max_draft_tokens_per_slot", "entries"}, "SPS profile")
    if root["schema_version"] != 2 or root["max_draft_tokens_per_slot"] != expected_max:
        raise ContractError("SPS profile schema or n_max does not match the selected tier")
    entries = root["entries"]
    if not isinstance(entries, list) or not entries or len(entries) > 4096:
        raise ContractError("SPS profile entries must be a non-empty bounded list")
    seen: dict[tuple[int, int, int], float] = {}
    for index, raw in enumerate(entries):
        entry = base.require_exact_keys(raw, {"context_tokens", "active_slots", "total_verify_rows", "cost_us"}, f"SPS profile entries[{index}]")
        context = _integer(entry["context_tokens"], "context_tokens")
        active = _integer(entry["active_slots"], "active_slots")
        rows = _integer(entry["total_verify_rows"], "total_verify_rows")
        cost = _finite(entry["cost_us"], "cost_us", positive=True)
        coordinate = (context, active, rows)
        if context <= 0 or active <= 0 or rows <= 0 or coordinate in seen:
            raise ContractError("SPS profile contains an invalid or duplicate coordinate")
        seen[coordinate] = cost
    parallel = int(tier["parallel"])
    required_context = 736 + 256
    matching = [(coordinate, cost) for coordinate, cost in seen.items() if coordinate[1] == parallel and coordinate[0] >= required_context]
    expected_rows = {parallel * (1 + cap) for cap in range(expected_max + 1)}
    if not matching or not expected_rows.issubset({coordinate[2] for coordinate, _ in matching}):
        raise ContractError("SPS profile does not cover the selected exact-active load and decode context")
    for left, left_cost in seen.items():
        for right, right_cost in seen.items():
            if all(a <= b for a, b in zip(left, right)) and left_cost > right_cost:
                raise ContractError("SPS profile cost table is not monotonic")
    return root


def validate_profiler_manifest(
    value: Any, *, server_sha: str, runtime_bundle: list[dict[str, str]],
    target_sha: str, draft_sha: str, gpu: dict[str, str], tier: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema") != base.MANIFEST_SCHEMA:
        raise ContractError("SPS profile directory is missing a valid profiler manifest")
    components = value.get("identity_components")
    source_identity = value.get("source_identity")
    if not isinstance(components, dict) or not isinstance(source_identity, str) or base.sha256_json(components) != source_identity:
        raise ContractError("SPS profiler manifest source identity is invalid")
    expected = {
        "server_sha256": server_sha, "runtime_bundle": runtime_bundle,
        "target_model_sha256": target_sha, "draft_model_sha256": draft_sha, "gpu": gpu,
    }
    for key, wanted in expected.items():
        if components.get(key) != wanted:
            raise ContractError(f"SPS profiler manifest is stale or incompatible: {key}")
    runtime = components.get("runtime")
    expected_runtime = {
        "parallel": tier["parallel"], "ctx_size": tier["ctx_size"],
        "batch_size": 2048, "ubatch_size": 128, "target_kv": "q8_0", "draft_kv": "q8_0",
        "dynamic_rs": False, "disable_unified_kv": False,
        "max_draft_tokens_per_slot": tier["arms"][-1]["n_max"],
    }
    if not isinstance(runtime, dict):
        raise ContractError("SPS profiler manifest runtime is missing")
    for key, wanted in expected_runtime.items():
        if runtime.get(key) != wanted:
            raise ContractError(f"SPS profiler runtime is incompatible: {key}")
    return value


def metric_snapshot(text: str) -> dict[str, float]:
    parsed = base.parse_metrics(text)
    missing = [metric for metric in REQUIRED_METRICS.values() if metric not in parsed]
    if missing:
        raise ContractError(f"required qualification metrics are missing: {missing}")
    return {name: _finite(parsed[metric], metric) for name, metric in REQUIRED_METRICS.items()}


def metric_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key in REQUIRED_METRICS:
        if key not in GAUGE_METRICS and after[key] < before[key]:
            raise ContractError(f"metric counter went backwards: {key}")
        result[key] = after[key] if key in GAUGE_METRICS else after[key] - before[key]
    return result


class AbortableSseConnections:
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

    def completion(self, base_url: str, body: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        parsed = urllib.parse.urlsplit(base_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"} or parsed.path not in {"", "/"}:
            raise ContractError("SSE endpoint must be a local HTTP origin")
        connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=timeout_s)
        started_at = time.monotonic()
        try:
            connection.connect()
            with self._lock:
                if self._closed:
                    raise ContractError("completion endpoint was closed")
                self._connections.add(connection)
            connection.request(
                "POST", "/completion", body=base.canonical(body).encode("utf-8"),
                headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
            )
            response = connection.getresponse()
            if response.status != 200:
                raise ContractError(f"/completion returned HTTP {response.status}")
            content_type = response.getheader("Content-Type", "").split(";", 1)[0].strip().lower()
            if content_type != "text/event-stream":
                raise ContractError("/completion did not return text/event-stream")
            return parse_completion_sse(response, started_at=started_at)
        except ContractError:
            raise
        except (OSError, TimeoutError, http.client.HTTPException) as error:
            raise ContractError("streaming /completion request failed") from error
        finally:
            with self._lock:
                self._connections.discard(connection)
            connection.close()


def parse_completion_sse(
    response: Any, clock: Callable[[], float] = time.monotonic, started_at: float | None = None,
) -> dict[str, Any]:
    started = clock() if started_at is None else started_at
    first_token_at: float | None = None
    tokens: list[int] = []
    contents: list[str] = []
    final: dict[str, Any] | None = None
    total_bytes = 0
    data_lines: list[bytes] = []

    def consume_event() -> None:
        nonlocal first_token_at, final, data_lines
        if not data_lines:
            return
        raw = b"\n".join(data_lines)
        data_lines = []
        try:
            event = json.loads(raw.decode("utf-8", errors="strict"), object_pairs_hook=base.reject_duplicate_keys)
        except (ContractError, UnicodeError, json.JSONDecodeError) as error:
            raise completion_error(
                ["sse_data_is_strict_utf8_json_without_duplicate_keys"],
                {"event_byte_count": len(raw)},
            ) from error
        if not isinstance(event, dict) or type(event.get("stop")) is not bool:
            raise completion_error(
                ["sse_event_is_object_with_boolean_stop"],
                {"event_type": type(event).__name__, "event_keys": sorted(event.keys()) if isinstance(event, dict) else []},
            )
        event_tokens = event.get("tokens")
        content = event.get("content")
        if not isinstance(event_tokens, list) or any(isinstance(token, bool) or not isinstance(token, int) for token in event_tokens):
            raise completion_error(
                ["sse_event_tokens_is_integer_array"],
                {"event_keys": sorted(event.keys()), "tokens_type": type(event_tokens).__name__},
            )
        if not isinstance(content, str):
            raise completion_error(
                ["sse_event_content_is_string"],
                {"event_keys": sorted(event.keys()), "content_type": type(content).__name__},
            )
        if final is not None:
            raise completion_error(
                ["terminal_event_is_last_data_event"],
                {"raw_partial_token_count": len(tokens), "extra_event_keys": sorted(event.keys())},
            )
        if event["stop"]:
            final = event
        else:
            if not event_tokens:
                # The native is_begin marker is converted to JSON null and
                # consumed by the HTTP layer solely to flush status/headers;
                # it is never serialized as an SSE data event. With
                # return_progress=false, an observed empty nonterminal event
                # is therefore invalid.
                raise completion_error(
                    ["nonterminal_event_contains_raw_token_ids"],
                    {
                        "raw_partial_token_count": len(tokens),
                        "partial_event_keys": sorted(event.keys()),
                        "partial_token_count": 0,
                        "partial_content_char_count": len(content),
                        "tokens_predicted": event.get("tokens_predicted"),
                        "tokens_evaluated": event.get("tokens_evaluated"),
                    },
                )
            if first_token_at is None:
                first_token_at = clock()
            tokens.extend(event_tokens)
            contents.append(content)

    while True:
        line = response.readline(1024 * 1024 + 1)
        if not line:
            consume_event()
            break
        total_bytes += len(line)
        if total_bytes > MAX_SSE_BYTES or len(line) > 1024 * 1024:
            raise completion_error(
                ["sse_response_is_within_size_bounds"],
                {
                    "total_byte_count": total_bytes,
                    "line_byte_count": len(line),
                    "maximum_total_byte_count": MAX_SSE_BYTES,
                    "maximum_line_byte_count": 1024 * 1024,
                },
            )
        stripped = line.rstrip(b"\r\n")
        if not stripped:
            consume_event()
        elif stripped.startswith(b":"):
            continue
        elif stripped.startswith(b"data:"):
            data_lines.append(stripped[5:].lstrip(b" "))
        elif stripped.startswith((b"event:", b"id:", b"retry:")):
            continue
        else:
            candidate, separator, _ = stripped.partition(b":")
            safe_field_name = (
                candidate.decode("ascii")
                if separator and re.fullmatch(rb"[A-Za-z][A-Za-z0-9_-]{0,31}", candidate)
                else None
            )
            raise completion_error(
                ["sse_line_uses_supported_field"],
                {
                    "line_byte_count": len(stripped),
                    "has_field_separator": bool(separator),
                    "has_safe_ascii_field_name": safe_field_name is not None,
                    "safe_ascii_field_name": safe_field_name,
                },
            )
    ended = clock()
    if final is None or first_token_at is None:
        raise completion_error(
            [name for name, passed in (
                ("terminal_event_is_present", final is not None),
                ("at_least_one_raw_token_event_is_present", first_token_at is not None),
            ) if not passed],
            {
                "raw_partial_token_count": len(tokens),
                "terminal_event_present": final is not None,
                "terminal_event_keys": sorted(final.keys()) if isinstance(final, dict) else [],
            },
        )
    timings = final.get("timings")
    timing_values = timings if isinstance(timings, dict) else {}
    predicates = {
        "raw_partial_token_count_is_256": len(tokens) == 256,
        "terminal_tokens_are_empty": final.get("tokens") == [],
        "terminal_content_is_empty": final.get("content") == "",
        "stop_type_is_limit": final.get("stop_type") == "limit",
        "truncated_is_false": final.get("truncated") is False,
        "tokens_predicted_is_256": final.get("tokens_predicted") == 256,
        "tokens_evaluated_is_736": final.get("tokens_evaluated") == 736,
        # Source semantics: final tokens_cached is the slot's current context
        # length (server-context.cpp send_final_response), not prompt-cache
        # reuse. Exact no-reuse is timings.cache_n == 0 and the Prometheus
        # prompt_tokens_cached_total delta checked by the arm.
        "tokens_cached_is_nonnegative_integer": (
            not isinstance(final.get("tokens_cached"), bool)
            and isinstance(final.get("tokens_cached"), int)
            and final.get("tokens_cached") >= 0
        ),
        "timings_is_object": isinstance(timings, dict),
        "timings_cache_n_is_0": timing_values.get("cache_n") == 0,
        "timings_prompt_n_is_736": timing_values.get("prompt_n") == 736,
        "timings_predicted_n_is_256": timing_values.get("predicted_n") == 256,
        "timings_predicted_ms_is_positive_finite": (
            not isinstance(timing_values.get("predicted_ms"), bool)
            and isinstance(timing_values.get("predicted_ms"), (int, float))
            and math.isfinite(float(timing_values.get("predicted_ms")))
            and float(timing_values.get("predicted_ms")) > 0
        ),
        "timings_predicted_per_second_is_positive_finite": (
            not isinstance(timing_values.get("predicted_per_second"), bool)
            and isinstance(timing_values.get("predicted_per_second"), (int, float))
            and math.isfinite(float(timing_values.get("predicted_per_second")))
            and float(timing_values.get("predicted_per_second")) > 0
        ),
    }
    failed = [name for name, passed in predicates.items() if not passed]
    if failed:
        diagnostic = {
            "schema": "ai-loader-qwen38-completion-contract-diagnostic/v1",
            "failed_predicates": failed,
            "observed": {
                "raw_partial_token_count": len(tokens),
                "terminal_event_keys": sorted(final.keys()),
                "terminal_token_count": len(final.get("tokens")) if isinstance(final.get("tokens"), list) else None,
                "terminal_content_char_count": len(final.get("content")) if isinstance(final.get("content"), str) else None,
                "stop_type": final.get("stop_type"), "truncated": final.get("truncated"),
                "tokens_predicted": final.get("tokens_predicted"),
                "tokens_evaluated": final.get("tokens_evaluated"),
                "tokens_cached": final.get("tokens_cached"),
                "timings_keys": sorted(timing_values.keys()),
                "timings": {
                    key: timing_values.get(key)
                    for key in ("cache_n", "prompt_n", "predicted_n", "predicted_ms", "predicted_per_second")
                },
            },
        }
        raise CompletionContractError(diagnostic)
    predicted_ms = float(timing_values["predicted_ms"])
    predicted_tps = float(timing_values["predicted_per_second"])
    content = "".join(contents)
    return {
        "tokens": tokens,
        "token_sha256": hashlib.sha256(base.canonical(tokens).encode("utf-8")).hexdigest(),
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "ttft_ms": (first_token_at - started) * 1000.0,
        "wall_ms": (ended - started) * 1000.0,
        "predicted_ms": predicted_ms,
        "predicted_tps": predicted_tps,
        "generated_tokens": len(tokens),
        "stop_type": final["stop_type"],
        "truncated": final["truncated"],
    }


def server_arguments(args: argparse.Namespace, tier: dict[str, Any], arm: dict[str, Any]) -> list[str]:
    common = args.plan_value["common_server_args"]
    values = [
        str(args.server), "--model", str(args.model), "--host", "127.0.0.1",
        "--port", str(args.port), "--alias", "qwen38-dspark-qualification",
        "--ctx-size", str(tier["ctx_size"]), "--parallel", str(tier["parallel"]),
        "--cache-type-k", common["target_kv"], "--cache-type-v", common["target_kv"],
        "--flash-attn", "on", "--batch-size", str(common["batch_size"]),
        "--ubatch-size", str(common["ubatch_size"]), "--n-gpu-layers", "all",
        "--device", "CUDA0", "--split-mode", "none", "--fit", "off",
        "--cache-ram", "0", "--ctx-checkpoints", "0", "--no-cache-idle-slots",
        "--no-cache-prompt", "--no-webui", "--metrics", "--reasoning", "off",
        "-lv", str(common.get("log_verbosity", 4)), "--kv-unified",
    ]
    if arm["speculation"] != "none":
        values += [
            "--spec-type", "draft-dspark", "--spec-draft-model", str(args.draft_model),
            "--spec-draft-n-max", str(arm["n_max"]), "--spec-draft-n-min", "0",
            "--spec-draft-p-min", "0", "--spec-draft-type-k", common["draft_kv"],
            "--spec-draft-type-v", common["draft_kv"], "--spec-draft-ngl", "all",
            "--spec-draft-device", "CUDA0", "--no-spec-draft-sps-shadow",
            "--no-spec-draft-adaptive", "--no-spec-draft-dynamic-rs",
        ]
        if arm["name"] == "sps-execution":
            values += ["--spec-draft-sps-profile", str(args.profile)]
    return values


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ContractError("percentile requires samples")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def paired_bootstrap_ci(candidate: list[float], baseline: list[float], *, seed: int = 20260819, resamples: int = 10000) -> dict[str, Any]:
    if len(candidate) != len(baseline) or len(candidate) < 2:
        raise ContractError("paired bootstrap requires equal sample vectors")
    gains = [100.0 * (new / old - 1.0) for new, old in zip(candidate, baseline) if old > 0]
    if len(gains) != len(candidate) or any(not math.isfinite(value) for value in gains):
        raise ContractError("paired bootstrap samples must be finite and positive")
    rng = random.Random(seed)
    estimates = [statistics.mean(rng.choices(gains, k=len(gains))) for _ in range(resamples)]
    return {
        "method": "paired nonparametric bootstrap of mean per-wave percent gain",
        "seed": seed, "resamples": resamples, "n": len(gains),
        "median_paired_gain_percent": statistics.median(gains),
        "mean_paired_gain_percent": statistics.mean(gains),
        "ci95_percent": [_percentile(estimates, 0.025), _percentile(estimates, 0.975)],
    }


def _read_metrics(base_url: str, timeout_s: float) -> tuple[str, dict[str, float]]:
    text = base.http_text(base_url, "/metrics", timeout_s)
    return text, metric_snapshot(text)


def wait_quiescent(base_url: str, timeout_s: float, process: Any, guard_check: Callable[[], Any]) -> tuple[str, dict[str, float]]:
    deadline = time.monotonic() + timeout_s
    while True:
        guard_check()
        if process.poll() is not None:
            raise ContractError("llama-server exited while waiting for idle metrics")
        text, metrics = _read_metrics(base_url, min(5.0, base.remaining(deadline, "server quiescence")))
        if metrics["requests_processing"] == 0 and metrics["requests_deferred"] == 0:
            return text, metrics
        time.sleep(min(0.05, base.remaining(deadline, "server quiescence")))


def run_wave(
    *, base_url: str, parallel: int, wave: int, workload: dict[str, Any], timeout_s: float,
    guard_check: Callable[[], Any], process: Any,
) -> tuple[dict[str, Any], dict[str, float], str, str]:
    guard_check()
    if process.poll() is not None:
        raise ContractError("llama-server exited before a qualification wave")
    before_text, before = wait_quiescent(base_url, timeout_s, process, guard_check)
    barrier = threading.Barrier(parallel + 1)
    connections = AbortableSseConnections()
    starts: list[float | None] = [None] * parallel
    ends: list[float | None] = [None] * parallel

    def one(client: int) -> dict[str, Any]:
        barrier.wait(timeout=timeout_s)
        starts[client] = time.monotonic()
        body = {
            "model": "qwen38-dspark-qualification",
            "prompt": [workload["prompt_token_id"]] * workload["prompt_tokens_per_request"],
            "n_predict": workload["output_tokens_per_request"],
            "temperature": workload["temperature"],
            "seed": workload["seed_base"] + client,
            "ignore_eos": workload["ignore_eos"], "cache_prompt": workload["cache_prompt"],
            "stream": True, "return_tokens": True,
        }
        result = connections.completion(base_url, body, timeout_s)
        ends[client] = time.monotonic()
        result.update({"client": client, "seed": body["seed"], "wave": wave})
        return result

    results: list[dict[str, Any] | None] = [None] * parallel
    errors: list[Exception | None] = [None] * parallel
    workers: list[threading.Thread] = []

    def worker(client: int) -> None:
        try:
            results[client] = one(client)
        except Exception as error:
            errors[client] = error
            connections.close_all()

    failed = True
    try:
        for client in range(parallel):
            thread = threading.Thread(target=worker, args=(client,), name=f"qualification-{client}", daemon=True)
            workers.append(thread)
            thread.start()
        barrier.wait(timeout=timeout_s)
        deadline = time.monotonic() + timeout_s
        while any(thread.is_alive() for thread in workers):
            guard_check()
            if process.poll() is not None:
                raise ContractError("llama-server exited during a qualification wave")
            for error in errors:
                if error is not None:
                    if isinstance(error, CompletionContractError):
                        raise error
                    raise ContractError(f"completion request failed: {error}") from error
            base.remaining(deadline, "qualification wave")
            time.sleep(0.05)
        for error in errors:
            if error is not None:
                if isinstance(error, CompletionContractError):
                    raise error
                raise ContractError(f"completion request failed: {error}") from error
        if any(result is None for result in results):
            raise ContractError("qualification worker ended without a result")
        failed = False
    finally:
        connections.close_all()
        if failed:
            barrier.abort()
        cleanup_deadline = time.monotonic() + min(30.0, timeout_s)
        for thread in workers:
            thread.join(timeout=max(0.0, cleanup_deadline - time.monotonic()))
        if any(thread.is_alive() for thread in workers):
            raise ContractError("completion workers did not exit after endpoint cleanup")
    wave_started = min(value for value in starts if value is not None)
    wave_ended = max(value for value in ends if value is not None)
    wave_ms = (wave_ended - wave_started) * 1000.0
    final_results = [result for result in results if result is not None]
    after_text, after = wait_quiescent(base_url, timeout_s, process, guard_check)
    delta = metric_delta(before, after)
    expected_tokens = parallel * workload["output_tokens_per_request"]
    if delta["tokens"] != expected_tokens:
        raise ContractError(f"server token counter delta {delta['tokens']} does not equal {expected_tokens}")
    overlap = max(value for value in starts if value is not None) < min(value for value in ends if value is not None)
    if parallel > 1 and not overlap:
        raise ContractError("parallel qualification requests did not overlap")
    return {
        "wave": wave, "parallel": parallel, "wall_ms": wave_ms,
        "goodput_tps": expected_tokens / (wave_ms / 1000.0), "clients": final_results,
        "request_overlap_proven": overlap,
        "metric_delta": delta,
    }, after, before_text, after_text


def _arm_paths(output: Path, arm_name: str) -> dict[str, Path]:
    root = output / "arms" / arm_name
    return {
        "root": root, "command": root / "command.json", "props": root / "props.json",
        "metrics_before": root / "metrics-before.txt", "metrics_after": root / "metrics-after.txt",
        "waves": root / "waves.json", "summary": root / "summary.json",
        "receipt": root / "receipt.json",
        "stdout": root / "server.stdout.log", "stderr": root / "server.stderr.log",
    }


def verify_exact_artifacts(args: argparse.Namespace) -> None:
    observed = {
        name: base.sha256_file(path)
        for name, path in args.artifact_paths_expected.items()
    }
    if observed != args.artifact_hashes_expected:
        raise ContractError("bound artifacts changed during qualification")
    if base.runtime_bundle_identity(args.server) != args.runtime_bundle_expected:
        raise ContractError("runtime bundle changed during qualification")


def _validate_props(props: dict[str, Any], tier: dict[str, Any], model: Path) -> None:
    # This qualification explicitly uses --kv-unified. llama-context therefore
    # sets n_ctx_seq == n_ctx (it divides by n_seq_max only for non-unified KV),
    # and /props reports that exact n_ctx_seq value. Prove the CLI context was
    # honored exactly, not merely that this short workload happens to fit.
    base.validate_props(props, int(tier["parallel"]), int(tier["ctx_size"]))
    n_ctx = props["default_generation_settings"]["n_ctx"]
    if n_ctx != int(tier["ctx_size"]):
        raise ContractError("/props per-sequence context does not match the exact unified-KV tier")
    raw_model = props.get("model_path")
    if not isinstance(raw_model, str) or Path(raw_model).resolve() != model.resolve():
        raise ContractError("/props model_path does not match the exact target model")
    if props.get("model_alias") != "qwen38-dspark-qualification":
        raise ContractError("/props model_alias does not match the qualification identity")


def _summarize_arm(arm: dict[str, Any], parallel: int, waves: list[dict[str, Any]], arm_before: dict[str, float], arm_after: dict[str, float]) -> dict[str, Any]:
    name = arm["name"]
    measured = waves[1:]
    if len(measured) != 6:
        raise ContractError("qualification arm did not retain exactly six measured waves")
    expected_depth = int(arm.get("n_max", 0)) if arm["speculation"] != "none" else 0
    for wave in measured:
        wave_metrics = wave["metric_delta"]
        if (
            wave_metrics["rs_configured_depth"] != expected_depth
            or wave_metrics["rs_resident_depth"] != expected_depth
            or wave_metrics["rs_required_depth"] != expected_depth
            or wave_metrics["rs_pending_depth"] != -1
        ):
            raise ContractError("recurrent snapshot depth gauges do not prove fixed tier storage on every measured wave")
    delta = metric_delta(arm_before, arm_after)
    measured_delta = {
        key: (max(float(wave["metric_delta"][key]) for wave in measured) if key in GAUGE_METRICS
              else sum(float(wave["metric_delta"][key]) for wave in measured))
        for key in REQUIRED_METRICS
    }
    goodputs = [float(wave["goodput_tps"]) for wave in measured]
    ttfts = [float(client["ttft_ms"]) for wave in measured for client in wave["clients"]]
    request_tps = [256.0 / (float(client["wall_ms"]) / 1000.0) for wave in measured for client in wave["clients"]]
    if measured_delta["decode_seconds"] <= 0:
        raise ContractError("backend decode-time counter did not advance")
    if measured_delta["tokens"] != parallel * 256 * 6:
        raise ContractError("measured server generation-token accounting is not exact")
    if measured_delta["prompt_tokens"] != parallel * 736 * 6 or measured_delta["cached_prompt_tokens"] != 0:
        raise ContractError("measured prompt accounting is not exactly 736 uncached tokens per request")
    if measured_delta["rs_resizes"] != 0:
        raise ContractError("recurrent snapshot storage resized despite Dynamic-RS=false")
    if (
        measured_delta["rs_configured_depth"] != expected_depth
        or measured_delta["rs_resident_depth"] != expected_depth
        or measured_delta["rs_required_depth"] != expected_depth
        or measured_delta["rs_pending_depth"] != -1
    ):
        raise ContractError("recurrent snapshot depth gauges do not prove fixed tier storage")
    if name == "sps-execution":
        if measured_delta["sps_ticks"] <= 0 or measured_delta["sps_fallback"] != 0 or measured_delta["sps_shadow"] != 0:
            raise ContractError("SPS execution used fallback/shadow or did not plan")
        if measured_delta["planned_rows"] > measured_delta["static_rows"]:
            raise ContractError("SPS planned more rows than its static comparison")
        if measured_delta["executed_rows"] != measured_delta["planned_rows"]:
            raise ContractError("SPS executed rows differ from planned rows")
    else:
        if any(
            measured_delta[key] != 0
            for key in ("sps_ticks", "sps_fallback", "sps_shadow", "static_rows", "planned_rows", "executed_rows")
        ):
            raise ContractError("non-SPS arm unexpectedly reported SPS planning")
    if arm["speculation"] == "none":
        if any(measured_delta[key] != 0 for key in ("draft_tokens", "accepted_tokens", "draft_steps")):
            raise ContractError("target-only arm unexpectedly reported speculative work")
    elif measured_delta["draft_tokens"] <= 0 or measured_delta["draft_steps"] <= 0:
        raise ContractError("DSpark arm did not perform speculative work")
    offered = measured_delta["draft_tokens"]
    accepted = measured_delta["accepted_tokens"]
    if accepted > offered:
        raise ContractError("accepted speculative tokens exceed offered tokens")
    return {
        "schema": ARM_SCHEMA, "name": name, "warmup_waves": 1, "measured_waves": 6,
        "goodput_tps_per_wave": goodputs, "goodput_tps_median": statistics.median(goodputs),
        "backend_decode_tps": measured_delta["tokens"] / measured_delta["decode_seconds"],
        "per_request_tps_median": statistics.median(request_tps),
        "per_request_tps_min": min(request_tps), "per_request_tps_max": max(request_tps),
        "ttft_ms_p50": base.nearest_rank(ttfts, 0.50), "ttft_ms_p95": base.nearest_rank(ttfts, 0.95),
        "acceptance_rate": accepted / offered if offered > 0 else None,
        "accepted_draft_tokens": accepted, "offered_draft_tokens": offered,
        "planned_verify_rows": measured_delta["planned_rows"],
        "executed_verify_rows": measured_delta["executed_rows"],
        "static_verify_rows": measured_delta["static_rows"],
        "sps_fallback_ticks": measured_delta["sps_fallback"],
        "sps_plan_ticks": measured_delta["sps_ticks"],
        "recurrent_snapshot_resizes": measured_delta["rs_resizes"],
        "measured_metric_delta": measured_delta, "whole_arm_metric_delta": delta,
    }


def server_environment(source: dict[str, str] | None = None) -> dict[str, str]:
    environment = dict(os.environ if source is None else source)
    cuda_visible_devices = environment.get("CUDA_VISIBLE_DEVICES", "")
    if not cuda_visible_devices:
        raise ContractError("exclusive guard did not bind CUDA_VISIBLE_DEVICES")
    for key in list(environment):
        upper = key.upper()
        if upper.startswith("LLAMA_") or upper.startswith("GGML_") or upper.startswith("CUDA_"):
            environment.pop(key, None)
    environment["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["LLAMA_ARG_SPEC_DRAFT_DYNAMIC_RS"] = "0"
    environment["LLAMA_ARG_SPEC_DRAFT_ADAPTIVE"] = "0"
    return environment


def server_environment_contract(environment: dict[str, str]) -> dict[str, Any]:
    return {
        "schema": "ai-loader-qwen38-server-environment/v1",
        "cuda_visible_devices": environment["CUDA_VISIBLE_DEVICES"],
        "cuda_device_order": environment["CUDA_DEVICE_ORDER"],
        "dynamic_rs": environment["LLAMA_ARG_SPEC_DRAFT_DYNAMIC_RS"],
        "adaptive_draft": environment["LLAMA_ARG_SPEC_DRAFT_ADAPTIVE"],
        "scrubbed_prefixes": ["LLAMA_", "GGML_", "CUDA_"],
    }


def validate_target_trace_runtime_bundle(bundle: list[dict[str, str]]) -> list[dict[str, str]]:
    names = {
        entry.get("name", "").lower()
        for entry in bundle
        if isinstance(entry, dict)
    }
    missing = sorted(TARGET_TRACE_REQUIRED_RUNTIME_DLLS - names)
    if missing:
        raise ContractError(f"target trace runtime bundle is missing mandatory adjacent DLLs: {missing}")
    return bundle


def run_arm(args: argparse.Namespace, tier: dict[str, Any], arm: dict[str, Any], guard_uuid: str, deadline: float) -> dict[str, Any]:
    if base.verify_guard_lease(args.repo_root) != guard_uuid:
        raise ContractError("exclusive GPU identity changed before arm startup")
    if base.port_is_open("127.0.0.1", args.port):
        raise ContractError(f"qualification port is already occupied: {args.port}")
    verify_exact_artifacts(args)
    paths = _arm_paths(args.output_dir, arm["name"])
    paths["root"].mkdir(parents=True, exist_ok=False)
    command = server_arguments(args, tier, arm)
    base.write_json(paths["command"], {
        "schema": ARM_SCHEMA, "tier": tier["name"], "arm": arm["name"],
        "dynamic_rs": False, "arguments": command, "command_sha256": base.sha256_json(command),
    }, False)
    job_name = os.environ.get(base.GUARD_JOB_NAME_ENV) if os.name == "nt" else None
    baseline = base.windows_job_active_process_count(job_name) if job_name else None
    if baseline is not None and baseline != 1:
        raise ContractError("exclusive GPU Job Object must contain only the qualification runner")
    environment = server_environment()
    if server_environment_contract(environment) != args.server_environment_expected:
        raise ContractError("server runtime environment changed during qualification")
    process = None
    stdout_stream = None
    stderr_stream = None
    completed_result: dict[str, Any] | None = None
    try:
        process, stdout_stream, stderr_stream = base.start_server(
            command, cwd=args.repo_root, environment=environment,
            stdout_path=paths["stdout"], stderr_path=paths["stderr"],
        )
        arm_deadline = min(deadline, time.monotonic() + args.arm_timeout_s)
        base_url = f"http://127.0.0.1:{args.port}"
        base.wait_ready(process, base_url, min(arm_deadline, time.monotonic() + args.readiness_timeout_s), args.http_timeout_s)
        props = base.http_json(base_url, "/props", min(args.http_timeout_s, base.remaining(arm_deadline, "arm")))
        _validate_props(props, tier, args.model)
        base.write_json(paths["props"], props, False)
        before_text, before = _read_metrics(base_url, min(args.http_timeout_s, base.remaining(arm_deadline, "arm")))
        base.write_text(paths["metrics_before"], before_text, False)
        waves: list[dict[str, Any]] = []
        wave_metric_texts: list[dict[str, str]] = []
        for wave in range(7):
            if base.verify_guard_lease(args.repo_root) != guard_uuid:
                raise ContractError("exclusive GPU identity changed during qualification")
            result, _, metrics_before, metrics_after = run_wave(
                base_url=base_url, parallel=int(tier["parallel"]), wave=wave,
                workload=args.plan_value["deterministic_workload"],
                timeout_s=min(args.http_timeout_s, base.remaining(arm_deadline, "arm")),
                guard_check=lambda: base.verify_guard_lease(args.repo_root), process=process,
            )
            waves.append(result)
            wave_metric_texts.append({"before": metrics_before, "after": metrics_after})
        after_text, after = _read_metrics(base_url, min(args.http_timeout_s, base.remaining(arm_deadline, "arm")))
        base.write_text(paths["metrics_after"], after_text, False)
        base.write_json(paths["waves"], {"schema": ARM_SCHEMA, "waves": waves, "metric_snapshots": wave_metric_texts}, False)
        summary = _summarize_arm(arm, int(tier["parallel"]), waves, before, after)
        base.write_json(paths["summary"], summary, False)
        completed_result = {"summary": summary, "waves": waves}
    finally:
        if process is not None:
            base.stop_process_tree(process, args.stop_timeout_s, job_name=job_name, expected_job_processes=baseline)
        if stdout_stream is not None:
            stdout_stream.close()
        if stderr_stream is not None:
            stderr_stream.close()
        stop_deadline = time.monotonic() + args.stop_timeout_s
        while base.port_is_open("127.0.0.1", args.port) and time.monotonic() < stop_deadline:
            time.sleep(0.1)
        if base.port_is_open("127.0.0.1", args.port):
            raise ContractError(f"server port remained open after cleanup: {args.port}")
        if baseline is not None and base.windows_job_active_process_count(job_name) != baseline:
            raise ContractError("server descendants remained after qualification arm cleanup")
        verify_exact_artifacts(args)
    if completed_result is None:
        raise ContractError("qualification arm ended without a result")
    stderr_text = paths["stderr"].read_text(encoding="utf-8", errors="strict")
    if re.search(r"(?im)(CUDA error|GGML_ASSERT|segmentation fault|terminate called|rollback.+(?:failed|error))", stderr_text):
        raise ContractError("server stderr contains a fatal CUDA/assert/rollback signature")
    bound_files = ["command", "props", "metrics_before", "metrics_after", "waves", "summary", "stdout", "stderr"]
    receipt = {
        "schema": ARM_SCHEMA, "tier": tier["name"], "arm": arm["name"],
        "dynamic_rs": False, "artifacts": {
            key: {"path": paths[key].name, "sha256": base.sha256_file(paths[key])}
            for key in bound_files
        }, "completed_at": base.utc_now(),
    }
    base.write_json(paths["receipt"], receipt, False)
    return completed_result


def _token_array_sha256(tokens: list[int]) -> str:
    return hashlib.sha256(base.canonical(tokens).encode("utf-8")).hexdigest()


def _parity_diagnostic(
    baseline_arm: str, candidate_arm: str,
    baseline_wave: dict[str, Any], candidate_wave: dict[str, Any],
    baseline_client: dict[str, Any], candidate_client: dict[str, Any],
) -> dict[str, Any]:
    baseline_tokens = baseline_client["tokens"]
    candidate_tokens = candidate_client["tokens"]
    overlap = min(len(baseline_tokens), len(candidate_tokens))
    differing = [index for index in range(overlap) if baseline_tokens[index] != candidate_tokens[index]]
    difference_count = len(differing) + abs(len(baseline_tokens) - len(candidate_tokens))
    first_difference = differing[0] if differing else (overlap if len(baseline_tokens) != len(candidate_tokens) else -1)
    baseline_token_sha256 = _token_array_sha256(baseline_tokens)
    candidate_token_sha256 = _token_array_sha256(candidate_tokens)
    return {
        "schema": "ai-loader-qwen38-parity-diagnostic/v1",
        "baseline_arm": baseline_arm,
        "candidate_arm": candidate_arm,
        "baseline_wave": baseline_wave.get("wave", -1),
        "candidate_wave": candidate_wave.get("wave", -1),
        "client": baseline_client.get("client", -1),
        "seed": baseline_client.get("seed", -1),
        "baseline_length": len(baseline_tokens),
        "candidate_length": len(candidate_tokens),
        "first_difference_index": first_difference,
        "difference_count": difference_count,
        "baseline_token_sha256": baseline_token_sha256,
        "candidate_token_sha256": candidate_token_sha256,
        "baseline_reported_token_sha256": baseline_client.get("token_sha256", ""),
        "candidate_reported_token_sha256": candidate_client.get("token_sha256", ""),
        "baseline_reported_hash_matches": baseline_client.get("token_sha256", baseline_token_sha256) == baseline_token_sha256,
        "candidate_reported_hash_matches": candidate_client.get("token_sha256", candidate_token_sha256) == candidate_token_sha256,
        "baseline_content_sha256": baseline_client.get("content_sha256", ""),
        "candidate_content_sha256": candidate_client.get("content_sha256", ""),
    }


def exact_parity(arm_results: list[dict[str, Any]]) -> dict[str, Any]:
    arm_names = [arm.get("summary", {}).get("name", f"arm-{index}") for index, arm in enumerate(arm_results)]
    for arm_index, arm in enumerate(arm_results):
        for wave in arm["waves"]:
            for client in wave["clients"]:
                if client.get("token_sha256", _token_array_sha256(client["tokens"])) != _token_array_sha256(client["tokens"]):
                    raise ParityContractError(_parity_diagnostic(
                        arm_names[arm_index], arm_names[arm_index], wave, wave, client, client,
                    ))
        first = arm["waves"][0]
        for later in arm["waves"][1:]:
            for expected, actual in zip(first["clients"], later["clients"]):
                if (
                    expected["client"] != actual["client"] or expected["tokens"] != actual["tokens"]
                    or expected["content_sha256"] != actual["content_sha256"]
                ):
                    raise ParityContractError(_parity_diagnostic(
                        arm_names[arm_index], arm_names[arm_index], first, later, expected, actual,
                    ))
    baseline = arm_results[0]["waves"]
    comparisons = 0
    for arm_index, arm in enumerate(arm_results[1:], start=1):
        if len(arm["waves"]) != len(baseline):
            raise ContractError("arm wave counts differ")
        for expected_wave, actual_wave in zip(baseline, arm["waves"]):
            for expected, actual in zip(expected_wave["clients"], actual_wave["clients"]):
                comparisons += 1
                if expected["client"] != actual["client"] or expected["seed"] != actual["seed"]:
                    raise ContractError("client identity differs across arms")
                if expected["tokens"] != actual["tokens"]:
                    raise ParityContractError(_parity_diagnostic(
                        arm_names[0], arm_names[arm_index], expected_wave, actual_wave, expected, actual,
                    ))
                if expected["content_sha256"] != actual["content_sha256"]:
                    raise ParityContractError(_parity_diagnostic(
                        arm_names[0], arm_names[arm_index], expected_wave, actual_wave, expected, actual,
                    ))
    return {"exact_token_and_content_parity": True, "comparisons": comparisons}


def command_validate(args: argparse.Namespace) -> int:
    plan_path = args.plan.resolve()
    _, tier = validate_plan(base.load_json(plan_path), plan_path, args.tier)
    print(base.canonical({"schema": PLAN_SCHEMA, "valid": True, "tier": tier["name"] if tier else None}))
    return 0


def _same_path(label: str, supplied: Path, configured: str, plan_path: Path) -> Path:
    resolved = base.require_file(supplied, label).resolve()
    expected = _resolve_configured(plan_path, configured)
    if resolved != expected:
        raise ContractError(f"{label} does not match the exact path in the qualification plan")
    return resolved


def _persist_failed_state(path: Path, state: dict[str, Any], error: Exception) -> None:
    """Atomically persist a content-free structured completion diagnostic."""
    state["status"] = "failed"
    state["error_type"] = type(error).__name__
    state["error"] = str(error)
    state.pop("completion_diagnostic", None)
    state.pop("parity_diagnostic", None)
    if isinstance(error, CompletionContractError):
        state["completion_diagnostic"] = error.diagnostic
    if isinstance(error, ParityContractError):
        state["parity_diagnostic"] = error.diagnostic
    state["updated_at"] = base.utc_now()
    base.write_json(path, state, True)


def command_run(args: argparse.Namespace) -> int:
    args.repo_root = Path(__file__).resolve().parent.parent
    guard_uuid = base.verify_guard_lease(args.repo_root)
    plan_path = args.plan.resolve()
    plan, tier = validate_plan(base.load_json(plan_path), plan_path, args.tier)
    assert tier is not None
    args.plan_value = plan
    artifacts = plan["artifacts"]
    args.server = _same_path("server", args.server, artifacts["server"], plan_path)
    args.model = _same_path("target model", args.model, artifacts["target_model"], plan_path)
    args.draft_model = _same_path("draft model", args.draft_model, artifacts["draft_model"], plan_path)
    args.profile = _same_path("SPS profile", args.profile, tier["profile_output"], plan_path)
    if base.sha256_file(args.model) != artifacts["target_model_sha256"] or base.sha256_file(args.draft_model) != artifacts["draft_model_sha256"]:
        raise ContractError("model SHA-256 does not match the qualification plan")
    validate_profile(base.load_json(args.profile), tier)
    args.port = int(plan["execution_policy"]["port"])
    if base.port_is_open("127.0.0.1", args.port):
        raise ContractError(f"qualification port is already occupied: {args.port}")
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ContractError("qualification output directory must be absent or empty (bounded no-resume runner)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gpu = base.gpu_identity(args.repo_root, guard_uuid)
    args.server_environment_expected = server_environment_contract(server_environment())
    args.runtime_bundle_expected = validate_target_trace_runtime_bundle(base.runtime_bundle_identity(args.server))
    profiler_manifest_path = args.profile.parent / "manifest.json"
    args.artifact_paths_expected = {
        "server": args.server, "model": args.model, "draft_model": args.draft_model,
        "profile": args.profile, "profiler_manifest": profiler_manifest_path,
    }
    args.artifact_hashes_expected = {
        name: base.sha256_file(path) for name, path in args.artifact_paths_expected.items()
    }
    profiler_manifest = validate_profiler_manifest(
        base.load_json(profiler_manifest_path), server_sha=base.sha256_file(args.server),
        runtime_bundle=args.runtime_bundle_expected, target_sha=base.sha256_file(args.model),
        draft_sha=base.sha256_file(args.draft_model), gpu=gpu, tier=tier,
    )
    components = {
        "schema": "ai-loader-qwen38-dspark-qualification-identity/v1",
        "source_commit": base.source_commit(args.repo_root), "python": sys.version.split()[0],
        "plan_sha256": base.sha256_file(plan_path), "runner_sha256": base.sha256_file(Path(__file__)),
        "wrapper_sha256": base.sha256_file(Path(__file__).with_name("Invoke-Qwen38DsparkQualification.ps1")),
        "guard_sha256": base.sha256_file(Path(__file__).with_name("Invoke-ExclusiveGpuTask.ps1")),
        "server_sha256": base.sha256_file(args.server), "runtime_bundle": args.runtime_bundle_expected,
        "server_environment": args.server_environment_expected,
        "target_model_sha256": base.sha256_file(args.model), "draft_model_sha256": base.sha256_file(args.draft_model),
        "profile_sha256": base.sha256_file(args.profile),
        "profiler_manifest_sha256": base.sha256_file(profiler_manifest_path),
        "profiler_source_identity": profiler_manifest["source_identity"], "tier": args.tier,
        "dynamic_rs": False, "gpu": gpu, "runtime_label": args.runtime_label,
        "arm_arguments": [
            {"name": arm["name"], "arguments": server_arguments(args, tier, arm)}
            for arm in tier["arms"]
        ],
    }
    identity = base.sha256_json(components)
    manifest = {"schema": MANIFEST_SCHEMA, "identity": identity, "components": components, "created_at": base.utc_now()}
    base.write_json(args.output_dir / "manifest.json", manifest, False)
    state = {"schema": STATE_SCHEMA, "identity": identity, "status": "running", "completed_arms": [], "updated_at": base.utc_now()}
    base.write_json(args.output_dir / "run-state.json", state, True)
    deadline = time.monotonic() + args.job_timeout_s
    results: list[dict[str, Any]] = []
    try:
        for arm in tier["arms"]:
            results.append(run_arm(args, tier, arm, guard_uuid, deadline))
            state["completed_arms"].append(arm["name"])
            state["updated_at"] = base.utc_now()
            base.write_json(args.output_dir / "run-state.json", state, True)
        parity = exact_parity(results)
        target = results[0]["summary"]
        sps = results[-1]["summary"]
        matched_static = results[1]["summary"]
        best_static = results[2]["summary"] if args.tier == "P4" else matched_static
        target_ci = paired_bootstrap_ci(sps["goodput_tps_per_wave"], target["goodput_tps_per_wave"])
        static_target_ci = paired_bootstrap_ci(best_static["goodput_tps_per_wave"], target["goodput_tps_per_wave"])
        matched_ci = paired_bootstrap_ci(sps["goodput_tps_per_wave"], matched_static["goodput_tps_per_wave"])
        best_ci = paired_bootstrap_ci(sps["goodput_tps_per_wave"], best_static["goodput_tps_per_wave"])
        ttft_limit = float(best_static["ttft_ms_p95"]) * 1.10
        promoted = (
            parity["exact_token_and_content_parity"]
            and best_ci["median_paired_gain_percent"] > 0
            and best_ci["ci95_percent"][0] > 0
            and target_ci["median_paired_gain_percent"] > 0
            and target_ci["ci95_percent"][0] > 0
            and float(sps["ttft_ms_p95"]) <= ttft_limit
        )
        summary = {
            "schema": SUMMARY_SCHEMA, "identity": identity, "tier": args.tier,
            "dynamic_rs": False, "status": "qualified" if promoted else "not-promoted",
            "parity": parity, "arms": [result["summary"] for result in results],
            "sps_vs_target": target_ci, "sps_vs_matched_static": matched_ci,
            "sps_vs_best_product_static": best_ci,
            "best_static_vs_target": static_target_ci,
            "promotion": {
                "promoted": promoted, "positive_median_gain": best_ci["median_paired_gain_percent"] > 0,
                "paired_ci_lower_above_zero": best_ci["ci95_percent"][0] > 0,
                "ttft_p95_regression_within_10_percent": float(sps["ttft_ms_p95"]) <= ttft_limit,
                "ttft_p95_limit_ms": ttft_limit,
                "best_static_promoted_over_target": (
                    static_target_ci["median_paired_gain_percent"] > 0
                    and static_target_ci["ci95_percent"][0] > 0
                ),
            },
            "completed_at": base.utc_now(),
        }
        base.write_json(args.output_dir / "summary.json", summary, False)
        reread_summary = base.load_json(args.output_dir / "summary.json")
        if reread_summary != summary:
            raise ContractError("atomic final summary reread did not match")
        base.write_json(args.output_dir / "validation.json", {
            "schema": SUMMARY_SCHEMA, "identity": identity, "valid": True,
            "summary_sha256": base.sha256_file(args.output_dir / "summary.json"),
            "arm_receipts": {
                arm["name"]: base.sha256_file(_arm_paths(args.output_dir, arm["name"])["receipt"])
                for arm in tier["arms"]
            }, "validated_at": base.utc_now(),
        }, False)
        state["status"] = "completed"
        state["summary_sha256"] = base.sha256_file(args.output_dir / "summary.json")
        state["updated_at"] = base.utc_now()
        base.write_json(args.output_dir / "run-state.json", state, True)
        print(base.canonical(summary))
        return 0
    except Exception as error:
        _persist_failed_state(args.output_dir / "run-state.json", state, error)
        raise


def command_target_trace(args: argparse.Namespace) -> int:
    """Capture one fresh P1 target-only trace using the qualification server path."""
    args.repo_root = Path(__file__).resolve().parent.parent
    guard_uuid = base.verify_guard_lease(args.repo_root)
    plan_path = args.plan.resolve()
    plan, tier = validate_plan(base.load_json(plan_path), plan_path, "P1")
    assert tier is not None
    args.plan_value = plan
    artifacts = plan["artifacts"]
    args.server = _same_path("server", args.server, artifacts["server"], plan_path)
    args.model = _same_path("target model", args.model, artifacts["target_model"], plan_path)
    model_sha256 = base.sha256_file(args.model)
    if model_sha256 != artifacts["target_model_sha256"]:
        raise ContractError("target model SHA-256 does not match the qualification plan")
    args.port = int(plan["execution_policy"]["port"])
    if base.port_is_open("127.0.0.1", args.port):
        raise ContractError(f"qualification port is already occupied: {args.port}")
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.name.lower() != "p1-final-v5":
        raise ContractError("target trace output directory leaf must be p1-final-v5")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ContractError("target trace output directory must be absent or empty (bounded no-resume runner)")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gpu = base.gpu_identity(args.repo_root, guard_uuid)
    args.server_environment_expected = server_environment_contract(server_environment())
    args.runtime_bundle_expected = base.runtime_bundle_identity(args.server)
    wrapper_path = Path(__file__).with_name("Invoke-Qwen38TargetTrace.ps1")
    guard_path = Path(__file__).with_name("Invoke-ExclusiveGpuTask.ps1")
    args.artifact_paths_expected = {
        "server": args.server, "model": args.model, "plan": plan_path,
        "runner": Path(__file__), "wrapper": wrapper_path, "guard": guard_path,
    }
    args.artifact_hashes_expected = {
        name: base.sha256_file(path) for name, path in args.artifact_paths_expected.items()
    }
    target_arm = tier["arms"][0]
    if target_arm != {"name": "target-only", "speculation": "none"}:
        raise ContractError("P1 target arm is not the exact non-speculative arm")
    components = {
        "schema": "ai-loader-qwen38-dspark-qualification-identity/v1",
        "source_commit": base.source_commit(args.repo_root), "python": sys.version.split()[0],
        "plan_sha256": base.sha256_file(plan_path), "runner_sha256": base.sha256_file(Path(__file__)),
        "wrapper_sha256": base.sha256_file(wrapper_path), "guard_sha256": base.sha256_file(guard_path),
        "server_sha256": base.sha256_file(args.server), "runtime_bundle": args.runtime_bundle_expected,
        "server_environment": args.server_environment_expected,
        "target_model_sha256": model_sha256, "tier": "P1", "dynamic_rs": False,
        "gpu": gpu, "runtime_label": args.runtime_label,
        "trace_mode": "target-only",
        "arm_arguments": [{"name": "target-only", "arguments": server_arguments(args, tier, target_arm)}],
    }
    identity = base.sha256_json(components)
    manifest = {"schema": MANIFEST_SCHEMA, "identity": identity, "components": components, "created_at": base.utc_now()}
    base.write_json(args.output_dir / "manifest.json", manifest, False)
    state = {
        "schema": STATE_SCHEMA, "identity": identity, "status": "running",
        "completed_arms": [], "mode": "target-only", "updated_at": base.utc_now(),
    }
    state_path = args.output_dir / "run-state.json"
    base.write_json(state_path, state, True)
    deadline = time.monotonic() + args.job_timeout_s
    try:
        result = run_arm(args, tier, target_arm, guard_uuid, deadline)
        state["completed_arms"].append("target-only")
        parity = exact_parity([result])
        parity["repeat_wave_comparisons"] = 6
        summary = {
            "schema": TARGET_TRACE_SUMMARY_SCHEMA, "identity": identity, "tier": "P1",
            "mode": "target-only", "dynamic_rs": False, "status": "completed",
            "repeat_parity": parity, "arm": result["summary"], "completed_at": base.utc_now(),
        }
        summary_path = args.output_dir / "summary.json"
        base.write_json(summary_path, summary, False)
        arm_paths = _arm_paths(args.output_dir, "target-only")
        first_client = result["waves"][0]["clients"][0]
        validation = {
            "schema": TARGET_TRACE_VALIDATION_SCHEMA, "identity": identity, "valid": True,
            "qualification_identity": identity, "model_sha256": model_sha256,
            "manifest_sha256": base.sha256_file(args.output_dir / "manifest.json"),
            "waves_sha256": base.sha256_file(arm_paths["waves"]),
            "token_trace_sha256": first_client["token_sha256"],
            "command_sha256": base.sha256_file(arm_paths["command"]),
            "receipt_sha256": base.sha256_file(arm_paths["receipt"]),
            "summary_sha256": base.sha256_file(summary_path),
            "runtime_bundle_sha256": base.sha256_json(args.runtime_bundle_expected),
            "runtime_bundle_count": len(args.runtime_bundle_expected),
            "validated_at": base.utc_now(),
        }
        base.write_json(args.output_dir / "validation.json", validation, False)
        state["status"] = "completed"
        state["summary_sha256"] = validation["summary_sha256"]
        state["validation_sha256"] = base.sha256_file(args.output_dir / "validation.json")
        state["updated_at"] = base.utc_now()
        base.write_json(state_path, state, True)
        print(base.canonical(summary))
        return 0
    except Exception as error:
        _persist_failed_state(state_path, state, error)
        raise


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ContractError(f"{field} must be a lowercase SHA-256")
    return value


def _validate_target_trace_wave(
    raw: Any, index: int, reference_tokens: list[int] | None, reference_content_sha256: str | None,
) -> tuple[list[int], str]:
    wave = base.require_exact_keys(raw, {
        "wave", "parallel", "wall_ms", "goodput_tps", "clients",
        "request_overlap_proven", "metric_delta",
    }, f"target trace waves[{index}]")
    _integer(wave["wave"], "wave", index)
    _integer(wave["parallel"], "parallel", 1)
    _finite(wave["wall_ms"], "wall_ms", positive=True)
    _finite(wave["goodput_tps"], "goodput_tps", positive=True)
    if wave["request_overlap_proven"] is not True:
        raise ContractError("target trace request overlap/lifetime proof is missing")
    clients = wave["clients"]
    if not isinstance(clients, list) or len(clients) != 1:
        raise ContractError("P1 target trace must contain exactly one client per wave")
    client = base.require_exact_keys(clients[0], {
        "tokens", "token_sha256", "content_sha256", "ttft_ms", "wall_ms",
        "predicted_ms", "predicted_tps", "generated_tokens", "stop_type",
        "truncated", "client", "seed", "wave",
    }, f"target trace waves[{index}].clients[0]")
    _integer(client["client"], "client", 0)
    _integer(client["seed"], "seed", 20260819)
    _integer(client["wave"], "client wave", index)
    _integer(client["generated_tokens"], "generated_tokens", 256)
    for field in ("ttft_ms", "wall_ms", "predicted_ms", "predicted_tps"):
        _finite(client[field], field, positive=True)
    if client["stop_type"] != "limit" or client["truncated"] is not False:
        raise ContractError("target trace completion did not stop at the exact token limit")
    tokens = client["tokens"]
    if (
        not isinstance(tokens, list) or len(tokens) != 256
        or any(isinstance(token, bool) or not isinstance(token, int) for token in tokens)
    ):
        raise ContractError("target trace client must contain exactly 256 integer token IDs")
    token_sha256 = _token_array_sha256(tokens)
    if client["token_sha256"] != token_sha256:
        raise ContractError("target trace client token hash does not match its raw token array")
    content_sha256 = _require_sha256(client["content_sha256"], "content_sha256")
    if reference_tokens is not None and tokens != reference_tokens:
        raise ContractError("target trace raw tokens do not repeat exactly across all seven waves")
    if reference_content_sha256 is not None and content_sha256 != reference_content_sha256:
        raise ContractError("target trace content hash does not repeat exactly across all seven waves")
    metrics = base.require_exact_keys(wave["metric_delta"], set(REQUIRED_METRICS), "target trace wave metric_delta")
    numeric = {key: _finite(metrics[key], f"metric_delta.{key}") for key in REQUIRED_METRICS}
    exact = {
        "tokens": 256.0, "prompt_tokens": 736.0, "cached_prompt_tokens": 0.0,
        "draft_tokens": 0.0, "accepted_tokens": 0.0, "draft_steps": 0.0,
        "rs_resizes": 0.0, "sps_ticks": 0.0, "sps_fallback": 0.0,
        "sps_shadow": 0.0, "static_rows": 0.0, "planned_rows": 0.0,
        "executed_rows": 0.0, "requests_processing": 0.0,
        "requests_deferred": 0.0, "rs_resident_depth": 0.0,
        "rs_configured_depth": 0.0, "rs_required_depth": 0.0,
        "rs_pending_depth": -1.0,
    }
    for key, expected in exact.items():
        if numeric[key] != expected:
            raise ContractError(f"target trace metric_delta.{key} must be {expected}")
    if numeric["decode_seconds"] <= 0:
        raise ContractError("target trace decode-time counter did not advance")
    return tokens, content_sha256


def command_validate_target_trace(args: argparse.Namespace) -> int:
    """CPU-only, fail-closed validation of a completed target trace bundle."""
    repo_root = Path(__file__).resolve().parent.parent
    plan_path = args.plan.resolve()
    plan, tier = validate_plan(base.load_json(plan_path), plan_path, "P1")
    assert tier is not None
    artifacts = plan["artifacts"]
    server = _same_path("server", args.server, artifacts["server"], plan_path)
    model = _same_path("target model", args.model, artifacts["target_model"], plan_path)
    model_sha256 = base.sha256_file(model)
    if model_sha256 != artifacts["target_model_sha256"]:
        raise ContractError("target model SHA-256 does not match the qualification plan")
    output = args.output_dir.resolve()
    if output.name.lower() != "p1-final-v5" or not output.is_dir():
        raise ContractError("target trace output must be an existing p1-final-v5 directory")
    wrapper_path = Path(__file__).with_name("Invoke-Qwen38TargetTrace.ps1")
    guard_path = Path(__file__).with_name("Invoke-ExclusiveGpuTask.ps1")
    manifest_path = output / "manifest.json"
    manifest = base.require_exact_keys(base.load_json(manifest_path), {
        "schema", "identity", "components", "created_at",
    }, "target trace manifest")
    if manifest["schema"] != MANIFEST_SCHEMA or not isinstance(manifest["created_at"], str) or not manifest["created_at"]:
        raise ContractError("target trace manifest schema/timestamp is invalid")
    components = base.require_exact_keys(manifest["components"], {
        "schema", "source_commit", "python", "plan_sha256", "runner_sha256",
        "wrapper_sha256", "guard_sha256", "server_sha256", "runtime_bundle",
        "server_environment", "target_model_sha256", "tier", "dynamic_rs", "gpu",
        "runtime_label", "trace_mode", "arm_arguments",
    }, "target trace manifest components")
    identity = _require_sha256(manifest["identity"], "manifest.identity")
    if identity != base.sha256_json(components):
        raise ContractError("target trace manifest identity does not match canonical components")
    expected_component_values = {
        "schema": "ai-loader-qwen38-dspark-qualification-identity/v1",
        "source_commit": base.source_commit(repo_root), "python": sys.version.split()[0],
        "plan_sha256": base.sha256_file(plan_path), "runner_sha256": base.sha256_file(Path(__file__)),
        "wrapper_sha256": base.sha256_file(wrapper_path), "guard_sha256": base.sha256_file(guard_path),
        "server_sha256": base.sha256_file(server),
        "runtime_bundle": validate_target_trace_runtime_bundle(base.runtime_bundle_identity(server)),
        "target_model_sha256": model_sha256, "tier": "P1", "dynamic_rs": False,
        "trace_mode": "target-only",
    }
    for key, expected in expected_component_values.items():
        if components[key] != expected:
            raise ContractError(f"target trace manifest current identity mismatch: {key}")
    if not isinstance(components["runtime_label"], str) or not components["runtime_label"]:
        raise ContractError("target trace runtime label is missing")
    gpu = base.require_exact_keys(components["gpu"], {
        "uuid", "name", "driver_version", "compute_capability", "cuda_driver_runtime",
    }, "target trace GPU identity")
    if gpu["uuid"] != args.gpu_uuid or not args.gpu_uuid.startswith("GPU-") or any(not isinstance(value, str) or not value for value in gpu.values()):
        raise ContractError("target trace GPU identity does not match the outer guard receipt")
    expected_environment = {
        "schema": "ai-loader-qwen38-server-environment/v1",
        "cuda_visible_devices": args.gpu_uuid, "cuda_device_order": "PCI_BUS_ID",
        "dynamic_rs": "0", "adaptive_draft": "0",
        "scrubbed_prefixes": ["LLAMA_", "GGML_", "CUDA_"],
    }
    if components["server_environment"] != expected_environment:
        raise ContractError("target trace server environment does not match the guarded GPU contract")
    target_arm = tier["arms"][0]
    runtime_args = argparse.Namespace(server=server, model=model, port=18136, plan_value=plan)
    expected_arguments = server_arguments(runtime_args, tier, target_arm)
    if components["arm_arguments"] != [{"name": "target-only", "arguments": expected_arguments}]:
        raise ContractError("target trace manifest arm arguments are not the exact P1 target command")

    paths = _arm_paths(output, "target-only")
    command = base.require_exact_keys(base.load_json(paths["command"]), {
        "schema", "tier", "arm", "dynamic_rs", "arguments", "command_sha256",
    }, "target trace command")
    if command != {
        "schema": ARM_SCHEMA, "tier": "P1", "arm": "target-only", "dynamic_rs": False,
        "arguments": expected_arguments, "command_sha256": base.sha256_json(expected_arguments),
    }:
        raise ContractError("target trace command artifact is not exact")
    props = base.load_json(paths["props"])
    _validate_props(props, tier, model)
    before_text = paths["metrics_before"].read_text(encoding="utf-8", errors="strict")
    after_text = paths["metrics_after"].read_text(encoding="utf-8", errors="strict")
    before = metric_snapshot(before_text)
    after = metric_snapshot(after_text)
    waves_artifact = base.require_exact_keys(base.load_json(paths["waves"]), {
        "schema", "waves", "metric_snapshots",
    }, "target trace waves artifact")
    if waves_artifact["schema"] != ARM_SCHEMA:
        raise ContractError("target trace waves schema is invalid")
    waves = waves_artifact["waves"]
    snapshots = waves_artifact["metric_snapshots"]
    if not isinstance(waves, list) or len(waves) != 7 or not isinstance(snapshots, list) or len(snapshots) != 7:
        raise ContractError("target trace must contain seven waves and seven raw metric snapshot pairs")
    reference_tokens: list[int] | None = None
    reference_content_sha256: str | None = None
    for index, wave in enumerate(waves):
        reference_tokens, reference_content_sha256 = _validate_target_trace_wave(
            wave, index, reference_tokens, reference_content_sha256,
        )
        snapshot = base.require_exact_keys(snapshots[index], {"before", "after"}, f"metric_snapshots[{index}]")
        if not isinstance(snapshot["before"], str) or not isinstance(snapshot["after"], str):
            raise ContractError("target trace raw metric snapshots must be text")
        observed_delta = metric_delta(metric_snapshot(snapshot["before"]), metric_snapshot(snapshot["after"]))
        if observed_delta != wave["metric_delta"]:
            raise ContractError("target trace raw metric snapshot delta does not match its wave")
    expected_arm_summary = _summarize_arm(target_arm, 1, waves, before, after)
    if base.load_json(paths["summary"]) != expected_arm_summary:
        raise ContractError("target trace arm summary does not recompute from its waves and metrics")
    stderr_text = paths["stderr"].read_text(encoding="utf-8", errors="strict")
    paths["stdout"].read_text(encoding="utf-8", errors="strict")
    if re.search(r"(?im)(CUDA error|GGML_ASSERT|segmentation fault|terminate called|rollback.+(?:failed|error))", stderr_text):
        raise ContractError("target trace server stderr contains a fatal signature")
    receipt = base.require_exact_keys(base.load_json(paths["receipt"]), {
        "schema", "tier", "arm", "dynamic_rs", "artifacts", "completed_at",
    }, "target trace arm receipt")
    if receipt["schema"] != ARM_SCHEMA or receipt["tier"] != "P1" or receipt["arm"] != "target-only" or receipt["dynamic_rs"] is not False:
        raise ContractError("target trace arm receipt identity is invalid")
    expected_artifact_names = {
        "command": "command.json", "props": "props.json", "metrics_before": "metrics-before.txt",
        "metrics_after": "metrics-after.txt", "waves": "waves.json", "summary": "summary.json",
        "stdout": "server.stdout.log", "stderr": "server.stderr.log",
    }
    artifacts_receipt = base.require_exact_keys(receipt["artifacts"], set(expected_artifact_names), "target trace receipt artifacts")
    for key, filename in expected_artifact_names.items():
        item = base.require_exact_keys(artifacts_receipt[key], {"path", "sha256"}, f"receipt artifact {key}")
        expected_path = paths["root"] / filename
        if item["path"] != filename or expected_path.resolve().parent != paths["root"].resolve() or item["sha256"] != base.sha256_file(expected_path):
            raise ContractError(f"target trace receipt does not canonically bind {key}")
    repeat_parity = exact_parity([{"summary": expected_arm_summary, "waves": waves}])
    repeat_parity["repeat_wave_comparisons"] = 6
    outer_summary = base.require_exact_keys(base.load_json(output / "summary.json"), {
        "schema", "identity", "tier", "mode", "dynamic_rs", "status",
        "repeat_parity", "arm", "completed_at",
    }, "target trace summary")
    expected_outer = {
        "schema": TARGET_TRACE_SUMMARY_SCHEMA, "identity": identity, "tier": "P1",
        "mode": "target-only", "dynamic_rs": False, "status": "completed",
        "repeat_parity": repeat_parity, "arm": expected_arm_summary,
    }
    for key, expected in expected_outer.items():
        if outer_summary[key] != expected:
            raise ContractError(f"target trace summary mismatch: {key}")
    validation = base.require_exact_keys(base.load_json(output / "validation.json"), {
        "schema", "identity", "valid", "qualification_identity", "model_sha256",
        "manifest_sha256", "waves_sha256", "token_trace_sha256", "command_sha256",
        "receipt_sha256", "summary_sha256", "runtime_bundle_sha256",
        "runtime_bundle_count", "validated_at",
    }, "target trace validation")
    expected_validation = {
        "schema": TARGET_TRACE_VALIDATION_SCHEMA, "identity": identity, "valid": True,
        "qualification_identity": identity, "model_sha256": model_sha256,
        "manifest_sha256": base.sha256_file(manifest_path), "waves_sha256": base.sha256_file(paths["waves"]),
        "token_trace_sha256": _token_array_sha256(reference_tokens or []),
        "command_sha256": base.sha256_file(paths["command"]), "receipt_sha256": base.sha256_file(paths["receipt"]),
        "summary_sha256": base.sha256_file(output / "summary.json"),
        "runtime_bundle_sha256": base.sha256_json(components["runtime_bundle"]),
        "runtime_bundle_count": len(components["runtime_bundle"]),
    }
    for key, expected in expected_validation.items():
        if validation[key] != expected:
            raise ContractError(f"target trace validation mismatch: {key}")
    state = base.require_exact_keys(base.load_json(output / "run-state.json"), {
        "schema", "identity", "status", "completed_arms", "mode", "updated_at",
        "summary_sha256", "validation_sha256",
    }, "target trace run state")
    if (
        state["schema"] != STATE_SCHEMA or state["identity"] != identity or state["status"] != "completed"
        or state["completed_arms"] != ["target-only"] or state["mode"] != "target-only"
        or state["summary_sha256"] != expected_validation["summary_sha256"]
        or state["validation_sha256"] != base.sha256_file(output / "validation.json")
    ):
        raise ContractError("target trace run state is not complete or does not bind final evidence")
    result = {
        "schema": TARGET_TRACE_VALIDATION_SCHEMA, "valid": True, "identity": identity,
        "gpu_uuid": args.gpu_uuid, "output_dir": str(output),
    }
    print(base.canonical(result))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="CPU-only validation of a qualification plan")
    validate.add_argument("--plan", type=Path, required=True)
    validate.add_argument("--tier", choices=sorted(ALLOWED_TIERS), required=True)
    validate.set_defaults(handler=command_validate)
    run = commands.add_parser("run", help="run one guarded sequential qualification tier")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--tier", choices=sorted(ALLOWED_TIERS), required=True)
    run.add_argument("--server", type=Path, required=True)
    run.add_argument("--model", type=Path, required=True)
    run.add_argument("--draft-model", type=Path, required=True)
    run.add_argument("--profile", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--runtime-label", required=True)
    run.add_argument("--job-timeout-s", type=float, default=3600.0)
    run.add_argument("--arm-timeout-s", type=float, default=1200.0)
    run.add_argument("--readiness-timeout-s", type=float, default=300.0)
    run.add_argument("--http-timeout-s", type=float, default=900.0)
    run.add_argument("--stop-timeout-s", type=float, default=30.0)
    run.set_defaults(handler=command_run)
    trace = commands.add_parser("target-trace", help="run one guarded P1 target-only trace")
    trace.add_argument("--plan", type=Path, required=True)
    trace.add_argument("--server", type=Path, required=True)
    trace.add_argument("--model", type=Path, required=True)
    trace.add_argument("--output-dir", type=Path, required=True)
    trace.add_argument("--runtime-label", required=True)
    trace.add_argument("--job-timeout-s", type=float, default=1800.0)
    trace.add_argument("--arm-timeout-s", type=float, default=1200.0)
    trace.add_argument("--readiness-timeout-s", type=float, default=300.0)
    trace.add_argument("--http-timeout-s", type=float, default=900.0)
    trace.add_argument("--stop-timeout-s", type=float, default=30.0)
    trace.set_defaults(handler=command_target_trace)
    validate_trace = commands.add_parser("validate-target-trace", help="CPU-only validation of a completed P1 target trace")
    validate_trace.add_argument("--plan", type=Path, required=True)
    validate_trace.add_argument("--server", type=Path, required=True)
    validate_trace.add_argument("--model", type=Path, required=True)
    validate_trace.add_argument("--output-dir", type=Path, required=True)
    validate_trace.add_argument("--gpu-uuid", required=True)
    validate_trace.set_defaults(handler=command_validate_target_trace)
    return parser


def main() -> int:
    try:
        args = build_parser().parse_args()
        for name in ("job_timeout_s", "arm_timeout_s", "readiness_timeout_s", "http_timeout_s", "stop_timeout_s"):
            if hasattr(args, name):
                _finite(getattr(args, name), name, positive=True)
        return int(args.handler(args))
    except ContractError as error:
        print(f"qualification contract error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("qualification interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
