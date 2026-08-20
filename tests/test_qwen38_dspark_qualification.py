import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("qualification", SCRIPTS / "qwen38_dspark_qualification.py")
qualification = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(qualification)


def plan_value():
    arms = {
        "P1": [
            {"name": "target-only", "speculation": "none"},
            {"name": "static-dspark", "speculation": "draft-dspark", "n_max": 7, "dynamic_rs": False},
            {"name": "sps-execution", "speculation": "draft-dspark", "n_max": 7, "sps_profile": "p1/profile.json", "dynamic_rs": False},
        ],
        "P4": [
            {"name": "target-only", "speculation": "none"},
            {"name": "static-dspark-matched-cap3", "speculation": "draft-dspark", "n_max": 3, "dynamic_rs": False},
            {"name": "static-dspark-best-cap2", "speculation": "draft-dspark", "n_max": 2, "dynamic_rs": False},
            {"name": "sps-execution", "speculation": "draft-dspark", "n_max": 3, "sps_profile": "p4/profile.json", "dynamic_rs": False},
        ],
        "P8": [
            {"name": "target-only", "speculation": "none"},
            {"name": "static-dspark-cap1", "speculation": "draft-dspark", "n_max": 1, "dynamic_rs": False},
            {"name": "sps-execution", "speculation": "draft-dspark", "n_max": 1, "sps_profile": "p8/profile.json", "dynamic_rs": False},
        ],
    }
    return {
        "schema": qualification.PLAN_SCHEMA,
        "status": "test", "created_for": "CPU test",
        "execution_policy": {
            "gpu_processes_max": 1, "server_arms_sequential_only": True, "outer_guard_required": True,
            "port": 18136, "preflight_requires_port_free": True, "preflight_requires_zero_llama_processes": True,
            "postflight_requires_zero_llama_processes": True, "stop_on_guard_violation": True,
            "stop_on_any_http_or_server_error": True, "max_used_vram_mib": 27000,
            "min_free_vram_mib": 5000, "min_free_system_ram_gib": 18, "poll_interval_ms": 250,
        },
        "artifacts": {"server": "server.exe", "target_model": "target.gguf", "draft_model": "draft.gguf"},
        "common_server_args": {
            "host": "127.0.0.1", "port": 18136, "batch_size": 2048, "ubatch_size": 128,
            "target_kv": "q8_0", "draft_kv": "q8_0", "flash_attn": True, "n_gpu_layers": "all",
            "device": "CUDA0", "split_mode": "none", "fit": "off", "cache_ram": 0,
            "ctx_checkpoints": 0, "cache_idle_slots": False, "cache_prompt": False,
            "kv_unified": True, "metrics": True, "reasoning": "off", "log_verbosity": 4,
        },
        "deterministic_workload": {
            "endpoint": "/completion", "stream": True, "prompt_token_id": 1,
            "prompt_tokens_per_request": 736, "output_tokens_per_request": 256,
            "temperature": 0, "seed_base": 20260819, "seed_rule": "seed_base + client_index",
            "ignore_eos": True, "cache_prompt": False, "warmup_waves": 1, "measured_waves": 6,
            "request_barrier": True, "exact_stop_type": "limit", "truncated": False,
        },
        "tiers": [
            {"name": name, "parallel": p, "ctx_size": ctx, "profile_output": f"{name.lower()}/profile.json", "arms": arms[name]}
            for name, (p, ctx) in qualification.ALLOWED_TIERS.items()
        ],
        "correctness_gates": {}, "reported_metrics": [], "promotion_gates": {},
        "estimated_runtime_minutes": {},
    }


class FakeSse:
    def __init__(self, lines):
        self.lines = iter(lines)

    def readline(self, _limit):
        return next(self.lines, b"")


def valid_sse(token_override=None):
    tokens = list(range(256)) if token_override is None else token_override
    lines = [b": ping\r\n", b"\r\n"]
    for ordinal, token in enumerate(tokens, start=1):
        # server_task_result_cmpl_partial::to_json_non_oaicompat
        event = {
            "index": 0, "content": "" if ordinal % 61 == 0 else "x", "tokens": [token], "stop": False,
            "id_slot": 0, "tokens_predicted": ordinal, "tokens_evaluated": 736,
        }
        lines += [("data: " + json.dumps(event) + "\r\n").encode(), b"\r\n"]
    final = {
        # server_task_result_cmpl_final::to_json_non_oaicompat
        "index": 0, "content": "", "tokens": [], "id_slot": 0, "stop": True,
        "model": "qwen38-dspark-qualification", "generation_settings": {"n_predict": 256},
        "prompt": "", "has_new_line": False, "truncated": False, "stop_type": "limit",
        "stopping_word": "",
        # Real native /completion source semantics: this is the current slot
        # context length, not reused prompt tokens. With 256 generated tokens,
        # the final sampled token has not yet been inserted, hence 736+255.
        "tokens_predicted": 256, "tokens_evaluated": 736, "tokens_cached": 991,
        "timings": {"cache_n": 0, "prompt_n": 736, "predicted_n": 256, "predicted_ms": 1000.0, "predicted_per_second": 256.0},
    }
    lines += [("data: " + json.dumps(final) + "\n").encode(), b"\n"]
    return lines


class QualificationTests(unittest.TestCase):
    def test_server_environment_removes_all_llama_arg_contamination(self):
        environment = qualification.server_environment({
            "PATH": "test",
            "CUDA_VISIBLE_DEVICES": "GPU-test",
            "CUDA_LAUNCH_BLOCKING": "1",
            "GGML_CUDA_DISABLE_GRAPHS": "1",
            "LLAMA_TRACE": "1",
            "LLAMA_ARG_CTX_SIZE": "999999",
            "llama_arg_device": "none",
            "LLAMA_ARG_SPEC_DRAFT_DYNAMIC_RS": "1",
            "LLAMA_ARG_SPEC_DRAFT_ADAPTIVE": "1",
        })
        self.assertEqual(environment["PATH"], "test")
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "GPU-test")
        self.assertEqual(environment["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        self.assertNotIn("CUDA_LAUNCH_BLOCKING", environment)
        self.assertNotIn("GGML_CUDA_DISABLE_GRAPHS", environment)
        self.assertNotIn("LLAMA_TRACE", environment)
        self.assertNotIn("LLAMA_ARG_CTX_SIZE", environment)
        self.assertNotIn("llama_arg_device", environment)
        self.assertEqual(environment["LLAMA_ARG_SPEC_DRAFT_DYNAMIC_RS"], "0")
        self.assertEqual(environment["LLAMA_ARG_SPEC_DRAFT_ADAPTIVE"], "0")
        self.assertEqual(
            sorted(key for key in environment if key.upper().startswith("LLAMA_ARG_")),
            ["LLAMA_ARG_SPEC_DRAFT_ADAPTIVE", "LLAMA_ARG_SPEC_DRAFT_DYNAMIC_RS"],
        )
        self.assertEqual(
            qualification.server_environment_contract(environment),
            {
                "schema": "ai-loader-qwen38-server-environment/v1",
                "cuda_visible_devices": "GPU-test",
                "cuda_device_order": "PCI_BUS_ID",
                "dynamic_rs": "0",
                "adaptive_draft": "0",
                "scrubbed_prefixes": ["LLAMA_", "GGML_", "CUDA_"],
            },
        )

    def test_plan_requires_four_arm_p4_and_explicit_dynamic_false(self):
        value = plan_value()
        _, tier = qualification.validate_plan(value, Path("plan.json"), "P4")
        self.assertEqual([arm["name"] for arm in tier["arms"]], qualification.ARM_NAMES["P4"])
        value["tiers"][1]["arms"][1].pop("dynamic_rs")
        with self.assertRaisesRegex(qualification.ContractError, "dynamic_rs=false"):
            qualification.validate_plan(value, Path("plan.json"), "P4")

    def test_plan_ram_floor_is_exactly_eighteen_gib(self):
        value = plan_value()
        qualification.validate_plan(value, Path("plan.json"), "P1")
        value["execution_policy"]["min_free_system_ram_gib"] = 17
        with self.assertRaisesRegex(qualification.ContractError, "must be 18"):
            qualification.validate_plan(value, Path("plan.json"), "P1")

    def test_target_has_no_spec_flags_and_dspark_is_explicitly_static_rs(self):
        class Args: pass
        args = Args()
        args.server, args.model, args.draft_model, args.profile = map(Path, ("server.exe", "target.gguf", "draft.gguf", "profile.json"))
        args.port = 18136
        args.plan_value = plan_value()
        tier = args.plan_value["tiers"][1]
        target = qualification.server_arguments(args, tier, tier["arms"][0])
        self.assertFalse(any(item.startswith("--spec") for item in target))
        static = qualification.server_arguments(args, tier, tier["arms"][1])
        self.assertIn("--no-spec-draft-dynamic-rs", static)
        self.assertIn("--no-spec-draft-adaptive", static)
        self.assertNotIn("--spec-draft-sps-profile", static)
        sps = qualification.server_arguments(args, tier, tier["arms"][-1])
        self.assertIn("--spec-draft-sps-profile", sps)

    def test_props_proves_exact_unified_kv_tier_context(self):
        props = {
            "total_slots": 8, "endpoint_metrics": True,
            "default_generation_settings": {"n_ctx": 8192},
            "model_path": str(Path("target.gguf").resolve()),
            "model_alias": "qwen38-dspark-qualification",
        }
        tier = plan_value()["tiers"][2]
        qualification._validate_props(props, tier, Path("target.gguf"))
        props["default_generation_settings"]["n_ctx"] = 1024
        with self.assertRaisesRegex(qualification.ContractError, "context"):
            qualification._validate_props(props, tier, Path("target.gguf"))

    def test_stream_accumulates_exact_raw_tokens_and_ttft(self):
        clock = iter((1.0, 1.125, 2.0)).__next__
        result = qualification.parse_completion_sse(FakeSse(valid_sse()), clock)
        self.assertEqual(result["tokens"], list(range(256)))
        self.assertEqual(result["ttft_ms"], 125.0)
        self.assertEqual(result["wall_ms"], 1000.0)

    def test_stream_rejects_wrong_final_accounting_and_duplicate_json_keys(self):
        lines = valid_sse()
        final_index = -2
        final = json.loads(lines[final_index][6:])
        final["timings"]["cache_n"] = 1
        lines[final_index] = ("data: " + json.dumps(final) + "\n").encode()
        with self.assertRaises(qualification.CompletionContractError) as caught:
            qualification.parse_completion_sse(FakeSse(lines))
        diagnostic = caught.exception.diagnostic
        self.assertEqual(diagnostic["failed_predicates"], ["timings_cache_n_is_0"])
        self.assertEqual(diagnostic["observed"]["raw_partial_token_count"], 256)
        self.assertEqual(diagnostic["observed"]["tokens_cached"], 991)
        self.assertNotIn("tokens", diagnostic["observed"])
        self.assertNotIn("content", diagnostic["observed"])
        bad = [b'data: {"stop":false,"stop":false,"tokens":[1],"content":"x"}\n', b"\n"]
        with self.assertRaisesRegex(qualification.ContractError, "duplicate"):
            qualification.parse_completion_sse(FakeSse(bad))

    def test_stream_rejects_empty_nonterminal_with_content_free_diagnostic(self):
        empty_partial = {
            "index": 0, "content": "", "tokens": [], "stop": False,
            "id_slot": 0, "tokens_predicted": 0, "tokens_evaluated": 736,
        }
        lines = [("data: " + json.dumps(empty_partial) + "\n").encode(), b"\n"]
        with self.assertRaises(qualification.CompletionContractError) as caught:
            qualification.parse_completion_sse(FakeSse(lines))
        diagnostic = caught.exception.diagnostic
        self.assertEqual(diagnostic["failed_predicates"], ["nonterminal_event_contains_raw_token_ids"])
        self.assertEqual(diagnostic["observed"]["partial_token_count"], 0)
        self.assertEqual(diagnostic["observed"]["partial_content_char_count"], 0)
        self.assertNotIn("tokens", diagnostic["observed"])
        self.assertNotIn("content", diagnostic["observed"])

    def test_stream_parser_diagnostics_never_copy_unsupported_line_content(self):
        secret = "generated-secret-text"
        with self.assertRaises(qualification.CompletionContractError) as caught:
            qualification.parse_completion_sse(FakeSse([(secret + "\n").encode()]))
        diagnostic_text = json.dumps(caught.exception.diagnostic, sort_keys=True)
        self.assertNotIn(secret, diagnostic_text)
        self.assertFalse(caught.exception.diagnostic["observed"]["has_field_separator"])

        safe_unknown = b"unknown_field: generated-secret-text\n"
        with self.assertRaises(qualification.CompletionContractError) as caught:
            qualification.parse_completion_sse(FakeSse([safe_unknown]))
        observed = caught.exception.diagnostic["observed"]
        self.assertEqual(observed["safe_ascii_field_name"], "unknown_field")
        self.assertNotIn("generated-secret-text", json.dumps(observed, sort_keys=True))

    def test_stream_size_limit_has_structured_content_free_diagnostic(self):
        secret = b"secret-payload-without-sse-fields\n"
        with mock.patch.object(qualification, "MAX_SSE_BYTES", 8):
            with self.assertRaises(qualification.CompletionContractError) as caught:
                qualification.parse_completion_sse(FakeSse([secret]))
        diagnostic = caught.exception.diagnostic
        self.assertEqual(diagnostic["failed_predicates"], ["sse_response_is_within_size_bounds"])
        self.assertNotIn("secret", json.dumps(diagnostic, sort_keys=True))

    def test_completion_diagnostic_is_atomically_persisted_in_failed_state(self):
        error = qualification.completion_error(
            ["tokens_predicted_is_256"],
            {"raw_partial_token_count": 255, "tokens_predicted": 255},
        )
        state = {
            "schema": qualification.STATE_SCHEMA, "identity": "test",
            "status": "running", "completed_arms": [], "updated_at": "old",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run-state.json"
            qualification._persist_failed_state(path, state, error)
            persisted = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(persisted["status"], "failed")
        self.assertEqual(persisted["error_type"], "CompletionContractError")
        self.assertEqual(persisted["completion_diagnostic"], error.diagnostic)
        self.assertNotIn("tokens", persisted["completion_diagnostic"]["observed"])
        self.assertNotIn("content", persisted["completion_diagnostic"]["observed"])

    def test_exact_parity_checks_full_tokens_and_repeat_waves(self):
        client = {"client": 0, "seed": 20260819, "tokens": [1, 2], "content_sha256": "a"}
        arm = {"waves": [{"clients": [dict(client)]} for _ in range(7)]}
        self.assertTrue(qualification.exact_parity([arm, arm])["exact_token_and_content_parity"])
        broken = {"waves": [{"clients": [dict(client)]} for _ in range(7)]}
        broken["waves"][4]["clients"][0]["tokens"] = [1, 3]
        with self.assertRaises(qualification.ParityContractError) as caught:
            qualification.exact_parity([broken, arm])
        self.assertEqual(caught.exception.diagnostic["first_difference_index"], 1)
        self.assertEqual(caught.exception.diagnostic["difference_count"], 1)
        self.assertNotIn("tokens", caught.exception.diagnostic)
        self.assertNotIn("content", caught.exception.diagnostic)

        state = {
            "schema": qualification.STATE_SCHEMA, "identity": "test",
            "status": "running", "completed_arms": [], "updated_at": "old",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run-state.json"
            qualification._persist_failed_state(path, state, caught.exception)
            persisted = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(persisted["parity_diagnostic"], caught.exception.diagnostic)

    def test_metric_parser_and_bootstrap_fail_closed_and_deterministic(self):
        text = "\n".join(f"{metric} 0" for metric in qualification.REQUIRED_METRICS.values())
        parsed = qualification.metric_snapshot(text)
        self.assertEqual(parsed["sps_fallback"], 0)
        with self.assertRaisesRegex(qualification.ContractError, "missing"):
            qualification.metric_snapshot("llamacpp:tokens_predicted_total 0")
        left = qualification.paired_bootstrap_ci([110, 111, 112, 113, 114, 115], [100] * 6, resamples=500)
        right = qualification.paired_bootstrap_ci([110, 111, 112, 113, 114, 115], [100] * 6, resamples=500)
        self.assertEqual(left, right)
        self.assertGreater(left["ci95_percent"][0], 0)

    def test_static_arm_requires_exact_fixed_recurrent_depth_gauges(self):
        metrics = {name: 0.0 for name in qualification.REQUIRED_METRICS}
        metrics.update({
            "tokens": 256.0, "decode_seconds": 1.0, "prompt_tokens": 736.0,
            "draft_tokens": 8.0, "accepted_tokens": 4.0, "draft_steps": 1.0,
            "rs_resident_depth": 7.0, "rs_configured_depth": 7.0,
            "rs_required_depth": 7.0, "rs_pending_depth": -1.0,
        })
        client = {"ttft_ms": 1.0, "wall_ms": 1000.0}
        waves = [
            {"metric_delta": dict(metrics), "goodput_tps": 256.0, "clients": [dict(client)]}
            for _ in range(7)
        ]
        counters = {name: 0.0 for name in qualification.REQUIRED_METRICS}
        arm = {"name": "static-dspark", "speculation": "draft-dspark", "n_max": 7}
        summary = qualification._summarize_arm(arm, 1, waves, counters, counters)
        self.assertEqual(summary["measured_metric_delta"]["rs_required_depth"], 7.0)

        waves[1]["metric_delta"]["rs_required_depth"] = 0.0
        with self.assertRaisesRegex(qualification.ContractError, "fixed tier storage"):
            qualification._summarize_arm(arm, 1, waves, counters, counters)
        waves[1]["metric_delta"]["rs_required_depth"] = 7.0
        waves[1]["metric_delta"]["rs_pending_depth"] = 7.0
        with self.assertRaisesRegex(qualification.ContractError, "fixed tier storage"):
            qualification._summarize_arm(arm, 1, waves, counters, counters)

    def test_profile_requires_exact_active_and_full_cap_rows(self):
        tier = plan_value()["tiers"][1]
        entries = [
            {"context_tokens": 1024, "active_slots": 4, "total_verify_rows": rows, "cost_us": 100 + rows}
            for rows in (4, 8, 12, 16)
        ]
        qualification.validate_profile({"schema_version": 2, "max_draft_tokens_per_slot": 3, "entries": entries}, tier)
        with self.assertRaisesRegex(qualification.ContractError, "cover"):
            qualification.validate_profile({"schema_version": 2, "max_draft_tokens_per_slot": 3, "entries": entries[:-1]}, tier)

    def test_fake_endpoint_proves_barrier_overlap_and_identical_payload_contract(self):
        parallel = 4
        before = {name: 0.0 for name in qualification.REQUIRED_METRICS}
        before["rs_pending_depth"] = -1.0
        after = dict(before)
        after.update({"tokens": 1024.0, "prompt_tokens": 2944.0, "decode_seconds": 4.0})
        bodies = []
        body_lock = threading.Lock()

        class Connections:
            def close_all(self): pass
            def completion(self, _url, body, _timeout):
                with body_lock:
                    bodies.append(body)
                time.sleep(0.02)
                tokens = list(range(256))
                return {
                    "tokens": tokens, "token_sha256": "t", "content_sha256": "c",
                    "ttft_ms": 1.0, "wall_ms": 20.0, "predicted_ms": 19.0,
                    "predicted_tps": 256.0 / 0.019, "generated_tokens": 256,
                    "stop_type": "limit", "truncated": False,
                }

        class Process:
            def poll(self): return None

        workload = plan_value()["deterministic_workload"]
        with mock.patch.object(qualification, "AbortableSseConnections", Connections), mock.patch.object(
            qualification, "_read_metrics", side_effect=[("before", before), ("after", after)]
        ):
            wave, _, _, _ = qualification.run_wave(
                base_url="http://127.0.0.1:18136", parallel=parallel, wave=2,
                workload=workload, timeout_s=2.0, guard_check=lambda: "GPU-test", process=Process(),
            )
        self.assertTrue(wave["request_overlap_proven"])
        self.assertEqual(len(bodies), parallel)
        for client, body in enumerate(sorted(bodies, key=lambda item: item["seed"])):
            self.assertEqual(body["prompt"], [1] * 736)
            self.assertEqual(body["seed"], 20260819 + client)
            self.assertTrue(body["stream"] and body["return_tokens"] and body["ignore_eos"])
            self.assertFalse(body["cache_prompt"])

    def test_target_trace_reuses_one_p1_target_arm_without_draft_or_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            server = root / "server.exe"
            model = root / "target.gguf"
            server.write_bytes(b"server")
            model.write_bytes(b"model")
            value = plan_value()
            value["artifacts"] = {
                "server": str(server), "target_model": str(model),
                "draft_model": str(root / "never-read.gguf"),
                "target_model_sha256": qualification.base.sha256_file(model),
                "draft_model_sha256": "0" * 64,
            }
            plan = root / "plan.json"
            plan.write_text(json.dumps(value), encoding="utf-8")
            output = root / "p1-final-v5"
            args = qualification.build_parser().parse_args([
                "target-trace", "--plan", str(plan), "--server", str(server),
                "--model", str(model), "--output-dir", str(output),
                "--runtime-label", "cpu-test",
            ])
            calls = []
            runtime_bundle = [
                {"name": name, "sha256": "a" * 64}
                for name in sorted(qualification.TARGET_TRACE_REQUIRED_RUNTIME_DLLS)
            ]

            def fake_run_arm(run_args, tier, arm, guard_uuid, _deadline):
                calls.append((tier["name"], dict(arm), guard_uuid))
                paths = qualification._arm_paths(run_args.output_dir, arm["name"])
                paths["root"].mkdir(parents=True)
                tokens = list(range(256))
                token_sha = qualification._token_array_sha256(tokens)
                delta = {name: 0.0 for name in qualification.REQUIRED_METRICS}
                delta.update({
                    "tokens": 256.0, "decode_seconds": 1.0, "prompt_tokens": 736.0,
                    "rs_pending_depth": -1.0,
                })
                waves = [{
                    "wave": wave, "parallel": 1, "wall_ms": 1000.0, "goodput_tps": 256.0,
                    "request_overlap_proven": True, "metric_delta": dict(delta),
                    "clients": [{
                        "client": 0, "seed": 20260819, "wave": wave, "tokens": tokens,
                        "token_sha256": token_sha, "content_sha256": "c" * 64,
                        "ttft_ms": 10.0, "wall_ms": 1000.0, "predicted_ms": 900.0,
                        "predicted_tps": 256.0 / 0.9, "generated_tokens": 256,
                        "stop_type": "limit", "truncated": False,
                    }],
                } for wave in range(7)]
                def metrics_text(multiplier):
                    values = {name: 0.0 for name in qualification.REQUIRED_METRICS}
                    values.update({
                        "tokens": 256.0 * multiplier, "decode_seconds": float(multiplier),
                        "prompt_tokens": 736.0 * multiplier, "rs_pending_depth": -1.0,
                    })
                    return "\n".join(
                        f"{qualification.REQUIRED_METRICS[name]} {values[name]}"
                        for name in qualification.REQUIRED_METRICS
                    ) + "\n"
                snapshots = [{"before": metrics_text(wave), "after": metrics_text(wave + 1)} for wave in range(7)]
                qualification.base.write_json(paths["command"], {
                    "schema": qualification.ARM_SCHEMA, "tier": "P1", "arm": "target-only",
                    "dynamic_rs": False, "arguments": qualification.server_arguments(run_args, tier, arm),
                    "command_sha256": qualification.base.sha256_json(qualification.server_arguments(run_args, tier, arm)),
                }, False)
                qualification.base.write_json(paths["props"], {
                    "total_slots": 1, "endpoint_metrics": True,
                    "default_generation_settings": {"n_ctx": 2048},
                    "model_path": str(run_args.model.resolve()),
                    "model_alias": "qwen38-dspark-qualification",
                }, False)
                qualification.base.write_text(paths["metrics_before"], metrics_text(0), False)
                qualification.base.write_text(paths["metrics_after"], metrics_text(7), False)
                qualification.base.write_json(paths["waves"], {
                    "schema": qualification.ARM_SCHEMA, "waves": waves, "metric_snapshots": snapshots,
                }, False)
                arm_summary = qualification._summarize_arm(
                    arm, 1, waves, qualification.metric_snapshot(metrics_text(0)),
                    qualification.metric_snapshot(metrics_text(7)),
                )
                qualification.base.write_json(paths["summary"], arm_summary, False)
                qualification.base.write_text(paths["stdout"], "", False)
                qualification.base.write_text(paths["stderr"], "", False)
                bound_files = ["command", "props", "metrics_before", "metrics_after", "waves", "summary", "stdout", "stderr"]
                qualification.base.write_json(paths["receipt"], {
                    "schema": qualification.ARM_SCHEMA, "tier": "P1", "arm": "target-only",
                    "dynamic_rs": False,
                    "artifacts": {
                        key: {"path": paths[key].name, "sha256": qualification.base.sha256_file(paths[key])}
                        for key in bound_files
                    }, "completed_at": "test",
                }, False)
                return {"summary": arm_summary, "waves": waves}

            clean_environment = {
                "PATH": "test", "CUDA_VISIBLE_DEVICES": "GPU-TEST",
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "LLAMA_ARG_SPEC_DRAFT_DYNAMIC_RS": "0",
                "LLAMA_ARG_SPEC_DRAFT_ADAPTIVE": "0",
            }
            with mock.patch.object(qualification.base, "verify_guard_lease", return_value="guard-test"), mock.patch.object(
                qualification.base, "gpu_identity", return_value={
                    "uuid": "GPU-TEST", "name": "test", "driver_version": "test",
                    "compute_capability": "test", "cuda_driver_runtime": "test",
                }
            ), mock.patch.object(
                qualification.base, "source_commit", return_value="dirty-test"
            ), mock.patch.object(
                qualification.base, "runtime_bundle_identity", return_value=runtime_bundle
            ), mock.patch.object(
                qualification.base, "port_is_open", return_value=False
            ), mock.patch.object(
                qualification, "server_environment", return_value=clean_environment
            ), mock.patch.object(qualification, "run_arm", side_effect=fake_run_arm):
                self.assertEqual(args.handler(args), 0)
                validate_args = qualification.build_parser().parse_args([
                    "validate-target-trace", "--plan", str(plan), "--server", str(server),
                    "--model", str(model), "--output-dir", str(output), "--gpu-uuid", "GPU-TEST",
                ])
                self.assertEqual(validate_args.handler(validate_args), 0)

            self.assertEqual(calls, [("P1", {"name": "target-only", "speculation": "none"}, "guard-test")])
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["components"]["trace_mode"], "target-only")
            self.assertEqual([arm["name"] for arm in manifest["components"]["arm_arguments"]], ["target-only"])
            self.assertNotIn("draft_model_sha256", manifest["components"])
            self.assertNotIn("profile_sha256", manifest["components"])
            validation = json.loads((output / "validation.json").read_text(encoding="utf-8"))
            self.assertTrue(validation["valid"])
            self.assertEqual(validation["token_trace_sha256"], qualification._token_array_sha256(list(range(256))))

            receipt_path = output / "arms" / "target-only" / "receipt.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["artifacts"].pop("stderr")
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            with mock.patch.object(qualification.base, "source_commit", return_value="dirty-test"), mock.patch.object(
                qualification.base, "runtime_bundle_identity", return_value=runtime_bundle
            ):
                with self.assertRaisesRegex(qualification.ContractError, "keys mismatch"):
                    qualification.command_validate_target_trace(validate_args)

    def test_target_trace_parser_has_no_draft_or_profile_inputs(self):
        args = qualification.build_parser().parse_args([
            "target-trace", "--plan", "plan.json", "--server", "server.exe",
            "--model", "model.gguf", "--output-dir", "p1-final-v5",
            "--runtime-label", "test",
        ])
        self.assertEqual(args.handler, qualification.command_target_trace)
        self.assertFalse(hasattr(args, "draft_model"))
        self.assertFalse(hasattr(args, "profile"))


if __name__ == "__main__":
    unittest.main()
