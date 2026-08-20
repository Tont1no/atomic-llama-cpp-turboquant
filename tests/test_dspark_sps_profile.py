#!/usr/bin/env python3

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "dspark_sps_profile.py"
SPEC = importlib.util.spec_from_file_location("dspark_sps_profile", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
profile = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(profile)


class DsparkSpsProfilePlanTests(unittest.TestCase):
    def make_plan(self, **overrides):
        values = {
            "seed": 20260818,
            "active_slots": [1, 2, 4, 8],
            "context_ceilings": [2048, 4096],
            "prefix_caps": [0, 1, 2, 3, 7],
            "max_draft_tokens_per_slot": 7,
            "repetitions": 4,
            "samples_per_cell": 64,
            "warmup_shape_runs": 3,
        }
        values.update(overrides)
        return profile.build_plan(**values)

    def test_plan_is_deterministic(self):
        first = self.make_plan()
        second = self.make_plan()
        self.assertEqual(profile.canonical(first), profile.canonical(second))
        self.assertEqual(first["schedule_sha256"], profile.sha256_json(first["schedule"]))
        self.assertEqual(len(first["cells"]), 4 * 2 * 5)
        self.assertEqual(len(first["schedule"]), len(first["cells"]) * 4)
        profile.validate_plan(first)

    def test_each_repetition_contains_every_cell_once(self):
        plan = self.make_plan()
        expected = {cell["cell_id"] for cell in plan["cells"]}
        for repetition in range(1, plan["repetitions"] + 1):
            observed = [
                entry["cell_id"]
                for entry in plan["schedule"]
                if entry["repetition"] == repetition
            ]
            self.assertEqual(len(observed), len(expected))
            self.assertEqual(set(observed), expected)

    def test_each_pair_is_position_reversed(self):
        plan = self.make_plan(repetitions=6)
        for first_repetition in (1, 3, 5):
            forward = [
                entry["cell_id"]
                for entry in plan["schedule"]
                if entry["repetition"] == first_repetition
            ]
            reverse = [
                entry["cell_id"]
                for entry in plan["schedule"]
                if entry["repetition"] == first_repetition + 1
            ]
            self.assertEqual(reverse, list(reversed(forward)))

    def test_pair_rotates_between_repetitions(self):
        plan = self.make_plan(repetitions=4)
        first = [
            entry["cell_id"]
            for entry in plan["schedule"]
            if entry["repetition"] == 1
        ]
        third = [
            entry["cell_id"]
            for entry in plan["schedule"]
            if entry["repetition"] == 3
        ]
        self.assertEqual(third, first[1:] + first[:1])

    def test_row_axes_are_physical_per_active(self):
        plan = self.make_plan(context_ceilings=[2048])
        self.assertEqual(plan["row_axes_by_active"]["1"], [1, 2, 3, 4, 8])
        self.assertEqual(plan["row_axes_by_active"]["2"], [2, 4, 6, 8, 16])
        self.assertEqual(plan["row_axes_by_active"]["4"], [4, 8, 12, 16, 32])
        self.assertEqual(plan["row_axes_by_active"]["8"], [8, 16, 24, 32, 64])

    def test_odd_repetitions_fail(self):
        with self.assertRaisesRegex(profile.ContractError, "even"):
            self.make_plan(repetitions=3)

    def test_missing_target_only_or_full_cap_fails(self):
        with self.assertRaisesRegex(profile.ContractError, "target-only"):
            self.make_plan(prefix_caps=[1, 2, 7])
        with self.assertRaisesRegex(profile.ContractError, "max_draft_tokens_per_slot"):
            self.make_plan(prefix_caps=[0, 1, 3])

    def test_tampering_and_unknown_fields_fail(self):
        plan = self.make_plan()
        tampered = copy.deepcopy(plan)
        tampered["schedule"][0]["cell_id"] = tampered["schedule"][1]["cell_id"]
        with self.assertRaisesRegex(profile.ContractError, "schedule_sha256"):
            profile.validate_plan(tampered)

        unknown = copy.deepcopy(plan)
        unknown["extra"] = True
        with self.assertRaisesRegex(profile.ContractError, "unknown"):
            profile.validate_plan(unknown)

    def test_duplicate_json_member_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"schema":"a","schema":"b"}', encoding="utf-8")
            with self.assertRaisesRegex(profile.ContractError, "duplicate"):
                profile.load_json(path)

    def test_cli_plan_and_validate_are_byte_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = [root / "first.json", root / "second.json"]
            for output in outputs:
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(SCRIPT),
                        "plan",
                        "--output",
                        str(output),
                        "--context-ceilings",
                        "2048,4096",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(outputs[0].read_bytes(), outputs[1].read_bytes())

            validated = subprocess.run(
                [sys.executable, str(SCRIPT), "validate", "--plan", str(outputs[0])],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(validated.returncode, 0, validated.stderr)
            receipt = json.loads(validated.stdout)
            self.assertTrue(receipt["ok"])
            self.assertEqual(receipt["cells"], 40)

    def test_run_is_fail_closed_without_guard_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = root / "plan.json"
            plan_path.write_text(
                json.dumps(self.make_plan(context_ceilings=[2048])), encoding="utf-8"
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "run",
                    "--plan",
                    str(plan_path),
                    "--server",
                    str(root / "missing-server.exe"),
                    "--model",
                    str(root / "missing-model.gguf"),
                    "--draft-model",
                    str(root / "missing-draft.gguf"),
                    "--output-dir",
                    str(root / "output"),
                    "--runtime-label",
                    "mock-cuda",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("requires a complete exclusive GPU guard proof", completed.stderr)
            self.assertFalse((root / "output").exists())


class DsparkSpsRunnerTests(unittest.TestCase):
    SOURCE_ID = "a" * 64

    def make_plan(self):
        return profile.build_plan(
            seed=20260818,
            active_slots=[1],
            context_ceilings=[2048],
            prefix_caps=[0, 7],
            max_draft_tokens_per_slot=7,
            repetitions=2,
            samples_per_cell=2,
            warmup_shape_runs=3,
        )

    def make_args(self, root: Path, plan_path: Path):
        for name in ("server.exe", "model.gguf", "draft.gguf", "ggml-cuda.dll"):
            path = root / name
            if not path.exists():
                path.write_bytes(name.encode("ascii"))
        return profile.build_parser().parse_args(
            [
                "run", "--plan", str(plan_path),
                "--server", str(root / "server.exe"),
                "--model", str(root / "model.gguf"),
                "--draft-model", str(root / "draft.gguf"),
                "--output-dir", str(root / "output"),
                "--runtime-label", "mock-cuda",
                "--samples-per-cell", "2",
                "--warmup-shape-runs", "3",
            ]
        )

    def write_arm_artifacts(self, output_dir, source_identity, entry, cell, args):
        paths = profile.arm_paths(output_dir, entry["ordinal"])
        paths["root"].mkdir(parents=True, exist_ok=True)
        command = {
            "schema": profile.ARM_SCHEMA,
            "source_identity": source_identity,
            "ordinal": entry["ordinal"],
            "cell": cell,
            "arguments": profile.server_arguments(args, cell, paths["record"], source_identity),
        }
        samples = [
            {
                "actual_context_tokens": cell["context_ceiling"] - 1 + index,
                "cost_us": float(10 + entry["ordinal"] + index),
                "execution_kind": "graph_replay",
                "prefixes": [cell["prefix_cap"]] * cell["active_slots"],
            }
            for index in range(args.samples_per_cell)
        ]
        coordinate = {
            "context_tokens": cell["context_ceiling"],
            "active_slots": cell["active_slots"],
            "total_verify_rows": cell["total_verify_rows"],
            "ready": True,
            "retained_samples": len(samples),
            "warmup_seen": args.warmup_shape_runs,
            "skipped_warmup": args.warmup_shape_runs,
            "skipped_capture": 0,
            "skipped_ineligible": 0,
            "skipped_full": 0,
            "p50_us": samples[0]["cost_us"],
            "p95_us": samples[-1]["cost_us"],
            "max_us": samples[-1]["cost_us"],
            "samples": samples,
        }
        sidecar = {
            "schema_version": 1,
            "profile_schema_version": 2,
            "source_identity": source_identity,
            "complete": True,
            "max_draft_tokens_per_slot": args.max_draft_tokens_per_slot,
            "context_buckets": [cell["context_ceiling"]],
            "verify_rows_by_active": {str(cell["active_slots"]): [cell["total_verify_rows"]]},
            "retained_samples_per_coordinate": args.samples_per_cell,
            "warmup_samples_per_coordinate": args.warmup_shape_runs,
            "aggregate_quantile": 0.95,
            "observations_total": len(samples) + args.warmup_shape_runs,
            "retained_total": len(samples),
            "skipped_warmup_total": args.warmup_shape_runs,
            "skipped_capture_total": 0,
            "skipped_ineligible_total": 0,
            "skipped_outside_grid_total": 0,
            "skipped_full_total": 0,
            "ready_coordinates": 1,
            "expected_coordinates": 1,
            "coordinates": [coordinate],
        }
        record = {
            "schema_version": 2,
            "max_draft_tokens_per_slot": args.max_draft_tokens_per_slot,
            "entries": [{
                "context_tokens": cell["context_ceiling"],
                "active_slots": cell["active_slots"],
                "total_verify_rows": cell["total_verify_rows"],
                "cost_us": samples[-1]["cost_us"],
            }],
        }
        client = {
            "schema": profile.ARM_SCHEMA,
            "ordinal": entry["ordinal"],
            "cell_id": cell["cell_id"],
            "active_slots": cell["active_slots"],
            "prompt_token_id": args.prompt_token_id,
            "prompt_tokens": 100,
            "output_tokens": args.output_tokens,
            "common_wall_ms": 100.0,
            "requests": [{
                "seed": 1,
                "prompt_tokens": 100,
                "generated_tokens": args.output_tokens,
                "wall_ms": 100.0,
                "predicted_ms": 90.0,
                "predicted_per_second": 10.0,
                "content_sha256": "b" * 64,
            }],
        }
        profile.write_json(paths["sidecar"], sidecar, True)
        profile.write_json(paths["record"], record, True)
        profile.write_json(paths["client"], client, True)
        profile.write_json(paths["command"], command, True)
        profile.write_json(paths["props"], {
            "total_slots": args.parallel,
            "endpoint_metrics": True,
            "default_generation_settings": {"n_ctx": args.ctx_size // args.parallel},
        }, True)
        metric_text = "llamacpp:spec_decode_sps_record_retained_samples_total 2\n"
        profile.write_text(paths["metrics_before"], metric_text, True)
        profile.write_text(paths["metrics_after"], metric_text, True)
        receipt = {
            "schema": profile.ARM_SCHEMA,
            "source_identity": source_identity,
            "ordinal": entry["ordinal"],
            "cell_id": cell["cell_id"],
            "status": "completed",
            "command_sha256": profile.sha256_file(paths["command"]),
            "props_sha256": profile.sha256_file(paths["props"]),
            "metrics_before_sha256": profile.sha256_file(paths["metrics_before"]),
            "metrics_after_sha256": profile.sha256_file(paths["metrics_after"]),
            "record_sha256": profile.sha256_file(paths["record"]),
            "sidecar_sha256": profile.sha256_file(paths["sidecar"]),
            "client_sha256": profile.sha256_file(paths["client"]),
            "completed_at": profile.utc_now(),
        }
        profile.write_json(paths["receipt"], receipt, True)

    def test_guarded_mock_run_merges_and_resume_revalidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self.make_plan()
            plan_path = root / "plan.json"
            profile.write_json(plan_path, plan, False)
            args = self.make_args(root, plan_path)
            cells = {cell["cell_id"]: cell for cell in plan["cells"]}
            calls = []

            def fake_run_arm(**kwargs):
                calls.append(kwargs["entry"]["ordinal"])
                self.write_arm_artifacts(
                    kwargs["output_dir"], kwargs["source_identity_value"],
                    kwargs["entry"], kwargs["cell"], kwargs["args"],
                )

            guard_env = {
                profile.GUARD_MARKER_ENV: "1",
                profile.GUARD_UUID_ENV: "GPU-MOCK",
                "CUDA_VISIBLE_DEVICES": "GPU-MOCK",
            }
            with mock.patch.dict(profile.os.environ, guard_env, clear=False), \
                    mock.patch.object(profile, "verify_guard_lease", return_value="GPU-MOCK"), \
                    mock.patch.object(profile, "gpu_identity", return_value={
                        "uuid": "GPU-MOCK", "name": "Mock", "driver_version": "1", "compute_capability": "12.0"
                    }), \
                    mock.patch.object(profile, "source_commit", return_value="c" * 40), \
                    mock.patch.object(profile, "run_arm", side_effect=fake_run_arm):
                self.assertEqual(profile.command_run(args), 0)
            self.assertEqual(calls, list(range(len(plan["schedule"]))))
            output = root / "output"
            final_profile = profile.load_json(output / "profile.json")
            profile.validate_final_profile(final_profile, plan)
            self.assertEqual(profile.load_json(output / "summary.json")["status"], "completed")

            calls.clear()
            args = self.make_args(root, plan_path)
            with mock.patch.dict(profile.os.environ, guard_env, clear=False), \
                    mock.patch.object(profile, "verify_guard_lease", return_value="GPU-MOCK"), \
                    mock.patch.object(profile, "gpu_identity", return_value={
                        "uuid": "GPU-MOCK", "name": "Mock", "driver_version": "1", "compute_capability": "12.0"
                    }), \
                    mock.patch.object(profile, "source_commit", return_value="c" * 40), \
                    mock.patch.object(profile, "run_arm", side_effect=AssertionError("completed arms must be skipped")):
                self.assertEqual(profile.command_run(args), 0)
            self.assertEqual(calls, [])

    def test_resume_rejects_corrupt_completed_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self.make_plan()
            plan_path = root / "plan.json"
            profile.write_json(plan_path, plan, False)
            args = self.make_args(root, plan_path)
            args.output_tokens = (2 + 3 + 16) * 8
            output = root / "output"
            output.mkdir()
            # A state may never cause a completed arm to be trusted without its artifacts.
            state = profile.new_state(self.SOURCE_ID, plan)
            state["completed_ordinals"] = [0]
            state["next_ordinal"] = 1
            with self.assertRaises(profile.ContractError):
                profile.validate_completed_arm(
                    profile.arm_paths(output, 0),
                    entry=plan["schedule"][0],
                    cell={cell["cell_id"]: cell for cell in plan["cells"]}[plan["schedule"][0]["cell_id"]],
                    source_identity_value=self.SOURCE_ID,
                    args=args,
                )

    def test_resume_rejects_changed_runtime_dll_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self.make_plan()
            plan_path = root / "plan.json"
            profile.write_json(plan_path, plan, False)
            args = self.make_args(root, plan_path)

            def fake_run_arm(**kwargs):
                self.write_arm_artifacts(
                    kwargs["output_dir"], kwargs["source_identity_value"],
                    kwargs["entry"], kwargs["cell"], kwargs["args"],
                )

            shared_patches = (
                mock.patch.object(profile, "verify_guard_lease", return_value="GPU-MOCK"),
                mock.patch.object(profile, "gpu_identity", return_value={
                    "uuid": "GPU-MOCK", "name": "Mock", "driver_version": "1",
                    "compute_capability": "12.0", "cuda_driver_runtime": "13.0",
                }),
                mock.patch.object(profile, "source_commit", return_value="c" * 40),
            )
            with shared_patches[0], shared_patches[1], shared_patches[2], \
                    mock.patch.object(profile, "run_arm", side_effect=fake_run_arm):
                self.assertEqual(profile.command_run(args), 0)

            (root / "cublasLt64_13.dll").write_bytes(b"changed-runtime")
            args = self.make_args(root, plan_path)
            with mock.patch.object(profile, "verify_guard_lease", return_value="GPU-MOCK"), \
                    mock.patch.object(profile, "gpu_identity", return_value={
                        "uuid": "GPU-MOCK", "name": "Mock", "driver_version": "1",
                        "compute_capability": "12.0", "cuda_driver_runtime": "13.0",
                    }), \
                    mock.patch.object(profile, "source_commit", return_value="c" * 40), \
                    mock.patch.object(profile, "run_arm", side_effect=AssertionError("must fail before resume")):
                with self.assertRaisesRegex(profile.ContractError, "manifest is incompatible"):
                    profile.command_run(args)

    def test_guard_is_reverified_before_each_server_arm(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self.make_plan()
            plan_path = root / "plan.json"
            profile.write_json(plan_path, plan, False)
            args = self.make_args(root, plan_path)

            def fake_run_arm(**kwargs):
                self.write_arm_artifacts(
                    kwargs["output_dir"], kwargs["source_identity_value"],
                    kwargs["entry"], kwargs["cell"], kwargs["args"],
                )

            run_arm = mock.Mock(side_effect=fake_run_arm)
            guard_checks = [
                "GPU-MOCK",
                "GPU-MOCK",
                profile.ContractError("exclusive GPU Job Object is not active"),
            ]
            with mock.patch.object(profile, "verify_guard_lease", side_effect=guard_checks), \
                    mock.patch.object(profile, "gpu_identity", return_value={
                        "uuid": "GPU-MOCK", "name": "Mock", "driver_version": "1",
                        "compute_capability": "12.0", "cuda_driver_runtime": "13.0",
                    }), \
                    mock.patch.object(profile, "source_commit", return_value="c" * 40), \
                    mock.patch.object(profile, "run_arm", run_arm):
                with self.assertRaisesRegex(profile.ContractError, "Job Object is not active"):
                    profile.command_run(args)
            self.assertEqual(run_arm.call_count, 1)

    def test_sidecar_rejects_incomplete_warmup_and_per_slot_overflow(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = profile.build_plan(
                seed=20260818,
                active_slots=[2],
                context_ceilings=[2048],
                prefix_caps=[0, 7],
                max_draft_tokens_per_slot=7,
                repetitions=2,
                samples_per_cell=2,
                warmup_shape_runs=3,
            )
            plan_path = root / "plan.json"
            profile.write_json(plan_path, plan, False)
            args = self.make_args(root, plan_path)
            args.samples_per_cell = 2
            args.warmup_shape_runs = 3
            entry = next(
                item for item in plan["schedule"]
                if next(cell for cell in plan["cells"] if cell["cell_id"] == item["cell_id"])["prefix_cap"] == 7
            )
            cell = {item["cell_id"]: item for item in plan["cells"]}[entry["cell_id"]]
            self.write_arm_artifacts(root / "output", self.SOURCE_ID, entry, cell, args)
            sidecar_path = profile.arm_paths(root / "output", entry["ordinal"])["sidecar"]
            original = profile.load_json(sidecar_path)

            incomplete_warmup = json.loads(json.dumps(original))
            incomplete_warmup["coordinates"][0]["warmup_seen"] = 2
            with self.assertRaisesRegex(profile.ContractError, "counters are inconsistent"):
                profile.validate_sidecar(
                    incomplete_warmup,
                    source_identity_value=self.SOURCE_ID,
                    cell=cell,
                    max_draft_tokens_per_slot=7,
                    samples_per_cell=2,
                    warmup_shape_runs=3,
                    require_ready=True,
                )

            overflow = json.loads(json.dumps(original))
            overflow["coordinates"][0]["samples"][0]["prefixes"] = [8, 6]
            with self.assertRaisesRegex(profile.ContractError, "violates the coordinate contract"):
                profile.validate_sidecar(
                    overflow,
                    source_identity_value=self.SOURCE_ID,
                    cell=cell,
                    max_draft_tokens_per_slot=7,
                    samples_per_cell=2,
                    warmup_shape_runs=3,
                    require_ready=True,
                )

    def test_run_arm_uses_one_process_and_always_cleans_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self.make_plan()
            plan_path = root / "plan.json"
            profile.write_json(plan_path, plan, False)
            args = self.make_args(root, plan_path)
            args.output_tokens = (2 + 3 + 16) * 8
            entry = plan["schedule"][0]
            cell = {cell["cell_id"]: cell for cell in plan["cells"]}[entry["cell_id"]]
            process = mock.Mock()
            process.poll.return_value = None
            process.returncode = None
            stdout = mock.Mock()
            stderr = mock.Mock()

            def fake_wave(**kwargs):
                self.write_arm_artifacts(root / "output", self.SOURCE_ID, entry, cell, args)
                return profile.load_json(profile.arm_paths(root / "output", entry["ordinal"])["client"])

            metrics = "llamacpp:spec_decode_sps_record_retained_samples_total 2\n"
            props = {"total_slots": 8, "endpoint_metrics": True, "default_generation_settings": {"n_ctx": 8192}}
            with mock.patch.object(profile, "port_is_open", return_value=False), \
                    mock.patch.object(profile, "start_server", return_value=(process, stdout, stderr)) as started, \
                    mock.patch.object(profile, "wait_ready", return_value={"status": "ok"}), \
                    mock.patch.object(profile, "http_json", return_value=props), \
                    mock.patch.object(profile, "http_text", return_value=metrics), \
                    mock.patch.object(profile, "run_wave", side_effect=fake_wave), \
                    mock.patch.object(profile, "stop_process_tree") as stopped:
                profile.run_arm(
                    repo_root=ROOT, output_dir=root / "output", source_identity_value=self.SOURCE_ID,
                    entry=entry, cell=cell, args=args, job_deadline=profile.time.monotonic() + 60,
                )
            self.assertEqual(started.call_count, 1)
            self.assertEqual(stopped.call_count, 1)
            stdout.close.assert_called_once()
            stderr.close.assert_called_once()

    def test_run_wave_aborts_server_without_waiting_for_http_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self.make_plan()
            plan_path = root / "plan.json"
            profile.write_json(plan_path, plan, False)
            args = self.make_args(root, plan_path)
            args.output_tokens = 128
            args.http_timeout_s = 900
            args.progress_timeout_s = 0.03
            args.poll_interval_s = 0.005
            entry = plan["schedule"][0]
            cell = {item["cell_id"]: item for item in plan["cells"]}[entry["cell_id"]]
            released = profile.threading.Event()

            def blocked_completion(**kwargs):
                kwargs["barrier"].wait(timeout=1)
                released.wait(timeout=2)
                raise profile.ContractError("endpoint closed")

            abort = mock.Mock(side_effect=released.set)
            started = profile.time.monotonic()
            with mock.patch.object(profile, "completion_request", side_effect=blocked_completion), \
                    mock.patch.object(
                        profile,
                        "http_text",
                        return_value="llamacpp:spec_decode_sps_record_retained_samples_total 0\n",
                    ):
                with self.assertRaisesRegex(profile.ContractError, "progress timeout"):
                    profile.run_wave(
                        base_url="http://127.0.0.1:1",
                        source_identity_value=self.SOURCE_ID,
                        entry=entry,
                        cell=cell,
                        args=args,
                        prompt_tokens=100,
                        sidecar_path=root / "missing.samples.json",
                        arm_deadline=profile.time.monotonic() + 60,
                        abort_server=abort,
                    )
            self.assertLess(profile.time.monotonic() - started, 0.5)
            abort.assert_called_once_with()

    def test_run_wave_aborts_other_requests_on_peer_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = profile.build_plan(
                seed=20260818,
                active_slots=[2],
                context_ceilings=[2048],
                prefix_caps=[0, 7],
                max_draft_tokens_per_slot=7,
                repetitions=2,
                samples_per_cell=2,
                warmup_shape_runs=3,
            )
            plan_path = root / "plan.json"
            profile.write_json(plan_path, plan, False)
            args = self.make_args(root, plan_path)
            args.output_tokens = 128
            args.http_timeout_s = 900
            args.progress_timeout_s = 30
            args.poll_interval_s = 0.005
            entry = plan["schedule"][0]
            cell = {item["cell_id"]: item for item in plan["cells"]}[entry["cell_id"]]
            released = profile.threading.Event()
            metrics_started = profile.threading.Event()
            sequence = iter((0, 1))
            sequence_lock = profile.threading.Lock()

            def peer_completion(**kwargs):
                kwargs["barrier"].wait(timeout=1)
                with sequence_lock:
                    index = next(sequence)
                if index == 0:
                    metrics_started.wait(timeout=1)
                    raise profile.ContractError("peer failed")
                released.wait(timeout=2)
                raise profile.ContractError("endpoint closed")

            def blocked_metrics(*unused, **unused_kwargs):
                metrics_started.set()
                released.wait(timeout=2)
                return "llamacpp:spec_decode_sps_record_retained_samples_total 0\n"

            abort = mock.Mock(side_effect=released.set)
            started = profile.time.monotonic()
            with mock.patch.object(profile, "completion_request", side_effect=peer_completion), \
                    mock.patch.object(profile, "http_text", side_effect=blocked_metrics):
                with self.assertRaisesRegex(profile.ContractError, "completion request failed"):
                    profile.run_wave(
                        base_url="http://127.0.0.1:1",
                        source_identity_value=self.SOURCE_ID,
                        entry=entry,
                        cell=cell,
                        args=args,
                        prompt_tokens=100,
                        sidecar_path=root / "missing.samples.json",
                        arm_deadline=profile.time.monotonic() + 60,
                        abort_server=abort,
                    )
            self.assertLess(profile.time.monotonic() - started, 0.5)
            abort.assert_called_once_with()

    def test_http_json_mock_rejects_duplicate_members(self):
        class Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *unused):
                return False

            def read(self, size):
                del size
                return b'{"status":"ok","status":"bad"}'

        opener = mock.Mock()
        opener.open.return_value = Response()
        with mock.patch.object(profile.urllib.request, "build_opener", return_value=opener):
            with self.assertRaisesRegex(profile.ContractError, "duplicate"):
                profile.http_json("http://127.0.0.1:1", "/health", 1)

    def test_abortable_http_close_interrupts_blocked_local_response(self):
        listener = profile.socket.socket(profile.socket.AF_INET, profile.socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        request_seen = profile.threading.Event()
        release_server = profile.threading.Event()

        def black_hole_server():
            connection, _ = listener.accept()
            try:
                connection.recv(65536)
                request_seen.set()
                release_server.wait(timeout=2)
            finally:
                connection.close()
                listener.close()

        server_thread = profile.threading.Thread(target=black_hole_server, daemon=True)
        server_thread.start()
        connections = profile.AbortableHttpConnections()
        client_error = []

        def client():
            try:
                connections.post_json(f"http://127.0.0.1:{port}", "/completion", 30, {"prompt": [1]})
            except Exception as error:
                client_error.append(error)

        client_thread = profile.threading.Thread(target=client, daemon=True)
        client_thread.start()
        self.assertTrue(request_seen.wait(timeout=1))
        started = profile.time.monotonic()
        connections.close_all()
        client_thread.join(timeout=0.5)
        release_server.set()
        server_thread.join(timeout=1)
        self.assertFalse(client_thread.is_alive())
        self.assertLess(profile.time.monotonic() - started, 0.5)
        self.assertTrue(client_error)

    def test_gpu_identity_binds_driver_runtime_and_guard_uuid(self):
        with mock.patch.object(profile, "subprocess_text", side_effect=[
            "GPU-MOCK, GeForce RTX 5090, 999.1, 12.0",
            "NVIDIA-SMI 999.1 Driver Version: 999.1 CUDA Version: 13.0",
        ]) as invoked:
            identity = profile.gpu_identity(ROOT, "GPU-MOCK")
        self.assertEqual(identity["uuid"], "GPU-MOCK")
        self.assertEqual(identity["cuda_driver_runtime"], "13.0")
        self.assertEqual(invoked.call_count, 2)

    def test_forged_environment_and_unlocked_lease_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lease_path = root / ".codex-deploy" / "gpu-exclusive.lock"
            nonce = "a" * 64
            job_name = f"Local\\AiLoaderGpuGuard-{nonce}"
            profile.write_json(lease_path, {
                "LeaseVersion": 2,
                "Nonce": nonce,
                "OwnerPid": profile.os.getppid(),
                "CreatedAt": profile.utc_now(),
                "OwnerStartTimeUtc": profile.utc_now(),
                "JobName": job_name,
                "Executable": "forged",
                "GpuIndex": 0,
                "GpuUuid": "GPU-MOCK",
            }, False)
            forged = {
                profile.GUARD_MARKER_ENV: "1",
                profile.GUARD_UUID_ENV: "GPU-MOCK",
                profile.GUARD_LEASE_PATH_ENV: str(lease_path),
                profile.GUARD_LEASE_NONCE_ENV: nonce,
                profile.GUARD_OWNER_PID_ENV: str(profile.os.getppid()),
                profile.GUARD_JOB_NAME_ENV: job_name,
                "CUDA_VISIBLE_DEVICES": "GPU-MOCK",
            }
            with mock.patch.dict(profile.os.environ, forged, clear=False):
                with self.assertRaisesRegex(profile.ContractError, "not actively write-locked"):
                    profile.verify_guard_lease(root)

    def test_read_only_unlocked_lease_is_not_lock_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lease_path = root / ".codex-deploy" / "gpu-exclusive.lock"
            nonce = "b" * 64
            job_name = f"Local\\AiLoaderGpuGuard-{nonce}"
            profile.write_json(lease_path, {
                "LeaseVersion": 2, "Nonce": nonce, "OwnerPid": profile.os.getppid(),
                "CreatedAt": profile.utc_now(), "OwnerStartTimeUtc": profile.utc_now(),
                "JobName": job_name, "Executable": "forged", "GpuIndex": 0,
                "GpuUuid": "GPU-MOCK",
            }, False)
            lease_path.chmod(stat.S_IREAD)
            forged = {
                profile.GUARD_MARKER_ENV: "1", profile.GUARD_UUID_ENV: "GPU-MOCK",
                profile.GUARD_LEASE_PATH_ENV: str(lease_path),
                profile.GUARD_LEASE_NONCE_ENV: nonce,
                profile.GUARD_OWNER_PID_ENV: str(profile.os.getppid()),
                profile.GUARD_JOB_NAME_ENV: job_name, "CUDA_VISIBLE_DEVICES": "GPU-MOCK",
            }
            try:
                with mock.patch.dict(profile.os.environ, forged, clear=False):
                    with self.assertRaisesRegex(profile.ContractError, "not actively write-locked"):
                        profile.verify_guard_lease(root)
            finally:
                lease_path.chmod(stat.S_IWRITE)

    @unittest.skipUnless(profile.os.name == "nt", "Windows Job Object test")
    def test_unrelated_lock_holder_cannot_forge_job_membership(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lease_path = root / ".codex-deploy" / "gpu-exclusive.lock"
            nonce = "c" * 64
            job_name = f"Local\\AiLoaderGpuGuard-{nonce}"
            profile.write_json(lease_path, {
                "LeaseVersion": 2, "Nonce": nonce, "OwnerPid": profile.os.getppid(),
                "CreatedAt": profile.utc_now(),
                "OwnerStartTimeUtc": profile.windows_process_start_time(profile.os.getppid()).isoformat(),
                "JobName": job_name, "Executable": "forged", "GpuIndex": 0,
                "GpuUuid": "GPU-MOCK",
            }, False)
            kernel32 = profile.ctypes.WinDLL("kernel32", use_last_error=True)
            create_file = kernel32.CreateFileW
            create_file.argtypes = (
                profile.ctypes.c_wchar_p, profile.ctypes.c_uint32, profile.ctypes.c_uint32,
                profile.ctypes.c_void_p, profile.ctypes.c_uint32, profile.ctypes.c_uint32,
                profile.ctypes.c_void_p,
            )
            create_file.restype = profile.ctypes.c_void_p
            handle = create_file(str(lease_path), 0xC0000000, 0x00000001, None, 3, 0x80, None)
            self.assertNotEqual(handle, profile.ctypes.c_void_p(-1).value)
            forged = {
                profile.GUARD_MARKER_ENV: "1", profile.GUARD_UUID_ENV: "GPU-MOCK",
                profile.GUARD_LEASE_PATH_ENV: str(lease_path),
                profile.GUARD_LEASE_NONCE_ENV: nonce,
                profile.GUARD_OWNER_PID_ENV: str(profile.os.getppid()),
                profile.GUARD_JOB_NAME_ENV: job_name, "CUDA_VISIBLE_DEVICES": "GPU-MOCK",
            }
            try:
                with mock.patch.dict(profile.os.environ, forged, clear=False):
                    with self.assertRaisesRegex(profile.ContractError, "Job Object is not active"):
                        profile.verify_guard_lease(root)
            finally:
                kernel32.CloseHandle(handle)


if __name__ == "__main__":
    unittest.main()
