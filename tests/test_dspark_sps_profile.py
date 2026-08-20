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

    def test_ramp_drain_active_slots_three_five_six_and_seven_validate(self):
        plan = self.make_plan(
            active_slots=[3, 5, 6, 7],
            context_ceilings=[2048],
            prefix_caps=[0, 1, 7],
        )
        validated = profile.validate_plan(plan)
        self.assertEqual(validated["active_slots"], [3, 5, 6, 7])
        self.assertEqual(validated["row_axes_by_active"]["3"], [3, 6, 24])
        self.assertEqual(validated["row_axes_by_active"]["5"], [5, 10, 40])
        self.assertEqual(validated["row_axes_by_active"]["6"], [6, 12, 48])
        self.assertEqual(validated["row_axes_by_active"]["7"], [7, 14, 56])
        self.assertEqual(
            {cell["active_slots"] for cell in validated["cells"]},
            {3, 5, 6, 7},
        )

    def test_active_slots_outside_one_through_eight_fail(self):
        for invalid in (0, 9):
            with self.subTest(active_slots=invalid), self.assertRaisesRegex(
                profile.ContractError, "integers from 1 through 8"
            ):
                self.make_plan(active_slots=[invalid])

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

    def make_args(self, root: Path, plan_path: Path, extra: list[str] | None = None):
        for name in ("server.exe", "model.gguf", "draft.gguf", "ggml-cuda.dll"):
            path = root / name
            if not path.exists():
                path.write_bytes(name.encode("ascii"))
        arguments = [
                "run", "--plan", str(plan_path),
                "--server", str(root / "server.exe"),
                "--model", str(root / "model.gguf"),
                "--draft-model", str(root / "draft.gguf"),
                "--output-dir", str(root / "output"),
                "--runtime-label", "mock-cuda",
                "--samples-per-cell", "2",
                "--warmup-shape-runs", "3",
            ]
        if extra is not None:
            arguments.extend(extra)
        return profile.build_parser().parse_args(arguments)

    def recorder_metrics(
        self,
        *,
        retained=0,
        warmup=0,
        ready=0,
        total=1,
        capture=0,
        mixed=0,
        partial=0,
        retry=0,
        other=0,
        write_errors=0,
    ):
        values = {
            profile.RECORDER_RETAINED_METRIC: retained,
            profile.RECORDER_WARMUP_METRIC: warmup,
            profile.RECORDER_READY_METRIC: ready,
            profile.RECORDER_TOTAL_METRIC: total,
            "llamacpp:spec_decode_sps_record_skipped_capture_total": capture,
            "llamacpp:spec_decode_sps_record_skipped_mixed_total": mixed,
            "llamacpp:spec_decode_sps_record_skipped_partial_total": partial,
            "llamacpp:spec_decode_sps_record_skipped_retry_total": retry,
            "llamacpp:spec_decode_sps_record_skipped_other_total": other,
            "llamacpp:spec_decode_sps_record_write_errors_total": write_errors,
        }
        return "".join(f"{name} {value}\n" for name, value in values.items())

    def test_dynamic_rs_is_explicit_and_part_of_immutable_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = root / "plan.json"
            default_args = self.make_args(root, plan_path)
            cell = self.make_plan()["cells"][0]
            default_command = profile.server_arguments(
                default_args, cell, root / "record.json", self.SOURCE_ID
            )
            self.assertIn("--no-spec-draft-dynamic-rs", default_command)
            self.assertNotIn("--spec-draft-dynamic-rs", default_command)
            self.assertFalse(profile.immutable_runtime_config(default_args)["dynamic_rs"])

            enabled_args = self.make_args(root, plan_path, ["--dynamic-rs"])
            enabled_command = profile.server_arguments(
                enabled_args, cell, root / "record-enabled.json", self.SOURCE_ID
            )
            self.assertIn("--spec-draft-dynamic-rs", enabled_command)
            self.assertNotIn("--no-spec-draft-dynamic-rs", enabled_command)
            self.assertTrue(profile.immutable_runtime_config(enabled_args)["dynamic_rs"])
            self.assertNotEqual(
                profile.immutable_runtime_config(default_args),
                profile.immutable_runtime_config(enabled_args),
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
        metric_text = self.recorder_metrics(retained=2, warmup=3, ready=1)
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

            ineligible = json.loads(json.dumps(original))
            ineligible["coordinates"][0]["skipped_ineligible"] = 1
            ineligible["skipped_ineligible_total"] = 1
            ineligible["observations_total"] += 1
            with self.assertRaisesRegex(profile.ContractError, "ineligible target decodes"):
                profile.validate_sidecar(
                    ineligible,
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

            metrics = self.recorder_metrics(retained=2, warmup=3, ready=1)
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

    def test_run_wave_uses_metrics_only_until_workers_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self.make_plan()
            args = self.make_args(root, root / "plan.json")
            args.poll_interval_s = 0.005
            entry = plan["schedule"][0]
            cell = {item["cell_id"]: item for item in plan["cells"]}[entry["cell_id"]]
            released = profile.threading.Event()

            def completion(**kwargs):
                kwargs["barrier"].wait(timeout=1)
                released.wait(timeout=2)
                return {
                    "seed": 1,
                    "prompt_tokens": 100,
                    "generated_tokens": args.output_tokens,
                    "wall_ms": 100.0,
                    "predicted_ms": 90.0,
                    "predicted_per_second": 10.0,
                    "content_sha256": "b" * 64,
                }

            complete_metrics = self.recorder_metrics(retained=2, warmup=3, ready=1)

            def metrics(*unused, **unused_kwargs):
                released.set()
                return complete_metrics

            with mock.patch.object(profile, "completion_request", side_effect=completion), \
                    mock.patch.object(profile, "http_text", side_effect=metrics) as polled, \
                    mock.patch.object(
                        profile,
                        "load_atomic_recorder_json",
                        side_effect=AssertionError("live wave opened recorder JSON"),
                    ) as recorder_open:
                client = profile.run_wave(
                    base_url="http://127.0.0.1:1",
                    source_identity_value=self.SOURCE_ID,
                    entry=entry,
                    cell=cell,
                    args=args,
                    prompt_tokens=100,
                    arm_deadline=profile.time.monotonic() + 60,
                    abort_server=mock.Mock(),
                    initial_metrics=profile.parse_metrics(self.recorder_metrics()),
                )

            self.assertEqual(len(client["requests"]), cell["active_slots"])
            self.assertGreaterEqual(polled.call_count, 2)
            recorder_open.assert_not_called()

    def test_run_wave_aborts_when_metrics_endpoint_disappears(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self.make_plan()
            args = self.make_args(root, root / "plan.json")
            args.poll_interval_s = 0.005
            entry = plan["schedule"][0]
            cell = {item["cell_id"]: item for item in plan["cells"]}[entry["cell_id"]]
            released = profile.threading.Event()

            def blocked_completion(**kwargs):
                kwargs["barrier"].wait(timeout=1)
                released.wait(timeout=2)
                raise profile.ContractError("endpoint closed")

            abort = mock.Mock(side_effect=released.set)
            with mock.patch.object(profile, "completion_request", side_effect=blocked_completion), \
                    mock.patch.object(
                        profile,
                        "http_text",
                        side_effect=profile.ContractError("metrics endpoint disappeared"),
                    ):
                with self.assertRaisesRegex(profile.ContractError, "metrics endpoint disappeared"):
                    profile.run_wave(
                        base_url="http://127.0.0.1:1",
                        source_identity_value=self.SOURCE_ID,
                        entry=entry,
                        cell=cell,
                        args=args,
                        prompt_tokens=100,
                        arm_deadline=profile.time.monotonic() + 60,
                        abort_server=abort,
                        initial_metrics=profile.parse_metrics(self.recorder_metrics()),
                    )
            abort.assert_called_once_with()

    def test_run_wave_rejects_invalid_recorder_counter_before_sidecar_io(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self.make_plan()
            args = self.make_args(root, root / "plan.json")
            entry = plan["schedule"][0]
            cell = {item["cell_id"]: item for item in plan["cells"]}[entry["cell_id"]]
            released = profile.threading.Event()

            def blocked_completion(**kwargs):
                kwargs["barrier"].wait(timeout=1)
                released.wait(timeout=2)
                raise profile.ContractError("endpoint closed")

            abort = mock.Mock(side_effect=released.set)
            invalid = profile.parse_metrics(self.recorder_metrics(write_errors=1))
            with mock.patch.object(profile, "completion_request", side_effect=blocked_completion), \
                    mock.patch.object(
                        profile,
                        "load_atomic_recorder_json",
                        side_effect=AssertionError("invalid metrics reached sidecar I/O"),
                    ) as recorder_open:
                with self.assertRaisesRegex(profile.ContractError, "retry/write-error"):
                    profile.run_wave(
                        base_url="http://127.0.0.1:1",
                        source_identity_value=self.SOURCE_ID,
                        entry=entry,
                        cell=cell,
                        args=args,
                        prompt_tokens=100,
                        arm_deadline=profile.time.monotonic() + 60,
                        abort_server=abort,
                        initial_metrics=invalid,
                    )
            abort.assert_called_once_with()
            recorder_open.assert_not_called()

    def test_recorder_metric_contract_is_strict_but_keeps_diagnostics(self):
        complete = profile.parse_metrics(
            self.recorder_metrics(retained=2, warmup=4, ready=1, capture=1, other=7)
        )
        is_complete, progress = profile.validate_recorder_metrics(
            complete,
            samples_per_cell=2,
            warmup_shape_runs=3,
            require_complete=True,
        )
        self.assertTrue(is_complete)
        self.assertEqual(progress, (2, 1))

        missing = dict(complete)
        del missing[profile.RECORDER_RETAINED_METRIC]
        with self.assertRaisesRegex(profile.ContractError, "missing"):
            profile.validate_recorder_metrics(
                missing, samples_per_cell=2, warmup_shape_runs=3, require_complete=True
            )

        noninteger = dict(complete)
        noninteger[profile.RECORDER_WARMUP_METRIC] = 3.5
        with self.assertRaisesRegex(profile.ContractError, "non-negative integer"):
            profile.validate_recorder_metrics(
                noninteger, samples_per_cell=2, warmup_shape_runs=3, require_complete=True
            )

        decreased = dict(complete)
        decreased[profile.RECORDER_RETAINED_METRIC] = 1.0
        with self.assertRaisesRegex(profile.ContractError, "decreased"):
            profile.validate_recorder_metrics(
                decreased,
                samples_per_cell=2,
                warmup_shape_runs=3,
                previous=complete,
                require_complete=False,
            )

        mixed = dict(complete)
        mixed["llamacpp:spec_decode_sps_record_skipped_mixed_total"] = 1.0
        with self.assertRaisesRegex(profile.ContractError, "invalid generation counter"):
            profile.validate_recorder_metrics(
                mixed, samples_per_cell=2, warmup_shape_runs=3, require_complete=True
            )

    def test_run_wave_warmup_only_does_not_extend_progress_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self.make_plan()
            args = self.make_args(root, root / "plan.json")
            args.progress_timeout_s = 0.08
            args.poll_interval_s = 0.005
            entry = plan["schedule"][0]
            cell = {item["cell_id"]: item for item in plan["cells"]}[entry["cell_id"]]
            released = profile.threading.Event()
            poll = 0

            def blocked_completion(**kwargs):
                kwargs["barrier"].wait(timeout=1)
                released.wait(timeout=2)
                raise profile.ContractError("endpoint closed")

            def warmup_only(*unused, **unused_kwargs):
                nonlocal poll
                poll += 1
                return self.recorder_metrics(warmup=poll, capture=poll, other=poll)

            abort = mock.Mock(side_effect=released.set)
            with mock.patch.object(profile, "completion_request", side_effect=blocked_completion), \
                    mock.patch.object(profile, "http_text", side_effect=warmup_only):
                with self.assertRaisesRegex(profile.ContractError, "progress timeout"):
                    profile.run_wave(
                        base_url="http://127.0.0.1:1",
                        source_identity_value=self.SOURCE_ID,
                        entry=entry,
                        cell=cell,
                        args=args,
                        prompt_tokens=100,
                        arm_deadline=profile.time.monotonic() + 60,
                        abort_server=abort,
                        initial_metrics=profile.parse_metrics(self.recorder_metrics()),
                    )
            self.assertGreaterEqual(poll, 1)
            abort.assert_called_once_with()

    def test_staggered_worker_completion_does_not_extend_recorder_timeout(self):
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
            args = self.make_args(root, root / "plan.json")
            args.progress_timeout_s = 0.10
            args.poll_interval_s = 0.005
            entry = plan["schedule"][0]
            cell = {item["cell_id"]: item for item in plan["cells"]}[entry["cell_id"]]
            released = profile.threading.Event()
            sequence = iter((0, 1))
            sequence_lock = profile.threading.Lock()

            def staggered_completion(**kwargs):
                kwargs["barrier"].wait(timeout=1)
                with sequence_lock:
                    index = next(sequence)
                if index == 0:
                    profile.time.sleep(0.07)
                    return {
                        "seed": 1,
                        "prompt_tokens": 100,
                        "generated_tokens": args.output_tokens,
                        "wall_ms": 70.0,
                        "predicted_ms": 60.0,
                        "predicted_per_second": 10.0,
                        "content_sha256": "b" * 64,
                    }
                released.wait(timeout=2)
                raise profile.ContractError("endpoint closed")

            abort = mock.Mock(side_effect=released.set)
            started = profile.time.monotonic()
            with mock.patch.object(profile, "completion_request", side_effect=staggered_completion), \
                    mock.patch.object(profile, "http_text", return_value=self.recorder_metrics()):
                with self.assertRaisesRegex(profile.ContractError, "progress timeout"):
                    profile.run_wave(
                        base_url="http://127.0.0.1:1",
                        source_identity_value=self.SOURCE_ID,
                        entry=entry,
                        cell=cell,
                        args=args,
                        prompt_tokens=100,
                        arm_deadline=profile.time.monotonic() + 60,
                        abort_server=abort,
                        initial_metrics=profile.parse_metrics(self.recorder_metrics()),
                    )
            self.assertLess(profile.time.monotonic() - started, 0.16)
            abort.assert_called_once_with()

    def test_run_arm_strictly_validates_sidecar_only_after_wave(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self.make_plan()
            plan_path = root / "plan.json"
            profile.write_json(plan_path, plan, False)
            args = self.make_args(root, plan_path)
            entry = plan["schedule"][0]
            cell = {item["cell_id"]: item for item in plan["cells"]}[entry["cell_id"]]
            process = mock.Mock()
            process.poll.return_value = None
            process.returncode = None
            stdout = mock.Mock()
            stderr = mock.Mock()
            wave_completed = profile.threading.Event()

            def fake_wave(**unused):
                self.write_arm_artifacts(root / "output", self.SOURCE_ID, entry, cell, args)
                sidecar_path = profile.arm_paths(root / "output", entry["ordinal"])["sidecar"]
                sidecar = profile.load_json(sidecar_path)
                sidecar["unknown"] = True
                profile.write_json(sidecar_path, sidecar, True)
                wave_completed.set()
                return profile.load_json(
                    profile.arm_paths(root / "output", entry["ordinal"])["client"]
                )

            original_load = profile.load_atomic_recorder_json

            def strict_load(*args_, **kwargs_):
                self.assertTrue(wave_completed.is_set())
                return original_load(*args_, **kwargs_)

            metrics = self.recorder_metrics(retained=2, warmup=3, ready=1)
            props = {"total_slots": 8, "endpoint_metrics": True, "default_generation_settings": {"n_ctx": 8192}}
            with mock.patch.object(profile, "port_is_open", return_value=False), \
                    mock.patch.object(profile, "start_server", return_value=(process, stdout, stderr)), \
                    mock.patch.object(profile, "wait_ready", return_value={"status": "ok"}), \
                    mock.patch.object(profile, "http_json", return_value=props), \
                    mock.patch.object(profile, "http_text", return_value=metrics), \
                    mock.patch.object(profile, "run_wave", side_effect=fake_wave), \
                    mock.patch.object(profile, "load_atomic_recorder_json", side_effect=strict_load), \
                    mock.patch.object(profile, "stop_process_tree"):
                with self.assertRaisesRegex(profile.ContractError, "keys mismatch"):
                    profile.run_arm(
                        repo_root=ROOT,
                        output_dir=root / "output",
                        source_identity_value=self.SOURCE_ID,
                        entry=entry,
                        cell=cell,
                        args=args,
                        job_deadline=profile.time.monotonic() + 60,
                    )

    @unittest.skipUnless(profile.os.name == "nt", "Windows process-handle contract")
    def test_stop_process_tree_uses_retained_handle_and_job_accounting(self):
        process = mock.Mock()
        process.pid = 12345
        process.poll.return_value = None
        process.wait.return_value = 0

        with mock.patch.object(
            profile.subprocess, "run", side_effect=AssertionError("numeric PID killing is forbidden")
        ), mock.patch.object(
            profile, "windows_job_active_process_count", side_effect=(2, 1)
        ) as job_count:
            profile.stop_process_tree(
                process,
                5.0,
                job_name="Local\\MockJob",
                expected_job_processes=1,
            )

        process.kill.assert_called_once_with()
        process.wait.assert_called_once_with(timeout=mock.ANY)
        self.assertEqual(job_count.call_count, 2)

    @unittest.skipUnless(profile.os.name == "nt", "Windows process-handle contract")
    def test_stop_process_tree_fails_closed_when_descendant_remains(self):
        process = mock.Mock()
        process.pid = 23456
        process.poll.return_value = 0

        with mock.patch.object(
            profile.subprocess, "run", side_effect=AssertionError("numeric PID killing is forbidden")
        ), mock.patch.object(
            profile, "windows_job_active_process_count", return_value=2
        ):
            with self.assertRaisesRegex(profile.ContractError, "descendants remained"):
                profile.stop_process_tree(
                    process,
                    0.01,
                    job_name="Local\\MockJob",
                    expected_job_processes=1,
                )

        process.kill.assert_not_called()
        process.wait.assert_not_called()

    def test_stop_process_tree_rejects_non_finite_deadline(self):
        process = mock.Mock()
        for timeout_s in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(timeout_s=timeout_s):
                with self.assertRaisesRegex(profile.ContractError, "positive and finite"):
                    profile.stop_process_tree(process, timeout_s)
        process.poll.assert_not_called()

    def test_command_run_rejects_non_finite_deadlines(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = root / "plan.json"
            profile.write_json(plan_path, self.make_plan(), False)
            for timeout_s in (float("nan"), float("inf"), float("-inf")):
                args = self.make_args(root, plan_path)
                args.stop_timeout_s = timeout_s
                with self.subTest(timeout_s=timeout_s), mock.patch.object(
                    profile, "verify_guard_lease", return_value="GPU-MOCK"
                ):
                    with self.assertRaisesRegex(profile.ContractError, "must be positive"):
                        profile.command_run(args)

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
                        return_value=self.recorder_metrics(),
                    ):
                with self.assertRaisesRegex(profile.ContractError, "progress timeout"):
                    profile.run_wave(
                        base_url="http://127.0.0.1:1",
                        source_identity_value=self.SOURCE_ID,
                        entry=entry,
                        cell=cell,
                        args=args,
                        prompt_tokens=100,
                        arm_deadline=profile.time.monotonic() + 60,
                        abort_server=abort,
                        initial_metrics=profile.parse_metrics(self.recorder_metrics()),
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
                return self.recorder_metrics()

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
                        arm_deadline=profile.time.monotonic() + 60,
                        abort_server=abort,
                        initial_metrics=profile.parse_metrics(self.recorder_metrics()),
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

    def test_atomic_recorder_read_retries_sharing_violation_then_succeeds(self):
        path = Path("record.samples.json")
        locked = profile.ContractError(f"cannot read strict UTF-8 JSON plan: {path}")
        locked.__cause__ = profile.ctypes.WinError(32)
        expected = {"complete": True}
        guard_check = mock.Mock()
        loader_name = (
            "_load_windows_shared_atomic_recorder_json"
            if profile.os.name == "nt"
            else "load_json"
        )

        with mock.patch.object(profile, loader_name, side_effect=(locked, expected)) as loaded:
            actual = profile.load_atomic_recorder_json(
                path,
                deadline=profile.time.monotonic() + 1.0,
                guard_check=guard_check,
            )

        self.assertEqual(actual, expected)
        self.assertEqual(loaded.call_count, 2)
        guard_check.assert_called_once_with()

    def test_atomic_recorder_read_rejects_persistent_invalid_utf8_and_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = {
                "invalid-utf8.json": b"\xff\xfe",
                "invalid-json.json": b'{"complete":',
            }
            for name, payload in cases.items():
                path = root / name
                path.write_bytes(payload)
                started = profile.time.monotonic()
                with self.subTest(name=name):
                    with self.assertRaisesRegex(
                        profile.ContractError, "cannot read strict UTF-8 JSON"
                    ):
                        profile.load_atomic_recorder_json(
                            path,
                            deadline=profile.time.monotonic() + 1.0,
                        )
                self.assertLess(profile.time.monotonic() - started, 0.25)

    def test_atomic_recorder_read_missing_and_locked_are_deadline_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.samples.json"
            started = profile.time.monotonic()
            with self.assertRaisesRegex(
                profile.ContractError,
                "cannot (stat|read strict UTF-8 JSON)",
            ):
                profile.load_atomic_recorder_json(
                    missing,
                    deadline=profile.time.monotonic() + 0.03,
                )
            self.assertLess(profile.time.monotonic() - started, 0.25)

            locked = profile.ContractError(
                f"cannot read strict UTF-8 JSON plan: {missing}"
            )
            locked.__cause__ = profile.ctypes.WinError(32)
            started = profile.time.monotonic()
            loader_name = (
                "_load_windows_shared_atomic_recorder_json"
                if profile.os.name == "nt"
                else "load_json"
            )
            with mock.patch.object(profile, loader_name, side_effect=locked) as loaded:
                with self.assertRaisesRegex(
                    profile.ContractError, "cannot read strict UTF-8 JSON"
                ):
                    profile.load_atomic_recorder_json(
                        missing,
                        deadline=profile.time.monotonic() + 0.03,
                    )
            self.assertGreaterEqual(loaded.call_count, 1)
            self.assertLess(profile.time.monotonic() - started, 0.25)

    @unittest.skipUnless(profile.os.name == "nt", "Windows shared-open contract")
    def test_windows_live_reader_does_not_block_repeated_atomic_replace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "record.samples.json"
            profile.write_json(path, {"generation": 0}, False)
            kernel32 = profile.ctypes.WinDLL("kernel32", use_last_error=True)
            replace_file = kernel32.ReplaceFileW
            replace_file.argtypes = (
                profile.ctypes.c_wchar_p,
                profile.ctypes.c_wchar_p,
                profile.ctypes.c_wchar_p,
                profile.ctypes.c_uint32,
                profile.ctypes.c_void_p,
                profile.ctypes.c_void_p,
            )
            replace_file.restype = profile.ctypes.c_int
            stop = profile.threading.Event()
            reader_errors = []
            observed = []

            def poll_reader():
                while not stop.is_set():
                    try:
                        value = profile.load_atomic_recorder_json(
                            path,
                            deadline=profile.time.monotonic() + 0.5,
                        )
                        observed.append(value["generation"])
                    except Exception as error:  # captured for the main test thread
                        reader_errors.append(error)
                        return

            reader = profile.threading.Thread(target=poll_reader, daemon=True)
            reader.start()
            try:
                for generation in range(1, 301):
                    incoming = root / "incoming.json"
                    profile.write_json(incoming, {"generation": generation}, False)
                    if not replace_file(str(path), str(incoming), None, 2, None, None):
                        raise profile.ctypes.WinError(profile.ctypes.get_last_error())
            finally:
                stop.set()
                reader.join(timeout=2)

            self.assertFalse(reader.is_alive())
            self.assertFalse(reader_errors)
            self.assertTrue(observed)
            self.assertEqual(profile.load_json(path), {"generation": 300})
            # A completed live read must not retain any file-generation handle.
            replacement = root / "replacement.json"
            profile.write_json(replacement, {"generation": 301}, False)
            profile.os.replace(replacement, path)
            path.unlink()
            self.assertFalse(path.exists())

    @unittest.skipUnless(profile.os.name == "nt", "Windows shared-open contract")
    def test_windows_shared_reader_is_bounded_and_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            oversized = root / "oversized.samples.json"
            with oversized.open("wb") as stream:
                stream.seek(profile.MAX_PLAN_BYTES)
                stream.write(b"x")
            with self.assertRaisesRegex(profile.ContractError, "plan size"):
                profile.load_atomic_recorder_json(
                    oversized,
                    deadline=profile.time.monotonic() + 1.0,
                )

            duplicate = root / "duplicate.samples.json"
            duplicate.write_bytes(b'{"generation":1,"generation":2}')
            with self.assertRaisesRegex(profile.ContractError, "duplicate"):
                profile.load_atomic_recorder_json(
                    duplicate,
                    deadline=profile.time.monotonic() + 1.0,
                )

    @unittest.skipUnless(profile.os.name == "nt", "Windows shared-open contract")
    def test_windows_shared_reader_closes_handle_on_read_and_size_failure(self):
        payload = b'{"complete":true}'

        def make_kernel(*, fail_size=False):
            kernel = mock.Mock()
            kernel.CreateFileW.return_value = 77

            def get_size(unused_handle, size_pointer):
                del unused_handle
                if fail_size:
                    profile.ctypes.set_last_error(5)
                    return 0
                size_pointer._obj.value = len(payload)
                return 1

            def read_file(unused_handle, buffer, count, read_pointer, unused_overlapped):
                del unused_handle, unused_overlapped
                profile.ctypes.memmove(buffer, payload, min(count, len(payload)))
                read_pointer._obj.value = min(count, len(payload))
                return 1

            kernel.GetFileSizeEx.side_effect = get_size
            kernel.ReadFile.side_effect = read_file
            kernel.CloseHandle.return_value = 1
            return kernel

        successful = make_kernel()
        with mock.patch.object(profile.ctypes, "WinDLL", return_value=successful):
            self.assertEqual(
                profile._windows_shared_atomic_recorder_bytes(Path("ok.json")),
                payload,
            )
        successful.CloseHandle.assert_called_once_with(77)

        failed = make_kernel(fail_size=True)
        with mock.patch.object(profile.ctypes, "WinDLL", return_value=failed):
            with self.assertRaises(OSError):
                profile._windows_shared_atomic_recorder_bytes(Path("failed.json"))
        failed.CloseHandle.assert_called_once_with(77)

    @unittest.skipUnless(profile.os.name == "nt", "Windows retry classification")
    def test_windows_live_reader_does_not_retry_non_transient_os_error(self):
        path = Path("record.samples.json")
        failure = profile.ContractError("read failed")
        failure.__cause__ = profile.ctypes.WinError(87)
        with mock.patch.object(
            profile,
            "_load_windows_shared_atomic_recorder_json",
            side_effect=failure,
        ) as loaded:
            with self.assertRaisesRegex(profile.ContractError, "read failed"):
                profile.load_atomic_recorder_json(
                    path,
                    deadline=profile.time.monotonic() + 1.0,
                )
        loaded.assert_called_once_with(path)

    def test_abortable_http_cleanup_is_bounded_after_server_release(self):
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
                connections.post_json(f"http://127.0.0.1:{port}", "/completion", 0.25, {"prompt": [1]})
            except Exception as error:
                client_error.append(error)

        client_thread = profile.threading.Thread(target=client, daemon=True)
        client_thread.start()
        self.assertTrue(request_seen.wait(timeout=1))
        started = profile.time.monotonic()
        connections.close_all()
        release_server.set()
        server_thread.join(timeout=1)
        client_thread.join(timeout=1)
        self.assertFalse(client_thread.is_alive())
        self.assertLess(profile.time.monotonic() - started, 1.5)
        self.assertTrue(client_error)

    def test_gpu_identity_accepts_cuda_version_banner(self):
        with mock.patch.object(profile, "subprocess_text", side_effect=[
            "GPU-MOCK, GeForce RTX 5090, 999.1, 12.0",
            "NVIDIA-SMI 999.1 Driver Version: 999.1 CUDA Version: 13.0",
        ]) as invoked:
            identity = profile.gpu_identity(ROOT, "GPU-MOCK")
        self.assertEqual(identity["uuid"], "GPU-MOCK")
        self.assertEqual(identity["cuda_driver_runtime"], "13.0")
        self.assertEqual(invoked.call_count, 2)

    def test_gpu_identity_accepts_cuda_umd_version_banner(self):
        with mock.patch.object(profile, "subprocess_text", side_effect=[
            "GPU-MOCK, GeForce RTX 5090, 610.88, 12.0",
            "NVIDIA-SMI 610.88 KMD Version: 610.88 CUDA UMD Version: 13.3",
        ]):
            identity = profile.gpu_identity(ROOT, "GPU-MOCK")
        self.assertEqual(identity["cuda_driver_runtime"], "13.3")

    def test_gpu_identity_rejects_banner_without_cuda_runtime(self):
        with mock.patch.object(profile, "subprocess_text", side_effect=[
            "GPU-MOCK, GeForce RTX 5090, 610.88, 12.0",
            "NVIDIA-SMI 610.88 KMD Version: 610.88",
        ]):
            with self.assertRaisesRegex(
                profile.ContractError, "did not report a CUDA driver runtime version"
            ):
                profile.gpu_identity(ROOT, "GPU-MOCK")

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
