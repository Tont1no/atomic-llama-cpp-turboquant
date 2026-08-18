#!/usr/bin/env python3

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


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

    def test_run_is_fail_closed_and_does_not_touch_runtime_paths(self):
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
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("GPU execution is disabled", completed.stderr)
            self.assertFalse((root / "output").exists())


if __name__ == "__main__":
    unittest.main()
