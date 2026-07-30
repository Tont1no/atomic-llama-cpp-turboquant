#!/usr/bin/env python3

import argparse
from collections import Counter
import os
import re
import subprocess
import sys
from typing import List, Optional, Tuple


ACTIVATION_MARKER = (
    "CUDA: enabling experimental SM89 Turbo4 symmetric-magnitude LUT "
    "for two-column VEC batches on all visible devices"
)
ONE_COLUMN_MARKER = (
    "CUDA: enabling experimental SM89 Turbo4 symmetric-magnitude LUT "
    "for one-column VEC decode on all visible devices"
)
KERNEL_HIT_PATTERN = re.compile(
    r"CUDA: Turbo4 ncols2 kernel-hit: D=(\d+) q_columns=(\d+) "
    r"odd_tail=(\d+) type_V=([a-z0-9_]+)"
)


def visible_cuda_devices() -> List[Tuple[str, str]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=compute_cap,uuid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    devices: List[Tuple[str, str]] = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", maxsplit=1)]
        if len(fields) == 2 and fields[1].startswith("GPU-"):
            devices.append((fields[0], fields[1]))
    return devices


def sm89_device_uuid(devices: List[Tuple[str, str]]) -> Optional[str]:
    for compute_capability, device_uuid in devices:
        if compute_capability == "8.9":
            return device_uuid
    return None


def run_parity(binary: str, device_uuid: str, enabled: bool) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device_uuid
    env["GGML_CUDA_TURBO4_SYM_LUT"] = "0"
    env["GGML_CUDA_TURBO4_SYM_LUT_NCOLS2"] = "1" if enabled else "0"
    env["GGML_CUDA_TURBO4_SYM_LUT_NCOLS2_TRACE"] = "1" if enabled else "0"
    command = [
        binary,
        "test",
        "-o",
        "FLASH_ATTN_EXT",
        "-b",
        "CUDA0",
        "-p",
        r"hsk=(128|256|512).*nb=(2|3|4|5|6|7|8).*type_K=turbo4",
        "-j",
        "1",
    ]
    result = subprocess.run(
        command,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    output = result.stdout + result.stderr
    sys.stdout.write(output)
    if result.returncode != 0:
        raise RuntimeError(
            f"Turbo4 SM89 ncols=2 {'opt-in' if enabled else 'baseline'} parity run failed "
            f"with exit code {result.returncode}"
        )
    match = re.search(r"(\d+)/(\d+) tests passed", output)
    if match is None or match.group(1) != "49" or match.group(2) != "49":
        raise RuntimeError(
            f"expected exactly 49 Turbo4 nb=2..8 parity cases, got "
            f"{match.group(0) if match else 'no test summary'}"
        )
    marker_present = ACTIVATION_MARKER in output
    if enabled and not marker_present:
        raise RuntimeError("Turbo4 ncols=2 run passed numerically but did not activate the experimental kernel")
    if not enabled and marker_present:
        raise RuntimeError("Turbo4 ncols=2 baseline unexpectedly activated the experimental kernel")
    if ONE_COLUMN_MARKER in output:
        raise RuntimeError("Turbo4 ncols=2 parity run unexpectedly activated the qualified one-column experiment")
    kernel_hits = Counter(
        (int(d), int(q_columns), int(odd_tail), type_v)
        for d, q_columns, odd_tail, type_v in KERNEL_HIT_PATTERN.findall(output)
    )
    if not enabled and kernel_hits:
        raise RuntimeError(f"baseline unexpectedly executed the ncols=2 candidate kernel: {kernel_hits}")
    if enabled:
        expected_shapes = (
            (128, "turbo4"),
            (256, "turbo4"),
            (128, "q8_0"),
            (128, "f16"),
            (512, "turbo4"),
            (512, "q8_0"),
            (512, "f16"),
        )
        expected_hits = Counter(
            (head_size, width, width % 2, value_type)
            for width in range(2, 9)
            for head_size, value_type in expected_shapes
        )
        if kernel_hits != expected_hits:
            raise RuntimeError(
                "candidate parity passed numerically but per-case kernel-hit evidence differed: "
                f"expected {expected_hits}, got {kernel_hits}"
            )


def run_unsupported_head_rejection(
    binary: str, device_uuid: str, head_size: int, params_filter: str
) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device_uuid
    env["GGML_CUDA_TURBO4_SYM_LUT"] = "0"
    env["GGML_CUDA_TURBO4_SYM_LUT_NCOLS2"] = "1"
    env["GGML_CUDA_TURBO4_SYM_LUT_NCOLS2_TRACE"] = "0"
    result = subprocess.run(
        [
            binary,
            "test",
            "-o",
            "FLASH_ATTN_EXT",
            "-b",
            "CUDA0",
            "-p",
            params_filter,
            "-j",
            "1",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    output = result.stdout + result.stderr
    sys.stdout.write(output)
    if result.returncode == 0:
        raise RuntimeError(
            f"unsupported Turbo4 D={head_size} request silently succeeded with the ncols=2 experiment enabled"
        )
    if f"reached unsupported Turbo4 head size {head_size}" not in output:
        raise RuntimeError(
            f"unsupported Turbo4 D={head_size} request failed without the fail-closed shape diagnostic"
        )
    if KERNEL_HIT_PATTERN.search(output):
        raise RuntimeError(
            f"unsupported Turbo4 D={head_size} request reached the candidate kernel before rejection"
        )


def run_single_column_non_candidate(binary: str, device_uuid: str) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device_uuid
    env["GGML_CUDA_TURBO4_SYM_LUT"] = "0"
    env["GGML_CUDA_TURBO4_SYM_LUT_NCOLS2"] = "1"
    env["GGML_CUDA_TURBO4_SYM_LUT_NCOLS2_TRACE"] = "1"
    result = subprocess.run(
        [
            binary,
            "test",
            "-o",
            "FLASH_ATTN_EXT",
            "-b",
            "CUDA0",
            "-p",
            r"hsk=(128|256).*nb=1.*type_K=turbo4",
            "-j",
            "1",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    output = result.stdout + result.stderr
    sys.stdout.write(output)
    if result.returncode != 0:
        raise RuntimeError("Turbo4 N=1 non-candidate reference cases failed")
    match = re.search(r"(\d+)/(\d+) tests passed", output)
    if match is None or match.group(1) != "4" or match.group(2) != "4":
        raise RuntimeError("expected exactly four Turbo4 N=1 non-candidate cases")
    if ACTIVATION_MARKER not in output:
        raise RuntimeError("N=1 gate did not prove the ncols2 experiment was enabled")
    if ONE_COLUMN_MARKER in output:
        raise RuntimeError("N=1 gate unexpectedly enabled the separate one-column experiment")
    if KERNEL_HIT_PATTERN.search(output):
        raise RuntimeError("N=1 incorrectly reached the ncols2 candidate kernel")


def run_mixed_visible_device_rejection(
    binary: str, devices: List[Tuple[str, str]], sm89_uuid: str
) -> None:
    non_sm89_uuid = next((uuid for cc, uuid in devices if cc != "8.9"), None)
    if non_sm89_uuid is None:
        print("SKIP subcase: no mixed SM89/non-SM89 visible-device set is available")
        return
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = f"{sm89_uuid},{non_sm89_uuid}"
    env["GGML_CUDA_TURBO4_SYM_LUT"] = "0"
    env["GGML_CUDA_TURBO4_SYM_LUT_NCOLS2"] = "1"
    env["GGML_CUDA_TURBO4_SYM_LUT_NCOLS2_TRACE"] = "0"
    result = subprocess.run(
        [
            binary,
            "test",
            "-o",
            "FLASH_ATTN_EXT",
            "-b",
            "CUDA0",
            "-p",
            r"hsk=128.*nb=2.*type_K=turbo4",
            "-j",
            "1",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    output = result.stdout + result.stderr
    sys.stdout.write(output)
    if result.returncode == 0:
        raise RuntimeError("mixed SM89/non-SM89 visible-device request silently succeeded")
    if "requires every visible CUDA device to be SM89" not in output:
        raise RuntimeError("mixed visible-device request failed without the fail-closed topology diagnostic")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend-ops", required=True)
    args = parser.parse_args()

    devices = visible_cuda_devices()
    device_uuid = sm89_device_uuid(devices)
    if device_uuid is None:
        print("SKIP: no SM89 CUDA device is visible")
        return 77

    run_parity(args.backend_ops, device_uuid, enabled=False)
    run_parity(args.backend_ops, device_uuid, enabled=True)
    run_single_column_non_candidate(args.backend_ops, device_uuid)
    run_unsupported_head_rejection(
        args.backend_ops,
        device_uuid,
        384,
        r"hsk=384.*nr23=\[1,1\].*nb=2.*type_K=turbo4",
    )
    run_unsupported_head_rejection(
        args.backend_ops,
        device_uuid,
        640,
        r"hsk=640.*hsv=512.*nr23=\[2,1\].*nb=2.*type_K=turbo4",
    )
    run_mixed_visible_device_rejection(args.backend_ops, devices, device_uuid)
    print(
        "PASS: SM89 Turbo4 ncols=2 baseline and opt-in each matched the CPU reference in 49/49 cases; "
        "D=512 covered Turbo4/Q8_0/F16 values at every nb=2..8; all opt-in cases emitted exact "
        "kernel-hit evidence, including every nb=3/5/7 odd tail; "
        "N=1 remained outside the ncols2 candidate while the complete N=2..8 surface was covered; "
        "unsupported D=384/D=640 rejected fail-closed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
