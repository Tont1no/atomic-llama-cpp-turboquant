#!/usr/bin/env python3

import argparse
import os
import re
import subprocess
import sys
from typing import Optional


ACTIVATION_MARKER = (
    "CUDA: enabling experimental SM89 Turbo4 symmetric-magnitude LUT "
    "for one-column VEC decode on all visible devices"
)


def sm89_device_uuid() -> Optional[str]:
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
        return None
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", maxsplit=1)]
        if len(fields) == 2 and fields[0] == "8.9" and fields[1].startswith("GPU-"):
            return fields[1]
    return None


def run_parity(binary: str, device_uuid: str, enabled: bool) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device_uuid
    env["GGML_CUDA_TURBO4_SYM_LUT"] = "1" if enabled else "0"
    command = [
        binary,
        "test",
        "-o",
        "FLASH_ATTN_EXT",
        "-b",
        "CUDA0",
        "-p",
        r"nb=1.*type_K=turbo4",
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
            f"Turbo4 SM89 {'opt-in' if enabled else 'baseline'} parity run failed "
            f"with exit code {result.returncode}"
        )
    match = re.search(r"(\d+)/(\d+) tests passed", output)
    if match is None or match.group(1) != "4" or match.group(2) != "4":
        raise RuntimeError(
            f"expected exactly 4 Turbo4 parity cases, got "
            f"{match.group(0) if match else 'no test summary'}"
        )
    marker_present = ACTIVATION_MARKER in output
    if enabled and not marker_present:
        raise RuntimeError("Turbo4 opt-in run passed numerically but did not activate the experimental kernel")
    if not enabled and marker_present:
        raise RuntimeError("Turbo4 baseline unexpectedly activated the experimental kernel")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend-ops", required=True)
    args = parser.parse_args()

    device_uuid = sm89_device_uuid()
    if device_uuid is None:
        print("SKIP: no SM89 CUDA device is visible")
        return 77

    run_parity(args.backend_ops, device_uuid, enabled=False)
    run_parity(args.backend_ops, device_uuid, enabled=True)
    print("PASS: SM89 Turbo4 baseline and opt-in each matched the CPU reference in 4/4 cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
