"""Kaggle CLI calls, devices."""

import os
import re
import subprocess
import sys

HOUR = 3600

# device: accelerator for `kaggle kernels push --accelerator`, session limit in seconds (None: no limit)
DEVICES = {
    "t4x2": ("NvidiaTeslaT4", 12 * HOUR),
    "tpu": ("TpuV5E8", 9 * HOUR),
    "cpu": (None, 12 * HOUR),
    "local": (None, None),
}
ACTIVE = {"QUEUED", "RUNNING"}  # run status holding device


def kaggle(*args, check=True):
    """stdout of Kaggle CLI call; stderr passes through, so failures show CLI message. `check`: exit on failure,
    else None."""
    result = subprocess.run(["kaggle", *args], stdout=subprocess.PIPE, text=True)

    if result.returncode:
        if not check:
            return None

        sys.exit(f"kaggle {' '.join(args)} failed: {result.stdout.strip()}")

    return result.stdout


def username():
    if "KAGGLE_USERNAME" in os.environ:
        return os.environ["KAGGLE_USERNAME"]

    return re.search(r"- username: (\S+)", kaggle("config", "view")).group(1)


def run_status(kernel_id):
    """Status of latest run: QUEUED, RUNNING, COMPLETE, ERROR, CANCEL_*; None when Kaggle has no such kernel."""
    out = kaggle("kernels", "status", kernel_id, check=False)
    match = out and re.search(r"KernelWorkerStatus\.(\w+)", out)

    return match.group(1) if match else None
