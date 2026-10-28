"""Kaggle GPU util: push job files (`kgpu.job`), track, pull, delete kernels.

`[tool.kgpu]` of `./pyproject.toml`: `home` (kernel folders, default `.kgpu`).
"""

import argparse
import subprocess
import sys
import time
import tomllib
from datetime import datetime
from pathlib import Path

from . import job, progress
from .kaggle import ACTIVE, DEVICES, kaggle
from .kernel import Kernel

TRACK_SECONDS = 30


def settings():
    path = Path("pyproject.toml")
    table = (
        tomllib.loads(path.read_text()).get("tool", {}).get("kgpu", {})
        if path.exists()
        else {}
    )

    return {"home": ".kgpu", **table}


def push(args, cfg):
    """Job file resolved, `check` run locally, then pushed (`--dry-run`: kernel folder only)."""
    spec = job.load(args.job, job.parse(args.overrides))

    if spec["device"] not in DEVICES:
        raise SystemExit(f"device: one of {DEVICES}")

    job.check(spec)
    Kernel(spec["name"], cfg["home"]).push(
        job.source(spec),
        spec["code"],
        spec["data"],
        spec["device"],
        spec["buffer"] * 60,
        args.dry_run,
    )


def show(kernel):
    """Device, push time, status, deadline, progress per task."""
    run = kernel.last()
    print(f"{kernel.name}: {run['status']} on {run['device']}, pushed {run['pushed']}")

    if run["status"] in ACTIVE and run["budget"]:
        print(
            f"deadline in ~{progress.duration(kernel.deadline_left(run))} (from push)"
        )

    print(progress.render(progress.parse(kernel.log())))


def status(args, cfg):
    if args.name:
        return show(Kernel(args.name, cfg["home"]))

    active = [k for k in Kernel.all(cfg["home"]) if k.last()["status"] in ACTIVE]

    for kernel in active:
        run = kernel.last()
        print(
            f"{kernel.name:<30} {run['status']:<8} {run['device']:<6} pushed {run['pushed']}"
        )

    if not active:
        print("no queued or running kernels")


def track(args, cfg):
    kernel = Kernel(args.name, cfg["home"])

    while True:
        if sys.stdout.isatty():
            print("\x1b[H\x1b[J", end="")  # clear screen

        show(kernel)

        if kernel.last()["status"] not in ACTIVE:
            return

        time.sleep(TRACK_SECONDS)


def log(args, cfg):
    kernel = Kernel(args.name, cfg["home"])

    if not args.follow:
        return print(kernel.log())

    if kernel.last()["device"] == "local":
        subprocess.run(["tail", "-f", str(kernel.log_file())])
    elif subprocess.run(
        ["kaggle", "kernels", "logs", kernel.id, "--follow"]
    ).returncode:
        sys.exit(
            f"log stream unavailable, retry in a minute or open kaggle.com/code/{kernel.id}"
        )


def devices(args, cfg):
    """Kernels holding each device; free when none. Kaggle GPU and TPU quota."""
    holders = {device: [] for device in DEVICES}

    for kernel in Kernel.all(cfg["home"]):
        run = kernel.last()

        if run["status"] in ACTIVE:
            holders[run["device"]].append(f"{kernel.name} ({run['status']})")

    for device, names in holders.items():
        print(f"{device:<6} {', '.join(names) or 'free'}")

    print()
    print(kaggle("quota"))


def history(args, cfg):
    kernels = [Kernel(args.name, cfg["home"])] if args.name else Kernel.all(cfg["home"])

    for kernel in kernels:
        kernel.last()  # refresh status of latest run

        for i, run in enumerate(kernel.runs(), 1):
            took = run["finished"] and progress.duration(
                (
                    datetime.fromisoformat(run["finished"])
                    - datetime.fromisoformat(run["pushed"])
                ).total_seconds()
            )
            print(
                f"{kernel.name:<30} #{i:<3} {run['device']:<6} {run['status']:<10} pushed {run['pushed']}  took {took or '-'}"
            )


def main(argv=None):
    parser = argparse.ArgumentParser(prog="kgpu", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    for command, run, help in [
        (
            "pull",
            lambda a, c: Kernel(a.name, c["home"]).pull(),
            "download outputs of finished run",
        ),
        (
            "delete",
            lambda a, c: Kernel(a.name, c["home"]).delete(),
            "delete kernel, running or not",
        ),
        ("track", track, "live progress until run finishes"),
    ]:
        p = commands.add_parser(command, help=help)
        p.add_argument("name")
        p.set_defaults(run=run)

    p = commands.add_parser("log", help="run log")
    p.add_argument("name")
    p.add_argument("-f", "--follow", action="store_true")
    p.set_defaults(run=log)

    for command, run, help in [
        (
            "status",
            status,
            "queued and running kernels; with name: its state and progress",
        ),
        ("history", history, "earlier runs"),
    ]:
        p = commands.add_parser(command, help=help)
        p.add_argument("name", nargs="?")
        p.set_defaults(run=run)

    p = commands.add_parser("devices", help="devices taken by kernels, quota")
    p.set_defaults(run=devices)

    p = commands.add_parser(
        "push", help="push job file: <job.yaml> [--key=value ...] [--dry-run]"
    )
    p.add_argument("job")
    p.add_argument("--dry-run", action="store_true", help="build kernel folder only")
    p.set_defaults(run=push)

    args, extra = parser.parse_known_args(argv)

    if args.command == "push":
        args.overrides = extra
    elif extra:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")

    args.run(args, settings())
