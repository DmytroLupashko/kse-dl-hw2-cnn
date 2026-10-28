"""Job file: YAML of one push, run in kernel by `kgpu.runner`.

Keys: `name` (kernel, default file stem); compute: `device`, `buffer` (minutes before session limit), `code`,
`pythonpath`, `data`; run: `setup` (`mkdir`, `pip`, `pip_missing`), `env`, `tasks` (`items`, `command`, `log`, `env`),
`vars` (placeholders); `check`: command run locally before push, placeholders `vars` and `{items}`.
`defaults: {<group>: <name>}` links `<job folder>/<group>/<name>.yaml` (e.g. `compute/t4x2.yaml`), merged in order under
job keys. CLI `--key=value` (dotted keys nested, values YAML) merged over all: `--device=local`, `'--tasks.items=[3]'`.
"""

import inspect
import os
import subprocess
from pathlib import Path

import yaml

from . import runner

DEFAULTS = {
    "device": "t4x2",
    "buffer": 30,
    "code": [],
    "pythonpath": [],
    "data": [],
    "setup": {"mkdir": [], "pip": [], "pip_missing": []},
    "env": {},
    "tasks": {"log": "logs/{item}.log", "env": {}},
    "vars": {},
    "check": None,
}
RUNNER_KEYS = ["device", "pythonpath", "setup", "env", "tasks", "vars"]


def merge(base, update):
    """`base` updated by `update`, nested dicts merged recursively."""
    out = dict(base)

    for key, value in update.items():
        out[key] = (
            merge(out[key], value)
            if isinstance(out.get(key), dict) and isinstance(value, dict)
            else value
        )

    return out


def parse(argv):
    """{dotted key: YAML value} of `--key=value` args, nested: `--tasks.items=[3]` -> {"tasks": {"items": [3]}}."""
    overrides = {}

    for arg in argv:
        key, sep, value = arg.removeprefix("--").partition("=")

        if not arg.startswith("--") or not sep:
            raise SystemExit(f"expected --key=value, got {arg}")

        *parents, leaf = key.split(".")
        node = overrides

        for parent in parents:
            node = node.setdefault(parent, {})

        node[leaf] = yaml.safe_load(value)

    return overrides


def load(path, overrides=None):
    """Resolved job of file `path`: `DEFAULTS`, linked files of `defaults`, job keys, `overrides`; unknown key fails."""
    path = Path(path)
    job = yaml.safe_load(path.read_text())
    resolved = {"name": path.stem, **DEFAULTS}

    for group, name in job.pop("defaults", {}).items():
        resolved = merge(
            resolved, yaml.safe_load((path.parent / group / f"{name}.yaml").read_text())
        )

    resolved = merge(merge(resolved, job), overrides or {})
    unknown = set(resolved) - {"name", *DEFAULTS}
    missing = {"items", "command"} - set(resolved["tasks"])

    if unknown or missing:
        raise SystemExit(
            f"{path}: unknown keys {sorted(unknown)}"
            if unknown
            else f"{path}: no tasks.{' tasks.'.join(sorted(missing))}"
        )

    return resolved


def source(job):
    """Notebook cell: `kgpu.runner` source, then `run` of runner part of `job`."""
    spec = {key: job[key] for key in RUNNER_KEYS}

    return f"# kgpu job {job['name']}\n{inspect.getsource(runner)}\n\nrun({spec!r})"


def check(job):
    """Job `check` command locally, `pythonpath` folders importable; failing command stops push."""
    if job["check"] is None:
        return

    paths = [str(Path(p).resolve()) for p in job["pythonpath"]]
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([*paths, os.environ.get("PYTHONPATH", "")]),
    }

    if subprocess.run(
        runner.argv(job["check"], **job["vars"], items=job["tasks"]["items"]), env=env
    ).returncode:
        raise SystemExit(f"check failed: {job['check']}")
